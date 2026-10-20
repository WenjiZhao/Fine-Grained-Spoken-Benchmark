"""
Per-record scoring for the tool-calling tasks in data/.

Tiers, cascading like VoiceAgentBench's evaluator (later tiers only meaningful
if earlier ones pass) but generalized to the 5-way gold.action space used in
data/ (call / call_multiple / no_call / clarify / cannot_answer), and with a
4th tier (grounding) enabled by real tool execution (tools/executor.py)
instead of a fixed reference response:

  0. action        - call/call_multiple come from the number of tool calls
                      the model actually made; no_call/clarify/cannot_answer
                      (zero calls made) are told apart by an LLM judge
                      (falls back to a "?"-in-response heuristic that can't
                      tell cannot_answer apart at all, if no LLM configured)
  1. tool_selection - multiset F1 of tool names (FDB-v3 style), for call(_multiple)
  2. schema         - required arguments present, per tools/tool_schemas.json;
                       a record whose gold needs a call but that made none
                       scores 0 (schema_valid_when_called keeps the old
                       called-only view, and `called` feeds call_rate)
  3. argument_value / clarify_quality - LLM judge (skipped if no LLM configured);
                       argument values equal up to case/punctuation/spacing
                       score 1 without asking the judge
  4. grounding      - LLM judge of final_response against REAL tool_results
                       (None if this record has no tool_results, i.e. was run
                       with a fixed/pre-written tool response)

Note: clarify_quality (the clarify half of tier 3) and grounding (tier 4) are
currently disabled in evaluate_record / failed_tiers; both fields are always None.

Except for *self_correction task_types, only the model's final calls are
scored (see final_calls): extra calls made before its last call to a gold
tool -- preparatory lookups, a wrong first attempt it then redid -- don't count
against it. Self-correction tasks keep every call, since acting on the
retracted request before the correction is exactly what they measure.
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import prompts
from prompts import (
    ARGUMENT_VALUE_PROMPT,
    CLARIFY_PROMPT,
    GROUNDING_PROMPT,
    classify_zero_call_action,
    llm_judge_score,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCHEMAS_PATH = ROOT / "tools" / "tool_schemas.json"


def load_schemas(path: Path = DEFAULT_SCHEMAS_PATH) -> Dict[str, dict]:
    return {t["name"]: t for t in json.loads(path.read_text())}


def _dig(obj: Any, path_parts: List[str]) -> Any:
    for part in path_parts:
        if isinstance(obj, list):
            if not part.lstrip("-").isdigit():
                return None
            idx = int(part)
            if idx >= len(obj) or idx < -len(obj):
                return None
            obj = obj[idx]
        elif isinstance(obj, dict):
            if part not in obj:
                return None
            obj = obj[part]
        else:
            return None
    return obj


def resolve_ref(value: Any, tool_results: Optional[List[dict]], expected_calls: Optional[List[dict]] = None) -> Any:
    """Resolve a gold-data placeholder against a real upstream tool result:

    - "<resolved_from_X_result>" (see data/multi_tool/sequential_tool_call):
      X is a tool name, looked up directly in tool_results.
    - "<result:N.path.to.field>" (see data/tool_pool.json's
      get_current_location/_date/_time/get_user_profile "implicit-context
      resolver" tools): N is the index of the resolver call within this
      record's own gold.calls list; expected_calls[N]["tool"] gives the tool
      name, which is then looked up in tool_results the same way, before
      digging "path.to.field" out of its real recorded result (dict keys or
      list indices).

    Either way, if the referenced call was never actually made (tool_results
    has no matching entry) there's no real value to compare against, so the
    raw placeholder is returned unchanged -- same as if it were never a
    placeholder at all.
    """
    if not isinstance(value, str):
        return value
    if value.startswith("<resolved_from_") and value.endswith("_result>"):
        source = value[len("<resolved_from_"):-len("_result>")]
        for call in tool_results or []:
            if call.get("name") == source:
                return call.get("result")
        return value
    if value.startswith("<result:") and value.endswith(">"):
        idx_str, _, path = value[len("<result:"):-1].partition(".")
        if not idx_str.isdigit() or not expected_calls or int(idx_str) >= len(expected_calls):
            return value
        source = expected_calls[int(idx_str)].get("tool")
        for call in tool_results or []:
            if call.get("name") == source:
                resolved = _dig(call.get("result"), path.split(".")) if path else call.get("result")
                return value if resolved is None else resolved
        return value
    return value


def expected_calls_from_gold(gold: dict) -> List[dict]:
    if gold.get("action") == "call":
        return [{"tool": gold["tool"], "arguments": gold.get("arguments") or {}}]
    if gold.get("action") == "call_multiple":
        return gold.get("calls", [])
    return []


def final_calls(expected_calls: List[dict], actual_calls: List[dict]) -> List[dict]:
    """The model's final result: for each gold tool, its last N calls to that
    tool (N = how many times gold calls it), plus any non-gold calls made after
    its last gold-tool call. Calls to non-gold tools before that point, and
    earlier repeats of a gold tool, are dropped. Order is preserved."""
    need = Counter(e["tool"] for e in expected_calls)
    last_gold = max((i for i, c in enumerate(actual_calls) if c["name"] in need), default=-1)
    kept = []
    for i in reversed(range(len(actual_calls))):
        call = actual_calls[i]
        if call["name"] in need:
            if need[call["name"]] > 0:
                kept.append(call)
                need[call["name"]] -= 1
        elif i > last_gold:
            kept.append(call)
    return kept[::-1]


def predicted_action(query: str, actual_calls: List[dict], final_response: str) -> str:
    n = len(actual_calls)
    if n == 0:
        classified = classify_zero_call_action(query, final_response)
        if classified is not None:
            return classified
        return "clarify" if "?" in (final_response or "") else "no_call"
    return "call" if n == 1 else "call_multiple"


def score_tool_selection(expected_calls: List[dict], actual_calls: List[dict]) -> dict:
    expected = [c["tool"] for c in expected_calls]
    actual = [c["name"] for c in actual_calls]
    exp_left, act_left = list(expected), list(actual)
    matched = 0
    for name in list(exp_left):
        if name in act_left:
            matched += 1
            exp_left.remove(name)
            act_left.remove(name)
    recall = matched / len(expected) if expected else 1.0
    precision = matched / len(actual) if actual else 1.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0
    return {"f1": round(f1, 3), "recall": round(recall, 3), "precision": round(precision, 3)}


def score_schema(actual_calls: List[dict], schemas: Dict[str, dict]) -> Optional[float]:
    if not actual_calls:
        return None
    valid = 0
    for call in actual_calls:
        schema = schemas.get(call["name"])
        required = schema["parameters"]["required"] if schema else []
        args = call.get("args", {})
        if all(k in args for k in required):
            valid += 1
    return valid / len(actual_calls)


FILLER_WORDS = {"in", "at", "on", "of", "the", "a", "an"}


def _normalize(value: Any) -> Any:
    """Surface-form-insensitive view of an argument value: strings lose case,
    punctuation, spacing and filler words ("EBT-145" == "ebt145",
    "500 Pine Street in Seattle" == "500 Pine Street, Seattle"), numeric
    strings become numbers, and lists/dicts are normalized element-wise."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", ""))
        except ValueError:
            # numeric ranges: "50 to 100" / "50–100" / "50~100" -> "50-100"
            value = re.sub(r"(\d)\s*(?:to|-|–|—|~)\s*(\d)", r"\1-\2", value.lower())
            tokens = re.findall(r"[^\W_]+", value)
            return "".join(t for t in tokens if t not in FILLER_WORDS) or "".join(tokens)
    if isinstance(value, list):
        return sorted((_normalize(v) for v in value), key=repr)
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    return value


def arguments_match_exactly(reference: dict, actual: dict) -> bool:
    """Every reference argument equals the actual one after _normalize (extra
    actual arguments are ignored, as in the LLM judge)."""
    return all(k in actual and _normalize(v) == _normalize(actual[k]) for k, v in reference.items())


def score_argument_values(query: str, expected_calls: List[dict], actual_calls: List[dict],
                           tool_results: Optional[List[dict]]) -> Optional[float]:
    if not expected_calls:
        return None
    by_name = {c["name"]: c for c in actual_calls}
    scores = []
    for exp in expected_calls:
        actual = by_name.get(exp["tool"])
        if actual is None:
            scores.append(0)
            continue
        reference = {k: resolve_ref(v, tool_results, expected_calls) for k, v in exp.get("arguments", {}).items()}
        if arguments_match_exactly(reference, actual.get("args", {})):
            scores.append(1)
            continue
        prompt = ARGUMENT_VALUE_PROMPT.format(query=query, reference=reference, actual=actual.get("args", {}))
        score = llm_judge_score(prompt)
        if score is not None:
            scores.append(score)
    return sum(scores) / len(scores) if scores else None


def score_clarify(gold: dict, final_response: str) -> Optional[int]:
    if gold.get("action") != "clarify" or "clarifying_question" not in gold:
        return None
    prompt = CLARIFY_PROMPT.format(reference_question=gold["clarifying_question"], actual_response=final_response)
    return llm_judge_score(prompt)


def score_grounding(query: str, tool_results: Optional[List[dict]], final_response: str) -> Optional[int]:
    if not tool_results:
        return None
    prompt = GROUNDING_PROMPT.format(query=query, tool_results=json.dumps(tool_results, ensure_ascii=False),
                                      final_response=final_response)
    return llm_judge_score(prompt)


def evaluate_record(record: dict, schemas: Dict[str, dict], grounding: bool = True) -> dict:
    gold = record["gold"]
    actual_calls = record.get("actual_tool_calls", [])
    tool_results = record.get("tool_results")
    final_response = record.get("final_response", record.get("generated_text", ""))

    expected_calls = expected_calls_from_gold(gold)
    n_calls_made = len(actual_calls)
    if expected_calls and "self_correction" not in record.get("task_type", ""):
        actual_calls = final_calls(expected_calls, actual_calls)
    failures_before = prompts.judge_failures
    predicted = predicted_action(record["query"], actual_calls, final_response)
    schema_when_called = score_schema(actual_calls, schemas) if expected_calls else None

    score = {
        "id": record.get("id"),
        "task_type": record.get("task_type"),
        "gold_action": gold.get("action"),
        "predicted_action": predicted,
        "action_correct": predicted == gold.get("action"),
        "tool_selection": score_tool_selection(expected_calls, actual_calls) if expected_calls else None,
        "called": bool(actual_calls) if expected_calls else None,
        "schema_valid": (schema_when_called or 0.0) if expected_calls else None,
        "schema_valid_when_called": schema_when_called,
        "argument_accuracy": score_argument_values(record["query"], expected_calls, actual_calls, tool_results),
        # clarify / grounding disabled: no judge call, fields stay None and do not count toward strict_success.
        # "clarify_quality": score_clarify(gold, final_response),
        # "result_grounding": score_grounding(record["query"], tool_results, final_response) if grounding else None,
        "clarify_quality": None,
        "result_grounding": None,
        "n_calls_ignored": n_calls_made - len(actual_calls),
    }
    score["strict_success"] = not failed_tiers(score)
    score["n_judge_failures"] = prompts.judge_failures - failures_before
    return score


def failed_tiers(score: dict) -> List[str]:
    """Which cascading tiers this record failed (empty list = fully correct on
    every tier that was actually evaluated for it)."""
    tiers = []
    if not score["action_correct"]:
        tiers.append("action")
    ts = score["tool_selection"]
    if ts is not None and ts["f1"] < 1.0:
        tiers.append("tool_selection")
    if score["schema_valid"] is not None and score["schema_valid"] < 1.0:
        tiers.append("schema")
    if score["argument_accuracy"] is not None and score["argument_accuracy"] < 1.0:
        tiers.append("argument")
    # clarify / grounding disabled: not counted as failed tiers.
    # if score["clarify_quality"] is not None and score["clarify_quality"] < 1:
    #     tiers.append("clarify")
    # if score["result_grounding"] is not None and score["result_grounding"] < 1:
    #     tiers.append("grounding")
    return tiers
