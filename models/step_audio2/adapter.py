"""Step-Audio2 adapter: dataset record -> normalized result dict.

Multi-step (up to MAX_STEPS) tool-calling loop over the vendored, un-fine-tuned StepAudio2 base
model (local `transformers` checkpoint, no vLLM server -- see the README.md
next to this file for why that route was picked over standing up the
official vLLM Docker image). VoxMind is itself Step-Audio2 plus a further
fine-tune, so this adapter mirrors models/voxmind/adapter.py's shape almost
exactly (same tool_schemas_for/resolved_audio_path/MockToolExecutor/
build_result_record plumbing, same two-turn structure: decide/call, feed
results back, produce a grounded final answer).

The one thing that's genuinely different from VoxMind: how tools are shown
to the model. VoxMind's own system-prompt template embeds the tool JSON
inline. The upstream Step-Audio2 demo (examples.py::tool_call_test) instead
uses a dedicated `tool_json_schemas` message role, fed as its own turn
alongside "system"/"human"/"assistant" -- that's the officially-demonstrated
mechanism for exposing tools to this exact base checkpoint, so this adapter
uses that instead of inventing a prompt-embedding convention. Each record's
distinct `available_tools` become a distinct tool_json_schemas turn per
run -- one independent generate() call per record, per MockToolExecutor's
own fresh-per-record contract.

RESOLVED CAVEAT -- native tool-call format, confirmed against real output:
an earlier version of this adapter taught the model the VoxMind-style
<tool_call>[{"name":..., "arguments":{...}}]</tool_call> format (see
models/voxmind/vendor/runtime/prompts.py), reasoning that a decent
instruction-following base model could pick up an in-context format even
without dedicated SFT for it. Two things turned out to be wrong with that.
First, upstream's own tool_call_test demo (examples.py) never actually
teaches a tag either -- its system prompt is purely narrative ("you have
the ability to call tools"), and the demo never even parses the model's
turn-1 text; it hard-codes a canned search result and continues regardless
of what the model said. So there was never an official example to copy.
Second, and more telling: stepfun-ai/Step-Audio-2-mini's own
tokenizer_config.json (added_tokens_decoder) defines <|CALL_START|>/
<|CALL_END|> (ids 151667/151668) as *this checkpoint's own* dedicated
tool-call tokens -- separate from the <tool_call>/</tool_call> pair (ids
151657/151658) it merely inherited from its Qwen2 base vocab and was never
necessarily fine-tuned to use. Live turn-1 output showed exactly the
failure this predicts: the model opens with its native <|CALL_START|>
function but closes with the taught </tool_call>, e.g.
'<|CALL_START|>function\nget_card_benefits{"card_type": "Platinum"}</tool_call>'
-- a name+args shape that matches neither the taught JSON-array format nor
the bare-JSON fallback (no top-level "name" key), so _extract_tool_calls()
silently returned zero calls despite the model clearly intending one.
SYSTEM_MESSAGE below teaches <|CALL_START|>function\nTOOL_NAME\n{...}<|CALL_END|>
-- this checkpoint's own native tokens, in the name+args shape it reached
for on its own even under the wrong (VoxMind-borrowed) in-context example
above -- rather than the borrowed <tool_call> tag. This is deliberately
*not* the hands-off, no-tag approach upstream's own demo takes: a first
attempt at dropping all format teaching (matching upstream) was tried and
made turn-1 output regress to zero attempts across the board, e.g. "I'll
look up the benefits of the Platinum Card." with no <|CALL_START|> at all --
this checkpoint, in this tool_json_schemas-turn setup, does not reliably
self-trigger its own call tokens from the tool list alone; it needs the
format spelled out, just spelled out *correctly* this time.
_extract_tool_calls() parses the checkpoint's own native <|CALL_START|> ...
<|CALL_END|> span first, keeping the old taught-tag and bare-JSON parses as
fallbacks in case the checkpoint ever reaches for its inherited Qwen vocab
instead. Smoke-test with --limit 3 --verbose and read the printed turn-1
raw text before trusting a full run.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BENCH_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from stepaudio2 import StepAudio2  # noqa: E402

from models.common.dataset import available_tool_names, resolved_audio_path, tool_schemas_for  # noqa: E402
from models.common.result_schema import build_result_record, normalize_calls  # noqa: E402
from tools.executor import MockToolExecutor  # noqa: E402

GENERATION_KWARGS = dict(
    max_new_tokens=256,
    temperature=0.6,
    repetition_penalty=1.05,
    top_p=0.9,
    do_sample=True,
)

# Step-Audio-2-mini's own tool-call tokens, per its tokenizer_config.json
# added_tokens_decoder (ids 151667/151668) -- distinct from the <tool_call>/
# </tool_call> pair (ids 151657/151658) it merely inherited from its Qwen2
# base vocab. See the module docstring's RESOLVED CAVEAT for how this was
# confirmed against real generation output.
CALL_START = "<|CALL_START|>"
CALL_END = "<|CALL_END|>"

# Kept only as a fallback parse target: the format this adapter used to teach
# in-context (VoxMind's convention, not this checkpoint's own). The model
# still has these tokens in-vocab from its Qwen2 base and may reach for them.
LEGACY_TOOL_CALL_START = "<tool_call>"
LEGACY_TOOL_CALL_END = "</tool_call>"

OBSERVATION_PREFIX = "Below is the return content of the function you used:"
OBSERVATION_SUFFIX = (
    "If the request still needs another tool, call it now using your call tokens; "
    "otherwise answer the user."
)

# Upper bound on generate() rounds per record (each round may emit tool calls
# whose results are fed back before the next round).
MAX_STEPS = 8

SYSTEM_MESSAGE = (
    "You are a helpful voice assistant. You understand spoken requests and decide whether "
    "to answer directly from your own knowledge or call your available tools.\n\n"
    "Tool Decision Rules:\n"
    "- If the request can be answered directly, answer normally -- do not call any tool.\n"
    "- If the request needs a capability listed in your available tools, call that tool "
    "instead of guessing the answer.\n"
    "- Never invent or call a tool name that is not literally one of your available tools.\n"
    "- Tool arguments must be values the user actually said or values returned by an "
    "earlier tool result; if a required argument is missing from both, ask the user "
    "instead of guessing.\n"
    "- If a tool call fails or returns an error, do not retry it for the same request -- "
    "tell the user the tool has an issue.\n"
    "- If the request needs several tools in sequence (e.g. search first, then act on a "
    "result), call them one after another: after each tool result comes back, call the next "
    "tool with the values from that result before answering the user.\n\n"
    "Tool Call Format:\n"
    "When you decide to call a tool, emit the call itself using your call tokens -- do not "
    "just describe in words that you are going to call it:\n"
    f"{CALL_START}function\n"
    "TOOL_NAME\n"
    '{"arg_name": "value"}\n'
    f"{CALL_END}\n"
    "A normal, non-tool-calling response must not contain "
    f"{CALL_START}."
)


def _openai_tool_schema(schema: dict) -> dict:
    """tools/tool_schemas.json's shape -> the {"type": "function", "function": {...}}
    shape examples.py::tool_call_test's `tool_json_schemas` message uses."""
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema.get("description", schema["name"]),
            "parameters": schema.get("parameters", {"type": "object", "properties": {}, "required": []}),
        },
    }


def _balanced_json_candidates(text: str) -> List[str]:
    """Top-level {...}/[...] substrings of `text`, respecting nesting -- used by the
    fallback branch of _extract_tool_calls when the model didn't use the taught tag."""
    candidates = []
    stack: List[str] = []
    start: Optional[int] = None
    for i, ch in enumerate(text):
        if ch in "{[":
            if not stack:
                start = i
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
            if not stack and start is not None:
                candidates.append(text[start : i + 1])
                start = None
    return candidates


_NATIVE_CALL_NAME_RE = re.compile(r"\A\s*function\s*\n?\s*([\w.]+)\s*\n?", re.DOTALL)
_LOOSE_CALL_RE = re.compile(r"([A-Za-z_][\w.]*)\s*\n?\s*(\{)")


# After a call-end token: either 'function\nNAME...' or a bare 'NAME\n{...}'
# (the model often drops the 'function\n' prefix on chained calls). The empty
# `bare` group records which form matched.
_FOLLOW_ON_CALL_RE = re.compile(
    r"\s*(?:(?=function\s*\n)|(?P<bare>)(?=[A-Za-z_][\w.]*[ \t]*\n\s*\{))"
)


def _native_call_blocks(text: str) -> List[str]:
    """Slice out the content between each <|CALL_START|> and the nearest of
    <|CALL_END|> / the legacy taught </tool_call> / end-of-text -- this base
    checkpoint is unverified and may not reliably close the span at all, per
    the module docstring's RESOLVED CAVEAT.

    Multiple calls in one turn were observed chained under a single
    <|CALL_START|>, e.g. '<|CALL_START|>function\nA\n{...}</tool_call>function\nB\n{...}</tool_call>'
    or, without the 'function\n' prefix, '...<|CALL_END|>\nB\n{...}<|CALL_END|>'
    -- a block closed by a call-end token and immediately followed by either
    form is treated as the start of another call block (bare-name blocks get
    'function\n' prepended so _NATIVE_CALL_NAME_RE parses them uniformly)."""
    blocks = []
    pos = 0
    prefix = ""
    start = text.find(CALL_START)
    if start != -1:
        start += len(CALL_START)
    while start != -1:
        end_candidates = [
            (p, tok) for tok in (CALL_END, LEGACY_TOOL_CALL_END) if (p := text.find(tok, start)) != -1
        ]
        if not end_candidates:
            blocks.append(prefix + text[start:])
            break
        end, close_tok = min(end_candidates)
        blocks.append(prefix + text[start:end])
        pos = end + len(close_tok)
        follow_on = _FOLLOW_ON_CALL_RE.match(text, pos)
        if follow_on:
            start = follow_on.end()
            prefix = "function\n" if follow_on.group("bare") is not None else ""
            continue
        prefix = ""
        start = text.find(CALL_START, pos)
        if start != -1:
            start += len(CALL_START)
    return blocks


def _json_call_object(candidate: str) -> Optional[Dict[str, Any]]:
    """{"name"|"function": "<tool>", "arguments": {...}} -> a call dict, else None.
    A string tool name is required, so echoed schemas ({"function": {...}})
    and hallucinated results ({"result": ...}) are rejected."""
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name", obj.get("function"))
    arguments = obj.get("arguments")
    if isinstance(name, str) and name and isinstance(arguments, dict):
        return {"name": name, "arguments": arguments}
    return None


def _extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse tool calls out of `text`, trying Step-Audio-2-mini's own native
    <|CALL_START|> ... <|CALL_END|> tokens first, then the legacy taught
    <tool_call>[...]</tool_call> format (kept in case the checkpoint reaches
    for its inherited Qwen vocab instead), then a bare-JSON fallback scan.
    See this module's docstring for how the native format was confirmed."""
    calls: List[Dict[str, Any]] = []
    for block in _native_call_blocks(text):
        name_match = _NATIVE_CALL_NAME_RE.match(block)
        if not name_match:
            continue
        name = name_match.group(1)
        for candidate in _balanced_json_candidates(block[name_match.end():]):
            try:
                arguments = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(arguments, dict):
                calls.append({"name": name, "arguments": arguments})
            break
    if calls:
        # A chained call is sometimes written as a JSON object instead, e.g.
        # '...</tool_call>\n{"function": "get_weather", "arguments": {...}}</tool_call>'
        # -- pick those up too, after the first native call.
        tail = text[text.find(CALL_START):]
        calls.extend(c for c in map(_json_call_object, _balanced_json_candidates(tail)) if c)
        return calls

    raw_blocks = [
        match.strip()
        for match in re.findall(
            rf"{re.escape(LEGACY_TOOL_CALL_START)}\s*(.*?)\s*{re.escape(LEGACY_TOOL_CALL_END)}", text, re.DOTALL
        )
    ]
    for block in raw_blocks:
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            calls.extend(item for item in parsed if isinstance(item, dict) and "name" in item)
        elif isinstance(parsed, dict) and "name" in parsed:
            calls.append(parsed)
    if calls:
        return calls

    # Fallback: model emitted a bare JSON object/array directly, without any
    # recognized wrapper tag.
    for candidate in _balanced_json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        items = parsed if isinstance(parsed, list) else [parsed]
        calls.extend(item for item in items if isinstance(item, dict) and "name" in item)
    if calls:
        return calls

    # Last resort: model dropped every wrapper (no <|CALL_START|>, no
    # <tool_call>) and just wrote `name\n{arguments}` directly, e.g.
    # 'book_appointment\n{"provider_name": ...}' -- scan for an identifier
    # sitting directly against a balanced JSON object with no "name" key of
    # its own (a real "name" key would already have been caught above).
    for match in _LOOSE_CALL_RE.finditer(text):
        name = match.group(1)
        brace_start = match.start(2)
        candidates = _balanced_json_candidates(text[brace_start:])
        if not candidates:
            continue
        try:
            arguments = json.loads(candidates[0])
        except json.JSONDecodeError:
            continue
        if isinstance(arguments, dict) and "name" not in arguments:
            calls.append({"name": name, "arguments": arguments})
    return calls


def _call_close_token(text: str) -> str:
    """The call-end token the model first used after a <|CALL_START|> in `text`
    (</tool_call> or <|CALL_END|>), defaulting to <|CALL_END|>."""
    start = text.find(CALL_START)
    found = [(p, tok) for tok in (CALL_END, LEGACY_TOOL_CALL_END) if (p := text.find(tok, max(start, 0))) != -1]
    return min(found)[1] if found else CALL_END


class StepAudio2Adapter:
    def __init__(self, checkpoint: str, device: str = "cuda", verbose: bool = False):
        self.model = StepAudio2(checkpoint, device=device)
        self.verbose = verbose

    def run(self, record: Dict[str, Any], all_schemas: Dict[str, dict]) -> Dict[str, Any]:
        tool_schemas = tool_schemas_for(record, all_schemas)
        tools_schema = [_openai_tool_schema(s) for s in tool_schemas]
        audio_path = resolved_audio_path(record)

        messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_MESSAGE}]
        if tools_schema:
            messages.append({"role": "tool_json_schemas", "content": json.dumps(tools_schema, ensure_ascii=False)})
        messages.append({"role": "human", "content": [{"type": "audio", "audio": str(audio_path)}]})
        messages.append({"role": "assistant", "content": None})

        executor = MockToolExecutor(allowed_tools=available_tool_names(record))  # fresh per record, per executor.py's own contract
        raw_calls: List[Dict[str, Any]] = []
        seen_calls = set()
        raw_generation: Dict[str, Any] = {}
        final_response = ""

        # Multi-step loop: keep feeding tool results back while the model keeps
        # emitting new calls, so sequential tasks (call B needs call A's result)
        # can actually reach call B. Stops on a text-only turn, on a repeat of
        # an already-made call (don't retry the same request), or at MAX_STEPS.
        for step in range(1, MAX_STEPS + 1):
            _, text, _ = self.model(messages, **GENERATION_KWARGS)
            raw_generation[f"turn{step}"] = text
            final_response = text
            if self.verbose:
                print(f"  [step_audio2] turn{step} raw text: {text!r}")

            new_calls = []
            for call in _extract_tool_calls(text):
                key = (call.get("name", ""), json.dumps(call.get("arguments") or {}, sort_keys=True))
                if key not in seen_calls:
                    seen_calls.add(key)
                    new_calls.append(call)
            if not new_calls:
                break

            n_before = len(executor.tool_results())
            for call in new_calls:
                executor.call(call.get("name", ""), **(call.get("arguments") or {}))
            raw_calls.extend(new_calls)

            # Replace the placeholder with only the clean call span(s): the raw
            # text can trail off into hallucinated results/refusals after the
            # call. Close each call with whichever end token the model itself
            # used (it usually reaches for </tool_call> rather than the taught
            # <|CALL_END|>), so the history matches its own output as closely
            # as possible -- upstream's tool_call_test feeds raw tokens back.
            close_tok = _call_close_token(text)
            clean_calls = "".join(
                f"{CALL_START}function\n{c.get('name', '')}\n"
                f"{json.dumps(c.get('arguments') or {}, ensure_ascii=False)}{close_tok}"
                for c in new_calls
            )
            messages[-1] = {"role": "assistant", "content": clean_calls}
            for observation in executor.tool_results()[n_before:]:
                messages.append({
                    "role": "input",
                    "content": f"{OBSERVATION_PREFIX}\n{json.dumps(observation, ensure_ascii=False)}\n{OBSERVATION_SUFFIX}",
                })
            messages.append({"role": "assistant", "content": None})

        return build_result_record(
            record, normalize_calls(raw_calls), executor.tool_results(), final_response, raw_generation
        )
