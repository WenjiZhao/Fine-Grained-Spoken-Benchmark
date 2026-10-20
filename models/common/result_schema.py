"""The one record shape every model adapter must produce.

evaluation/scorer.py (copied verbatim from ../evaluation/) reads exactly:
    record["query"], record["gold"], record.get("actual_tool_calls", []),
    record.get("tool_results"), record.get("final_response", ...)
with actual_tool_calls entries shaped {"name": ..., "args": ...} (score_schema
does call["name"] / call.get("args", {})) and tool_results shaped
{"name", "args", "result"} (the exact shape tools/executor.py's
MockToolExecutor.tool_results() already returns).

Because every adapter normalizes into this same dict, evaluation/evaluate.py
runs unmodified against any model's results.json -- no per-model evaluator
needed.
"""

from typing import Any, Dict, List, Optional


def normalize_calls(raw_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """[{"name", "arguments"}] (VoxMind / VoiceChat's native shape) -> [{"name", "args"}]."""
    return [{"name": c.get("name", ""), "args": c.get("arguments", {}) or {}} for c in raw_calls]


def build_result_record(
    record: Dict[str, Any],
    actual_tool_calls: List[Dict[str, Any]],
    tool_results: Optional[List[dict]],
    final_response: str,
    raw_generation: Optional[Any] = None,
    tool_call_trace: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    result = {
        "id": record.get("id"),
        "task_type": record.get("task_type"),
        "query": record.get("transcript"),
        "gold": record.get("gold"),
        "actual_tool_calls": actual_tool_calls,
        "tool_results": tool_results,
        "final_response": final_response,
    }
    if raw_generation is not None:
        # Debug-only: the model's raw decoded text per turn (think block, <tool_call>
        # block, everything) -- evaluation/scorer.py never reads this key.
        result["raw_generation"] = raw_generation
    if tool_call_trace is not None:
        # Debug-only, same as raw_generation: one entry per tool-call token the model emitted,
        # INCLUDING ones that never made it into actual_tool_calls (malformed/superseded
        # attempts), with "model_emitted_at"/"server_received_at" timestamps (absolute and
        # relative to the record's own audio start) for latency inspection. Currently only
        # populated by models/voicechat_interactive/adapter.py -- evaluation/scorer.py never
        # reads this key.
        result["tool_call_trace"] = tool_call_trace
    return result
