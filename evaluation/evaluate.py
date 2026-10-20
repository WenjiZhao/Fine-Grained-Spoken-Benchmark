"""
CLI entry point.

Usage:
    python evaluate.py --src_file results.json
    python evaluate.py --src_file results.json --schemas custom_schemas.json
    python evaluate.py --src_file results.json --no_grounding   # skip the grounding judge

Input: a JSON list of records, one per evaluated example --
    {"id", "task_type", "query", "gold", "actual_tool_calls",
     "tool_results" (optional, from tools/executor.py), "final_response"}
Output: <src_file>_results.json (per-record scores) + a printed summary;
with --no_grounding, <src_file>_nogrounding_results.json instead (grounding is
not judged, so it never fails a record's strict_success).

Note: clarify and grounding are currently disabled in scorer.py, so with or
without --no_grounding both fields are None and neither counts toward strict_success.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scorer import DEFAULT_SCHEMAS_PATH, evaluate_record, failed_tiers, load_schemas


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def aggregate(scored: list) -> dict:
    def summarize(rows):
        return {
            "n": len(rows),
            "action_accuracy": _mean([r["action_correct"] for r in rows]),
            "tool_selection_f1": _mean([r["tool_selection"]["f1"] for r in rows if r["tool_selection"]]),
            "call_rate": _mean([r.get("called") for r in rows]),
            "schema_valid": _mean([r["schema_valid"] for r in rows]),
            "schema_valid_when_called": _mean([r.get("schema_valid_when_called") for r in rows]),
            "argument_accuracy": _mean([r["argument_accuracy"] for r in rows]),
            "clarify_quality": _mean([r["clarify_quality"] for r in rows]),
            "result_grounding": _mean([r["result_grounding"] for r in rows]),
            "n_grounding_judged": sum(r["result_grounding"] is not None for r in rows),
            "strict_success": _mean([r.get("strict_success") for r in rows]),
            "n_judge_failures": sum(r.get("n_judge_failures") or 0 for r in rows),
        }

    by_task = defaultdict(list)
    for r in scored:
        by_task[r["task_type"]].append(r)

    return {"overall": summarize(scored), "by_task_type": {k: summarize(v) for k, v in by_task.items()}}


def collect_errors(records: list, scored: list) -> list:
    """Pull out every record that failed at least one evaluated tier, paired
    with enough of the original record to inspect why."""
    errors = []
    for record, score in zip(records, scored):
        tiers = failed_tiers(score)
        if not tiers:
            continue
        errors.append({
            "id": record.get("id"),
            "task_type": record.get("task_type"),
            "failed_tiers": tiers,
            "query": record.get("query"),
            "gold": record.get("gold"),
            "actual_tool_calls": record.get("actual_tool_calls"),
            "tool_results": record.get("tool_results"),
            "final_response": record.get("final_response", record.get("generated_text", "")),
            "scores": score,
        })
    return errors


def summarize_errors(errors: list) -> dict:
    by_tier = defaultdict(int)
    by_task = defaultdict(int)
    for e in errors:
        by_task[e["task_type"]] += 1
        for t in e["failed_tiers"]:
            by_tier[t] += 1
    return {"n_errors": len(errors), "by_failed_tier": dict(by_tier), "by_task_type": dict(by_task)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", required=True)
    parser.add_argument("--schemas", default=str(DEFAULT_SCHEMAS_PATH))
    parser.add_argument("--no_grounding", action="store_true", help="skip the result-grounding judge")
    args = parser.parse_args()

    records = json.loads(Path(args.src_file).read_text())
    schemas = load_schemas(Path(args.schemas))

    scored = [evaluate_record(r, schemas, grounding=not args.no_grounding) for r in records]
    report = {"scores": scored, "summary": aggregate(scored)}

    errors = collect_errors(records, scored)
    error_report = {"summary": summarize_errors(errors), "errors": errors}

    stem = args.src_file.rsplit(".json", 1)[0] + ("_nogrounding" if args.no_grounding else "")
    out_path = stem + "_results.json"
    errors_path = stem + "_errors.json"
    Path(out_path).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    Path(errors_path).write_text(json.dumps(error_report, indent=2, ensure_ascii=False))

    print(json.dumps(report["summary"], indent=2))
    print(f"\nWrote {out_path}")
    print(json.dumps(error_report["summary"], indent=2))
    print(f"Wrote {errors_path}")


if __name__ == "__main__":
    main()
