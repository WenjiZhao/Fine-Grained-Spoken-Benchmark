"""Entry point: run the OpenAI Realtime API (gpt-realtime) adapter over the curated
dataset, streaming each record's full_duplex audio in real time (see
models/common/duplex_audio.py and adapter.py's module docstrings for why full-duplex
audio specifically, and why real-time pacing).

Usage:
    export OPENAI_API_KEY=sk-...
    python -m models.gpt_realtime.run_infer \\
        [--model gpt-realtime] [--dataset data/] [--output results/gpt_realtime_results.json] \\
        [--limit 10] [--verbose]

(--dataset accepts the data/ root -- loads every task_type's dataset.json -- or a
single data/<category>/<task_type>/dataset.json to run just one.)

This adapter is verified end to end against a live session (see adapter.py's module
docstring and README.md) -- the core connect/tool-call/final-answer path is confirmed
correct, but parallel/multiple function calls in one round and error paths beyond
what that one run triggered are not. Run --limit 1 --verbose first on anything new
and read the printed event stream before trusting a full run.

Then evaluate with the shared evaluator (unchanged, no per-model branching):
    python evaluation/evaluate.py --src_file results/gpt_realtime_results.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[2]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from models.common.dataset import DEFAULT_DATASET_PATH, DEFAULT_SCHEMAS_PATH, load_dataset, load_tool_schemas  # noqa: E402
from models.gpt_realtime.adapter import GPTRealtimeAdapter  # noqa: E402

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-realtime", help="Realtime API model snapshot")
    p.add_argument("--api-key", default=None, help="defaults to $OPENAI_API_KEY")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    p.add_argument("--output", default=str(BENCH_ROOT / "results" / f"gpt_realtime_results_{timestamp}.json"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    dataset = load_dataset(Path(args.dataset))
    if args.limit:
        dataset = dataset[: args.limit]
    schemas = load_tool_schemas(DEFAULT_SCHEMAS_PATH)

    print(f"Connecting to OpenAI Realtime API ({args.model}) ...")
    adapter = GPTRealtimeAdapter(model=args.model, api_key=args.api_key, verbose=args.verbose)
    print(f"Running {len(dataset)} records ...")

    results = []
    for i, record in enumerate(dataset, start=1):
        t0 = time.time()
        try:
            results.append(adapter.run(record, schemas))
        except Exception as exc:  # a single bad record shouldn't kill the whole run
            print(f"[{i}/{len(dataset)}] {record.get('id')} FAILED: {exc}")
            results.append({
                "id": record.get("id"), "task_type": record.get("task_type"),
                "query": record.get("transcript"), "gold": record.get("gold"),
                "actual_tool_calls": [], "tool_results": None,
                "final_response": f"[adapter error: {exc}]",
            })
            continue
        print(f"[{i}/{len(dataset)}] {record.get('id')} ({time.time() - t0:.1f}s)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(results)} results -> {out_path}")


if __name__ == "__main__":
    main()
