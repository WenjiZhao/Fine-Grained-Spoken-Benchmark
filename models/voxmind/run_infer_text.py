"""Entry point 1 (text variant): run the VoxMind text adapter over the
curated dataset, feeding each record's ground-truth transcript directly as
text instead of its TTS audio. Isolates ASR/audio-understanding errors from
tool-calling errors -- compare against results/voxmind_results_*.json.

Usage:
    python -m models.voxmind.run_infer_text --checkpoint /path/to/VoxMind \\
        [--dataset data/] [--output results/voxmind_text_results.json] \\
        [--limit 10] [--device cuda]

(--dataset accepts the data/ root -- loads every task_type's dataset.json --
or a single data/<category>/<task_type>/dataset.json to run just one.)

Then evaluate with the shared evaluator (unchanged, no per-model branching):
    python evaluation/evaluate.py --src_file results/voxmind_text_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[2]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from models.common.dataset import DEFAULT_DATASET_PATH, DEFAULT_SCHEMAS_PATH, load_dataset, load_tool_schemas  # noqa: E402
from models.voxmind.text_adapter import VoxMindTextAdapter  # noqa: E402
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="local VoxMind HF checkpoint dir")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    p.add_argument("--output", default=str(BENCH_ROOT / "results" / f"voxmind_text_results_{timestamp}.json"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dataset = load_dataset(Path(args.dataset))
    if args.limit:
        dataset = dataset[: args.limit]
    schemas = load_tool_schemas(DEFAULT_SCHEMAS_PATH)

    print(f"Loading VoxMind from {args.checkpoint} ...")
    adapter = VoxMindTextAdapter(args.checkpoint, device=args.device)
    print(f"Running {len(dataset)} records (text input) ...")

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
