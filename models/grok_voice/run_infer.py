"""Entry point: run the xAI Grok Voice Agent API adapter over the curated dataset,
streaming each record's full_duplex audio in real time (see
models/common/duplex_audio.py and adapter.py's module docstrings).

Usage:
    python -m models.grok_voice.run_infer \\
        [--model grok-voice-latest] [--voice eve] \\
        [--dataset data/] [--output results/grok_voice_results.json] \\
        [--limit 10] [--verbose]

The API key comes from --api-key, else $XAI_API_KEY, else the `grok_key` line in
.env (repo root).

Run --limit 1 --verbose first and read the printed event stream before trusting a
full run.

Then evaluate with the shared evaluator:
    python evaluation/evaluate.py --src_file results/grok_voice_results.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[2]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from models.common.dataset import DEFAULT_DATASET_PATH, DEFAULT_SCHEMAS_PATH, load_dataset, load_tool_schemas  # noqa: E402
from models.grok_voice.adapter import DEFAULT_MODEL, DEFAULT_VOICE, GrokVoiceAdapter  # noqa: E402

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_api_key(cli_key):
    """--api-key > $XAI_API_KEY > `grok_key = ...` in .env (repo root)."""
    if cli_key:
        return cli_key
    if os.environ.get("XAI_API_KEY"):
        return os.environ["XAI_API_KEY"]
    env_path = BENCH_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() in ("grok_key", "XAI_API_KEY"):
                return v.strip().strip("'\"")
    raise SystemExit("No xAI API key: pass --api-key, set $XAI_API_KEY, or add grok_key to .env (repo root)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL, help="Grok Voice model id")
    p.add_argument("--voice", default=DEFAULT_VOICE)
    p.add_argument("--api-key", default=None, help="defaults to $XAI_API_KEY, then grok_key in .env")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    p.add_argument("--output", default=str(BENCH_ROOT / "results" / f"grok_voice_results_{timestamp}.json"))
    p.add_argument("--offset", type=int, default=0, help="skip the first N records")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    dataset = load_dataset(Path(args.dataset))[args.offset:]
    if args.limit:
        dataset = dataset[: args.limit]
    schemas = load_tool_schemas(DEFAULT_SCHEMAS_PATH)

    print(f"Connecting to xAI Grok Voice Agent API ({args.model}) ...")
    adapter = GrokVoiceAdapter(model=args.model, api_key=_load_api_key(args.api_key),
                               voice=args.voice, verbose=args.verbose)
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
