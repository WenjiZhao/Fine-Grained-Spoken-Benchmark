"""Entry point: run the Amazon Nova 2.5 Sonic (Bedrock bidirectional stream) adapter over the curated dataset,
streaming each record's full_duplex audio in real time (see
models/common/duplex_audio.py and adapter.py's module docstrings).

Usage:
    python -m models.nova_sonic.run_infer \\
        [--model amazon.nova-2-5-sonic] [--region us-east-1] [--voice matthew] \\
        [--dataset data/] [--output results/nova_sonic_results.json] \\
        [--limit 10] [--verbose]

Run with this model's own venv (aws_sdk_bedrock_runtime needs Python >= 3.12):
    models/nova_sonic/.venv/bin/python -m models.nova_sonic.run_infer ...

Auth: IAM access keys (SigV4) from $AWS_ACCESS_KEY_ID/$AWS_SECRET_ACCESS_KEY or the
`nova_access_ID` / `nova_access_key` lines in .env (repo root); otherwise a Bedrock
API key (--api-key, $AWS_BEARER_TOKEN_BEDROCK, or `nova_key`). See _load_api_key.

Run --limit 1 --verbose first and read the printed event stream before trusting a
full run.

Then evaluate with the shared evaluator:
    python evaluation/evaluate.py --src_file results/nova_sonic_results.json
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
from models.nova_sonic.adapter import DEFAULT_MODEL, DEFAULT_REGION, DEFAULT_VOICE, NovaSonicAdapter  # noqa: E402

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_env_file() -> dict:
    env_path = BENCH_ROOT / ".env"
    values = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip().strip("'\"")
    return values


def _load_api_key(cli_key):
    """Pick Bedrock credentials. Returns a Bedrock API key for Bearer auth, or None
    to use SigV4 with $AWS_ACCESS_KEY_ID/$AWS_SECRET_ACCESS_KEY.

    --api-key > IAM keys already in the environment > IAM keys in .env (repo root)
    (nova_access_ID + nova_access_key, exported to $AWS_ACCESS_KEY_ID/...) >
    $AWS_BEARER_TOKEN_BEDROCK > nova_key in .env. IAM keys go first because this
    account's org SCP denies bedrock:CallWithBearerToken, so API keys are rejected."""
    if cli_key:
        return cli_key
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return None
    env = _read_env_file()
    key_id = env.get("nova_access_ID") or env.get("nova_access_key_id") or env.get("AWS_ACCESS_KEY_ID")
    secret = (env.get("nova_access_key") or env.get("nove_access_key")
              or env.get("nova_secret_access_key") or env.get("AWS_SECRET_ACCESS_KEY"))
    if key_id and secret:
        os.environ["AWS_ACCESS_KEY_ID"] = key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = secret
        return None
    if os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        return os.environ["AWS_BEARER_TOKEN_BEDROCK"]
    if env.get("nova_key") or env.get("AWS_BEARER_TOKEN_BEDROCK"):
        return env.get("nova_key") or env.get("AWS_BEARER_TOKEN_BEDROCK")
    raise SystemExit("No Bedrock credentials: set nova_access_ID + nova_access_key (IAM) or nova_key "
                     "(API key) in .env (repo root), or the matching AWS_* environment variables")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL, help="Bedrock model id")
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--voice", default=DEFAULT_VOICE)
    p.add_argument("--api-key", default=None, help="Bedrock API key; defaults to $AWS_BEARER_TOKEN_BEDROCK, then nova_key in .env")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    p.add_argument("--output", default=str(BENCH_ROOT / "results" / f"nova_sonic_results_{timestamp}.json"))
    p.add_argument("--offset", type=int, default=0, help="skip the first N records")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    dataset = load_dataset(Path(args.dataset))[args.offset:]
    if args.limit:
        dataset = dataset[: args.limit]
    schemas = load_tool_schemas(DEFAULT_SCHEMAS_PATH)

    print(f"Connecting to Amazon Nova 2.5 Sonic (Bedrock bidirectional stream) ({args.model}) ...")
    adapter = NovaSonicAdapter(model=args.model, api_key=_load_api_key(args.api_key),
                               region=args.region, voice=args.voice, verbose=args.verbose)
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
