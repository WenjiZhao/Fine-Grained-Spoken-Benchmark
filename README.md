# Fine-Grained Spoken Agent Benchmark

A diagnostic benchmark for **tool calling from spoken input**. Each record is a spoken user
request (synthesized speech, in a half-duplex and a full-duplex rendering), a set of
available tools, and a gold action. Models run against a deterministic mock backend, so
every tool call is actually executed and multi-step behaviour can be scored.

The benchmark splits tool-use ability into **9 task types in 4 families**, so a failure can
be attributed to a specific skill instead of being reported as a single accuracy number.

| Family | Task type | What it tests |
|---|---|---|
| Intent Understanding | `call_or_not` | Whether to act at all: `call` vs `no_call` vs `cannot_answer` |
| | `tool_selection_hard_negative` | Picking the right tool among 5–8 genuinely confusable ones |
| | `parallel_tool_call` | Issuing two independent calls |
| Information Extraction & Argument Binding | `explicit_argument_binding` | Binding spoken values to the right arguments |
| | `resolvable_implicit_argument` | Resolving "where I am" / "today" / "my card" through a resolver tool first |
| | `missing_argument` | Asking for a missing required argument instead of guessing |
| Result Reuse | `sequential_tool_call` | A second call that depends on the first call's result |
| Self-Correction | `tool_self_correction` | The user changes their mind about *which tool* mid-utterance |
| | `argument_self_correction` | The user corrects *one argument value* mid-utterance |

> **Data in this repository.** The full benchmark has 723 records. This repository currently
> ships **2 example records per task type** (18 records, with both audio renderings) under
> `data/final_data/`, enough to run every model end to end. The full set will be released
> separately. Point `--data` at it once you have it.

## Models

| Model key | Model | Access | Input | Needs |
|---|---|---|---|---|
| `gpt_realtime` | OpenAI GPT-Realtime | Realtime API (websocket) | full-duplex stream | `OPENAI_API_KEY` |
| `gemini_live` | Google Gemini Live | Live API (`google-genai`) | full-duplex stream | `GEMINI_API_KEY` |
| `grok_voice` | xAI Grok Voice | Voice Agent API (websocket) | full-duplex stream | `XAI_API_KEY` |
| `nova_sonic` | Amazon Nova Sonic | Bedrock bidirectional stream | full-duplex stream | AWS credentials, Python ≥ 3.12 |
| `step_audio2` | [Step-Audio-2-mini](https://huggingface.co/stepfun-ai/Step-Audio-2-mini) | local `transformers` | half-duplex clip | 1 CUDA GPU (reference runs: A100) |
| `voxmind` | [VoxMind](https://huggingface.co/leungtianle/VoxMind) | local `transformers` | half-duplex clip | 1 CUDA GPU (reference runs: A100) |

Each model has its own adapter, `run_infer.py` CLI and `requirements.txt` under
`models/<model key>/`. Each one gets its own virtualenv, so their dependencies never clash. The open-source adapters vendor the
minimal upstream inference code they need (`models/*/vendor/`, see `VENDORED_FROM.md`); weights
are downloaded from Hugging Face on first run.

## Quick start

```bash
git clone https://github.com/WenjiZhao/Fine-Grained-Spoken-Benchmark.git
cd Fine-Grained-Spoken-Benchmark
cp .env.example .env          # fill in the keys you need (OPENAI_API_KEY is used by the judge)

bash run.sh gpt_realtime      # one API model
bash run.sh api               # all four API models
bash run.sh voxmind           # an open-source model (GPU; downloads the checkpoint)
bash run.sh all               # everything
```

`run.sh` does everything for each model:

1. creates `.venvs/<model>` and installs `models/<model>/requirements.txt` + `requirements.txt`;
2. downloads the checkpoint to `ckpt/` (open-source models only);
3. runs inference on every task type → `results/<model>/<model>_<task>.json`;
4. scores each file → `results/<model>/<model>_<task>_results.json` (+ `_errors.json`);
5. builds the tables for every model in `results/` → `results/summary.md`.

Useful options:

```bash
bash run.sh voxmind --tasks call_or_not,missing_argument --limit 1   # quick check
bash run.sh voxmind --setup-only        # only build the venv + download weights
bash run.sh gemini_live --skip-infer    # re-score existing outputs and rebuild tables
bash run.sh voxmind --data /path/to/full/final_data
DEVICE=cuda:1 CKPT_DIR=/scratch/ckpt bash run.sh step_audio2
```

Python ≥ 3.10 is required (3.12 recommended; `nova_sonic` needs ≥ 3.12). `run.sh` picks the
newest `python3.1x` on your `PATH`, or set `PYTHON=/path/to/python`.

## Running on Spartan (UniMelb HPC)

```bash
# login node: modules, venvs and checkpoints (compute nodes may lack fast internet)
source scripts/spartan/modules.sh
bash run.sh voxmind --setup-only
bash run.sh step_audio2 --setup-only

# GPU job: VoxMind + Step-Audio2 on all task types, then scoring + tables
sbatch scripts/spartan/run_open_source.sbatch

# optional: API models on a CPU node (put the keys in .env first)
bash run.sh api --setup-only && sbatch scripts/spartan/run_api_models.sbatch
```

Edit the `#SBATCH --account/--partition/--qos` lines for your allocation. Restrict a job with
`MODELS="voxmind" RUN_ARGS="--tasks call_or_not --limit 2" sbatch ...`. Logs go to `logs/`.

## Evaluation

`evaluation/evaluate.py` scores one output file; `evaluation/make_tables.py` aggregates
everything under `results/`. Both are called by `run.sh`, and you can also run them by hand:

```bash
python evaluation/evaluate.py --src_file results/voxmind/voxmind_call_or_not.json
python evaluation/make_tables.py --results results --out results/summary.md
```

Scoring is a cascade over the model's tool calls (as actually executed by `tools/executor.py`):

| Metric | Definition |
|---|---|
| **Action Acc** | Predicted action matches gold. `call` / `call_multiple` come from the number of calls made. For zero-call responses, an LLM judge decides between `no_call`, `clarify` and `cannot_answer`. |
| **Tool-F1** | Multiset F1 of called tool names vs gold (records that need a call). |
| **Schema Valid** | All required arguments present per `tools/tool_schemas.json` (records where the model called). |
| **Arg Acc** | Argument values match gold. Values equal up to case, punctuation or spacing pass directly; the rest go to an LLM judge that tolerates ASR-style mishearings but not wrong values. |
| **Strict Success** | All of the above correct for the record. |

Except for the self-correction tasks, only the model's final calls are scored, so preparatory
lookups or a retried first attempt don't count against it. The self-correction tasks keep every
call, because acting on the retracted request is exactly what they measure. Their 2×2 table
(decision completeness × correction distance) also reports a **failure-to-update rate**.

The LLM judge is OpenAI `gpt-4o` (`evaluation/prompts.py`, via `OPENAI_API_KEY`). Without
a key, the zero-call action and the non-exact argument values fall back to heuristics, so set it
for comparable numbers. Clarify-quality and result-grounding judges exist in the code but are
disabled. Eight records of the full set with known data issues are excluded from the tables
(`EXCLUDED_IDS` in `make_tables.py`).

## Reference results (full set, 715 records)

Macro-average over the 9 task types, ± 95 % bootstrap CI.

| Metric | GPT-Realtime | Gemini-Live | Grok-Voice | Nova-Sonic | Step-Audio2 | VoxMind |
|---|---|---|---|---|---|---|
| Strict Success | 0.926 ±0.019 | 0.924 ±0.021 | **0.937** ±0.018 | 0.833 ±0.030 | 0.488 ±0.023 | 0.577 ±0.032 |
| Action Acc | 0.948 ±0.017 | 0.952 ±0.017 | **0.960** ±0.014 | 0.921 ±0.018 | 0.538 ±0.019 | 0.731 ±0.022 |
| Tool-F1 | 0.961 ±0.013 | 0.972 ±0.012 | **0.981** ±0.010 | 0.971 ±0.012 | 0.857 ±0.012 | 0.929 ±0.013 |
| Schema Valid | **1.000** ±0.000 | 0.999 ±0.002 | **1.000** ±0.000 | **1.000** ±0.000 | 0.999 ±0.002 | 0.986 ±0.007 |
| Arg Acc | 0.943 ±0.017 | 0.952 ±0.015 | **0.972** ±0.013 | 0.889 ±0.029 | 0.761 ±0.019 | 0.780 ±0.029 |

Strict Success by task type:

| Task (n) | GPT-Realtime | Gemini-Live | Grok-Voice | Nova-Sonic | Step-Audio2 | VoxMind |
|---|---|---|---|---|---|---|
| call_or_not (89) | 0.933 | **0.966** | 0.944 | 0.876 | 0.562 | 0.551 |
| tool_selection_hard_negative (48) | 0.958 | 0.917 | **0.979** | 0.875 | 0.938 | 0.875 |
| parallel_tool_call (45) | **1.000** | 0.978 | 0.956 | 0.822 | 0.178 | 0.711 |
| explicit_argument_binding (90) | 0.956 | 0.989 | **1.000** | 0.911 | 0.989 | 0.889 |
| resolvable_implicit_argument (76) | **0.947** | 0.842 | 0.882 | 0.855 | 0.000 | 0.158 |
| missing_argument (90) | **0.967** | 0.944 | 0.922 | 0.900 | 0.022 | 0.056 |
| sequential_tool_call (48) | 0.812 | 0.917 | **0.938** | 0.729 | 0.042 | 0.521 |
| tool_self_correction (121) | 0.843 | **0.884** | 0.868 | 0.744 | 0.785 | 0.719 |
| argument_self_correction (108) | 0.917 | 0.880 | **0.944** | 0.787 | 0.880 | 0.713 |

API models are nondeterministic (server-side VAD and sampling), so individual records can flip
between runs. VoxMind is run with `--response-mode speech` (its native agent mode; only the
decoded text is scored).

### Protocol notes

Each adapter is kept exactly as it was used for the reference results above:

- The four API adapters append a short per-task rule to the system prompt for `call_or_not`,
  `missing_argument` and `resolvable_implicit_argument` (`TASK_TYPE_SYSTEM_SUFFIXES` in
  `models/common/dataset.py`). The open-source adapters do not.
- The open-source adapters restrict the mock executor to the record's `available_tools`
  (`MockToolExecutor(allowed_tools=...)`), so a call to a tool that wasn't offered fails. The API
  adapters use the unrestricted executor.

## Data format

`data/final_data/<task_type>.json` is a list of records:

```jsonc
{
  "id": "sequential_tool_call_0001",
  "task_type": "sequential_tool_call",
  "available_tools": [{"name": "events_3_find_events"}, {"name": "events_3_buy_event_tickets"}],
  "transcript": "Find me a comedy event in Chicago and buy 2 tickets for the first event that comes back. ...",
  "gold": {
    "action": "call_multiple",          // call | call_multiple | no_call | clarify | cannot_answer
    "calls": [
      {"tool": "events_3_find_events", "arguments": {"location": "Chicago", "category": "comedy"}},
      {"tool": "events_3_buy_event_tickets",
       "arguments": {"event_id": "<result:0.results.0.event_id>", "quantity": 2}}  // reference into call 0's result
    ]
  },
  "half_duplex_audio_paths": ["data/final_data/half_duplex/sequential_tool_call_audio/sequential_tool_call_0001.wav"],
  "full_duplex_audio_paths": ["data/final_data/full_duplex/sequential_tool_call_audio/sequential_tool_call_0001.wav"],
  "annotations": { ... }                // task-specific: speech phenomena, correction spans, completeness, distance, ...
}
```

- **Half-duplex audio** is just the request. Offline models get the whole clip at once.
- **Full-duplex audio** is the request followed by ~20 s of silence. Realtime APIs receive it as a
  real-time-paced stream so their server-side VAD decides when the user's turn has ended.
- Single-call golds use `{"action", "tool", "arguments"}` directly. Multi-call golds use `calls`.

The tool definitions shown to models are in `tools/tool_schemas.json` (34 tools), backed by
deterministic mock data in `tools/db/`. `data/tool_pool.json` documents why each tool was chosen.

## Adding a model

Copy one of the API adapters (e.g. `models/grok_voice/`), implement the provider session, and
emit one record per input via `models/common/result_schema.build_result_record` with
`actual_tool_calls`, `tool_results` and `final_response`. Then add `requirements.txt` and run
`bash run.sh <your_model>`. Anything that writes the same output schema can be scored by
`evaluation/evaluate.py`.

## Repository layout

```
run.sh                      one-click pipeline (setup -> infer -> score -> tables)
requirements.txt            evaluation dependencies (installed into every model venv)
.env.example                API keys template
data/
  final_data/<task>.json    example records (2 per task type)
  final_data/{half,full}_duplex/<task>_audio/*.wav
  tool_pool.json            tool selection rationale
tools/
  tool_schemas.json         tool definitions shown to the models and used for scoring
  executor.py, db/          deterministic mock backend
models/
  common/                   dataset loading, duplex audio streaming, result schema
  gpt_realtime/ gemini_live/ grok_voice/ nova_sonic/ step_audio2/ voxmind/
                            adapter.py, run_infer.py, requirements.txt (+ vendor/ for open-source)
evaluation/
  evaluate.py, scorer.py, prompts.py   per-record scoring + LLM judge
  make_tables.py                       result tables -> results/summary.md
scripts/spartan/            module setup + SLURM jobs for Spartan
```
