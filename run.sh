#!/usr/bin/env bash
# One-click pipeline: environment -> (checkpoint) -> inference -> scoring -> tables.
#
#   bash run.sh <model> [options]
#
# Models:
#   gpt_realtime | gemini_live | grok_voice | nova_sonic   (API, no GPU)
#   step_audio2  | voxmind                                 (open-source, GPU)
#   api   -> all four API models        all -> every model
#
# Options:
#   --tasks a,b,c     task types to run (default: all 9)
#   --limit N         only the first N records of each task type
#   --data DIR        dataset directory (default: data/final_data)
#   --setup-only      create the venv (+ download the checkpoint) and stop
#   --skip-setup      reuse the existing venv as-is
#   --skip-infer      only re-score existing results and rebuild the tables
#
# Environment (read from .env at the repo root if present; see .env.example):
#   OPENAI_API_KEY    LLM judge used by evaluation/ (also gpt_realtime itself)
#   GEMINI_API_KEY / XAI_API_KEY / AWS_* or nova_* keys for the API models
#   PYTHON            interpreter used to create venvs (default: newest python3.1x found)
#   VENV_DIR          where venvs live (default: .venvs)
#   CKPT_DIR          where checkpoints live (default: ckpt)
#   DEVICE            torch device for open-source models (default: cuda)

set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)

ALL_TASKS=(call_or_not tool_selection_hard_negative parallel_tool_call
           explicit_argument_binding resolvable_implicit_argument missing_argument
           sequential_tool_call tool_self_correction argument_self_correction)
API_MODELS=(gpt_realtime gemini_live grok_voice nova_sonic)
OSS_MODELS=(step_audio2 voxmind)

usage() { sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
[ $# -ge 1 ] || usage 1
case "$1" in -h|--help) usage 0 ;; esac

MODEL_ARG=$1; shift
TASKS=("${ALL_TASKS[@]}"); LIMIT=""; DATA_DIR="data/final_data"
SETUP_ONLY=0; SKIP_SETUP=0; SKIP_INFER=0
while [ $# -gt 0 ]; do
    case "$1" in
        --tasks) IFS=',' read -r -a TASKS <<< "$2"; shift 2 ;;
        --limit) LIMIT=$2; shift 2 ;;
        --data) DATA_DIR=$2; shift 2 ;;
        --setup-only) SETUP_ONLY=1; shift ;;
        --skip-setup) SKIP_SETUP=1; shift ;;
        --skip-infer) SKIP_INFER=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 1 ;;
    esac
done

case "$MODEL_ARG" in
    api) MODELS=("${API_MODELS[@]}") ;;
    all) MODELS=("${API_MODELS[@]}" "${OSS_MODELS[@]}") ;;
    *) [ -f "models/$MODEL_ARG/run_infer.py" ] || { echo "unknown model: $MODEL_ARG" >&2; usage 1; }
       MODELS=("$MODEL_ARG") ;;
esac

if [ -f .env ]; then set -a; . ./.env; set +a; fi
VENV_DIR=${VENV_DIR:-$ROOT/.venvs}
CKPT_DIR=${CKPT_DIR:-$ROOT/ckpt}
DEVICE=${DEVICE:-cuda}

# ---------------------------------------------------------------------------
min_python() {  # nova_sonic's Bedrock SDK needs 3.12; everything else 3.10
    [ "$1" = nova_sonic ] && echo 12 || echo 10
}

pick_python() {
    local need=$1 cand
    for cand in ${PYTHON:-} python3.13 python3.12 python3.11 python3.10 python3; do
        command -v "$cand" >/dev/null 2>&1 || continue
        if "$cand" -c "import sys; sys.exit(sys.version_info < (3, $need))" 2>/dev/null; then
            echo "$cand"; return 0
        fi
    done
    return 1
}

checkpoint_of() {
    case "$1" in
        voxmind) echo "leungtianle/VoxMind $CKPT_DIR/VoxMind" ;;
        step_audio2) echo "stepfun-ai/Step-Audio-2-mini $CKPT_DIR/Step-Audio-2-mini" ;;
    esac
}

setup_model() {
    local m=$1 venv="$VENV_DIR/$1" py need
    if [ ! -x "$venv/bin/python" ]; then
        need=$(min_python "$m")
        py=$(pick_python "$need") || { echo "FATAL: $m needs Python >= 3.$need (set PYTHON=...)" >&2; return 1; }
        echo "== [$m] creating venv with $py -> $venv"
        "$py" -m venv "$venv" || return 1
    fi
    echo "== [$m] installing requirements"
    "$venv/bin/python" -m pip install -q --upgrade pip || return 1
    "$venv/bin/python" -m pip install -q -r "models/$m/requirements.txt" -r requirements.txt || return 1

    local ck; ck=$(checkpoint_of "$m")
    if [ -n "$ck" ]; then
        set -- $ck
        if [ ! -f "$2/config.json" ]; then
            echo "== [$m] downloading checkpoint $1 -> $2"
            "$venv/bin/python" -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], local_dir=sys.argv[2])" "$1" "$2" || return 1
        fi
    fi
}

infer_args() {
    case "$1" in
        voxmind) echo "--checkpoint $CKPT_DIR/VoxMind --device $DEVICE --response-mode speech" ;;
        step_audio2) echo "--checkpoint $CKPT_DIR/Step-Audio-2-mini --device $DEVICE" ;;
    esac
}

FAILED=()
for m in "${MODELS[@]}"; do
    echo "================ $m ================"
    if [ "$SKIP_SETUP" -eq 0 ]; then
        setup_model "$m" || { echo "FAILED: setup $m"; FAILED+=("$m/setup"); continue; }
    fi
    [ "$SETUP_ONLY" -eq 1 ] && continue
    PY="$VENV_DIR/$m/bin/python"
    [ -x "$PY" ] || { echo "FATAL: no venv for $m at $VENV_DIR/$m (drop --skip-setup)"; FAILED+=("$m/venv"); continue; }
    mkdir -p "results/$m"

    for t in "${TASKS[@]}"; do
        ds="$DATA_DIR/$t.json"; out="results/$m/${m}_$t.json"
        [ -f "$ds" ] || { echo "skip $t: $ds not found"; continue; }
        if [ "$SKIP_INFER" -eq 0 ]; then
            echo "-- [$m] infer $t"
            # shellcheck disable=SC2046
            "$PY" -u -m "models.$m.run_infer" --dataset "$ds" --output "$out" \
                ${LIMIT:+--limit "$LIMIT"} $(infer_args "$m") \
                || { echo "FAILED: $m infer $t"; FAILED+=("$m/$t"); continue; }
        fi
        [ -f "$out" ] || { echo "skip scoring $t: $out not found"; continue; }
        echo "-- [$m] score $t"
        "$PY" evaluation/evaluate.py --src_file "$out" > "results/$m/${m}_${t}_eval.log" \
            || { echo "FAILED: $m score $t (see results/$m/${m}_${t}_eval.log)"; FAILED+=("$m/$t/score"); }
    done
done

if [ "$SETUP_ONLY" -eq 0 ]; then
    [ -n "${OPENAI_API_KEY:-}" ] || echo "WARNING: OPENAI_API_KEY not set -- the LLM judge was skipped, so action (no_call/clarify/cannot_answer) and non-exact argument values were scored by fallback heuristics."
    echo "================ tables ================"
    PY_TABLES=$(pick_python 10 || echo python3)
    "$PY_TABLES" evaluation/make_tables.py --results results --data "$DATA_DIR" --out results/summary.md | tail -n 1
fi

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "Completed with failures:"; printf '  - %s\n' "${FAILED[@]}"; exit 1
fi
echo "Done."
