#!/bin/bash
# Run the four API models (GPT-Realtime, Gemini-Live, Grok-Voice, Nova-Sonic) on a
# CPU node. They only stream audio to the providers, so no GPU is needed. Put the
# API keys in .env at the repo root (see .env.example), then:
#   source scripts/spartan/modules.sh && bash run.sh api --setup-only   # login node
#   sbatch scripts/spartan/run_api_models.sbatch

#SBATCH --job-name=fgsb-api-models
#SBATCH --account=punim2341
#SBATCH --partition=sapphire
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs
source scripts/spartan/modules.sh

# shellcheck disable=SC2086
bash run.sh ${MODELS:-api} --skip-setup ${RUN_ARGS:-}
