#!/bin/bash
# Run the open-source models (VoxMind, Step-Audio2) on all 9 task types on one
# A100, then score them and rebuild results/summary.md.
#
# Do the one-time setup on the LOGIN node first (venvs + ~30 GB of checkpoints;
# compute nodes may not have fast internet):
#   source scripts/spartan/modules.sh
#   bash run.sh voxmind --setup-only
#   bash run.sh step_audio2 --setup-only
# then submit from the repo root:
#   sbatch scripts/spartan/run_open_source.sbatch
#
# Edit --account / --partition / --qos for your own allocation. Override the
# model list with MODELS="voxmind" sbatch ..., and pass extra run.sh options
# with RUN_ARGS="--tasks call_or_not --limit 2" sbatch ...

#SBATCH --job-name=fgsb-open-source
#SBATCH --account=punim2341
#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs
source scripts/spartan/modules.sh
nvidia-smi --query-gpu=name,memory.total --format=csv

STATUS=0
for m in ${MODELS:-voxmind step_audio2}; do
    # shellcheck disable=SC2086
    bash run.sh "$m" --skip-setup ${RUN_ARGS:-} || STATUS=1
done
exit $STATUS
