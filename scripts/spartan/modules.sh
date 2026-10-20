# Source this on Spartan (University of Melbourne HPC) before running run.sh:
#   source scripts/spartan/modules.sh
# Provides Python 3.12, FFmpeg (needed by torchcodec / torchaudio.load) and CUDA.
module purge
module load foss/2022a
module load GCCcore/13.3.0
module load Python/3.12.3
module load FFmpeg/7.0.2
module load CUDA/12.4.1
module load cuDNN/9.6.0.74-CUDA-12.4.1

# Keep ~/.local packages out of the venvs.
export PYTHONNOUSERSITE=1
# torchcodec dlopens FFmpeg's shared libraries.
FFMPEG_ROOT=$(dirname "$(dirname "$(command -v ffmpeg)")")
export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib:${LD_LIBRARY_PATH:-}"
export PYTHON=python3.12
