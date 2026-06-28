#!/bin/bash
# Five-fold image/metadata fusion, reusing saved CNN checkpoints.
#SBATCH --job-name=skyresid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX:-./.conda/skyfinder}"
export XDG_CACHE_HOME="${SCRATCH_ROOT:-/gscratch/stf/${USER}}/.cache"
export TORCH_HOME="${XDG_CACHE_HOME}/torch"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export PYTHONUNBUFFERED=1
mkdir -p slurm/logs
python -u -m skyfinder.analysis.residual_probe --variant "${VARIANT:-dinov2_vits14}" --img-dir data/images
