#!/bin/bash
# One array task fine-tunes DINOv2 ViT-S/14 on one LOCO fold.
#SBATCH --job-name=skydinoft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX:-./.conda/skyfinder}"
export XDG_CACHE_HOME="${SCRATCH_ROOT:-/gscratch/stf/${USER}}/.cache"
export TORCH_HOME="${XDG_CACHE_HOME}/torch"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
mkdir -p slurm/logs results_dino_finetune
python -u dino_finetune.py --config "${CONFIG:-configs/dino_finetune.yaml}" --task-id "${SLURM_ARRAY_TASK_ID:?array task required}"
