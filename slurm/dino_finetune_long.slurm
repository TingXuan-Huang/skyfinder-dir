#!/bin/bash
# Longer-schedule DINOv2 ViT-S/14 partial fine-tune (40 epochs) with per-epoch
# held-out test logging. One array task = one LOCO fold. Mirrors the resources
# the 20-epoch run actually used (ckpt-all / ckpt-stf / H200). Resumes from
# <run_name>_last.pt after preemption, so any single segment length is fine.
# Override any directive on the sbatch command line if your allocation differs.
#SBATCH --job-name=skydinoftL
#SBATCH --account=ckpt-stf
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Any idle ckpt GPU (A100/H200/L40S/A40/...) so the job starts on whichever frees
# first. This ViT-S/14 partial finetune is small and data-loading-bound, so card
# type barely matters. Pin a type on the sbatch line if you want, e.g.
# `sbatch --gpus=a100:1 ...` (overrides this).
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
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
mkdir -p slurm/logs results_dino_finetune_long
python -u dino_finetune.py --config "${CONFIG:-configs/dino_finetune_long.yaml}" --task-id "${SLURM_ARRAY_TASK_ID:?array task required}"
