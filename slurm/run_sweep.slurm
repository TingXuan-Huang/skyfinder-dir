#!/bin/bash
# One array task = one (experiment, fold) cell of the sweep.
# Submit via: bash submit_sweep.sh [concurrent] [config]
#
# Hyak-specific notes:
# - Account/partition/GPU flags are supplied by submit_sweep.sh so they can be
#   changed without editing this file.
# - On Klone, `module load conda` is available on compute nodes, not login nodes.
#   Build the env first with `sbatch slurm/setup_env.slurm`.
#SBATCH --job-name=skyfinder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err
#SBATCH --export=ALL

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/main.yaml}"
ENV_PREFIX="${ENV_PREFIX:-${REPO_ROOT}/.conda/skyfinder}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/gscratch/stf/${USER}}"

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh 2>/dev/null || true
fi
module load conda
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

export XDG_CACHE_HOME="${SCRATCH_ROOT}/.cache"
export TORCH_HOME="${XDG_CACHE_HOME}/torch"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

mkdir -p "${TORCH_HOME}" slurm/logs results

echo "[task] ${SLURM_ARRAY_TASK_ID:-none}  host $(hostname)  $(date)"
nvidia-smi || true
python - <<'PY'
import torch
print(f"[torch] version={torch.__version__} cuda_available={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
PY

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  python -u run_sweep.py --config "${CONFIG}" --task-id "${SLURM_ARRAY_TASK_ID}" --skip-existing
else
  python -u run_sweep.py --config "${CONFIG}" --skip-existing
fi

echo "[end] $(date)"
