#!/bin/bash
# CPU-only recovery for runs that completed training but failed writing JSON.
# Submit example:
#   sbatch --array=0-1,20-38%6 slurm/recover_json_cpu.slurm
# This keeps *_last.pt by default.
#SBATCH --job-name=sky-json-cpu
#SBATCH --account=stf
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

mkdir -p slurm/logs results

echo "[recover-task] ${SLURM_ARRAY_TASK_ID:-none} host $(hostname) $(date)"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  python -u recover_results_from_last.py --config "${CONFIG}" --task-id "${SLURM_ARRAY_TASK_ID}"
else
  python -u recover_results_from_last.py --config "${CONFIG}" --task-id "${TASK_IDS:?set TASK_IDS or use --array}"
fi
echo "[end] $(date)"
