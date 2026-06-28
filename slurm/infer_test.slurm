#!/bin/bash
# One array task performs test inference for one completed sweep cell.
# Submission supplies CONFIG and a typed GPU request.
#SBATCH --job-name=skyinfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
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
export PYTHONUNBUFFERED=1
mkdir -p slurm/logs
python -u infer_test.py --config "${CONFIG:?CONFIG is required}" --task-id "${SLURM_ARRAY_TASK_ID:?array task required}"
