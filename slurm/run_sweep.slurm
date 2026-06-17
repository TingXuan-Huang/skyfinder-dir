#!/bin/bash
# One array task = one (experiment, fold) cell of the sweep.
# Submit via:  bash submit_sweep.sh [concurrent]   (computes the array size for you)
#SBATCH --job-name=skyfinder-sweep
#SBATCH --account=stf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err
#SBATCH --export=ALL
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=thuang27@uw.edu

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# Activate conda env (adjust if your env lives elsewhere).
source ~/.bashrc
conda activate ./.conda/skyfinder || conda activate skyfinder

export SCRATCH=/mmfs1/gscratch/stf/${USER}
export XDG_CACHE_HOME=$SCRATCH/.cache
export TORCH_HOME=$SCRATCH/.cache/torch
mkdir -p "$TORCH_HOME" slurm/logs

echo "[task] ${SLURM_ARRAY_TASK_ID}  host $(hostname)  $(date)"
nvidia-smi || true

python run_sweep.py --config configs/main.yaml --task-id "${SLURM_ARRAY_TASK_ID}" --skip-existing

echo "[end] $(date)"
