#!/bin/bash
# CPU-only metadata feature ablations plus validation-selected FDS/C2 blend.
#SBATCH --job-name=skymeta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX:-./.conda/skyfinder}"
export PYTHONUNBUFFERED=1
export SKYFINDER_METADATA_N_JOBS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p slurm/logs results/_analysis
python -u -m skyfinder.analysis.metadata_ablations --out results/_analysis/metadata_ablations.json
python -u -m skyfinder.analysis.ensemble --cnn fds_resnet50 --selection validation \
  --out results/_analysis/ensemble_validation_selected.json
