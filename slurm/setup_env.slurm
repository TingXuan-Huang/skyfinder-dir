#!/bin/bash
# Build the Python environment for this checkout on a Hyak compute node.
#
# Submit once from the repo root:
#   sbatch --account=stf --partition=compute slurm/setup_env.slurm
#
# If you want to verify CUDA at setup time, submit to a GPU partition instead:
#   sbatch --account=stf --partition=gpu-rtx6k --gpus=1 slurm/setup_env.slurm
#SBATCH --job-name=skyfinder-env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"

ENV_PREFIX="${ENV_PREFIX:-${REPO_ROOT}/.conda/skyfinder}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${REPO_ROOT}/.conda/pkgs}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/gscratch/stf/${USER}}"
SETUP_CLONE_FROM="${SETUP_CLONE_FROM:-/gscratch/stf/${USER}/DIR-SkyFinder/.conda/skyfinder}"

export CONDA_PKGS_DIRS
export XDG_CACHE_HOME="${SCRATCH_ROOT}/.cache"
export TORCH_HOME="${XDG_CACHE_HOME}/torch"
export PIP_CACHE_DIR="${XDG_CACHE_HOME}/pip"
mkdir -p slurm/logs "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "${TORCH_HOME}"

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh 2>/dev/null || true
fi
module load conda
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

FRESH_ENV=0
if [[ ! -d "${ENV_PREFIX}" ]]; then
  if [[ -d "${SETUP_CLONE_FROM}" ]]; then
    echo "[setup] cloning existing package env: ${SETUP_CLONE_FROM}"
    conda create --yes --prefix "${ENV_PREFIX}" --clone "${SETUP_CLONE_FROM}"
  else
    echo "[setup] no clone source found; creating env from package indexes"
    conda create --yes --prefix "${ENV_PREFIX}" python=3.11
    FRESH_ENV=1
  fi
fi

conda activate "${ENV_PREFIX}"
python -m pip install --upgrade pip wheel setuptools
if [[ "${FRESH_ENV}" == "1" ]]; then
  # A fresh Python environment has none of the project dependencies. This may
  # require an enabled package index or a populated pip cache on the compute node.
  python -m pip install -e .
else
  python -m pip install --no-deps -e .
fi
python - <<'PY'
missing = []
for mod in ("torch", "torchvision", "numpy", "pandas", "yaml", "sklearn", "scipy", "pyarrow"):
    try:
        __import__(mod)
    except Exception:
        missing.append(mod)
if missing:
    raise SystemExit("[setup] missing required packages after env creation: " + ", ".join(missing))
PY

if [[ "${PREFETCH_WEIGHTS:-1}" == "1" ]]; then
  python - <<'PY'
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights

for weights in (ResNet50_Weights.IMAGENET1K_V2, ViT_B_16_Weights.IMAGENET1K_V1):
    path = weights.get_state_dict(progress=True)
    print(f"[ok] cached {weights.name}")
PY
fi

python - <<'PY'
import torch
import torchvision
import skyfinder
print(f"[ok] torch={torch.__version__} torchvision={torchvision.__version__}")
print(f"[ok] cuda_available={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
print(f"[ok] skyfinder={skyfinder.__file__}")
PY
