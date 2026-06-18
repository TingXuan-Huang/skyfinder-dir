#!/usr/bin/env bash
# Submit the full sweep as a SLURM array. Array size = #cells from run_sweep --list.
# Usage:
#   bash submit_sweep.sh [concurrent] [config]
#
# Defaults target UW Hyak Klone GPU partitions. Override from the shell, e.g.:
#   ACCOUNT=stf PARTITION=gpu-l40s GPUS=1 bash submit_sweep.sh 2
#   GPU_FLAG=--gpus-per-node=2080ti:1 PARTITION=ckpt-all bash submit_sweep.sh 2
set -euo pipefail
cd "$(dirname "$0")"

CONCURRENT="${1:-${CONCURRENT:-2}}"
CONFIG="${2:-${CONFIG:-configs/main.yaml}}"
ACCOUNT="${ACCOUNT:-stf}"
PARTITION="${PARTITION:-gpu-rtx6k}"
GPUS="${GPUS:-1}"
GPU_FLAG="${GPU_FLAG:---gpus=${GPUS}}"
ENV_PREFIX="${ENV_PREFIX:-$(pwd)/.conda/skyfinder}"

count_cells() {
  awk '
    /^[[:space:]]*-[[:space:]]*name:/ { in_exp=1; next }
    in_exp && /^[[:space:]]*folds:[[:space:]]*\[/ {
      line=$0
      sub(/.*\[/, "", line)
      sub(/\].*/, "", line)
      gsub(/[[:space:]]/, "", line)
      n=split(line, folds, ",")
      total += n
      in_exp=0
    }
    END { print total + 0 }
  ' "$1"
}

N="$(count_cells "${CONFIG}")"
if [[ "${N}" -le 0 ]]; then
  echo "[error] could not determine sweep size from ${CONFIG}" >&2
  exit 1
fi

echo "[submit] config=${CONFIG} cells=${N} concurrent=${CONCURRENT}"
echo "[submit] account=${ACCOUNT} partition=${PARTITION} ${GPU_FLAG} env=${ENV_PREFIX}"
mkdir -p slurm/logs
sbatch \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  "${GPU_FLAG}" \
  --array="0-$((N - 1))%${CONCURRENT}" \
  --export=ALL,CONFIG="${CONFIG}",ENV_PREFIX="${ENV_PREFIX}" \
  slurm/run_sweep.slurm
