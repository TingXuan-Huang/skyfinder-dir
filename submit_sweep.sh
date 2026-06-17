#!/usr/bin/env bash
# Submit the full sweep as a SLURM array. Array size = #cells from run_sweep --list.
# Usage:  bash submit_sweep.sh [concurrent]      (default 2 concurrent)
set -euo pipefail
cd "$(dirname "$0")"
CONCURRENT="${1:-2}"
N=$(python run_sweep.py --config configs/main.yaml --list | wc -l | tr -d ' ')
echo "[submit] ${N} cells, ${CONCURRENT} concurrent"
mkdir -p slurm/logs
sbatch --array=0-$((N - 1))%"${CONCURRENT}" slurm/run_sweep.slurm
