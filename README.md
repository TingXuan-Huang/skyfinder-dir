# dir-skyfinder

Clean rebuild of the DIR-on-SkyFinder training pipeline (temperature regression from outdoor
webcam images). This is the go-forward codebase; the original repo (`DIR_Code`, branch
`refactor-2026-05`) is kept as reference, including its analysis/figure layer and the full
restart Q&A trail.

Every module here was triaged module-by-module from the original and verified
(`build_loaders(cfg)` + `run_baseline` end-to-end smoke). See the reference repo's
`experiments/restart-2026-05-24/` for the design docs, Q&A log, and the key finding.

## Layout

```
skyfinder/training/   refactored training package (11 modules)
skyfinder/analysis/   C1 (constant) + C2 (metadata GBM) baselines — CPU only
run_sweep.py          GPU sweep harness: experiment x fold -> run_baseline(cfg)
run_baselines.py      CPU baselines runner (C1 + C2)
configs/main.yaml     the 8-experiment x 5-fold matrix
data/*.py             data-prep scripts (build labels + splits + download images)
slurm/run_sweep.slurm + submit_sweep.sh   Hyak array-job submission
```

## Key design point

Paths are **Config fields** (`labels_path`/`splits_path`/`img_dir`) passed into each run,
not module globals. `run_sweep.py` builds one `Config` per (experiment, fold). The eval metric
`per_bin_mae` uses **`bin_w=1.0 °C`** (paper-faithful; the few-bin is sparse on SkyFinder — see
the reference repo's U3 finding).

## Quickstart

```bash
pip install -e .

# 1. Build data (run on a login node — Hyak compute nodes have no internet):
python data/prep_labels.py && python data/download_images.py \
  && python data/filter_to_images.py && python data/splits.py

# 2. Inspect the run matrix (no data needed):
python run_sweep.py --list            # 40 cells: 8 configs x 5 folds
python run_sweep.py --dry-run         # resolved Configs

# 3. Smoke one cell locally:
python run_sweep.py --experiment baseline_resnet50 --fold 0   # (set epochs small in a smoke config first)

# 4. Full sweep on Hyak (array job, 2 concurrent):
bash submit_sweep.sh 2

# 5. CPU baselines (anytime):
python run_baselines.py --config configs/main.yaml
```

## Status

- Training package: complete, verified.
- Run harness (`run_sweep.py`, `run_baselines.py`): present; `--list`/`--dry-run` verified locally.
- SLURM (`slurm/`, `submit_sweep.sh`): `account=stf`, scratch + conda paths preset — **verify
  partition/account/env on Hyak before the real submit** (untested off-cluster).
- Analysis/figures layer: NOT ported (lives in the reference repo). Add if needed.
