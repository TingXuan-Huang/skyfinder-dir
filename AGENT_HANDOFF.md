# SkyFinder DIR — Agent Handoff

This document records the post-review repair pass completed on 2026-06-20.
Read it before changing experiment orchestration, result persistence, or split
handling.

## Non-negotiable invariants

1. Every run carries its own `Config.results_dir`. Do not route artifacts by
   mutating `skyfinder.training.config.RESULTS_DIR`.
2. A split file is valid only for the exact `labels_with_images.csv` used to
   generate it. New split files contain a SHA-256 fingerprint.
3. A checkpoint is loaded from the same configured result root it was saved
   to. This applies to training, test inference, recovery, fusion, and the
   camera-conditioned driver.
4. Camera IDs are categorical metadata, not ordered continuous numbers.

## What changed

### Result routing

- `Config` now has `results_dir`.
- `run_sweep.py` builds a fully path-bound `Config` for every matrix cell.
- `trainer.py`, `infer_test.py`, `recover_results_from_last.py`, and
  `cam_cond_train.py` pass `cfg.results_dir` directly to checkpoint helpers.
- YAML `skip_existing` is now honored for direct `run_sweep.py` calls as well
  as Slurm calls.
- `--dry-run` no longer creates a results directory.

This fixes the prior failure mode where random-control and camera-conditioned
runs wrote under `results/` while later steps searched `results_random/` or
`results_cam_cond/`.

### Split manifests

- `data/splits.py` and `data/splits_random.py` now write a manifest containing
  `schema_version`, `labels_sha256`, `n_rows`, and `folds`.
- `skyfinder.training.splits.load_splits()` validates the row count and file
  hash before returning folds.
- Existing list-only split files remain readable but issue a `RuntimeWarning`.
- Consumers select a fold by recorded `fold` ID rather than list position.

Required action after regenerating or reordering labels:

```bash
python data/splits.py
# If using the random control:
python data/splits_random.py
```

Do this whenever `data/labels_with_images.csv` is regenerated or reordered.

Status on 2026-06-20: both local split files were regenerated from the linked
81,453-row labels table and now carry valid manifests.

### Checkpoint reconstruction and analysis

- `build_training_model(cfg, pretrained=False)` reconstructs vanilla and FDS
  architectures for checkpoint-only loading without downloading torchvision
  weights.
- Fusion reads the saved run config, supports FDS checkpoints, and one-hot
  encodes `CamId`, `Hour`, and `Month`; it no longer treats camera codes as a
  numeric feature.
- Ensemble analysis checks saved test-prediction lengths and targets against
  the configured split before blending.
- C1/C2 compact metric artifacts are included by `aggregate.py` at their
  native 1 °C bin width. Use `--bin-w 1.0` when results contain C1/C2 files.
- Camera-distance analysis now uses nearest-camera great-circle distance in km.
- DINOv2 is loaded once per invocation rather than once per fold.

### Reliability and configuration cleanup

- Main and camera-conditioned training seed Python, NumPy, and Torch. Resume
  checkpoints additionally preserve Python and CUDA RNG state when available.
- Camera-conditioned lookup is vectorized; it no longer synchronizes each GPU
  camera ID through a Python loop.
- The camera-conditioned dataset now tolerates truncated images just like the
  main dataset.
- Unsupported corruption experiment hooks and family entries were removed from
  the public configuration surface. `FAMILIES` lists only checked-in configs.
- `setup_env.slurm` now installs dependencies when it must create a fresh
  Conda environment rather than a clone.
- `results_random/` and `results_cam_cond/` are ignored by Git.
- Result migration is runnable with:

```bash
python -m skyfinder.training.migrate --dry-run --root results/
python -m skyfinder.training.migrate --root results/
# Repair previously nested directories written to the wrong result root:
python -m skyfinder.training.migrate --repair-misrouted --dry-run
python -m skyfinder.training.migrate --repair-misrouted
```

The repair mode only relocates known old-root `*_rand` and `cam_cond_*`
directories and refuses to overwrite destination directories. Confirm no
SkyFinder jobs are running before executing it.

Status on 2026-06-20: it relocated `baseline_resnet50_rand` and
`baseline_vit_rand` to `results_random/`, and `cam_cond_resnet50` to
`results_cam_cond/`.

## Development checks

Focused standard-library regression tests live in `tests/`:

```bash
python -m unittest discover -s tests -v
```

The checked run passed 9 tests covering result routing, invalid subset config,
split fingerprints and legacy warnings, compact C1/C2 aggregation, explicit
analysis method labels, JSON-backed per-camera analysis, geographical distance,
and misrouted-results repair planning/execution.

Optional developer tools are defined in `pyproject.toml`:

```bash
pip install -e '.[dev]'
pytest
ruff check skyfinder data tests run_sweep.py run_baselines.py infer_test.py \
  recover_results_from_last.py cam_cond_train.py
```

Also run these lightweight checks after touching orchestration:

```bash
python run_sweep.py --config configs/main_random.yaml --dry-run
bash -n submit_sweep.sh slurm/*.slurm
```

## Deliberate limits

- This repair pass did not run a full GPU training cell or full-data inference.
- Legacy split files only warn; regenerate them before reporting new results.
- The C1/C2 outputs store aggregate 1 °C metrics, not raw predictions, so they
  cannot be recomputed at a different bin width.
- `pytest` and `ruff` were not installed in the existing project environment;
  the standard-library test suite was used for local verification.
