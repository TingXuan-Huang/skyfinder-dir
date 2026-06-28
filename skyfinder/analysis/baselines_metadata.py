"""C2: metadata-only baseline.

Trains a bounded tree ensemble per fold on (CamId, Hour, Month, Latitude,
Longitude) -- no image features at all. Reports val + test per-bin MAE.

Output: `<baselines_metadata_path>` (from config). Same per_fold schema as the constant baselines, so
`analysis/aggregate.py` ingests them with one code path.

Implication if C2 beats the CNN: vision isn't adding much; the CNN is mostly
learning per-camera priors.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from skyfinder.training.engine import per_bin_mae
from skyfinder.training.splits import load_splits

METADATA_FEATURES = ["CamId", "Hour", "Month", "Latitude", "Longitude"]
CATEGORICAL_FEATURES = ["CamId", "Hour", "Month"]
NUMERIC_FEATURES = ["Latitude", "Longitude"]


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def make_metadata_regressor(seed: int = 0, features: list[str] | None = None):
    """Return the C2 metadata-only model.

    The original HistGBR categorical splitter is prohibitively slow on Klone for
    this data shape. A one-hot random forest keeps C2 nonlinear and metadata-only
    while giving bounded runtime for repeated fold refits.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    n_jobs = _env_int(
        "SKYFINDER_METADATA_N_JOBS",
        _env_int("SLURM_CPUS_PER_TASK", 4),
    )
    n_estimators = _env_int("SKYFINDER_METADATA_TREES", 100)
    min_samples_leaf = _env_int("SKYFINDER_METADATA_MIN_LEAF", 20)

    features = list(METADATA_FEATURES if features is None else features)
    unknown = set(features) - set(METADATA_FEATURES)
    if unknown:
        raise ValueError(f"unknown metadata feature(s): {sorted(unknown)}")
    categorical = [name for name in CATEGORICAL_FEATURES if name in features]
    numeric = [name for name in NUMERIC_FEATURES if name in features]
    if not categorical and not numeric:
        raise ValueError("at least one metadata feature is required")

    transformers = []
    if categorical:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)
        )
    if numeric:
        transformers.append(("num", "passthrough", numeric))
    preprocess = ColumnTransformer(transformers)
    regressor = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=n_jobs,
    )
    return make_pipeline(preprocess, regressor)


def run_baselines_metadata(config: dict, out_path: Path | str | None = None, seed: int = 0) -> dict:
    """Fit the metadata-only regressor per fold and save per-fold MAE.

    Reports MAE on:
      val  -- held-out images from train cameras (in-distribution; matches CNN's val)
      test -- LOCO held-out cameras (out-of-distribution; the honest comparison)

    Historical note: this was the "C2" ablation in `docs/ablation_catalog.md`.
    """
    labels_path = Path(config["labels_path"])
    splits_path = Path(config["splits_path"])
    out_path = Path(out_path) if out_path is not None else Path(config["baselines_metadata_path"])

    df = pd.read_csv(labels_path)
    splits = load_splits(splits_path, labels_path, len(df))

    results: dict = {"method": "c2_metadata", "bin_width": 1.0, "per_fold": []}
    for f in splits:
        fold = f["fold"]
        train_df = df.iloc[f["train"]]
        val_df = df.iloc[f["val"]]
        test_df = df.iloc[f["test"]]

        X_train = train_df[METADATA_FEATURES]
        y_train = train_df["TempM"].to_numpy()

        model = make_metadata_regressor(seed=seed)
        model.fit(X_train, y_train)

        val_pred = model.predict(val_df[METADATA_FEATURES])
        test_pred = model.predict(test_df[METADATA_FEATURES])

        val_mae = per_bin_mae(val_df["TempM"].to_numpy(), val_pred, y_train)
        test_mae = per_bin_mae(test_df["TempM"].to_numpy(), test_pred, y_train)

        results["per_fold"].append({
            "fold": fold,
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "test_cams": f["test_cams"],
            "val": val_mae,
            "test": test_mae,
        })
        print(
            f"[fold {fold}] val={val_mae['overall']:.3f}  test={test_mae['overall']:.3f}  "
            f"(val few={val_mae['few']:.3f}  test few={test_mae['few']:.3f})",
            flush=True,
        )

    # Summary across folds.
    val_overall = np.array([r["val"]["overall"] for r in results["per_fold"]])
    test_overall = np.array([r["test"]["overall"] for r in results["per_fold"]])
    results["summary"] = {
        "val_mae_mean":  float(val_overall.mean()),  "val_mae_std":  float(val_overall.std()),
        "test_mae_mean": float(test_overall.mean()), "test_mae_std": float(test_overall.std()),
        "seed": seed,
    }
    print(f"[summary] val MAE:  {results['summary']['val_mae_mean']:.3f} "
          f"± {results['summary']['val_mae_std']:.3f}", flush=True)
    print(f"[summary] test MAE: {results['summary']['test_mae_mean']:.3f} "
          f"± {results['summary']['test_mae_std']:.3f}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[saved] {out_path}", flush=True)
    return results
