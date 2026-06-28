"""LOCO metadata feature ablations.

This establishes whether the C2 advantage comes from time, location, their
combination, or the full feature set. Every variant is fit only on the fold's
training rows and evaluated on the same validation/test rows as C2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from skyfinder.analysis.baselines_metadata import METADATA_FEATURES, make_metadata_regressor
from skyfinder.training.engine import per_bin_mae
from skyfinder.training.splits import load_splits


VARIANTS = {
    "time": ["Hour", "Month"],
    "location": ["Latitude", "Longitude"],
    "time_location": ["Hour", "Month", "Latitude", "Longitude"],
    "full_c2": METADATA_FEATURES,
}


def run_variant(df: pd.DataFrame, splits: list[dict], name: str, features: list[str], seed: int) -> dict:
    per_fold = []
    for fold in splits:
        train = df.iloc[fold["train"]]
        val = df.iloc[fold["val"]]
        test = df.iloc[fold["test"]]
        model = make_metadata_regressor(seed=seed, features=features)
        y_train = train["TempM"].to_numpy()
        model.fit(train[features], y_train)
        val_pred = model.predict(val[features])
        test_pred = model.predict(test[features])
        val_metric = per_bin_mae(val["TempM"].to_numpy(), val_pred, y_train)
        test_metric = per_bin_mae(test["TempM"].to_numpy(), test_pred, y_train)
        per_fold.append({
            "fold": int(fold["fold"]),
            "n_train": int(len(train)),
            "n_val": int(len(val)),
            "n_test": int(len(test)),
            "val": val_metric,
            "test": test_metric,
        })
        print(
            f"[{name} fold {fold['fold']}] val={val_metric['overall']:.3f} "
            f"test={test_metric['overall']:.3f}",
            flush=True,
        )
    val = np.asarray([row["val"]["overall"] for row in per_fold])
    test = np.asarray([row["test"]["overall"] for row in per_fold])
    return {
        "features": features,
        "per_fold": per_fold,
        "summary": {
            "val_mae_mean": float(val.mean()),
            "val_mae_std": float(val.std(ddof=1)),
            "test_mae_mean": float(test.mean()),
            "test_mae_std": float(test.std(ddof=1)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/labels_with_images.csv")
    parser.add_argument("--splits", default="data/splits/loco_5fold.json")
    parser.add_argument("--out", default="results/_analysis/metadata_ablations.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    labels_path = Path(args.labels)
    df = pd.read_csv(labels_path)
    splits = load_splits(args.splits, labels_path, len(df))
    output = {"method": "metadata_ablations", "seed": args.seed, "variants": {}}
    for name, features in VARIANTS.items():
        output["variants"][name] = run_variant(df, splits, name, features, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2))
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
