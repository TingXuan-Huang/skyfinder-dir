"""LOCO metadata feature ablations.

This establishes whether the C2 advantage comes from time, location, their
combination, or the full feature set. Every variant is fit only on the fold's
training rows and evaluated on the same validation/test rows as C2.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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


def solar_elevation(lat_deg, month, hour):
    """Physically-derived solar-elevation angle (degrees), vectorized.

    There is no elevation column in the labels; this reconstructs a transferable
    solar-geometry feature from Latitude, Month, Hour. Inputs are array-like.
    """
    lat_deg = np.asarray(lat_deg, dtype=float)
    month = np.asarray(month, dtype=float)
    hour = np.asarray(hour, dtype=float)
    doy = 30.4 * (month - 0.5)
    decl_deg = 23.44 * np.sin(2.0 * np.pi * (284.0 + doy) / 365.0)
    H_deg = 15.0 * (hour - 12.0)
    lat = np.radians(lat_deg)
    decl = np.radians(decl_deg)
    H = np.radians(H_deg)
    sin_elev = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(H)
    return np.degrees(np.arcsin(np.clip(sin_elev, -1.0, 1.0)))


def run_solar_variant(df: pd.DataFrame, splits: list[dict], seed: int) -> dict:
    features = ["Hour", "Month", "Latitude", "Longitude", "SolarElev"]
    cat = ["Hour", "Month"]
    num = ["Latitude", "Longitude", "SolarElev"]
    df = df.copy()
    df["SolarElev"] = solar_elevation(df["Latitude"], df["Month"], df["Hour"])
    per_fold = []
    for fold in splits:
        train = df.iloc[fold["train"]]
        val = df.iloc[fold["val"]]
        test = df.iloc[fold["test"]]
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
            ("num", "passthrough", num),
        ])
        model = Pipeline([
            ("pre", pre),
            ("rf", RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=20,
                random_state=seed,
                n_jobs=(os.cpu_count() or 4),
            )),
        ])
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
            f"[time_location_solar fold {fold['fold']}] val={val_metric['overall']:.3f} "
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
    output["variants"]["time_location_solar"] = run_solar_variant(df, splits, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2))
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
