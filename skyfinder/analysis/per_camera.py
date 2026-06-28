"""Per-camera LOCO test failure analysis (story II).

For a CNN run, compute each held-out camera's test MAE and correlate it with:
  - climate distance: |cam_mean_TempM - train_mean_TempM|   (report found r ~ +0.56)
  - geo distance:     great-circle km to nearest TRAIN camera
Strong climate correlation + weak geo correlation suggests a missing per-camera prior;
failure is driven by temperature distance from the training mean, not geography.

Reads per-fold run JSONs (test_preds/test_ys from infer_test). The i-th test pred maps to
df.iloc[fold['test'][i]] (infer_test uses shuffle=False), so CamId/lat/lon are recoverable.

Usage:
    python -m skyfinder.analysis.per_camera --cnn baseline_resnet50 --results results \
        --labels data/labels_with_images.csv --splits data/splits/loco_5fold.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from skyfinder.training.checkpoint import load_results, subdir_for
from skyfinder.training.splits import load_splits


def nearest_geo_distance_km(latlon: np.ndarray, train_latlon: np.ndarray) -> float:
    """Great-circle distance to the nearest train camera, in kilometres."""
    lat1, lon1 = np.radians(latlon)
    lat2 = np.radians(train_latlon[:, 0])
    lon2 = np.radians(train_latlon[:, 1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float((2 * 6371.0088 * np.arcsin(np.sqrt(a))).min())


def per_camera_stats(df: pd.DataFrame, splits: list, results_dir, cnn: str) -> pd.DataFrame:
    """One row per held-out camera: test MAE, climate distance, geo distance."""
    rows = []
    for fold in splits:
        name = f"{cnn}_fold{fold['fold']}"
        jp = Path(results_dir) / subdir_for(name) / f"{name}.json"
        if not jp.exists():
            continue
        d = load_results(jp)
        if not d.get("test_preds"):
            continue
        test_df = df.iloc[fold["test"]].reset_index(drop=True)
        preds = np.asarray(d["test_preds"], float)
        ys = np.asarray(d["test_ys"], float)
        if len(preds) != len(test_df):
            continue
        train_df = df.iloc[fold["train"]]
        train_mean = train_df["TempM"].mean()
        train_cams = train_df.groupby("CamId")[["Latitude", "Longitude"]].first().to_numpy()

        g = pd.DataFrame({"CamId": test_df["CamId"].to_numpy(), "err": np.abs(preds - ys),
                          "T": ys, "lat": test_df["Latitude"].to_numpy(),
                          "lon": test_df["Longitude"].to_numpy()})
        for cam, grp in g.groupby("CamId"):
            latlon = grp[["lat", "lon"]].iloc[0].to_numpy()
            geo = nearest_geo_distance_km(latlon, train_cams)
            rows.append({"CamId": int(cam), "fold": fold["fold"], "n": len(grp),
                         "mae": float(grp["err"].mean()),
                         "climate_dist": float(abs(grp["T"].mean() - train_mean)),
                         "geo_dist_km": geo})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", default="baseline_resnet50")
    ap.add_argument("--results", default="results")
    ap.add_argument("--labels", default="data/labels_with_images.csv")
    ap.add_argument("--splits", default="data/splits/loco_5fold.json")
    args = ap.parse_args()

    df = pd.read_csv(args.labels)
    splits = load_splits(args.splits, args.labels, len(df))
    stats = per_camera_stats(df, splits, args.results, args.cnn)
    if stats.empty:
        print(f"no test preds for '{args.cnn}' — run infer_test.py first")
        return
    r_clim = stats["mae"].corr(stats["climate_dist"])
    r_geo = stats["mae"].corr(stats["geo_dist_km"])
    print(f"=== per-camera test MAE, CNN={args.cnn}  (n_cams={len(stats)}) ===")
    print(f"climate-distance r = {r_clim:+.2f}   nearest-camera distance (km) r = {r_geo:+.2f}")
    worst = stats.sort_values("mae", ascending=False).head(5)
    print("worst 5 cameras (CamId / mae / climate_dist):")
    for _, r in worst.iterrows():
        print(f"  cam {int(r['CamId']):>6d}  mae={r['mae']:5.2f}  climate_dist={r['climate_dist']:5.2f}")


if __name__ == "__main__":
    main()
