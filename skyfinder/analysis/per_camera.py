"""Per-camera LOCO test failure analysis (story II).

For a CNN run, compute each held-out camera's test MAE and correlate it with:
  - climate distance: |cam_mean_TempM - train_mean_TempM|   (report found r ~ +0.56)
  - geo distance:     L2(lat,lon) to nearest TRAIN camera   (report found r ~ -0.17)
Strong climate correlation + weak geo correlation => the model lacks a per-camera prior;
failure is driven by temperature distance from the training mean, not geography.

Reads per-fold run JSONs (test_preds/test_ys from infer_test). The i-th test pred maps to
df.iloc[fold['test'][i]] (infer_test uses shuffle=False), so CamId/lat/lon are recoverable.

Usage:
    python -m skyfinder.analysis.per_camera --cnn baseline_resnet50 --results results \
        --labels data/labels_with_images.csv --splits data/splits/loco_5fold.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from skyfinder.training.checkpoint import subdir_for


def per_camera_stats(df: pd.DataFrame, splits: list, results_dir, cnn: str) -> pd.DataFrame:
    """One row per held-out camera: test MAE, climate distance, geo distance."""
    rows = []
    for fold in splits:
        name = f"{cnn}_fold{fold['fold']}"
        jp = Path(results_dir) / subdir_for(name) / f"{name}.json"
        if not jp.exists():
            continue
        d = json.load(open(jp))
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
            geo = float(np.sqrt(((train_cams - latlon) ** 2).sum(axis=1)).min())
            rows.append({"CamId": int(cam), "fold": fold["fold"], "n": len(grp),
                         "mae": float(grp["err"].mean()),
                         "climate_dist": float(abs(grp["T"].mean() - train_mean)),
                         "geo_dist": geo})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", default="baseline_resnet50")
    ap.add_argument("--results", default="results")
    ap.add_argument("--labels", default="data/labels_with_images.csv")
    ap.add_argument("--splits", default="data/splits/loco_5fold.json")
    args = ap.parse_args()

    df = pd.read_csv(args.labels)
    splits = json.loads(Path(args.splits).read_text())
    stats = per_camera_stats(df, splits, args.results, args.cnn)
    if stats.empty:
        print(f"no test preds for '{args.cnn}' — run infer_test.py first")
        return
    r_clim = stats["mae"].corr(stats["climate_dist"])
    r_geo = stats["mae"].corr(stats["geo_dist"])
    print(f"=== per-camera test MAE, CNN={args.cnn}  (n_cams={len(stats)}) ===")
    print(f"climate-distance r = {r_clim:+.2f}   geo-distance r = {r_geo:+.2f}")
    worst = stats.sort_values("mae", ascending=False).head(5)
    print("worst 5 cameras (CamId / mae / climate_dist):")
    for _, r in worst.iterrows():
        print(f"  cam {r['CamId']:>6d}  mae={r['mae']:5.2f}  climate_dist={r['climate_dist']:5.2f}")


if __name__ == "__main__":
    main()
