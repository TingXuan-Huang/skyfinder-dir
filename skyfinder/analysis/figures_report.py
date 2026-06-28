"""Report figures for the distribution-shift story (story II).

Reuses per_camera.per_camera_stats to build per-held-out-camera test MAE, then
writes two figures into --out:
  1) <cnn>_climate_scatter.png  — test MAE vs climate_dist, title shows Pearson r.
  2) <cnn>_worst_cameras.png    — 10 worst cameras by MAE (horizontal bar).
Also prints per-fold test-count line and the worst-camera list.

Usage:
    python -m skyfinder.analysis.figures_report --cnn baseline_resnet50 \
        --results results --labels data/labels_with_images.csv \
        --splits data/splits/loco_5fold.json --out figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from skyfinder.analysis.per_camera import per_camera_stats
from skyfinder.training.splits import load_splits


def climate_scatter(stats: pd.DataFrame, cnn: str, out: Path) -> Path:
    r = stats["mae"].corr(stats["climate_dist"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(stats["climate_dist"], stats["mae"], s=24, alpha=0.7)
    ax.set_xlabel("climate distance |cam_mean - train_mean| (C)")
    ax.set_ylabel("per-camera test MAE (C)")
    ax.set_title(f"{cnn}: test MAE vs climate distance  (Pearson r = {r:+.2f})")
    fig.tight_layout()
    p = out / f"{cnn}_climate_scatter.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def worst_cameras_bar(stats: pd.DataFrame, cnn: str, out: Path) -> Path:
    worst = stats.sort_values("mae", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = [str(int(c)) for c in worst["CamId"]]
    ax.barh(labels, worst["mae"])
    ax.invert_yaxis()
    ax.set_xlabel("per-camera test MAE (C)")
    ax.set_ylabel("CamId")
    ax.set_title(f"{cnn}: 10 worst held-out cameras by test MAE")
    fig.tight_layout()
    p = out / f"{cnn}_worst_cameras.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", default="baseline_resnet50")
    ap.add_argument("--results", default="results")
    ap.add_argument("--labels", default="data/labels_with_images.csv")
    ap.add_argument("--splits", default="data/splits/loco_5fold.json")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    df = pd.read_csv(args.labels)
    splits = load_splits(args.splits, args.labels, len(df))
    stats = per_camera_stats(df, splits, args.results, args.cnn)
    if stats.empty:
        print(f"[figures] no test preds for '{args.cnn}' — run infer_test.py first")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[figures] per-fold test counts (n_test):")
    for fold, n in stats.groupby("fold")["n"].sum().items():
        print(f"  fold {int(fold)}: {int(n)}")

    p1 = climate_scatter(stats, args.cnn, out)
    p2 = worst_cameras_bar(stats, args.cnn, out)

    worst = stats.sort_values("mae", ascending=False).head(10)
    print("[figures] worst cameras (CamId / mae / climate_dist):")
    for _, r in worst.iterrows():
        print(f"  cam {int(r['CamId']):>6d}  mae={r['mae']:5.2f}  climate_dist={r['climate_dist']:5.2f}")

    print(f"[figures] saved {p1}")
    print(f"[figures] saved {p2}")


if __name__ == "__main__":
    main()
