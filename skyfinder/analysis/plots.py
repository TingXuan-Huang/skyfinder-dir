"""Figures from the aggregated per-bin table.

plot_per_bin: grouped bars of per-bin MAE (method x bin), error bars = std across folds.
Makes the LDS body-tail tradeoff legible (helps medium/few, hurts overall/many).

Usage:
    python -m skyfinder.analysis.plots --results results --split val --model resnet50 --out figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from skyfinder.analysis.aggregate import BINS, build_table, summarize

KIND_ORDER = ["baseline", "lds", "fds", "lds_fds"]


def plot_per_bin(tbl, model: str, split: str, out_path):
    """Grouped bar chart: per-bin MAE by method, error bars = std across folds."""
    out, n = summarize(tbl[tbl["model"] == model])
    kinds = [k for k in KIND_ORDER if (model, k) in out.index]
    x = np.arange(len(BINS))
    w = 0.8 / max(len(kinds), 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, k in enumerate(kinds):
        row = out.loc[(model, k)]
        means = [row[(b, "mean")] for b in BINS]
        stds = [row[(b, "std")] for b in BINS]
        ax.bar(x + i * w, means, w, yerr=stds, capsize=3, label=f"{k} (n={n[(model, k)]})")
    ax.set_xticks(x + w * (len(kinds) - 1) / 2)
    ax.set_xticklabels(BINS)
    ax.set_ylabel(f"{split} MAE (°C)")
    ax.set_title(f"{model}: per-bin MAE by method ({split})")
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--labels", default="data/labels_with_images.csv")
    ap.add_argument("--splits", default="data/splits/loco_5fold.json")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--bin-w", type=float, default=1.0)
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    tbl = build_table(args.results, args.labels, args.splits, args.split, args.bin_w)
    if tbl.empty:
        print("no runs with raw predictions found under", args.results)
        return
    p = plot_per_bin(tbl, args.model, args.split,
                     Path(args.out) / f"per_bin_{args.model}_{args.split}.png")
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
