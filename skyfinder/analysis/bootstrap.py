"""Paired bootstrap CIs for method comparisons — the rigor the per-bin table lacks.

paired_bootstrap(a, b): a, b are paired per-unit MAE (per-fold now; per-camera once test
inference exists). Returns mean(a - b) and a percentile CI. Few units (e.g. 5 folds) -> an
honestly wide CI.

compare_to_baseline ALIGNS folds: it only pairs a method with baseline on the folds BOTH
completed — so partial/misaligned sweeps still give a valid (if smaller-n) comparison.

Usage:
    python -m skyfinder.analysis.bootstrap --results results --split val --model resnet50 --bin overall
"""
from __future__ import annotations

import argparse

import numpy as np

from skyfinder.analysis.aggregate import BINS, build_table

KIND_ORDER = ["lds", "fds", "lds_fds"]


def paired_bootstrap(a, b, n_boot=10000, ci=95, seed=0):
    """mean(a - b) and a percentile CI from resampling the paired units."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    diffs = a - b
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(diffs, len(diffs), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(diffs.mean()), float(lo), float(hi)


def compare_to_baseline(tbl, model: str, bin_name: str, **kw):
    """For each DIR method: paired bootstrap of (method - baseline) MAE over COMMON folds only."""
    sub = tbl[tbl["model"] == model]
    base = sub[sub["kind"] == "baseline"].set_index("fold")[bin_name]
    out = {}
    for k in KIND_ORDER:
        m = sub[sub["kind"] == k].set_index("fold")[bin_name]
        folds = sorted(set(base.index) & set(m.index))  # ALIGN: only folds both ran
        if len(folds) < 2:
            out[k] = (folds, None)
            continue
        d, lo, hi = paired_bootstrap(m.loc[folds].to_numpy(), base.loc[folds].to_numpy(), **kw)
        out[k] = (folds, (d, lo, hi))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--labels", default="data/labels_with_images.csv")
    ap.add_argument("--splits", default="data/splits/loco_5fold.json")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--bin-w", type=float, default=1.0)
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--bin", default="overall", choices=BINS)
    args = ap.parse_args()

    tbl = build_table(args.results, args.labels, args.splits, args.split, args.bin_w)
    if tbl.empty:
        print("no runs with raw predictions found under", args.results)
        return
    res = compare_to_baseline(tbl, args.model, args.bin)
    print(f"=== {args.model} {args.split} {args.bin}: method - baseline "
          f"(paired bootstrap, common folds; * = CI excludes 0) ===")
    for k, (folds, ci) in res.items():
        if ci is None:
            print(f"{k:8s}: only {len(folds)} common fold(s) — need >=2 for a CI")
            continue
        d, lo, hi = ci
        star = "" if lo <= 0 <= hi else "  *"
        print(f"{k:8s}: Δ={d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  (folds {folds}){star}")


if __name__ == "__main__":
    main()
