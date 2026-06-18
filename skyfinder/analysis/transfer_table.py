"""Transfer-rate table: SkyFinder DIR %-improvement vs the paper's IMDB-WIKI + AgeDB.

%-improvement = (baseline - method) / baseline * 100 per bin (positive = DIR helps). Compares
SkyFinder's rates (from the run JSONs) to Yang et al. 2021's image benchmarks, side by side —
so you can read whether the tail-direction transfers and whether the overall-direction flips.

Usage:
    python -m skyfinder.analysis.transfer_table --results results --split val --model resnet50
"""
from __future__ import annotations

import argparse

from skyfinder.analysis.aggregate import BINS, build_table, summarize

# Yang et al. 2021 (arXiv:2102.09554), ResNet-50 test MAE. Order: [All, Many, Medium, Few].
PAPER = {
    "IMDB-WIKI": {"baseline": [8.06, 7.23, 15.12, 26.33], "lds": [7.83, 7.31, 12.43, 22.51],
                  "fds": [7.85, 7.18, 13.35, 24.12], "lds_fds": [7.78, 7.20, 12.61, 22.19]},
    "AgeDB": {"baseline": [7.77, 6.62, 9.55, 13.67], "lds": [7.67, 6.98, 8.86, 10.89],
              "fds": [7.55, 6.50, 8.97, 13.01], "lds_fds": [7.55, 7.01, 8.24, 10.79]},
}
METHODS = ["lds", "fds", "lds_fds"]


def pct_improve(base, method):
    return [(b - m) / b * 100 for b, m in zip(base, method)]


def rates_from_mae(mae_by_kind: dict) -> dict:
    base = mae_by_kind["baseline"]
    return {k: pct_improve(base, mae_by_kind[k]) for k in METHODS if k in mae_by_kind}


def skyfinder_rates(tbl, model):
    out, _ = summarize(tbl[tbl["model"] == model])
    if (model, "baseline") not in out.index:
        return None
    mae = {kind: [out.loc[(model, kind)][(b, "mean")] for b in BINS]
           for kind in ["baseline"] + METHODS if (model, kind) in out.index}
    return rates_from_mae(mae)


def _print_block(name, rates):
    print(f"\n{name}  (%-improvement vs baseline; + = better)")
    print(f"  {'method':8s} " + "  ".join(f"{b:>7s}" for b in BINS))
    for k in METHODS:
        if k not in rates:
            continue
        print(f"  {k:8s} " + "  ".join(f"{v:+6.1f}%" for v in rates[k]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--labels", default="data/labels_with_images.csv")
    ap.add_argument("--splits", default="data/splits/loco_5fold.json")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--bin-w", type=float, default=1.0)
    ap.add_argument("--model", default="resnet50")
    args = ap.parse_args()

    for name, mae in PAPER.items():
        _print_block(name, rates_from_mae(mae))

    tbl = build_table(args.results, args.labels, args.splits, args.split, args.bin_w)
    sky = skyfinder_rates(tbl, args.model) if not tbl.empty else None
    if sky:
        _print_block(f"SkyFinder {args.model} ({args.split})", sky)
        print("\nRead: does the tail (Medium/Few) direction match the paper? Does Overall flip sign?")
    else:
        print(f"\n(no SkyFinder {args.model} runs yet — paper reference shown above)")


if __name__ == "__main__":
    main()
