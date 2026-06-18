"""Aggregate sweep run-JSONs into a tidy per-bin MAE table.

For each <run_name>.json that carries raw predictions, recompute per-bin MAE at `bin_w`
from `<split>_preds`/`<split>_ys` (so the table is at one consistent granularity regardless
of when the run was saved). Returns a long-form DataFrame; the CLI prints the condition x bin
summary (mean +/- std across folds).

Usage:
    python -m skyfinder.analysis.aggregate --results results \
        --labels data/labels_with_images.csv --splits data/splits/loco_5fold.json --split val
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from skyfinder.training.engine import per_bin_mae

BINS = ["overall", "many", "medium", "few"]
_KIND = {(False, False): "baseline", (True, False): "lds",
         (False, True): "fds", (True, True): "lds_fds"}


def config_kind(cfg: dict) -> str:
    return _KIND[(bool(cfg.get("use_lds")), bool(cfg.get("use_fds")))]


def build_table(results_dir, labels_path, splits_path, split="val", bin_w=1.0) -> pd.DataFrame:
    """Long-form rows: one per (run, fold) with the 4 per-bin MAEs for `split`."""
    df = pd.read_csv(labels_path)
    splits = json.loads(Path(splits_path).read_text())
    temp = df["TempM"].to_numpy()

    rows = []
    for jp in sorted(glob.glob(str(Path(results_dir) / "**" / "*.json"), recursive=True)):
        d = json.load(open(jp))
        if "config" not in d or f"{split}_preds" not in d or not d.get(f"{split}_preds"):
            continue
        if str(d.get("run_name", "")).startswith("tune"):  # skip hyperparam-sweep runs
            continue
        cfg = d["config"]
        fold = int(cfg["fold"])
        train_y = temp[splits[fold]["train"]]
        ys = np.asarray(d[f"{split}_ys"], dtype=float)
        ps = np.asarray(d[f"{split}_preds"], dtype=float)
        m = per_bin_mae(ys, ps, train_y, bin_w=bin_w)
        rows.append({"model": cfg["model"], "kind": config_kind(cfg), "fold": fold,
                     **{b: m.get(b, np.nan) for b in BINS}})
    return pd.DataFrame(rows)


def summarize(tbl: pd.DataFrame):
    g = tbl.groupby(["model", "kind"])
    return g[BINS].agg(["mean", "std"]), g.size().rename("folds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--labels", default="data/labels_with_images.csv")
    ap.add_argument("--splits", default="data/splits/loco_5fold.json")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--bin-w", type=float, default=1.0)
    args = ap.parse_args()

    tbl = build_table(args.results, args.labels, args.splits, args.split, args.bin_w)
    if tbl.empty:
        print("no runs with raw predictions found under", args.results)
        return
    out, n = summarize(tbl)
    print(f"=== {args.split} per-bin MAE (bin_w={args.bin_w}) ===")
    order = ["baseline", "lds", "fds", "lds_fds"]
    for model in sorted(tbl["model"].unique()):
        for kind in order:
            if (model, kind) not in out.index:
                continue
            row = out.loc[(model, kind)]
            cells = "  ".join(f"{b}={row[(b, 'mean')]:5.2f}±{row[(b, 'std')]:4.2f}" for b in BINS)
            print(f"{model:9s} {kind:8s} | {cells} | folds={n[(model, kind)]}")


if __name__ == "__main__":
    main()
