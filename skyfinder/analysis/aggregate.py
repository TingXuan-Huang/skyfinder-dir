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
from skyfinder.training.splits import get_fold, load_splits

BINS = ["overall", "many", "medium", "few"]
_KIND = {(False, False): "baseline", (True, False): "lds",
         (False, True): "fds", (True, True): "lds_fds"}


def config_kind(cfg: dict) -> str:
    if cfg.get("method"):
        return str(cfg["method"])
    return _KIND[(bool(cfg.get("use_lds")), bool(cfg.get("use_fds")))]


def build_table(results_dir, labels_path, splits_path, split="val", bin_w=1.0) -> pd.DataFrame:
    """Long-form rows: one per (run, fold) with the 4 per-bin MAEs for `split`."""
    df = pd.read_csv(labels_path)
    splits = load_splits(splits_path, labels_path, len(df))
    temp = df["TempM"].to_numpy()

    rows = []
    for jp in sorted(glob.glob(str(Path(results_dir) / "**" / "*.json"), recursive=True)):
        with Path(jp).open() as f:
            d = json.load(f)

        if "config" in d and d.get(f"{split}_preds"):
            if str(d.get("run_name", "")).startswith("tune"):
                continue
            cfg = d["config"]
            fold = int(cfg["fold"])
            train_y = temp[get_fold(splits, fold)["train"]]
            ys = np.asarray(d[f"{split}_ys"], dtype=float)
            ps = np.asarray(d[f"{split}_preds"], dtype=float)
            m = per_bin_mae(ys, ps, train_y, bin_w=bin_w)
            rows.append({"model": cfg["model"], "kind": config_kind(cfg), "fold": fold,
                         **{b: m.get(b, np.nan) for b in BINS}})
            continue

        # C1/C2 files store already-computed 1 °C metrics rather than raw
        # predictions. Include them in the table at their native resolution.
        if "per_fold" not in d:
            continue
        if bin_w != 1.0:
            raise ValueError("C1/C2 artifacts only support aggregation with --bin-w 1.0")
        for record in d["per_fold"]:
            metric = record.get(split)
            if not isinstance(metric, dict):
                continue
            kind = f"c1_{record['predictor']}" if "predictor" in record else d.get("method", "c2_metadata")
            rows.append({"model": "metadata_baselines", "kind": kind, "fold": int(record["fold"]),
                         **{b: metric.get(b, np.nan) for b in BINS}})
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
    preferred_order = ["baseline", "lds", "fds", "lds_fds", "cam_conditioned"]
    for model in sorted(tbl["model"].unique()):
        present = set(tbl.loc[tbl["model"] == model, "kind"])
        order = [kind for kind in preferred_order if kind in present]
        order.extend(sorted(present - set(order)))
        for kind in order:
            if (model, kind) not in out.index:
                continue
            row = out.loc[(model, kind)]
            cells = "  ".join(f"{b}={row[(b, 'mean')]:5.2f}±{row[(b, 'std')]:4.2f}" for b in BINS)
            print(f"{model:9s} {kind:8s} | {cells} | folds={n[(model, kind)]}")


if __name__ == "__main__":
    main()
