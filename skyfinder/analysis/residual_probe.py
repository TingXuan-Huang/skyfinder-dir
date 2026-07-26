"""Residual probe: do image pixels add anything on top of the metadata climatology prior?

Per fold, first fit a metadata climatology on TRANSFERABLE features (Hour, Month, Latitude,
Longitude — CamId is dropped because test cameras are unseen under LOCO). Then fit a frozen
DINOv2 + Ridge probe on the RESIDUAL (TempM - metadata prediction). The final prediction is
metadata + residual. If residual-fusion beats metadata-only on LOCO test, image pixels carry
marginal signal the climatology prior misses; if not, the prior already explains the test error.

No training. Needs torch.hub (downloads DINOv2 weights once) + GPU recommended.

Usage:
    python -m skyfinder.analysis.residual_probe --labels data/labels_with_images.csv \
        --splits data/splits/loco_5fold.json --img-dir data/images --variant dinov2_vits14
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from skyfinder.analysis.dino_probe import load_dino, extract, fit_probe
from skyfinder.analysis.baselines_metadata import make_metadata_regressor
from skyfinder.training.engine import get_device, per_bin_mae
from skyfinder.training.splits import load_splits

# CamId dropped: under LOCO the test cameras are unseen, so a CamId term cannot transfer.
RESIDUAL_META_FEATURES = ["Hour", "Month", "Latitude", "Longitude"]


def run_residual_probe(df, fold, net, img_dir, device) -> dict:
    """Metadata climatology prior + frozen-image Ridge probe on the residual."""
    train = df.iloc[fold["train"]]
    m = make_metadata_regressor(0, features=RESIDUAL_META_FEATURES)
    m.fit(train[RESIDUAL_META_FEATURES], train["TempM"])

    # Predict the prior on each split in df.iloc row order (matches the image extract order).
    yhat_meta = {s: m.predict(df.iloc[fold[s]][RESIDUAL_META_FEATURES])
                 for s in ("train", "val", "test")}

    # Frozen image features; ys aligns with yhat_meta on the same df.iloc rows in the same order.
    feats = {s: extract(net, df.iloc[fold[s]].reset_index(drop=True), img_dir, device)
             for s in ("train", "val", "test")}

    resid_train = feats["train"][1] - yhat_meta["train"]
    scaler, ridge = fit_probe(feats["train"][0], resid_train)

    train_y = train["TempM"].to_numpy()
    out = {}
    for s in ("val", "test"):
        Xf, ys = feats[s]
        resid_pred = ridge.predict(scaler.transform(Xf))
        final = yhat_meta[s] + resid_pred
        out[s] = per_bin_mae(ys, final, train_y)
        out[f"meta_{s}"] = per_bin_mae(ys, yhat_meta[s], train_y)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels_with_images.csv")
    ap.add_argument("--splits", default="data/splits/loco_5fold.json")
    ap.add_argument("--img-dir", default="data/images")
    ap.add_argument("--variant", default="dinov2_vits14")
    args = ap.parse_args()

    df = pd.read_csv(args.labels)
    splits = load_splits(args.splits, args.labels, len(df))
    device = get_device()
    net = load_dino(args.variant, device)

    mv, mt, fv, ft = [], [], [], []
    for fold in splits:
        r = run_residual_probe(df, fold, net, Path(args.img_dir), device)
        mv.append(r["meta_val"]["overall"])
        mt.append(r["meta_test"]["overall"])
        fv.append(r["val"]["overall"])
        ft.append(r["test"]["overall"])
        print(f"[fold {fold['fold']}] "
              f"meta_only val={r['meta_val']['overall']:.3f} test={r['meta_test']['overall']:.3f}  "
              f"residual_fusion val={r['val']['overall']:.3f} test={r['test']['overall']:.3f}")
    if ft:
        print(f"[summary] metadata-only    val={np.mean(mv):.3f}  test={np.mean(mt):.3f}")
        print(f"[summary] residual-fusion  val={np.mean(fv):.3f}  test={np.mean(ft):.3f}")
        print(f"[summary] pixel marginal   val={np.mean(mv) - np.mean(fv):+.3f}  "
              f"test={np.mean(mt) - np.mean(ft):+.3f}  (positive = pixels help)")


if __name__ == "__main__":
    main()
