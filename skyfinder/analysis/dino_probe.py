"""DINOv2 frozen-feature probe: is fine-tuning even helping?

Extract FROZEN DINOv2 features (no training), fit Ridge on each fold's train -> TempM, eval
val+test per-bin MAE. If frozen DINOv2 + Ridge >= the trained ResNet on LOCO test, fine-tuning
the CNN is harmful (it overfits train-camera appearance) — a major reframe.

No training. Needs torch.hub (downloads DINOv2 weights once) + GPU recommended.

Usage:
    python -m skyfinder.analysis.dino_probe --labels data/labels_with_images.csv \
        --splits data/splits/loco_5fold.json --img-dir data/images --variant dinov2_vits14
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from skyfinder.training.dataloader import EVAL_TF, SkyFinderDataset
from skyfinder.training.engine import get_device, per_bin_mae
from skyfinder.training.splits import load_splits


def load_dino(variant: str, device: str):
    """Frozen DINOv2 ViT (vits14=384-d, vitb14=768-d CLS embedding)."""
    net = torch.hub.load("facebookresearch/dinov2", variant)
    return net.eval().to(device)


def extract(net, sub_df, img_dir, device, bs=128):
    loader = DataLoader(SkyFinderDataset(sub_df, EVAL_TF, img_dir=img_dir),
                        batch_size=bs, shuffle=False, num_workers=4)
    feats, ys = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            feats.append(net(x.to(device)).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(feats), np.concatenate(ys)


def fit_probe(Xtr, ytr, alpha=1.0, alphas=None):
    """StandardScaler + Ridge — the standard linear probe.

    If alphas is given, use RidgeCV to pick alpha by CV (a fair probe);
    otherwise fall back to a fixed Ridge(alpha).
    """
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(Xtr)
    model = (RidgeCV(alphas=alphas) if alphas is not None else Ridge(alpha=alpha))
    model.fit(scaler.transform(Xtr), ytr)
    return scaler, model


def run_probe(df, fold, net, img_dir, device, alphas=None) -> dict:
    parts = {s: extract(net, df.iloc[fold[s]].reset_index(drop=True), img_dir, device)
             for s in ("train", "val", "test")}
    scaler, model = fit_probe(*parts["train"], alphas=alphas)
    if hasattr(model, "alpha_"):
        print(f"[fold {fold['fold']}] RidgeCV selected alpha={model.alpha_:g}")
    train_y = df.iloc[fold["train"]]["TempM"].to_numpy()
    return {s: per_bin_mae(parts[s][1], model.predict(scaler.transform(parts[s][0])), train_y)
            for s in ("val", "test")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels_with_images.csv")
    ap.add_argument("--splits", default="data/splits/loco_5fold.json")
    ap.add_argument("--img-dir", default="data/images")
    ap.add_argument("--variant", default="dinov2_vits14")
    ap.add_argument("--alphas", default="0.1,1,10,100,1000",
                    help="comma-separated RidgeCV alpha grid")
    args = ap.parse_args()

    alphas = tuple(float(a) for a in args.alphas.split(","))

    df = pd.read_csv(args.labels)
    splits = load_splits(args.splits, args.labels, len(df))
    device = get_device()
    net = load_dino(args.variant, device)
    val_o, test_o = [], []
    for fold in splits:
        r = run_probe(df, fold, net, Path(args.img_dir), device, alphas=alphas)
        val_o.append(r["val"]["overall"])
        test_o.append(r["test"]["overall"])
        print(f"[fold {fold['fold']}] DINOv2+Ridge val={r['val']['overall']:.3f} "
              f"test={r['test']['overall']:.3f}")
    if val_o:
        print(f"[summary] {args.variant}+Ridge  val={np.mean(val_o):.3f}  test={np.mean(test_o):.3f}")


if __name__ == "__main__":
    main()
