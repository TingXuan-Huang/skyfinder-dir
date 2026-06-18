"""Image+metadata fusion: HistGBR on (CNN penultimate features ++ metadata).

Tests whether image and metadata carry COMPLEMENTARY signal. Loads a trained CNN checkpoint,
extracts penultimate features on each fold's train/val/test images, concatenates the 5 metadata
columns, fits a HistGradientBoostingRegressor, reports val+test per-bin MAE. If fusion beats
BOTH CNN-alone and C2-alone, fusion is the right model.

No new training (reuses a checkpoint). Heavy — extracts features over all fold images — so run
on a GPU node.

Usage:
    python -m skyfinder.analysis.fusion --cnn baseline_resnet50 --results results \
        --labels data/labels_with_images.csv --splits data/splits/loco_5fold.json --img-dir data/images
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from skyfinder.training.checkpoint import load_model_weights
from skyfinder.training.dataloader import EVAL_TF, SkyFinderDataset
from skyfinder.training.engine import get_device, per_bin_mae
from skyfinder.training.model import build_model

META = ["CamId_cat", "Hour", "Month", "Latitude", "Longitude"]


def strip_head(net: nn.Module) -> nn.Module:
    """Replace the regression head with Identity so forward returns penultimate features."""
    if hasattr(net, "fc"):
        net.fc = nn.Identity()
    else:
        net.heads.head = nn.Identity()
    return net


def extract(net, sub_df, img_dir, device, bs=128):
    loader = DataLoader(SkyFinderDataset(sub_df, EVAL_TF, img_dir=img_dir),
                        batch_size=bs, shuffle=False, num_workers=4)
    feats, ys = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            feats.append(net(x.to(device)).flatten(1).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(feats), np.concatenate(ys)


def run_fusion(df, splits, fold, cnn, model_name, img_dir, device) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor
    name = f"{cnn}_fold{fold['fold']}"
    net = strip_head(build_model(model_name))
    net.load_state_dict(load_model_weights(name))
    net.eval().to(device)

    parts = {}
    for split in ("train", "val", "test"):
        sub = df.iloc[fold[split]].reset_index(drop=True)
        feat, ys = extract(net, sub, img_dir, device)
        parts[split] = (np.hstack([feat, sub[META].to_numpy()]), ys)

    Xtr, ytr = parts["train"]
    gbm = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=6, random_state=0)
    gbm.fit(Xtr, ytr)
    train_y = df.iloc[fold["train"]]["TempM"].to_numpy()
    return {split: per_bin_mae(parts[split][1], gbm.predict(parts[split][0]), train_y)
            for split in ("val", "test")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", default="baseline_resnet50")
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--results", default="results")
    ap.add_argument("--labels", default="data/labels_with_images.csv")
    ap.add_argument("--splits", default="data/splits/loco_5fold.json")
    ap.add_argument("--img-dir", default="data/images")
    args = ap.parse_args()

    df = pd.read_csv(args.labels)
    cam_dtype = pd.CategoricalDtype(categories=sorted(df["CamId"].unique()))
    df["CamId_cat"] = df["CamId"].astype(cam_dtype).cat.codes
    splits = json.loads(Path(args.splits).read_text())
    device = get_device()

    val_o, test_o = [], []
    for fold in splits:
        name = f"{args.cnn}_fold{fold['fold']}"
        try:
            r = run_fusion(df, splits, fold, args.cnn, args.model, Path(args.img_dir), device)
        except FileNotFoundError:
            print(f"[skip] {name}: no checkpoint")
            continue
        val_o.append(r["val"]["overall"])
        test_o.append(r["test"]["overall"])
        print(f"[fold {fold['fold']}] fusion val={r['val']['overall']:.3f} test={r['test']['overall']:.3f}")
    if val_o:
        print(f"[summary] fusion val={np.mean(val_o):.3f}  test={np.mean(test_o):.3f}  (n={len(val_o)})")


if __name__ == "__main__":
    main()
