"""Test-split (LOCO) inference for completed runs.

Loads each <run>.pt, predicts on the fold's held-out TEST cameras (splits[fold]["test"]),
and writes test_preds/test_ys + test per-bin MAE back into <run>.json. Training produces
val only; this adds the LOCO-test side (the honest generalisation number).

Net construction mirrors the trainer (FDSModel wrap for use_fds) so checkpoints load cleanly.
FDS calibration is train-only; at eval the forward returns plain predictions.

Usage:
    python infer_test.py --config configs/main.yaml --task-id 0   # one cell (SLURM array)
    python infer_test.py --config configs/main.yaml               # all completed runs
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from run_sweep import build_matrix
from skyfinder.training import config as cfg_module
from skyfinder.training.checkpoint import (load_model_weights, load_results, save_results,
                                           subdir_for)
from skyfinder.training.config import Config
from skyfinder.training.dataloader import EVAL_TF, SkyFinderDataset
from skyfinder.training.engine import get_device, per_bin_mae, predict_split
from skyfinder.training.families import completed, load_yaml
from skyfinder.training.fds import FDS
from skyfinder.training.lds import MIN_TEMP
from skyfinder.training.model import FDSModel, build_model


def build_net(cfg: Config, device: str) -> torch.nn.Module:
    """Reconstruct the trained architecture (mirrors trainer.run_baseline)."""
    vanilla = build_model(cfg.model, freeze_backbone=cfg.freeze_backbone)
    if not cfg.use_fds:
        return vanilla.to(device)
    feat = 2048 if cfg.model == "resnet50" else 768
    buckets = int(np.ceil((55.0 - MIN_TEMP) / cfg.bin_width))
    fds = FDS(feature_dim=feat, bucket_num=buckets, kernel=cfg.fds_kernel, ks=cfg.fds_ks,
              sigma=cfg.fds_sigma, momentum=cfg.fds_momentum, start_smooth=cfg.fds_start_smooth)
    return FDSModel(vanilla, fds, bin_width=cfg.bin_width, min_temp=MIN_TEMP).to(device)


def infer_one(name: str, cfg: Config, df: pd.DataFrame, splits: list, device: str):
    fold = splits[cfg.fold]
    test_df = df.iloc[fold["test"]].reset_index(drop=True)
    loader = DataLoader(SkyFinderDataset(test_df, EVAL_TF, img_dir=cfg.img_dir),
                        batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    net = build_net(cfg, device)
    net.load_state_dict(load_model_weights(name))
    net.eval()
    preds, ys = predict_split(net, loader, device)
    train_y = df["TempM"].to_numpy()[fold["train"]]
    binned = per_bin_mae(ys, preds, train_y, bin_w=cfg.bin_width)
    return preds, ys, binned, len(test_df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--task-id", type=int, default=None)
    args = ap.parse_args()

    ycfg = load_yaml(args.config)
    matrix, results_dir = build_matrix(ycfg)
    cfg_module.RESULTS_DIR = results_dir
    cfg0 = matrix[0][1]
    df = pd.read_csv(cfg0.labels_path)
    splits = json.loads(cfg0.splits_path.read_text())
    device = get_device()

    sel = [matrix[args.task_id]] if args.task_id is not None else matrix
    for name, cfg in sel:
        if not completed(name):
            print(f"[skip] {name}: no results JSON (training not finished)")
            continue
        path = results_dir / subdir_for(name) / f"{name}.json"
        results = load_results(path)
        if results.get("test_preds"):
            print(f"[skip] {name}: test already present")
            continue
        preds, ys, binned, n = infer_one(name, cfg, df, splits, device)
        results["test_preds"] = preds.tolist()
        results["test_ys"] = ys.tolist()
        results["test_final"] = binned
        results["n_test"] = n
        save_results(results)
        print(f"[ok] {name}: test overall={binned['overall']:.3f} few={binned['few']:.3f} (n={n})")


if __name__ == "__main__":
    main()
