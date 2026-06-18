"""Train the camera-conditioned model (standalone — does NOT touch the verified trainer).

Per fold: cam-aware loader (returns CamId), train CamConditionedModel with weighted L1, eval on
val + LOCO test (unknown token handles unseen cameras), save results JSON in the run_baseline
schema (so aggregate.py / per_camera.py ingest it). GPU node.

Usage:
    python cam_cond_train.py --config configs/cam_cond.yaml --task-id 0
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from skyfinder.training import config as cfg_module
from skyfinder.training.cam_conditioned import CamConditionedModel, build_cam_id_to_idx
from skyfinder.training.checkpoint import save_model_weights, save_results
from skyfinder.training.config import Config
from skyfinder.training.dataloader import EVAL_TF, TRAIN_TF
from skyfinder.training.engine import get_device, per_bin_mae
from skyfinder.training.families import expand_experiment, load_yaml, resolve_path
from skyfinder.training.lds import weighted_l1_loss


class CamDataset(Dataset):
    """Like SkyFinderDataset but returns (image, temp, CamId) — no LDS weight."""

    def __init__(self, df, transform, img_dir):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.img_dir / str(row["CamId"]) / row["Filename"]).convert("RGB")
        return (self.transform(img), torch.tensor(row["TempM"], dtype=torch.float32),
                torch.tensor(int(row["CamId"]), dtype=torch.long))


def _loader(df, tf, cfg, shuffle):
    return DataLoader(CamDataset(df, tf, cfg.img_dir), batch_size=cfg.batch_size,
                      shuffle=shuffle, num_workers=cfg.num_workers)


def _predict(net, loader, device):
    net.eval()
    preds, ys = [], []
    with torch.no_grad():
        for x, y, cam in loader:
            preds.append(net(x.to(device), cam.to(device)).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(preds), np.concatenate(ys)


def train_cam(cfg, df, splits, device, emb_dim, cam_dropout, save=True) -> dict:
    fold = splits[cfg.fold]
    train_df = df.iloc[fold["train"]].reset_index(drop=True)
    val_df = df.iloc[fold["val"]].reset_index(drop=True)
    test_df = df.iloc[fold["test"]].reset_index(drop=True)
    cam_idx = build_cam_id_to_idx(train_df["CamId"])

    net = CamConditionedModel(cfg.model, cam_idx, emb_dim=emb_dim,
                              cam_dropout_prob=cam_dropout, freeze_backbone=cfg.freeze_backbone).to(device)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))
    tl = _loader(train_df, TRAIN_TF, cfg, shuffle=True)
    vl = _loader(val_df, EVAL_TF, cfg, shuffle=False)
    tel = _loader(test_df, EVAL_TF, cfg, shuffle=False)
    train_y = train_df["TempM"].to_numpy()

    best, best_state, best_vp, best_vy = 1e9, None, None, None
    for ep in range(cfg.epochs):
        net.train()
        for x, y, cam in tl:
            loss = weighted_l1_loss(net(x.to(device), cam.to(device)), y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        vp, vy = _predict(net, vl, device)
        vmae = float(np.mean(np.abs(vp - vy)))
        print(f"ep {ep:2d} val_mae={vmae:.3f}")
        if vmae < best:
            best = vmae
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            best_vp, best_vy = vp, vy

    net.load_state_dict(best_state)
    tp, ty = _predict(net, tel, device)
    results = {
        "run_name": cfg.run_name, "config": asdict(cfg), "device": device,
        "final_val": per_bin_mae(best_vy, best_vp, train_y),
        "test_final": per_bin_mae(ty, tp, train_y), "best_val_mae": best,
        "val_preds": best_vp.tolist(), "val_ys": best_vy.tolist(),
        "test_preds": tp.tolist(), "test_ys": ty.tolist(),
    }
    if save:
        save_model_weights(best_state, cfg.run_name)
        save_results(results)
    print(f"[done] {cfg.run_name} val={best:.3f} test={results['test_final']['overall']:.3f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cam_cond.yaml")
    ap.add_argument("--task-id", type=int, default=None)
    args = ap.parse_args()

    ycfg = load_yaml(args.config)
    root = resolve_path(Path.cwd(), ycfg.get("project_root", ".")).resolve()
    p = ycfg["paths"]
    paths = dict(labels_path=resolve_path(root, p["labels"]),
                 splits_path=resolve_path(root, p["splits"]),
                 img_dir=resolve_path(root, p["images"]))
    cfg_module.RESULTS_DIR = resolve_path(root, p["results"])
    df = pd.read_csv(paths["labels_path"])
    splits = json.loads(paths["splits_path"].read_text())
    device = get_device()

    matrix = []
    for exp in ycfg["experiments"]:
        emb = int(exp.get("cam_embedding_dim", 64))
        drop = float(exp.get("cam_dropout_prob", 0.05))
        for spec in expand_experiment(exp):
            matrix.append((spec, emb, drop))
    sel = [matrix[args.task_id]] if args.task_id is not None else matrix
    for spec, emb, drop in sel:
        cfg = Config(**{k: v for k, v in spec.items() if k in Config.__dataclass_fields__}, **paths)
        print(f"\n===== {cfg.run_name} (emb={emb}, cam_dropout={drop}) =====")
        train_cam(cfg, df, splits, device, emb, drop, save=ycfg.get("save", True))


if __name__ == "__main__":
    main()
