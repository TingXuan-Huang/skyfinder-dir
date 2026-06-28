"""Partial DINOv2 fine-tuning for SkyFinder LOCO regression.

The frozen DINO probe does not test adaptation. This driver fine-tunes only the
last N transformer blocks plus the regression head, preserving most of the
foundation representation while adapting to SkyFinder imagery. It saves the
same raw val/test prediction fields as the main sweep and resumes after Slurm
preemption from ``*_last.pt``.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from skyfinder.analysis.dino_probe import load_dino
from skyfinder.training.checkpoint import (load_training_state, save_model_weights,
                                           save_results, save_training_state, subdir_for)
from skyfinder.training.dataloader import EVAL_TF, TRAIN_TF, SkyFinderDataset
from skyfinder.training.engine import get_device, per_bin_mae, predict_split, seed_everything
from skyfinder.training.families import load_yaml, resolve_path
from skyfinder.training.splits import get_fold, load_splits


@dataclass
class DinoFineTuneConfig:
    variant: str
    fold: int
    epochs: int
    batch_size: int
    head_lr: float
    backbone_lr: float
    unfreeze_blocks: int
    num_workers: int
    seed: int
    labels_path: Path
    splits_path: Path
    img_dir: Path
    results_dir: Path
    run_name: str
    method: str = "dino_v2_partial_finetune"


class DinoRegressor(nn.Module):
    def __init__(self, variant: str, unfreeze_blocks: int):
        super().__init__()
        self.backbone = load_dino(variant, "cpu")
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None or unfreeze_blocks < 1 or unfreeze_blocks > len(blocks):
            raise ValueError(f"unfreeze_blocks must be in [1, {len(blocks) if blocks is not None else 0}]")
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for block in blocks[-unfreeze_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for parameter in self.backbone.norm.parameters():
            parameter.requires_grad_(True)
        embed_dim = int(getattr(self.backbone, "embed_dim"))
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if isinstance(features, dict):
            features = features["x_norm_clstoken"]
        return self.head(features).squeeze(-1)


def build_loaders(cfg: DinoFineTuneConfig, df: pd.DataFrame, splits: list[dict]):
    fold = get_fold(splits, cfg.fold)
    train = df.iloc[fold["train"]].reset_index(drop=True)
    val = df.iloc[fold["val"]].reset_index(drop=True)
    test = df.iloc[fold["test"]].reset_index(drop=True)
    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    return (
        DataLoader(SkyFinderDataset(train, TRAIN_TF, cfg.img_dir), shuffle=True, **common),
        DataLoader(SkyFinderDataset(val, EVAL_TF, cfg.img_dir), shuffle=False, **common),
        DataLoader(SkyFinderDataset(test, EVAL_TF, cfg.img_dir), shuffle=False, **common),
        train,
    )


def train_one_epoch(net: nn.Module, loader: DataLoader, optimizer, device: str) -> float:
    net.train()
    total = count = 0
    for x, y, _ in loader:
        prediction = net(x.to(device))
        loss = F.l1_loss(prediction, y.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * len(y)
        count += len(y)
    return total / count


def run_fold(cfg: DinoFineTuneConfig) -> dict:
    seed_everything(cfg.seed)
    df = pd.read_csv(cfg.labels_path)
    splits = load_splits(cfg.splits_path, cfg.labels_path, len(df))
    train_loader, val_loader, test_loader, train_df = build_loaders(cfg, df, splits)
    device = get_device()
    net = DinoRegressor(cfg.variant, cfg.unfreeze_blocks).to(device)
    head_params = list(net.head.parameters())
    backbone_params = [p for name, p in net.named_parameters() if name != "head.weight" and name != "head.bias" and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": cfg.backbone_lr}, {"params": head_params, "lr": cfg.head_lr}],
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    train_y = train_df["TempM"].to_numpy()
    best_mae, best_state, best_preds, best_ys, best_epoch = float("inf"), None, None, None, -1
    best_test_mae, best_test_epoch = float("inf"), -1
    history = []
    start_epoch = 0
    resume = load_training_state(cfg.run_name, cfg.results_dir)
    if resume is not None:
        net.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        history = resume["history"]
        best_mae, best_state = resume["best_mae"], resume["best_state"]
        best_preds, best_ys, best_epoch = resume["best_preds"], resume["best_ys"], resume["best_epoch"]
        best_test_mae = resume.get("best_test_mae", float("inf"))
        best_test_epoch = resume.get("best_test_epoch", -1)
        start_epoch = int(resume["epoch"]) + 1
        print(f"[resume] epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, cfg.epochs):
        started = time.time()
        train_mae = train_one_epoch(net, train_loader, optimizer, device)
        scheduler.step()
        val_preds, val_ys = predict_split(net, val_loader, device)
        val_mae = float(np.mean(np.abs(val_preds - val_ys)))
        # Per-epoch LOCO test trajectory. Diagnostic only: model selection still
        # uses validation MAE (the honest protocol), but evaluating the held-out
        # cameras every epoch reveals whether the unseen-camera optimum precedes
        # the validation optimum, and how the medium/few tail bins move over time.
        test_preds_ep, test_ys_ep = predict_split(net, test_loader, device)
        test_mae_ep = float(np.mean(np.abs(test_preds_ep - test_ys_ep)))
        test_bins_ep = per_bin_mae(test_ys_ep, test_preds_ep, train_y)
        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
            best_preds, best_ys, best_epoch = val_preds, val_ys, epoch
            save_model_weights(best_state, cfg.run_name, cfg.results_dir)
        if test_mae_ep < best_test_mae:
            best_test_mae, best_test_epoch = test_mae_ep, epoch
        history.append({"epoch": epoch, "train_mae": train_mae, "val_mae": val_mae,
                        "test_mae": test_mae_ep, "test_bins": test_bins_ep,
                        "sec": time.time() - started, "is_best": val_mae == best_mae})
        save_training_state({"epoch": epoch, "model": net.state_dict(), "optimizer": optimizer.state_dict(),
                             "scheduler": scheduler.state_dict(), "history": history, "best_mae": best_mae,
                             "best_state": best_state, "best_preds": best_preds, "best_ys": best_ys,
                             "best_epoch": best_epoch, "best_test_mae": best_test_mae,
                             "best_test_epoch": best_test_epoch}, cfg.run_name, cfg.results_dir)
        print(f"epoch={epoch:02d} train_mae={train_mae:.3f} val_mae={val_mae:.3f} "
              f"test_mae={test_mae_ep:.3f} best_val={best_mae:.3f}@{best_epoch} "
              f"best_test={best_test_mae:.3f}@{best_test_epoch}", flush=True)

    if best_state is None or best_preds is None or best_ys is None:
        raise RuntimeError("fine-tuning produced no validation checkpoint")
    net.load_state_dict(best_state)
    test_preds, test_ys = predict_split(net, test_loader, device)
    result = {
        "run_name": cfg.run_name,
        "config": asdict(cfg),
        "device": device,
        "n_train": len(train_df),
        "n_val": len(best_ys),
        "n_test": len(test_ys),
        "history": history,
        "best_epoch": best_epoch,
        "best_val_mae": best_mae,
        "best_test_epoch": best_test_epoch,
        "best_test_mae": best_test_mae,
        "final_val": per_bin_mae(best_ys, best_preds, train_y),
        "test_final": per_bin_mae(test_ys, test_preds, train_y),
        "val_preds": best_preds.tolist(),
        "val_ys": best_ys.tolist(),
        "test_preds": test_preds.tolist(),
        "test_ys": test_ys.tolist(),
    }
    save_results(result, cfg.results_dir)
    last = cfg.results_dir / subdir_for(cfg.run_name) / f"{cfg.run_name}_last.pt"
    if last.exists():
        last.unlink()
    print(f"[done] {cfg.run_name} val={best_mae:.3f} test={result['test_final']['overall']:.3f}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dino_finetune.yaml")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raw = load_yaml(args.config)
    root = resolve_path(Path.cwd(), raw.get("project_root", ".")).resolve()
    paths = raw["paths"]
    folds = list(raw["folds"])
    run_name_prefix = raw.get("run_name_prefix", "dino_v2_finetune")
    matrix = []
    for fold in folds:
        matrix.append(DinoFineTuneConfig(
            variant=raw["variant"], fold=int(fold), epochs=int(raw["epochs"]), batch_size=int(raw["batch_size"]),
            head_lr=float(raw["head_lr"]), backbone_lr=float(raw["backbone_lr"]),
            unfreeze_blocks=int(raw["unfreeze_blocks"]), num_workers=int(raw["num_workers"]), seed=int(raw["seed"]),
            labels_path=resolve_path(root, paths["labels"]), splits_path=resolve_path(root, paths["splits"]),
            img_dir=resolve_path(root, paths["images"]), results_dir=resolve_path(root, paths["results"]),
            run_name=f"{run_name_prefix}_fold{fold}",
        ))
    selected = [matrix[args.task_id]] if args.task_id is not None else matrix
    for cfg in selected:
        result_path = cfg.results_dir / subdir_for(cfg.run_name) / f"{cfg.run_name}.json"
        if raw.get("skip_existing", True) and result_path.exists():
            print(f"[skip] {cfg.run_name}: result exists")
            continue
        if args.dry_run:
            print(asdict(cfg))
            continue
        run_fold(cfg)


if __name__ == "__main__":
    main()
