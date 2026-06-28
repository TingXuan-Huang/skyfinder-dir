"""Recover final results JSON from a completed *_last.pt training checkpoint.

This is for runs that finished training but failed while writing JSON. It does
not retrain and it keeps *_last.pt by default.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from run_sweep import build_matrix
from skyfinder.training.checkpoint import load_training_state, save_model_weights, save_results, subdir_for
from skyfinder.training.dataloader import build_loaders
from skyfinder.training.engine import per_bin_mae
from skyfinder.training.families import completed, load_yaml


def _parse_task_ids(values: list[str]) -> list[int]:
    task_ids: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-", 1)
                task_ids.extend(range(int(start), int(end) + 1))
            else:
                task_ids.append(int(part))
    return task_ids


def _tolist(value):
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def recover_task(task_id: int, config: str, *, overwrite: bool, save_missing_weights: bool,
                 remove_last: bool, dry_run: bool) -> Path | None:
    ycfg = load_yaml(config)
    matrix, results_dir = build_matrix(ycfg)
    name, cfg = matrix[task_id]
    run_dir = results_dir / subdir_for(name)
    last_path = run_dir / f"{name}_last.pt"
    json_path = run_dir / f"{name}.json"
    weights_path = run_dir / f"{name}.pt"

    print(f"[recover] task={task_id} run={name}")
    print(f"[paths] last={last_path} json={json_path}")

    if completed(name, results_dir) and not overwrite:
        print(f"[skip] {name} already has results JSON")
        return json_path
    if not last_path.exists():
        raise FileNotFoundError(f"missing resume checkpoint: {last_path}")
    if dry_run:
        print("[dry-run] checkpoint exists; no JSON written")
        return None

    state = load_training_state(name, results_dir)
    if state is None:
        raise FileNotFoundError(f"could not load resume checkpoint: {last_path}")

    completed_epoch = int(state["epoch"])
    if completed_epoch + 1 < cfg.epochs:
        raise RuntimeError(
            f"{name} only reached epoch {completed_epoch}; expected final epoch {cfg.epochs - 1}"
        )

    _, _, train_df, val_df = build_loaders(cfg)
    best_preds = state.get("best_preds")
    best_ys = state.get("best_ys")
    if best_preds is None or best_ys is None:
        raise RuntimeError(f"{name} checkpoint has no best predictions/targets")

    best_preds_arr = np.asarray(_tolist(best_preds), dtype=float)
    best_ys_arr = np.asarray(_tolist(best_ys), dtype=float)
    train_y = train_df["TempM"].to_numpy()
    binned = per_bin_mae(best_ys_arr, best_preds_arr, train_y, bin_w=cfg.bin_width)

    checkpoint_path: Path | None = weights_path if weights_path.exists() else None
    if checkpoint_path is None and save_missing_weights:
        best_state = state.get("best_state")
        if best_state is None:
            raise RuntimeError(f"{name} checkpoint has no best_state to save")
        checkpoint_path = save_model_weights(best_state, name, results_dir)

    results = {
        "run_name": name,
        "config": asdict(cfg),
        "device": "recovered_from_last",
        "n_train": len(train_df),
        "n_val": len(val_df),
        "history": state["history"],
        "best_epoch": state["best_epoch"],
        "best_val_mae": state["best_val_mae"],
        "final_val": binned,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "val_preds": best_preds_arr.tolist(),
        "val_ys": best_ys_arr.tolist(),
    }
    out = save_results(results, results_dir)
    print(f"[ok] {name} best_epoch={state['best_epoch']} best_val_mae={state['best_val_mae']:.4f}")

    if remove_last:
        last_path.unlink()
        print(f"[removed] {last_path}")

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--task-id", action="append", default=[], help="task id, comma list, or range")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-missing-weights", action="store_true")
    parser.add_argument("--remove-last", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    task_ids = _parse_task_ids(args.task_id)
    if not task_ids:
        raise SystemExit("provide at least one --task-id")

    for task_id in task_ids:
        recover_task(
            task_id,
            args.config,
            overwrite=args.overwrite,
            save_missing_weights=args.save_missing_weights,
            remove_last=args.remove_last,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
