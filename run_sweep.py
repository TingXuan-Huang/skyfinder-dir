"""Run the main DIR sweep: each experiment x fold -> run_baseline(cfg).

Paths come from the YAML `paths:` block and are passed INTO each Config (Config-threading).
We do NOT mutate module-level paths: a dataclass default captures the module constant at
class-definition time, so module mutation would not reach a freshly-built Config.

Usage:
    python run_sweep.py --list                         # print the (idx, run_name) matrix
    python run_sweep.py --dry-run                       # print resolved Configs, don't train
    python run_sweep.py --task-id 7                     # run one cell (for SLURM arrays)
    python run_sweep.py --experiment lds_resnet50       # one experiment, all its folds
    python run_sweep.py --experiment lds_resnet50 --fold 3
    python run_sweep.py                                 # run everything
"""
from __future__ import annotations

import argparse
from pathlib import Path

from skyfinder.training import config as cfg_module
from skyfinder.training.config import Config
from skyfinder.training.families import load_yaml, resolve_path, expand_experiment, completed
from skyfinder.training.trainer import run_baseline


def build_matrix(ycfg: dict):
    """Flatten YAML experiments x folds into an ordered list of (run_name, Config)."""
    root = resolve_path(Path.cwd(), ycfg.get("project_root", ".")).resolve()
    p = ycfg["paths"]
    paths = dict(
        labels_path=resolve_path(root, p["labels"]),
        splits_path=resolve_path(root, p["splits"]),
        img_dir=resolve_path(root, p["images"]),
    )
    results_dir = resolve_path(root, p["results"])
    matrix = []
    for exp in ycfg["experiments"]:
        for spec in expand_experiment(exp):
            matrix.append((spec["run_name"], Config(**spec, **paths)))
    return matrix, results_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--task-id", type=int, default=None, help="run only the Nth cell (SLURM array)")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--list", action="store_true", help="print the (idx, run_name) matrix and exit")
    ap.add_argument("--dry-run", action="store_true", help="print resolved Configs, don't train")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    ycfg = load_yaml(args.config)
    matrix, results_dir = build_matrix(ycfg)
    cfg_module.RESULTS_DIR = results_dir  # checkpoint I/O reads this at call time

    if args.list:
        for i, (name, _) in enumerate(matrix):
            print(f"{i:3d}  {name}")
        return

    if args.task_id is not None:
        sel = [matrix[args.task_id]]
    else:
        sel = matrix
        if args.experiment is not None:
            sel = [(n, c) for n, c in sel if n.startswith(args.experiment + "_fold")]
        if args.fold is not None:
            sel = [(n, c) for n, c in sel if c.fold == args.fold]

    results_dir.mkdir(parents=True, exist_ok=True)
    for name, cfg in sel:
        if args.skip_existing and completed(name):
            print(f"[skip] {name} (results exist)")
            continue
        if args.dry_run:
            print(f"{name}: model={cfg.model} fold={cfg.fold} epochs={cfg.epochs} "
                  f"use_lds={cfg.use_lds} use_fds={cfg.use_fds} "
                  f"labels={cfg.labels_path.name} splits={cfg.splits_path.name}")
            continue
        print(f"\n===== {name} =====")
        run_baseline(cfg, save=ycfg.get("save", True))


if __name__ == "__main__":
    main()
