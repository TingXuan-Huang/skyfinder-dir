"""Run the CPU metadata baselines C1 (constant) + C2 (metadata GBM). No GPU, no images.

Outputs results/_analysis/{c1_constants,c2_metadata_only}.json.
Usage:  python run_baselines.py --config configs/main.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from skyfinder.training.families import load_yaml, resolve_path
from skyfinder.analysis.baselines_constant import run_baselines_constant
from skyfinder.analysis.baselines_metadata import run_baselines_metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main.yaml")
    args = ap.parse_args()

    ycfg = load_yaml(args.config)
    root = resolve_path(Path.cwd(), ycfg.get("project_root", ".")).resolve()
    p = ycfg["paths"]
    out = resolve_path(root, p["results"]) / "_analysis"
    cfg = {
        "labels_path": str(resolve_path(root, p["labels"])),
        "splits_path": str(resolve_path(root, p["splits"])),
        "baselines_constant_path": str(out / "c1_constants.json"),
        "baselines_metadata_path": str(out / "c2_metadata_only.json"),
    }
    print("=== C1 constant baselines ===")
    run_baselines_constant(cfg)
    print("\n=== C2 metadata GBM ===")
    run_baselines_metadata(cfg)


if __name__ == "__main__":
    main()
