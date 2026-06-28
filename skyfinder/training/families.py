"""Experiment-family metadata, YAML loading, and expansion helpers.

A family is a named, checked-in YAML configuration. The public registry only
contains configurations that exist in this repository.
"""
from __future__ import annotations

import fnmatch
import sys
from pathlib import Path

import yaml

from .checkpoint import subdir_for

# ============================================================
# Registry: family -> (YAML config, name-glob pattern, description)
# ============================================================

FAMILIES: dict[str, dict] = {
    "main": {
        "config": "configs/main.yaml",
        "pattern": None,
        "desc": "Main sweep: baseline / LDS / FDS / LDS+FDS x 5 folds (ResNet + ViT)",
    },
    "random_control": {
        "config": "configs/main_random.yaml",
        "pattern": None,
        "desc": "Random row-split control: ResNet and ViT baselines",
    },
    "smoke": {
        "config": "configs/smoke.yaml",
        "pattern": None,
        "desc": "One-fold one-epoch ResNet smoke run",
    },
    "camera_conditioned": {
        "config": "configs/cam_cond.yaml",
        "pattern": None,
        "desc": "Camera-conditioned ResNet sweep (run with cam_cond_train.py)",
    },
}


# Keys read from a YAML experiment block and passed as kwargs to `run_baseline`.
RUN_KEYS = (
    "model", "fold", "epochs", "batch_size", "lr", "num_workers",
    "train_subset", "val_subset", "seed", "run_name",
    "use_lds", "lds_kernel", "lds_ks", "lds_sigma", "lds_reweight",
    "use_fds", "fds_kernel", "fds_ks", "fds_sigma", "fds_momentum", "fds_start_smooth",
    "bin_width",
    "freeze_backbone",                      # linear-probe hook
    "snapshot_every",                      # trajectory hook
)


# ============================================================
# YAML loading and path resolution
# ============================================================

def load_yaml(path: Path | str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def completed(run_name: str, results_dir: Path) -> bool:
    """True iff a results JSON exists for this run under the nested layout.

    (For old flat-layout runs, migrate before checking completion.)
    """
    path = Path(results_dir) / subdir_for(run_name) / f"{run_name}.json"
    return path.exists()


# ============================================================
# Experiment expansion + filtering
# ============================================================

def filter_experiments(experiments: list[dict], pattern: str | None,
                       experiment_name: str | None) -> list[dict]:
    out = list(experiments)
    if pattern is not None:
        out = [e for e in out if fnmatch.fnmatch(e["name"], pattern)]
    if experiment_name is not None:
        out = [e for e in out if e["name"] == experiment_name]
    return out


def expand_experiment(exp: dict) -> list[dict]:
    """One YAML experiment block + N folds -> N specs (one per fold) with `run_name`."""
    specs = []
    for fold in exp["folds"]:
        spec = {k: exp[k] for k in RUN_KEYS if k in exp}
        spec["fold"] = fold
        spec["run_name"] = exp["run_name_template"].format(
            name=exp["name"], model=exp["model"], fold=fold, epochs=exp["epochs"],
        )
        if "method" in exp:
            spec["method"] = exp["method"]
        specs.append(spec)
    return specs


# ============================================================
# Discovery helpers
# ============================================================

def enumerate_family(family: str) -> tuple[Path, list[str]]:
    """Returns (yaml_path, list_of_experiment_names) for one family."""
    if family not in FAMILIES:
        raise KeyError(family)
    entry = FAMILIES[family]
    yaml_path = Path(entry["config"])
    if not yaml_path.exists():
        return yaml_path, []
    data = load_yaml(yaml_path) or {}
    names = [e["name"] for e in data.get("experiments", [])]
    if entry.get("pattern"):
        names = [n for n in names if fnmatch.fnmatch(n, entry["pattern"])]
    return yaml_path, names


def print_list(family: str | None, names_only: bool) -> None:
    if family is None:
        if names_only:
            for fam in FAMILIES:
                print(fam)
            return
        width = max(len(f) for f in FAMILIES)
        print(f"{'family'.ljust(width)}  count  description")
        print("-" * (width + 25))
        for fam, entry in FAMILIES.items():
            try:
                _, names = enumerate_family(fam)
            except KeyError:
                names = []
            print(f"{fam.ljust(width)}  {len(names):>5d}  {entry['desc']}")
        return

    if family not in FAMILIES:
        print(f"unknown family: {family!r}", file=sys.stderr)
        print(f"known families: {', '.join(FAMILIES)}", file=sys.stderr)
        sys.exit(2)

    yaml_path, names = enumerate_family(family)
    if names_only:
        for n in names:
            print(n)
        return
    print(f"# family={family}  config={yaml_path}  ({len(names)} experiments)")
    print(f"# {FAMILIES[family]['desc']}")
    for n in names:
        print(n)


def print_count(family: str) -> None:
    if family not in FAMILIES:
        print(f"unknown family: {family!r}", file=sys.stderr)
        sys.exit(2)
    _, names = enumerate_family(family)
    print(len(names))
