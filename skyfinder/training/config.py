"""Training config dataclass and default repository paths.

`Config` holds every training option (baseline, LDS, FDS, snapshots).
All LDS/FDS fields default off so existing baseline calls are unchanged.

Paths default to the repository constants but are Config fields, so each run carries
its complete input and output locations instead of relying on mutable module globals:
    Config(labels_path=Path("/content/drive/.../labels_with_images.csv"), ...)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Project root = parent of skyfinder/ package.
# __file__ = .../skyfinder/training/config.py -> parents[2] = repo root.
PROJ = Path(__file__).resolve().parents[2]
DATA = PROJ / "data"
LABELS = DATA / "labels_with_images.csv"
SPLITS = DATA / "splits" / "loco_5fold.json"
IMG_DIR = DATA / "images"
RESULTS_DIR = PROJ / "results"


@dataclass
class Config:
    """Training config. Pass to `run_baseline(cfg=...)` or via kwargs."""

    model: str = "resnet50"          # "resnet50" or "vit_b_16"
    fold: int = 0                    # 0 through 4 for the 5-fold split
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    num_workers: int = 2
    train_subset: int | None = None  # cap train rows for smoke tests
    val_subset: int | None = None
    seed: int = 0
    run_name: str | None = None      # auto-generated from model/fold/epochs/time if None

    # Data paths (injectable; default to the module-level constants).
    labels_path: Path = LABELS
    splits_path: Path = SPLITS
    img_dir: Path = IMG_DIR
    results_dir: Path = RESULTS_DIR

    # LDS (loss-side reweighting; no architecture change)
    use_lds: bool = False
    lds_kernel: str = "gaussian"     # "gaussian" | "triang" | "laplace"
    lds_ks: int = 5                  # odd
    lds_sigma: float = 2.0
    lds_reweight: str = "sqrt_inv"   # "none" | "sqrt_inv" | "inverse"

    # FDS (architecture-side feature calibration)
    use_fds: bool = False
    fds_kernel: str = "gaussian"
    fds_ks: int = 5
    fds_sigma: float = 2.0
    fds_momentum: float = 0.9
    fds_start_smooth: int = 1        # epoch to begin applying calibration

    # Bucket scheme — applies to both LDS and FDS when active
    bin_width: float = 1.0           # °C per bucket; matches DIR paper convention

    # Ablation hooks
    freeze_backbone: bool = False    # D4: linear probe — train only the regression head
    method: str | None = None         # explicit analysis label for non-DIR runs

    # Checkpoint-trajectory hook.
    # When >0, dumps state_dict to <run>_ep{N}.pt at epoch 0 and every Nth epoch.
    snapshot_every: int = 0

    def __post_init__(self) -> None:
        """Normalize path values and reject invalid runs before allocating a model."""
        for field in ("labels_path", "splits_path", "img_dir", "results_dir"):
            setattr(self, field, Path(getattr(self, field)))
        if self.fold < 0:
            raise ValueError("fold must be non-negative")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.lr <= 0 or self.bin_width <= 0:
            raise ValueError("lr and bin_width must be positive")
        for name, value in (("train_subset", self.train_subset), ("val_subset", self.val_subset)):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1 when set")
