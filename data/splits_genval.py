from __future__ import annotations

"""Create 5-fold LOCO splits with a CAMERA-DISJOINT validation set (gen-val control).

Same as data/splits.py for the TEST partition: the trainable cameras are shuffled
with numpy RandomState(0) and array_split into 5 disjoint groups; each fold's
held-out group is the test set. The difference is VAL selection: instead of a
random 10% of the remaining rows, we deterministically hold out
VAL_CAMS_PER_FOLD=2 *cameras* from the remaining (training) cameras and use ALL
of their rows as val. So train / val / test camera sets are mutually disjoint.

This is the selection control for "does picking checkpoints on cameras you've
already seen leak generalization signal?" — here val cameras are unseen too.

Writes data/splits/genval_5fold.json: a manifest with the source-labels
fingerprint plus a list of {fold, test_cams, train, val, test}. Row indices are
valid only for that exact data/labels_with_images.csv file.

Usage (from project root):  python data/splits_genval.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from skyfinder.training.splits import make_split_manifest

DATA_DIR = Path("data")
LABELS_PATH = DATA_DIR / "labels_with_images.csv"
OUT_PATH = DATA_DIR / "splits" / "genval_5fold.json"

N_FOLDS = 5
VAL_CAMS_PER_FOLD = 2
SEED = 0


def build_genval_folds(df, n_folds: int = 5, val_cams_per_fold: int = 2, seed: int = 0):
    """Pure: return a list of fold dicts with disjoint train/val/test cameras.

    test_cams come from a RandomState(seed)-shuffled array_split (same as
    data/splits.py). From the remaining cameras (in their shuffled order), the
    first `val_cams_per_fold` become val cameras; the rest are train cameras.
    """
    n = len(df)
    rng = np.random.RandomState(seed)

    cams = sorted(df["CamId"].unique().tolist())
    rng.shuffle(cams)
    cam_groups = [g.tolist() for g in np.array_split(cams, n_folds)]

    folds = []
    for f, test_cams in enumerate(cam_groups):
        test_cams_set = set(test_cams)
        # Remaining cameras in the shuffled order, with test cams removed.
        rem_cams = [c for c in cams if c not in test_cams_set]
        val_cams = rem_cams[:val_cams_per_fold]
        train_cams = rem_cams[val_cams_per_fold:]

        test_idx = df.index[df["CamId"].isin(test_cams)].tolist()
        val_idx = df.index[df["CamId"].isin(val_cams)].tolist()
        train_idx = df.index[df["CamId"].isin(train_cams)].tolist()

        # Sanity: disjoint and complete coverage.
        assert len(set(train_idx) | set(val_idx) | set(test_idx)) == n
        assert len(train_idx) + len(val_idx) + len(test_idx) == n

        folds.append({
            "fold": f,
            "test_cams": sorted(test_cams),
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        })

    return folds


def main() -> None:
    df = pd.read_csv(LABELS_PATH)
    n = len(df)

    folds = build_genval_folds(df, n_folds=N_FOLDS, val_cams_per_fold=VAL_CAMS_PER_FOLD, seed=SEED)

    for fold in folds:
        n_val_cams = len(df.loc[fold["val"], "CamId"].unique())
        print(f"[fold {fold['fold']}]  test_cams={len(fold['test_cams']):2d}  "
              f"val_cams={n_val_cams:2d}  "
              f"train={len(fold['train']):>6,}  val={len(fold['val']):>5,}  "
              f"test={len(fold['test']):>6,}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(make_split_manifest(folds, LABELS_PATH, n), indent=2))
    print(f"[saved] {OUT_PATH}  ({OUT_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
