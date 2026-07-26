from __future__ import annotations

"""Tests for data/splits_genval.py: train/val/test cameras are mutually disjoint.

Usage:  python -m unittest tests.test_splits_genval
"""
import unittest

import pandas as pd

from data.splits_genval import VAL_CAMS_PER_FOLD, build_genval_folds

N_FOLDS = 5


def _synthetic_df():
    rows = []
    for cam in range(10):
        for k in range(3):
            rows.append({"CamId": cam, "TempM": float(cam + k)})
    return pd.DataFrame(rows)


class GenvalSplitTest(unittest.TestCase):
    def test_camera_sets_disjoint_and_complete(self) -> None:
        df = _synthetic_df()
        folds = build_genval_folds(df, n_folds=N_FOLDS,
                                   val_cams_per_fold=VAL_CAMS_PER_FOLD, seed=0)
        self.assertEqual(len(folds), N_FOLDS)

        for fold in folds:
            train_cams = set(df.loc[fold["train"], "CamId"])
            val_cams = set(df.loc[fold["val"], "CamId"])
            test_cams = set(fold["test_cams"])

            # Camera sets mutually disjoint.
            self.assertTrue(train_cams.isdisjoint(val_cams))
            self.assertTrue(train_cams.isdisjoint(test_cams))
            self.assertTrue(val_cams.isdisjoint(test_cams))

            # Exactly VAL_CAMS_PER_FOLD validation cameras.
            self.assertEqual(len(val_cams), VAL_CAMS_PER_FOLD)

            # Row indices: disjoint and cover all rows.
            idx = fold["train"] + fold["val"] + fold["test"]
            self.assertEqual(len(idx), len(df))
            self.assertEqual(set(idx), set(df.index))


if __name__ == "__main__":
    unittest.main()
