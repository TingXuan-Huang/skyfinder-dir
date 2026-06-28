import json
from pathlib import Path
import tempfile
import unittest

from skyfinder.training.splits import load_splits, make_split_manifest


class SplitManifestTest(unittest.TestCase):
    def test_split_manifest_rejects_changed_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            labels = tmp_path / "labels.csv"
            labels.write_text("Filename,TempM\na.jpg,1\nb.jpg,2\n")
            folds = [{"fold": 0, "train": [0], "val": [], "test": [1], "test_cams": [1]}]
            split_path = tmp_path / "splits.json"
            split_path.write_text(json.dumps(make_split_manifest(folds, labels, n_rows=2)))

            self.assertEqual(load_splits(split_path, labels, n_rows=2), folds)
            labels.write_text("Filename,TempM\nb.jpg,2\na.jpg,1\n")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_splits(split_path, labels, n_rows=2)


    def test_legacy_split_file_warns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            labels = tmp_path / "labels.csv"
            labels.write_text("Filename,TempM\na.jpg,1\n")
            split_path = tmp_path / "legacy.json"
            split_path.write_text('[{"fold": 0, "train": [0], "val": [], "test": [], "test_cams": []}]')

            with self.assertWarnsRegex(RuntimeWarning, "legacy split"):
                self.assertEqual(len(load_splits(split_path, labels, n_rows=1)), 1)
