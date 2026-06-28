import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from skyfinder.analysis.aggregate import build_table, config_kind
from skyfinder.analysis.per_camera import nearest_geo_distance_km, per_camera_stats
from skyfinder.training.splits import make_split_manifest


class AnalysisMetadataTest(unittest.TestCase):
    def test_explicit_analysis_method_overrides_dir_flags(self) -> None:
        self.assertEqual(
            config_kind(
                {"method": "cam_conditioned", "use_lds": False, "use_fds": False}
            ),
            "cam_conditioned",
        )


    def test_nearest_geo_distance_uses_great_circle_distance(self) -> None:
        # One degree of longitude on the equator is approximately 111.2 km.
        distance = nearest_geo_distance_km(np.array([0.0, 0.0]), np.array([[0.0, 1.0]]))
        self.assertGreater(distance, 111.0)
        self.assertLess(distance, 112.0)

    def test_aggregate_includes_compact_c1_c2_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "labels.csv"
            labels.write_text("Filename,TempM\na.jpg,1.0\n")
            splits = root / "splits.json"
            folds = [{"fold": 0, "train": [0], "val": [], "test": [], "test_cams": []}]
            splits.write_text(json.dumps(make_split_manifest(folds, labels, n_rows=1)))
            metrics = {"overall": 1.0, "many": 1.0, "medium": 1.0, "few": 1.0}
            (root / "c1.json").write_text(
                json.dumps(
                    {
                        "method": "c1_constants",
                        "per_fold": [
                            {"fold": 0, "predictor": "global_mean", "val": metrics, "test": metrics}
                        ],
                    }
                )
            )
            (root / "c2.json").write_text(
                json.dumps(
                    {
                        "method": "c2_metadata",
                        "per_fold": [{"fold": 0, "val": metrics, "test": metrics}],
                    }
                )
            )

            table = build_table(root, labels, splits, split="test")
            self.assertEqual(set(table["kind"]), {"c1_global_mean", "c2_metadata"})

    def test_per_camera_stats_reads_saved_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "baseline_resnet50"
            run_dir.mkdir()
            (run_dir / "baseline_resnet50_fold0.json").write_text(
                json.dumps({"test_preds": [1.5], "test_ys": [1.0]})
            )
            labels = pd.DataFrame(
                {
                    "CamId": [1, 2],
                    "TempM": [0.0, 1.0],
                    "Latitude": [0.0, 0.0],
                    "Longitude": [0.0, 1.0],
                }
            )
            splits = [{"fold": 0, "train": [0], "val": [], "test": [1]}]

            stats = per_camera_stats(labels, splits, root, "baseline_resnet50")
            self.assertEqual(len(stats), 1)
            self.assertAlmostEqual(stats.iloc[0]["mae"], 0.5)
