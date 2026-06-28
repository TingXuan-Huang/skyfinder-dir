from pathlib import Path
import unittest

from skyfinder.training.checkpoint import load_results, save_results
from skyfinder.training.config import Config
from skyfinder.training.families import completed


class CheckpointRoutingTest(unittest.TestCase):
    def test_results_are_saved_to_the_configured_root(self) -> None:
        # unittest does not supply a temporary-path fixture, so isolate artifacts
        # with TemporaryDirectory.
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            results_dir = tmp_path / "alternate-results"
            cfg = Config(run_name="baseline_resnet50_fold0", results_dir=results_dir)
            artifact = save_results({"run_name": cfg.run_name, "config": {}}, cfg.results_dir)

            self.assertEqual(artifact, results_dir / "baseline_resnet50" / "baseline_resnet50_fold0.json")
            self.assertTrue(completed(cfg.run_name, cfg.results_dir))
            self.assertEqual(load_results(artifact)["run_name"], cfg.run_name)
            self.assertFalse((tmp_path / "results" / artifact.name).exists())


    def test_config_rejects_invalid_subset_sizes(self) -> None:
        with self.assertRaisesRegex(ValueError, "train_subset"):
            Config(train_subset=0)
