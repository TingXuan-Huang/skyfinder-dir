from pathlib import Path
import tempfile
import unittest

from skyfinder.training.migrate import plan_misrouted_repair, run_misrouted_repair


class MisroutedResultsRepairTest(unittest.TestCase):
    def test_repairs_only_known_misrouted_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            random_results = root / "results_random"
            cam_results = root / "results_cam_cond"
            (results / "baseline_resnet50_rand").mkdir(parents=True)
            (results / "cam_cond_resnet50").mkdir()
            (results / "baseline_resnet50").mkdir()

            moves, conflicts = plan_misrouted_repair(results, random_results, cam_results)
            self.assertEqual(conflicts, [])
            self.assertEqual(
                {(source.name, destination.parent.name) for source, destination in moves},
                {
                    ("baseline_resnet50_rand", "results_random"),
                    ("cam_cond_resnet50", "results_cam_cond"),
                },
            )
            self.assertEqual(run_misrouted_repair(results, random_results, cam_results, dry_run=False), 0)
            self.assertTrue((random_results / "baseline_resnet50_rand").is_dir())
            self.assertTrue((cam_results / "cam_cond_resnet50").is_dir())
            self.assertTrue((results / "baseline_resnet50").is_dir())
