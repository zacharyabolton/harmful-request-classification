"""Check challenger comparisons without loading model weights."""

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("laya_experiment", ROOT / "scripts/experiment_laya.py")
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


class LayaComparisonChecks(unittest.TestCase):
    def test_threshold_keeps_tied_benign_scores_together(self):
        rows = [{"label": 0}] * 500 + [{"label": 1}] * 500
        scores = [0.1] * 474 + [0.8] * 26 + [0.9] * 450 + [0.8] * 50
        point = experiment.operating_point(rows, scores)
        self.assertEqual(point["confusion"], {"tn": 500, "fp": 0, "fn": 50, "tp": 450})
        self.assertEqual(point["threshold"], 0.9)

    def test_paired_comparison_checks_identity_and_class(self):
        rows = [{"id": "a", "label": 0}, {"id": "b", "label": 1}]
        reference = [{"id": "a", "score": 0.8}, {"id": "b", "score": 0.8}]
        paired = experiment.paired_errors(rows, [0.2, 0.2], 0.5, reference, 0.5)
        self.assertEqual(paired["benign"]["fixed"], 1)
        self.assertEqual(paired["harmful"]["introduced"], 1)
        with self.assertRaisesRegex(ValueError, "order mismatch"):
            experiment.paired_errors(rows, [0.2, 0.2], 0.5, reference[::-1], 0.5)


if __name__ == "__main__":
    unittest.main()
