"""Exercise fail-closed preflight against real temporary test suites."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from preflight import run_checks


class PreflightTests(unittest.TestCase):
    def test_skipped_and_empty_suites_fail(self):
        for body in (
            "",
            "import unittest\n@unittest.skip('deliberate')\nclass TestSkipped(unittest.TestCase):\n    def test_skip(self): pass\n",
        ):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "tests").mkdir()
                (root / "tests/test_probe.py").write_text(body)
                with self.assertRaisesRegex(RuntimeError, "failed or skipped"):
                    run_checks(root)
                report = json.loads((root / ".local/checks/preflight.json").read_text())
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["checks"][0]["mode"], "normal")

    def test_optimized_failure_is_not_hidden_by_normal_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "tests/test_probe.py").write_text(
                "import unittest\nclass TestMode(unittest.TestCase):\n"
                "    def test_mode(self): self.assertTrue(__debug__)\n"
            )
            with self.assertRaisesRegex(RuntimeError, "optimized tests failed"):
                run_checks(root)
            report = json.loads((root / ".local/checks/preflight.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual([c["returncode"] for c in report["checks"]], [0, 1])
