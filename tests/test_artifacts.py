import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("artifact_check", ROOT / "scripts/check_artifacts.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class DistributionChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "model.bin").write_bytes(b"recorded model")
        self.files = {"model.bin": hashlib.sha256(b"recorded model").hexdigest()}
        self.manifest()

    def manifest(self):
        (self.root / "DISTRIBUTION.json").write_text(json.dumps({"files": self.files}))

    def test_complete_distribution(self):
        self.assertEqual(checker.check(self.root), {"status": "passed", "files": 1})

    def test_changed_model_is_rejected(self):
        (self.root / "model.bin").write_bytes(b"changed model")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            checker.check(self.root)

    def test_unlisted_file_is_rejected(self):
        (self.root / "extra.txt").write_text("unlisted")
        with self.assertRaisesRegex(ValueError, "unlisted"):
            checker.check(self.root)

    def test_parent_path_is_rejected(self):
        self.files = {"../model.bin": self.files["model.bin"]}
        self.manifest()
        with self.assertRaisesRegex(ValueError, "invalid distribution path"):
            checker.check(self.root)


if __name__ == "__main__":
    unittest.main()
