"""Reject changed historical inputs before starting a fixed reproduction."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import reproduce_encoder
import diagnose_tfidf
import session
from state import dump, sha


class FixedReproductionTests(unittest.TestCase):
    def test_snapshot_preserves_gpu_pins_and_license(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            for name, content in (("requirements.colab.txt", "torch==2.11.0+cu130\n"), ("LICENSE", "Original code license\n")):
                (root / name).write_text(content)
            destination = Path(tmp) / "snapshot"
            with patch.object(session, "ROOT", root):
                before = session.snapshot(destination)
                (root / "requirements.colab.txt").write_text("torch==2.8.0\n")
                after = session.source_hashes()
            self.assertNotEqual(before["requirements.colab.txt"], after["requirements.colab.txt"])
            for name in ("requirements.colab.txt", "LICENSE"):
                self.assertEqual(sha(destination / name), before[name])

    def test_required_historical_match_fails_and_keeps_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "diagnostic.json"
            result = {"historical_match": False, "fits": {"native": {"maximum_score_error": 0.04}}}
            with patch.object(diagnose_tfidf, "diagnose", return_value=result), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit, "historical feature selection or scores differ"):
                    diagnose_tfidf.main(["--data", tmp, "--output", str(output), "--require-historical-match"])
            import json
            self.assertEqual(json.loads(output.read_text()), result)
            with patch.object(diagnose_tfidf, "diagnose") as fit, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    diagnose_tfidf.main(["--data", tmp, "--output", str(output)])
                fit.assert_not_called()
            self.assertEqual(json.loads(output.read_text()), result)

    def test_changed_split_cannot_start_training(self):
        for changed in ("train", "validation"):
            with self.subTest(split=changed), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data = root / "inputs"
                data.mkdir()
                for split in ("train", "validation"):
                    (data / (split + ".jsonl")).write_text('{"id":"original"}\n')
                dump(root / "data/split-index.json", {
                    "split_sha256": {s: sha(data / (s + ".jsonl")) for s in ("train", "validation")}
                })
                (data / (changed + ".jsonl")).write_text('{"id":"changed"}\n')
                output = root / "output"
                with patch.object(reproduce_encoder, "ROOT", root), patch.object(reproduce_encoder, "train_encoder") as train:
                    with self.assertRaisesRegex(ValueError, "split checksum mismatch: " + changed):
                        reproduce_encoder.reproduce(data, output)
                    train.assert_not_called()
                self.assertFalse(output.exists())

    def test_existing_reproduction_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            output.mkdir()
            sentinel = output / "report.json"
            sentinel.write_text("existing result\n")
            with patch.object(reproduce_encoder, "train_encoder") as train:
                with self.assertRaisesRegex(ValueError, "output exists"):
                    reproduce_encoder.reproduce(Path(tmp) / "absent", output)
                train.assert_not_called()
            self.assertEqual(sentinel.read_text(), "existing result\n")


if __name__ == "__main__":
    unittest.main()
