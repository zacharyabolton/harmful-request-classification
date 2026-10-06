"""Check decision gates and recovery with synthetic data."""

import contextlib
import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as Args
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from state import read, dump, tree, jsonl
from session import start, state, gates
from wildjailbreak import prepare as prepare_source
from smoke_test import fixture_rows, approve
from planning import plan, approve_plan, check_plan
from runtime import configure, lineage
from checkpoints import checkpoint, restore
from settings import encoder_settings, planned_steps
from models import training_batches


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.cfg = read(ROOT / "tests/fixtures/config.json")
        self.cfg["fixture_only"] = True
        jsonl(self.base / "rows.jsonl", fixture_rows())
        prepare_source(self.base / "rows.jsonl", self.base / "data", self.cfg)
        self.out = self.base / "run"
        self.args = Args(
            input=self.base / "data", out=self.out, kind="fixture", replication_of=None
        )
        with patch("session.save_environment"):
            start(self.args, self.cfg)
        approve(self.out)

    def plan_args(self):
        return Args(
            out=self.out,
            experiment="tfidf",
            model=None,
            config=None,
            hypothesis="Separate the synthetic labels.",
            stop_rule="Stop after one synthetic fit.",
            author="Synthetic tester",
            reason="Accept defaults for synthetic verification.",
        )

    def test_cli_defaults_to_exploratory(self):
        from run import parser

        args = parser().parse_args(["start", "--input", "data", "--out", "run"])
        self.assertEqual(args.mode, "exploratory")

    def test_approval_binds_settings_and_code(self):
        a = self.plan_args()
        run, cfg = state(self.out)
        with self.assertRaisesRegex(ValueError, "plan and approve"):
            check_plan(a, run, cfg)
        plan(a)
        with self.assertRaisesRegex(ValueError, "human approval"):
            check_plan(a, run, cfg)
        approve_plan(a)
        self.assertTrue(check_plan(a, run, cfg))
        with self.assertRaisesRegex(ValueError, "changed"):
            check_plan(a, run, {**cfg, "C": 0.2})
        with patch("planning.source_hashes", return_value={"modified": "code"}):
            with self.assertRaisesRegex(ValueError, "changed"):
                check_plan(a, run, cfg)

    def test_runtime_preserves_clock_and_invalidates_plan(self):
        a = self.plan_args()
        plan(a)
        approve_plan(a)
        run = read(self.out / "run.json")
        run["start_epoch"] -= 10000
        dump(self.out / "run.json", run)
        with self.assertRaises(TimeoutError):
            state(self.out)
        configure(
            Args(
                out=self.out,
                mode="exploratory",
                verification_timeout=900,
                author="Synthetic tester",
                reason="Continue an untimed synthetic exercise.",
            )
        )
        restored, cfg = state(self.out)
        self.assertEqual(restored["start_epoch"], run["start_epoch"])
        with self.assertRaisesRegex(ValueError, "changed"):
            check_plan(a, restored, cfg)
        (self.out / "freeze.json").write_text("{}")
        with self.assertRaises(ValueError):
            configure(
                Args(
                    out=self.out,
                    mode="timed",
                    verification_timeout=None,
                    author="Synthetic tester",
                    reason="Cannot change frozen controls.",
                )
            )

    def test_approval_binds_early_stopping_policy(self):
        a = self.plan_args()
        a.experiment = "deberta-early-stop"
        a.model = "deberta"
        run, cfg = state(self.out)
        cfg["model_options"] = {
            "epochs": 6,
            "early_stopping": {"patience": 2, "min_delta": 0},
        }
        a.config = self.base / "early-stopping.json"
        dump(a.config, cfg)
        plan(a)
        approve_plan(a)
        self.assertTrue(check_plan(a, run, state(self.out)[1]))
        cfg["model_options"]["early_stopping"]["patience"] = 3
        dump(a.config, cfg)
        with self.assertRaisesRegex(ValueError, "changed"):
            check_plan(a, run, state(self.out)[1])

    def test_prepared_exposure_cannot_be_reset(self):
        manifest = read(self.base / "data/manifest.json")
        manifest["heldout_exposure"] = "previously-evaluated"
        dump(self.base / "data/manifest.json", manifest)
        for exposure in ("unknown", "unseen"):
            args = Args(
                input=self.base / "data", out=self.base / exposure,
                kind="fixture", replication_of=None, heldout_exposure=exposure,
            )
            with patch("session.save_environment"):
                start(args, self.cfg)
            self.assertEqual(
                lineage(args.out, state(args.out)[0])["heldout_exposure"],
                "previously-evaluated",
            )

    def test_linked_run_retains_exposure_and_explicit_review(self):
        (self.out / "evaluation").mkdir()
        child = self.base / "child"
        a = Args(
            input=self.base / "data",
            out=child,
            kind="fixture",
            replication_of=None,
            parent=self.out,
            reuse_review=True,
            heldout_exposure="unseen",
        )
        with patch("session.save_environment"):
            start(a, self.cfg)
        run, _ = state(child)
        self.assertEqual(
            lineage(child, run)["heldout_exposure"], "previously-evaluated"
        )
        gates(child, run)
        self.assertEqual(read(child / "review.json"), read(self.out / "review.json"))
        a.out = self.base / "unreviewed"
        a.reuse_review = False
        with patch("session.save_environment"):
            start(a, self.cfg)
        with self.assertRaises(ValueError):
            gates(a.out, state(a.out)[0])

    def test_checkpoint_roundtrip_and_unsafe_archive(self):
        archive = self.base / "checkpoint.tar.gz"
        checkpoint(Args(out=self.out, output=archive))
        restored = self.base / "restored"
        restore(Args(input=archive, out=restored))
        self.assertEqual(tree(restored), tree(self.out))
        with self.assertRaises(ValueError):
            restore(Args(input=archive, out=restored))
        with tarfile.open(archive, "w:gz") as output:
            for name in ("checkpoint-manifest.json", "run/../escape"):
                item = tarfile.TarInfo(name)
                item.size = 2
                output.addfile(item, io.BytesIO(b"{}"))
        with self.assertRaisesRegex(ValueError, "unsafe"):
            restore(Args(input=archive, out=self.base / "unsafe"))
        self.assertFalse((self.base / "unsafe").exists())

    def test_legacy_state_requires_no_new_metadata(self):
        run = read(self.out / "run.json")
        for key in ("workflow_version", "controls_hash", "lineage_hash"):
            run.pop(key)
        dump(self.out / "run.json", run)
        (self.out / "controls.json").unlink()
        (self.out / "lineage.json").unlink()
        self.assertEqual(state(self.out)[0], run)
        self.assertIsNone(check_plan(self.plan_args(), run, self.cfg))



class TrainingBudgetTests(unittest.TestCase):
    def test_epochs_partial_batches_and_no_hidden_cap(self):
        import numpy as np

        random = Args(randperm=lambda n: np.arange(n))
        options = encoder_settings({"model_options": {"epochs": 2, "batch_size": 8}})
        batches = list(training_batches(random, 17, options))
        self.assertEqual([epoch for epoch, _, _ in batches], [0, 0, 0, 1, 1, 1])
        self.assertEqual(
            sum(len(order[start : start + 8]) for _, order, start in batches), 34
        )
        self.assertEqual(planned_steps(4000, options), 1000)
        options["max_steps"] = 4
        self.assertEqual(len(list(training_batches(random, 17, options))), 4)
        for bad in (
            {"epochs": 0},
            {"learning_rate": float("nan")},
            {"max_length": 513},
            {"batch_size": True},
            {"unknown": 1},
        ):
            with self.assertRaises(ValueError):
                encoder_settings({"model_options": bad})


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        unittest.main()
