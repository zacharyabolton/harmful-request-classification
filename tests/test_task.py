"""Task isolation with synthetic inputs."""

import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as Args
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from state import read, jsonl, lines, validate_config
from preparation import prepare
from session import start, state
from runtime import lineage
from measurement import threshold_sweep
from early_stopping import EarlyStopping, validation_operating_point
from smoke_test import fixture_rows, approve
from planning import plan, approve_plan
from experiments import train, freeze
from workflow import select


def task_config():
    cfg = read(ROOT / "tests/fixtures/config.json")
    cfg.update(
        fixture_only=True,
        max_chars=20000,
        max_rows=12000,
        run_budget_seconds=18000,
        objective="recall_at_fpr",
        max_fpr=0.25,
    )
    cfg["task"] = {
        "task_id": "synthetic-routing",
        "author": "Synthetic tester",
        "target": "Route synthetic support tickets",
        "inputs": "Ticket text only",
        "label_map": {"0": "general", "1": "specialist"},
        "data_permissions": "Synthetic fixtures; no real task data",
        "collection": {
            "extra_sources": {"use": False, "reason": "Use supplied fixtures."},
            "synthetic_augmentation": {"use": False, "reason": "No added examples."},
            "sources": [
                {
                    "source": "fixture",
                    "revision": "synthetic-v1",
                    "origin": "Local test fixtures",
                    "permission": "Test use",
                    "labels": "Synthetic routing labels",
                    "role": "supplied",
                }
            ],
            "model_checks": [
                {
                    "model": "tfidf",
                    "revision": "local",
                    "status": "not_applicable",
                    "evidence": "No pretrained weights",
                    "limits": "Does not assess encoder pretraining",
                }
            ],
        },
        "split_policy": "supplied",
        "review_per_class": 2,
        "minimums": {
            s: {"rows": 4, "per_class": 2, "groups": 2}
            for s in ("train", "validation", "test")
        },
    }
    return cfg


class TaskTests(unittest.TestCase):
    def test_incomplete_template_and_unsupported_labels_are_rejected(self):
        f1_config = task_config()
        f1_config.update(objective="max_f1", max_fpr=None)
        validate_config(f1_config)
        cfg = task_config()
        self.assertEqual(validate_config(cfg)["run_budget_seconds"], 18000)
        cfg["task"]["label_map"]["2"] = "third class"
        with self.assertRaisesRegex(ValueError, "two distinct"):
            validate_config(cfg)

    def test_task_support_review_and_supplied_imbalance_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg = task_config()
            rows = [
                r
                for r in fixture_rows()
                if r["label"] == 0 or int(r["id"].split(r["split"])[1]) < 6
            ]
            jsonl(base / "rows.jsonl", rows)
            prepare(base / "rows.jsonl", base / "data", cfg)
            self.assertEqual(
                sum(
                    len(lines(base / "data" / f"{s}.jsonl"))
                    for s in cfg["task"]["minimums"]
                ),
                len(rows),
            )
            self.assertEqual(len(read(base / "data/review.json")["items"]), 4)
            cfg["task"]["minimums"]["validation"]["per_class"] = 100
            with self.assertRaisesRegex(ValueError, "declared support"):
                prepare(base / "rows.jsonl", base / "insufficient", cfg)
            for r in rows:
                r.pop("split")
            jsonl(base / "unsplit.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "supply task-specific"):
                prepare(base / "unsplit.jsonl", base / "unsplit", task_config())

    def test_custom_fpr_is_used_by_threshold_and_stopping(self):
        cfg = task_config()
        y = [0] * 4 + [1] * 4
        scores = [0.9, 0.1, 0.1, 0.1] + [0.8] * 4
        threshold, _ = threshold_sweep(y, scores, cfg["objective"], 2, cfg["max_fpr"])
        self.assertEqual(threshold, 0.8)
        p = validation_operating_point(
            None, [{"label": label} for label in y], cfg, scores
        )
        self.assertEqual(p["fpr"], 0.25)
        self.assertTrue(
            EarlyStopping(2, 0, max_fpr=0.25).observe(1, p)["checkpoint_improved"]
        )
        with self.assertRaisesRegex(ValueError, "FPR constraint"):
            EarlyStopping(2, 0).observe(1, p)

    def test_task_starts_empty_and_rejects_foreign_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg = task_config()
            jsonl(base / "rows.jsonl", fixture_rows())
            prepare(base / "rows.jsonl", base / "data", cfg)
            a = Args(
                input=base / "data",
                out=base / "first",
                kind="fixture",
                replication_of=None,
            )
            with patch("session.save_environment"):
                start(a, cfg)
            run, _ = state(a.out)
            self.assertEqual(
                lineage(a.out, run), {"heldout_exposure": "unknown", "baselines": {}}
            )
            self.assertEqual(
                read(a.out / "contract.json")["target"], cfg["task"]["target"]
            )
            a.parent = a.out
            a.out = base / "second"
            with patch("session.save_environment"):
                start(a, cfg)
            self.assertEqual(
                lineage(a.out, state(a.out)[0])["parent_run_id"], run["run_id"]
            )
            other = copy.deepcopy(cfg)
            other["task"]["task_id"] = "unrelated-task"
            prepare(base / "rows.jsonl", base / "other-data", other)
            a.input = base / "other-data"
            a.out = base / "foreign"
            with self.assertRaisesRegex(ValueError, "same declared task"):
                start(a, other)
            legacy = read(ROOT / "tests/fixtures/config.json")
            legacy["fixture_only"] = True
            prepare(base / "rows.jsonl", base / "legacy-data", legacy)
            legacy_args = Args(
                input=base / "legacy-data",
                out=base / "legacy",
                kind="fixture",
                replication_of=None,
            )
            with patch("session.save_environment"):
                start(legacy_args, legacy)
            a.input = base / "data"
            a.parent = legacy_args.out
            a.out = base / "from-research"
            with self.assertRaisesRegex(ValueError, "undeclared task history"):
                start(a, cfg)

    def test_unfinished_decisions_name_the_next_step(self):
        from session import gates, check_freeze
        from workflow import confirm

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg = task_config()
            jsonl(base / "rows.jsonl", fixture_rows())
            prepare(base / "rows.jsonl", base / "data", cfg)
            args = Args(
                input=base / "data",
                out=base / "run",
                kind="fixture",
                replication_of=None,
                author="Synthetic tester",
                target=cfg["task"]["target"],
            )
            with patch("session.save_environment"):
                start(args, cfg)
            run, _ = state(args.out)
            with self.assertRaisesRegex(ValueError, "run confirm"):
                gates(args.out, run)
            confirm(args, run)
            with self.assertRaisesRegex(ValueError, "finish review"):
                gates(args.out, run)
            approve(args.out)
            args.experiment = None
            with self.assertRaisesRegex(ValueError, "select a trained experiment"):
                freeze(args, cfg)
            with self.assertRaisesRegex(ValueError, "freeze the selected model"):
                check_freeze(args.out)

    def test_task_labels_survive_real_synthetic_fit_and_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg = task_config()
            jsonl(base / "rows.jsonl", fixture_rows())
            prepare(base / "rows.jsonl", base / "data", cfg)
            a = Args(
                input=base / "data",
                out=base / "run",
                kind="fixture",
                replication_of=None,
                experiment="tfidf",
                model=None,
                config=None,
                hypothesis="Test synthetic routing.",
                stop_rule="One synthetic fit.",
                author="Synthetic tester",
                reason="Synthetic validation choice.",
            )
            with patch("session.save_environment"):
                start(a, cfg)
            approve(a.out)
            plan(a)
            approve_plan(a)
            train(a, cfg)
            select(a, state(a.out)[0])
            freeze(a, cfg)
            self.assertEqual(
                read(a.out / "freeze.json")["label_map"], cfg["task"]["label_map"]
            )

