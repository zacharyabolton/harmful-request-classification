"""Stopping decisions and best-model restoration, without pretrained downloads."""

import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from early_stopping import EarlyStopping, validation_operating_point
from settings import encoder_settings, effective_settings


def point(recall, fpr=0.04, threshold=0.5):
    return {"recall": recall, "fpr": fpr, "threshold": threshold}


class EarlyStoppingTests(unittest.TestCase):
    def test_patience_resets_only_on_strict_recall_improvement(self):
        tracker = EarlyStopping(patience=2, min_delta=0)
        self.assertTrue(tracker.observe(1, point(0.8))["patience_reset"])
        # Better FPR saves the checkpoint but does not reset recall patience.
        entry = tracker.observe(2, point(0.8, 0.02))
        self.assertTrue(entry["checkpoint_improved"])
        self.assertFalse(entry["patience_reset"])
        self.assertEqual(entry["bad_epochs"], 1)
        self.assertTrue(tracker.observe(3, point(0.802))["patience_reset"])
        self.assertFalse(tracker.observe(4, point(0.79))["stop"])
        self.assertTrue(tracker.observe(5, point(0.802))["stop"])
        self.assertEqual(tracker.best["epoch"], 3)
        with self.assertRaisesRegex(ValueError, "already triggered"):
            tracker.observe(6, point(0.9))

    def test_min_delta_is_strict_and_best_checkpoint_is_separate(self):
        tracker = EarlyStopping(patience=2, min_delta=0.125)
        tracker.observe(1, point(0.5))
        entry = tracker.observe(2, point(0.625))
        self.assertTrue(entry["checkpoint_improved"])
        self.assertFalse(entry["patience_reset"])
        entry = tracker.observe(3, point(0.625))
        self.assertTrue(entry["stop"])
        self.assertEqual(tracker.best["epoch"], 2)

    def test_invalid_metrics_and_skipped_epochs_fail(self):
        for bad in (point(float("nan")), point(0.9, 0.051), point(1.1)):
            with self.assertRaises(ValueError):
                EarlyStopping(2, 0).observe(1, bad)
        with self.assertRaisesRegex(ValueError, "consecutive"):
            EarlyStopping(2, 0).observe(2, point(0.9))

    def test_stopping_configuration_is_explicit_and_visible(self):
        cfg = json.loads((ROOT / "tests/fixtures/config.json").read_text())
        self.assertIsNone(encoder_settings(cfg)["early_stopping"])
        for bad in (
            True,
            {},
            {"patience": 2},
            {"patience": True, "min_delta": 0},
            {"patience": 0, "min_delta": 0},
            {"patience": 2, "min_delta": float("nan")},
            {"patience": 2, "min_delta": -0.1},
            {"patience": 2, "min_delta": True},
            {"patience": 2, "min_delta": 0, "monitor": "training_loss"},
        ):
            with self.assertRaises(ValueError):
                encoder_settings({**cfg, "model_options": {"early_stopping": bad}})
        options = {"early_stopping": {"patience": 2, "min_delta": 0}}
        for change in ({"max_steps": 10},):
            with self.assertRaises(ValueError):
                encoder_settings({**cfg, "model_options": {**options, **change}})
        with self.assertRaises(ValueError):
            encoder_settings({**cfg, "objective": "max_f1", "model_options": options})
        settings = effective_settings("deberta", {**cfg, "model_options": options})
        self.assertEqual(settings["early_stopping"], options["early_stopping"])
        self.assertIn(
            "best validation checkpoint",
            settings["early_stopping_policy"]["return_model"],
        )

    def test_epoch_metric_uses_only_supplied_validation_and_reselects_threshold(self):
        rows = [{"label": 0}] * 5 + [{"label": 1}] * 5
        encoder = SimpleNamespace(
            scores=lambda seen, cfg: [0.5] * 5 + [0.9, 0.8, 0.7, 0.1, 0.1]
        )
        result = validation_operating_point(
            encoder, rows, {"fixture_only": True, "objective": "recall_at_fpr"}
        )
        self.assertEqual(result["threshold"], 0.7)
        self.assertEqual(result["recall"], 0.6)
        self.assertEqual(result["fpr"], 0)


class EncoderStoppingIntegrationTests(unittest.TestCase):
    def run_toy(self, *, epochs=6, patience=1, budget_failure=False, enabled=True):
        import numpy as np
        import torch
        from models import train_encoder

        class Batch(dict):
            def to(self, device):
                return self

        class Tokenizer:
            def __call__(self, texts, **kwargs):
                if isinstance(texts, str):
                    return {"input_ids": [1, 2]}
                return Batch(inputs=torch.ones(len(texts), 1))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Linear(1, 2)
                self.register_buffer("updates", torch.tensor(0))

            def forward(self, inputs, labels):
                self.updates += 1
                return SimpleNamespace(
                    loss=torch.nn.functional.cross_entropy(
                        self.classifier(inputs), labels
                    )
                )

        class ToyEncoder:
            def __init__(self, path, device="cpu", max_length=256, revision=None):
                self.model = Model()
                self.tokenizer = Tokenizer()
                self.device = device
                self.max_length = max_length
                self.torch = torch
                if Path(path).is_dir():
                    self.model.load_state_dict(
                        torch.load(Path(path) / "weights.pt", weights_only=True)
                    )

            def sync(self):
                pass

            def scores(self, rows, cfg):
                epoch = int(self.model.updates) // 2
                positive = {
                    0: [0.9, 0.8, 0.1, 0.1, 0.1],
                    1: [0.9, 0.8, 0.1, 0.1, 0.1],
                    2: [0.9, 0.8, 0.7, 0.6, 0.1],
                }.get(epoch, [0.9, 0.8, 0.7, 0.1, 0.1])
                return np.array(([0.5] * 5 + positive)[: len(rows)])

            def save(self, path):
                if budget_failure:
                    clock[0] = 2000
                Path(path).mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), Path(path) / "weights.pt")

        train = [{"raw": {"text": "synthetic"}, "label": i % 2} for i in range(8)]
        val = [{"raw": {"text": "synthetic"}, "label": int(i >= 5)} for i in range(10)]
        cfg = json.loads((ROOT / "tests/fixtures/config.json").read_text())
        cfg.update(fixture_only=True, encoder_device="cpu")
        cfg["model_options"] = {
            "epochs": epochs,
            "batch_size": 4,
            "early_stopping": {"patience": patience, "min_delta": 0}
            if enabled
            else None,
        }
        clock = [0.0]

        def tick():
            clock[0] += 0.01
            return clock[0]

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("models.Encoder", ToyEncoder),
            patch("models.time.monotonic", side_effect=tick),
        ):
            path = Path(tmp) / "encoder"
            encoder, diagnostics = train_encoder(train, val, cfg, path)
            history_path = path.parent / "epoch_history.json"
            history = (
                json.loads(history_path.read_text()) if history_path.exists() else None
            )
            self.assertTrue(math.isfinite(diagnostics["seconds"]))
            return encoder, diagnostics, history

    def test_early_stop_restores_better_earlier_epoch(self):
        encoder, diagnostics, history = self.run_toy()
        self.assertEqual(diagnostics["steps"], 6)
        self.assertEqual(diagnostics["planned_steps"], 12)
        self.assertEqual(diagnostics["completed_epochs"], 3)
        self.assertEqual(diagnostics["early_stopping"]["best_epoch"], 2)
        self.assertEqual(
            diagnostics["early_stopping"]["stop_reason"], "patience_exhausted"
        )
        self.assertEqual(int(encoder.model.updates), 4)
        self.assertEqual(history[-1]["validation"]["recall"], 0.6)
        self.assertEqual(
            diagnostics["early_stopping"]["best_validation"]["recall"], 0.8
        )

    def test_epoch_limit_also_returns_best_checkpoint(self):
        encoder, diagnostics, _ = self.run_toy(epochs=3, patience=5)
        self.assertEqual(diagnostics["early_stopping"]["stop_reason"], "epoch_limit")
        self.assertEqual(int(encoder.model.updates), 4)

    def test_disabled_stopping_preserves_final_epoch_behavior(self):
        encoder, diagnostics, history = self.run_toy(epochs=3, enabled=False)
        self.assertNotIn("early_stopping", diagnostics)
        self.assertIsNone(history)
        self.assertEqual(int(encoder.model.updates), 6)

    def test_epoch_checkpoint_timeout_fails_instead_of_returning_best(self):
        with self.assertRaisesRegex(
            TimeoutError, "epoch validation/checkpoint deadline"
        ):
            self.run_toy(budget_failure=True)


if __name__ == "__main__":
    unittest.main()
