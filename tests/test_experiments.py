"""Freeze and reload the selected experiment with its saved code."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from session import snapshot
from state import dump, read, jsonl
from smoke_test import fixture_rows, approve, write_report


MODEL = """from state import dump, read


def train(train_rows, validation_rows, cfg, path):
    if cfg.get("model_options", {}).get("fail"):
        raise ValueError("deliberate test failure")
    dump(path / "weights.json", {"value": cfg["C"]})
    return cfg["C"], {}


def load(path, cfg, device):
    return read(path / "weights.json")["value"]


def scores(model, rows, cfg):
    return [model * FACTOR for _ in rows]
"""


class ExperimentLifecycleTests(unittest.TestCase):
    def test_develop_refine_and_freeze_saved_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            workspace = base / "workspace"
            snapshot(workspace)
            run = base / "run"
            settings = read(ROOT / "tests/fixtures/config.json")
            settings["fixture_only"] = True
            cfg = base / "settings.json"
            dump(cfg, settings)
            source = base / "rows.jsonl"
            jsonl(source, fixture_rows())

            def cli(*args, good=True):
                result = subprocess.run(
                    [
                        sys.executable,
                        *(["-O"] if sys.flags.optimize else []),
                        str(workspace / "src/run.py"),
                        *map(str, args),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(
                    result.returncode == 0, good, result.stdout + result.stderr
                )
                return result

            cli(
                "prepare", "--input", source, "--output", base / "data", "--config", cfg
            )
            cli(
                "start",
                "--input",
                base / "data",
                "--out",
                run,
                "--config",
                cfg,
                "--kind",
                "fixture",
            )
            approve(run)
            started = read(run / "run.json")["start_epoch"]
            plugin = workspace / "src/example_model.py"
            plugin.write_text(MODEL.replace("FACTOR", "1.0"))

            def approve_fit(name):
                cli(
                    "plan",
                    "--out",
                    run,
                    "--experiment",
                    name,
                    "--model",
                    "example_model",
                    "--config",
                    cfg,
                    "--hypothesis",
                    "Exercise saved synthetic model settings.",
                    "--stop-rule",
                    "Stop after one synthetic fit.",
                )
                cli(
                    "approve",
                    "--out",
                    run,
                    "--experiment",
                    name,
                    "--author",
                    "Fixture tester",
                    "--reason",
                    "Accept these synthetic test settings.",
                )

            settings["C"] = 0.2
            settings["model_options"] = {"fail": True}
            dump(cfg, settings)
            approve_fit("failed")
            cli(
                "train",
                "--out",
                run,
                "--experiment",
                "failed",
                "--model",
                "example_model",
                "--config",
                cfg,
                good=False,
            )
            failed = next((run / "attempts").glob("failed-*.json"))
            self.assertEqual(read(failed)["status"], "failed")
            self.assertTrue(
                (run / ".experiment-failed/source/src/example_model.py").exists()
            )

            settings["model_options"] = {}
            dump(cfg, settings)
            approve_fit("first")
            cli(
                "train",
                "--out",
                run,
                "--experiment",
                "first",
                "--model",
                "example_model",
                "--config",
                cfg,
            )
            plugin.write_text(MODEL.replace("FACTOR", "0.5"))
            settings["C"] = 0.8
            dump(cfg, settings)
            approve_fit("second")
            cli(
                "train",
                "--out",
                run,
                "--experiment",
                "second",
                "--model",
                "example_model",
                "--config",
                cfg,
            )
            self.assertEqual(read(run / "experiments/first/settings.json")["C"], 0.2)
            self.assertEqual(read(run / "experiments/second/settings.json")["C"], 0.8)
            cli(
                "train",
                "--out",
                run,
                "--experiment",
                "first",
                "--model",
                "example_model",
                good=False,
            )
            cli("status", "--out", run)
            cli(
                "select",
                "--out",
                run,
                "--experiment",
                "first",
                "--author",
                "Fixture tester",
                "--reason",
                "Keep the first synthetic attempt.",
            )
            cli("freeze", "--out", run)
            plugin.write_text(
                'raise RuntimeError("working code is no longer usable")\n'
            )
            cli(
                "train",
                "--out",
                run,
                "--experiment",
                "third",
                "--model",
                "example_model",
                good=False,
            )
            cli(
                "predict",
                "--model",
                run / "experiments/first",
                "--input",
                source,
                "--output",
                base / "direct-predictions.jsonl",
            )
            direct = [
                json.loads(line)
                for line in (base / "direct-predictions.jsonl").read_text().splitlines()
            ]
            self.assertTrue(all(row["score"] == 0.2 for row in direct))
            self.assertTrue(
                all(row["preprocessing"]["truncated"] is None for row in direct)
            )
            cli("evaluate", "--out", run)
            scores = [
                json.loads(line)["score"]
                for line in (run / "evaluation/predictions.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertTrue(all(score == 0.2 for score in scores))
            self.assertEqual(
                read(run / "evaluation/latency.json")["device"], "unspecified"
            )
            cli("verify", "--out", run)
            write_report(run)
            cli("report", "--out", run, "--input", run / "writeup.md")
            self.assertEqual(read(run / "run.json")["start_epoch"], started)
            cli("evaluate", "--out", run, good=False)
            (run / "source/src/example_model.py").write_text("# changed saved file\n")
            cli("verify", "--out", run, good=False)


if __name__ == "__main__":
    unittest.main()
