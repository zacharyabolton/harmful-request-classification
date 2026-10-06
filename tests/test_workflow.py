import copy
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from data import model_text, load_inference, dump
from state import check_review, review_template, validate_config, timed
from measurement import metrics, threshold_sweep, wilson, failure_samples
from wildjailbreak import convert, components, prepare, supports
from smoke_test import fixture_rows, approval
from models import finite

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def test_unlabeled_metadata_exclusion_and_overflow(self):
        r = {
            "id": "one",
            "raw": {"text": "visible", "outcome": "CANARY"},
            "label": True,
            "source": "CANARY",
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.jsonl"
            p.write_text(json.dumps(r) + "\n")
            rows = load_inference(p)
            self.assertEqual(model_text(rows[0])[0], "visible")
            del r["label"]
            self.assertEqual(model_text(r)[0], model_text(rows[0])[0])
        r = {
            "id": "x",
            "target_index": 0,
            "raw": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "a" * 100, "outcome": "CANARY"}
                        ],
                        "outcome": "CANARY",
                    },
                    {"role": "assistant", "content": "FUTURE"},
                ]
            },
        }
        with self.assertRaisesRegex(ValueError, "adaptation"):
            model_text(r, 20)
        text, _ = model_text(r)
        self.assertNotIn("CANARY", text)
        self.assertNotIn("FUTURE", text)
        for raw in (None, [], {"text": ""}):
            with self.assertRaises(ValueError):
                model_text({"raw": raw})

    def test_configuration_rejected_before_work(self):
        cfg = json.loads((ROOT / "tests/fixtures/config.json").read_text())
        for field, value in [
            ("device", "cuda"),
            ("encoder_device", "mps"),
            ("objective", "imaginary"),
            ("fixture_only", 1),
            ("seed", True),
            ("max_rows", 7000),
        ]:
            with self.assertRaises(ValueError):
                validate_config({**cfg, field: value})

    def test_operating_point_hand_computed_boundaries(self):
        t, sweep = threshold_sweep(
            [0, 0, 1, 1], [0.1, 0.4, 0.3, 0.8], "recall_at_fpr", 2
        )
        self.assertEqual(t, 0.8)
        self.assertEqual(metrics([0, 0, 1, 1], [0.1, 0.4, 0.3, 0.8], t)["recall"], 0.5)
        t, _ = threshold_sweep([0, 1], [1.0, 1.0], "recall_at_fpr", 1)
        self.assertGreater(t, 1)
        self.assertEqual(
            metrics([0, 1], [1.0, 1.0], t)["confusion"],
            {"tn": 1, "fp": 0, "fn": 1, "tp": 0},
        )
        m = metrics([0, 1, 1], [0.1, 0.8, 0.8], 0.8)
        self.assertEqual(m["f1"], 1)
        self.assertEqual(m["average_precision"], 1)
        self.assertAlmostEqual(wilson(0, 50)[1], 0.0713476, places=6)
        self.assertTrue(metrics([], [], 0.5)["empty_slice"])
        self.assertIsNone(metrics([1], [0.2], 0.5)["fpr"])
        for scores in ([float("nan"), 0.2], [-0.1, 0.3], [1.1, 0.3]):
            with self.assertRaises(ValueError):
                threshold_sweep([0, 1], scores)
        with self.assertRaises(ValueError):
            threshold_sweep([0, 1], [0.1, 0.9], "recall_at_fpr", 500)

    def test_bound_actual_review_and_order_invariance(self):
        rows = fixture_rows()
        review = review_template(rows, "protocol")
        self.assertEqual(review, review_template(list(reversed(rows)), "protocol"))
        with self.assertRaises(ValueError):
            check_review(review, rows, "protocol", "fixture")
        for item in review["items"]:
            item.update(approval(), human_decision="agree", resolution="retain")
        check_review(review, rows, "protocol", "fixture")
        with self.assertRaises(ValueError):
            check_review(review, rows, "protocol", "research")
        mutated = copy.deepcopy(rows)
        mutated[0]["raw"]["text"] = "changed"
        with self.assertRaises(ValueError):
            check_review(review, mutated, "protocol", "fixture")
        for key, value in [
            ("human_decision", "uncertain"),
            ("author_kind", "agent"),
            ("row_hash", "wrong"),
        ]:
            bad = copy.deepcopy(review)
            bad["items"][0][key] = value
            if key == "author_kind":
                bad["items"][0]["fixture_only"] = False
            with self.assertRaises(ValueError):
                check_review(bad, rows, "protocol", "fixture")

    def test_source_mapping_and_seed_exact_links(self):
        r = convert(
            {
                "vanilla": "seed",
                "adversarial": "input",
                "completion": "CANARY",
                "data_type": "adversarial_harmful",
            },
            1,
        )
        self.assertEqual(model_text(r)[0], "input")
        self.assertEqual(r["label"], 1)
        for bad in (
            {"data_type": "unknown"},
            {"data_type": "vanilla_benign", "vanilla": ""},
            {"data_type": "adversarial_harmful", "vanilla": "seed", "adversarial": ""},
        ):
            with self.assertRaises(ValueError):
                convert(bad, 1)
        a, b = fixture_rows()[:2]
        b["raw"] = a["raw"]
        self.assertEqual(len(components([a, b])), 1)

    def test_supplied_conflicts_and_support_floor(self):
        cfg = json.loads((ROOT / "tests/fixtures/config.json").read_text())
        cfg["fixture_only"] = True
        rows = fixture_rows()
        rows[-1]["group_id"] = rows[0]["group_id"]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.jsonl"
            p.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaisesRegex(ValueError, "supplied partition conflicts"):
                prepare(p, Path(tmp) / "prepared", cfg)
        with self.assertRaises(ValueError):
            supports(fixture_rows(), False)

    def test_failure_sampling_stable_unique_complete(self):
        rows = fixture_rows()[:24]
        predictions = [
            {"id": r["id"], "score": 0.6 + i * 0.01, "prediction": 1 - r["label"]}
            for i, r in enumerate(rows)
        ]
        a = failure_samples(rows, predictions, 0.5)
        b = failure_samples(list(reversed(rows)), list(reversed(predictions)), 0.5)
        self.assertEqual(a, b)
        for group in a.values():
            self.assertEqual(len(group["sample"]), 10)
            self.assertEqual(len({x["id"] for x in group["sample"]}), 10)
            self.assertEqual(
                [x["selection_reason"] for x in group["sample"]].count(
                    "seeded remainder"
                ),
                4,
            )
        small = failure_samples(rows[:8], predictions[:8], 0.5)
        self.assertEqual(sum(len(x["sample"]) for x in small.values()), 8)

    def test_real_subprocess_deadline_and_finite_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            start = time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired):
                timed(
                    [sys.executable, "-c", "import time; time.sleep(10)"],
                    Path(tmp) / "timeout.log",
                    0.15,
                )
            self.assertLess(time.monotonic() - start, 3)
        with self.assertRaises(ValueError):
            finite([float("inf")])


class AdditionalContracts(unittest.TestCase):
    def test_review_correction_regenerates_and_preserves_holdout(self):
        from state import read, jsonl, sha
        from wildjailbreak import revise_training

        cfg = json.loads((ROOT / "tests/fixtures/config.json").read_text())
        cfg["fixture_only"] = True
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source.jsonl"
            jsonl(source, fixture_rows())
            prepared = base / "prepared"
            prepare(source, prepared, cfg)
            review = read(prepared / "review.json")
            for item in review["items"]:
                item.update(approval(), human_decision="agree", resolution="retain")
            review["items"][0].update(
                human_decision="disagree",
                resolution="correct",
                corrected_label=1 - review["items"][0]["original_label"],
                rationale="Synthetic test correction, not a human label decision.",
            )
            dump(base / "review.json", review)
            revised = base / "revised"
            revise_training(prepared, base / "review.json", revised, cfg, "fixture")
            self.assertEqual(sha(prepared / "test.jsonl"), sha(revised / "test.jsonl"))
            self.assertEqual(
                sha(prepared / "validation.jsonl"), sha(revised / "validation.jsonl")
            )
            self.assertNotEqual(
                sha(prepared / "train.jsonl"), sha(revised / "train.jsonl")
            )
            self.assertTrue(
                all(
                    x["human_decision"] == "pending"
                    for x in read(revised / "review.json")["items"]
                )
            )
            self.assertNotEqual(
                read(prepared / "review.json")["protocol_hash"],
                read(revised / "review.json")["protocol_hash"],
            )

    def test_group_allocation_is_order_invariant(self):
        from state import jsonl, lines

        cfg = json.loads((ROOT / "tests/fixtures/config.json").read_text())
        cfg["fixture_only"] = True
        # Enough groups for hash allocation to satisfy every fixture floor.
        records = []
        for copy_no in range(6):
            for r in fixture_rows():
                r = copy.deepcopy(r)
                r.pop("split")
                r["id"] = str(copy_no) + r["id"]
                r["group_id"] = r["id"]
                r["parent_id"] = r["id"]
                r["raw"]["text"] += " " + " ".join(
                    r["id"] + "unique" + str(j) for j in range(20)
                )
                records.append(r)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for name, rs in [("a", records), ("b", list(reversed(records)))]:
                jsonl(base / (name + ".jsonl"), rs)
                prepare(base / (name + ".jsonl"), base / name, cfg)
            for split in ("train", "validation", "test"):
                self.assertEqual(
                    lines(base / "a" / (split + ".jsonl")),
                    lines(base / "b" / (split + ".jsonl")),
                )

    def test_fit_unknown_warning_and_nonfinite_are_not_suppressed(self):
        import warnings
        from unittest.mock import patch
        from models import fit

        cfg = json.loads((ROOT / "tests/fixtures/config.json").read_text())

        def noisy(*args, **kwargs):
            warnings.warn("synthetic unknown numerical warning", RuntimeWarning)
            return None

        with patch("models.Pipeline.fit", side_effect=noisy):
            with self.assertRaisesRegex(RuntimeError, "fit warnings"):
                fit(fixture_rows(), cfg)

    def test_metrics_hand_enumerated_ap_and_equality(self):
        m = metrics([0, 1, 0, 1], [0.9, 0.8, 0.7, 0.6], 0.7)
        self.assertEqual(m["confusion"], {"tn": 0, "fp": 2, "fn": 1, "tp": 1})
        self.assertAlmostEqual(m["average_precision"], 0.5)
        self.assertAlmostEqual(m["f1"], 0.4)
        self.assertEqual(m["recall"], 0.5)
        self.assertEqual(m["fpr"], 1.0)

    def test_inference_csv_equivalence(self):
        import csv

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            r = {
                "id": "x",
                "raw": {"messages": [{"role": "user", "content": "hello"}]},
                "target_index": 0,
            }
            (base / "x.jsonl").write_text(json.dumps(r) + "\n")
            with (base / "x.csv").open("w") as f:
                w = csv.DictWriter(f, fieldnames=list(r))
                w.writeheader()
                w.writerow({**r, "raw": json.dumps(r["raw"])})
            self.assertEqual(
                load_inference(base / "x.jsonl"), load_inference(base / "x.csv")
            )


class DecisionCommandsTests(unittest.TestCase):
    def test_confirm_cannot_invalidate_a_started_fit(self):
        from types import SimpleNamespace
        from workflow import confirm

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "attempts").mkdir()
            dump(out / "attempts/tfidf.json", {"status": "passed"})
            with self.assertRaisesRegex(ValueError, "before training"):
                confirm(SimpleNamespace(out=out), {})

    def test_interactive_review_revisits_uncertain_answers(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from workflow import review
        from state import read

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            dump(
                out / "review.json",
                {
                    "items": [
                        {
                            "id": "one",
                            "text": "Example",
                            "original_label": 0,
                            "human_decision": "uncertain",
                        }
                    ]
                },
            )
            args = SimpleNamespace(out=out, id=None, author="Tester")
            with (
                patch("workflow.sys.stdin.isatty", return_value=True),
                patch("builtins.input", return_value="a"),
                patch("builtins.print"),
            ):
                review(args, {"run_kind": "fixture"})
            self.assertEqual(
                read(out / "review.json")["items"][0]["human_decision"], "agree"
            )

    def test_markdown_report_requires_complete_unique_headings(self):
        from workflow import report_sections
        from reporting import SECTIONS

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for name in (
                "freeze.json",
                "evaluation/metrics.json",
                "verification/complete.json",
            ):
                dump(out / name, {})
            path = out / "writeup.md"
            text = "\n".join(
                "## " + name.title() + "\nA measured conclusion.\n" for name in SECTIONS
            )
            path.write_text(text)
            result = report_sections(path, out, {"run_id": "test"}, SECTIONS)
            self.assertEqual(set(result["sections"]), set(SECTIONS))
            self.assertEqual(result["sections"]["target"]["author_kind"], "provided")
            for invalid in (
                text + "## Target\nDuplicate",
                text.replace("## Errors", "## Other"),
            ):
                path.write_text(invalid)
                with self.assertRaises(ValueError):
                    report_sections(path, out, {"run_id": "test"}, SECTIONS)


class FreezePolicyTests(unittest.TestCase):
    def test_user_can_choose_either_experiment(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from state import read, sha
        from experiments import freeze

        for available, selected in (
            (("tfidf", "deberta"), "tfidf"),
            (("tfidf", "deberta"), "deberta"),
            (("deberta",), "deberta"),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                run = {
                    "run_id": "fixture",
                    "run_kind": "fixture",
                    "protocol_hash": "protocol",
                    "source_hashes": {},
                    "config_hash": "config",
                }
                for experiment in available:
                    directory = out / "experiments" / experiment
                    (directory / "source").mkdir(parents=True)
                    dump(directory / "experiment.json", {"experiment": experiment})
                    dump(
                        directory / "validation.json",
                        {"recall": 0.5, "threshold": 0.7},
                    )
                dump(
                    out / "decisions.json",
                    {
                        **approval(),
                        "experiment": selected,
                        "rationale": "Synthetic selection based on user judgment.",
                        "experiment_hashes": {
                            k: sha(out / "experiments" / k / "experiment.json")
                            for k in available
                        },
                    },
                )
                with (
                    patch("experiments.state", return_value=(run, {})),
                    patch("experiments.gates", return_value={}),
                    patch("experiments.experiment_check"),
                    patch("experiments.event"),
                ):
                    freeze(SimpleNamespace(out=out, experiment=selected), {})
                self.assertEqual(read(out / "freeze.json")["experiment_id"], selected)


if __name__ == "__main__":
    unittest.main()
