import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import load_local, model_text, split_records, validate
from measurement import metrics, threshold_sweep
from models import fit


def row(i=0, **kw):
    return {
        "id": str(i),
        "label": i % 2,
        "source": "fixture",
        "revision": "v1",
        "group_id": str(i),
        "label_provenance": "fixture",
        "raw": {"text": "ordinary words"},
        **kw,
    }


class SchemaTests(unittest.TestCase):
    def test_boundary_tool_calls_and_parts(self):
        r = row(
            raw={
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "question"}]},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"q":"needle"}',
                                }
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "1", "content": "result"},
                    {"role": "assistant", "content": "FUTURE_RESPONSE"},
                ],
                "policy": "POLICY",
                "context": "CONTEXT",
                "target_action": "inspect",
                "outcome": "SECRET_LABEL",
            },
            target_index=2,
        )
        text, trace = model_text(r)
        for token in (
            "question",
            "needle",
            "result",
            "tool_call_id",
            "POLICY",
            "CONTEXT",
        ):
            self.assertIn(token, text)
        self.assertNotIn("FUTURE_RESPONSE", text)
        self.assertNotIn("SECRET_LABEL", text)
        self.assertEqual(trace["future_messages_excluded"], 1)
        self.assertEqual(r["raw"]["messages"][3]["content"], "FUTURE_RESPONSE")
        with self.assertRaises(ValueError):
            model_text({**r, "target_index": None})

    def test_truncation_metadata_and_labels(self):
        r = row(raw={"text": "a" * 50}, category="METADATA", label=1)
        text, trace = model_text(r, 10)
        self.assertEqual(len(text), 10)
        self.assertTrue(trace["truncated"])
        self.assertEqual(len(r["raw"]["text"]), 50)
        self.assertNotIn("METADATA", text)
        for label in (None, "safe", True, 2):
            with self.assertRaises(ValueError):
                validate([row(label=label)])
        with self.assertRaises(ValueError):
            validate([row(), row()])

    def test_jsonl_csv_equivalence(self):
        r = row()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "data.jsonl").write_text(json.dumps(r) + "\n")
            with (base / "data.csv").open("w") as f:
                w = csv.DictWriter(f, fieldnames=list(r))
                w.writeheader()
                w.writerow({**r, "raw": json.dumps(r["raw"])})
            self.assertEqual(
                load_local(base / "data.jsonl"), load_local(base / "data.csv")
            )

    def test_official_splits_preserved_and_overlap_reported(self):
        rows = [row(0, split="train"), row(1, split="test")]
        result, audit = split_records(rows, 42, 0.8)
        self.assertEqual([r["split"] for r in result], ["train", "test"])
        self.assertEqual(len(audit["cross_split_pairs"]), 1)
        self.assertFalse(audit["cross_split_pairs"][0]["label_agrees"])

    def test_group_integrity_and_training_only_vocabulary(self):
        rows = [row(i, raw={"text": f"word{i} unique{i}"}) for i in range(50)]
        rows[1]["group_id"] = rows[0]["group_id"]
        result, audit = split_records(rows, 42, 0.8)
        self.assertEqual(result[0]["split"], result[1]["split"])
        self.assertFalse(audit["cross_split_groups"])
        cfg = {
            "max_features": 100,
            "C": 1,
            "max_iter": 1000,
            "seed": 42,
            "max_chars": 100,
        }
        model = fit(
            [
                row(0, split="train", raw={"text": "harmless daily question"}),
                row(1, split="train", raw={"text": "harmful dangerous instruction"}),
                row(2, split="test", raw={"text": "heldoutcanary"}),
            ],
            cfg,
        )
        self.assertNotIn("heldoutcanary", model["tfidf"].vocabulary_)


class MetricTests(unittest.TestCase):
    def test_threshold_and_uncertainty(self):
        threshold, _ = threshold_sweep([0, 0, 1, 1], [0.1, 0.4, 0.3, 0.8])
        self.assertEqual(threshold, 0.3)
        m = metrics([0, 0, 1], [0.1, 0.2, 0.9], 0.5)
        self.assertGreater(m["fpr_wilson_95"][1], 0)
        self.assertIsNone(metrics([1, 1], [0.1, 0.9], 0.5)["average_precision"])
        self.assertIsNone(metrics([0, 0], [0.1, 0.2], 0.5)["recall"])


if __name__ == "__main__":
    unittest.main()
