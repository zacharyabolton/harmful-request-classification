"""Collection gates use synthetic rows only."""

import copy
import unittest
from test_task import task_config
from collection import check_rows, validate_collection
from smoke_test import fixture_rows


class CollectionTests(unittest.TestCase):
    def record(self):
        return task_config()["task"]["collection"]

    def test_unknown_is_valid_but_missing_choices_and_evidence_are_not(self):
        record = self.record()
        record["model_checks"][0]["status"] = "unknown"
        validate_collection(record)
        for field in (
            "extra_sources",
            "synthetic_augmentation",
            "sources",
            "model_checks",
        ):
            bad = copy.deepcopy(record)
            del bad[field]
            with self.assertRaises(ValueError):
                validate_collection(bad)
        record["model_checks"][0]["evidence"] = ""
        with self.assertRaises(ValueError):
            validate_collection(record)

    def test_sources_must_be_registered_and_allowed(self):
        record = self.record()
        rows = fixture_rows()
        check_rows(rows, record)
        rows[0]["revision"] = "unregistered"
        with self.assertRaisesRegex(ValueError, "missing from collection"):
            check_rows(rows, record)
        record["sources"][0]["role"] = "external"
        with self.assertRaisesRegex(ValueError, "contradicts"):
            validate_collection(record)

    def test_generation_cannot_use_evaluation_parents_or_cross_groups(self):
        record = self.record()
        record["synthetic_augmentation"]["use"] = True
        record["sources"].append(
            dict(
                source="generated",
                revision="v1",
                origin="Local recipe",
                permission="Synthetic test",
                labels="Reviewed",
                role="generated",
                generator="Test generator",
                generator_revision="v1",
                recipe="Local fixture",
                review="Fixture labels checked",
                overlap_check="Unknown; fixture only",
            )
        )
        rows = fixture_rows()
        parent = next(r for r in rows if r["split"] == "train")
        child = dict(
            parent,
            id="generated-child",
            source="generated",
            revision="v1",
            parent_id=parent["id"],
        )
        check_rows(rows + [child], record)
        variants = [
            dict(child, split="validation"),
            dict(child, group_id="different"),
            dict(child, parent_id="missing"),
            dict(child, parent_id=child["id"]),
        ]
        heldout = next(r for r in rows if r["split"] == "test")
        variants.append(
            dict(child, parent_id=heldout["id"], group_id=heldout["group_id"])
        )
        for bad in variants:
            with (
                self.subTest(row=bad),
                self.assertRaisesRegex(ValueError, "original training parent"),
            ):
                check_rows(rows + [bad], record)

    def test_preparation_and_review_keep_generated_families_together(self):
        from pathlib import Path
        import tempfile
        from preparation import prepare, revise_training
        from state import read, dump, jsonl, lines, sha
        from smoke_test import approval

        cfg = task_config()
        cfg["fixture_only"] = False
        record = cfg["task"]["collection"]
        record["synthetic_augmentation"]["use"] = True
        record["sources"].append(
            dict(
                source="generated",
                revision="v1",
                origin="Synthetic regression",
                permission="Test fixture",
                labels="Fixture codes",
                role="generated",
                generator="fixture",
                generator_revision="v1",
                recipe="Local fixture",
                review="Fixture only",
                overlap_check="No external model used",
            )
        )
        rows = fixture_rows()
        children = [
            dict(
                row,
                id=row["id"] + "-child",
                parent_id=row["id"],
                source="generated",
                revision="v1",
                raw={"text": "Distinct generated variant " + row["id"]},
            )
            for row in rows
            if row["split"] == "train"
        ]
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            jsonl(base / "rows.jsonl", rows + children)
            prepare(base / "rows.jsonl", base / "prepared", cfg)
            training = {r["id"]: r for r in lines(base / "prepared/train.jsonl")}
            for child in children:
                self.assertEqual(
                    training[child["id"]]["component_id"],
                    training[child["parent_id"]]["component_id"],
                )
            self.assertIsNone(read(base / "prepared/audit.json")["official_eval_used"])
            review = read(base / "prepared/review.json")
            for item in review["items"]:
                item.update(approval(), human_decision="agree", resolution="retain")
            item = review["items"][0]
            selected = training[item["id"]]
            parent_id = (
                selected["parent_id"]
                if selected["source"] == "generated"
                else selected["id"]
            )
            item.update(
                human_decision="disagree",
                resolution="exclude-training-group",
                rationale="Exclude this fixture family",
            )
            dump(base / "review.json", review)
            revise_training(
                base / "prepared",
                base / "review.json",
                base / "revised",
                cfg,
                "fixture",
            )
            remaining = lines(base / "revised/train.jsonl")
            self.assertNotIn(parent_id, {r["id"] for r in remaining})
            self.assertNotIn(parent_id + "-child", {r["id"] for r in remaining})
            check_rows(remaining, record)
            for split in ("validation", "test"):
                self.assertEqual(
                    sha(base / f"prepared/{split}.jsonl"),
                    sha(base / f"revised/{split}.jsonl"),
                )

    def test_proposal_requires_exact_encoder_provenance(self):
        from planning import proposal
        from types import SimpleNamespace
        from unittest.mock import patch

        cfg = task_config()
        args = SimpleNamespace(
            config=None, model="deberta", experiment="encoder", out=None
        )
        run = {"run_id": "fixture-run", "protocol_hash": "fixture-protocol"}
        with (
            patch("planning.gates", return_value={}),
            patch("planning.source_hashes", return_value={}),
            patch("planning.rows", return_value=[{}]),
        ):
            with self.assertRaisesRegex(ValueError, "exact DeBERTa"):
                proposal(args, run, cfg)
            check = dict(
                model="microsoft/deberta-v3-xsmall",
                revision="wrong",
                status="unknown",
                evidence="No full corpus available",
                limits="Cannot establish absence",
            )
            cfg["task"]["collection"]["model_checks"].append(check)
            with self.assertRaisesRegex(ValueError, "exact DeBERTa"):
                proposal(args, run, cfg)
            check["revision"] = cfg["encoder_revision"]
            self.assertEqual(proposal(args, run, cfg)["model"], "deberta")
            check["status"] = "not_applicable"
            with self.assertRaisesRegex(ValueError, "exact DeBERTa"):
                proposal(args, run, cfg)
