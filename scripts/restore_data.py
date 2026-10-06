"""Restore the recorded splits from the pinned source release."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import dump
from state import bound, jsonl, read, review_template, sha, tree, validate_config
from wildjailbreak import acquire, convert


def restore(source, output, config=ROOT / "config.json"):
    output = Path(output)
    if output.exists():
        raise ValueError("output exists")
    index = read(ROOT / "data/split-index.json")
    metadata = read(source)
    if (metadata.get("source"), metadata.get("revision")) != (
        index["source"], index["revision"]
    ):
        raise ValueError("source revision mismatch")
    tsv = Path(source).parent / "train/train.tsv"
    if sha(tsv) != index["source_sha256"]:
        raise ValueError("source checksum mismatch")
    cfg = validate_config(read(config))
    selected = {item["row"] for split in index["splits"].values() for item in split}
    records = {}
    csv.field_size_limit(10**7)
    with tsv.open() as stream:
        for number, record in enumerate(csv.DictReader(stream, delimiter="\t")):
            if number in selected:
                records[number] = convert(record, number)
    if records.keys() != selected:
        raise ValueError("source rows missing")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".restore-", dir=output.parent))
    try:
        all_rows = []
        for split, items in index["splits"].items():
            rows = [
                {**records[item["row"]], "split": split, "component_id": item["component_id"]}
                for item in items
            ]
            jsonl(stage / (split + ".jsonl"), rows)
            if sha(stage / (split + ".jsonl")) != index["split_sha256"][split]:
                raise ValueError("restored split checksum mismatch: " + split)
            all_rows.extend(rows)
        shutil.copyfile(ROOT / "data/audit.json", stage / "audit.json")
        (stage / "NOTICE.txt").write_text(
            "Contains information from WildJailbreak, Allen Institute for AI (2024).\n"
            "https://huggingface.co/datasets/allenai/wildjailbreak\n"
            "ODC-BY 1.0: https://opendatacommons.org/licenses/by/1-0/\n"
            "Custom splits of the training release. Previously evaluated.\n"
        )
        manifest = {
            "source": {
                "source": index["source"], "revision": index["revision"],
                "release": "train/train.tsv", "sha256": index["source_sha256"],
                "license": "ODC-BY-1.0", "official_eval_used": False,
            },
            "split_origin": "restored_recorded_training_release_splits",
            "heldout_exposure": "previously-evaluated",
            "schema_validation": "passed", "splits": index["support"],
            "fixture_only": False, "config_hash": bound(cfg), "files": tree(stage),
        }
        dump(stage / "manifest.json", manifest)
        dump(stage / "review.json", review_template(all_rows, bound(manifest), cfg))
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {"splits": {s: len(v) for s, v in index["splits"].items()}, "checksums": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="Acquired source.json; otherwise download")
    parser.add_argument("--cache", type=Path, default=ROOT / ".local/cache")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    source = args.source or acquire(args.cache)
    print(json.dumps(restore(source, args.output, args.config)))


if __name__ == "__main__":
    main()
