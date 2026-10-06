"""One bounded pinned training-release adapter; official eval is never requested."""

import csv
import shutil
import time
from collections import Counter
from pathlib import Path
from data import dump, normalized
from state import bound, sha, jsonl, read

REPO = "allenai/wildjailbreak"
REVISION = "5ddc12a7894f842b0619b8e1c7ee496b198af009"
LABELS = {
    "vanilla_benign": 0,
    "adversarial_benign": 0,
    "vanilla_harmful": 1,
    "adversarial_harmful": 1,
}


def acquire(output):
    from huggingface_hub import HfApi, hf_hub_download

    begin = time.monotonic()
    output = Path(output) / REVISION
    if output.exists():
        m = read(output / "source.json")
        for name, h in m["files"].items():
            if sha(output / name) != h:
                raise ValueError("cached source checksum mismatch")
        return output / "source.json"
    info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
    sizes = {x.rfilename: x.size for x in info.siblings}
    if sizes.get("train/train.tsv") != 530920110:
        raise ValueError("unexpected pinned source size")
    output.mkdir(parents=True)
    files = {}
    for name in ("train/train.tsv", "README.md"):
        if sizes.get(name, 0) > 1024**3:
            raise ValueError("download cap")
        p = hf_hub_download(
            REPO, name, repo_type="dataset", revision=REVISION, local_dir=output
        )
        files[name] = sha(p)
        if time.monotonic() - begin > 600:
            raise TimeoutError("acquisition cap")
    notice = "WildJailbreak, Allen Institute for AI (2024). Dataset: https://huggingface.co/datasets/allenai/wildjailbreak\nODC-BY 1.0: https://opendatacommons.org/licenses/by/1-0/\nResponsible use: https://allenai.org/responsible-use\nCustom training-release subsets; not official evaluation.\n"
    (output / "NOTICE.txt").write_text(notice)
    files["NOTICE.txt"] = sha(output / "NOTICE.txt")
    dump(
        output / "source.json",
        {
            "schema_version": 1,
            "source": REPO,
            "revision": REVISION,
            "release": "train/train.tsv",
            "files": files,
            "bytes": sum((output / x).stat().st_size for x in files),
            "seconds": time.monotonic() - begin,
            "label_map": LABELS,
            "license": "ODC-BY-1.0",
            "official_eval_used": False,
        },
    )
    return output / "source.json"


def convert(record, index):
    kind = record.get("data_type")
    if kind not in LABELS:
        raise ValueError("unknown WildJailbreak data_type")
    seed = record.get("vanilla")
    if not isinstance(seed, str) or not normalized(seed):
        raise ValueError("missing vanilla ancestry")
    text = record.get("adversarial" if kind.startswith("adversarial") else "vanilla")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("missing feature text")
    parent = bound([REPO, normalized(seed)])
    return {
        "id": f"{REPO}:{REVISION}:train:{index}",
        "label": LABELS[kind],
        "source": REPO,
        "revision": REVISION,
        "group_id": parent,
        "parent_id": parent,
        "label_provenance": "authors:data_type:" + kind,
        "source_type": kind,
        "raw": {"text": text},
        "audit": {"source_row": index, "original_label": LABELS[kind]},
    }


def load_source(input_path, cfg, begin):
    from data import load_local

    if input_path.suffix == ".json" and read(input_path).get("source") == REPO:
        source = read(input_path)
        if cfg["fixture_only"]:
            raise ValueError("fixture exemption forbidden for real source")
        for name, h in source["files"].items():
            if sha(input_path.parent / name) != h:
                raise ValueError("source changed")
        csv.field_size_limit(10**7)
        rows = []
        rejected = Counter()
        with (input_path.parent / "train/train.tsv").open() as f:
            for i, r in enumerate(csv.DictReader(f, delimiter="\t")):
                if i >= 300000:
                    raise ValueError("source row cap")
                if (
                    r.get("data_type") in LABELS
                    and not (
                        r.get(
                            "adversarial"
                            if r["data_type"].startswith("adversarial")
                            else "vanilla"
                        )
                        or ""
                    ).strip()
                ):
                    rejected["empty_feature_text"] += 1
                    continue
                row = convert(r, i)
                rows.append(row)
                if time.monotonic() - begin > 600:
                    raise TimeoutError("preparation cap")
        source["parsed_rows"] = len(rows) + sum(rejected.values())
        source["rejected_rows"] = dict(rejected)
        source["eligible_rows"] = len(rows)
    else:
        rows = load_local(input_path, cfg["max_rows"])
        source = {"local_sha256": sha(input_path), "source": "local"}
    return rows, source


def export_source(input_path, output, source, selected, fixture):
    if source["source"] == REPO:
        selected_indices = {r["audit"]["source_row"] for r in selected}
        with (input_path.parent / "train/train.tsv").open() as f:
            original_records = [
                {"row": i, "record": r}
                for i, r in enumerate(csv.DictReader(f, delimiter="\t"))
                if i in selected_indices
            ]
        jsonl(output / "source-records.jsonl", original_records)
        shutil.copy2(input_path.parent / "NOTICE.txt", output / "NOTICE.txt")
    else:
        (output / "NOTICE.txt").write_text(
            "Synthetic fixtures; no real-data quality evidence.\n"
            if fixture
            else "Local source: record data permissions and attribution.\n"
        )


# Shared data preparation entry points.
from preparation import (
    components as components,
    supports as supports,
    prepare as prepare,
    revise_training as revise_training,
)
