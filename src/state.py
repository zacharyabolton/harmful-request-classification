"""Run state, file integrity and recorded decisions."""

import hashlib
import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from data import dump as dump, model_text
from task import validate_task, budget_key

VERSION = "classification-experiments"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bound(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows)
    )
    os.replace(tmp, path)


def lines(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def tree(path):
    return {
        str(p.relative_to(path)): sha(p)
        for p in sorted(Path(path).rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def verify_tree(path, hashes):
    if tree(path) != hashes:
        raise ValueError("artifact tree changed: " + str(path))


def envelope(run, inputs, **values):
    return {
        "schema_version": 1,
        "producer": VERSION,
        "run_id": run["run_id"],
        "inputs": inputs,
        **values,
    }


def timed(cmd, log, timeout, **kwargs):
    begin = time.monotonic()
    try:
        p = subprocess.run(
            [str(x) for x in cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            **kwargs,
        )
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        Path(log).write_text(p.stdout + p.stderr)
        result = {
            "command": [str(x) for x in cmd],
            "exit": p.returncode,
            "seconds": time.monotonic() - begin,
            "log": str(log),
        }
        if p.returncode:
            raise RuntimeError("subprocess failed; see " + str(log))
        return result
    except subprocess.TimeoutExpired:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        Path(log).write_text("deadline exceeded; child killed and waited\n")
        raise


def review_template(rows, protocol, cfg=None):
    count = cfg["task"]["review_per_class"] if cfg and cfg.get("task") else 5
    train = sorted((r for r in rows if r["split"] == "train"), key=lambda r: r["id"])
    rng = random.Random(42)
    sample = [
        r
        for label in (0, 1)
        for r in rng.sample([r for r in train if r["label"] == label], count)
    ]
    return {
        "schema_version": 1,
        "protocol_hash": protocol,
        "training_hash": bound(train),
        "seed": 42,
        "items": [
            {
                "id": r["id"],
                "row_hash": bound(r),
                "text": model_text(r, cfg["max_chars"] if cfg else 12000)[0],
                "original_label": r["label"],
                "human_decision": "pending",
                "resolution": None,
            }
            for r in sample
        ],
    }


def human(record, kind):
    fixture = record.get("fixture_only") is True
    if fixture and kind != "fixture":
        raise ValueError("fixture decision in real run")
    if not fixture and record.get("author_kind") != "human":
        raise ValueError("human authorship required")
    for k in ("author", "timestamp", "input_reference"):
        v = record.get(k)
        if (
            not isinstance(v, str)
            or len(v.strip()) < 3
            or v.strip().lower() in ("yes", "ok", "approved")
            or any(x in v.upper() for x in ("PENDING", "TODO", "PLACEHOLDER"))
        ):
            raise ValueError("missing human attribution: " + k)
    datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))


def check_review(review, rows, protocol, kind, allow_changes=False, cfg=None):
    expected = review_template(rows, protocol, cfg)
    for k in ("protocol_hash", "training_hash", "seed"):
        if review.get(k) != expected[k]:
            raise ValueError("stale review: " + k)
    items = review.get("items", [])
    if [x.get("id") for x in items] != [x["id"] for x in expected["items"]]:
        raise ValueError("review sample changed")
    for item, exp in zip(items, expected["items"]):
        if any(item.get(k) != exp[k] for k in ("row_hash", "text", "original_label")):
            raise ValueError("review content changed")
        decision = item.get("human_decision")
        resolution = item.get("resolution")
        if decision not in ("agree", "disagree"):
            raise ValueError(
                "unresolved human review; finish review before planning or training"
            )
        human(item, kind)
        if resolution not in ("retain", "correct", "exclude-training-group"):
            raise ValueError("unresolved resolution")
        if decision == "agree" and resolution != "retain":
            raise ValueError("agreement must retain")
        if decision == "disagree" and not item.get("rationale", "").strip():
            raise ValueError("disagreement needs rationale")
        # A correction/exclusion changes the review population. Apply before any fit,
        # then prepare a new dataset and review sample; never mutate val/test here.
        if resolution == "correct" and (
            type(item.get("corrected_label")) is not int
            or item["corrected_label"] not in (0, 1)
            or item["corrected_label"] == item["original_label"]
        ):
            raise ValueError("corrected_label must change binary label")
        if resolution != "retain" and not allow_changes:
            raise ValueError(
                "apply training correction/exclusion in new preparation and regenerate review before fitting"
            )
    return bound(review)


def validate_config(c):
    task = "task" in c
    if task:
        validate_task(c["task"])
    keys = {
        "schema_version",
        "seed",
        "max_chars",
        "max_rows",
        "max_features",
        "C",
        "max_iter",
        "solver",
        "objective",
        "max_fpr",
        "device",
        "encoder_device",
        "encoder_revision",
        "fixture_only",
        budget_key(c),
    }
    if task:
        keys.add("task")
    if set(c) - {"model_options"} != keys:
        raise ValueError("unknown/missing configuration keys: " + str(set(c) ^ keys))
    if c["objective"] not in ("recall_at_fpr", "max_f1"):
        raise ValueError("unsupported objective")
    if c["device"] != "cpu" or c["encoder_device"] not in ("cpu", "cuda"):
        raise ValueError("unsupported device")
    if c["solver"] not in ("lbfgs", "liblinear"):
        raise ValueError("unsupported solver")
    for k in (
        "max_chars",
        "max_rows",
        "max_features",
        "max_iter",
        budget_key(c),
    ):
        if type(c[k]) is not int or c[k] <= 0:
            raise ValueError("positive integer " + k + " required")
    if not task and (
        c["run_budget_seconds"] > 7200
        or c["max_rows"] > 6000
        or c["max_fpr"] != 0.05
    ):
        raise ValueError("configuration violates fixed protocol")
    if not (task and c["objective"] == "max_f1" and c["max_fpr"] is None) and (
        isinstance(c["max_fpr"], bool)
        or not isinstance(c["max_fpr"], (int, float))
        or not 0 < c["max_fpr"] < 1
    ):
        raise ValueError("max_fpr must be explicitly specified between zero and one")
    if type(c["seed"]) is not int or c["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (
        not isinstance(c["C"], (int, float))
        or isinstance(c["C"], bool)
        or not 0 < c["C"] < float("inf")
    ):
        raise ValueError("C must be finite and positive")
    if not isinstance(c.get("model_options", {}), dict):
        raise ValueError("model_options must be an object")
    if type(c["fixture_only"]) is not bool:
        raise ValueError("fixture_only must be boolean")
    if c["encoder_revision"] is not None and (
        len(c["encoder_revision"]) != 40
        or any(x not in "0123456789abcdef" for x in c["encoder_revision"])
    ):
        raise ValueError("pin encoder revision to commit")
    return c
