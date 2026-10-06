"""Create runs, record decisions and guard saved inputs."""

import re
import platform
import shutil
import sys
import time
import uuid
from pathlib import Path
from data import dump
from state import (
    read,
    sha,
    bound,
    tree,
    verify_tree,
    human,
    check_review,
    validate_config,
    utc,
    lines,
)
from preparation import revise_training
from task import budget_key

ROOT = Path(__file__).resolve().parents[1]


def code_files():
    patterns = (
        "src/*.py",
        "scripts/*.py",
        "tests/*.py",
        "notices/*",
        "docs/*.md",
        "templates/*.md",
        "templates/*.py",
        "templates/*.json",
        "tests/fixtures/*.json",
    )
    files = [path for pattern in patterns for path in ROOT.glob(pattern)]
    files += [
        ROOT / name
        for name in (
            "README.md",
            "LICENSE",
            "requirements.txt",
            "requirements.encoder.txt",
            "requirements.colab.txt",
            "requirements.dev.txt",
            "pyproject.toml",
            "config.json",
        )
    ]
    return sorted(files)


def source_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for p in code_files() if p.exists()}


def experiment_name(name):
    if not name or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", name):
        raise ValueError(
            "experiment name must use letters, digits, underscores or hyphens"
        )
    return name


def experiment_names(out):
    return sorted(p.name for p in (out / "experiments").glob("*") if p.is_dir())


def snapshot(dest):
    hashes = source_hashes()
    for name in hashes:
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    verify_tree(dest, hashes)
    return hashes


def state(out, enforce_time=True):
    if not (out / "run.json").exists():
        raise ValueError("run.json missing; create a run with start")
    run = read(out / "run.json")
    cfg = validate_config(read(out / "config.json"))
    if bound(cfg) != run["config_hash"]:
        raise ValueError("run configuration changed")
    if (out / "freeze.json").exists():
        verify_tree(out / "source", run["source_hashes"])
    manifest = read(out / "data/manifest.json")
    if bound(manifest) != run["protocol_hash"]:
        raise ValueError("protocol changed")
    for name, h in manifest["files"].items():
        if sha(out / "data" / name) != h:
            raise ValueError("prepared data changed: " + name)
    from runtime import controls, lineage

    policy = controls(out, run)
    lineage(out, run)
    if (
        enforce_time
        and policy["mode"] == "timed"
        and time.time() - run["start_epoch"] > cfg[budget_key(cfg)]
        and not (out / "completed.json").exists()
    ):
        raise TimeoutError("run budget exceeded")
    return run, cfg


def rows(out, split):
    return lines(out / "data" / (split + ".jsonl"))


def development(out):
    return rows(out, "train") + rows(out, "validation")


def gates(out, run):
    contract = read(out / "contract.json")
    if (
        contract.get("confirmed") is not True
        or contract.get("protocol_hash") != run["protocol_hash"]
    ):
        raise ValueError(
            "task not confirmed for this protocol; run confirm before planning or training"
        )
    human(contract, run["run_kind"])
    rh = check_review(
        read(out / "review.json"),
        rows(out, "train"),
        run["protocol_hash"],
        run["run_kind"],
        cfg=read(out / "config.json"),
    )
    return {"contract": bound(contract), "review": rh}


def event(out, name, begin):
    t = read(out / "timing.json")
    t["stages"].append(
        {
            "stage": name,
            "seconds": time.monotonic() - begin,
            "utc": utc(),
            "elapsed_since_start": time.time() - t["start_epoch"],
        }
    )
    dump(out / "timing.json", t)


def save_environment(out):
    import subprocess

    deps = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout
    dump(
        out / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": deps.splitlines(),
            "native_thread_request": 1,
        },
    )


def start(a, cfg):
    begin = time.monotonic()
    validate_config(cfg)
    if a.out.exists():
        raise ValueError("run collision; new directory required")
    manifest = read(a.input / "manifest.json")
    if manifest["config_hash"] != bound(cfg):
        raise ValueError("prepared configuration mismatch")
    if a.kind != "fixture" and (cfg["fixture_only"] or manifest["fixture_only"]):
        raise ValueError("fixture data/config in real run")
    if a.kind == "fixture" and not cfg["fixture_only"]:
        raise ValueError("fixture run requires fixture config")
    if a.kind == "replication" and not a.replication_of:
        raise ValueError("replication_of required")
    for name, h in manifest["files"].items():
        if sha(a.input / name) != h:
            raise ValueError("prepared integrity mismatch")
    from runtime import parent_record, inherit_review

    lineage = parent_record(a, manifest, cfg)
    controls = {
        "mode": getattr(a, "mode", "timed"),
        "verification_timeout_seconds": getattr(a, "verification_timeout", 600),
    }
    if (
        controls["mode"] not in ("timed", "exploratory")
        or controls["verification_timeout_seconds"] <= 0
    ):
        raise ValueError("invalid runtime settings")
    a.out.mkdir(parents=True)
    shutil.copytree(a.input, a.out / "data")
    shutil.copy2(a.input / "review.json", a.out / "review.json")
    run = {
        "schema_version": 1,
        "workflow_version": 2,
        "producer": "classification-experiments",
        "run_id": a.out.name + "-" + uuid.uuid4().hex[:12],
        "run_kind": a.kind,
        "replication_of": a.replication_of,
        "start_epoch": time.time(),
        "protocol_hash": bound(manifest),
        "config_hash": bound(cfg),
        "source_hashes": {},
        "controls_hash": bound(controls),
        "lineage_hash": bound(lineage),
    }
    dump(a.out / "controls.json", controls)
    dump(a.out / "lineage.json", lineage)
    dump(a.out / "run.json", run)
    dump(a.out / "config.json", cfg)
    dump(
        a.out / "contract.json",
        {
            "protocol_hash": run["protocol_hash"],
            "confirmed": False,
            "author_kind": None,
            "target": cfg["task"]["target"]
            if cfg.get("task")
            else "harmful request classification",
            "official_benchmark": None if cfg.get("task") else False,
        },
    )
    dump(
        a.out / "decisions.json",
        {"experiment": None, "rationale": None, "author_kind": None},
    )
    inherit_review(a)
    shutil.copy2(ROOT / "templates/REPORT.md", a.out / "writeup.md")
    dump(a.out / "timing.json", {"start_epoch": run["start_epoch"], "stages": []})
    save_environment(a.out)
    event(a.out, "start", begin)


def resolve_review(a, cfg):
    begin = time.monotonic()
    run, cfg = state(a.out)
    if (
        (a.out / "freeze.json").exists()
        or any((a.out / "experiments").glob("*"))
        or any((a.out / "attempts").glob("*.json"))
    ):
        raise ValueError("training changes must precede all fitting attempts")
    pending = a.out / ".revised-data"
    revise_training(
        a.out / "data", a.out / "review.json", pending, cfg, run["run_kind"]
    )
    history = a.out / "review-history" / str(time.time_ns())
    history.mkdir(parents=True)
    for name in ("review.json", "contract.json", "decisions.json", "run.json"):
        shutil.copy2(a.out / name, history / name)
    (a.out / "data").rename(history / "data")
    pending.rename(a.out / "data")
    run["protocol_hash"] = bound(read(a.out / "data/manifest.json"))
    dump(a.out / "run.json", run)
    shutil.copy2(a.out / "data/review.json", a.out / "review.json")
    dump(
        a.out / "contract.json",
        {
            "protocol_hash": run["protocol_hash"],
            "confirmed": False,
            "author_kind": None,
            "target": cfg["task"]["target"]
            if cfg.get("task")
            else "harmful request classification",
            "official_benchmark": None if cfg.get("task") else False,
        },
    )
    dump(
        a.out / "decisions.json",
        {"experiment": None, "rationale": None, "author_kind": None},
    )
    event(a.out, "resolve review; original start time retained", begin)


def experiment_check(out, run, name):
    p = out / "experiments" / experiment_name(name)
    evidence = read(p / "experiment.json")
    if (
        evidence["run_id"] != run["run_id"]
        or evidence["inputs"]["protocol"] != run["protocol_hash"]
    ):
        raise ValueError("experiment ownership mismatch")
    if evidence["inputs"]["gates"] != gates(out, run):
        raise ValueError("experiment review changed")
    actual = tree(p)
    actual.pop("experiment.json", None)
    if actual != evidence["files"]:
        raise ValueError("experiment changed")
    if (
        "plan" in evidence["inputs"]
        and sha(p / "plan.json") != evidence["inputs"]["plan"]
    ):
        raise ValueError("experiment plan changed")
    return evidence


def check_freeze(out):
    run, cfg = state(out)
    if not (out / "freeze.json").exists():
        raise ValueError("freeze the selected model before evaluation or verification")
    f = read(out / "freeze.json")
    if (
        f["run_id"] != run["run_id"]
        or f["inputs"]["protocol"] != run["protocol_hash"]
        or f["inputs"]["config"] != run["config_hash"]
        or f["inputs"]["source"] != bound(run["source_hashes"])
    ):
        raise ValueError("freeze identity mismatch")
    for key in ("controls_hash", "lineage_hash"):
        if key in run and f["inputs"].get(key) != run[key]:
            raise ValueError("frozen runtime policy or lineage changed")
    if f["inputs"]["gates"] != gates(out, run) or f["inputs"]["decision"] != bound(
        read(out / "decisions.json")
    ):
        raise ValueError("frozen human decisions changed")
    for name, h in f["inputs"]["experiments"].items():
        experiment_check(out, run, name)
        if sha(out / "experiments" / name / "experiment.json") != h:
            raise ValueError("frozen experiment mismatch")
    verify_tree(out / "experiments" / f["experiment_id"], f["model_files"])
    if (
        f["threshold"]
        != read(out / "experiments" / f["experiment_id"] / "validation.json")[
            "threshold"
        ]
    ):
        raise ValueError("threshold changed")
    return f
