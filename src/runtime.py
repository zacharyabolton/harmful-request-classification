"""Recorded run budgets and exposure-aware links between development runs."""

import shutil
import time

from state import read, dump, bound, sha, human


def controls(out, run):
    if run.get("workflow_version", 1) < 2:
        return {"mode": "timed", "verification_timeout_seconds": 120}
    value = read(out / "controls.json")
    if bound(value) != run["controls_hash"]:
        raise ValueError("run controls changed without a recorded decision")
    if value.get("mode") not in ("timed", "exploratory"):
        raise ValueError("invalid run mode")
    timeout = value.get("verification_timeout_seconds")
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("verification timeout must be a positive integer")
    return value


def lineage(out, run):
    if run.get("workflow_version", 1) < 2:
        return {"heldout_exposure": "unknown", "baselines": {}}
    value = read(out / "lineage.json")
    if bound(value) != run["lineage_hash"]:
        raise ValueError("run lineage changed")
    return value


def parent_record(args, manifest, cfg=None):
    from session import state, experiment_check, experiment_names, gates

    exposure = getattr(args, "heldout_exposure", "unknown")
    if manifest.get("heldout_exposure") == "previously-evaluated":
        exposure = "previously-evaluated"
    result = {"heldout_exposure": exposure, "baselines": {}}
    parent = getattr(args, "parent", None)
    if not parent:
        if getattr(args, "reuse_review", False):
            raise ValueError("reuse-review requires a parent run")
        return result
    parent_run, parent_cfg = state(parent, enforce_time=False)
    if cfg and cfg.get("task"):
        if not parent_cfg.get("task") or parent_cfg["task"] != cfg["task"]:
            raise ValueError(
                "task parent must belong to the same declared task; undeclared task history cannot be inherited"
            )
    old = read(parent / "data/manifest.json")
    same = all(
        old["files"][name] == manifest["files"][name]
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl")
    )
    if not same:
        raise ValueError(
            "linked comparison requires identical train/validation/test files"
        )
    if parent_run["run_kind"] == "fixture" and args.kind != "fixture":
        raise ValueError("fixture parent cannot supply a real run's decisions")
    if (parent / "evaluation").exists() or lineage(parent, parent_run)[
        "heldout_exposure"
    ] == "previously-evaluated":
        result["heldout_exposure"] = "previously-evaluated"
    result["parent_run_id"] = parent_run["run_id"]
    result["parent_protocol_hash"] = parent_run["protocol_hash"]
    result["comparison_split_hashes"] = {
        k: manifest["files"][k]
        for k in ("train.jsonl", "validation.jsonl", "test.jsonl")
    }
    for name in experiment_names(parent):
        evidence = experiment_check(parent, parent_run, name)
        result["baselines"][name] = {
            "experiment_sha256": sha(parent / "experiments" / name / "experiment.json"),
            "validation": read(parent / "experiments" / name / "validation.json"),
            "seconds": evidence["seconds"],
            "model": evidence.get("model", name),
        }
    if getattr(args, "reuse_review", False):
        if bound(manifest) != parent_run["protocol_hash"]:
            raise ValueError("review reuse requires the identical prepared protocol")
        gates(parent, parent_run)
        result["inherited_decisions"] = {
            name: sha(parent / name) for name in ("contract.json", "review.json")
        }
    return result


def inherit_review(args):
    if getattr(args, "reuse_review", False):
        for name in ("contract.json", "review.json"):
            shutil.copy2(args.parent / name, args.out / name)


def configure(args, cfg=None):
    from session import state
    from workflow import author_record, editable

    run, _ = state(args.out, enforce_time=False)
    editable(args.out)
    if run.get("workflow_version", 1) < 2:
        raise ValueError(
            "legacy run: create a linked run; do not rewrite archived code"
        )
    if not args.reason or len(args.reason.strip()) < 10:
        raise ValueError("runtime change needs a reason")
    value = controls(args.out, run)
    if args.mode is not None:
        value["mode"] = args.mode
    if args.verification_timeout is not None:
        if args.verification_timeout <= 0:
            raise ValueError("verification timeout must be positive")
        value["verification_timeout_seconds"] = args.verification_timeout
    decision = {
        **author_record(args, run),
        "reason": args.reason,
        "actual_elapsed_seconds": time.time() - run["start_epoch"],
    }
    human(decision, run["run_kind"])
    value.setdefault("changes", []).append(decision)
    dump(args.out / "controls.json", value)
    run["controls_hash"] = bound(value)
    dump(args.out / "run.json", run)
    print(
        "Runtime policy updated; original clock retained. Unstarted plans need reapproval."
    )
