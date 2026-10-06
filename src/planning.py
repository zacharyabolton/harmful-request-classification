"""Bind a human-approved hypothesis to the exact code, data and fit settings."""

import json
import shutil

from data import dump
from state import read, bound, sha, human, utc, validate_config
from session import state, gates, source_hashes, experiment_name, rows
from settings import effective_settings, encoder_settings, planned_steps


def fit_settings(args, cfg):
    if not getattr(args, "config", None):
        return cfg
    settings = validate_config(read(args.config))
    tunable = {
        "seed",
        "max_chars",
        "max_features",
        "C",
        "max_iter",
        "solver",
        "encoder_device",
        "encoder_revision",
        "model_options",
    }
    if any(
        settings.get(k) != cfg.get(k)
        for k in set(settings) | set(cfg)
        if k not in tunable
    ):
        raise ValueError(
            "experiment config must keep the run's data and objective settings"
        )
    return settings


def proposal(args, run, cfg):
    settings = fit_settings(args, cfg)
    kind = getattr(args, "model", None) or args.experiment
    from models import custom_model

    if kind not in ("tfidf", "deberta"):
        custom_model(kind)
    if settings.get("task") and kind == "deberta":
        from collection import check_encoder

        check_encoder(settings["task"]["collection"], settings["encoder_revision"])
    result = {
        "run_id": run["run_id"],
        "experiment": experiment_name(args.experiment),
        "model": kind,
        "config": settings,
        "effective_settings": effective_settings(kind, settings),
        "protocol": run["protocol_hash"],
        "gates": gates(args.out, run),
        "source_hashes": source_hashes(),
        "objective": {"name": settings["objective"], "max_fpr": settings["max_fpr"]},
        "controls_hash": run.get("controls_hash"),
        "lineage_hash": run.get("lineage_hash"),
    }
    if kind == "deberta":
        result["planned_optimizer_steps"] = planned_steps(
            len(rows(args.out, "train")), encoder_settings(settings)
        )
    return result


def unstarted(out, name):
    if (out / "freeze.json").exists():
        raise ValueError("run is frozen")
    if any((out / "attempts").glob(name + "-*.json")):
        raise ValueError("experiment has an attempt; use a new name")


def plan(args, cfg=None):
    run, cfg = state(args.out)
    unstarted(args.out, args.experiment)
    if len(args.hypothesis.strip()) < 10 or len(args.stop_rule.strip()) < 10:
        raise ValueError("plan needs a hypothesis and stopping rule")
    record = {
        "proposal": proposal(args, run, cfg),
        "hypothesis": args.hypothesis,
        "stopping_rule": args.stop_rule,
        "created": utc(),
        "approval": None,
    }
    dump(args.out / "plans" / (args.experiment + ".json"), record)
    view = {
        **record,
        "proposal": {
            k: v for k, v in record["proposal"].items() if k != "source_hashes"
        },
    }
    print(json.dumps(view, indent=2))
    print(
        "Review these settings, then approve --experiment NAME --author NAME --reason REASON."
    )


def approve_plan(args, cfg=None):
    from types import SimpleNamespace
    from workflow import author_record

    run, cfg = state(args.out)
    unstarted(args.out, args.experiment)
    path = args.out / "plans" / (args.experiment + ".json")
    record = read(path)
    proposed = record["proposal"]
    # Recompute using the settings stored in the proposal, not an external file.
    probe = SimpleNamespace(
        out=args.out, experiment=args.experiment, model=proposed["model"], config=None
    )
    if proposal(probe, run, proposed["config"]) != proposed:
        raise ValueError("plan is stale; regenerate before approving")
    if not args.reason or len(args.reason.strip()) < 10:
        raise ValueError(
            "approval needs a rationale or explicit acceptance of defaults"
        )
    record["approval"] = {
        **author_record(args, run),
        "reason": args.reason,
        "proposal_hash": bound({k: v for k, v in record.items() if k != "approval"}),
    }
    dump(path, record)
    print("Experiment plan approved. Train with the exact planned settings.")


def check_plan(args, run, cfg):
    if run.get("workflow_version", 1) < 2:
        return None
    path = args.out / "plans" / (args.experiment + ".json")
    if not path.exists():
        raise ValueError("plan and approve this experiment before training")
    record = read(path)
    approval = record.get("approval")
    if not approval:
        raise ValueError("experiment plan needs human approval")
    human(approval, run["run_kind"])
    if approval.get("proposal_hash") != bound(
        {k: v for k, v in record.items() if k != "approval"}
    ):
        raise ValueError("approved plan changed")
    if record["proposal"] != proposal(args, run, cfg):
        raise ValueError(
            "code, settings or decisions changed; regenerate and approve the plan"
        )
    return sha(path)


def save_plan(out, name, work):
    path = out / "plans" / (name + ".json")
    shutil.copy2(path, work / "plan.json")
