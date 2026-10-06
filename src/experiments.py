"""Train named experiments and freeze the user's selection."""

from task import validation_minimum, label_map

import shutil
import time
import joblib
import numpy as np
from data import dump
from state import read, sha, bound, tree, envelope, human, utc, jsonl
from session import (
    state,
    rows,
    gates,
    event,
    experiment_check,
    experiment_names,
    experiment_name,
    snapshot,
    source_hashes,
    save_environment,
)
from models import fit, predict_scores, train_encoder, load, custom_model
from planning import fit_settings, check_plan, save_plan
from settings import encoder_settings
from measurement import threshold_sweep, metrics


def train(a, cfg):
    begin = time.monotonic()
    run, cfg = state(a.out)
    if (a.out / "freeze.json").exists():
        raise ValueError("fit forbidden after freeze")
    gate = gates(a.out, run)
    experiment_name(a.experiment)
    kind = getattr(a, "model", None) or a.experiment
    if kind not in ("tfidf", "deberta"):
        custom_model(kind)
    plan_hash = check_plan(a, run, cfg)
    cfg = fit_settings(a, cfg)
    dest = a.out / "experiments" / a.experiment
    if dest.exists():
        raise ValueError("successful experiment cannot be overwritten")
    work = a.out / (".experiment-" + a.experiment)
    if work.exists():
        raise ValueError("previous attempt retained; use a new experiment name")
    attempts = a.out / "attempts"
    attempts.mkdir(exist_ok=True)
    attempt = attempts / (a.experiment + "-" + str(time.time_ns()) + ".json")
    inputs = {"protocol": run["protocol_hash"], "gates": gate}
    if plan_hash:
        inputs["plan"] = plan_hash
    record = envelope(
        run,
        inputs,
        experiment_id=a.experiment,
        status="running",
        started=utc(),
    )
    dump(attempt, record)
    work.mkdir()
    if plan_hash:
        save_plan(a.out, a.experiment, work)
    hashes = snapshot(work / "source")
    dump(work / "settings.json", cfg)
    save_environment(work)
    record["source_hashes"] = hashes
    record["config"] = cfg
    record["model"] = kind
    record["artifacts"] = work.name
    dump(attempt, record)
    try:
        train_rows = rows(a.out, "train")
        val = rows(a.out, "validation")
        config = {
            "experiment": a.experiment,
            "model": kind,
            "config": cfg,
            "run_id": run["run_id"],
        }
        if kind == "tfidf":
            model = fit(train_rows, cfg)
            joblib.dump(model, work / "model.joblib")
            diagnostics = {
                "finite_weights": True,
                "converged": True,
                "warnings": [],
                "solver": cfg["solver"],
            }
        elif kind == "deberta":
            if not cfg["encoder_revision"]:
                raise ValueError("model revision not pinned")
            model, diagnostics = train_encoder(train_rows, val, cfg, work / "encoder")
        else:
            model, diagnostics = custom_model(kind).train(train_rows, val, cfg, work)
        scores = predict_scores(model, val, config)
        threshold, sweep = threshold_sweep(
            [r["label"] for r in val],
            scores,
            cfg["objective"],
            validation_minimum(cfg),
            cfg["max_fpr"],
        )
        dump(work / "config.json", config)
        dump(work / "sweep.json", sweep)
        dump(work / "diagnostics.json", diagnostics)
        jsonl(
            work / "validation_predictions.jsonl",
            [
                {"id": r["id"], "label": r["label"], "score": float(s)}
                for r, s in zip(val, scores)
            ],
        )
        dump(
            work / "validation.json",
            {
                "threshold": threshold,
                **metrics([r["label"] for r in val], scores, threshold),
            },
        )
        loaded, _ = load(work, cfg["encoder_device"] if kind == "deberta" else "cpu")
        if not np.allclose(
            scores,
            predict_scores(loaded, val, config),
            rtol=0,
            atol=0 if kind == "tfidf" else 1e-5,
        ):
            raise ValueError("save/reload failed")
        if (
            kind == "deberta"
            and time.monotonic() - begin > encoder_settings(cfg)["max_stage_seconds"]
        ):
            raise TimeoutError("complete DeBERTa stage exceeded configured budget")
        if source_hashes() != hashes:
            raise ValueError(
                "code changed during training; retry with a new experiment name"
            )
        evidence = envelope(
            run,
            inputs,
            experiment_id=a.experiment,
            model=kind,
            files=tree(work),
            seconds=time.monotonic() - begin,
        )
        dump(work / "experiment.json", evidence)
        dest.parent.mkdir(exist_ok=True)
        work.rename(dest)
        record.update(
            status="passed",
            artifacts=str(dest.relative_to(a.out)),
            seconds=time.monotonic() - begin,
        )
        dump(attempt, record)
        event(a.out, "train " + a.experiment, begin)
    except Exception as exc:
        record.update(
            status="failed", reason=str(exc), seconds=time.monotonic() - begin
        )
        dump(attempt, record)
        raise


def freeze(a, cfg):
    begin = time.monotonic()
    run, cfg = state(a.out)
    if (a.out / "freeze.json").exists():
        raise ValueError("already frozen")
    gates_now = gates(a.out, run)
    decision = read(a.out / "decisions.json")
    chosen = decision.get("experiment")
    if not chosen or (a.experiment and a.experiment != chosen):
        raise ValueError("select a trained experiment before freezing")
    human(decision, run["run_kind"])
    experiments = {}
    for name in experiment_names(a.out):
        path = a.out / "experiments" / name
        if path.exists():
            experiment_check(a.out, run, name)
            experiments[name] = sha(path / "experiment.json")
    if chosen not in experiments or decision.get("experiment_hashes") != experiments:
        raise ValueError(
            "experiment results changed; review the comparison and select again"
        )
    reason = decision.get("rationale", "")
    if not reason.strip() or any(
        word in reason.upper() for word in ("TODO", "PENDING", "PLACEHOLDER")
    ):
        raise ValueError("selection needs a reason")
    selected = a.out / "experiments" / chosen
    validation = read(selected / "validation.json")
    if (a.out / "source").exists():
        shutil.rmtree(a.out / "source")
    shutil.copytree(selected / "source", a.out / "source")
    run["source_hashes"] = tree(a.out / "source")
    dump(a.out / "run.json", run)
    dump(
        a.out / "freeze.json",
        envelope(
            run,
            {
                "protocol": run["protocol_hash"],
                "gates": gates_now,
                "decision": bound(decision),
                "experiments": experiments,
                "source": bound(run["source_hashes"]),
                "config": run["config_hash"],
                **{
                    key: run[key]
                    for key in ("controls_hash", "lineage_hash")
                    if key in run
                },
            },
            experiment_id=chosen,
            threshold=validation["threshold"],
            model_files=tree(a.out / "experiments" / chosen),
            label_map=label_map(cfg),
            frozen_utc=utc(),
        ),
    )
    event(a.out, "freeze", begin)
