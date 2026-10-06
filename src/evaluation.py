"""Heldout measurements and saved-model checks."""

from task import validation_minimum

import os
import sys
import time
import numpy as np
from data import dump, load_inference
from state import read, sha, tree, lines, jsonl, envelope, timed
from session import ROOT, rows, check_freeze, event
from models import load, fit, score
from measurement import (
    metrics,
    prediction_rows,
    latency,
    failure_samples,
    threshold_sweep,
)


def evaluation_check(out):
    f = check_freeze(out)
    e = read(out / "evaluation/complete.json")
    run = read(out / "run.json")
    if e["run_id"] != run["run_id"] or e["inputs"]["freeze"] != sha(
        out / "freeze.json"
    ):
        raise ValueError("evaluation mismatch")
    actual = tree(out / "evaluation")
    actual.pop("complete.json", None)
    if actual != e["files"]:
        raise ValueError("evaluation changed")
    return f


def evaluate(a, cfg):
    begin = time.monotonic()
    f = check_freeze(a.out)
    run = read(a.out / "run.json")
    cfg = read(a.out / "experiments" / f["experiment_id"] / "config.json")["config"]
    dest = a.out / "evaluation"
    if (dest / "complete.json").exists():
        raise ValueError("evaluation already completed")
    dest.mkdir(exist_ok=True)
    checkpoint = dest / "predictions.jsonl"
    binding = dest / "binding.json"
    inputs = {"freeze": sha(a.out / "freeze.json")}
    if binding.exists() and read(binding) != inputs:
        raise ValueError("interrupted evaluation has different freeze")
    dump(binding, inputs)
    t = time.monotonic()
    model, config = load(
        a.out / "experiments" / f["experiment_id"],
        cfg["encoder_device"],
    )
    cold = time.monotonic() - t
    test = rows(a.out, "test")
    if checkpoint.exists():
        if not (dest / "checkpoint.json").exists() or read(
            dest / "checkpoint.json"
        ) != {
            "predictions_sha256": sha(checkpoint),
            "freeze_sha256": sha(a.out / "freeze.json"),
        }:
            raise ValueError("interrupted predictions lack valid binding")
        predictions = lines(checkpoint)
    else:
        predictions = prediction_rows(model, test, config, f["threshold"])
        jsonl(checkpoint, predictions)
        dump(
            dest / "checkpoint.json",
            {
                "predictions_sha256": sha(checkpoint),
                "freeze_sha256": sha(a.out / "freeze.json"),
            },
        )
    if [p["id"] for p in predictions] != [r["id"] for r in test]:
        raise ValueError("checkpoint IDs mismatch")
    scores = [p["score"] for p in predictions]
    result = {
        "threshold": f["threshold"],
        "overall": metrics([r["label"] for r in test], scores, f["threshold"]),
        "AP_definition": "stepwise average precision; not trapezoidal PR area",
        "slices": {},
    }
    if run.get("workflow_version", 1) >= 2:
        from runtime import lineage

        exposure = lineage(a.out, run)["heldout_exposure"]
        result["evaluation_context"] = {
            "heldout_exposure": exposure,
            "interpretation": (
                "Exploratory evaluation on a previously evaluated split; not fresh independent evidence."
                if exposure == "previously-evaluated"
                else "Exposure not established; interpret cautiously."
                if exposure == "unknown"
                else "Declared unseen before this run; supplied provenance and grouping limits still apply."
            ),
        }
    for field in ("source", "source_type", "group_id", "truncation"):
        keys = [
            (
                "unavailable"
                if p["preprocessing"]["truncated"] is None
                else str(
                    p["preprocessing"]["truncated"]
                    or p["preprocessing"].get("token_truncated", False)
                )
            )
            if field == "truncation"
            else str(r.get(field, "unavailable"))
            for r, p in zip(test, predictions)
        ]
        result["slices"][field] = {
            key: metrics(
                [r["label"] for r, k in zip(test, keys) if k == key],
                [s for s, k in zip(scores, keys) if k == key],
                f["threshold"],
            )
            for key in sorted(set(keys))
        }
    dump(dest / "metrics.json", result)
    dump(dest / "failures.json", failure_samples(test, predictions, f["threshold"]))
    dump(
        dest / "latency.json",
        {"cold_load_seconds": cold, **latency(model, test, config, f["threshold"])},
    )
    dump(
        dest / "complete.json",
        envelope(run, inputs, files=tree(dest), seconds=time.monotonic() - begin),
    )
    event(a.out, "evaluate", begin)


def predict(a, cfg=None):
    model, config = load(a.model, getattr(a, "device", "cpu"))
    threshold = (
        read(a.model / "threshold.json")["threshold"]
        if (a.model / "threshold.json").exists()
        else read(a.model / "validation.json")["threshold"]
    )
    rs = load_inference(
        a.input, config["config"]["max_rows"], config["config"]["max_chars"]
    )
    if a.output.exists():
        raise ValueError("prediction output exists")
    jsonl(a.output, prediction_rows(model, rs, config, threshold))


def compare(expected, actual, tol):
    if (
        len(expected) != len(actual)
        or [p["id"] for p in expected] != [p["id"] for p in actual]
        or [p["prediction"] for p in expected] != [p["prediction"] for p in actual]
    ):
        raise ValueError("prediction IDs/decisions mismatch")
    if not np.allclose(
        [p["score"] for p in expected], [p["score"] for p in actual], rtol=0, atol=tol
    ):
        raise ValueError("prediction score mismatch")


def verify(a, cfg):
    begin = time.monotonic()
    dest = a.out / "verification"
    dest.mkdir(exist_ok=True)
    success = dest / "complete.json"
    success.unlink(missing_ok=True)
    f = evaluation_check(a.out)
    run = read(a.out / "run.json")
    from runtime import controls

    timeout_seconds = controls(a.out, run)["verification_timeout_seconds"]
    if (dest / "reload.log").exists():
        history = a.out / "verification-attempts"
        history.mkdir(exist_ok=True)
        import shutil

        shutil.copy2(dest / "reload.log", history / (str(time.time_ns()) + ".log"))
    jsonl(
        dest / "unlabeled.jsonl",
        [
            {k: r[k] for k in ("id", "raw", "target_index") if k in r}
            for r in rows(a.out, "test")
        ],
    )
    output = dest / "predictions.jsonl"
    output.unlink(missing_ok=True)
    command = timed(
        [
            sys.executable,
            ROOT / "src/run.py",
            "predict",
            "--model",
            a.out / "experiments" / f["experiment_id"],
            "--input",
            dest / "unlabeled.jsonl",
            "--output",
            output,
        ],
        dest / "reload.log",
        timeout_seconds,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    compare(
        lines(a.out / "evaluation/predictions.jsonl"),
        lines(output),
        0
        if read(a.out / "experiments" / f["experiment_id"] / "config.json").get(
            "model", f["experiment_id"]
        )
        == "tfidf"
        else 1e-5,
    )
    dump(
        success,
        envelope(
            run,
            {
                "evaluation": sha(a.out / "evaluation/complete.json"),
                "freeze": sha(a.out / "freeze.json"),
            },
            command=command,
            timeout_seconds=timeout_seconds,
            files=tree(dest),
            seconds=time.monotonic() - begin,
        ),
    )
    event(a.out, "verify", begin)


def verification_check(out):
    evaluation_check(out)
    v = read(out / "verification/complete.json")
    if v["run_id"] != read(out / "run.json")["run_id"] or v["inputs"] != {
        "evaluation": sha(out / "evaluation/complete.json"),
        "freeze": sha(out / "freeze.json"),
    }:
        raise ValueError("stale verification")
    actual = tree(out / "verification")
    actual.pop("complete.json", None)
    if actual != v["files"]:
        raise ValueError("verification changed")


def reproduce(a, cfg):
    f = evaluation_check(a.out)
    config = read(a.out / "experiments" / f["experiment_id"] / "config.json")
    cfg = config["config"]
    if config.get("model", config["experiment"]) != "tfidf":
        raise ValueError("reproduce currently supports TF-IDF only")
    model = fit(rows(a.out, "train"), cfg)
    val = rows(a.out, "validation")
    threshold, _ = threshold_sweep(
        [r["label"] for r in val],
        score(model, val, cfg),
        cfg["objective"],
        validation_minimum(cfg),
        cfg["max_fpr"],
    )
    if threshold != f["threshold"]:
        raise ValueError("reproduced threshold mismatch")
    actual = prediction_rows(model, rows(a.out, "test"), {"config": cfg}, threshold)
    compare(lines(a.out / "evaluation/predictions.jsonl"), actual, 1e-12)
    dump(
        a.out / "reproduction.json",
        {
            "replication_of": read(a.out / "run.json")["run_id"],
            "protocol": read(a.out / "run.json")["protocol_hash"],
            "same_protocol_not_independent_evaluation": True,
        },
    )
