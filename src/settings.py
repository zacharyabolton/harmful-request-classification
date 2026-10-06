"""Explicit built-in model settings and reproducible training budgets."""

import math


ENCODER_DEFAULTS = {
    "epochs": 1,
    "batch_size": 8,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "max_length": "auto",
    "max_steps": None,
    "max_stage_seconds": 1800,
    "pilot_steps": 20,
    "early_stopping": None,
}


def encoder_settings(cfg):
    supplied = cfg.get("model_options", {})
    unknown = set(supplied) - set(ENCODER_DEFAULTS)
    if unknown:
        raise ValueError("unknown DeBERTa options: " + ", ".join(sorted(unknown)))
    result = {**ENCODER_DEFAULTS, **supplied}
    for key in ("epochs", "batch_size", "max_stage_seconds", "pilot_steps"):
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(key + " must be a positive integer")
    if result["max_stage_seconds"] < 30:
        raise ValueError("max_stage_seconds must allow at least 30 seconds")
    steps = result["max_steps"]
    if steps is not None and (type(steps) is not int or steps <= 0):
        raise ValueError("max_steps must be null or a positive integer")
    length = result["max_length"]
    if length != "auto" and (type(length) is not int or not 1 <= length <= 512):
        raise ValueError("max_length must be auto or an integer from 1 to 512")
    for key in ("learning_rate", "weight_decay"):
        value = result[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (key == "learning_rate" and value == 0)
        ):
            raise ValueError(key + " must be finite and nonnegative (learning rate >0)")
    stopping = result["early_stopping"]
    if stopping is not None:
        if not isinstance(stopping, dict) or set(stopping) != {"patience", "min_delta"}:
            raise ValueError("early_stopping needs exactly patience and min_delta")
        if type(stopping["patience"]) is not int or stopping["patience"] <= 0:
            raise ValueError("early-stopping patience must be a positive integer")
        delta = stopping["min_delta"]
        if (
            isinstance(delta, bool)
            or not isinstance(delta, (int, float))
            or not math.isfinite(delta)
            or not 0 <= delta <= 1
        ):
            raise ValueError(
                "early-stopping min_delta must be finite and within [0, 1]"
            )
        if cfg.get("objective") != "recall_at_fpr":
            raise ValueError("early stopping requires the recall_at_fpr objective")
        if result["max_steps"] is not None:
            raise ValueError(
                "early stopping uses complete epochs; max_steps must be null"
            )
    return result


def planned_steps(n, options):
    total = math.ceil(n / options["batch_size"]) * options["epochs"]
    return min(total, options["max_steps"]) if options["max_steps"] else total


def effective_settings(kind, cfg):
    common = {"seed": cfg["seed"], "max_chars": cfg["max_chars"]}
    if kind == "deberta":
        settings = {
            **common,
            **encoder_settings(cfg),
            "checkpoint": "microsoft/deberta-v3-xsmall",
            "revision": cfg["encoder_revision"],
            "device": cfg["encoder_device"],
            "initialization": "pretrained checkpoint; fresh optimizer (not resume)",
            "length_policy": "auto: 512 if >5% of train+validation exceed 256 tokens; else 256",
        }
        if settings["early_stopping"] is not None:
            settings["early_stopping_policy"] = {
                "metric": f"validation recall at empirical FPR <={cfg['max_fpr']:.1%}",
                "evaluation_frequency": "after each complete epoch",
                "threshold": "reselected on validation each epoch",
                "patience_reset": "recall exceeds previous best recall by more than min_delta",
                "checkpoint_selection": "highest recall, then lower FPR, then earlier epoch",
                "return_model": "best validation checkpoint, not last epoch",
                "step_budget": "maximum; early stopping may use fewer steps",
            }
        return settings
    if kind == "tfidf":
        return {
            **common,
            "ngram_range": [1, 2],
            **{k: cfg[k] for k in ("C", "max_features", "max_iter", "solver")},
            "device": "cpu",
        }
    return {**common, "model_options": cfg.get("model_options", {})}
