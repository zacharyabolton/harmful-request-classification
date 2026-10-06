"""Operating points and descriptive measurements, independent of orchestration."""

import math
import random
import time
from collections import Counter
import numpy as np
from sklearn.metrics import average_precision_score
from data import model_text
from models import finite, predict_scores, Encoder


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [max(0, center - half), min(1, center + half)]


def metrics(y, scores, threshold):
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    finite(scores)
    if len(y) != len(scores) or any(
        type(v) not in (int, np.int64, np.int32) or v not in (0, 1) for v in y
    ):
        raise ValueError("invalid metric labels")
    if not math.isfinite(threshold) or ((scores < 0) | (scores > 1)).any():
        raise ValueError("invalid score/threshold")
    p = scores >= threshold
    tp = int(((y == 1) & p).sum())
    fp = int(((y == 0) & p).sum())
    fn = int(((y == 1) & ~p).sum())
    tn = int(((y == 0) & ~p).sum())
    return {
        "n": len(y),
        "positives": tp + fn,
        "negatives": tn + fp,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "average_precision": float(average_precision_score(y, scores))
        if len(set(y)) == 2
        else None,
        "fpr": fp / (tn + fp) if tn + fp else None,
        "fpr_wilson_95": wilson(fp, tn + fp),
        "recall_wilson_95": wilson(tp, tp + fn),
        "one_class_slice": len(set(y)) < 2,
        "empty_slice": len(y) == 0,
        "interval_caveat": "Row Wilson ignores group correlation; validation also ignores threshold selection. Artificial prevalence affects precision/AP.",
    }


def threshold_sweep(y, scores, objective="max_f1", min_negatives=1, max_fpr=0.05):
    if len(y) != len(scores) or set(y) != {0, 1} or y.count(0) < min_negatives:
        raise ValueError("threshold support insufficient")
    thresholds = sorted(
        set([0.0, 1.0, math.nextafter(1.0, math.inf), *[float(s) for s in scores]])
    )
    sweep = [{"threshold": t, **metrics(y, scores, t)} for t in thresholds]
    if objective == "recall_at_fpr":
        chosen = max(
            (x for x in sweep if x["fpr"] <= max_fpr),
            key=lambda x: (x["recall"], -x["fpr"], x["threshold"]),
        )
    elif objective == "max_f1":
        chosen = max(sweep, key=lambda x: (x["f1"] or 0, -x["fpr"], x["threshold"]))
    else:
        raise ValueError("unknown objective")
    return chosen["threshold"], sweep


def failure_samples(rows, predictions, threshold):
    errors = {}
    for kind, label in (("false_positives", 0), ("false_negatives", 1)):
        population = sorted(
            [
                {
                    **p,
                    "group_id": r["group_id"],
                    "category": r.get("category"),
                    "raw": r["raw"],
                    "margin": abs(p["score"] - threshold),
                }
                for r, p in zip(rows, predictions)
                if r["label"] == label and p["prediction"] != label
            ],
            key=lambda x: x["id"],
        )
        if len(population) <= 10:
            chosen = [{**r, "selection_reason": "all"} for r in population]
        else:
            far = sorted(population, key=lambda r: (-r["margin"], r["id"]))[:3]
            used = {r["id"] for r in far}
            near = sorted(
                [r for r in population if r["id"] not in used],
                key=lambda r: (r["margin"], r["id"]),
            )[:3]
            used |= {r["id"] for r in near}
            rest = random.Random(42).sample(
                [r for r in population if r["id"] not in used], 4
            )
            chosen = [
                {**r, "selection_reason": reason}
                for items, reason in (
                    (far, "farthest"),
                    (near, "nearest"),
                    (rest, "seeded remainder"),
                )
                for r in items
            ]

        def coverage(items):
            return {
                field: dict(Counter(str(r.get(field)) for r in items))
                for field in ("category", "group_id")
            }

        errors[kind] = {
            "total": len(population),
            "sample": chosen,
            "seed": 42,
            "population_coverage": coverage(population),
            "sample_coverage": coverage(chosen),
            "omitted_categories": sorted(
                set(str(r.get("category")) for r in population)
                - set(str(r.get("category")) for r in chosen)
            ),
            "interpretation": "Margins are distances from threshold, not confidence; explanations are hypotheses.",
        }
    return errors


def prediction_rows(model, rows, config, threshold):
    scores = predict_scores(model, rows, config)
    result = []
    for r, s in zip(rows, scores):
        kind = config.get("model", config.get("experiment", "tfidf"))
        trace = (
            model_text(r, config["config"]["max_chars"])[1]
            if kind in ("tfidf", "deberta")
            else {"implementation": kind, "truncated": None}
        )
        if isinstance(model, Encoder):
            length = len(
                model.tokenizer(
                    model_text(r, config["config"]["max_chars"])[0], truncation=False
                )["input_ids"]
            )
            trace.update(
                token_count=length,
                token_truncated=length > model.max_length,
                max_tokens=model.max_length,
            )
        result.append(
            {
                "id": r["id"],
                "score": float(s),
                "prediction": int(s >= threshold),
                "preprocessing": trace,
            }
        )
    return result


def latency(model, rows, config, threshold):
    rows = sorted(rows, key=lambda r: r["id"])[:100]
    start = time.monotonic()

    def sync():
        if isinstance(model, Encoder):
            model.sync()

    for _ in range(5):
        prediction_rows(model, rows[:1], config, threshold)
        sync()
    result = {}
    for batch in (1, 8):
        durations = []
        count = 0
        for _ in range(3):
            for i in range(0, len(rows), batch):
                sync()
                t = time.perf_counter()
                prediction_rows(model, rows[i : i + batch], config, threshold)
                sync()
                durations.append(time.perf_counter() - t)
                count += len(rows[i : i + batch])
                if time.monotonic() - start > 120:
                    raise TimeoutError("latency cap")
        result[str(batch)] = {
            "p50_p95_max_ms": (np.quantile(durations, [0.5, 0.95, 1]) * 1000).tolist(),
            "throughput_rows_second": count / sum(durations),
            "repetitions": 3,
            "rows_per_repetition": len(rows),
        }
    return {
        "batches": result,
        "warmups": 5,
        "device": (
            "cpu"
            if config.get("model", config.get("experiment", "tfidf")) == "tfidf"
            else str(getattr(model, "device", "unspecified"))
        ),
        "includes": "render/tokenize/forward/threshold/output construction",
        "excludes": "disk/network/model load",
        "length_chars": [
            model_text(r, config["config"]["max_chars"])[1]["original_chars"]
            for r in rows
        ],
    }
