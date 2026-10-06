"""Check saved evidence without downloading data or loading models."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"


def read(path: Path):
    return json.loads(path.read_text())


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def close(actual: float, expected: float, name: str):
    require(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), name)


def check_metrics(predictions, labels, threshold, saved, name):
    require(len({r["id"] for r in predictions}) == len(predictions), name + ": duplicate ID")
    require({r["id"] for r in predictions} == set(labels), name + ": row IDs")
    confusion = dict(tp=0, fn=0, fp=0, tn=0)
    for row in predictions:
        label = labels[row["id"]]
        if "label" in row:
            require(row["label"] == label, name + ": label")
        predicted = int(row["score"] >= threshold)
        if "prediction" in row:
            require(row["prediction"] == predicted, name + ": threshold")
        confusion[{(1, 1): "tp", (1, 0): "fn", (0, 1): "fp", (0, 0): "tn"}[label, predicted]] += 1
    require(confusion == saved["confusion"], name + ": confusion")
    tp, fn, fp, tn = (confusion[k] for k in ["tp", "fn", "fp", "tn"])
    require(saved["n"] == len(predictions), name + ": support")
    require(saved["positives"] == tp + fn, name + ": positive support")
    require(saved["negatives"] == tn + fp, name + ": negative support")
    for key, value in {
        "recall": tp / (tp + fn),
        "fpr": fp / (tn + fp),
        "precision": tp / (tp + fp),
        "f1": 2 * tp / (2 * tp + fp + fn),
    }.items():
        close(value, saved[key], name + ": " + key)
    score_groups = {}
    for row in predictions:
        score_groups.setdefault(row["score"], []).append(labels[row["id"]])
    seen = positive_seen = 0
    average_precision = 0.0
    for score in sorted(score_groups, reverse=True):
        group = score_groups[score]
        seen += len(group)
        gained = sum(group)
        positive_seen += gained
        average_precision += gained / (tp + fn) * positive_seen / seen
    close(average_precision, saved["average_precision"], name + ": average precision")
    return confusion


def paired(old, new, labels):
    result = {}
    for label, name in [(0, "benign"), (1, "harmful")]:
        counts = dict(both_correct=0, old_only_correct=0, new_only_correct=0, both_wrong=0)
        for row_id, truth in labels.items():
            if truth != label:
                continue
            key = {
                (True, True): "both_correct",
                (True, False): "old_only_correct",
                (False, True): "new_only_correct",
                (False, False): "both_wrong",
            }[(old[row_id] == truth, new[row_id] == truth)]
            counts[key] += 1
        result[name] = counts
    return result


def main():
    index = read(EVIDENCE / "index.json")
    for item in index["artifacts"]:
        path = EVIDENCE / item["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == item["sha256"], item["path"] + ": hash")

    wj = EVIDENCE / "wildjailbreak"
    split_rows = rows(wj / "split-labels.jsonl")
    require(len({r["id"] for r in split_rows}) == 6000, "Unique split IDs")
    groups = {}
    split_labels = {}
    for split, n, group_count in [("train", 4000, 3899), ("validation", 1000, 976), ("test", 1000, 975)]:
        selected = [r for r in split_rows if r["split"] == split]
        require(len(selected) == n, split + ": rows")
        require(sum(r["label"] for r in selected) == n // 2, split + ": label balance")
        groups[split] = {r["group_id"] for r in selected}
        require(len(groups[split]) == group_count, split + ": groups")
        split_labels[split] = {r["id"]: r["label"] for r in selected}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        require(not groups[a] & groups[b], a + "/" + b + ": group overlap")

    expected = {
        "wj-tfidf": {"validation": (402, 98, 25, 475)},
        "wj-deberta-1ep": {"validation": (438, 62, 24, 476), "test": (436, 64, 18, 482)},
        "wj-deberta-2ep": {"validation": (461, 39, 25, 475)},
        "wj-deberta-6ep": {"validation": (483, 17, 25, 475), "test": (483, 17, 36, 464)},
    }
    predictions_by_model = {}
    for model, splits in expected.items():
        for split, counts in splits.items():
            saved = read(wj / model / (split + ".json"))
            threshold = saved["threshold"]
            predictions = rows(wj / model / (split + "-predictions.jsonl"))
            result = check_metrics(predictions, split_labels[split], threshold, saved.get("overall", saved), model + "/" + split)
            require(tuple(result[k] for k in ["tp", "fn", "fp", "tn"]) == counts, model + ": published counts")
            if split == "validation":
                predictions_by_model[model] = {r["id"]: int(r["score"] >= threshold) for r in predictions}
            else:
                source_types = read(wj / model / "test.json")["slices"]["source_type"]
                for source_type, metrics in source_types.items():
                    labels = {r["id"]: r["label"] for r in split_rows if r["split"] == split and r["source_type"] == source_type}
                    source_counts = dict(tp=0, fn=0, fp=0, tn=0)
                    for row in predictions:
                        if row["id"] in labels:
                            y, pred = labels[row["id"]], int(row["score"] >= threshold)
                            source_counts[{(1, 1): "tp", (1, 0): "fn", (0, 1): "fp", (0, 0): "tn"}[y, pred]] += 1
                    require(source_counts == metrics["confusion"], model + ": source-type counts")

    comparison = read(wj / "paired-validation.json")
    for model, key in [("wj-tfidf", "tfidf"), ("wj-deberta-1ep", "deberta-one-epoch"), ("wj-deberta-2ep", "deberta-two-epoch")]:
        result = paired(predictions_by_model[model], predictions_by_model["wj-deberta-6ep"], split_labels["validation"])
        require(result == comparison["six_epoch_vs_baselines"][key], model + ": paired counts")
    result = paired(predictions_by_model["wj-deberta-1ep"], predictions_by_model["wj-deberta-2ep"], split_labels["validation"])
    require(result == comparison["two_vs_one_epoch"]["counts"], "Two vs one epoch")
    epochs = read(wj / "epochs.json")
    require(len(epochs) == 6 and not any(e["stop"] for e in epochs), "Epoch limit")
    require(epochs[-1]["validation"] == read(wj / "wj-deberta-6ep/validation.json"), "Selected epoch")
    chronology = read(wj / "chronology.json")
    require(chronology["series"][1]["parent"] == "wj-initial", "Parent series")
    require(chronology["series"][1]["heldout_exposure"] == "previously-evaluated", "Test exposure")
    xstest = read(EVIDENCE / "xstest/results.json")
    require(xstest["heldout"]["overall"]["confusion"] == dict(tp=23, fn=2, fp=47, tn=3), "XSTest test counts")
    print("Evidence passed: hashes, splits, six prediction sets, metrics, slices, paired counts and exposure.")


if __name__ == "__main__":
    main()
