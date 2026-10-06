"""Validate labeled inputs and render inference-time features."""

import csv
import hashlib
import json
import re
import os
from collections import Counter
from pathlib import Path


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    os.replace(temp, path)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def normalized(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def model_text(record, max_chars=12000):
    """Only explicit inference-time allowlist; raw record remains unchanged.

    target_index ends the visible conversation. Unknown fields remain in raw storage
    but not features. Non-text parts are marked unsupported, not semantically decoded.
    """
    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    try:
        json.dumps(record, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("record must be finite JSON") from exc
    raw = record.get("raw")
    if not isinstance(raw, dict):
        raise ValueError("raw must be an object")
    if type(max_chars) is not int or max_chars <= 0:
        raise ValueError("positive max_chars required")
    if "messages" in raw:
        messages = raw["messages"]
        target = record.get("target_index")
        if (
            not isinstance(messages, list)
            or not isinstance(target, int)
            or isinstance(target, bool)
        ):
            raise ValueError("messages require explicit integer target_index")
        if not 0 <= target < len(messages):
            raise ValueError("invalid target_index")
        for m in messages:
            if not isinstance(m, dict) or not isinstance(m.get("role"), str):
                raise ValueError("message must have a role")
            content = m.get("content")
            if content is not None and not isinstance(content, (str, list)):
                raise ValueError("content must be string, text-part list, or null")
            if isinstance(content, list) and any(
                not isinstance(p, dict) for p in content
            ):
                raise ValueError("content parts must be objects")
        visible_messages = []
        for message in messages[: target + 1]:
            visible_message = {
                k: v
                for k, v in message.items()
                if k in {"role", "content", "name", "tool_calls", "tool_call_id"}
            }
            if isinstance(message.get("content"), list):
                visible_message["content"] = [
                    {k: v for k, v in part.items() if k in {"type", "text"}}
                    if part.get("type") in ("text", "input_text", "output_text")
                    else {"type": part.get("type"), "unsupported_nontext": True}
                    for part in message["content"]
                ]
            if "tool_calls" in message:
                if not isinstance(message["tool_calls"], list):
                    raise ValueError("tool_calls must be a list")
                calls = []
                for call in message["tool_calls"]:
                    if not isinstance(call, dict):
                        raise ValueError("tool_call must be an object")
                    visible_call = {
                        k: v for k, v in call.items() if k in {"id", "type"}
                    }
                    if "function" in call:
                        if not isinstance(call["function"], dict):
                            raise ValueError("tool function must be an object")
                        visible_call["function"] = {
                            k: v
                            for k, v in call["function"].items()
                            if k in {"name", "arguments"}
                        }
                    calls.append(visible_call)
                visible_message["tool_calls"] = calls
            visible_messages.append(visible_message)
        visible = {"messages": visible_messages, "target_index": target}
        for key in ("policy", "context", "target_action"):
            if key in raw:
                visible[key] = raw[key]
        text = json.dumps(visible, ensure_ascii=False, sort_keys=True)
        if len(text) > max_chars:
            raise ValueError(
                "structured message exceeds max_chars; explicit task adaptation required"
            )
        excluded = len(messages) - target - 1
    else:
        if not isinstance(raw.get("text"), str) or not raw["text"].strip():
            raise ValueError("nonempty text required")
        # Policy and context must exist at inference time.
        text = raw["text"]
        for key in ("policy", "context", "target_action"):
            if key in raw:
                text += (
                    "\n<"
                    + key
                    + ">\n"
                    + json.dumps(raw[key], ensure_ascii=False, sort_keys=True)
                )
        excluded = 0
    return text[:max_chars], {
        "original_chars": len(text),
        "model_chars": min(len(text), max_chars),
        "truncated": len(text) > max_chars,
        "future_messages_excluded": excluded,
        "raw_fields_not_features": sorted(
            set(raw) - {"text", "messages", "policy", "context", "target_action"}
        ),
    }


def validate(records, max_chars=12000):
    if not records:
        raise ValueError("empty labeled dataset")
    ids = set()
    for r in records:
        if not isinstance(r, dict):
            raise ValueError("labeled row must be an object")
        for k in ("id", "source", "revision", "group_id", "label_provenance"):
            if not isinstance(r.get(k), str) or not r[k].strip():
                raise ValueError(f"missing/nonstring {k}")
        if r["id"] in ids:
            raise ValueError("duplicate stable ID")
        ids.add(r["id"])
        if type(r.get("label")) is not int or r["label"] not in (0, 1):
            raise ValueError(
                "label must be 0 or 1; missing labels never become class 0"
            )
        if r.get("split") not in (None, "", "train", "validation", "test"):
            raise ValueError("invalid split")
        model_text(r, max_chars)
    supplied = {bool(r.get("split")) for r in records}
    if len(supplied) > 1:
        raise ValueError("partially supplied splits not allowed")


def load_local(path, max_rows=450, max_chars=12000):
    """Canonical JSONL or CSV. CSV raw column is a JSON object; label is 0/1."""
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    elif path.suffix == ".csv":
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        for r in rows:
            r["raw"] = json.loads(r["raw"])
            r["label"] = int(r["label"])
            if r.get("target_index"):
                r["target_index"] = int(r["target_index"])
    else:
        raise ValueError("use .csv or .jsonl")
    if len(rows) > max_rows:
        raise ValueError("row cap exceeded; explicitly prepare a bounded dataset")
    validate(rows, max_chars)
    return rows


def overlap_pairs(rows, threshold):
    """Bounded all-pairs audit, without fitting a vocabulary on held-out text."""
    texts = [normalized(model_text(r)[0]) for r in rows]
    tokens = [set(x.split()) for x in texts]
    pairs = []
    for i in range(len(rows)):
        for j in range(i):
            union = tokens[i] | tokens[j]
            score = len(tokens[i] & tokens[j]) / len(union) if union else 1.0
            exact = texts[i] == texts[j]
            if exact or score >= threshold:
                pairs.append(
                    {
                        "a": rows[j]["id"],
                        "b": rows[i]["id"],
                        "exact": exact,
                        "jaccard": score,
                        "label_agrees": rows[j]["label"] == rows[i]["label"],
                    }
                )
    return pairs


def split_records(rows, seed, threshold):
    from sklearn.model_selection import StratifiedGroupKFold

    pairs = overlap_pairs(rows, threshold)
    # Explicit partitions are immutable. Diagnose overlap; caller must not silently fix it.
    if all(r.get("split") for r in rows):
        return rows, audit(rows, pairs)
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def join(a, b):
        parents[find(a)] = find(b)

    seen = {}
    by_id = {r["id"]: i for i, r in enumerate(rows)}
    for i, r in enumerate(rows):
        for kind, val in [("group", r["group_id"]), ("parent", r.get("parent_id"))]:
            if val:
                key = (r["source"], kind, val)
                if key in seen:
                    join(i, seen[key])
                seen[key] = i
    for p in pairs:
        join(by_id[p["a"]], by_id[p["b"]])
    groups = [find(i) for i in range(len(rows))]
    if len(set(groups)) < 5:
        raise ValueError(
            "too few independent groups for five folds; revise protocol before fitting"
        )
    for i, r in enumerate(rows):
        r["original_group_id"] = r["group_id"]
        r["group_id"] = "component:" + str(groups[i])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (_, idx) in enumerate(
        splitter.split([""] * len(rows), [r["label"] for r in rows], groups)
    ):
        for i in idx:
            rows[i]["split"] = (
                "test" if fold == 0 else "validation" if fold == 1 else "train"
            )
    for split in ("train", "validation", "test"):
        if {r["label"] for r in rows if r["split"] == split} != {0, 1}:
            raise ValueError(
                f"{split} needs both classes; revise protocol before fitting"
            )
    result = audit(rows, pairs)
    if result["cross_split_pairs"] or result["cross_split_groups"]:
        raise ValueError("cross-split contamination")
    return rows, result


def audit(rows, pairs):
    by_id = {r["id"]: r for r in rows}
    groups = {}
    for r in rows:
        for key in (r["group_id"], r.get("parent_id")):
            if key:
                groups.setdefault((r["source"], key), set()).add(r["split"])
    return {
        "method": "normalized exact + all-pairs token-set Jaccard; no learned embeddings",
        "pairs_checked": len(rows) * (len(rows) - 1) // 2,
        "exact_pairs": [p for p in pairs if p["exact"]],
        "near_pairs": [p for p in pairs if not p["exact"]],
        "cross_split_pairs": [
            p for p in pairs if by_id[p["a"]]["split"] != by_id[p["b"]]["split"]
        ],
        "cross_split_groups": [str(g) for g, s in groups.items() if len(s) > 1],
        "residual_risk": "category families + lexical components are proxies, not verified seed genealogy; semantic variants below threshold can remain",
    }


def manifest(rows, max_chars):
    import numpy as np

    out = {}
    for split in ("train", "validation", "test"):
        subset = [r for r in rows if r["split"] == split]
        lengths = [model_text(r, max_chars)[1]["original_chars"] for r in subset]
        out[split] = {
            "n": len(subset),
            "labels": dict(Counter(r["label"] for r in subset)),
            "categories": dict(Counter(r.get("category", "unknown") for r in subset)),
            "sources": dict(Counter(r["source"] for r in subset)),
            "source_types": dict(
                Counter(r.get("source_type", "unavailable") for r in subset)
            ),
            "groups": len({r["group_id"] for r in subset}),
            "length_chars_min_median_p95_max": [
                float(x) for x in np.quantile(lengths, [0, 0.5, 0.95, 1])
            ]
            if lengths
            else None,
            "missing_parent_ids": sum(not r.get("parent_id") for r in subset),
            "missing_policy": sum("policy" not in r["raw"] for r in subset),
            "truncated": sum(model_text(r, max_chars)[1]["truncated"] for r in subset),
        }
    return out


def load_inference(path, max_rows=6000, max_chars=12000):
    """Labels and provenance are optional and never validated as features."""
    path = Path(path)
    with path.open() as handle:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in handle if line.strip()]
        elif path.suffix == ".csv":
            rows = list(csv.DictReader(handle))
            for r in rows:
                r["raw"] = json.loads(r["raw"])
                if r.get("target_index"):
                    r["target_index"] = int(r["target_index"])
        else:
            raise ValueError("use .csv or .jsonl")
    if not rows or len(rows) > max_rows:
        raise ValueError("empty input or row cap")
    ids = set()
    for r in rows:
        if (
            not isinstance(r, dict)
            or not isinstance(r.get("id"), str)
            or not r["id"].strip()
            or r["id"] in ids
        ):
            raise ValueError("unique nonempty string IDs required")
        ids.add(r["id"])
        model_text(r, max_chars)
    return rows
