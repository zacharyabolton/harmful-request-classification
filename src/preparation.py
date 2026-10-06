"""Task-local preparation, partition auditing and label-review revisions."""

import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from data import dump, model_text, normalized, validate, manifest
from state import bound, sha, jsonl, read, review_template, tree, validate_config


def components(rows):
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, r in enumerate(rows):
        for key in (
            ("seed", r["source"], r["group_id"]),
            ("parent", r["source"], r.get("parent_id") or r["group_id"]),
            (
                "text",
                normalized(r["raw"].get("text", model_text(r, max_chars=10**8)[0])),
            ),
        ):
            if key in seen:
                parents[find(i)] = find(seen[key])
            else:
                seen[key] = i
    by_id = {row["id"]: i for i, row in enumerate(rows)}
    for i, row in enumerate(rows):
        parent = by_id.get(row.get("parent_id"))
        if parent is not None:
            parents[find(i)] = find(parent)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[find(i)].append(r)
    return {min(r["id"] for r in group): group for group in groups.values()}


def supports(rows, fixture, cfg=None):
    result = manifest(rows, cfg["max_chars"] if cfg else 12000)
    for split, s in result.items():
        labs = s["labels"]
        if cfg and cfg.get("task"):
            minimum = cfg["task"]["minimums"][split]
            if (
                s["n"] < minimum["rows"]
                or s["groups"] < minimum["groups"]
                or any(labs.get(label, 0) < minimum["per_class"] for label in (0, 1))
            ):
                raise ValueError("insufficient declared support: " + split)
            continue
        minimum = 5 if fixture else (250 if split == "train" else 100)
        if (
            labs.get(0, 0) < (minimum if fixture or split == "train" else 500)
            or labs.get(1, 0) < minimum
        ):
            raise ValueError("insufficient label support: " + split)
        if s["groups"] < (2 if fixture else 100):
            raise ValueError("insufficient seed groups: " + split)
        if split == "train" and s["n"] < (10 if fixture else 1000):
            raise ValueError("insufficient training rows")
    return result


def overlap(rows, cap=2_000_000):
    """Check exact duplicates and a bounded sample of similar pairs."""
    rows = sorted(rows, key=lambda r: r["id"])
    tokens = [set(normalized(model_text(r, max_chars=10**8)[0]).split()) for r in rows]
    freq = Counter(t for ts in tokens for t in ts)
    index = defaultdict(list)
    pairs_to_check = set()
    exact = {}
    conflicts = []
    checked = 0
    for i, (r, ts) in enumerate(zip(rows, tokens)):
        key = normalized(model_text(r, max_chars=10**8)[0])
        if key in exact and rows[exact[key]]["split"] != r["split"]:
            conflicts.append((exact[key], i, 1.0))
        exact[key] = i
        for token in sorted(ts, key=lambda t: (freq[t], t))[:8]:
            for j in index[token]:
                if rows[j]["split"] != r["split"] and len(pairs_to_check) < cap:
                    pairs_to_check.add((j, i))
            # bounded index gives predictable runtime even with boilerplate-only inputs
            if len(index[token]) < 256:
                index[token].append(i)
    for j, i in sorted(pairs_to_check):
        a, b = tokens[j], tokens[i]
        if min(len(a), len(b)) < 0.8 * max(len(a), len(b)):
            continue
        checked += 1
        union = len(a | b)
        similarity = len(a & b) / union if union else 1
        if similarity >= 0.8:
            conflicts.append((j, i, similarity))
    pairs = [
        {
            "a": rows[j]["id"],
            "b": rows[i]["id"],
            "jaccard": s,
            "splits": [rows[j]["split"], rows[i]["split"]],
        }
        for j, i, s in sorted(set(conflicts))
    ]
    return {
        "method": "8 rarest-token index, at most 256 postings/token; stable ID order",
        "candidate_pairs": len(pairs_to_check),
        "comparisons": checked,
        "cap": cap,
        "exact_coverage": "all input rows",
        "near_coverage": "nonexhaustive",
        "seed": 42,
        "cross_split_pairs": pairs,
    }


def prepare(input_path, output, cfg):
    from data import load_local

    begin = time.monotonic()
    output = Path(output)
    if output.exists():
        raise ValueError("prepared destination exists")
    input_path = Path(input_path)
    validate_config(cfg)
    if cfg.get("task"):
        rows = load_local(input_path, cfg["max_rows"], cfg["max_chars"])
        from collection import check_rows

        check_rows(rows, cfg["task"]["collection"])
        source = {"local_sha256": sha(input_path), "source": "local"}
    else:
        from wildjailbreak import load_source

        rows, source = load_source(input_path, cfg, begin)
    validate(rows, cfg["max_chars"])
    fixture = cfg["fixture_only"]
    if fixture and any(r["source"] != "fixture" for r in rows):
        raise ValueError("fixture exemption requires fixture source")
    supplied = all(r.get("split") for r in rows)
    if cfg.get("task") and not supplied:
        raise ValueError(
            "supply task-specific train/validation/test partitions before preparation"
        )
    groups = components(rows)
    allocations = {}
    pool = defaultdict(list)
    for key, group in sorted(groups.items()):
        if supplied:
            splits = {r["split"] for r in group}
            if len(splits) != 1:
                raise ValueError("supplied partition conflicts; unchanged")
            split = next(iter(splits))
        else:
            bucket = int(bound([42, key])[:16], 16) % 6
            split = "test" if bucket == 0 else "validation" if bucket == 1 else "train"
        allocations[key] = split
        for r in group:
            r["split"] = split
            r["component_id"] = bound(key)
            pool[split].append(r)
    excluded = set()

    def select(split):
        available = sorted(
            (r for r in pool[split] if r["component_id"] not in excluded),
            key=lambda r: (bound([42, r["id"]]), r["id"]),
        )
        if supplied:
            return available
        cap = (16 if fixture else 2000) if split == "train" else (8 if fixture else 500)
        return [
            r
            for label in (0, 1)
            for r in [x for x in available if x["label"] == label][:cap]
        ]

    selected = [r for s in ("train", "validation", "test") for r in select(s)]
    fixed_test = [r["id"] for r in selected if r["split"] == "test"]
    passes = []
    for attempt in range(2):
        audit = overlap(selected)
        passes.append(audit)
        if not audit["cross_split_pairs"]:
            break
        if supplied:
            raise ValueError("supplied partition overlap; unchanged")
        byid = {r["id"]: r for r in selected}
        for pair in audit["cross_split_pairs"]:
            for key in ("a", "b"):
                r = byid[pair[key]]
                if r["split"] != "test":
                    excluded.add(r["component_id"])
        if attempt == 0:
            selected = [r for s in ("train", "validation", "test") for r in select(s)]
    if passes[-1]["cross_split_pairs"]:
        raise ValueError("cross-split near overlap after bounded repair")
    if fixed_test != [r["id"] for r in selected if r["split"] == "test"]:
        raise RuntimeError("heldout allocation changed")
    counts = supports(selected, fixture, cfg)
    output.mkdir(parents=True)
    for split in ("train", "validation", "test"):
        jsonl(
            output / (split + ".jsonl"),
            sorted([r for r in selected if r["split"] == split], key=lambda r: r["id"]),
        )
    if cfg.get("task"):
        (output / "NOTICE.txt").write_text(cfg["task"]["data_permissions"] + "\n")
    else:
        from wildjailbreak import export_source

        export_source(input_path, output, source, selected, fixture)
    dump(
        output / "audit.json",
        {
            "passes": passes,
            "excluded_development_components": sorted(excluded),
            "known_parent_and_exact_components": len(groups),
            "official_eval_used": None if cfg.get("task") else False,
        },
    )
    protocol = {
        "source": source,
        "split_origin": "supplied" if supplied else "custom_grouped_training_release",
        "schema_validation": "passed",
        "splits": counts,
        "fixture_only": fixture,
        "config_hash": bound(cfg),
        "files": tree(output),
        "seconds": time.monotonic() - begin,
    }
    if supplied:
        protocol["supplied_partitions_preserved"] = True
    dump(output / "manifest.json", protocol)
    dump(output / "review.json", review_template(selected, bound(protocol), cfg))
    if time.monotonic() - begin > 600:
        raise TimeoutError("preparation cap")
    return output / "manifest.json"


def revise_training(prepared, review_path, output, cfg, kind):
    """Apply explicit human corrections to training only, then regenerate pending review.

    This is a new immutable preparation; existing runs and all eval rows stay intact.
    Invoke before fitting. Excluded groups are never silently refilled.
    """
    from state import check_review, lines

    prepared = Path(prepared)
    output = Path(output)
    if output.exists():
        raise ValueError("revised preparation destination exists")
    manifest = read(prepared / "manifest.json")
    if manifest["config_hash"] != bound(cfg):
        raise ValueError("configuration mismatch")
    if cfg["fixture_only"] and kind != "fixture":
        raise ValueError("fixture exemption in real revision")
    for name, h in manifest["files"].items():
        if sha(prepared / name) != h:
            raise ValueError("prepared integrity mismatch")
    training = lines(prepared / "train.jsonl")
    review = read(review_path)
    check_review(review, training, bound(manifest), kind, allow_changes=True, cfg=cfg)
    byid = {r["id"]: r for r in training}
    changes = []
    excluded = set()
    for item in review["items"]:
        r = byid[item["id"]]
        if item["resolution"] == "exclude-training-group":
            excluded.add(r.get("component_id", r["group_id"]))
        if item["resolution"] == "correct":
            r = dict(r)
            r["original_label"] = r.get("original_label", r["label"])
            r["label"] = item["corrected_label"]
            r["original_label_provenance"] = r.get(
                "original_label_provenance", r["label_provenance"]
            )
            r["label_provenance"] = "human correction; see training-changes.json"
            byid[r["id"]] = r
        if item["resolution"] != "retain":
            changes.append(item)
    if not changes:
        raise ValueError("no training corrections/exclusions")
    effective = sorted(
        [
            r
            for r in byid.values()
            if r.get("component_id", r["group_id"]) not in excluded
        ],
        key=lambda r: r["id"],
    )
    all_rows = (
        effective
        + lines(prepared / "validation.jsonl")
        + lines(prepared / "test.jsonl")
    )
    if cfg.get("task"):
        from collection import check_rows

        check_rows(all_rows, cfg["task"]["collection"])
    counts = supports(all_rows, cfg["fixture_only"], cfg)
    shutil.copytree(prepared, output)
    jsonl(output / "train.jsonl", effective)
    dump(
        output / "training-changes.json",
        {
            "parent_protocol": bound(manifest),
            "review": review,
            "changes": changes,
            "excluded_components": sorted(excluded),
            "refilled": False,
        },
    )
    new = {
        **manifest,
        "splits": counts,
        "parent_protocol": bound(manifest),
        "training_revision": True,
    }
    files = tree(output)
    files.pop("manifest.json")
    files.pop("review.json")
    new["files"] = files
    dump(output / "manifest.json", new)
    dump(output / "review.json", review_template(all_rows, bound(new), cfg))
    return output / "manifest.json"
