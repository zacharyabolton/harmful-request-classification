"""Check the synthetic CPU workflow and rejected actions."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from state import read, dump, jsonl, sha, utc, lines
from reporting import SECTIONS


def fixture_rows(tag="A"):
    result = []
    for split, n in (("train", 32), ("validation", 16), ("test", 16)):
        for i in range(n):
            label = i % 2
            unique = f"{tag}{split}{i}"
            words = (
                (
                    "garden peaceful ordinary"
                    if not label
                    else "attack harmful dangerous"
                )
                if tag == "A"
                else (
                    "friendly flowers benign"
                    if not label
                    else "malicious violence unsafe"
                )
            )
            result.append(
                {
                    "id": unique,
                    "label": label,
                    "source": "fixture",
                    "revision": "synthetic-v1",
                    "group_id": unique,
                    "parent_id": unique,
                    "split": split,
                    "label_provenance": "synthetic fixture; no quality evidence",
                    "raw": {
                        "text": words
                        + " "
                        + " ".join(unique + str(j) for j in range(8))
                    },
                }
            )
    return result


def approval():
    return {
        "fixture_only": True,
        "author_kind": "fixture",
        "author": "synthetic verification",
        "timestamp": utc(),
        "input_reference": "synthetic gate fixture; not human input",
    }


def approve(run):
    review = read(run / "review.json")
    for item in review["items"]:
        item.update(approval(), human_decision="agree", resolution="retain")
    dump(run / "review.json", review)
    contract = read(run / "contract.json")
    contract.update(approval(), confirmed=True)
    dump(run / "contract.json", contract)


def select(run):
    dump(
        run / "decisions.json",
        {
            **approval(),
            "experiment": "tfidf",
            "rationale": "TF-IDF is the single successful synthetic experiment.",
            "experiment_hashes": {
                "tfidf": sha(run / "experiments/tfidf/experiment.json")
            },
        },
    )


def write_report(run):
    evidence = {
        name: sha(run / name)
        for name in (
            "freeze.json",
            "evaluation/metrics.json",
            "verification/complete.json",
        )
    }
    m = read(run / "evaluation/metrics.json")["overall"]
    prose = {
        "target": "Exercise the workflow with synthetic binary labels, not measure real model quality.",
        "data": "Use 32 training, 16 validation and 16 heldout rows in separate groups.",
        "method": "Fit word TF-IDF and logistic regression, then choose a validation operating point.",
        "selection": "Choose TF-IDF as the single successful experiment in this fixture run.",
        "results": "Synthetic confusion counts: "
        + json.dumps(m["confusion"])
        + ". These are mechanics checks.",
        "errors": "Sample errors by threshold margin and seeded selection, preserving unique IDs.",
        "limitations": "Synthetic results say nothing about performance on real requests.",
        "reproduction": "Reload the saved predictor offline and compare IDs, scores and decisions.",
    }
    (run / "writeup.md").write_text(
        "\n\n".join("## " + name + "\n\n" + prose[name] for name in SECTIONS) + "\n"
    )
    (run / "reports").mkdir(exist_ok=True)
    dump(
        run / "reports/sections.json",
        {
            "run_id": read(run / "run.json")["run_id"],
            "sections": {
                key: {
                    "status": "complete",
                    "author_kind": "agent",
                    "text": prose[key],
                    "evidence": evidence,
                }
                for key in SECTIONS
            },
        },
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    results = []

    def cli(args, good=True, opt=False):
        import subprocess

        cmd = (
            [sys.executable]
            + (["-O"] if opt or sys.flags.optimize else [])
            + [str(ROOT / "src/run.py"), *map(str, args)]
        )
        t = time.monotonic()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        log = out / f"command-{len(results):03d}.log"
        log.write_text(proc.stdout + proc.stderr)
        results.append(
            {
                "command": cmd,
                "exit": proc.returncode,
                "expected_success": good,
                "seconds": time.monotonic() - t,
                "log": log.name,
            }
        )
        dump(out / "checks.json", results)
        if (proc.returncode == 0) != good:
            raise RuntimeError("unexpected command result: " + str(log))

    cfg = read(ROOT / "tests/fixtures/config.json")
    cfg["fixture_only"] = True
    dump(out / "config.json", cfg)
    runs = []
    for tag in ("A", "B"):
        source = out / f"fixture-{tag}.jsonl"
        jsonl(source, fixture_rows(tag))
        prepared = out / f"prepared-{tag}"
        run = out / f"fixture-run-{tag}"
        runs.append(run)
        cli(
            [
                "prepare",
                "--config",
                out / "config.json",
                "--input",
                source,
                "--output",
                prepared,
            ]
        )
        cli(
            [
                "start",
                "--config",
                out / "config.json",
                "--input",
                prepared,
                "--out",
                run,
                "--kind",
                "fixture",
            ]
        )
        cli(
            [
                "start",
                "--config",
                out / "config.json",
                "--input",
                prepared,
                "--out",
                run,
                "--kind",
                "fixture",
            ],
            False,
        )
        cli(["train", "--out", run, "--experiment", "tfidf"], False)
        cli(
            [
                "confirm",
                "--out",
                run,
                "--author",
                "Synthetic tester",
                "--target",
                "Classify synthetic fixture labels",
            ]
        )
        cli(["review", "--out", run])
        for item in read(run / "review.json")["items"]:
            cli(
                [
                    "review",
                    "--out",
                    run,
                    "--id",
                    item["id"],
                    "--decision",
                    "agree",
                    "--author",
                    "Synthetic tester",
                ]
            )
        cli(["train", "--out", run, "--experiment", "tfidf"], False)
        cli(
            [
                "plan",
                "--out",
                run,
                "--experiment",
                "tfidf",
                "--hypothesis",
                "TF-IDF should separate the synthetic fixture.",
                "--stop-rule",
                "Stop after one fit; synthetic verification only.",
            ]
        )
        cli(
            [
                "approve",
                "--out",
                run,
                "--experiment",
                "tfidf",
                "--author",
                "Synthetic tester",
                "--reason",
                "Accept defaults for synthetic lifecycle verification.",
            ]
        )
        cli(["train", "--out", run, "--experiment", "tfidf"])
        if (run / "freeze.json").exists():
            raise RuntimeError("training froze implicitly")
        cli(["freeze", "--out", run, "--experiment", "tfidf"], False)
        cli(["status", "--out", run])
        cli(
            [
                "select",
                "--out",
                run,
                "--experiment",
                "tfidf",
                "--author",
                "Synthetic tester",
                "--reason",
                "TF-IDF is sufficient for this synthetic fixture.",
            ]
        )
        cli(["freeze", "--out", run, "--experiment", "tfidf"])
        cli(["train", "--out", run, "--experiment", "tfidf"], False, True)
        cli(["evaluate", "--out", run])
        cli(["evaluate", "--out", run], False)
        cli(["verify", "--out", run])
        write_report(run)
        cli(["report", "--out", run, "--input", run / "writeup.md"])
    # Run-owned artifact swaps must fail even under optimized Python.
    for name in (
        "experiments/tfidf/model.joblib",
        "review.json",
        "reports/sections.json",
        "evaluation/predictions.jsonl",
        "source/src/models.py",
    ):
        target = runs[0] / name
        original = target.read_bytes()
        target.write_bytes(
            (runs[1] / name).read_bytes()
            if name != "source/src/models.py"
            else original + b"\n# changed\n"
        )
        try:
            cli(["report", "--out", runs[0]], False, True)
            if (runs[0] / "reports/complete.json").exists():
                raise RuntimeError("stale success after failed report")
        finally:
            target.write_bytes(original)
    cli(["report", "--out", runs[0]])
    if time.monotonic() - started > 300:
        raise TimeoutError("local fixture suite cap")
    dump(
        out / "complete.json",
        {
            "status": "passed",
            "fixture_only": True,
            "seconds": time.monotonic() - started,
            "checks": results,
            "runs": [str(r) for r in runs],
        },
    )
    print("Synthetic CPU workflow passed")


if __name__ == "__main__":
    main()
