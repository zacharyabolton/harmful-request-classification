"""Commands for the decisions between experiment stages."""

import re
import sys
from pathlib import Path

from data import dump
from state import read, sha, utc


def author_record(args, run):
    if not args.author or len(args.author.strip()) < 3:
        raise ValueError("--author must name the person making this decision")
    return {
        "author": args.author,
        "author_kind": "fixture" if run["run_kind"] == "fixture" else "human",
        "fixture_only": run["run_kind"] == "fixture",
        "timestamp": utc(),
        "input_reference": "command-line decision",
    }


def editable(out):
    if (out / "freeze.json").exists():
        raise ValueError("run is frozen; decisions cannot change")


def confirm(args, run):
    editable(args.out)
    if any((args.out / "attempts").glob("*.json")):
        raise ValueError("task confirmation must finish before training begins")
    if not args.target or not args.target.strip():
        raise ValueError("--target should state what the model predicts")
    cfg = read(args.out / "config.json")
    if cfg.get("task") and args.target != cfg["task"]["target"]:
        raise ValueError("confirmation target must match the declared task")
    record = read(args.out / "contract.json")
    record.update(author_record(args, run), confirmed=True, target=args.target)
    dump(args.out / "contract.json", record)
    print("Task confirmed. Review the sampled training labels next.")


def record_review(args, run, worksheet, item):
    if not args.decision:
        raise ValueError("--decision is required when --id is provided")
    resolution = args.resolution or ("retain" if args.decision == "agree" else None)
    if args.decision == "disagree" and (not args.reason or not resolution):
        raise ValueError("disagreement needs --reason and --resolution")
    if args.decision == "agree" and resolution != "retain":
        raise ValueError("agreement retains the label")
    if resolution == "correct" and (
        args.label is None or args.label == item["original_label"]
    ):
        raise ValueError("--label must give a different binary label")
    item.update(
        author_record(args, run), human_decision=args.decision, resolution=resolution
    )
    item["rationale"] = args.reason or ""
    item.pop("corrected_label", None)
    if resolution == "correct":
        item["corrected_label"] = args.label
    dump(args.out / "review.json", worksheet)


def review(args, run):
    worksheet = read(args.out / "review.json")
    if args.id:
        editable(args.out)
        if any((args.out / "attempts").glob("*.json")):
            raise ValueError("review must finish before training begins")
        item = next(
            (item for item in worksheet["items"] if item["id"] == args.id), None
        )
        if item is None:
            raise ValueError("ID is not in the training review sample")
        record_review(args, run, worksheet, item)
        print("Review saved. Run status to see what remains.")
        return

    interactive = args.author is not None
    if interactive:
        editable(args.out)
        if not sys.stdin.isatty():
            raise ValueError(
                "interactive review needs a terminal; use --id and --decision for individual entries"
            )
        if any((args.out / "attempts").glob("*.json")):
            raise ValueError("review must finish before training begins")
    for item in worksheet["items"]:
        print(
            f"\n{item['id']} | label {item['original_label']} | {item['human_decision']}"
        )
        print(item["text"])
        if not interactive or item["human_decision"] not in ("pending", "uncertain"):
            continue
        answer = (
            input(
                "Agree [a], retain with reason [r], correct [c], exclude group [e], uncertain [u]: "
            )
            .strip()
            .lower()
        )
        if answer not in ("a", "r", "c", "e", "u"):
            raise ValueError(
                "unrecognized response; saved earlier decisions are retained"
            )
        args.decision = (
            "agree" if answer == "a" else "uncertain" if answer == "u" else "disagree"
        )
        args.resolution = {
            "a": "retain",
            "r": "retain",
            "c": "correct",
            "e": "exclude-training-group",
            "u": None,
        }[answer]
        args.reason = input("Reason: ").strip() if answer in ("r", "c", "e") else ""
        args.label = 1 - item["original_label"] if answer == "c" else None
        record_review(args, run, worksheet, item)
    if interactive:
        print(
            "\nReview saved. If you corrected or excluded rows, run resolve-review next."
        )


def select(args, run):
    from session import experiment_check, gates, experiment_names

    editable(args.out)
    gates(args.out, run)
    if not args.reason or len(args.reason.strip()) < 10:
        raise ValueError("--reason should explain the validation result and tradeoff")
    experiments = {}
    for name in experiment_names(args.out):
        path = args.out / "experiments" / name
        if path.exists():
            experiment_check(args.out, run, name)
            experiments[name] = sha(path / "experiment.json")
    if args.experiment not in experiments:
        raise ValueError("train that experiment before selecting it")
    dump(
        args.out / "decisions.json",
        {
            **author_record(args, run),
            "experiment": args.experiment,
            "rationale": args.reason,
            "experiment_hashes": experiments,
        },
    )
    print(f"Selected {args.experiment}. Run freeze when development is finished.")


def status(args, run):
    from session import gates, experiment_check, experiment_names

    print(f"Run: {args.out.name} ({run['run_kind']})")
    from runtime import controls, lineage

    policy = controls(args.out, run)
    history = lineage(args.out, run)
    print(f"Mode: {policy['mode']}; heldout exposure: {history['heldout_exposure']}")
    for name, baseline in history["baselines"].items():
        result = baseline["validation"]
        print(
            f"Parent {name}: validation recall={result['recall']:.3f}, FPR={result['fpr']:.3f}"
        )
    worksheet = read(args.out / "review.json")
    reviewed = sum(
        item["human_decision"] in ("agree", "disagree") for item in worksheet["items"]
    )
    print(f"Training review: {reviewed}/{len(worksheet['items'])} decisions")
    for name in experiment_names(args.out):
        path = args.out / "experiments" / name
        if path.exists():
            experiment_check(args.out, run, name)
            result = read(path / "validation.json")
            print(
                f"{name}: validation recall={result['recall']:.3f}, FPR={result['fpr']:.3f}, threshold={result['threshold']:.6g}"
            )
    for attempt in sorted((args.out / "attempts").glob("*.json")):
        result = read(attempt)
        if result.get("status") == "failed":
            print(f"Failed {result['experiment_id']}: {result['reason']}")
    if (args.out / "reports/complete.json").exists():
        next_step = "complete; preserve the run directory"
    elif (args.out / "verification/complete.json").exists():
        next_step = "finish writeup.md, then report --out RUN --input RUN/writeup.md"
    elif (args.out / "evaluation/complete.json").exists():
        next_step = "verify --out RUN"
    elif (args.out / "freeze.json").exists():
        next_step = "evaluate --out RUN"
    else:
        try:
            gates(args.out, run)
        except ValueError as error:
            next_step = f"confirm/review --out RUN ({error})"
        else:
            decision = read(args.out / "decisions.json")
            next_step = (
                "freeze --out RUN"
                if decision.get("experiment")
                else "plan and approve an experiment, train, compare validation results, then select"
            )
    print("Next: " + next_step)


def report_sections(path, out, run, required):
    text = Path(path).read_text()
    parts = re.split(r"^##\s+([^\n]+)\n", text, flags=re.MULTILINE)
    sections = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        key = name.strip().lower()
        if key in sections:
            raise ValueError("duplicate report section: " + key)
        sections[key] = body.strip()
    if set(sections) != set(required):
        raise ValueError("use the section headings in templates/REPORT.md")
    evidence = {
        name: sha(out / name)
        for name in (
            "freeze.json",
            "evaluation/metrics.json",
            "verification/complete.json",
        )
    }
    return {
        "run_id": run["run_id"],
        "sections": {
            name: {
                "status": "complete",
                "author_kind": "provided",
                "text": body,
                "evidence": evidence,
            }
            for name, body in sections.items()
        },
    }


def dispatch(args):
    from session import state

    run, _ = state(args.out, enforce_time=args.command != "status")
    {"status": status, "confirm": confirm, "review": review, "select": select}[
        args.command
    ](args, run)
