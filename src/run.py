"""Run text-classification experiments."""

import argparse
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "1")

from session import ROOT, start, resolve_review, experiment_name
from experiments import train, freeze
from evaluation import evaluate, verify, predict, reproduce
from reporting import report
from planning import plan, approve_plan
from checkpoints import checkpoint, restore
from runtime import configure
from state import read, validate_config
from preparation import prepare as prepare_source, revise_training


COMMANDS = {
    "start": start,
    "train": train,
    "freeze": freeze,
    "evaluate": evaluate,
    "verify": verify,
    "predict": predict,
    "report": report,
    "reproduce": reproduce,
    "plan": plan,
    "approve": approve_plan,
    "checkpoint": checkpoint,
    "restore": restore,
    "runtime": configure,
}


def parser():
    cli = argparse.ArgumentParser(
        description="Run and compare text-classification experiments."
    )
    cli.set_defaults(
        out=None,
        output=None,
        input=None,
        model=None,
        review=None,
        experiment=None,
        author=None,
        target=None,
        reason=None,
        id=None,
        decision=None,
        resolution=None,
        label=None,
        replication_of=None,
        config=ROOT / "config.json",
        kind="research",
    )
    commands = cli.add_subparsers(dest="command", required=True)
    descriptions = {
        "acquire": "Download WildJailbreak.",
        "prepare": "Validate data and prepare grouped splits.",
        "start": "Create a run and start its clock.",
        "status": "Show progress, validation results and the next step.",
        "confirm": "Record the prediction task you chose.",
        "review": "Inspect and decide on sampled training labels.",
        "resolve-review": "Apply label corrections before fitting.",
        "train": "Train one named experiment.",
        "select": "Record your experiment choice and reason.",
        "freeze": "End development and lock the chosen model.",
        "evaluate": "Evaluate the frozen model on the test split.",
        "verify": "Check saved-model predictions in a fresh process.",
        "report": "Build the report from your writeup.",
        "predict": "Predict from a saved model.",
        "reproduce": "Repeat TF-IDF training with the same settings.",
        "revise-training": "Apply review changes to a new prepared dataset.",
        "plan": "Show exact fit settings, hypothesis and stopping rule for review.",
        "approve": "Approve an experiment plan before training.",
        "checkpoint": "Export a checksummed run archive for durable storage.",
        "restore": "Restore an archive to a new run directory.",
        "runtime": "Record an explicit runtime-policy change before freezing.",
    }
    for name, description in descriptions.items():
        command = commands.add_parser(name, help=description, description=description)
        if name in ("acquire", "prepare", "revise-training", "predict"):
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument(
                "--out", type=Path, required=True, help="Run directory"
            )
        if name in ("prepare", "revise-training", "start", "predict", "restore"):
            command.add_argument("--input", type=Path, required=True)
        if name in ("acquire", "prepare", "revise-training", "start"):
            command.add_argument("--config", type=Path, default=ROOT / "config.json")
        if name in ("start", "revise-training"):
            command.add_argument(
                "--kind",
                choices=["research", "fixture", "replication"],
                default="research",
            )
        if name in ("train", "plan"):
            command.add_argument(
                "--config",
                type=Path,
                default=None,
                help="Experiment settings; defaults to the run settings",
            )
            command.add_argument(
                "--model",
                help="tfidf, deberta, or a module in src; defaults to the experiment name",
            )
        if name == "start":
            command.add_argument("--replication-of")
            command.add_argument(
                "--mode", choices=["timed", "exploratory"], default="exploratory"
            )
            command.add_argument("--verification-timeout", type=int, default=600)
            command.add_argument(
                "--parent", type=Path, help="Link an earlier run using identical splits"
            )
            command.add_argument(
                "--reuse-review",
                action="store_true",
                help="Explicitly inherit the parent's task and label decisions",
            )
            command.add_argument(
                "--heldout-exposure",
                choices=["unknown", "unseen", "previously-evaluated"],
                default="unknown",
            )
        if name == "runtime":
            command.add_argument("--mode", choices=["timed", "exploratory"])
            command.add_argument("--verification-timeout", type=int)
        if name == "checkpoint":
            command.add_argument("--output", type=Path, required=True)
        if name == "plan":
            command.add_argument("--hypothesis", required=True)
            command.add_argument("--stop-rule", required=True)
        if name == "revise-training":
            command.add_argument("--review", type=Path, required=True)
        if name in ("train", "select", "freeze", "plan", "approve"):
            command.add_argument(
                "--experiment", type=experiment_name, required=name != "freeze"
            )
        if name in ("confirm", "select", "review", "approve", "runtime"):
            command.add_argument(
                "--author", required=name != "review", help="Person making the decision"
            )
        if name == "confirm":
            command.add_argument(
                "--target", required=True, help="What the model predicts"
            )
        if name in ("select", "review", "approve", "runtime"):
            command.add_argument("--reason", required=name != "review")
        if name == "review":
            command.add_argument("--id", help="Review one sample by ID")
            command.add_argument(
                "--decision", choices=["agree", "disagree", "uncertain"]
            )
            command.add_argument(
                "--resolution", choices=["retain", "correct", "exclude-training-group"]
            )
            command.add_argument("--label", type=int, choices=[0, 1])
        if name == "report":
            command.add_argument(
                "--input", type=Path, help="Markdown writeup with the template headings"
            )
        if name == "predict":
            command.add_argument("--model", type=Path, required=True)
            command.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return cli


def main():
    p = parser()
    a = p.parse_args()
    for k in ("out", "output", "input") + (
        ("model",) if a.command == "predict" else ()
    ):
        if getattr(a, k) is not None:
            setattr(a, k, getattr(a, k).resolve())
    saved = None
    if a.command == "predict" and (a.model / "source/src/run.py").exists():
        from state import tree

        files = tree(a.model)
        files.pop("experiment.json", None)
        if files != read(a.model / "experiment.json")["files"]:
            raise ValueError("saved experiment changed")
        saved = a.model / "source/src/run.py"
    elif (
        a.command in ("evaluate", "verify", "report", "reproduce")
        and (a.out / "freeze.json").exists()
    ):
        from session import state

        state(a.out)
        saved = a.out / "source/src/run.py"
    if saved is not None and saved.resolve() != Path(__file__).resolve():
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                *(["-O"] if sys.flags.optimize else []),
                str(saved),
                *sys.argv[1:],
            ]
        )
        raise SystemExit(result.returncode)
    cfg = (
        validate_config(read(a.config))
        if a.command in ("acquire", "prepare", "revise-training", "start")
        else (
            read(a.out / "config.json")
            if a.out and (a.out / "config.json").exists()
            else None
        )
    )
    if a.command == "acquire":
        if cfg.get("task"):
            raise ValueError(
                "task data acquisition must follow its declared permissions; use supplied local inputs"
            )
        from wildjailbreak import acquire

        print(acquire(a.output))
    elif a.command == "revise-training":
        if a.review is None:
            p.error("--review required")
        print(revise_training(a.input, a.review, a.output, cfg, a.kind))
    elif a.command == "prepare":
        print(prepare_source(a.input, a.output, cfg))
    elif a.command in ("status", "confirm", "review", "select"):
        from workflow import dispatch

        dispatch(a)
    elif a.command == "resolve-review":
        resolve_review(a, cfg)
    else:
        COMMANDS[a.command](a, cfg)


if __name__ == "__main__":
    import sys

    try:
        main()
    except (ValueError, RuntimeError, TimeoutError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
