"""Build a short report from prose and measured results."""

import time
from pathlib import Path
from data import dump
from state import read, sha, envelope
from session import event
from evaluation import verification_check

SECTIONS = (
    "target",
    "data",
    "method",
    "selection",
    "results",
    "errors",
    "limitations",
    "reproduction",
)


def report(a, cfg):
    begin = time.monotonic()
    dest = a.out / "reports"
    dest.mkdir(exist_ok=True)
    (dest / "complete.json").unlink(missing_ok=True)
    verification_check(a.out)
    run = read(a.out / "run.json")
    if a.input is not None:
        from workflow import report_sections

        dump(dest / "sections.json", report_sections(a.input, a.out, run, SECTIONS))
    sections = read(dest / "sections.json")
    if sections.get("run_id") != run["run_id"] or set(
        sections.get("sections", {})
    ) != set(SECTIONS):
        raise ValueError("report sections missing/ownership mismatch")
    for section, item in sections["sections"].items():
        if item.get("status") != "complete" or item.get("author_kind") not in (
            "agent",
            "human",
            "provided",
        ):
            raise ValueError("incomplete report")
        text = item.get("text", "")
        if len(text.strip()) < 30 or any(
            x in text.upper() for x in ("TODO", "PENDING", "PLACEHOLDER")
        ):
            raise ValueError("placeholder report")
        if not item.get("evidence"):
            raise ValueError("evidence required")
        for name, h in item["evidence"].items():
            p = Path(name)
            if p.is_absolute() or ".." in p.parts or sha(a.out / p) != h:
                raise ValueError("report evidence mismatch")
    content = (
        "\n\n".join(
            "## " + key + "\n\n" + sections["sections"][key]["text"] for key in SECTIONS
        )
        + "\n"
    )
    (dest / "writeup.md").write_text(content)
    dump(
        dest / "complete.json",
        envelope(
            run,
            {
                "verification": sha(a.out / "verification/complete.json"),
                "sections": sha(dest / "sections.json"),
                "writeup": sha(dest / "writeup.md"),
            },
            machine_metrics=read(a.out / "evaluation/metrics.json"),
            human_selection=read(a.out / "decisions.json"),
        ),
    )
    if not (a.out / "completed.json").exists():
        dump(
            a.out / "completed.json",
            {
                "end_epoch": time.time(),
                "elapsed_seconds": time.time() - run["start_epoch"],
            },
        )
    event(a.out, "report", begin)


def report_check(out):
    verification_check(out)
    r = read(out / "reports/complete.json")
    if r["run_id"] != read(out / "run.json")["run_id"] or r["inputs"] != {
        "verification": sha(out / "verification/complete.json"),
        "sections": sha(out / "reports/sections.json"),
        "writeup": sha(out / "reports/writeup.md"),
    }:
        raise ValueError("stale report")
    for item in read(out / "reports/sections.json")["sections"].values():
        for name, h in item["evidence"].items():
            if (
                Path(name).is_absolute()
                or ".." in Path(name).parts
                or sha(out / name) != h
            ):
                raise ValueError("stale report evidence")
    if r["machine_metrics"] != read(out / "evaluation/metrics.json") or r[
        "human_selection"
    ] != read(out / "decisions.json"):
        raise ValueError("report facts mismatch")
