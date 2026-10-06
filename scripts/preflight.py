"""Mandatory offline behavior checks, including encoder tests, in both Python modes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKER = """
import sys, unittest
import torch
suite = unittest.defaultTestLoader.discover('tests')
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() and result.testsRun and not result.skipped else 1)
"""


def run_checks(root=ROOT, python=sys.executable, report=None):
    root = Path(root).resolve()
    report = Path(report or root / ".local/checks/preflight.json").resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    record = {"status": "running", "checks": []}
    report.write_text(json.dumps(record, indent=2))
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    try:
        for mode, flags in [("normal", []), ("optimized", ["-O"])]:
            command = [str(python), *flags, "-c", WORKER]
            proc = subprocess.run(
                command, cwd=root, env=env, capture_output=True, text=True, timeout=600
            )
            log = report.with_name(report.stem + "-" + mode + ".log")
            log.write_text(proc.stdout + proc.stderr)
            record["checks"].append(
                {"mode": mode, "returncode": proc.returncode, "log": log.name}
            )
            report.write_text(json.dumps(record, indent=2))
            if proc.returncode:
                raise RuntimeError(f"{mode} tests failed or skipped; see {log}")
        record["status"] = "passed"
    except Exception as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        report.write_text(json.dumps(record, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_checks(report=args.report), indent=2))


if __name__ == "__main__":
    main()
