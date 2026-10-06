"""Create the local Python environment and check its dependencies."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default=ROOT / ".venv", type=Path)
    p.add_argument("--report", default=ROOT / ".local/checks/setup.json", type=Path)
    a = p.parse_args()
    a.env = a.env.resolve()
    a.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "started_epoch": time.time(),
        "python": sys.version,
        "steps": [],
        "cache_boundary": "pip cache may be warm",
    }

    def run(cmd, timeout=240):
        begin = time.monotonic()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        log = a.report.parent / ("setup-%d.log" % len(report["steps"]))
        log.write_text(proc.stdout + proc.stderr)
        report["steps"].append(
            {
                "command": cmd,
                "exit": proc.returncode,
                "seconds": time.monotonic() - begin,
                "log": log.name,
            }
        )
        a.report.write_text(json.dumps(report, indent=2))
        return proc.returncode

    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("run setup with Python 3.11")
    if not a.env.exists():
        if run([sys.executable, "-m", "venv", str(a.env)]):
            raise RuntimeError("venv failed")
    py = str(a.env / "bin/python")
    req = ROOT / "requirements.txt"
    report["requirements_sha256"] = hashlib.sha256(req.read_bytes()).hexdigest()
    if run([py, "-m", "pip", "install", "-r", str(req)]):
        raise RuntimeError("install failed")
    check = [
        py,
        "-c",
        'import scipy.sparse.linalg, sklearn, numpy; print("numerical imports passed")',
    ]
    if run(check):
        if sys.platform != "darwin":
            raise RuntimeError("imports failed")
        url = "https://files.pythonhosted.org/packages/4a/4a/66ba30abe5ad1a3ad15bfb0b59d22174012e8056ff448cb1644deccbfed2/scipy-1.15.3-cp311-cp311-macosx_12_0_arm64.whl#sha256=34716e281f181a02341ddeaad584205bd2fd3c242063bd3423d61ac259ca7eba"
        report["workaround"] = "same-version official macOS12 wheel"
        if run(
            [py, "-m", "pip", "install", "--no-deps", "--force-reinstall", url]
        ) or run(check):
            raise RuntimeError("wheel workaround failed")
    if run([py, "-m", "pip", "install", "-r", str(ROOT / "requirements.encoder.txt")]):
        raise RuntimeError("encoder dependencies failed")
    if run([py, "-m", "pip", "check"]):
        raise RuntimeError("pip check failed")
    lock = subprocess.run(
        [py, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout
    (a.report.parent / "resolved-requirements.txt").write_text(lock)
    from preflight import run_checks

    run_checks(ROOT, py, a.report.parent / "preflight.json")
    report.update(
        status="passed",
        seconds=time.time() - report["started_epoch"],
        environment=str(a.env),
    )
    a.report.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
