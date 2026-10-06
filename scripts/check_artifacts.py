"""Check the listed distribution files and hashes."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = {".local", ".venv", ".git", "__pycache__", ".ruff_cache", ".pytest_cache"}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check(root=ROOT):
    root = Path(root)
    manifest = json.loads((root / "DISTRIBUTION.json").read_text())
    expected = manifest["files"]
    for name, checksum in expected.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("invalid distribution path")
        file = root / path
        if file.is_symlink() or not file.is_file() or digest(file) != checksum:
            raise ValueError("missing or changed file: " + name)
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and not any(part in LOCAL or part.startswith(".venv-") for part in path.relative_to(root).parts)
        and path.name != "DISTRIBUTION.json"
    }
    if actual != set(expected):
        raise ValueError("unlisted files: " + ", ".join(sorted(actual - set(expected))))
    return {"status": "passed", "files": len(expected)}


if __name__ == "__main__":
    print(json.dumps(check()))
