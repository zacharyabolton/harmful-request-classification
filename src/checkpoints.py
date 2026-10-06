"""Portable, checksummed run archives; restore never overwrites a run."""

import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

from session import state
from state import read, tree, sha


def checkpoint(args, cfg=None):
    run, _ = state(args.out, enforce_time=False)
    output = args.output.resolve()
    if output.is_relative_to(args.out.resolve()):
        raise ValueError("checkpoint must be outside the run directory")
    if output.exists():
        raise ValueError("checkpoint output already exists")
    if any(
        read(p).get("status") == "running"
        for p in (args.out / "attempts").glob("*.json")
    ):
        raise ValueError(
            "cannot checkpoint an active or interrupted fit; finish or inspect it first"
        )
    files = tree(args.out)
    manifest = json.dumps(
        {"schema_version": 1, "run_id": run["run_id"], "files": files}
    ).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=output.parent)
    os.close(fd)
    try:
        with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
            info = tarfile.TarInfo("checkpoint-manifest.json")
            info.size = len(manifest)
            archive.addfile(info, io.BytesIO(manifest))
            for name in files:
                path = args.out / name
                if path.is_symlink():
                    raise ValueError("checkpoint cannot include symlinks")
                archive.add(path, arcname="run/" + name, recursive=False)
        if tree(args.out) != files:
            raise ValueError(
                "run changed during checkpoint; retry after the command finishes"
            )
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(
        json.dumps(
            {"archive": str(output), "sha256": sha(output), "run_id": run["run_id"]}
        )
    )
    print("Copy this archive off the runtime for durable recovery.")


def restore(args, cfg=None):
    if args.out.exists():
        raise ValueError("restore destination already exists")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".restore-", dir=args.out.parent
    ) as temporary:
        staging = Path(temporary) / "run"
        staging.mkdir()
        with tarfile.open(args.input, "r:gz") as archive:
            members = archive.getmembers()
            names = [m.name for m in members]
            if len(names) != len(set(names)):
                raise ValueError("duplicate checkpoint member")
            if "checkpoint-manifest.json" not in names:
                raise ValueError("checkpoint manifest missing")
            for member in members:
                path = PurePosixPath(member.name)
                if (
                    not member.isfile()
                    or path.is_absolute()
                    or ".." in path.parts
                    or "\\" in member.name
                    or str(path) != member.name
                ):
                    raise ValueError("unsafe checkpoint member")
                if member.name != "checkpoint-manifest.json" and (
                    not member.name.startswith("run/") or len(path.parts) < 2
                ):
                    raise ValueError("unexpected checkpoint member")
            manifest = json.load(archive.extractfile("checkpoint-manifest.json"))
            if manifest.get("schema_version") != 1:
                raise ValueError("unsupported checkpoint manifest")
            for member in members:
                if member.name == "checkpoint-manifest.json":
                    continue
                target = staging / PurePosixPath(member.name).relative_to("run")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as dest:
                    import shutil

                    shutil.copyfileobj(source, dest)
        if tree(staging) != manifest["files"]:
            raise ValueError("checkpoint checksum mismatch")
        run, _ = state(staging, enforce_time=False)
        if run["run_id"] != manifest["run_id"]:
            raise ValueError("checkpoint run identity mismatch")
        staging.rename(args.out)
    print(
        "Restored "
        + run["run_id"]
        + "; original timestamps and exposure history retained."
    )
