"""Repeat the fixed encoder fit; keep new checks separate from historical evidence."""

import argparse
import importlib.metadata
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from models import Encoder, train_encoder
from early_stopping import validation_operating_point
from state import dump, jsonl, lines, read, sha, tree, utc


def checked_splits(data):
    index = read(ROOT / "data/split-index.json")
    for split in ("train", "validation"):
        if sha(data / (split + ".jsonl")) != index["split_sha256"][split]:
            raise ValueError("split checksum mismatch: " + split)
    return lines(data / "train.jsonl"), lines(data / "validation.jsonl")


def reproduce(data, output, device="cuda"):
    if output.exists():
        raise ValueError("output exists")
    train, validation = checked_splits(data)
    historical = ROOT / "evidence/wildjailbreak/wj-deberta-6ep"
    cfg = read(historical / "method.json")["settings"]
    cfg["encoder_device"] = device
    expected = lines(historical / "validation-predictions.jsonl")
    if [r["id"] for r in validation] != [r["id"] for r in expected]:
        raise ValueError("validation order mismatch")
    output.mkdir(parents=True)
    dump(output / "settings.json", cfg)
    dependencies = read(historical / "method.json")["historical_runtime"]["core_dependencies"]
    runtime = {
        "python": sys.version, "platform": platform.platform(),
        "dependencies": {name.split("==")[0]: importlib.metadata.version(name.split("==")[0])
                         for name in dependencies},
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if device == "cuda" else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "source_hashes": tree(ROOT / "src"),
    }
    dump(output / "runtime.json", runtime)
    dump(output / "status.json", {"status": "running", "started": utc()})
    try:
        model, diagnostics = train_encoder(train, validation, cfg, output / "encoder")
        scores = model.scores(validation, cfg)
        loaded = Encoder(output / "encoder", device, model.max_length)
        reloaded = loaded.scores(validation, cfg)
        threshold = validation_operating_point(model, validation, cfg, scores)["threshold"]
        old_scores = np.array([r["score"] for r in expected])
        old_threshold = read(historical / "validation.json")["threshold"]
        predictions_path = output / "validation-predictions.jsonl"
        jsonl(predictions_path, [
            {"id": row["id"], "label": row["label"], "score": float(value),
             "prediction": int(value >= threshold)}
            for row, value in zip(validation, scores)
        ])
        historical_epochs = read(ROOT / "evidence/wildjailbreak/epochs.json")
        epoch_comparison = [
            {
                "epoch": entry["epoch"],
                "recall_difference": entry["validation"]["recall"] - old["validation"]["recall"],
                "fpr_difference": entry["validation"]["fpr"] - old["validation"]["fpr"],
                "threshold_difference": entry["validation"]["threshold"] - old["validation"]["threshold"],
                "historical_best_epoch": old["best_epoch"],
                "reproduced_best_epoch": entry["best_epoch"],
            }
            for entry, old in zip(diagnostics["early_stopping"]["history"], historical_epochs)
        ]
        losses = diagnostics.pop("losses")
        diagnostics["loss_summary"] = {"steps": len(losses), "first": losses[0], "last": losses[-1]}
        report = {
            "date_utc": utc(), "status": "completed",
            "settings": cfg, "runtime": runtime,
            "split_sha256": {s: sha(data / (s + ".jsonl")) for s in ("train", "validation")},
            "validation": validation_operating_point(model, validation, cfg, scores),
            "diagnostics": diagnostics,
            "historical_comparison": {
                "maximum_score_difference": float(np.max(np.abs(scores - old_scores))),
                "changed_decisions": int(np.sum((scores >= threshold) != (old_scores >= old_threshold))),
                "threshold_difference": float(threshold - old_threshold),
                "epochs": epoch_comparison,
                "historical_best_epoch": historical_epochs[-1]["best_epoch"],
                "reproduced_best_epoch": diagnostics["early_stopping"]["best_epoch"],
            },
            "saved_reload": {
                "rows": len(validation),
                "maximum_score_difference": float(np.max(np.abs(scores - reloaded))),
                "changed_decisions": int(np.sum((scores >= threshold) != (reloaded >= threshold))),
            },
            "test_evaluated": False,
            "validation_predictions_sha256": sha(predictions_path),
            "interpretation": "Fixed reproduction on previously used validation; no independent test evidence.",
        }
        dump(output / "report.json", report)
        dump(output / "status.json", {"status": "completed", "finished": utc()})
        print(f"Completed {diagnostics['completed_epochs']} epochs; "
              f"best={diagnostics['early_stopping']['best_epoch']}, "
              f"recall={report['validation']['recall']:.3f}, "
              f"FPR={report['validation']['fpr']:.3f}, "
              f"reload difference={report['saved_reload']['maximum_score_difference']:.3g}")
        return report
    except Exception as error:
        dump(output / "status.json", {"status": "failed", "reason": str(error), "finished": utc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    reproduce(args.data, args.output, args.device)
