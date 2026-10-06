"""Check TF-IDF numerical stability on synthetic data."""

import argparse
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from state import read, dump, sha
from models import fit, score
from sklearn.metrics import log_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("diagnostic output exists")
    cfg = read(ROOT / "tests/fixtures/config.json")
    rng = random.Random(42)
    start = time.monotonic()
    records = [
        {
            "id": str(i),
            "label": i % 2,
            "split": "train",
            "raw": {
                "text": (
                    "benign ordinary garden "
                    if i % 2 == 0
                    else "harmful dangerous attack "
                )
                + " ".join("token" + str(rng.randrange(10000)) for _ in range(100))
            },
        }
        for i in range(4000)
    ]
    result = {
        "fixture_only": True,
        "rows": len(records),
        "solver": cfg["solver"],
        "command": [sys.executable, *sys.argv],
        "source_sha256": sha(__file__),
        "models_sha256": sha(ROOT / "src/models.py"),
        "config_sha256": sha(ROOT / "tests/fixtures/config.json"),
        "python": sys.version,
    }
    try:
        model = fit(records, cfg)
        scores = score(model, records, cfg)
        loss = float(log_loss([r["label"] for r in records], scores))
        if not math.isfinite(loss):
            raise ValueError("nonfinite loss")
        result.update(
            status="passed",
            features=len(model["tfidf"].vocabulary_),
            iterations=model["classifier"].n_iter_.tolist(),
            finite_weights_scores_loss=True,
            loss=loss,
            warnings=[],
        )
    except Exception as exc:
        result.update(status="failed", error=str(exc))
        raise
    finally:
        result["seconds"] = time.monotonic() - start
        dump(a.output, result)


if __name__ == "__main__":
    main()
