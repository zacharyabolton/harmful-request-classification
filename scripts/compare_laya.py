"""Compare saved classifiers and inference cost on the same Colab validation rows."""

import argparse
import gc
import os
from pathlib import Path
import sys
import time

os.environ["USE_TF"] = "0"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def compare(data, result):
    import numpy as np
    import torch
    import laya
    from data import model_text
    from models import load, predict_scores
    from measurement import metrics
    from state import dump, lines, read, sha
    from experiment_laya import QUESTIONS, operating_point

    if read(result / "status.json")["status"] != "completed":
        raise ValueError("complete the experiment first")
    if sha(data / "validation.jsonl") != read(ROOT / "data/split-index.json")["split_sha256"]["validation"]:
        raise ValueError("validation hash mismatch")
    validation = lines(data / "validation.jsonl")
    report = read(result / "report.json")
    raw_predictions = lines(result / "validation-predictions.jsonl")
    if [r["id"] for r in validation] != [r["id"] for r in raw_predictions]:
        raise ValueError("prediction order mismatch")
    final_threshold = report["validation"]["threshold"]
    output = {
        "script_sha256": sha(Path(__file__)),
        "validation_sha256": sha(data / "validation.jsonl"),
        "test_evaluated": False,
        "scope": "Same 100 validation rows, five warmups, three repetitions; render/tokenize/forward/threshold/output construction. Excludes loading, disk, network and queues.",
        "models": {},
    }

    def benchmark(infer, threshold, cuda):
        probe = validation[:100]

        def sync():
            if cuda:
                torch.cuda.synchronize()

        def predict(rows):
            values = infer(rows)
            return [{"id": row["id"], "score": float(score),
                     "prediction": int(score >= threshold)}
                    for row, score in zip(rows, values)]

        for _ in range(5):
            predict(probe[:1])
        sync()
        measurements = {}
        for batch in (1, 8):
            durations = []
            count = 0
            started = time.monotonic()
            for _ in range(3):
                for start in range(0, len(probe), batch):
                    sync()
                    begin = time.monotonic()
                    predict(probe[start:start + batch])
                    sync()
                    durations.append(time.monotonic() - begin)
                    count += len(probe[start:start + batch])
                    if time.monotonic() - started > 120:
                        raise TimeoutError("inference cost probe exceeded 120 seconds")
            measurements[str(batch)] = {
                "p50_ms": float(np.percentile(durations, 50) * 1000),
                "p95_ms": float(np.percentile(durations, 95) * 1000),
                "requests_per_second": count / sum(durations),
            }
        return measurements

    for name, device in (("wj-tfidf", "cpu"), ("wj-deberta-6ep", "cuda")):
        path = ROOT / "models" / name
        model, cfg = load(path, device)
        threshold = read(path / "threshold.json")["threshold"]

        def infer(rows):
            return predict_scores(model, rows, cfg)

        scores = infer(validation)
        point = {"threshold": threshold, **metrics([r["label"] for r in validation], scores, threshold)}
        timing = benchmark(infer, threshold, device == "cuda")
        entry = {"validation_at_saved_threshold": point, "latency": timing, "device": device}
        expected = lines(ROOT / "evidence/wildjailbreak" / name / "validation-predictions.jsonl")
        if [r["id"] for r in expected] != [r["id"] for r in validation]:
            raise ValueError("reference order mismatch")
        entry["historical_maximum_score_difference"] = float(np.max(np.abs(
            scores - np.array([r["score"] for r in expected]))))
        output["models"][name] = entry
        del model, infer
        gc.collect()
        torch.cuda.empty_cache()
        dump(result / "comparison.json", output)
        print(name, "recall", point["recall"], "FPR", point["fpr"], "p95 ms", timing["1"]["p95_ms"], flush=True)

    with laya.load(str(result / "best"), device="cuda", backend="eager") as agent:
        def infer(rows):
            texts = [model_text(row, 12000)[0] for row in rows]
            answers = agent.predict_batch(texts, QUESTIONS, batch_size=8, max_len=512, head_max_len=192)
            return np.asarray([a["answers"]["harm"]["probabilities"]["B"] for a in answers])

        scores = infer(validation)
        rounded = np.array([r["score"] for r in raw_predictions])
        difference = float(np.max(np.abs(scores - rounded)))
        if difference > 1e-4:
            raise ValueError("saved model SDK parity mismatch: " + str(difference))
        point = {"threshold": final_threshold,
                 **metrics([r["label"] for r in validation], scores, final_threshold)}
        sdk_point = operating_point(validation, scores)
        dump(result / "best/sdk-threshold.json", {
            "threshold": sdk_point["threshold"], "score": "SDK P(choice B), rounded to four decimals",
            "selected_on": "previously used validation", "max_fpr": 0.05,
        })
        output["models"]["laya"] = {
            "device": "cuda", "latency": benchmark(infer, final_threshold, True),
            "sdk_maximum_score_difference": difference,
            "sdk_changed_decisions_at_unrounded_threshold": int(np.sum(
                (scores >= final_threshold) != (rounded >= final_threshold))),
            "sdk_validation_at_unrounded_threshold": point,
            "sdk_validation_at_refitted_rounded_threshold": sdk_point,
            "sdk_threshold_sha256": sha(result / "best/sdk-threshold.json"),
        }
    dump(result / "comparison.json", output)
    print("Completed saved-model comparison", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    compare(args.data, args.result)
