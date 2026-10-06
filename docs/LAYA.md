# Laya

Laya did not beat saved DeBERTa.
All training, inference and tests ran on a Colab T4.

| Model | Validation recall | False positives |
|---|---:|---:|
| TF-IDF | 80.4% | 5.0% |
| Laya baseline | 31.6% | 5.0% |
| Laya fine-tuned, raw scores | 94.0% | 5.0% |
| Laya fine-tuned, SDK scores | 93.6% | 4.6% |
| DeBERTa, six epochs | 96.6% | 5.0% |

Existing splits: 4,000 train, 1,000 validation. Full fine-tuning; seed 42.
Best: epoch 3 of 4. Settings and pins are in [results](../checks/laya-colab-2026-10-06.json).

Saving FP16 weights changed one decision. Refitting the threshold gave 94.0% recall.
Reload scores matched exactly. Use `sdk-threshold.json` with rounded SDK scores;
use `threshold.json` with raw scores. SDK rounding reduces recall.
Single-request p95: Laya 122 ms; DeBERTa 75 ms on this T4.

One seed; reused validation; no test evaluation. No production claim.

Weights and full logs: `.local/laya-colab-2026-10-06/result/`.
Report error fixed without retraining; logs retained.

In Colab, restore data and download `convaiinnovations/laya` at the recorded revision:

```sh
pip install -r requirements.laya-colab.txt
python scripts/experiment_laya.py --data /content/data --output /content/result --model-dir /content/laya-base
python scripts/compare_laya.py --data /content/data --result /content/result
```
