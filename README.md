# Harmful-request classification

Compare models for detecting harmful requests while keeping false alarms low.
Saved DeBERTa remains best on validation. Fine-tuned Laya was slower and missed more harmful requests.

## Results

Custom WildJailbreak splits: 4,000 training and 1,000 validation requests, each half harmful.
Recall: harmful requests caught. False-positive rate: benign requests flagged.
These are custom-split results.

| Model | Validation recall | False-positive rate |
|---|---:|---:|
| TF-IDF | 80.4% | 5.0% |
| DeBERTa, 1 epoch | 87.6% | 4.8% |
| DeBERTa, 2 epochs | 92.2% | 5.0% |
| DeBERTa, 6 epochs | 96.6% | 5.0% |
| DeBERTa, six-epoch refit | 96.0% | 5.0% |
| Laya baseline | 31.6% | 5.0% |
| Laya fine-tuned, raw scores | 94.0% | 5.0% |
| Laya fine-tuned, rounded SDK scores | 93.6% | 4.6% |

Laya selected epoch 3 of 4. On the same T4 probe, 95% of single-request predictions finished within 122 ms for Laya and 75 ms for DeBERTa.

Earlier XSTest baseline: 47/50 benign test requests flagged. Different data and objective.

## Limits and checks

One seed; reused validation. DeBERTa's false-positive rate rose to 7.2% on reused test data. Laya had no test evaluation. Blocking deployment needs fresh, representative evaluation.

Saved models reload consistently. CPU sorting changes TF-IDF refits; the DeBERTa refit changed 41/1,000 validation decisions.

Latest Colab checks: 65 tests passed in normal and optimized Python. Laya cost about 0.74 compute units; runtime closed. Laya weights and logs are local, not bundled.

## Details

- [Experiments](docs/EXPERIMENTS.md): original fits, test results and errors.
- [Laya](docs/LAYA.md): method, thresholds and saved outputs.
- [Verification](docs/VERIFICATION.md): reloads and refits.
- [Reproduce](docs/REPRODUCE.md): setup and commands.
- [Data](docs/DATA.md) · [Harness](docs/HARNESS.md) · [Attribution](notices/README.md).

Original code and documentation: [0BSD](LICENSE). [Third-party terms](notices/README.md) cover data and model artifacts.
