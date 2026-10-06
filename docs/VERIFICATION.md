# Verification

Checked 2026-10-06 UTC. Reproduction records are in `checks/`; original evidence
and weights remain in `evidence/` and `models/`. Validation and test were previously
used. These checks add no independent test evidence.

## Data and local checks

A fresh download of the pinned WildJailbreak release restored all 6,000 rows.
The source and three original split hashes matched.
[Acquisition](../checks/acquisition-2026-10-06.json).

A clean project copy and virtual environment ran outside the source workspace on
macOS ARM64, Python 3.11.13. [Results](../checks/local-2026-10-06.json).

| Check | Result |
|---|---|
| Setup and dependencies | Passed |
| Full suite | 63 tests in normal and optimized Python; no skips |
| Synthetic CPU workflow | 64 commands passed |
| Numerical regression | 4,000 synthetic rows; finite outputs; no warnings |
| Saved TF-IDF inference | All 1,000 validation scores within 2.22e-16; no changed decisions |
| Saved encoder inference | All 1,000 validation scores within 4.41e-6; no changed decisions |

The workflow checks model extensions, saved-code loading and changes to frozen
artifacts. The macOS SciPy import required the setup script's same-version wheel fallback.

## TF-IDF: resolved

NumPy's SIMD argsort changes which tied features enter the 20,000-feature vocabulary.
Training produced 160,675 features: 16,095 occurred more than four times; 4,635 tied
at four occurrences for 3,905 remaining slots. Every differing feature was tied there.

An isolated Linux x86-64 environment used Python 3.13.15, NumPy 2.2.6, SciPy 1.15.3,
scikit-learn 1.7.2 and one numerical thread. Dependency checks passed. Text and full
feature-count hashes matched across all four runs.

| Sorting path | Historical vocabulary overlap | Largest validation score difference |
|---|---:|---:|
| [AVX512](../checks/tfidf-linux-native-2026-10-06.json) | 20,000 / 20,000 | 0 |
| [AVX2](../checks/tfidf-linux-avx2-2026-10-06.json) | 19,493 / 20,000 | 0.04676247 |
| [Scalar x86](../checks/tfidf-linux-scalar-2026-10-06.json) | 19,312 / 20,000 | 0.04505655 |
| [ARM64](../checks/tfidf-arm64-2026-10-06.json) | 19,312 / 20,000 | 0.04505655 |

The AVX512 fit independently selected every historical feature and exactly reproduced
all 1,000 scores and threshold 0.5571463167. Scalar x86 reproduced the ARM64 gap.
The separate saved-vocabulary fits isolated downstream fitting; they are not independent
feature-selection checks. Preprocessing, counts and downstream fitting do not explain
the gap. See the pinned [scikit-learn feature limit](https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/feature_extraction/text.py)
and [NumPy sorting dispatch](https://github.com/numpy/numpy/blob/v2.2.6/numpy/_core/src/npysort/quicksort.cpp).

The pinned runtime and historical-match gate preserve the original selection rule.
Other sorting paths still vary. [Commands](REPRODUCE.md).

## Encoder: full fixed fit

A Colab T4 repeated all six epochs and 3,000 steps with the pinned model revision,
original splits, seed, optimizer settings and patience rule. It reached the epoch
limit; early stopping did not trigger. [Report](../checks/encoder-colab-2026-10-06.json)
and [validation predictions](../checks/encoder-colab-validation-predictions-2026-10-06.jsonl).

| Fit | Best epoch | TP / FN / FP / TN | Recall | FPR | Threshold |
|---|---:|---:|---:|---:|---:|
| Historical | 6 | 483 / 17 / 25 / 475 | 96.6% | 5.0% | 0.0156799033 |
| Reproduction | 5 | 480 / 20 / 25 / 475 | 96.0% | 5.0% | 0.1021510139 |

The report compares every epoch. The fits changed 41/1,000 validation decisions at
their selected thresholds; the largest score difference was 0.98296058. Reloading
the reproduced checkpoint matched all 1,000 scores exactly. This was successful
fixed training, not exact reproduction of the historical encoder fit.

Recorded core versions matched, including Python 3.13.15 and Torch 2.11.0+cu130.
The rest of Colab was not isolated or fully compared; unused preinstalled packages
had dependency conflicts. CUDA/cuDNN flags came from a separate process probe,
not instrumentation of the active training process. The cause of encoder variation
is unresolved. One fixed repeat cannot establish its distribution.

The T4 reproduction used 0.28 prepaid compute units.
[Resource record](../checks/colab-resources-2026-10-06.json).
