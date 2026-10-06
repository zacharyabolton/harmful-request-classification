# Experiments

Can request text distinguish harmful from benign requests with few false alarms?
The labels describe requests, not successful attacks or model responses.

Recall is the share of harmful requests detected. False-positive rate (FPR) is
the share of benign requests flagged. Precision is the share of flags labeled harmful.

## 1. XSTest: a failed baseline

**Question.** Can word counts generalize across request families?

**Method.** TF-IDF weights word counts; logistic regression maps them to a score.
The model used words and word pairs. Related categories and lexical duplicates
stayed together. Training had 275 rows across four groups; validation had 100
across two; test had 75 from one discrimination-related group. Validation F1,
the harmonic mean of precision and recall, selected threshold 0.3186046031.

| Split | Harmful / benign | TP / FN / FP / TN | Recall | FPR | F1 |
|---|---:|---:|---:|---:|---:|
| Validation | 50 / 50 | 49 / 1 / 36 / 14 | 98.0% | 72.0% | 0.726 |
| Test | 25 / 50 | 23 / 2 / 47 / 3 | 92.0% | 94.0% | 0.484 |

TP/FN count detected/missed harmful requests; FP/TN count flagged/unflagged benign requests.

**Result.** The model flagged 47 of 50 benign test requests. Benign nonhuman
targets were frequent errors; two discriminatory requests were missed.
Test precision was 32.9%. This result does not support deployment as a blocking filter.

**Limits.** One test family cannot establish broad performance. All 101,025
row pairs were checked for similar wording; seven near-duplicate pairs stayed
within splits. Semantic overlap remains unknown. An agent inspected ten training
examples; human label validation was pending. Numerical warnings remained
unexplained despite finite coefficients and matching independent logits.
A later synthetic diagnostic passed with the liblinear solver; this motivated the solver
change, not a measured quality gain.
[Diagnostic](../evidence/xstest/numerical-followup.json). No encoder was tested in this series.

[Results](../evidence/xstest/results.json) · [Settings](../evidence/xstest/method.json) ·
[Pinned source](../evidence/xstest/source.json)

## 2. WildJailbreak: model and duration comparisons

**Question.** Does a pretrained encoder improve recall at validation FPR ≤5%?
Does longer training help?

**Data.** Custom splits came from the training release at revision
`5ddc12a7894f842b0619b8e1c7ee496b198af009`. These are not official benchmark results.
Author labels were retained after human review of five training examples per class.
Each split had equal benign and harmful counts.

| Split | Rows | Harmful / benign | Groups |
|---|---:|---:|---:|
| Train | 4,000 | 2,000 / 2,000 | 3,899 |
| Validation | 1,000 | 500 / 500 | 976 |
| Test | 1,000 | 500 / 500 | 975 |

Known seed families and exact duplicates stayed in one split. A bounded check
compared 47,052 text pairs and found no cross-split matches. It was not exhaustive.
Public dataset-card examples had already been viewed; their IDs were not recorded.
Pretraining overlap is unknown. [Split counts, hashes and audit](../evidence/wildjailbreak/data.json).

**Method.** The CPU baseline used TF-IDF and logistic regression. DeBERTa-v3-xsmall
fits started from the same pinned pretrained weights with new optimizers.
They used seed 42, AdamW, learning rate 0.00002, weight decay 0.01 and batches of eight.
Training and validation text lengths set the limit at 512 tokens: 377/5,000 rows exceeded
256 tokens; 10 exceeded 512. Test lengths did not choose this limit.

Each model's threshold maximized validation recall at FPR ≤5%, then favored lower
FPR and a higher threshold. The six-epoch fit also selected checkpoints by recall, then lower FPR, then
earlier epoch. Its stopping rule allowed two epochs without recall improvement.
Epoch six won; the epoch limit was reached, so early stopping never triggered.

| Model | Validation TP / FN / FP / TN | Recall | FPR | Threshold | Stage seconds |
|---|---:|---:|---:|---:|---:|
| TF-IDF | 402 / 98 / 25 / 475 | 80.4% | 5.0% | 0.5571463167 | 4.9 |
| DeBERTa, 1 epoch | 438 / 62 / 24 / 476 | 87.6% | 4.8% | 0.5956808925 | 168.4 |
| DeBERTa, 2 epochs | 461 / 39 / 25 / 475 | 92.2% | 5.0% | 0.5178803205 | 287.4 |
| DeBERTa, 6 epochs | 483 / 17 / 25 / 475 | 96.6% | 5.0% | 0.0156799033 | 790.0 |

Stage time includes fitting and model checks. It excludes installation and data preparation.
[Exact metrics, settings and predictions](../evidence/index.json).

**Recorded decisions.** The initial series selected one epoch for higher validation
recall, accepting larger weights and slower inference. Its child series kept the
same splits and prior test exposure. It selected six epochs over two: 22 fewer
misses with 25 false positives each. All labels stayed unchanged.
[Decision chronology](../evidence/wildjailbreak/chronology.json).

**Test results.** Each selected threshold was frozen before evaluation.
The child reused the initial series' evaluated test rows.

| Model | Evidence status | TP / FN / FP / TN | Recall | FPR |
|---|---|---:|---:|---:|
| 1 epoch | First recorded evaluation; source exposure uncertain | 436 / 64 / 18 / 482 | 87.2% | 3.6% |
| 6 epochs | Reused test; exploratory | 483 / 17 / 36 / 464 | 96.6% | 7.2% |

The six-epoch validation FPR target did not transfer. Its test FPR interval was
5.2–9.8%; recall interval was 94.6–97.9%. The one-epoch intervals were 2.3–5.6%
and 84.0–89.8%. These 95% Wilson intervals treat rows as independent;
they omit group dependence, repeated selection and exposure. They do not prove
population FPR control. TF-IDF and two epochs have no recorded test evaluation.

**Errors.** One epoch missed 55/309 adversarial harmful requests and 9/191 direct
harmful requests. Six epochs flagged 30/303 adversarial benign requests and
6/197 direct benign requests.

Against two epochs, six epochs fixed 25 harmful validation misses but introduced
three. It fixed 16 benign errors and introduced 16 others. All three new misses
and four selected false positives were reviewed. One harmful label was disputed
but retained. Benign connection-building and garden-entrance requests regressed.
The sample cannot estimate error-type prevalence. Reliance on words or framing is an interpretation, not a tested mechanism.
[Paired counts](../evidence/wildjailbreak/paired-validation.json) ·
[Recorded judgments](../evidence/wildjailbreak/error-review.json).

**Cost.** On the same 100 validation rows, warmed single-request p95 was 2.5 ms
for TF-IDF on CPU and 42.3 ms for one epoch on T4. Six epochs measured 39.9 ms
on T4. Batch-eight throughput was about 2,198, 92 and 91 requests/second,
respectively. These in-process probes exclude network, queues and model loading.
They do not measure concurrent service or worst-case latency.

**Limits.** One seed and repeated validation selection limit generalization claims.
Epoch-five recall fell to 92.8%, then recovered; more epochs did not improve every
checkpoint. The separate two-epoch fit is not epoch two of the six-epoch fit.
Balanced synthetic data does not establish production precision. No external,
multilingual or multimodal test was run. Quantization, extra data and contrastive
learning remained proposals. XSTest and WildJailbreak differ in data and objectives;
their scores do not measure progress on one common benchmark.

## Evidence and reproduction

The [evidence index](../evidence/index.json) links curated records and their source
hashes. Predictions preserve public dataset IDs. Original metrics are unchanged.
Source records listed only by hash are not included.

TF-IDF and six-epoch weights are included. One- and two-epoch weights are omitted.
Raw dataset text requires acquisition. XSTest retains results and settings;
its historical pipeline and exact split artifacts are not supplied as a runnable bundle.
Historical reload checks are [separate](../evidence/wildjailbreak/historical-verification.json)
from current [reproduction checks](REPRODUCE.md).

TF-IDF's feature cap cuts through tied counts. NumPy's CPU-specific sort changes
tie order: AVX-512 reproduced all 20,000 features and 1,000 validation scores
exactly; disabling AVX2 and AVX-512 reproduced the ARM mismatch.

A full T4 encoder refit reached six epochs but selected epoch five: 96.0% recall
at 5% validation FPR, versus historical 96.6%. All 1,000 saved-reload scores
matched; 41 decisions differed from history. No test evaluation was added.
See [current verification](VERIFICATION.md).
