# Harness

The CLI stores data, plans, models, decisions, and checks in a run directory.
Paths below are examples. Each output directory must be new.

## Train

Restore data as described in [reproduction](REPRODUCE.md), then start a run:

```sh
python src/run.py start --input .local/data --out .local/runs/repeat-01 --mode exploratory --heldout-exposure previously-evaluated
python src/run.py confirm --out .local/runs/repeat-01 --author researcher --target 'harmful request classification'
python src/run.py review --out .local/runs/repeat-01
```

Inspect each displayed training label. Record each decision with `review --out RUN --id ID
--decision agree --author researcher`. For disputed labels, use `--decision disagree
--reason TEXT --resolution retain|correct|exclude-training-group`; corrections need
`--label 0|1`. Run `resolve-review` before any fit to apply changes. Confirm and review
again afterward. Validation and test rows stay fixed.

```sh
python src/run.py plan --out .local/runs/repeat-01 --experiment tfidf --hypothesis 'Repeat the fixed lexical baseline.' --stop-rule 'One fit with the saved settings.'
python src/run.py approve --out .local/runs/repeat-01 --experiment tfidf --author researcher --reason 'Use the fixed baseline settings.'
python src/run.py train --out .local/runs/repeat-01 --experiment tfidf
python src/run.py select --out .local/runs/repeat-01 --experiment tfidf --author researcher --reason 'Select the fixed baseline for reproduction.'
python src/run.py freeze --out .local/runs/repeat-01
python src/run.py evaluate --out .local/runs/repeat-01
python src/run.py verify --out .local/runs/repeat-01
```

`status --out RUN` shows results and the next action. Training needs a confirmed target,
resolved label review, and an approved plan. Changing settings or code invalidates approval.
Freeze locks the model and threshold. Evaluation rejects a second completed pass.
A linked run (`start --parent RUN`) must use identical splits and inherits test exposure.
`--reuse-review` explicitly inherits the parent's task and label decisions.

`reproduce --out RUN` repeats a frozen TF-IDF fit and compares predictions. It needs
completed evaluation. It records a same-split check, not independent evidence.
`checkpoint` and `restore` move checksummed runs. `--help` lists their required paths.

## Models and settings

TF-IDF converts words and word pairs into weighted counts. Logistic regression predicts
the binary label. Only training rows build the vocabulary. Validation chooses the
threshold with highest recall subject to the configured false-positive limit.

`--model deberta` uses the pinned DeBERTa encoder. Its default training device is CUDA.
Use a separate config file with explicit `model_options`: `epochs`, `batch_size`,
`learning_rate`, `weight_decay`, `max_length`, and optional `early_stopping`.
Early stopping needs `patience` and `min_delta`; it restores the best validation epoch.
Pass the same config to `plan` and `train`. See `src/settings.py` for defaults and limits.
Historical settings are in the evidence records.

For another model, add `src/NAME.py` with three functions:

```python
def train(train_rows, validation_rows, cfg, path):
    # Save weights under path. Return the model and a diagnostics dict.
    ...

def load(path, cfg, device):
    # Load only the saved weights.
    ...

def scores(model, rows, cfg):
    # Return one finite class-1 probability per row, in input order.
    ...
```

Use `--model NAME --experiment FIT_NAME`. Each fit snapshots its code and settings.
Saved prediction uses that code after freezing. Failed fits keep their reason.
The extension test in `tests/test_experiments.py` checks this behavior.

## Other binary tasks

The default config fixes the WildJailbreak protocol. To change labels, splits, or
support limits, declare a `task` block. See `tests/test_task.py:task_config` for a
synthetic example and `src/task.py` for required fields. Supply partitions and record
data rights and model provenance. Cross-task parent links are rejected.
