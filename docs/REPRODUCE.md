# Reproduce

Local checks use Python 3.11. Run commands from the project root.
Setup installs pinned CPU and encoder dependencies. Allow 2 GB free disk space.
No GPU or Hugging Face account is needed for the tests.

The encoder checkpoint uses Git LFS. For a Git checkout, install Git LFS and
fetch the model before setup:

```sh
git lfs install --local
git lfs pull
```

Use a clone with LFS files downloaded; source archives may contain pointers.

```sh
python3.11 scripts/setup.py
source .venv/bin/activate
python scripts/smoke_test.py --output .local/cpu-check
```

Setup runs the full suite in normal and optimized Python. Both must pass without skips.
The CPU check trains two small synthetic TF-IDF models. It tests saved predictions,
source snapshots, blocked actions, and changed artifacts. It prints
`Synthetic CPU workflow passed`. Use a new output directory on each run.
These checks measure software behavior, not historical model quality.

## Saved models

```sh
python src/run.py predict --model models/wj-tfidf --input examples/requests.jsonl --output .local/tfidf-predictions.jsonl
python src/run.py predict --model models/wj-deberta-6ep --input examples/requests.jsonl --output .local/encoder-predictions.jsonl
```

Both run on CPU without network access. Output has one ID, score, binary prediction,
and preprocessing record per input. Scores estimate class 1; they are not calibrated risks.
Existing output files are rejected. Load only trusted model files: joblib uses Python pickle.

## Historical data

```sh
python scripts/restore_data.py --output .local/data
```

WildJailbreak requires Hugging Face access. Accept its research-use terms and AI2
Responsible Use Guidelines, then run `hf auth login` outside this project.
Keep credentials outside the export.
The download is about 531 MB. It restores the pinned split membership.
See [data](DATA.md) for fields and split limits, and [notices](../notices/README.md) for rights.

Historical metrics are saved evidence. Installation and inference checks do not repeat
historical training. GPU training can vary across software and hardware.
Use a new run for every reproduction. Mark this test split `previously-evaluated`.
Never replace files under `evidence/` or `models/`.

## Fixed training checks

Use the restored splits. New results belong outside `evidence/` and `models/`.
Copy the project and restored splits into a Colab T4 runtime.
The checked runtime used Python 3.13.15. Install the recorded core versions:

```sh
python -m pip install -r requirements.colab.txt
python scripts/reproduce_encoder.py --data .local/data --output .local/encoder-repeat
```

This repeats the six-epoch plan and recorded stopping rule. It saves the best
checkpoint, validation predictions, reload comparison and runtime. It never evaluates test.
The checked run selected epoch five; historical training selected epoch six.
Matching core versions did not reproduce every score or checkpoint.

For TF-IDF, the checked historical profile used Linux x86-64, Python 3.13.15,
NumPy 2.2.6 and AVX512 sorting. Disabling SIMD changes tied feature selection.
Check the refit separately in that profile:

```sh
python scripts/diagnose_tfidf.py --data .local/data --output .local/tfidf-refit.json --require-historical-match
```

The gate checks independently selected features and historical validation scores.
A mismatch exits nonzero and keeps its report. Saved-vocabulary fitting is a
separate diagnostic. See [verification](VERIFICATION.md) for the supported runtime.

## Check the files

```sh
python scripts/check_artifacts.py
python scripts/check_evidence.py
```

The artifact manifest pins this distribution. Edits invalidate its hashes.
The evidence check recomputes saved metrics without training.

After code edits:

```sh
python scripts/preflight.py
python scripts/check_numerics.py --output .local/numerics.json
```

See [verification](VERIFICATION.md) for completed checks.
See [the harness](HARNESS.md) to train or add a model.
