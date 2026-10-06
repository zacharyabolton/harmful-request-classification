# Harmful-request classification

I compared TF-IDF and DeBERTa to ask how much harmful-request recall I could get within a false-positive budget.
For a deployed blocking filter, false positives are a real constraint: every false alarm blocks a legitimate request.
The six-epoch encoder reached 96.6% validation recall at 5% false-positive rate, but that rate rose to 7.2% on reused test data.
I see this as a small-scale artifact-versus-behaviour problem: a checkpoint and a good validation score do not establish safe behaviour beyond the selected split.
I kept the failed baseline, split exposure, error reviews and reproduction checks visible so the claims can be challenged.

My XSTest lexical baseline flagged 47/50 benign test requests. On the custom WildJailbreak splits, TF-IDF reached 80.4% validation recall at the same 5% false-positive rate.

## What I'd conclude

- I wouldn't deploy these models as blocking filters from this evidence. I'd first need a fresh, representative evaluation showing that the false-positive budget holds.
- I see the encoder's recall gain as promising on these custom splits. I don't treat reused test data or repeated validation selection as independent confirmation.
- I'd verify both the saved artifact and the fitting procedure. Reloads matched, but refits exposed CPU-dependent TF-IDF feature selection and different encoder decisions.

## Details and reproduction

- [Experiments](docs/EXPERIMENTS.md): methods, results, errors, and limits.
- [Reproduce](docs/REPRODUCE.md): setup, saved-model inference, and training checks.
- [Data](docs/DATA.md): labels, splits, and input format.
- [Harness](docs/HARNESS.md): workflow and model extensions.
- [Attribution](notices/README.md): data and model terms.

Original code and documentation: [0BSD](LICENSE). [Third-party terms](notices/README.md) cover data and model artifacts.
