# Harmful-request classification

Can text classifiers detect harmful requests with few false alarms?

An XSTest baseline flagged 47/50 benign test requests. On WildJailbreak, validation recall was 80.4% for TF-IDF and 96.6% for six-epoch DeBERTa, both at 5% false-positive rate. The latter reached 7.2% false positives on reused test data. Early stopping never triggered.

- [Experiments](docs/EXPERIMENTS.md): methods, results, errors, and limits.
- [Reproduce](docs/REPRODUCE.md): setup, saved-model inference, and CPU checks.
- [Data](docs/DATA.md): labels, splits, and input format.
- [Harness](docs/HARNESS.md): workflow and model extensions.
- [Attribution](notices/README.md): data and model terms.

Original code and documentation: [0BSD](LICENSE). [Third-party terms](notices/README.md) cover data and model artifacts.
