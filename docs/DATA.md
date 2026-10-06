# Data

The historical series use XSTest and WildJailbreak. Split counts, label checks, and test exposure are in [Experiments](EXPERIMENTS.md).
WildJailbreak keeps author labels: benign `0`, harmful `1`.
Raw dataset text is not bundled. See [acquisition](REPRODUCE.md) and [attribution](../notices/README.md).

## Inputs

Use JSONL: one object per line.

```json
{"id":"row-1","label":0,"source":"dataset","revision":"v1","group_id":"seed-1","label_provenance":"source labels","raw":{"text":"A request to classify"},"split":"train"}
```

IDs must be unique. Labels are integers `0` or `1`. `group_id` joins related examples.
Source, revision, label provenance, and split are metadata; the model does not see them.
CSV accepts the same fields with JSON in the `raw` column.

Prediction needs only an ID and raw input:

```json
{"id":"request-1","raw":{"text":"A request to classify"}}
```

Conversation inputs use `raw.messages` and a top-level integer `target_index`.
Only messages through that index are visible. Role, text, and tool calls are retained.
Unknown fields and future outcomes are excluded. Nontext parts are marked unsupported.
Optional `raw.policy`, `raw.context`, and `raw.target_action` are model inputs; use them only when available at prediction time.
Text is truncated at `max_chars`, with a trace. Oversized conversations fail. DeBERTa also records token truncation.

## New data

Declare labels, sources, permissions, model-overlap checks, and split minimums in the task configuration.
Supply `train`, `validation`, or `test` for each row. Preparation preserves supplied rows and splits; cross-split overlap fails.
Added generated rows need a training parent in the same group and cannot enter evaluation splits.
Keep evaluation prompts out of generation. Record label checks; a requested label is not ground truth.

The harness supports binary text classification. Other targets need code and test changes.
