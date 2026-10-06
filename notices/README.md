# Attribution

Original code and documentation use [0BSD](../LICENSE), which permits reuse without attribution. It does not relicense upstream data, dataset extracts, model weights, tokenizer files, or third-party notices. Dependencies installed by setup retain their own licenses; their code is not bundled.

XSTest: Paul Röttger and coauthors. [Pinned source](https://github.com/paul-rottger/xstest/tree/d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d), revision `d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d`. [CC BY 4.0](https://github.com/paul-rottger/xstest/blob/d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d/LICENSE). Only results and source metadata are bundled.

WildJailbreak: Liwei Jiang and coauthors, Allen Institute for AI (2024). [Pinned source](https://huggingface.co/datasets/allenai/wildjailbreak/tree/5ddc12a7894f842b0619b8e1c7ee496b198af009), revision `5ddc12a7894f842b0619b8e1c7ee496b198af009`. [ODC-BY 1.0](ODC-BY-1.0.txt) covers database rights; it does not grant every right in individual contents. Acquisition also requires the source card's research-use agreement and [AI2 Responsible Use Guidelines](https://allenai.org/responsible-use.pdf). Bundled IDs, labels, groups and predictions use custom training-release splits. Raw text is omitted. Attribute WildJailbreak when sharing these extracts or models trained from them.

DeBERTa-v3-xsmall: Microsoft. [Pinned model card](https://huggingface.co/microsoft/deberta-v3-xsmall/blob/4b419818330868dff6a60ad3e6b1c730f8b8c0c6/README.md), revision `4b419818330868dff6a60ad3e6b1c730f8b8c0c6`. Its MIT terms cover the bundled encoder and tokenizer; retain the [Microsoft license](DeBERTa-LICENSE.txt). Encoder weights were fine-tuned on WildJailbreak. The TF-IDF vocabulary also derives from that data.

Laya: Convai Innovations. [Source](https://github.com/NandhaKishorM/laya/tree/a4a8921afebfd852bba0000475cfb6ab737a124c) and [base weights](https://huggingface.co/convaiinnovations/laya/tree/7b928d828b7b0e022f929d9bd2e44165aa270148) use [Apache 2.0](https://github.com/NandhaKishorM/laya/blob/a4a8921afebfd852bba0000475cfb6ab737a124c/LICENSE). Fine-tuned on WildJailbreak. Laya source and weights are not bundled.

Upstream terms checked 2026-10-06 UTC.
