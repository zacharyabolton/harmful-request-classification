"""Narrow fits and offline saved inference; neural dependencies are lazy."""

import importlib
import re
import math
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from data import model_text, dump
from state import read
from settings import encoder_settings, planned_steps
from early_stopping import EarlyStopping, validation_operating_point


def finite(values):
    if not np.isfinite(values).all():
        raise ValueError("nonfinite numerical output")


def fit(rows, cfg):
    training = [r for r in rows if r["split"] == "train"]
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=cfg["max_features"],
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=cfg["C"],
                    max_iter=cfg["max_iter"],
                    random_state=cfg["seed"],
                    solver=cfg.get("solver", "lbfgs"),
                ),
            ),
        ]
    )
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(
            [model_text(r, cfg["max_chars"])[0] for r in training],
            [r["label"] for r in training],
        )
    if caught:
        raise RuntimeError("fit warnings: " + "; ".join(str(w.message) for w in caught))
    if model["classifier"].n_iter_.max() >= cfg["max_iter"]:
        raise RuntimeError("did not converge")
    finite(model["classifier"].coef_)
    finite(model["classifier"].intercept_)
    return model


def score(model, rows, cfg):
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values = model.predict_proba(
            [model_text(r, cfg["max_chars"])[0] for r in rows]
        )[:, 1]
    if caught:
        raise RuntimeError(
            "score warnings: " + "; ".join(str(w.message) for w in caught)
        )
    finite(values)
    if ((values < 0) | (values > 1)).any():
        raise ValueError("score domain")
    return values


class Encoder:
    def __init__(self, path, device="cpu", max_length=256, revision=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        if device not in ("cpu", "cuda"):
            raise ValueError("unsupported encoder device")
        self.torch = torch
        self.device = device
        self.max_length = max_length
        local = Path(path).is_dir()
        kw = {"local_files_only": True} if local else {"revision": revision}
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=False, **kw)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(path), num_labels=2, **kw
        ).to(device)

    def sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def scores(self, rows, cfg):
        self.model.eval()
        result = []
        with self.torch.no_grad():
            for i in range(0, len(rows), 8):
                texts = [model_text(r, cfg["max_chars"])[0] for r in rows[i : i + 8]]
                batch = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                result.extend(
                    self.model(**batch).logits.softmax(-1)[:, 1].cpu().tolist()
                )
        finite(result)
        return np.asarray(result)

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        dump(Path(path) / "inference.json", {"max_length": self.max_length})


def train_encoder(train, val, cfg, path, steps_cap=None):
    import torch

    torch.manual_seed(cfg["seed"])
    started = time.monotonic()
    options = encoder_settings(cfg)
    stopping = (
        EarlyStopping(**options["early_stopping"], max_fpr=cfg["max_fpr"])
        if options["early_stopping"] is not None
        else None
    )
    # A step cap can end a fit before an epoch ends.
    if steps_cap is not None:
        if stopping:
            raise ValueError("early stopping cannot use a partial-epoch smoke-test cap")
        if type(steps_cap) is not int or steps_cap <= 0:
            raise ValueError("steps_cap must be positive")
        options["max_steps"] = min(options["max_steps"] or steps_cap, steps_cap)
    target_steps = planned_steps(len(train), options)
    stage_limit = options["max_stage_seconds"]
    reserve = min(120, stage_limit / 4)
    enc = Encoder(
        "microsoft/deberta-v3-xsmall",
        cfg["encoder_device"],
        revision=cfg["encoder_revision"],
    )
    texts = [model_text(r, cfg["max_chars"])[0] for r in train + val]
    lengths = [len(enc.tokenizer(x, truncation=False)["input_ids"]) for x in texts]
    enc.max_length = options["max_length"]
    if enc.max_length == "auto":
        enc.max_length = (
            512 if sum(x > 256 for x in lengths) / len(lengths) > 0.05 else 256
        )
    optimizer = torch.optim.AdamW(
        enc.model.parameters(),
        lr=options["learning_rate"],
        weight_decay=options["weight_decay"],
    )
    batch_size = options["batch_size"]
    losses = []
    steps = 0
    completed_epochs = 0
    examples_seen = 0
    pilot_start = time.monotonic()
    reference = next(enc.model.classifier.parameters()).detach().clone()
    try:
        for epoch, order, start in training_batches(torch, len(train), options):
            enc.model.train()
            batch_rows = [train[i] for i in order[start : start + batch_size]]
            batch = enc.tokenizer(
                [model_text(r, cfg["max_chars"])[0] for r in batch_rows],
                padding=True,
                truncation=True,
                max_length=enc.max_length,
                return_tensors="pt",
            ).to(enc.device)
            labels = torch.tensor([r["label"] for r in batch_rows], device=enc.device)
            optimizer.zero_grad()
            loss = enc.model(**batch, labels=labels).loss
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in enc.model.parameters()
            ):
                raise ValueError("nonfinite gradients")
            optimizer.step()
            enc.sync()
            steps += 1
            examples_seen += len(batch_rows)
            if start + batch_size >= len(order):
                completed_epochs = epoch + 1
            losses.append(float(loss.detach().cpu()))
            elapsed = time.monotonic() - pilot_start
            pilot_steps = min(options["pilot_steps"], target_steps)
            if steps <= pilot_steps and elapsed > min(180, stage_limit / 4):
                raise TimeoutError("neural pilot exceeded configured pilot budget")
            if steps == pilot_steps:
                # Measured forward validation plus conservative checkpoint/reload reserve.
                t = time.monotonic()
                enc.scores(val[:8], cfg)
                enc.sync()
                validation_projection = (time.monotonic() - t) * math.ceil(len(val) / 8)
                if stopping:
                    validation_projection *= 1 + options["epochs"]
                projection = (
                    elapsed / steps * target_steps + validation_projection + reserve
                ) * 1.25
                if projection > stage_limit:
                    raise TimeoutError(
                        "DeBERTa projected beyond configured stage budget"
                    )
            if time.monotonic() - started > stage_limit - reserve:
                raise TimeoutError("DeBERTa fit deadline")
            if stopping and start + batch_size >= len(order):
                validation_started = time.monotonic()
                point = validation_operating_point(enc, val, cfg)
                entry = stopping.observe(completed_epochs, point)
                if entry["checkpoint_improved"]:
                    enc.save(path)
                entry["validation_and_checkpoint_seconds"] = (
                    time.monotonic() - validation_started
                )
                dump(Path(path).parent / "epoch_history.json", stopping.history)
                print(
                    f"Epoch {completed_epochs}: validation recall={point['recall']:.4f}, "
                    f"FPR={point['fpr']:.4f}, best epoch={entry['best_epoch']}, "
                    f"patience={entry['bad_epochs']}/{stopping.patience}",
                    flush=True,
                )
                if time.monotonic() - started > stage_limit - reserve:
                    raise TimeoutError("DeBERTa epoch validation/checkpoint deadline")
                if entry["stop"]:
                    break
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError("OOM: experiment failed; no unrecorded retry") from exc
    changed = not torch.equal(
        reference, next(enc.model.classifier.parameters()).detach()
    )
    if not changed:
        raise ValueError("no actual parameter update")
    if stopping:
        # Reload the saved best epoch before the ordinary final validation checks.
        # Drop training references first so restoration does not retain its optimizer.
        max_length = enc.max_length
        del optimizer, enc, batch, labels, loss
        enc = Encoder(path, cfg["encoder_device"], max_length)
    scores = enc.scores(val, cfg)
    if stopping is None:
        enc.save(path)
    reload = Encoder(path, enc.device, enc.max_length)
    actual = reload.scores(val, cfg)
    if not np.allclose(scores, actual, atol=1e-5, rtol=0):
        raise ValueError("encoder reload mismatch")
    if time.monotonic() - started > stage_limit:
        raise TimeoutError("DeBERTa cap")
    diagnostics = {
        "steps": steps,
        "planned_steps": target_steps,
        "completed_epochs": completed_epochs,
        "examples_seen": examples_seen,
        "settings": options,
        "losses": losses,
        "finite_gradients": True,
        "parameter_updated": changed,
        "max_length": enc.max_length,
        "length_audit": {
            "n": len(lengths),
            "over_256": sum(x > 256 for x in lengths),
            "over_512": sum(x > 512 for x in lengths),
        },
        "seconds": time.monotonic() - started,
        "device": enc.device,
        "batch_size": batch_size,
    }
    if stopping:
        diagnostics["early_stopping"] = stopping.summary(
            "patience_exhausted" if stopping.history[-1]["stop"] else "epoch_limit"
        )
        selected = validation_operating_point(enc, val, cfg, scores=scores)
        if selected != diagnostics["early_stopping"]["best_validation"]:
            raise ValueError(
                "restored best epoch validation differs from recorded result"
            )
        diagnostics["seconds"] = time.monotonic() - started
        if diagnostics["seconds"] > stage_limit:
            raise TimeoutError("DeBERTa best-checkpoint verification deadline")
    return enc, diagnostics


def training_batches(torch, n, options):
    """Shuffle each epoch; a stated step cap may stop partway through an epoch."""
    remaining = planned_steps(n, options)
    for epoch in range(options["epochs"]):
        order = torch.randperm(n).tolist()
        for start in range(0, n, options["batch_size"]):
            if remaining == 0:
                return
            yield epoch, order, start
            remaining -= 1


def load(path, device="cpu"):
    path = Path(path)
    config = read(path / "config.json")
    kind = config.get("model", config["experiment"])
    if kind == "tfidf":
        return joblib.load(path / "model.joblib"), config
    if kind != "deberta":
        return custom_model(kind).load(path, config["config"], device), config
    enc = Encoder(
        path / "encoder", device, read(path / "encoder/inference.json")["max_length"]
    )
    return enc, config


def custom_model(name):
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", name):
        raise ValueError("model must name a Python module in src")
    try:
        module = importlib.import_module(name)
    except ImportError as error:
        raise ValueError(f"cannot import model {name}: {error}") from error
    if Path(module.__file__).resolve().parent != Path(__file__).resolve().parent:
        raise ValueError("custom model must be a module in src")
    if not all(
        callable(getattr(module, method, None))
        for method in ("train", "load", "scores")
    ):
        raise ValueError("model module must define train, load and scores")
    return module


def predict_scores(model, rows, config):
    kind = config.get("model", config.get("experiment", "tfidf"))
    if kind not in ("tfidf", "deberta"):
        values = np.asarray(
            custom_model(kind).scores(model, rows, config["config"]), dtype=float
        )
        finite(values)
        if values.shape != (len(rows),) or ((values < 0) | (values > 1)).any():
            raise ValueError("model must return one probability per row")
        return values
    return (
        model.scores(rows, config["config"])
        if isinstance(model, Encoder)
        else score(model, rows, config["config"])
    )
