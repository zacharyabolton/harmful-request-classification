"""Validation checkpoint ranking and an explicit patience rule."""

from task import validation_minimum

import math


def validation_operating_point(encoder, rows, cfg, scores=None):
    # Imported here because measurement also uses model inference helpers.
    from measurement import threshold_sweep

    threshold, sweep = threshold_sweep(
        [row["label"] for row in rows],
        encoder.scores(rows, cfg) if scores is None else scores,
        cfg["objective"],
        validation_minimum(cfg),
        cfg.get("max_fpr", 0.05),
    )
    return next(point for point in sweep if point["threshold"] == threshold)


class EarlyStopping:
    def __init__(self, patience, min_delta, max_fpr=0.05):
        self.max_fpr = max_fpr
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad_epochs = 0
        self.history = []

    def observe(self, epoch, validation):
        recall, fpr = validation["recall"], validation["fpr"]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (recall, fpr)):
            raise ValueError("invalid early-stopping metric")
        if fpr > self.max_fpr:
            raise ValueError(
                "early-stopping operating point exceeds the FPR constraint"
            )
        if epoch != len(self.history) + 1:
            raise ValueError("early stopping requires consecutive complete epochs")
        if self.history and self.history[-1]["stop"]:
            raise ValueError("early stopping already triggered")
        reset = self.best is None or recall - self.best["recall"] > self.min_delta
        improved = self.best is None or (recall, -fpr) > (
            self.best["recall"],
            -self.best["fpr"],
        )
        self.bad_epochs = 0 if reset else self.bad_epochs + 1
        if improved:
            self.best = {"epoch": epoch, **validation}
        entry = {
            "epoch": epoch,
            "validation": validation,
            "checkpoint_improved": improved,
            "patience_reset": reset,
            "bad_epochs": self.bad_epochs,
            "best_epoch": self.best["epoch"],
            "stop": self.bad_epochs >= self.patience,
        }
        self.history.append(entry)
        return entry

    def summary(self, stop_reason):
        return {
            "monitor": f"validation recall at empirical FPR <={self.max_fpr:.1%}",
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_epoch": self.best["epoch"],
            "best_validation": {k: v for k, v in self.best.items() if k != "epoch"},
            "stop_reason": stop_reason,
            "history": self.history,
        }
