"""Two-layer fixed hypergraph propagation predictor.

Model (frozen architecture, no bias, no dropout):

    logits = S_all ReLU(S_all X W1) W2

``S_all`` is the paper1 dynamic-normalisation operator with every candidate
edge active (kappa=1), i.e. the predictor's propagation matrix is FIXED.
The monitored operator S(Y) later in the pipeline is a different object built
from the predictor's own output; the fixed predictor matrix never changes.

Training: full batch, Adam (lr 0.01, weight decay 5e-4 as L2), cross-entropy
on TRAIN nodes only.  Checkpoint selection uses VALIDATION loss only.  Test
labels are not an input to this module.  Randomness per run is the weight
initialisation seed; the data split and graph are fixed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PredictorConfig:
    hidden: int = 64
    epochs: int = 200
    epoch_candidates: tuple = (100, 200, 400)
    lr: float = 0.01
    weight_decay: float = 0.0005
    dropout: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8
    val_every: int = 5
    init: str = "glorot_normal"
    batch: str = "full"
    grad_clip: float = 0.0

    @classmethod
    def from_dict(cls, d):
        known = {f for f in cls.__dataclass_fields__}
        cfg = cls(**{k: v for k, v in d.items() if k in known})
        cfg.epoch_candidates = tuple(int(v) for v in cfg.epoch_candidates)
        return cfg


def softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _forward(G, S, W1, W2):
    Z = G @ W1
    H = np.maximum(Z, 0.0)
    logits = S @ (H @ W2)
    return Z, H, logits


def _ce(logits, labels, mask):
    m = np.asarray(mask, dtype=bool)
    z = logits[m]
    y = labels[m]
    z = z - z.max(axis=1, keepdims=True)
    logsum = np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(np.mean(logsum[:, 0] - z[np.arange(len(y)), y]))


def evaluate(G, S, W1, W2, labels, mask):
    _Z, _H, logits = _forward(G, S, W1, W2)
    probs = softmax(logits)
    pred = np.argmax(probs, axis=1).astype(np.int64)
    m = np.asarray(mask, dtype=bool)
    loss = _ce(logits, labels, m)
    acc = float(np.mean(pred[m] == labels[m]))
    return {"probs": probs, "pred": pred, "loss": loss, "acc": acc}


def _init_params(rng, d, hidden, n_classes):
    W1 = rng.standard_normal((d, hidden)) * np.sqrt(2.0 / d)
    W2 = rng.standard_normal((hidden, n_classes)) * np.sqrt(2.0 / hidden)
    return W1, W2


def train_predictor(G, S, labels, train_mask, val_mask, n_classes, seed,
                    cfg: PredictorConfig, epoch_limits=None, deadline=None,
                    collect_history=False):
    """Train once; return one checkpoint per requested epoch limit.

    ``epoch_limits`` defaults to ``(cfg.epochs,)``.  A checkpoint for limit E
    is the lowest-validation-loss state among evaluated epochs <= E, exactly
    the protocol that would result from training with a hard E-epoch budget.
    """
    G = np.asarray(G, dtype=np.float64)
    S = S.tocsr()
    labels = np.asarray(labels, dtype=np.int64)
    train_mask = np.asarray(train_mask, dtype=bool)
    val_mask = np.asarray(val_mask, dtype=bool)
    n, d = G.shape
    limits = tuple(sorted(int(v) for v in (epoch_limits or (cfg.epochs,))))
    max_epochs = max(limits)
    if min(limits) < 1:
        raise ValueError("epoch limits must be positive")
    if train_mask.sum() == 0 or val_mask.sum() == 0:
        raise ValueError("train and validation masks must be non-empty")
    if cfg.batch != "full":
        raise ValueError("only full-batch training is part of this protocol")

    rng = np.random.default_rng(int(seed))
    W1, W2 = _init_params(rng, d, cfg.hidden, n_classes)
    m1 = np.zeros_like(W1); v1 = np.zeros_like(W1)
    m2 = np.zeros_like(W2); v2 = np.zeros_like(W2)
    ntr = int(train_mask.sum())
    onehot = np.eye(n_classes)
    ytr = labels[train_mask]

    best = {lim: {"val_loss": np.inf, "val_acc": np.nan, "epoch": 0,
                  "W1": None, "W2": None} for lim in limits}
    history = []
    t0 = time.time()
    status = "ok"
    last_train_loss = np.nan

    for ep in range(1, max_epochs + 1):
        if deadline is not None and time.time() > deadline:
            status = "timeout"
            break
        Z, H, logits = _forward(G, S, W1, W2)
        loss = _ce(logits, labels, train_mask)
        last_train_loss = loss
        if not np.isfinite(loss):
            status = "diverged"
            break

        # backward: dL only on train rows; S is symmetric.
        P = softmax(logits)
        dlogits = P * train_mask[:, None]
        dlogits[np.arange(n)[train_mask], labels[train_mask]] -= 1.0
        dlogits /= ntr
        dU = S @ dlogits
        dW2 = H.T @ dU + cfg.weight_decay * W2
        dH = dU @ W2.T
        dZ = dH * (Z > 0.0)
        dW1 = G.T @ dZ + cfg.weight_decay * W1
        if cfg.grad_clip > 0:
            for grad in (dW1, dW2):
                nrm = float(np.linalg.norm(grad))
                if nrm > cfg.grad_clip:
                    grad *= cfg.grad_clip / nrm

        m1 = cfg.beta1 * m1 + (1 - cfg.beta1) * dW1
        v1 = cfg.beta2 * v1 + (1 - cfg.beta2) * (dW1 * dW1)
        m2 = cfg.beta1 * m2 + (1 - cfg.beta1) * dW2
        v2 = cfg.beta2 * v2 + (1 - cfg.beta2) * (dW2 * dW2)
        bc1 = 1 - cfg.beta1 ** ep
        bc2 = 1 - cfg.beta2 ** ep
        W1 -= cfg.lr * (m1 / bc1) / (np.sqrt(v1 / bc2) + cfg.adam_eps)
        W2 -= cfg.lr * (m2 / bc1) / (np.sqrt(v2 / bc2) + cfg.adam_eps)

        if ep % cfg.val_every == 0 or ep == max_epochs:
            # Validation forward MUST use the post-update weights that are
            # checkpointed (the training forward above is pre-update).
            _Zv, _Hv, val_logits = _forward(G, S, W1, W2)
            val_loss = _ce(val_logits, labels, val_mask)
            val_pred = np.argmax(val_logits[val_mask], axis=1)
            val_acc = float(np.mean(val_pred == labels[val_mask]))
            for lim in limits:
                if ep <= lim and val_loss < best[lim]["val_loss"]:
                    best[lim] = {"val_loss": float(val_loss),
                                 "val_acc": val_acc, "epoch": ep,
                                 "W1": W1.copy(), "W2": W2.copy()}
            if collect_history:
                history.append({"epoch": ep, "train_loss": float(loss),
                                "val_loss": float(val_loss),
                                "val_acc": val_acc})

    train_seconds = time.time() - t0
    results = {}
    for lim in limits:
        b = best[lim]
        if b["W1"] is None:
            # No evaluated epoch <= limit (e.g. timeout before val_every).
            results[lim] = {"status": status if status != "ok" else "empty",
                            "train_seconds": train_seconds,
                            "train_loss": last_train_loss}
            continue
        ev = evaluate(G, S, b["W1"], b["W2"], labels, val_mask)
        results[lim] = {
            "status": status, "epoch": int(b["epoch"]),
            "val_loss": float(b["val_loss"]), "val_acc": float(ev["acc"]),
            "train_loss": float(last_train_loss),
            "train_seconds": train_seconds,
            "W1": b["W1"], "W2": b["W2"],
        }
    return {"checkpoints": results, "history": history,
            "epoch_limits": limits, "status": status,
            "train_seconds": train_seconds}


def predict_full(G, S, W1, W2):
    """Probability matrix and argmax vector for all objects."""
    _Z, _H, logits = _forward(G, S, W1, W2)
    probs = softmax(logits)
    return probs, np.argmax(probs, axis=1).astype(np.int64)
