"""Data loading, hash verification and known node-law profiles.

Profiles follow the manuscript coding (paper/paper1/.../native_risk/
native_protocol.py): class index 1 is the majority class in binary profiles.
All profiles here are known synthetic laws (L1), defined using benchmark
labels; they are model validation, not label-free deployment estimates.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np

ROOT = str(Path(__file__).resolve().parents[3])

FROZEN_HASHES = {
    "cora-cc/hypergraph.pickle":
        "49abc7fff72a07de24e50d9d740d5cc63a32a40b2508995c16816fcb2729de80",
    "cora-cc/labels.pickle":
        "0e6edd0d971dde22c1116fa662362262163a7a6d557eafd60ac6857de4569527",
    "dblp-ca/hypergraph.pickle":
        "a8dbbed2041a8c1b836cbc29986ca777128eb9d75f2e006da899a801971f3278",
    "dblp-ca/labels.pickle":
        "8c95becadfa803f4dcfa56d535758619622a5f80ed18b481ab854754cd5927c8",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hashes(root=ROOT):
    out = {}
    for rel, want in FROZEN_HASHES.items():
        got = sha256_file(f"{root}/data/{rel}")
        out[rel] = (got == want, got)
    return out


def load_graph(tag, root=ROOT):
    """Return (n, edges, labels); edge slots preserved in pickle order."""
    with open(f"{root}/data/{tag}/hypergraph.pickle", "rb") as f:
        hg = pickle.load(f)
    with open(f"{root}/data/{tag}/labels.pickle", "rb") as f:
        labels = np.asarray(pickle.load(f))
    edges = [tuple(sorted(int(u) for u in e)) for e in hg.values()]
    return int(len(labels)), edges, labels


def _labels_and_majority(labels):
    labels = np.asarray(labels)
    classes, inverse, counts = np.unique(
        labels, return_inverse=True, return_counts=True)
    majority = classes[int(np.argmax(counts))]
    return labels, classes, inverse, majority


def majority_binary_profile(labels, epsilon):
    """Manuscript binary flip profile (class 1 = majority, else 0)."""
    labels, _, inverse, majority = _labels_and_majority(labels)
    P = np.empty((len(labels), 2), dtype=float)
    P[:, 1] = np.where(labels == majority, 1.0 - epsilon, epsilon)
    P[:, 0] = 1.0 - P[:, 1]
    return P


def multiclass_profile(labels, epsilon):
    """Symmetric multiclass profile over original labels (manuscript coding)."""
    labels, classes, inverse, _ = _labels_and_majority(labels)
    C = len(classes)
    P = np.full((len(labels), C), epsilon / (C - 1))
    P[np.arange(len(labels)), inverse] = 1.0 - epsilon
    return P


def hetero_binary_profile(labels, degrees, eps_low, eps_high,
                          hub_quantile=0.90):
    """Known heterogeneous law: hubs (structural degree >= quantile) get eps_high.

    Structural degree = number of edge slots containing u (all-active),
    independent of predictions.
    """
    labels, _, _, majority = _labels_and_majority(labels)
    degrees = np.asarray(degrees, dtype=float)
    cut = float(np.quantile(degrees, hub_quantile))
    eps_u = np.where(degrees >= cut, float(eps_high), float(eps_low))
    P = np.empty((len(labels), 2), dtype=float)
    P[:, 1] = np.where(labels == majority, 1.0 - eps_u, eps_u)
    P[:, 0] = 1.0 - P[:, 1]
    return P
