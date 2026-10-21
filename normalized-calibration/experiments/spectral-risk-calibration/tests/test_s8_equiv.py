"""Numerical equivalence: sparse-row-normalized gcn_crossfit vs the
manuscript's dense path (small graph, fixed seeds)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import s8_predictor_matrix as s8  # noqa: E402


def dense_reference(Xsp, labels, A, n_cls, seed, epochs=20):
    X = Xsp.toarray().astype(np.float64)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    n = len(labels)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), 5)
    pred = np.zeros(n, dtype=int)
    for f in range(5):
        test = folds[f]
        train = np.concatenate([folds[g] for g in range(5) if g != f])
        mask = np.zeros(n, dtype=bool)
        mask[train] = True
        W1, W2 = s8.train_gcn_masked(X, A, labels, n_cls, seed * 10 + f,
                                     mask, epochs=epochs)
        p = s8.predict_gcn(X, A, W1, W2)
        pred[test] = p[test]
    return pred


def make_case():
    rng = np.random.default_rng(0)
    n, d, C = 40, 12, 3
    X = sp.csr_matrix((rng.random((n, d)) < 0.3) * rng.random((n, d)))
    labels = rng.integers(0, C, n)
    rows, cols = [], []
    for _ in range(25):
        e = rng.choice(n, 3, replace=False)
        for a in e:
            for b in e:
                rows.append(a); cols.append(b)
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = A + sp.eye(n, format="csr")
    deg = np.asarray(A.sum(axis=1)).ravel()
    dis = 1.0 / np.sqrt(deg)
    A = (sp.diags(dis) @ A @ sp.diags(dis)).tocsr()
    return X, labels, A, C


def test_sparse_dense_equiv():
    X, labels, A, C = make_case()
    p_sparse, _ = s8.gcn_crossfit(X, labels, A, C, 7, epochs=20)
    p_dense = dense_reference(X, labels, A, C, 7, epochs=20)
    agree = float(np.mean(p_sparse == p_dense))
    assert agree >= 0.95, f"agreement {agree}"


def test_acc_reasonable_on_blobs():
    rng = np.random.default_rng(1)
    n, d, C = 60, 8, 3
    centers = rng.random((C, d)) * 3
    lab = rng.integers(0, C, n)
    Xd = centers[lab] + 0.15 * rng.random((n, d))
    X = sp.csr_matrix(Xd)
    rows, cols = [], []
    for i in range(0, n - 2):
        e = (i, i + 1, i + 2)
        for a in e:
            for b in e:
                rows.append(a); cols.append(b)
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = A + sp.eye(n, format="csr")
    deg = np.asarray(A.sum(axis=1)).ravel()
    dis = 1.0 / np.sqrt(deg)
    A = (sp.diags(dis) @ A @ sp.diags(dis)).tocsr()
    pred, val_acc = s8.gcn_crossfit(X, lab, A, C, 3, epochs=60)
    # learning sanity on synthetic data only (protocol verification is the
    # Cora run reproducing the manuscript held-out acc ~0.67)
    assert val_acc > 0.4, f"held-out acc {val_acc}"
