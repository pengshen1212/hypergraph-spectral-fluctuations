"""Unit tests for src/hyperops.py (contract v1.0). Run with pytest."""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hyperops as ho  # noqa: E402


def tiny_case():
    # edges incl. a duplicate node set across slots and a shared-hub triangle
    edges = [(0, 1, 2), (0, 1, 2), (2, 3), (3, 4, 5)]
    P = np.array([
        [0.7, 0.3], [0.2, 0.8], [0.5, 0.5],
        [0.9, 0.1], [0.4, 0.6], [0.95, 0.05],
    ])
    return ho.RelationOperator(6, edges, kappa=1.0), P


def dense_B(op, a):
    M = np.zeros((op.n, op.n))
    for i, e in enumerate(op.edges):
        for u in e:
            for v in e:
                M[u, v] += a[i] / len(e)
    return M


def dense_S(op, a):
    A = dense_B(op, a) + op.kappa * np.eye(op.n)
    d = A @ np.ones(op.n)
    dis = 1.0 / np.sqrt(d)
    return A * np.outer(dis, dis)


def test_pattern_matches_dense():
    op, P = tiny_case()
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = (rng.random(op.m) < 0.5).astype(float)
        B1 = op.csr(op.raw_b_data(a)).toarray()
        assert np.allclose(B1, dense_B(op, a))
        S1 = op.csr(op.S_data(a)).toarray()
        assert np.allclose(S1, dense_S(op, a))


def test_degrees_identity():
    op, P = tiny_case()
    rng = np.random.default_rng(1)
    for _ in range(20):
        a = (rng.random(op.m) < 0.7).astype(float)
        A = dense_B(op, a) + op.kappa * np.eye(op.n)
        assert np.allclose(op.degrees(a), A @ np.ones(op.n))
        # degree >= kappa, and equals kappa + #active edges through u
        assert np.all(op.degrees(a) >= op.kappa)


def test_lambda_max_is_one():
    op, P = tiny_case()
    rng = np.random.default_rng(2)
    for _ in range(10):
        a = (rng.random(op.m) < 0.6).astype(float)
        S = dense_S(op, a)
        d = op.degrees(a)
        assert np.allclose(S @ np.sqrt(d), np.sqrt(d), atol=1e-12)
        assert np.max(np.abs(np.linalg.eigvalsh(S))) <= 1.0 + 1e-12


def test_flags_from_Y():
    op, P = tiny_case()
    Y = np.array([1, 1, 0, 0, 0, 0])
    a = op.flags_from_Y(Y)
    assert a.tolist() == [0.0, 0.0, 1.0, 1.0]


def test_marginals_match_enumeration():
    op, P = tiny_case()
    p = op.edge_marginals(P)
    for i, e in enumerate(op.edges):
        brute = sum(np.prod([P[u, c] for u in e]) for c in range(P.shape[1]))
        assert abs(p[i] - brute) < 1e-12


def test_exact_ES_node_vs_monte_carlo():
    op, P = tiny_case()
    es = op.exact_ES_node(P)
    sampler = ho.bound_node_sampler(op, P)
    rng = np.random.default_rng(7)
    acc = np.zeros(op.n_entries)
    for _ in range(60000):
        a, _ = sampler(rng)
        acc += op.S_data(a)
    mc = acc / 60000
    assert np.max(np.abs(es - mc)) < 5e-3


def test_exact_ES_edge_differs_from_node():
    op, P = tiny_case()
    p = op.edge_marginals(P)
    es_node = op.exact_ES_node(P)
    es_edge = op.exact_ES_edge(p)
    d = np.linalg.norm(op.csr(es_node - es_edge).toarray(), 2)
    # dynamic normalization breaks mean matching in general
    assert d > 0.0


def test_fixed_degree_means_match_exactly():
    op, P = tiny_case()
    p = op.edge_marginals(P)
    di0 = op.S0_inverse_degrees(p)
    # ES0 under both arms: linear map of a -> identical expectation
    es_node_S0 = None
    # enumerate node states
    n, C = P.shape
    acc_n = np.zeros(op.n_entries)
    for st in itertools.product(range(C), repeat=n):
        prob = np.prod([P[u, c] for u, c in enumerate(st)])
        if prob:
            acc_n += prob * op.S0_data(op.flags_from_Y(st), di0)
    # enumerate edge states
    m = op.m
    acc_e = np.zeros(op.n_entries)
    for bits in itertools.product((0, 1), repeat=m):
        a = np.asarray(bits, dtype=float)
        prob = 1.0
        for i in range(m):
            prob *= a[i] * p[i] + (1 - a[i]) * (1 - p[i])
        if prob:
            acc_e += prob * op.S0_data(a, di0)
    assert np.allclose(acc_n, acc_e, atol=1e-12)


def test_rank_threshold_rule():
    scores = np.arange(1.0, 21.0)  # M=20
    q5 = ho.rank_threshold(scores, 0.05)   # k=ceil(21*0.95)=20 -> T_(20)=20
    assert q5 == 20.0
    q1 = ho.rank_threshold(scores, 0.01)   # k=ceil(21*0.99)=21 > 20 -> inf
    assert math.isinf(q1)
    # alpha=0.10: k=ceil(21*0.9)=19 -> T_(19)=19.0
    assert ho.rank_threshold(scores, 0.10) == 19.0


def test_ties_conservative():
    scores = np.zeros(50)  # all ties
    q = ho.rank_threshold(scores, 0.05)
    assert q == 0.0
    # strict exceedance of any new 0-score is impossible
    assert (np.zeros(10) > q).sum() == 0


def test_sample_Y_rows():
    P = np.array([[0.9, 0.1], [0.3, 0.7], [0.0, 1.0]])
    rng = np.random.default_rng(3)
    Ys = np.stack([ho.sample_Y(P, rng) for _ in range(3000)])
    fr = (Ys == 1).mean(axis=0)
    assert abs(fr[0] - 0.1) < 0.02
    assert abs(fr[1] - 0.7) < 0.02
    assert (Ys[:, 2] == 1).all()


def test_spectral_norm_dense_matches_numpy():
    op, P = tiny_case()
    rng = np.random.default_rng(4)
    a = (rng.random(op.m) < 0.5).astype(float)
    s = op.S_data(a)
    nrm, res = ho.spectral_norm_entries(op, s, np.zeros_like(s))
    assert res == 0.0
    assert abs(nrm - np.linalg.norm(dense_S(op, a), 2)) < 1e-10


def test_clopper_pearson_brackets():
    lo, hi = ho.clopper_pearson(50, 1000)
    assert lo < 0.05 < hi
    lo0, hi0 = ho.clopper_pearson(0, 100)
    assert lo0 == 0.0 and hi0 > 0.0
