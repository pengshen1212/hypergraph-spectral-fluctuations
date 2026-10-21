"""Tests for the S9a linear-response decomposition (hyperops.linear_response).

1. Implementation consistency: dense formula vs class data arrays.
2. Uniform-scaling identity: with kappa=0 semantics, E = alpha*A0 gives
   J_A(E) = alpha*S0 - (alpha*S0 + alpha*S0)/2 = 0 exactly.
3. First-order accuracy: residual/true shrinks as the perturbation scale
   shrinks (paired draws at small vs large flag deviations).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hyperops as ho  # noqa: E402


def dense_mat(op, data):
    return op.csr(data).toarray()


def test_response_matches_dense_formula():
    edges = [(0, 1, 2), (0, 1, 2), (2, 3, 4), (4, 5), (1, 5)]
    P = np.array([[0.7, 0.3], [0.2, 0.8], [0.55, 0.45],
                  [0.9, 0.1], [0.4, 0.6], [0.65, 0.35]])
    op = ho.RelationOperator(6, edges, kappa=1.0)
    p = op.edge_marginals(P)
    basis = op.response_basis(p)
    rng = np.random.default_rng(0)
    a = (rng.random(op.m) < 0.5).astype(float)
    r1, rdeg, j, true, resid = op.linear_response(a, basis)

    E = dense_mat(op, op.raw_b_data(a) - basis["EB"])
    d0 = basis["d0"]
    dis = basis["di0"]
    s0 = dense_mat(op, basis["S0"])
    r1_d = dis[:, None] * E * dis[None, :]
    cvec = (E @ np.ones(6)) / d0
    rdeg_d = -0.5 * (np.outer(cvec, np.ones(6)) +
                     np.outer(np.ones(6), cvec)) * s0
    j_d = r1_d + rdeg_d
    true_d = dense_mat(op, op.S_data(a)) - s0
    assert np.allclose(dense_mat(op, r1), r1_d, atol=1e-12)
    assert np.allclose(dense_mat(op, rdeg), rdeg_d, atol=1e-12)
    assert np.allclose(dense_mat(op, j), j_d, atol=1e-12)
    assert np.allclose(dense_mat(op, true), true_d, atol=1e-12)
    assert np.allclose(dense_mat(op, resid), true_d - j_d, atol=1e-12)


def test_uniform_scaling_identity():
    """Abstract algebra check (kappa-free): E = alpha * A0 => J = 0."""
    rng = np.random.default_rng(3)
    n = 9
    M = rng.random((n, n)); B0 = (M + M.T) / 2 * 0.3
    np.fill_diagonal(B0, 0.0)
    d0 = B0.sum(axis=1) + 1.0
    A0 = B0 + np.diag(np.ones(n) * 0.0)  # pure-B base for the identity
    # recompute consistently: use A0 = base operator (no kappa), d0 = A0 1
    A0m = B0 + np.diag(np.abs(rng.random(n)) + 0.5)  # positive diag, PD-ish
    d0 = A0m @ np.ones(n)
    dis = 1.0 / np.sqrt(d0)
    S0 = dis[:, None] * A0m * dis[None, :]
    alpha = 0.37
    E = alpha * A0m
    C = np.diag((E @ np.ones(n)) / d0)  # = alpha I
    J = dis[:, None] * E * dis[None, :] - 0.5 * (C @ S0 + S0 @ C)
    assert np.max(np.abs(J)) < 1e-10


def test_first_order_derivative_scaling():
    """Continuous epsilon test: flipping one edge with strength eps.

    J is linear in the perturbation, so S(f0 + eps*flip) - S0 = eps*J_unit
    + O(eps^2). The relative residual must shrink like O(eps).
    """
    edges = [(0, 1, 2), (2, 3), (3, 4, 5), (1, 4), (0, 5)]
    op = ho.RelationOperator(6, edges, kappa=1.0)
    f0 = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    basis = op.response_basis(f0)
    flip_edge = 1  # currently 0 -> activate with strength eps

    def ratio(eps):
        a = f0.copy()
        a[flip_edge] = eps
        _, _, j, true, resid = op.linear_response(a, basis)
        return float(np.linalg.norm(dense_mat(op, resid)) /
                     max(np.linalg.norm(dense_mat(op, true)), 1e-15))

    r_big = ratio(0.2)
    r_small = ratio(0.002)
    assert r_small < 0.02 * r_big or r_small < 1e-3,         f"first-order scaling failed: {r_big} vs {r_small}"
