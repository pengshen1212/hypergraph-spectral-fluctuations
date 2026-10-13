"""Tests for node_influence (first-order node-influence decomposition).

Run: python test_node_influence.py
"""

import itertools
import unittest

import numpy as np

from node_influence import (
    conditional_retention,
    exhaustive_decomposition,
    first_order_components,
    influence_profile,
    retention_probability,
)


class ProbabilityTests(unittest.TestCase):
    def test_mixed_edge_conditional_probability(self):
        probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.1, 0.9]])
        self.assertAlmostEqual(retention_probability(probs), 0.09)
        self.assertAlmostEqual(conditional_retention(probs[1:], 1), 0.81)

    def test_zero_probability_is_valid(self):
        probs = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(retention_probability(probs), 0.0)
        self.assertEqual(conditional_retention(probs[1:], 1), 1.0)

    def test_retention_is_symmetric_under_class_relabel(self):
        probs = np.array([[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]])
        self.assertAlmostEqual(retention_probability(probs),
                               retention_probability(probs[:, ::-1]))

    def test_conditional_equals_product_without_center(self):
        # zero-probability class must not be handled by division
        probs = np.array([[0.0, 1.0], [0.5, 0.5]])
        self.assertAlmostEqual(conditional_retention(probs, 0), 0.0)
        self.assertAlmostEqual(conditional_retention(probs, 1), 0.5)


class ComponentTests(unittest.TestCase):
    def test_components_match_definition(self):
        # 4 nodes, 3 edges with pairwise overlap but no containment
        edges = [(0, 1, 2), (0, 3), (2, 3)]
        P = np.array([[0.8, 0.2], [0.6, 0.4], [0.9, 0.1], [0.7, 0.3]])
        comps = first_order_components(edges, P)
        # independent reference construction of F_u(z)
        for u in range(4):
            for z in range(2):
                ref = np.zeros((4, 4))
                for e in edges:
                    if u not in e:
                        continue
                    others = [v for v in e if v != u]
                    pe = retention_probability(P[list(e)])
                    gz = conditional_retention(P[others], z)
                    for a_ in e:
                        for b_ in e:
                            ref[a_, b_] += (gz - pe) / len(e)
                self.assertTrue(np.allclose(comps[u][z], ref, atol=1e-14))

    def test_homogeneous_star_closed_form(self):
        # star: center 0, edges (0, j, j+5) disjoint leaves, flip p
        p = 0.2
        r, d = 3, 4
        edges = [(0, 1 + 2 * j, 2 + 2 * j) for j in range(d)]
        P = np.full((1 + 2 * d, 2), p)
        P[:, 0] = 1 - p  # true label 0, flip to 1 w.p. p
        comps = first_order_components(edges, P)
        dG = (1 - p) ** (r - 1) - p ** (r - 1)
        x = np.zeros(1 + 2 * d)
        x[0] = 1.0
        Phi, V1 = influence_profile(edges, P, x)
        q_r = 1 - (1 - p) ** r - p ** r
        # center influence: |Phi| = d * |dG| * h_e, h_e = 1/r (the paper's
        # dG uses true-label direction; here phi_center = p^2-(1-p)^2 = -dG)
        self.assertAlmostEqual(abs(Phi[0]), d * abs(dG) / r, places=12)
        # V1 = p(1-p)[Phi_center^2 + d(r-1)(dG/r)^2]
        expected = p * (1 - p) * ((d * dG / r) ** 2
                                  + d * (r - 1) * (dG / r) ** 2)
        self.assertAlmostEqual(V1, expected, places=12)
        # variance ratio vs B recovers cor:star-var 1+(d-1)*kappa
        h = 1.0 / r
        varB = d * q_r * (1 - q_r) * h * h
        kappa = p * (1 - p) * dG ** 2 / (q_r * (1 - q_r))
        # exhaustive Var_A includes the remainder; check V1 <= Var_A and
        # the paper's exact ratio separately in the exhaustive test below
        self.assertGreaterEqual(V1, 0.0)


class ExhaustiveTests(unittest.TestCase):
    def test_exhaustive_decomposition_identities(self):
        rng = np.random.default_rng(2026)
        for trial in range(6):
            n, m = 6, 4
            edges = [tuple(sorted(int(w) for w in
                                  rng.choice(n, 3, replace=False)))
                     for _ in range(m)]
            P = rng.random((n, 2))
            P /= P.sum(axis=1, keepdims=True)
            x = rng.normal(size=n)
            res = exhaustive_decomposition(edges, P, x=x)
            # E[X_A] = 0
            self.assertLess(np.abs(res["EX"]).max(), 1e-12)
            # E[R | pred_u = z] = 0 for all u, z
            self.assertLess(np.abs(res["ER_given"]).max(), 1e-12)
            # orthogonality: E<F_u, F_v> = 0 (u != v), E<F_u, R> = 0
            self.assertLess(np.abs(res["cross_F"]).max(), 1e-12)
            self.assertLess(np.abs(res["FR"]).max(), 1e-12)
            # Var identity: Var_A(x) = V1(x) + Var(R-part) for random x
            self.assertLess(abs(res["varA"] - res["V1"] - res["varR"]),
                            1e-12)
            # profile formula equals the component-based V1
            self.assertLess(abs(res["V1_profile"] - res["V1"]), 1e-12)
            # A/B matched marginal means: E[a_e] = p_e (MC-free, exact)
            self.assertLess(np.abs(res["pa_err"]).max(), 1e-12)

    def test_exhaustive_star_variance_ratio(self):
        # cor:star-var exact ratio for small d, checked by enumeration
        p, r, d = 0.3, 3, 3
        edges = [(0, 1 + 2 * j, 2 + 2 * j) for j in range(d)]
        P = np.full((1 + 2 * d, 2), p)
        P[:, 0] = 1 - p
        x = np.zeros(1 + 2 * d)
        x[0] = 1.0
        res = exhaustive_decomposition(edges, P, x=x)
        q_r = 1 - (1 - p) ** r - p ** r
        dG = (1 - p) ** (r - 1) - p ** (r - 1)
        varB = d * q_r * (1 - q_r) / r ** 2
        ratio = res["varA"] / varB
        expected = 1 + (d - 1) * (p * (1 - p) * dG ** 2
                                  / (q_r * (1 - q_r)))
        self.assertAlmostEqual(ratio, expected, places=10)
        # V1 <= VarA with equality iff remainder degenerate (not here)
        self.assertGreater(res["varA"], res["V1"])


class CanonicalStructureTests(unittest.TestCase):
    """The six canonical examples required by the upgrade plan:
    disjoint edges / shared center / multiple shared nodes / opposite
    label arrangements / mixed-label edges / zero-class probability."""

    STRUCTS = {
        "disjoint": ([(0, 1, 2), (3, 4, 5)], 6),
        "shared_center": ([(0, 1, 2), (0, 3, 4), (0, 5, 6)], 7),
        "multi_shared": ([(0, 1, 2), (0, 1, 3), (0, 1, 4)], 5),
        "opposite_arrangement": ([(0, 1, 2), (0, 1, 3)], 4),
        "mixed_labels": ([(0, 1, 2), (2, 3, 4), (1, 3, 5)], 6),
        "zero_class": ([(0, 1, 2), (1, 2, 3)], 4),
    }

    def probs(self, tag, n):
        rng = np.random.default_rng(hash(tag) % 2**31)
        P = rng.random((n, 2)) + 0.05
        if tag == "opposite_arrangement":
            P[:] = [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]]
        if tag == "zero_class":
            P[0] = [1.0, 0.0]
            P[3] = [0.0, 1.0]
        return P / P.sum(axis=1, keepdims=True)

    def test_identities_hold_on_all_six(self):
        for tag, (edges, n) in self.STRUCTS.items():
            P = self.probs(tag, n)
            x = np.linspace(1.0, -1.0, n)
            res = exhaustive_decomposition(edges, P, x=x)
            self.assertLess(np.abs(res["EX"]).max(), 1e-12, tag)
            self.assertLess(np.abs(res["ER_given"]).max(), 1e-12, tag)
            self.assertLess(np.abs(res["cross_F"]).max(), 1e-12, tag)
            self.assertLess(np.abs(res["FR"]).max(), 1e-12, tag)
            self.assertLess(abs(res["varA"] - res["V1"] - res["varR"]),
                            1e-12, tag)
            self.assertLess(abs(res["V1_profile"] - res["V1"]), 1e-12, tag)
            self.assertLess(np.abs(res["pa_err"]).max(), 1e-12, tag)

    def test_cancellation_flat_edge(self):
        # opposite arrangement: the competing pair (0,1) cancels the
        # first-order influence of every other node in their edges
        P = np.array([[0.9, 0.1], [0.1, 0.9], [0.4, 0.6], [0.6, 0.4]])
        edges = [(0, 1, 2), (0, 1, 3)]
        x = np.array([0.3, -0.2, 1.0, 0.7])
        Phi, V1 = influence_profile(edges, P, x)
        self.assertAlmostEqual(Phi[2], 0.0, places=12)
        self.assertAlmostEqual(Phi[3], 0.0, places=12)
        # nodes 0 and 1 keep their own influence (their co-arrangement
        # only cancels *other* nodes' influence on the shared edges)
        self.assertNotAlmostEqual(Phi[0], 0.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
