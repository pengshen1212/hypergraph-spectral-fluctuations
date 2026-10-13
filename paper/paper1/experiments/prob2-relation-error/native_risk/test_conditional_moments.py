import itertools
import unittest

import numpy as np

from conditional_moments import (
    conditional_edge_rate,
    conditional_edge_moments,
    conditional_frobenius_residual_second_moment,
    conditional_joint_retention,
    conditional_state_probability,
    directional_conditional_moments,
    enumerate_group_states,
)
from risk_certificate import (
    cantelli_tail_lower,
    frobenius_tail_upper,
    mixture_tail_bounds,
)
from native_protocol import (
    majority_binary_profile,
    multiclass_profile,
    pilot_seed_key,
    select_structural_groups,
)


def exhaustive_conditional(edges, P, group, state, x):
    n, classes = P.shape
    group = tuple(group)
    fixed = dict(zip(group, state))
    rest = [u for u in range(n) if u not in fixed]
    p_e = [float(np.prod(P[list(e)], axis=0).sum()) for e in edges]
    atoms = []
    for labels in itertools.product(range(classes), repeat=len(rest)):
        assignment = dict(fixed)
        assignment.update(zip(rest, labels))
        probability = float(np.prod([P[u, assignment[u]] for u in rest]))
        if probability == 0:
            continue
        X = np.zeros((n, n))
        for edge, pe in zip(edges, p_e):
            retained = len({assignment[u] for u in edge}) == 1
            weight = float(retained) - pe
            idx = np.asarray(edge, dtype=int)
            X[np.ix_(idx, idx)] += weight / len(edge)
        atoms.append((probability, X))
    norm = sum(prob for prob, _ in atoms)
    mean = sum((prob * X for prob, X in atoms), np.zeros((n, n))) / norm
    var_direction = sum(
        prob * (float(x @ X @ x) - float(x @ mean @ x)) ** 2
        for prob, X in atoms
    ) / norm
    frobenius_second = sum(
        prob * float(np.linalg.norm(X - mean, "fro") ** 2)
        for prob, X in atoms
    ) / norm
    return mean, var_direction, frobenius_second


class ConditionalProbabilityTests(unittest.TestCase):
    def test_fixed_and_conflicting_labels(self):
        P = np.array([[0.6, 0.4], [0.3, 0.7], [0.8, 0.2]])
        self.assertAlmostEqual(
            conditional_edge_rate((0, 1), P, (0,), (1,)), 0.7
        )
        self.assertEqual(
            conditional_edge_rate((0, 1), P, (0, 1), (0, 1)), 0.0
        )

    def test_shared_free_vertex_has_exact_joint_probability(self):
        P = np.array([[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])
        group, state = (0,), (1,)
        e, f = (0, 1), (0, 1, 2)
        joint = conditional_joint_retention(e, f, P, group, state)
        ge = conditional_edge_rate(e, P, group, state)
        gf = conditional_edge_rate(f, P, group, state)
        self.assertAlmostEqual(joint, 0.42)
        self.assertAlmostEqual(joint - ge * gf, 0.126)

    def test_edges_sharing_only_fixed_nodes_are_conditionally_independent(self):
        P = np.array([[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])
        e, f = (0, 1), (0, 2)
        group, state = (0,), (1,)
        joint = conditional_joint_retention(e, f, P, group, state)
        ge = conditional_edge_rate(e, P, group, state)
        gf = conditional_edge_rate(f, P, group, state)
        self.assertAlmostEqual(joint, ge * gf)

    def test_zero_probability_state_is_defined_without_division(self):
        P = np.array([[1.0, 0.0], [0.3, 0.7]])
        self.assertEqual(conditional_state_probability(P, (0,), (1,)), 0.0)
        self.assertAlmostEqual(
            conditional_edge_rate((0, 1), P, (0,), (1,)), 0.7
        )

    def test_group_states_sum_to_one_and_skip_impossible_states(self):
        P = np.array([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5]])
        states = enumerate_group_states(P, (0, 1))
        self.assertEqual(len(states), 3)
        self.assertAlmostEqual(sum(prob for _, prob in states), 1.0)


class ConditionalMomentEnumerationTests(unittest.TestCase):
    def test_sparse_conditional_moments_match_full_enumeration(self):
        rng = np.random.default_rng(20261011)
        P = rng.uniform(0.1, 1.0, size=(6, 3))
        P /= P.sum(axis=1, keepdims=True)
        edges = [
            (0, 2, 3),
            (0, 2, 4),
            (1, 3, 5),
            (3, 4, 5),
            (3, 4, 5),
        ]
        group, state = (0, 1), (0, 2)
        x = np.array([0.3, -0.2, 0.0, 1.0, 0.7, 0.0])
        expected_mean, expected_var, expected_frob = exhaustive_conditional(
            edges, P, group, state, x
        )

        got_mean = conditional_edge_moments(
            edges, P, group, state
        )["mean"].toarray()
        got_direction_mean, got_var = directional_conditional_moments(
            edges, P, group, state, x
        )
        got_frob = conditional_frobenius_residual_second_moment(
            edges, P, group, state
        )

        self.assertTrue(np.allclose(got_mean, expected_mean, atol=1e-12))
        self.assertAlmostEqual(
            got_direction_mean, float(x @ expected_mean @ x), places=11
        )
        self.assertAlmostEqual(got_var, expected_var, places=11)
        self.assertAlmostEqual(got_frob, expected_frob, places=11)

    def test_mean_mixture_is_zero_and_duplicate_slots_are_not_deduplicated(self):
        P = np.array([[0.4, 0.6], [0.3, 0.7], [0.8, 0.2]])
        edges = [(0, 1), (0, 1), (0, 2)]
        states = enumerate_group_states(P, (0,))
        matrices = [
            (prob, conditional_edge_moments(
                edges, P, (0,), state
            )["mean"].toarray())
            for state, prob in states
        ]
        mixture = sum(
            (prob * matrix for prob, matrix in matrices),
            np.zeros((3, 3)),
        )
        self.assertLess(np.abs(mixture).max(), 1e-12)

    def test_tail_bounds_bracket_exhaustive_grouped_operator_tail(self):
        P = np.array([
            [0.8, 0.2],
            [0.3, 0.7],
            [0.6, 0.4],
            [0.45, 0.55],
            [0.7, 0.3],
        ])
        edges = [(0, 1, 2), (0, 2, 3), (1, 3, 4), (1, 3, 4)]
        group = (0, 1)
        mixed_lower = {tau: 0.0 for tau in (0.25, 0.75, 1.5)}
        mixed_upper = {tau: 0.0 for tau in (0.25, 0.75, 1.5)}
        mixed_tail = {tau: 0.0 for tau in (0.25, 0.75, 1.5)}
        for state, probability in enumerate_group_states(P, group):
            moments = conditional_edge_moments(edges, P, group, state)
            dense_mean = moments["mean"].toarray()
            values, vectors = np.linalg.eigh(dense_mean)
            k = int(np.argmax(np.abs(values)))
            x = vectors[:, k]
            mean_y, variance_y = directional_conditional_moments(
                edges, P, group, state, x
            )
            mu = abs(mean_y)
            residual2 = conditional_frobenius_residual_second_moment(
                edges, P, group, state
            )
            _, _, exact_norm_tail = exhaustive_conditional(
                edges, P, group, state, x
            )
            fixed = dict(zip(group, state))
            rest = [u for u in range(len(P)) if u not in fixed]
            outcomes = []
            for labels in itertools.product(range(P.shape[1]), repeat=len(rest)):
                assignment = dict(fixed)
                assignment.update(zip(rest, labels))
                probability_rest = float(np.prod([
                    P[u, assignment[u]] for u in rest
                ]))
                if probability_rest == 0:
                    continue
                weights = []
                for edge in edges:
                    same = len({assignment[u] for u in edge}) == 1
                    pe = float(np.prod(P[list(edge)], axis=0).sum())
                    weights.append(float(same) - pe)
                X = np.zeros((len(P), len(P)))
                for edge, weight in zip(edges, weights):
                    idx = np.asarray(edge)
                    X[np.ix_(idx, idx)] += weight / len(edge)
                outcomes.append((probability_rest, np.linalg.norm(X, 2)))

            for tau in (0.25, 0.75, 1.5):
                exact_tail = sum(w for w, norm in outcomes if norm > tau)
                lower = cantelli_tail_lower(mu, variance_y, tau)
                upper = frobenius_tail_upper(
                    np.linalg.norm(dense_mean, "fro"), residual2, tau
                )
                self.assertLessEqual(lower, exact_tail + 1e-12)
                self.assertGreaterEqual(upper + 1e-12, exact_tail)
                mixed_lower[tau] += probability * lower
                mixed_upper[tau] += probability * upper
                mixed_tail[tau] += probability * exact_tail
        for tau in mixed_tail:
            self.assertLessEqual(mixed_lower[tau], mixed_tail[tau] + 1e-12)
            self.assertGreaterEqual(mixed_upper[tau] + 1e-12, mixed_tail[tau])


class TailInequalityTests(unittest.TestCase):
    def test_cantelli_handles_zero_variance_and_nonpositive_gap(self):
        self.assertEqual(cantelli_tail_lower(2.0, 0.0, 1.0), 1.0)
        self.assertEqual(cantelli_tail_lower(1.0, 0.3, 1.0), 0.0)

    def test_mixture_bounds_require_a_probability_distribution(self):
        states = [{
            "probability": 0.25,
            "mean": 2.0,
            "variance": 0.5,
            "mean_norm_upper": 1.0,
            "residual_second_moment": 0.2,
        }, {
            "probability": 0.75,
            "mean": 0.0,
            "variance": 1.0,
            "mean_norm_upper": 0.0,
            "residual_second_moment": 0.1,
        }]
        lower, upper = mixture_tail_bounds(states, 1.5)
        self.assertGreaterEqual(lower, 0.0)
        self.assertLessEqual(lower, upper)
        with self.assertRaises(ValueError):
            mixture_tail_bounds(states[:1], 1.5)


class NativeProtocolTests(unittest.TestCase):
    def test_pilot_seed_key_uses_frozen_protocol_names(self):
        self.assertEqual(pilot_seed_key("cora-cc", 0.05), "cora-epsilon-005")
        self.assertEqual(pilot_seed_key("dblp-ca", 0.15), "dblp-epsilon-015")

    def test_selection_uses_degree_then_weighted_overlap_and_stable_ties(self):
        edges = [(0, 1), (0, 2, 3), (0, 2), (2, 3)]
        selected = select_structural_groups(edges, n_nodes=4)
        self.assertEqual(selected["hub"], 0)
        self.assertEqual(selected["partner"], 2)
        self.assertAlmostEqual(selected["partner_score"], 1.0 / 3.0 + 0.5)

        tied = select_structural_groups([(0, 1), (0, 2)], n_nodes=3)
        self.assertEqual(tied["hub"], 0)
        self.assertEqual(tied["partner"], 1)

    def test_repeated_edge_slots_contribute_to_degree_and_overlap(self):
        selected = select_structural_groups(
            [(0, 1), (0, 1), (0, 2)], n_nodes=3
        )
        self.assertEqual(selected["hub"], 0)
        self.assertEqual(selected["partner"], 1)
        self.assertEqual(selected["degree"][0], 3)
        self.assertEqual(selected["partner_score"], 1.0)

    def test_binary_and_multiclass_profiles_keep_labels_and_row_sums(self):
        labels = np.array([2, 1, 2, 3])
        binary, majority = majority_binary_profile(labels, 0.1)
        self.assertEqual(majority, 2)
        self.assertTrue(np.allclose(binary.sum(axis=1), 1.0))
        self.assertAlmostEqual(binary[0, 1], 0.9)
        self.assertAlmostEqual(binary[1, 1], 0.1)

        multiclass, classes = multiclass_profile(labels, 0.1)
        self.assertEqual(classes, [1, 2, 3])
        self.assertTrue(np.allclose(multiclass.sum(axis=1), 1.0))
        self.assertAlmostEqual(multiclass[0, classes.index(2)], 0.9)
        self.assertAlmostEqual(multiclass[0, classes.index(1)], 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
