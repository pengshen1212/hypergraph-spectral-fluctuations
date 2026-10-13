"""Protocol-contract tests for N4 evaluation bookkeeping."""

import unittest
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

from finite_sample import tolerance_order_statistic_rank
from n4_runner import (classify_enclosure, count_envelope,
                       exact_binomial_interval, order_statistic_threshold,
                       rational_profile, retry_enclosure,
                       primary_task_consequence_gate, controlled_profile,
                       validate_seed_plan, _deterministic_witness)


class N4RunnerTests(unittest.TestCase):
    def test_strict_threshold_and_ambiguous_interval_rules(self):
        self.assertEqual(classify_enclosure(1.01, 1.02, 1.0), "alarm")
        self.assertEqual(classify_enclosure(.99, 1.0, 1.0), "nonalarm")
        self.assertEqual(classify_enclosure(.99, 1.01, 1.0), "ambiguous")

    def test_ambiguous_draws_expand_full_rate_envelope(self):
        rows = [(1.1, 1.2), (0.1, .2), (.9, 1.1), (1.2, 1.3)]
        out = count_envelope(rows, 1.)
        self.assertEqual(out["definite_alarm"], 2)
        self.assertEqual(out["definite_nonalarm"], 1)
        self.assertEqual(out["ambiguous"], 1)
        self.assertEqual(out["rate_envelope"], [0.5, 0.75])

    def test_tolerance_rank_threshold_uses_conservative_upper_scores(self):
        self.assertEqual(tolerance_order_statistic_rank(
            2000, Fraction(1, 20), Fraction(1, 800)), 1929)
        self.assertEqual(order_statistic_threshold([4., 1., 3., 2.], 3), 3.)
        with self.assertRaises(ValueError):
            order_statistic_threshold([1., 2.], 3)

    def test_exact_profiles_use_frozen_majority_and_class_coding(self):
        labels = np.array([7, 3, 7, 5, 7])
        binary = rational_profile(labels, .05, "binary_majority_vs_rest")
        self.assertEqual(binary[0], [Fraction(1, 20), Fraction(19, 20)])
        self.assertEqual(binary[1], [Fraction(19, 20), Fraction(1, 20)])
        multi = rational_profile(labels, .15, "symmetric_multiclass")
        self.assertEqual(multi[0], [Fraction(3, 40), Fraction(3, 40), Fraction(17, 20)])
        self.assertEqual(multi[0][2], Fraction(17, 20))

    def test_clopper_pearson_interval_has_expected_edges(self):
        lo, hi = exact_binomial_interval(0, 10, .95)
        self.assertEqual(lo, 0.)
        self.assertAlmostEqual(hi, 1 - .025 ** .1, places=12)
        lo, hi = exact_binomial_interval(10, 10, .95)
        self.assertAlmostEqual(lo, .025 ** .1, places=12)
        self.assertEqual(hi, 1.)

    def test_binary_profile_maps_original_majority_to_class_zero(self):
        labels = np.array([7, 3, 7, 5, 7])
        profile = rational_profile(labels, .05, "binary_majority_vs_rest")
        self.assertEqual(profile[0], [Fraction(1, 20), Fraction(19, 20)])
        self.assertEqual(profile[1], [Fraction(19, 20), Fraction(1, 20)])

    def test_controlled_profiles_are_exact_rational_rows(self):
        edges, profile = controlled_profile({"generator": "existing family_B_competition",
                                             "params": {"d": 2, "gamma": .95}})
        self.assertEqual(len(edges), 2)
        self.assertTrue(all(sum(row) == 1 for row in profile))
        self.assertEqual(profile[0], [Fraction(19, 20), Fraction(1, 20)])
        self.assertEqual(profile[2], [Fraction(1, 20), Fraction(19, 20)])

    def test_enclosure_retry_reuses_same_draw_and_keeps_unresolved_ambiguous(self):
        class Builder:
            def __init__(self):
                self.calls = []

            def build(self, indicators):
                self.calls.append(tuple(indicators))
                return object(), 0.

        calls = []

        def solve(matrix, assembly_error=0., witness=None):
            calls.append(witness)
            if len(calls) == 1:
                raise ArithmeticError("first solve failure")
            raise ArithmeticError("fallback solve failure")

        builder = Builder()
        result = retry_enclosure(builder, np.array([1, 0]), solve=solve)
        self.assertEqual(builder.calls, [(1, 0), (1, 0)])
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(result["lo"], 0.)
        self.assertEqual(result["hi"], np.finfo(float).max)

    def test_task_gate_requires_precommitted_cell_and_both_batches(self):
        batch = {"rate_envelope": [.12, .13],
                 "simultaneous_adjusted_ci_envelope": [.11, .15]}
        self.assertTrue(primary_task_consequence_gate(
            [batch, batch], certificate_lower=Fraction(7, 100),
            pooled_lower=.11, alpha=.05))
        self.assertFalse(primary_task_consequence_gate(
            [batch, {**batch, "rate_envelope": [.01, .12]}],
            certificate_lower=Fraction(7, 100), pooled_lower=.11, alpha=.05))
        self.assertFalse(primary_task_consequence_gate(
            [batch, batch], certificate_lower=Fraction(69, 1000),
            pooled_lower=.11, alpha=.05))

    def test_all_registered_control_generators_build_exact_probability_models(self):
        protocol_path = Path(__file__).with_name("formal_protocol_n4.json")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        validate_seed_plan(protocol)
        controls = protocol["controlled_boundary"]["cells"]
        self.assertEqual(len(controls), 5)
        for spec in controls:
            edges, profile = controlled_profile(spec)
            self.assertTrue(edges)
            self.assertTrue(all(sum(row) == 1 for row in profile))
        negative = next(row for row in controls if row["id"] == "node-disjoint-negative-control")
        edges, _ = controlled_profile(negative)
        self.assertEqual(len(edges), 60)
        self.assertEqual(len(set(node for edge in edges for node in edge)), 180)

    def test_frozen_direction_is_sparse_low_degree_diagonal_witness(self):
        from scipy.sparse import csr_matrix
        mean = csr_matrix(np.array([[5., .8, 0.], [.8, -3., .1],
                                    [0., .1, 2.]]))
        direction, score = _deterministic_witness(mean, "frozen-token",
                                                  node_degrees=[9, 2, 1])
        self.assertEqual(np.flatnonzero(direction).tolist(), [2])
        self.assertEqual(direction[2], 1.)
        self.assertEqual(score, 2.)
        direction, score = _deterministic_witness(mean, "frozen-token")
        self.assertEqual(np.flatnonzero(direction).tolist(), [0])
        self.assertEqual(direction[0], 1.)
        self.assertEqual(score, 5.)
        off_diagonal = csr_matrix(np.array([[0., 2.], [2., 0.]]))
        direction, score = _deterministic_witness(off_diagonal, "frozen-token")
        self.assertEqual(np.count_nonzero(direction), 1)
        self.assertEqual(np.linalg.norm(direction), 1.)
        self.assertEqual(score, 0.)


if __name__ == "__main__":
    unittest.main()
