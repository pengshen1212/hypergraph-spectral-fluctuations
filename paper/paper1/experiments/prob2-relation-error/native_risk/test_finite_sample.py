from fractions import Fraction
import unittest

from finite_sample import (
    binomial_upper_tail_exact,
    tolerance_order_statistic_rank,
)


class FiniteSampleCalibrationTests(unittest.TestCase):
    def test_planned_calibration_rank_is_exact_minimal_rank(self):
        n = 2000
        alpha = Fraction(1, 20)
        delta = Fraction(1, 800)
        rank = tolerance_order_statistic_rank(n, alpha, delta)
        self.assertEqual(rank, 1929)
        self.assertLessEqual(
            binomial_upper_tail_exact(n, rank, 1 - alpha), delta
        )
        self.assertGreater(
            binomial_upper_tail_exact(n, rank - 1, 1 - alpha), delta
        )

    def test_bonferroni_allocation_is_at_least_099_joint_confidence(self):
        delta = Fraction(1, 800)
        per_cell_failure = binomial_upper_tail_exact(
            2000, 1929, Fraction(19, 20)
        )
        self.assertLessEqual(8 * per_cell_failure, Fraction(1, 100))

    def test_invalid_rank_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            tolerance_order_statistic_rank(0, Fraction(1, 20), Fraction(1, 100))
        with self.assertRaises(ValueError):
            binomial_upper_tail_exact(10, 0, Fraction(1, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
