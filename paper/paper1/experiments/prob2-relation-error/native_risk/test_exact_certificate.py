from fractions import Fraction
import itertools
import unittest

from exact_certificate import (
    exact_directional_moments,
    exact_group_probability,
    exact_cantelli_lower,
)


class ExactDirectionalCertificateTests(unittest.TestCase):
    def test_exact_moments_match_conditioned_enumeration(self):
        half = Fraction(1, 2)
        P = [[half, half] for _ in range(3)]
        edges = [(0, 1), (0, 2), (1, 2)]
        group, state = (0,), (1,)
        direction = [Fraction(1), Fraction(-2), Fraction(1, 2)]

        result = exact_directional_moments(
            edges, P, group, state, direction
        )
        outcomes = []
        for free_labels in itertools.product(range(2), repeat=2):
            labels = (1,) + free_labels
            probability = Fraction(1, 4)
            edge_values = [
                int(len({labels[u] for u in edge}) == 1) for edge in edges
            ]
            edge_rates = [Fraction(1, 2)] * len(edges)
            weights = [
                sum(direction[u] for u in edge) ** 2 / len(edge)
                for edge in edges
            ]
            value = sum(
                (Fraction(a) - p) * h
                for a, p, h in zip(edge_values, edge_rates, weights)
            )
            outcomes.append((probability, value))

        exact_mean = sum(p * value for p, value in outcomes)
        exact_variance = sum(
            p * (value - exact_mean) ** 2 for p, value in outcomes
        )
        self.assertEqual(result["mean"], exact_mean)
        self.assertEqual(result["variance"], exact_variance)
        self.assertEqual(result["norm_squared"], sum(v * v for v in direction))

    def test_cantelli_bound_is_exact_and_respects_strict_threshold(self):
        self.assertEqual(
            exact_cantelli_lower(Fraction(3), Fraction(4), Fraction(1), Fraction(1)),
            Fraction(1, 2),
        )
        self.assertEqual(
            exact_cantelli_lower(Fraction(1), Fraction(0), Fraction(1), Fraction(1)),
            Fraction(0),
        )
        self.assertEqual(
            exact_cantelli_lower(Fraction(-3), Fraction(4), Fraction(1), Fraction(1)),
            Fraction(1, 2),
        )

    def test_group_state_probability_uses_exact_inputs(self):
        P = [
            [Fraction(19, 20), Fraction(1, 20)],
            [Fraction(17, 20), Fraction(3, 20)],
        ]
        self.assertEqual(exact_group_probability(P, (0, 1), (0, 1)),
                         Fraction(57, 400))


if __name__ == "__main__":
    unittest.main(verbosity=2)
