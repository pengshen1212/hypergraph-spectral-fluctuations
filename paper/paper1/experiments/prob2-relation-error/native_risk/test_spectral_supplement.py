from fractions import Fraction
import itertools
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from exact_certificate import exact_directional_moments
from spectral_supplement import supported_witness


class SupportedSupplementTests(unittest.TestCase):
    def test_large_embedding_has_exact_zero_tail(self):
        matrix = csr_matrix(([4.0, 1.0, 1.0, -2.0],
                             ([37, 37, 901, 901], [37, 901, 37, 901])),
                            shape=(20000, 20000))
        witness = supported_witness(matrix)
        self.assertEqual(set(np.flatnonzero(witness["x"])), {37, 901})
        self.assertLess(witness["residual"], 1e-12)

    def test_multiclass_duplicate_slots_match_complete_enumeration(self):
        P = [[Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)],
             [Fraction(1, 3), Fraction(1, 6), Fraction(1, 2)],
             [Fraction(1, 6), Fraction(1, 2), Fraction(1, 3)]]
        edges = [(0, 1), (0, 1), (1, 2), (0, 1, 2)]
        q = [Fraction(1, 2), Fraction(-1, 4), Fraction(3, 8)]
        marginals = [sum((P[e[0]][c] * P[e[1]][c] *
                         (P[e[2]][c] if len(e) == 3 else 1)
                         for c in range(3)), Fraction(0)) for e in edges]
        weights = [sum((q[u] for u in e), Fraction(0)) ** 2 / len(e) for e in edges]
        for fixed in range(3):
            outcomes = []
            for a, b in itertools.product(range(3), repeat=2):
                labels = (fixed, a, b)
                value = sum(((int(len({labels[u] for u in e}) == 1) - p) * w
                             for e, p, w in zip(edges, marginals, weights)), Fraction(0))
                outcomes.append((P[1][a] * P[2][b], value))
            mean = sum((p * x for p, x in outcomes), Fraction(0))
            variance = sum((p * (x - mean) ** 2 for p, x in outcomes), Fraction(0))
            moments = exact_directional_moments(edges, P, (0,), (fixed,), q)
            self.assertEqual(moments["mean"], mean)
            self.assertEqual(moments["variance"], variance)

    def test_zero_mean_gets_valid_nonzero_fallback(self):
        witness = supported_witness(csr_matrix((12, 12)))
        self.assertEqual(witness["x"][0], 1.0)
        self.assertEqual(np.count_nonzero(witness["x"]), 1)


if __name__ == "__main__":
    unittest.main()
