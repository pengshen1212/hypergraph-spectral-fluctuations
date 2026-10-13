"""Enclosure regressions against exact rational and dense reference calculations."""

import unittest
from fractions import Fraction
from unittest.mock import patch

import numpy as np
from scipy import sparse

from verified_norm import ExactOperatorBuilder, enclose_norm, positive_matvec_upper


class VerifiedNormTests(unittest.TestCase):
    def test_positive_sparse_product_is_outward_bounded(self):
        a = sparse.csr_matrix([[0.1, 0.3], [1e-300, 1e-300]])
        w = np.array([0.7, 1e-20])
        upper = positive_matvec_upper(a, w)
        for i in range(2):
            exact = sum(Fraction.from_float(float(a[i, j])) *
                        Fraction.from_float(float(w[j])) for j in range(2))
            self.assertGreaterEqual(Fraction.from_float(float(upper[i])), exact)

    def test_hidden_extreme_is_covered_even_with_bad_witness(self):
        a = sparse.diags([100.0, -120.0, 1.0], format="csr")
        result = enclose_norm(a, witness=np.array([0., 0., 1.]))
        self.assertLessEqual(result["lo"], 120)
        self.assertGreaterEqual(result["hi"], 120)
        self.assertLess(result["lo"], 2)

    def test_signed_matrix_and_square_fill_fallback(self):
        a = sparse.csr_matrix([[2., -3., 0.], [-3., -4., 1.], [0., 1., 1.]])
        target = max(abs(np.linalg.eigvalsh(a.toarray())))
        for cap in (0, 10000):
            result = enclose_norm(a, square_path_cap=cap)
            self.assertLessEqual(result["lo"], target + 1e-13)
            self.assertGreaterEqual(result["hi"], target - 1e-13)
        self.assertEqual(enclose_norm(a, square_path_cap=0)["power"], 1)

    def test_exact_rational_operator_error_and_duplicate_slots(self):
        edges = [(0, 1), (0, 1), (1, 2, 3)]
        p = [Fraction(13, 100), Fraction(13, 100), Fraction(17, 1000)]
        builder = ExactOperatorBuilder(4, edges, p)
        for indicators in ([0, 1, 0], [1, 1, 1], [0, 0, 0]):
            a, error = builder.build(indicators)
            exact = [[Fraction(0) for _ in range(4)] for _ in range(4)]
            for e, pe, b in zip(edges, p, indicators):
                for i in e:
                    for j in e:
                        exact[i][j] += (b - pe) / len(e)
            row_error = max(sum(abs(exact[i][j] - Fraction.from_float(float(a[i, j])))
                                for j in range(4)) for i in range(4))
            self.assertGreaterEqual(Fraction.from_float(error), row_error)
            result = enclose_norm(a, assembly_error=error)
            target = max(abs(np.linalg.eigvalsh(np.array(exact, dtype=float))))
            self.assertLessEqual(result["lo"], target + 1e-13)
            self.assertGreaterEqual(result["hi"], target - 1e-13)

    def test_zero_subnormal_and_invalid_inputs(self):
        result = enclose_norm(sparse.csr_matrix((3, 3)))
        self.assertEqual(result["lo"], 0.)
        self.assertEqual(result["hi"], 0.)
        a = sparse.diags([np.nextafter(0., 1.), 0.], format="csr")
        self.assertGreaterEqual(enclose_norm(a)["hi"], np.nextafter(0., 1.))
        with self.assertRaises(ValueError):
            enclose_norm(sparse.csr_matrix([[0., 1.], [0., 0.]]))
        with self.assertRaises(ValueError):
            enclose_norm(sparse.diags([np.inf]))

    def test_upper_bound_passes_exact_positive_definiteness_check(self):
        rng = np.random.default_rng(8091)
        for _ in range(12):
            integers = rng.integers(-50, 51, size=(6, 6))
            a = (integers + integers.T) / 128.0
            result = enclose_norm(sparse.csr_matrix(a))
            t = Fraction.from_float(result["hi"])
            for sign in (-1, 1):
                shifted = [[(t if i == j else Fraction(0)) -
                            sign * Fraction.from_float(float(a[i, j]))
                            for j in range(6)] for i in range(6)]
                for k in range(6):
                    pivot = shifted[k][k]
                    self.assertGreater(pivot, 0)
                    for i in range(k + 1, 6):
                        for j in range(k + 1, 6):
                            shifted[i][j] -= shifted[i][k] * shifted[k][j] / pivot

    def test_lower_endpoint_is_below_exact_rational_rayleigh(self):
        a = np.array([[1., -7., 2.], [-7., 5., 3.], [2., 3., -1.]]) / 32
        q = np.array([.31, -.67, .19])
        result = enclose_norm(sparse.csr_matrix(a), witness=q)
        q = q / max(abs(q))
        qr = [Fraction.from_float(float(x)) for x in q]
        rayleigh = abs(sum(qr[i] * Fraction.from_float(float(a[i, j])) * qr[j]
                          for i in range(3) for j in range(3))) / sum(x*x for x in qr)
        self.assertLessEqual(Fraction.from_float(result["lo"]), rayleigh)

    def test_integer_marginals_remain_exact(self):
        builder = ExactOperatorBuilder(3, [(0, 1, 2)], [0])
        a, error = builder.build([1])
        actual = abs(Fraction.from_float(float(a[0, 0])) - Fraction(1, 3)) * 3
        self.assertGreaterEqual(Fraction.from_float(error), actual)

    def test_duplicate_csr_storage_is_rejected_without_mutation(self):
        a = sparse.csr_matrix(
            (np.tile([0.5, 2.**-60, -0.5], 4),
             np.array([0, 0, 0, 1, 1, 1] * 2), np.array([0, 6, 12])),
            shape=(2, 2))
        original = a.data.copy()
        with self.assertRaises(ValueError):
            enclose_norm(a, witness=np.array([1., 0.]))
        np.testing.assert_array_equal(original, a.data)

    def test_dense_eigensolve_failure_keeps_valid_interval(self):
        a = sparse.diags([1., -2.], format="csr")
        with patch("verified_norm.np.linalg.eigh", side_effect=np.linalg.LinAlgError):
            result = enclose_norm(a)
        self.assertLessEqual(result["lo"], 2)
        self.assertGreaterEqual(result["hi"], 2)

    def test_empty_builder_is_exactly_zero(self):
        matrix, error = ExactOperatorBuilder(2, [], []).build([])
        self.assertEqual(error, 0)
        self.assertEqual(matrix.nnz, 0)
        result = enclose_norm(matrix, assembly_error=error)
        self.assertEqual(result["hi"], 0)


if __name__ == "__main__":
    unittest.main()
