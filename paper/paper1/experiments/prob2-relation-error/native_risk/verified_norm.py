"""Sparse norm enclosures under IEEE binary64 round-to-nearest arithmetic.

Proof and operational assumptions: notes/native-risk/norm_enclosure.md.
Eigensolvers only propose a lower-bound witness. The upper bound uses a
positive weighted row sum, optionally applied to a verified matrix square.
"""

from fractions import Fraction

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


UNIT_ROUNDOFF = Fraction(1, 2**53)
ETA = Fraction(1, 2**1074)


def _up(value):
    return np.nextafter(value, np.inf)


def _down(value):
    return np.nextafter(value, -np.inf)


def fraction_upper(value):
    result = float(value)
    if not np.isfinite(result):
        raise ArithmeticError("rational conversion overflow")
    if Fraction.from_float(result) < value:
        result = float(_up(result))
    return result


def _gamma_sigma(operations):
    su = operations * UNIT_ROUNDOFF
    if su >= Fraction(1, 4):
        raise ArithmeticError("rounding operation budget exceeded")
    return (fraction_upper(su / (1 - su)),
            fraction_upper(operations * ETA / (1 - su)))


def _finite(value):
    if not np.all(np.isfinite(value)):
        raise ArithmeticError("overflow/nonfinite intermediate in norm enclosure")
    return value


def _point_csr(matrix):
    if not sparse.isspmatrix_csr(matrix) or matrix.dtype != np.float64:
        raise ValueError("a binary64 CSR point matrix is required")
    matrix = matrix.copy()
    matrix.sort_indices()
    if not matrix.has_canonical_format:
        raise ValueError("duplicate CSR storage must be assembled with an error budget")
    return matrix


def _arithmetic_probe():
    tiny = np.finfo(float).tiny / 2
    probe = sparse.diags([tiny], format="csr")
    identity = sparse.eye(1, format="csr")
    if (np.nextafter(0., 1.) * 1.0 == 0 or
            np.dot(np.array([tiny]), np.ones(1)) != tiny or
            (probe @ np.ones(1))[0] != tiny or
            (probe @ identity)[0, 0] != tiny):
        raise ArithmeticError("gradual-underflow probes failed")


def positive_matvec_upper(matrix, vector):
    """Upper enclosure of each exact nonnegative sparse dot product."""
    matrix = sparse.csr_matrix(matrix, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    if np.any(matrix.data < 0) or np.any(vector < 0):
        raise ValueError("nonnegative inputs required")
    _finite(matrix.data)
    _finite(vector)
    k = int(np.max(np.diff(matrix.indptr), initial=0))
    gamma, sigma = _gamma_sigma(2 * k + 2)
    value = _finite(matrix @ vector)
    return _finite(_up(_up(value + sigma) / _down(1.0 - gamma)))


def _positive_dot_upper(left, right):
    gamma, sigma = _gamma_sigma(2 * len(left) + 2)
    value = float(np.dot(left, right))
    return float(_finite(_up(_up(value + sigma) / _down(1.0 - gamma))))


def _weights(matrix):
    """Any strictly positive dyadic vector is valid; iteration only tightens it."""
    w = np.ones(matrix.shape[0])
    for _ in range(32):
        value = matrix @ w
        maximum = np.max(value, initial=0.0)
        if maximum == 0 or not np.isfinite(maximum):
            break
        w = np.maximum(value / maximum, 1e-6)
    return w


class ExactOperatorBuilder:
    """Build a symmetric float operator and bound its exact rational row error."""

    def __init__(self, n, edges, marginals):
        self.n = int(n)
        self.edges = [tuple(sorted(int(u) for u in e)) for e in edges]
        if self.n < 1 or len(self.edges) != len(marginals):
            raise ValueError("invalid node/edge dimensions")
        if any(len(e) < 2 or len(set(e)) != len(e) or min(e) < 0 or max(e) >= n
               for e in self.edges):
            raise ValueError("invalid edge slot")
        if any(not isinstance(p, (int, Fraction)) or p < 0 or p > 1
               for p in marginals):
            raise ValueError("exact rational probabilities required")
        rows, cols, slots, inc_rows, inc_cols = [], [], [], [], []
        self.sizes = np.array([len(e) for e in self.edges], dtype=float)
        self.terms = np.zeros((2, len(self.edges)))
        self.errors = np.zeros_like(self.terms)
        for j, (e, p) in enumerate(zip(self.edges, marginals)):
            p = Fraction(p)
            for b in (0, 1):
                exact = (b - p) / len(e)
                value = float(exact)
                self.terms[b, j] = value
                self.errors[b, j] = fraction_upper(abs(Fraction.from_float(value) - exact))
            for i, u in enumerate(e):
                inc_rows.append(u)
                inc_cols.append(j)
                for v in e[i:]:
                    rows.append(u)
                    cols.append(v)
                    slots.append(j)
        self.rows = np.array(rows, dtype=np.intp)
        self.cols = np.array(cols, dtype=np.intp)
        self.slots = np.array(slots, dtype=np.intp)
        self.incidence = sparse.csr_matrix(
            (np.ones(len(inc_rows)), (inc_rows, inc_cols)),
            shape=(n, len(self.edges)))
        self.max_degree = int(np.max(np.diff(self.incidence.indptr), initial=0))

    def build(self, indicators):
        indicators = np.asarray(indicators)
        if indicators.shape != (len(self.edges),) or np.any((indicators != 0) & (indicators != 1)):
            raise ValueError("one binary indicator per edge slot required")
        if not self.edges:
            return sparse.csr_matrix((self.n, self.n), dtype=float), 0.0
        index = np.arange(len(self.edges))
        terms = self.terms[indicators.astype(int), index]
        errors = self.errors[indicators.astype(int), index]
        upper = sparse.coo_matrix((terms[self.slots], (self.rows, self.cols)),
                                  shape=(self.n, self.n)).tocsr().tocoo()
        off = upper.row != upper.col
        # Mirror the already summed upper triangle, preserving bitwise symmetry.
        matrix = sparse.coo_matrix((np.concatenate((upper.data, upper.data[off])),
                                   (np.concatenate((upper.row, upper.col[off])),
                                    np.concatenate((upper.col, upper.row[off])))),
                                  shape=(self.n, self.n)).tocsr()
        gamma, sigma = _gamma_sigma(2 * self.max_degree + 2)
        absolute_rows = positive_matvec_upper(self.incidence, _up(self.sizes * abs(terms)))
        coefficient_rows = positive_matvec_upper(self.incidence, _up(self.sizes * errors))
        row_error = _up(_up(gamma * absolute_rows) + coefficient_rows)
        row_error = _up(row_error + _up(self.n * sigma))
        return matrix, float(np.max(row_error, initial=0.0))


def _rayleigh_lower(matrix, witness, assembly_error):
    q = np.asarray(witness, dtype=float)
    if q.shape != (matrix.shape[0],) or not np.all(np.isfinite(q)) or not np.any(q):
        raise ValueError("finite nonzero witness required")
    q = q / np.max(abs(q))
    y = _finite(matrix @ q)
    value = float(_finite(np.dot(q, y)))
    k = int(np.max(np.diff(matrix.indptr), initial=0))
    gamma_mv, sigma_mv = _gamma_sigma(2 * k + 2)
    row_magnitude = positive_matvec_upper(abs(matrix), abs(q))
    row_error = _up(_up(gamma_mv * row_magnitude) + sigma_mv)
    mv_error = _positive_dot_upper(abs(q), row_error)
    gamma_dot, sigma_dot = _gamma_sigma(2 * len(q) + 2)
    dot_magnitude = _positive_dot_upper(abs(q), abs(y))
    total_error = float(_up(_up(gamma_dot * dot_magnitude) + _up(mv_error + sigma_dot)))
    numerator_lower = max(0.0, float(_down(abs(value) - total_error)))
    norm_upper = _positive_dot_upper(abs(q), abs(q))
    return max(0.0, float(_down(_down(numerator_lower / norm_upper) - assembly_error)))


def enclose_norm(matrix, assembly_error=0.0, witness=None,
                 square_path_cap=20_000_000, relative_goal=0.002):
    """Return [lo,hi] for the exact symmetric target within assembly_error."""
    matrix = _point_csr(matrix)
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("nonempty square matrix required")
    if not np.all(np.isfinite(matrix.data)) or not np.isfinite(assembly_error) or assembly_error < 0:
        raise ValueError("finite matrix and nonnegative error required")
    if (matrix - matrix.T).nnz:
        raise ValueError("matrix must be exactly symmetric")
    _arithmetic_probe()
    if not matrix.nnz or np.max(abs(matrix.data), initial=0.0) == 0:
        return {"lo": 0.0, "hi": float(assembly_error), "power": 1,
                "assembly_error": float(assembly_error), "square_paths": 0}
    if witness is None:
        try:
            if matrix.shape[0] <= 8 or np.max(abs(matrix.data)) < 1e-200:
                values, vectors = np.linalg.eigh(matrix.toarray())
                witness = vectors[:, int(np.argmax(abs(values)))]
            else:
                v0 = np.random.default_rng(41011).normal(size=matrix.shape[0])
                _, vectors = eigsh(matrix, k=1, which="LM", v0=v0, tol=1e-9,
                                   maxiter=50000)
                witness = vectors[:, 0]
        except Exception:
            # A failed witness solve only weakens the lower endpoint.
            witness = np.zeros(matrix.shape[0])
            witness[int(np.argmax(abs(matrix.diagonal())))] = 1.0
    lo = _rayleigh_lower(matrix, witness, assembly_error)
    absolute = abs(matrix)
    w = _weights(absolute)
    hi = float(_up(np.max(_up(positive_matvec_upper(absolute, w) / w)) + assembly_error))
    power = 1
    counts = np.diff(matrix.indptr).astype(np.int64)
    paths = int(np.dot(counts, counts))
    if hi > (1 + relative_goal) * lo and paths <= square_path_cap:
        square = _finite_sparse(matrix @ matrix)
        w2 = _weights(abs(square))
        gamma, sigma = _gamma_sigma(2 * int(np.max(counts)) + 2)
        square_error = positive_matvec_upper(absolute,
                                            positive_matvec_upper(absolute, w2))
        row_upper = _up(positive_matvec_upper(abs(square), w2) + _up(gamma * square_error))
        sum_w = _positive_dot_upper(w2, np.ones(len(w2)))
        row_upper = _up(row_upper + _up(sigma * sum_w))
        squared_bound = float(np.max(_up(row_upper / w2)))
        hi2 = float(_up(_up(np.sqrt(squared_bound)) + assembly_error))
        if hi2 < hi:
            hi, power = hi2, 2
    if hi < lo:
        raise ArithmeticError("inconsistent norm enclosure")
    return {"lo": lo, "hi": hi, "power": power,
            "assembly_error": float(assembly_error), "square_paths": paths}


def _finite_sparse(matrix):
    _finite(matrix.data)
    return _point_csr(matrix)
