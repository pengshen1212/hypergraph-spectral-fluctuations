"""Core library implementing docs/operator_contract.md (v1.0).

Operators (fixed w_e = 1, kappa = 1):

    B(a)   = sum_e a_e 1_e 1_e^T / |e|
    A(a)   = B(a) + kappa I
    D(a)   = diag(A(a) 1)            recomputed for every draw
    S(a)   = D^{-1/2} A D^{-1/2}
    S0(a)  = D0^{-1/2} A(a) D0^{-1/2},  D0 = diag((E B + kappa I) 1)

All sparse structures use a FIXED (row, col) pattern derived from the clique
expansion (including self-loop positions), so every operator is a data array
over the same pattern; averages (centers) are exact elementwise means.

Numerical note: spectral norms are two-sided (max |lambda|); dense eigvalsh
below DENSE_LIMIT nodes, otherwise scipy eigsh with residuals recorded.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.stats import beta as _beta

DENSE_LIMIT = 600
EIGSH_TOL = 1e-8


class RelationOperator:
    """Fixed candidate hypergraph with precomputed clique-expansion pattern."""

    def __init__(self, n, edges, kappa=1.0):
        self._v0 = None  # deterministic eigsh start vector (set lazily)
        edges = tuple(tuple(int(u) for u in e) for e in edges)
        for e in edges:
            if len(e) < 2:
                raise ValueError("each hyperedge slot needs >=2 nodes")
            if len(set(e)) != len(e):
                raise ValueError("nodes inside a slot must be distinct")
            if min(e) < 0 or max(e) >= int(n):
                raise ValueError("hyperedge node outside graph")
        self.n = int(n)
        self.kappa = float(kappa)
        self.edges = edges
        self.m = len(edges)
        self.sizes = np.array([len(e) for e in edges], dtype=np.int64)

        # per-node agreement evaluation: flat membership + segment starts
        self.flat = np.concatenate(
            [np.asarray(e, dtype=np.int64) for e in edges]
        ) if self.m else np.zeros(0, dtype=np.int64)
        self.starts = np.concatenate(
            [[0], np.cumsum(self.sizes)]
        )[:-1].astype(np.int64) if self.m else np.zeros(0, dtype=np.int64)

        # incidence matrix M (n x m): deg(u) = kappa + (M @ a)[u]
        if self.m:
            edge_of_member = np.repeat(np.arange(self.m), self.sizes)
            self.incidence = sp.csr_matrix(
                (np.ones(len(self.flat)), (self.flat, edge_of_member)),
                shape=(self.n, self.m),
            )
        else:
            self.incidence = sp.csr_matrix((self.n, 0))

        # clique term expansion: ordered pairs (u, v) per edge, incl. diagonal
        term_rows, term_cols, term_edge, term_scale = self._expand_terms()
        keys = term_rows * self.n + term_cols
        entries, self.entry_of_term = np.unique(keys, return_inverse=True)
        self.erows = (entries // self.n).astype(np.int64)
        self.ecols = (entries % self.n).astype(np.int64)
        self.n_entries = len(entries)
        self.diag_ids = np.flatnonzero(self.erows == self.ecols)
        self.term_edge = term_edge
        self.term_scale = term_scale

    def _expand_terms(self):
        rows, cols, eidx, scale = [], [], [], []
        for i, e in enumerate(self.edges):
            r = len(e)
            arr = np.asarray(e, dtype=np.int64)
            rows.append(np.repeat(arr, r))
            cols.append(np.tile(arr, r))
            eidx.append(np.full(r * r, i))
            scale.append(np.full(r * r, 1.0 / r))
        if not rows:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty.copy(), empty.copy(), np.zeros(0)
        return (np.concatenate(rows), np.concatenate(cols),
                np.concatenate(eidx), np.concatenate(scale))

    # ------------------------------------------------------------------
    # flags and marginals
    # ------------------------------------------------------------------
    def flags_from_Y(self, Y):
        """a_e = 1{all predictions on edge e agree}, binary class labels."""
        lab = np.asarray(Y)[self.flat]
        mn = np.minimum.reduceat(lab, self.starts)
        mx = np.maximum.reduceat(lab, self.starts)
        return (mn == mx).astype(float)

    def edge_marginals(self, P):
        """p_e = sum_c prod_{u in e} P[u, c] (exact)."""
        P = np.asarray(P, dtype=float)
        out = np.zeros(self.m)
        for i, e in enumerate(self.edges):
            out[i] = float(np.prod(P[list(e)], axis=0).sum())
        return out

    # ------------------------------------------------------------------
    # operator data arrays over the fixed pattern
    # ------------------------------------------------------------------
    def degrees(self, a):
        return self.kappa + np.asarray(self.incidence @ np.asarray(a)).ravel()

    def raw_b_data(self, a):
        return np.bincount(
            self.entry_of_term,
            weights=np.asarray(a)[self.term_edge] * self.term_scale,
            minlength=self.n_entries,
        )

    def A_data(self, a):
        vals = self.raw_b_data(a)
        vals[self.diag_ids] += self.kappa
        return vals

    def S_data(self, a):
        vals = self.A_data(a)
        di = 1.0 / np.sqrt(self.degrees(a))
        return vals * di[self.erows] * di[self.ecols]

    def S0_inverse_degrees(self, p):
        """d0 = (E B + kappa I) 1 (deterministic); returns 1/sqrt(d0)."""
        d0 = self.degrees(p)
        return 1.0 / np.sqrt(d0)

    def S0_data(self, a, di0):
        return self.A_data(a) * di0[self.erows] * di0[self.ecols]

    # ------------------------------------------------------------------
    # linearization of dynamic normalization around a reference point (S9a)
    # ------------------------------------------------------------------
    def response_basis(self, p):
        """Base operator at expected flags p: S0 data + inverse degrees."""
        p = np.asarray(p, dtype=float)
        di0 = self.S0_inverse_degrees(p)
        s0 = self.S0_data(p, di0)
        d0 = self.degrees(p)
        return {"di0": di0, "S0": s0, "d0": d0, "p": p,
                "EB": self.raw_b_data(p)}

    def linear_response(self, a, basis):
        """Split S(a) - S0 into relation / degree / linear / residual parts.

        With E = B(a) - B(p), C = D0^{-1} diag(E 1):

            J_A(E) = D0^{-1/2} E D0^{-1/2} - (C S0 + S0 C)/2

        Returns (R1, Rdeg, J, true, resid) data arrays over the fixed
        pattern; Rdeg carries the negative sign as it enters J.
        Uniform-scaling identity (E = alpha A0 in the pure-normalization
        sense, i.e. degree shift consistent with the entry shift) yields
        J = alpha S0 - (alpha S0 + alpha S0)/2 = 0 — see tests.
        """
        di0, s0, d0 = basis["di0"], basis["S0"], basis["d0"]
        e_data = self.raw_b_data(a) - basis["EB"]
        c = (self.degrees(a) - d0) / d0           # diag(C) = E1 / d0
        r1 = e_data * di0[self.erows] * di0[self.ecols]
        rdeg = -0.5 * (c[self.erows] + c[self.ecols]) * s0
        j = r1 + rdeg
        true = self.S_data(a) - s0
        return r1, rdeg, j, true, true - j

    def EB_data(self, p):
        return np.bincount(
            self.entry_of_term,
            weights=np.asarray(p)[self.term_edge] * self.term_scale,
            minlength=self.n_entries,
        )

    # ------------------------------------------------------------------
    # csr assembly
    # ------------------------------------------------------------------
    def csr(self, data):
        return sp.csr_matrix((data, (self.erows, self.ecols)),
                             shape=(self.n, self.n))

    def B_csr(self, a):
        return self.csr(self.raw_b_data(a))

    def S_csr(self, a):
        return self.csr(self.S_data(a))

    # ------------------------------------------------------------------
    # exact expectations via enumeration (small scale only)
    # ------------------------------------------------------------------
    def exact_ES_node(self, P, max_states=200_000):
        """Exact E S under independent node law, by full state enumeration."""
        n, C = P.shape
        total = C ** n
        if total > max_states:
            raise ValueError(f"enumeration too large: {total} states")
        acc = np.zeros(self.n_entries)
        states = itertools.product(range(C), repeat=n)
        probs = []
        for state in states:
            p = 1.0
            for u, c in enumerate(state):
                p *= P[u, c]
            probs.append(p)
        probs = np.asarray(probs)
        keep = probs > 0
        idxs = np.flatnonzero(keep)
        all_states = list(itertools.product(range(C), repeat=n))
        for j in idxs:
            state = all_states[j]
            a = self.flags_from_Y(state)
            acc += probs[j] * self.S_data(a)
        return acc

    def exact_ES_edge(self, p, max_states=65_536):
        """Exact E S under the independent-edge arm (2^m enumeration)."""
        m = self.m
        if 2 ** m > max_states:
            raise ValueError(f"edge enumeration too large: 2^{m}")
        acc = np.zeros(self.n_entries)
        for bits in itertools.product((0.0, 1.0), repeat=m):
            a = np.asarray(bits)
            prob = 1.0
            for i in range(m):
                prob *= a[i] * p[i] + (1 - a[i]) * (1 - p[i])
            if prob == 0.0:
                continue
            acc += prob * self.S_data(a)
        return acc

    def exact_conditional_ES_node(self, P, group, state, max_states=200_000):
        """Exact E[S | Y_group = state] and the conditioning probability."""
        n, C = P.shape
        group = tuple(int(u) for u in group)
        state = tuple(int(c) for c in state)
        acc = np.zeros(self.n_entries)
        prob_sum = 0.0
        for st in itertools.product(range(C), repeat=n):
            if any(st[u] != c for u, c in zip(group, state)):
                continue
            p = 1.0
            for u in range(n):
                p *= P[u, st[u]]
            if p == 0.0:
                continue
            acc += p * self.S_data(self.flags_from_Y(st))
            prob_sum += p
        if prob_sum == 0.0:
            raise ValueError("conditioning event has probability zero")
        return acc / prob_sum, prob_sum

    def dense(self, data):
        return self.csr(data).toarray()


# ----------------------------------------------------------------------
# sampling
# ----------------------------------------------------------------------
def sample_Y(P, rng):
    """Independent node predictions from row-stochastic P (n x C)."""
    P = np.asarray(P, dtype=float)
    n, C = P.shape
    cdf = np.cumsum(P, axis=1)
    u = rng.random((n, 1))
    return np.clip((u >= cdf).sum(axis=1), 0, C - 1)


def node_flags(op, P, rng):
    Y = sample_Y(P, rng)
    return op.flags_from_Y(Y), Y


def edge_flags_iid(p, rng):
    return (rng.random(len(p)) < p).astype(float)


# ----------------------------------------------------------------------
# spectral norms
# ----------------------------------------------------------------------
def _start_vector(op):
    if op._v0 is None:
        op._v0 = np.random.default_rng(20261021).standard_normal(op.n)
    return op._v0


def spectral_norm_entries(op, data_a, data_b, dense_limit=DENSE_LIMIT):
    """Two-sided ||M(a) - M(b)||_2 over a fixed pattern; returns (norm, resid).

    Deterministic: dense eigvalsh for n <= dense_limit; otherwise eigsh with a
    FIXED start vector. On non-convergence the fallback is the Frobenius upper
    bound with resid=inf (the score then estimates ||.||_F for that draw and
    must be flagged approximate by the caller; calibration and evaluation must
    share one deterministic approximation pipeline).
    """
    if op.n <= dense_limit:
        diff = op.csr(data_a - data_b).toarray()
        return float(np.max(np.abs(np.linalg.eigvalsh(diff)))), 0.0
    diff = op.csr(data_a - data_b)
    v0 = _start_vector(op)
    frob = float(np.linalg.norm(data_a - data_b))
    best, worst_resid = -np.inf, 0.0
    big = op.n > 20000
    for which in ("LA", "SA"):
        maxiter = 300 if big else min(1500, max(3 * op.n, 600))
        tol = 3e-6 if big else EIGSH_TOL
        try:
            vals, vecs = eigsh(diff, k=1, which=which, tol=tol,
                               maxiter=maxiter, v0=v0)
            lam = float(vals[0])
            x = np.asarray(vecs[:, 0]).ravel()
            resid = float(np.linalg.norm(diff @ x - lam * x))
        except Exception as exc:
            lam_vals = getattr(exc, "eigenvalues", None)
            if lam_vals is None or len(lam_vals) == 0:
                return frob, math.inf
            lam = float(lam_vals[0])
            resid = math.inf
        worst_resid = max(worst_resid, resid)
        best = max(best, abs(lam))
    return float(best), worst_resid


def spectral_norm_matrix(op, data):
    return spectral_norm_entries(op, data, np.zeros_like(data))


# ----------------------------------------------------------------------
# thresholding (contract section 5)
# ----------------------------------------------------------------------
def rank_threshold(scores, alpha):
    """Ascending-order rule: k = ceil((M+1)(1-alpha)); q = T_(k) or +inf."""
    M = len(scores)
    k = math.ceil((M + 1) * (1.0 - alpha))
    if k > M:
        return math.inf
    asc = np.sort(np.asarray(scores))
    return float(asc[k - 1])


def clopper_pearson(k, n, conf=0.95):
    """Two-sided Clopper-Pearson interval for exceedance rate k/n."""
    if n <= 0:
        return (math.nan, math.nan)
    a = (1.0 - conf) / 2.0
    lo = 0.0 if k == 0 else float(_beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta.ppf(1.0 - a, k + 1, n - k))
    return lo, hi


# ----------------------------------------------------------------------
# calibration batch driver (contract section 5)
# ----------------------------------------------------------------------
def draw_scores(op, sampler, count, rng, center_data):
    """Scores T_b = ||S(draw) - center||_2; sampler(rng) -> (a_flags, extra)."""
    scores = np.empty(int(count))
    resids = np.empty(int(count))
    for b in range(int(count)):
        a, _ = sampler(rng)
        s = op.S_data(a)
        scores[b], resids[b] = spectral_norm_entries(op, s, center_data)
    return scores, resids


def estimate_center(op, sampler, count, rng):
    """Frozen reference mean M_hat as elementwise data mean over the pattern."""
    acc = np.zeros(op.n_entries)
    for b in range(int(count)):
        a, _ = sampler(rng)
        acc += op.S_data(a)
    return acc / float(count)


def bound_node_sampler(op, P):
    def sampler(rng):
        return node_flags(op, P, rng)
    return sampler


def bound_edge_sampler(op, p):
    def sampler(rng):
        a = edge_flags_iid(p, rng)
        return a, None
    return sampler
