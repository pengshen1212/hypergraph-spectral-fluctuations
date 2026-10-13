"""Structure families and samplers for the influence validation (paper1).

Family A (amplification, non-star overlapping block):
    t node-disjoint blocks; block i has weak hub u_i (P[u_i,1]=eps),
    a pool of d/kappa strong co-members w (P[w,1]=1-eps, each reused in
    kappa edges), and fresh strong leaves. Edges {u_i, w_j, leaf}.
    phi_{u_i,e} = (1-eps)+(1-eps)-1 = 1-2eps on every incident edge.

Family B (competition sweep, fresh co-members):
    one hub u (P[u,1]=eps) with d edges {u, w_j, leaf_j}, all co-members
    fresh; w strongly class-1 (P=1-eps), leaf swept: P[leaf,1]=gamma.
    phi_{u,e} = (1-eps) + gamma - 1 = gamma - eps: zero at gamma=eps
    (exact second-order cancellation), max at gamma=1-eps.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as spsp
from scipy.sparse.linalg import eigsh

EPS_WEAK = 0.05
EPS_STRONG = 0.95


def build_family_A(t, d, kappa=5, seed=0):
    """Nodes: per block 1 hub + d/kappa pooled w + d leaves."""
    rng = np.random.default_rng(seed)
    edges, P, hubs = [], [], []
    base = 0
    for i in range(t):
        u = base
        nw = max(1, d // kappa)
        w_pool = list(range(base + 1, base + 1 + nw))
        leaf0 = base + 1 + nw
        block_edges = []
        for j in range(d):
            w = w_pool[rng.integers(nw)]
            leaf = leaf0 + j
            block_edges.append((u, int(w), leaf))
        edges.extend(block_edges)
        hubs.append(u)
        base = leaf0 + d
    n = base
    P = np.zeros((n, 2))
    P[:, 1] = EPS_STRONG
    for u in hubs:
        P[u] = [1 - EPS_WEAK, EPS_WEAK]
    return edges, P, hubs


def build_family_B(d, gamma, seed=0):
    """Single hub, fresh co-members; gamma = P[leaf,1]."""
    n = 1 + 2 * d
    edges = [(0, 1 + 2 * j, 2 + 2 * j) for j in range(d)]
    P = np.zeros((n, 2))
    P[0] = [1 - EPS_WEAK, EPS_WEAK]
    for j in range(d):
        P[1 + 2 * j] = [1 - EPS_STRONG, EPS_STRONG]  # w strong class-1
        P[2 + 2 * j] = [1 - gamma, gamma]            # swept leaf
    return edges, P, [0]


def edge_probabilities(edges, P):
    prod1 = np.array([np.prod(P[list(e), 1]) for e in edges])
    prod0 = np.array([np.prod(P[list(e), 0]) for e in edges])
    return prod1 + prod0


def phi_profile(edges, P):
    """phi_{u,e} for binary laws: g_u(1)-g_u(0) = prod P[v,1] - prod P[v,0]
    over v != u (sign convention: z=1 minus z=0)."""
    n = len(P)
    phi = {u: {} for u in range(n)}
    for e in edges:
        idx = list(e)
        p1 = np.prod(P[idx, 1])
        p0 = np.prod(P[idx, 0])
        for u in idx:
            pu1, pu0 = P[u, 1], P[u, 0]
            if pu1 > 0 and pu0 > 0:
                g1 = p1 / pu1
                g0 = p0 / pu0
            else:
                others = [v for v in idx if v != u]
                g1 = np.prod(P[others, 1]) if others else 1.0
                g0 = np.prod(P[others, 0]) if others else 1.0
            phi[u][e] = g1 - g0
    return phi


def influence_at_witness(edges, P, x):
    """Phi_u(x), V1(x) restricted to binary laws."""
    n = len(P)
    phi = phi_profile(edges, P)
    h = {}
    for e in edges:
        idx = list(e)
        h[e] = np.sum(x[idx]) ** 2 / len(e)
    Phi = np.zeros(n)
    V1 = 0.0
    for u in range(n):
        w1 = sum(phi[u][e] * h[e] for e in edges if u in e)
        # binary: psi_u(1)=P0*Phi_u, psi_u(0)=-P1*Phi_u
        Phi[u] = w1
        V1 += P[u, 0] * P[u, 1] * w1 ** 2
    return Phi, V1


def sample_weights_A(edges, P, rng):
    """a_e - p_e under independent node predictions."""
    preds = (rng.random(len(P)) < P[:, 1]).astype(int)
    w = []
    for e in edges:
        idx = list(e)
        same = len(set(preds[idx].tolist())) == 1
        w.append(1.0 if same else 0.0)
    w = np.array(w)
    return w - edge_probabilities(edges, P)


def sample_weights_B(edges, P, rng):
    pe = edge_probabilities(edges, P)
    b = (rng.random(len(edges)) < pe).astype(float)
    return b - pe


def build_operator(n, edges, w):
    rows, cols, vals = [], [], []
    for j, e in enumerate(edges):
        for a_ in e:
            for b_ in e:
                rows.append(a_)
                cols.append(b_)
                vals.append(w[j] / len(e))
    return spsp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def beta_centered(n, edges, w, tol=1e-6):
    M = build_operator(n, edges, w)
    if M.nnz == 0:
        return 0.0
    try:
        la = eigsh(M, k=1, which="LA", tol=tol, maxiter=50000,
                   return_eigenvectors=False)[0]
        li = eigsh(M, k=1, which="SA", tol=tol, maxiter=50000,
                   return_eigenvectors=False)[0]
    except Exception:
        la = eigsh(M, k=1, which="LA", tol=1e-4, maxiter=100000,
                   return_eigenvectors=False)[0]
        li = eigsh(M, k=1, which="SA", tol=1e-4, maxiter=100000,
                   return_eigenvectors=False)[0]
    return float(max(abs(la), abs(li)))


def quadratic_at(edges, w, x):
    s = 0.0
    for j, e in enumerate(edges):
        idx = list(e)
        s += w[j] * np.sum(x[idx]) ** 2 / len(e)
    return float(s)


def family_A_realized_lambda2(edges, hubs, block_size):
    """Exact Lambda^2 = max_i sum_{u != u_i} L_u(x_i)^2 from the realized
    structure (family A edges are (u, w, leaf): pooled w contributes
    reuse_w/3, fresh leaf contributes 1/3)."""
    lam2 = 0.0
    for u in hubs:
        lu = {}
        for e in edges:
            if e[0] != u:
                continue
            lu[e[1]] = lu.get(e[1], 0.0) + 1.0 / 3.0
            lu[e[2]] = lu.get(e[2], 0.0) + 1.0 / 3.0
        lam2 = max(lam2, sum(v * v for v in lu.values()))
    return lam2


def theory_bounds_family_A(t, d, kappa, lam2=None, eps=EPS_WEAK, eta=0.01,
                           eta2=0.01, delta=0.01):
    """T4 certificate pieces for family A with witness x=e_{u_i}.

    lam2: exact realized co-occurrence budget (preferred). If None, the
    uniform-reuse idealization (d/kappa)(kappa/3)^2 + d/9 is used, which
    underestimates the budget when pool reuse is random-with-replacement
    (Jensen gap on sum of squared reuse counts)."""
    n = t * (1 + max(1, d // kappa) + d)
    phi_u = (1 - eps) + (1 - eps) - 1
    Phi_hub = d * phi_u / 3.0
    drift = (1 - eps) * abs(Phi_hub)
    V_minus = 0.0                      # all co-members flat by design
    phi_max_minus = 0.0
    if lam2 is None:
        lam2 = (d // kappa) * (kappa / 3.0) ** 2 + d * (1 / 3.0) ** 2
    Lam2 = lam2
    pe = eps * (1 - eps)               # per-edge retention mean
    # v_B: norm of sum pe(1-pe) H_e; per block ~ pe*(d+2)/3 (disjoint blocks)
    v_B = pe * (d + 2) / 3.0
    Ln = np.log(2 * n / delta)
    U_B = np.sqrt(2 * v_B * Ln) + 2 * Ln / 3
    # two-sided allowances use log(2/eta) per the corrected theorem
    U_rest = (np.sqrt(2 * V_minus * np.log(2 / eta))
              + phi_max_minus / 3 * np.log(2 / eta)
              + np.sqrt(2 * Lam2 * np.log(2 / eta2)))
    L_A = drift - U_rest
    return dict(n=n, Phi_hub=Phi_hub, drift=drift, Lam2=Lam2, U_rest=U_rest,
                U_B=U_B, L_A=L_A, certified=bool(L_A > U_B),
                sep_prob_lb=1 - (1 - eps) ** t - eta - eta2 - delta)
