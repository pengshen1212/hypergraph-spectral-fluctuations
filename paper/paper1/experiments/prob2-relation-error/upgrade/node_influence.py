"""Node-influence decomposition for centered hypergraph relation operators.

Model: independent node predictions with per-node categorical laws P[u, c].
For each candidate edge e (size r_e >= 2, arbitrary overlap):

    a_e = 1{all predictions on e agree},  p_e = E a_e,
    X_A = sum_e (a_e - p_e) H_e,  H_e = 1_e 1_e^T / r_e   (centered model A)
    X_B = sum_e (b_e - p_e) H_e,  b_e ~ Bern(p_e) iid     (matched model B)

First-order (leave-one-out) decomposition (proof_targets.md T1-T3):

    g_{e,u}(z) = E[a_e | pred_u = z] = prod_{v in e \\ {u}} P[v, z]
    F_u(z)     = sum_{e contains u} (g_{e,u}(z) - p_e) H_e
    X_A        = sum_u F_u(pred_u) + R,   E[R | pred_u] = 0 for all u
    V_1(x)     = sum_u Var_{Z ~ P[u,.]} sum_{e contains u} g_{e,u}(Z) h_e(x)
               = sum_u P[u,1] P[u,0] Phi_u(x)^2          (binary classes)
    Phi_u(x)   = sum_{e contains u} phi_{u,e} h_e(x),
    phi_{u,e}  = g_{e,u}(1) - g_{e,u}(0),  h_e(x) = (sum_{u in e} x_u)^2/r_e

Small-scale exhaustive checks are exact (no sampling). Native-scale
quantities use per-edge / per-node sparse accumulation only.
"""

from __future__ import annotations

import numpy as np


def retention_probability(node_probs):
    """p_e = sum_c prod_{u in e} P[u, c] for one edge's probability rows."""
    node_probs = np.asarray(node_probs, dtype=float)
    return float(np.prod(node_probs, axis=0).sum())


def conditional_retention(other_node_probs, center_class):
    """E[a_e | center predicts center_class] = prod P[v, center_class]
    over the remaining nodes (no division: zero-probability classes safe)."""
    other = np.asarray(other_node_probs, dtype=float)
    return float(np.prod(other[:, center_class]))


def edge_rates(edges, P):
    """Per-edge p_e and conditional kernels g_{e,u}(z)."""
    rates = {}
    for e in edges:
        idx = list(e)
        pe = retention_probability(P[idx])
        gs = {u: np.array([conditional_retention(
            P[[v for v in idx if v != u]], z)
            for z in range(P.shape[1])]) for u in idx}
        rates[e] = (pe, gs)
    return rates


def first_order_components(edges, P):
    """F_u(z) as dense (n, n) matrices, indexed [u][z]. Small scale only."""
    n, C = P.shape
    rates = edge_rates(edges, P)
    comps = []
    for u in range(n):
        Fz = [np.zeros((n, n)) for _ in range(C)]
        for e in edges:
            if u not in e:
                continue
            pe, gs = rates[e]
            idx = list(e)
            for z in range(C):
                Fz[z][np.ix_(idx, idx)] += (gs[u][z] - pe) / len(e)
        comps.append(Fz)
    return comps


def h_of_edge(e, x):
    idx = list(e)
    return float(np.sum(x[idx]) ** 2 / len(e))


def influence_profile(edges, P, x):
    """(Phi_u(x) array, V_1(x)).

    Phi is returned for binary problems (z-sensitivity contrast); V_1 is
    computed in the class-general form sum_u Var_Z[sum g h].
    """
    n, C = P.shape
    x = np.asarray(x, dtype=float)
    rates = edge_rates(edges, P)
    Phi = np.zeros(n)
    V1 = 0.0
    for u in range(n):
        w = np.zeros(C)  # w[z] = sum_{e contains u} g_{e,u}(z) h_e
        for e in edges:
            if u not in e:
                continue
            pe, gs = rates[e]
            he = h_of_edge(e, x)
            w += gs[u] * he
        mean = float(w @ P[u])
        V1 += float(((w - mean) ** 2) @ P[u])
        if C == 2:
            Phi[u] = w[1] - w[0]
    return Phi, V1


def quadratic_form(edges, weights, x):
    """x^T (sum_e weights_e H_e) x for given per-edge weights."""
    s = 0.0
    for e, w in zip(edges, weights):
        idx = list(e)
        s += w * np.sum(x[idx]) ** 2 / len(e)
    return float(s)


def exhaustive_decomposition(edges, P, x=None):
    """Exact enumeration of all prediction configurations (binary laws or
    general C; 2^n / C^n configs). Checks the T1-T3 identities and returns
    the pieces needed by the regression tests."""
    n, C = P.shape
    edges = [tuple(e) for e in edges]
    rates = edge_rates(edges, P)
    comps = first_order_components(edges, P)
    N = n
    EX = np.zeros((N, N))
    ER = {}
    EFu = [np.zeros((N, N)) for _ in range(n)]
    EFuFv = np.zeros((n, n))
    EFuR = np.zeros(n)
    Ea = np.zeros(len(edges))
    xa, xr, xf = [], [], []
    for assign in itertools.product(range(C), repeat=n):
        assign = np.array(assign)
        prob = float(np.prod([P[u, assign[u]] for u in range(n)]))
        if prob == 0.0:
            continue
        a = np.array([1.0 if len(set(assign[list(e)].tolist())) == 1
                      else 0.0 for e in edges])
        Ea += prob * a
        XA = np.zeros((N, N))
        for e, ae in zip(edges, a):
            pe, _ = rates[e]
            XA[np.ix_(list(e), list(e))] += (ae - pe) / len(e)
        Fs = [comps[u][assign[u]] for u in range(n)]
        R = XA - sum(Fs)
        EX += prob * XA
        EFu += [prob * F for F in Fs]
        for u in range(n):
            EFuFv[u, :] += [prob * float(np.sum(Fs[u] * Fs[v]))
                            for v in range(n)]
            EFuR[u] += prob * float(np.sum(Fs[u] * R))
            key = (u, int(assign[u]))
            ER[key] = ER.get(key, np.zeros((N, N))) + prob * R
        if x is not None:
            xa.append((prob, quadratic_form(edges, a - _p_vec(edges,
                                                              rates), x)))
            xf.append((prob, sum(quadratic_form_single(
                Fs[u], x) for u in range(n))))
            xr.append((prob, quadratic_form_single(R, x)))
    out = dict(EX=EX,
               ER_given=np.array([[ER.get((u, z), np.zeros((N, N)))
                                   for z in range(C)] for u in range(n)]),
               cross_F=EFuFv - np.diag(np.diag(EFuFv)),
               FR=EFuR,
               EFu=EFu,
               pa_err=np.abs(Ea - np.array([rates[e][0]
                                            for e in edges])))
    if x is not None:
        def moment(pairs, k=2):
            return sum(p * v ** k for p, v in pairs)
        out["varA"] = moment(xa)
        out["V1"] = moment(xf)
        out["varR"] = moment(xr)
        Phi, V1_profile = influence_profile(edges, P, x)
        out["V1_profile"] = V1_profile
        out["Phi"] = Phi
    return out


def quadratic_form_single(M, x):
    return float(x @ M @ x)


def _p_vec(edges, rates):
    return np.array([rates[e][0] for e in edges])


import itertools  # noqa: E402  (kept at bottom to mirror usage scope)


def main():
    """Tiny smoke demo of the decomposition pieces."""
    edges = [(0, 1, 2), (0, 3), (1, 2)]
    P = np.array([[0.8, 0.2], [0.6, 0.4], [0.9, 0.1], [0.7, 0.3]])
    x = np.array([1.0, 0.5, -0.5, 0.0])
    res = exhaustive_decomposition(edges, P, x=x)
    print("E[X_A] max abs:", np.abs(res["EX"]).max())
    print("E[R|pred_u] max abs:", np.abs(res["ER_given"]).max())
    print("cross_F max abs:", np.abs(res["cross_F"]).max())
    print("FR max abs:", np.abs(res["FR"]).max())
    print("varA - V1 - varR =", res["varA"] - res["V1"] - res["varR"])
    print("V1 profile:", res["V1_profile"])


if __name__ == "__main__":
    main()
