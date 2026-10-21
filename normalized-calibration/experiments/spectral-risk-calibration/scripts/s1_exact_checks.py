"""S1: exact small-graph verification (plan section 9, stage A0).

Checks (all exact by full enumeration unless stated):
  1. Relation probabilities: node-resampled edge flags match analytic p_e.
  2. Dynamic degree recomputation vs dense A(a) @ 1.
  3. First-order influence decomposition under DYNAMIC normalization:
     F_u^S(z) = E[S|Y_u=z] - E S ; R = S - E S - sum_u F_u^S(Y_u);
     exact E[R | Y_u=z] = 0, Frobenius orthogonality and energy identity.
  4. Fixed-degree control: E S_0 matches EXACTLY across arms while dynamic
     E S_A != E S_B in general; report the mean shift.
  5. Dense eigvalsh vs sparse eigsh on a medium controlled graph.
  6. Rank threshold: boundary k>M, tie conservatism, and empirical marginal
     exceedance of the finite-sample rule under exchangeable node draws.

Writes results/exact_checks.json. No Monte Carlo is used inside any PASS
criterion except check 1 (rate recovery) and check 6 (rule behavior).
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

import hyperops as ho  # noqa: E402
import families as fam  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def dense_S(op, a):
    B = np.zeros((op.n, op.n))
    for i, e in enumerate(op.edges):
        B[np.ix_(e, e)] += a[i] / len(e)
    A = B + op.kappa * np.eye(op.n)
    d = A @ np.ones(op.n)
    dis = 1.0 / np.sqrt(d)
    return A * np.outer(dis, dis)


def all_states(n, C):
    return list(itertools.product(range(C), repeat=n))


def state_prob(P, st):
    return float(np.prod([P[u, c] for u, c in enumerate(st)]))


# ----------------------------------------------------------------------
def check1_rate_recovery():
    """Node-resampled per-edge rates vs analytic p_e (Monte Carlo)."""
    edges = [(0, 1, 2), (0, 1, 2), (2, 3), (3, 4, 5), (1, 4)]
    rng = np.random.default_rng(101)
    P = np.array([[0.7, 0.3], [0.2, 0.8], [0.5, 0.5],
                  [0.9, 0.1], [0.4, 0.6], [0.95, 0.05]])
    op = ho.RelationOperator(6, edges)
    p = op.edge_marginals(P)
    sampler = ho.bound_node_sampler(op, P)
    N = 200_000
    acc = np.zeros(op.m)
    for _ in range(N):
        a, _ = sampler(rng)
        acc += a
    rate = acc / N
    # per-edge binomial sd at least ~ sqrt(0.25/N)=3.5e-4
    max_dev = float(np.max(np.abs(rate - p)))
    return {"mc_draws": N, "max_abs_rate_dev": max_dev,
            "tol": 5e-3, "pass": bool(max_dev < 5e-3)}


def check2_dynamic_degree():
    edges = [(0, 1, 2), (2, 3), (3, 4, 5), (0, 5)]
    P = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4],
                  [0.55, 0.45], [0.1, 0.9], [0.45, 0.55]])
    op = ho.RelationOperator(6, edges)
    worst = 0.0
    for st in all_states(6, 2):
        a = op.flags_from_Y(st)
        B = np.zeros((6, 6))
        for i, e in enumerate(op.edges):
            B[np.ix_(e, e)] += a[i] / len(e)
        Ad = B + op.kappa * np.eye(6)
        dd = Ad @ np.ones(6)
        worst = max(worst, float(np.max(np.abs(op.degrees(a) - dd))))
    return {"max_abs_degree_dev": worst,
            "states_enumerated": 2 ** 6,
            "pass": bool(worst < 1e-12)}


def check3_influence_decomposition():
    """Exact F^S decomposition + residual orthogonality, n=7 binary."""
    edges = [(0, 1, 2), (0, 1, 2), (2, 3, 4), (4, 5), (5, 6), (1, 6)]
    P = np.array([[0.7, 0.3], [0.2, 0.8], [0.55, 0.45], [0.9, 0.1],
                  [0.4, 0.6], [0.65, 0.35], [0.3, 0.7]])
    n, C = 7, 2
    op = ho.RelationOperator(n, edges)
    ES = op.exact_ES_node(P)
    states = all_states(n, C)
    probs = np.array([state_prob(P, st) for st in states])

    # conditional means E[S | Y_u=z] (exact enumeration over remaining nodes)
    F = {}
    for u in range(n):
        for z in range(C):
            cm, _ = op.exact_conditional_ES_node(P, [u], [z])
            F[(u, z)] = cm - ES

    # residual per state and exact moments
    cross_max = 0.0
    E_Fu_max = 0.0
    sumF = np.zeros(len(states), dtype=object)
    R_arr = []
    for j, st in enumerate(states):
        if probs[j] == 0.0:
            R_arr.append(None)
            continue
        s = op.S_data(op.flags_from_Y(st))
        tot = s - ES
        for u in range(n):
            tot = tot - F[(u, st[u])]
        R_arr.append(tot)
        # E[F_u(Y_u)] = sum_z P[u,z] F_u(z)
        EFu = sum(P[u, z] * F[(u, z)] for z in range(C))
        E_Fu_max = max(E_Fu_max, float(np.max(np.abs(EFu))))
    # E[R | Y_u = z] exact
    for u in range(n):
        for z in range(C):
            acc = np.zeros(op.n_entries)
            tot_p = 0.0
            for j, st in enumerate(states):
                if st[u] == z and probs[j] > 0:
                    acc += probs[j] * R_arr[j]
                    tot_p += probs[j]
            cross_max = max(cross_max, float(np.max(np.abs(acc / tot_p))))

    # Frobenius energy identity via dense matrices
    X2 = 0.0      # E||S-ES||_F^2
    F1 = 0.0      # E||sum_u F_u(Y_u)||_F^2
    Rr = 0.0      # E||R||_F^2
    cross_uv = 0.0
    cross_uR = 0.0
    denseF = {k: op.csr(v).toarray() for k, v in F.items()}
    dense_ES = op.csr(ES).toarray()
    for j, st in enumerate(states):
        if probs[j] == 0.0:
            continue
        s = dense_S(op, op.flags_from_Y(st))
        X = s - dense_ES
        sumFm = np.zeros_like(s)
        for u in range(n):
            sumFm += denseF[(u, st[u])]
        Rm = X - sumFm
        X2 += probs[j] * float(np.sum(X * X))
        F1 += probs[j] * float(np.sum(sumFm * sumFm))
        Rr += probs[j] * float(np.sum(Rm * Rm))
        for u in range(n):
            v_u = denseF[(u, st[u])]
            cross_uR += probs[j] * float(np.sum(v_u * Rm))
            for v in range(u + 1, n):
                cross_uv += probs[j] * float(np.sum(v_u * denseF[(v, st[v])]))
    identity_gap = abs(X2 - (F1 + Rr))
    scale = max(X2, 1e-12)
    return {
        "E_Fu_maxabs": E_Fu_max,
        "E[R|Y_u=z]_maxabs": cross_max,
        "energy_X": X2, "energy_first_order": F1, "energy_residual": Rr,
        "energy_identity_gap_frac": identity_gap / scale,
        "cross_FuFv_abs": abs(cross_uv) / scale,
        "cross_FuR_abs": abs(cross_uR) / scale,
        "pass": bool(cross_max < 1e-9 and E_Fu_max < 1e-9
                     and identity_gap / scale < 1e-9
                     and abs(cross_uv) / scale < 1e-9
                     and abs(cross_uR) / scale < 1e-9),
    }


def check4_mean_separation():
    """Exact: fixed-degree means match; dynamic means differ; report shift."""
    edges = [(0, 1, 2), (0, 1, 2), (2, 3, 4), (4, 5), (5, 6), (1, 6)]
    P = np.array([[0.7, 0.3], [0.2, 0.8], [0.55, 0.45], [0.9, 0.1],
                  [0.4, 0.6], [0.65, 0.35], [0.3, 0.7]])
    n = 7
    op = ho.RelationOperator(n, edges)
    p = op.edge_marginals(P)

    ES_A = op.exact_ES_node(P)
    ES_B = op.exact_ES_edge(p)
    shift_dyn = float(np.linalg.norm(op.csr(ES_A - ES_B).toarray(), 2))

    di0 = op.S0_inverse_degrees(p)
    states = all_states(n, 2)
    acc_n = np.zeros(op.n_entries)
    for st in states:
        pr = state_prob(P, st)
        if pr:
            acc_n += pr * op.S0_data(op.flags_from_Y(st), di0)
    acc_e = np.zeros(op.n_entries)
    for bits in itertools.product((0, 1), repeat=op.m):
        a = np.asarray(bits, dtype=float)
        pr = 1.0
        for i in range(op.m):
            pr *= a[i] * p[i] + (1 - a[i]) * (1 - p[i])
        if pr:
            acc_e += pr * op.S0_data(a, di0)
    s0_gap = float(np.max(np.abs(acc_n - acc_e)))

    # raw linear means identical
    EB = op.EB_data(p)
    acc_b_raw = np.zeros(op.n_entries)
    for st in states:
        pr = state_prob(P, st)
        if pr:
            acc_b_raw += pr * op.raw_b_data(op.flags_from_Y(st))
    raw_gap = float(np.max(np.abs(acc_b_raw - EB)))
    return {"raw_EB_gap": raw_gap,
            "fixed_degree_mean_gap": s0_gap,
            "dynamic_mean_shift_spectral": shift_dyn,
            "pass": bool(raw_gap < 1e-12 and s0_gap < 1e-12)}


def check5_dense_vs_lanczos():
    edges, P, _ = fam.build_family_A(t=2, d=120, pool=5, seed=11)
    n = 2 * (1 + 120 // 5 + 120)
    op = ho.RelationOperator(n, edges)
    sampler = ho.bound_node_sampler(op, P)
    rng = np.random.default_rng(5)
    center = ho.estimate_center(op, sampler, 60, rng)
    a, _ = sampler(rng)
    s = op.S_data(a)
    diff = op.csr(s - center)
    exact = float(np.max(np.abs(np.linalg.eigvalsh(diff.toarray()))))
    approx, resid = ho.spectral_norm_entries(
        op, s, center, dense_limit=0)  # force sparse path
    rel = abs(approx - exact) / max(exact, 1e-12)
    return {"n": n, "exact": exact, "lanczos": approx,
            "residual": resid, "relerr": rel,
            "pass": bool(rel < 1e-3 and resid < 1e-4)}


def check6_rank_threshold():
    edges = [(0, 1, 2), (0, 1, 2), (2, 3), (3, 4, 5), (1, 4)]
    P = np.array([[0.7, 0.3], [0.2, 0.8], [0.5, 0.5],
                  [0.9, 0.1], [0.4, 0.6], [0.95, 0.05]])
    op = ho.RelationOperator(6, edges)
    sampler = ho.bound_node_sampler(op, P)
    rng = np.random.default_rng(23)
    center = ho.estimate_center(op, sampler, 400, rng)

    rows = []
    for alpha in (0.01, 0.05, 0.10):
        hits = []
        for rep in range(40):
            rng2 = np.random.default_rng(1000 + rep)
            cal, _ = ho.draw_scores(op, sampler, 400, rng2, center)
            q = ho.rank_threshold(cal, alpha)
            new, _ = ho.draw_scores(op, sampler, 200, rng2, center)
            hits.append(float(np.mean(new > q)))
        m = float(np.mean(hits))
        lo, hi = ho.clopper_pearson(int(round(m * 40)), 40)
        rows.append({"alpha": alpha, "mean_exceed": m, "cp95": [lo, hi]})

    inf_check = ho.rank_threshold(np.arange(5.0), 0.05)  # k=ceil(6*.95)=6>5
    tie = ho.rank_threshold(np.zeros(50), 0.05)
    return {"exchange": rows,
            "k_gt_M_returns_inf": bool(math.isinf(ho.rank_threshold(
                np.arange(5.0), 0.05))),
            "tie_threshold": float(tie),
            "pass": bool(all(r["mean_exceed"] <= r["alpha"] + 0.03 for r in rows)
                         and math.isinf(inf_check) and tie == 0.0)}


def main():
    out = {}
    out["check1_rate_recovery"] = check1_rate_recovery()
    out["check2_dynamic_degree"] = check2_dynamic_degree()
    out["check3_influence_decomposition"] = check3_influence_decomposition()
    out["check4_mean_separation"] = check4_mean_separation()
    out["check5_dense_vs_lanczos"] = check5_dense_vs_lanczos()
    out["check6_rank_threshold"] = check6_rank_threshold()
    out["all_pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    with open(RESULTS / "exact_checks.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
