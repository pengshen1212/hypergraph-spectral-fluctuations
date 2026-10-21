"""S5-a: numerical verification of the two audit-theory candidates.

T4 engineering lemma (plan 5.T4). For symmetric nonnegative A, A' with
min-degree >= d0 > 0 and S = D^{-1/2} A D^{-1/2}:

    ||S - S'||_2  <=  ||A - A'||_2 / d0  +  ||A'||_2 ||D - D'||_2 / d0^2

Proof (recorded in docs/theorem_notes.md): three-term triangle split of
S - S' plus the Lipschitz bound |x^{-1/2}-y^{-1/2}| <= |x-y|/(2 d0^{3/2})
for x, y >= d0. Here d0 = kappa (degree = kappa + #active edges >= kappa).
Verification: random flag pairs on every structure; report LHS/RHS and
tightness fraction.

T2 Efron-Stein route (plan 5.T2). With independent node coordinates Y_u and
S^(u) = S with Y_u replaced by an independent draw:

    V_ES = (1/2) sum_u E ||S - S^(u)||_F^2,
    P(||S - E S||_2 >= t) <= E||S-ES||_F^2 / t^2 <= V_ES / t^2.

Estimator (plan 6.3): draw u with structural probability p_u = deg_u/sum(deg);
Z = ||S - S^(u)||_F^2 / (2 p_u) is unbiased for V_ES. Reported as empirical
mean (NOT a certificate until an upper confidence bound exists; see notes).

Writes results/theory_checks.json.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

import hyperops as ho  # noqa: E402
import families as fam  # noqa: E402
import data_io  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def t4_check(op, P, n_trials, seed):
    rng = np.random.default_rng(seed)
    sampler = ho.bound_node_sampler(op, P)
    worst_ratio = 0.0
    viol = 0
    for _ in range(n_trials):
        a, _ = sampler(rng)
        b, _ = sampler(rng)
        A = op.A_data(a)
        B2 = op.A_data(b)
        S_a = op.S_data(a)
        S_b = op.S_data(b)
        dA = op.csr(A - B2)
        dS = op.csr(S_a - S_b)
        lhs = ho.spectral_norm_entries(op, S_a, S_b)[0]
        nrm_A2 = ho.spectral_norm_entries(op, B2, np.zeros(op.n_entries))[0]
        nrm_dA = ho.spectral_norm_entries(op, A - B2, np.zeros(op.n_entries))[0]
        deg_a = op.degrees(a)
        deg_b = op.degrees(b)
        if op.n <= 600:
            nrm_dD = float(np.max(np.abs(deg_a - deg_b)))
        else:
            nrm_dD = float(np.max(np.abs(deg_a - deg_b)))  # diagonal 2-norm
        d0 = op.kappa
        rhs = nrm_dA / d0 + nrm_A2 * nrm_dD / d0 ** 2
        ratio = lhs / max(rhs, 1e-15)
        worst_ratio = max(worst_ratio, ratio)
        if lhs > rhs * (1 + 1e-9):
            viol += 1
    return {"trials": n_trials, "violations": viol,
            "max_lhs_over_rhs": worst_ratio}


def es_v_check(op, P, n_draws, seed):
    """Estimate V_ES via importance-sampled node replacements + tail check.

    Z = ||S(Y) - S(Y^(u))||_F^2 / (2 p_u), p_u = structural degree share,
    E Z = V_ES (plan 6.3). Tail columns: the score distribution and the
    replacement batch share the same frozen ES center from an independent
    reference batch; exceedance vs V_hat/t^2 is an in-model empirical check.
    """
    rng = np.random.default_rng(seed)
    sampler = ho.bound_node_sampler(op, P)
    struct_deg = np.asarray(op.incidence @ np.ones(op.m)).ravel() + 1e-12
    p_u = struct_deg / struct_deg.sum()
    n_ref = min(n_draws, 1000)
    ES = ho.estimate_center(op, sampler, n_ref, np.random.default_rng(seed + 1))
    zs = []
    scores = []
    x2 = 0.0
    for _ in range(n_draws):
        a, Y = sampler(rng)
        d = op.S_data(a) - ES
        x2 += float(d @ d)
        scores.append(ho.spectral_norm_entries(op, op.S_data(a), ES)[0])
        u = int(rng.choice(op.n, p=p_u))
        Yp = Y.copy()
        P_u = P[u]
        # independent replacement draw for node u (may coincide with Y_u)
        Yp[u] = int((rng.random() >= np.cumsum(P_u)).sum())
        a2 = op.flags_from_Y(Yp)
        dd = op.S_data(a) - op.S_data(a2)
        zs.append(float(dd @ dd) / (2.0 * p_u[u]))
    V_hat = float(np.mean(zs))
    se_V = float(np.std(zs, ddof=1) / math.sqrt(len(zs)))
    eff_n = float(np.sum(zs) ** 2 / max(np.sum(np.square(zs)), 1e-30))
    x2 = x2 / n_draws
    scores = np.sort(np.array(scores))
    tail = []
    for t in (float(np.quantile(scores, 0.99)),
              float(np.quantile(scores, 0.999)),
              float(scores[-1])):
        if t <= 0:
            continue
        emp = float(np.mean(scores > t))
        bound = min(1.0, V_hat / t ** 2)
        tail.append({"t": t, "emp_exceed": emp, "V_ES_bound": bound,
                     "trivially_true": bool(bound >= 1.0 - 1e-12),
                     "informative_valid": bool(bound < 1.0 and
                                               emp <= bound + 1e-12)})
    return {"V_ES_hat": V_hat, "se_V": se_V,
            "V_ci95": [V_hat - 1.96 * se_V, V_hat + 1.96 * se_V],
            "effective_n": eff_n, "n_replacements": len(zs),
            "X2_frob": x2,
            "ratio_V_over_X2": V_hat / max(x2, 1e-18),
            "ratio_violation_within_se": bool(
                (x2 - V_hat) <= 1.96 * se_V),
            "tail": tail,
            "note": "V_hat is a point estimate with recorded standard error; "
                    "a value below X2 within ~2 se is NOT by itself an "
                    "implementation error proof (heavy-tailed estimator). "
                    "Tail rows with bound>=1 are trivially true, not "
                    "validation. Certificate requires a UCB (T2b)."}


def main():
    out = {"t4": {}, "es": {}}
    structures = []
    eA, PA, _ = fam.build_family_A(t=2, d=120, pool=5, seed=11)
    structures.append(("familyA", ho.RelationOperator(PA.shape[0], eA), PA))
    eB, PB, _ = fam.build_family_B(d=120, gamma=1 - fam.EPS_WEAK, seed=12)
    structures.append(("familyB_max", ho.RelationOperator(PB.shape[0], eB), PB))
    cora_n, cora_edges, cora_labels = data_io.load_graph("cora-cc")
    Pc = data_io.majority_binary_profile(cora_labels, 0.05)
    structures.append(("cora-cc", ho.RelationOperator(cora_n, cora_edges), Pc))
    dblp_n, dblp_edges, dblp_labels = data_io.load_graph("dblp-ca")
    Pd = data_io.majority_binary_profile(dblp_labels, 0.05)
    structures.append(("dblp-ca", ho.RelationOperator(dblp_n, dblp_edges), Pd))

    for name, op, P in structures:
        t0 = time.time()
        n_t4 = 60 if op.n > 10000 else 200
        out["t4"][name] = t4_check(op, P, n_t4,
                             seed=abs(sum(map(ord, name))) % 10000)
        out["t4"][name]["secs"] = round(time.time() - t0, 1)
        print(f"T4 {name}: {out['t4'][name]}", flush=True)
        if name == "dblp-ca":
            n_draws = 200
        elif name == "cora-cc":
            n_draws = 400
        else:
            n_draws = 800
        t1 = time.time()
        out["es"][name] = es_v_check(op, P, n_draws, 977)
        out["es"][name]["secs"] = round(time.time() - t1, 1)
        print(f"ES {name}: {out['es'][name]}", flush=True)

    with open(RESULTS / "theory_checks.json", "w") as f:
        json.dump(out, f, indent=2)
    print("S5-a done.")


if __name__ == "__main__":
    main()
