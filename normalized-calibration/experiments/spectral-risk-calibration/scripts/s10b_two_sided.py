"""S10b: two-sided calibrated detection (review round 3, priority method fix).

Rank-based conformal p-values on one frozen calibration batch (size M):

    p_up(T)  = (#{T_cal >= T} + 1) / (M + 1)
    p_lo(T)  = (#{T_cal <= T} + 1) / (M + 1)
    one-sided reject: p_up <= alpha
    two-sided reject: min(1, 2*min(p_up, p_lo)) <= alpha

Ties handled by the >= / <= counting (conservative direction for alarms).
Normal-mechanism achieved FPR is reported for every test as a check that
two-sidedness does not inflate the type-I rate beyond 2x nominal.

Matrix: structures {cora, dblp, familyA, familyBmax} x mechanisms
{dial x=0.75/0.5/0.25/0 (dependence LOSS from the A-calibrated normal),
 M1_x2, M2_hubs, M2_lowdeg (drift)} x statistics {norm, inf, var, rand}
x tests {one, two}, all at alpha = 0.05.

This directly addresses the report inconsistency flagged by the review:
zero alarm on dependence loss may be a one-sided artifact, not evidence
that the spectral norm is uninformative.

Writes results/two_sided_detection_v2.csv after the reference-direction
repair; historical two_sided_detection.csv is not modified.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hyperops as ho  # noqa: E402
import families as fam  # noqa: E402
import data_io  # noqa: E402
import s9b_directional_detection as s9b
import s9c_dependence_dial as s9c

RESULTS = HERE / "results"
ALPHA = 0.05


def p_up(cal, t):
    return (float(np.sum(cal >= t)) + 1.0) / (len(cal) + 1.0)


def p_lo(cal, t):
    return (float(np.sum(cal <= t)) + 1.0) / (len(cal) + 1.0)


def eval_mech_scores(op, groups, sigma, M_hat, sampler, N, rng):
    tvals = s9b.dir_tvals(op, groups, M_hat, sampler, N, rng)
    out = {g: s9b.standardized(tvals[g], sigma[g]) for g in groups}
    out["norm"] = np.array([
        ho.spectral_norm_entries(op, op.S_data(a), M_hat)[0]
        for a in (sampler(rng)[0] for _ in range(N))])
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", default="cora",
                    choices=["cora", "dblp", "familyA", "familyBmax"])
    args = ap.parse_args()
    tag = {"cora": "cora-cc", "dblp": "dblp-ca",
           "familyA": "familyA_coherent",
           "familyBmax": "familyB_flat"}[args.unit]

    n, edges, labels, P = None, None, None, None
    if args.unit == "familyA":
        edges, P, _ = fam.build_family_A(t=2, d=120, pool=5, seed=11)
        n = P.shape[0]
    elif args.unit == "familyBmax":
        edges, P, _ = fam.build_family_B(d=120, gamma=1 - fam.EPS_WEAK,
                                         seed=12)
        n = P.shape[0]
    else:
        n, edges, labels = data_io.load_graph(tag)
        P = data_io.majority_binary_profile(labels, 0.05)
    op = ho.RelationOperator(n, edges)
    p = op.edge_marginals(P)
    deg = np.zeros(n)
    for e in edges:
        for u in e:
            deg[u] += 1
    sampler = ho.bound_node_sampler(op, P)

    if args.unit == "cora":
        N_CAL, N_EVAL, N_ALT = 600, 400, 300
    elif args.unit == "dblp":
        N_CAL, N_EVAL, N_ALT = 300, 200, 150
    else:
        N_CAL, N_EVAL, N_ALT = 600, 400, 300

    t0 = time.time()
    M_hat = ho.estimate_center(op, sampler, min(N_CAL, 400),
                               np.random.default_rng(41001))
    d_inf, _sel = s9b.influence_directions(op, P, deg, 2201, 800)
    groups = {"inf": d_inf,
              "var": s9b.fluctuation_directions(op, P, M_hat, 2202),
              "rand": s9b.random_directions(op, 2203)}
    T0 = s9b.dir_tvals(op, groups, M_hat, sampler, 300,
                       np.random.default_rng(41002))
    sigma = {g: np.std(T0[g], axis=0, ddof=1) for g in groups}

    cal = eval_mech_scores(op, groups, sigma, M_hat, sampler, N_CAL,
                           np.random.default_rng(41003))

    # mechanisms
    mechs = []
    for x in (0.75, 0.5, 0.25, 0.0):
        mechs.append((f"dial_x{x:.2f}",
                      lambda rng, x=x: s9c.mix_flags(op, P, p, x, rng)))
    if labels is not None:
        hub = deg >= float(np.quantile(deg, 0.90))
        for f, lab in ((2.0, "M1_x2"),):
            Pa = data_io.majority_binary_profile(labels, 0.05 * f)
            mechs.append((lab, lambda rng, Pa=Pa:
                          ho.bound_node_sampler(op, Pa)(rng)[0]))
        e = np.full(n, 0.05)
        e[hub] = min(0.05 * 4, 0.49)
        mechs.append(("M2_hubs", lambda rng, e=e:
                      ho.bound_node_sampler(
                          op, s9b.s9b_binary_from_eps(labels, e))(rng)[0]))
        order = np.argsort(deg)
        low = np.zeros(n, dtype=bool)
        low[[u for u in order if not hub[u]][: int(hub.sum())]] = True
        e2 = np.full(n, 0.05)
        e2[low] = min(0.05 * 4, 0.49)
        mechs.append(("M2_lowdeg", lambda rng, e2=e2:
                      ho.bound_node_sampler(
                          op, s9b.s9b_binary_from_eps(labels, e2))(rng)[0]))

    rows = []
    # achieved FPR on fresh normal draws (one and two sided)
    def rates(cal_v, ev_v):
        r = {}
        for test in ("one", "two"):
            rej = 0
            for t in ev_v:
                pu = p_up(cal_v, t)
                pl = p_lo(cal_v, t)
                pv = pu if test == "one" else min(1.0, 2 * min(pu, pl))
                rej += int(pv <= ALPHA)
            r[test] = rej / len(ev_v)
        return r

    ev_norm = eval_mech_scores(op, groups, sigma, M_hat, sampler, N_EVAL,
                               np.random.default_rng(41004))
    for stat in ("norm", "inf", "var", "rand"):
        fr = rates(cal[stat], ev_norm[stat])
        rows.append({"structure": tag, "mechanism": "normal_FPR_check",
                     "statistic": stat, "test": "one",
                     "reject": fr["one"], "n": N_EVAL, "alpha": ALPHA})
        rows.append({"structure": tag, "mechanism": "normal_FPR_check",
                     "statistic": stat, "test": "two",
                     "reject": fr["two"], "n": N_EVAL, "alpha": ALPHA})

    for mlab, gen in mechs:
        rng = np.random.default_rng(42000 + len(rows))
        ev = {stat: np.empty(N_ALT) for stat in ("norm", "inf", "var", "rand")}
        tvals = {g: np.empty((N_ALT, len(groups[g])))
                 for g in groups}
        Mc = op.csr(M_hat)
        const = {g: np.array([float(v @ (Mc @ v)) for v in vs])
                 for g, vs in groups.items()}
        for b in range(N_ALT):
            a = gen(rng)
            s_data = op.S_data(a)
            Sc = op.csr(s_data)
            for g in groups:
                tvals[g][b] = np.array([float(v @ (Sc @ v))
                                        for v in groups[g]]) - const[g]
                ev[g][b] = np.max(np.abs(tvals[g][b])
                                  / (sigma[g] + s9b.EPS_SIG))
            ev["norm"][b] = ho.spectral_norm_entries(op, s_data, M_hat)[0]
        for stat in ("norm", "inf", "var", "rand"):
            rr = rates(cal[stat], ev[stat])
            for test in ("one", "two"):
                ex = rr[test]
                lo, hi = ho.clopper_pearson(int(round(ex * N_ALT)), N_ALT)
                rows.append({"structure": tag, "mechanism": mlab,
                             "statistic": stat, "test": test,
                             "reject": ex, "n": N_ALT, "alpha": ALPHA})
            print(f"{tag} {mlab} {stat}: one={rr['one']:.3f} "
                  f"two={rr['two']:.3f}", flush=True)

    outp = RESULTS / "two_sided_detection_v2.csv"
    exists = outp.exists()
    with open(outp, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(rows)
    print(f"S10b {tag} done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
