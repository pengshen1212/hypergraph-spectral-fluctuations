"""S12: independent new-seed confirmation of the three load-bearing claims
(review round 4: exploration seeds must not double as confirmation).

C1  calibration recovery (S3): cora binary-eps0.05, node calibration,
    alpha=5%, 3 fresh reference/calibration/evaluation triples.
C2  direction detection on DBLP (S9b): frozen new reference
    (M_hat, directions, sigma), M2_hubs evaluation, inf vs norm power.
C3  two-sided artifact (S10b): cora dial x=0, norm one-sided vs two-sided
    rejection rate on dependence LOSS.

Random streams use a separate seed namespace. This is a new-seed rerun,
not a retrospectively registered confirmation. Version 2 writes
results/confirm_results_v2.csv and never overwrites the historical CSV.
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
import data_io  # noqa: E402
import s9b_directional_detection as s9b  # noqa: E402
import s9c_dependence_dial as s9c  # noqa: E402
import s10b_two_sided as s10b  # noqa: E402
from audit_evidence import summarize_confirmation

RESULTS = HERE / "results"
ALPHA = 0.05
MASTER = 20261021


def c1_cora_calibration():
    n, edges, labels = data_io.load_graph("cora-cc")
    op = ho.RelationOperator(n, edges)
    P = data_io.majority_binary_profile(labels, 0.05)
    sampler = ho.bound_node_sampler(op, P)
    ex_list, qs = [], []
    for b in range(3):
        sd = MASTER + 100 + 17 * b
        M_hat = ho.estimate_center(op, sampler, 200,
                                   np.random.default_rng(sd))
        cal, _ = ho.draw_scores(op, sampler, 500, np.random.default_rng(sd + 1),
                                M_hat)
        ev, _ = ho.draw_scores(op, sampler, 1000, np.random.default_rng(sd + 2),
                               M_hat)
        q = ho.rank_threshold(cal, ALPHA)
        ex_list.append(float(np.mean(ev > q)))
        qs.append(q)
    ex = float(np.mean(ex_list))
    return [{"claim": "C1_S3_calibration_recovery", "structure": "cora-cc",
             "metric": "node_cal_exceedance_alpha05", "estimate": ex,
             "cp_lo": "", "cp_hi": "", "n": 3000, "batches": 3,
             "q_mean": float(np.mean(qs))}]


def c2_dblp_inf():
    n, edges, labels = data_io.load_graph("dblp-ca")
    op = ho.RelationOperator(n, edges)
    P = data_io.majority_binary_profile(labels, 0.05)
    deg = np.zeros(n)
    for e in edges:
        for u in e:
            deg[u] += 1
    sampler = ho.bound_node_sampler(op, P)
    sd = MASTER + 500
    M_hat = ho.estimate_center(op, sampler, 150, np.random.default_rng(sd))
    d_inf, _ = s9b.influence_directions(op, P, deg, sd + 1, 800)
    groups = {"inf": d_inf,
              "var": s9b.fluctuation_directions(op, P, M_hat, sd + 2),
              "rand": s9b.random_directions(op, sd + 3)}
    T0 = s9b.dir_tvals(op, groups, M_hat, sampler, 250,
                       np.random.default_rng(sd + 4))
    sigma = {g: np.std(T0[g], axis=0, ddof=1) for g in groups}
    Tc = s9b.dir_tvals(op, groups, M_hat, sampler, 400,
                       np.random.default_rng(sd + 5))
    cal = {g: s9b.standardized(Tc[g], sigma[g]) for g in groups}
    cal["norm"], _ = ho.draw_scores(op, sampler, 400,
                                    np.random.default_rng(sd + 6), M_hat)
    q = {g: ho.rank_threshold(cal[g], ALPHA) for g in cal}
    hub = deg >= float(np.quantile(deg, 0.90))
    e = np.full(n, 0.05)
    e[hub] = min(0.05 * 4, 0.49)
    P_alt = s9b.s9b_binary_from_eps(labels, e)
    s_alt = ho.bound_node_sampler(op, P_alt)
    N = 250
    Ta = s9b.dir_tvals(op, groups, M_hat, s_alt, N,
                       np.random.default_rng(sd + 7))
    stat = {g: s9b.standardized(Ta[g], sigma[g]) for g in groups}
    stat["norm"], _ = ho.draw_scores(op, s_alt, N,
                                     np.random.default_rng(sd + 8), M_hat)
    out = []
    for g in stat:
        ex = float(np.mean(stat[g] > q[g]))
        lo, hi = ho.clopper_pearson(int(round(ex * N)), N)
        out.append({"claim": "C2_S9b_M2hubs_detection", "structure": "dblp-ca",
                    "metric": f"M2_hubs_power_{g}", "estimate": ex,
                    "cp_lo": lo, "cp_hi": hi, "n": N, "batches": 1,
                    "q_mean": q[g]})
    return out


def c3_cora_two_sided():
    n, edges, labels = data_io.load_graph("cora-cc")
    op = ho.RelationOperator(n, edges)
    P = data_io.majority_binary_profile(labels, 0.05)
    p = op.edge_marginals(P)
    sampler = ho.bound_node_sampler(op, P)
    sd = MASTER + 900
    M_hat = ho.estimate_center(op, sampler, 400, np.random.default_rng(sd))
    cal, _ = ho.draw_scores(op, sampler, 600, np.random.default_rng(sd + 1),
                            M_hat)
    rng = np.random.default_rng(sd + 2)
    ev = np.array([ho.spectral_norm_entries(
        op, op.S_data(s9c.mix_flags(op, P, p, 0.0, rng)), M_hat)[0]
        for _ in range(400)])
    one = float(np.mean(ev > ho.rank_threshold(cal, ALPHA)))
    two = float(np.mean([min(1.0, 2 * min(s10b.p_up(cal, t),
                                            s10b.p_lo(cal, t))) <= ALPHA
                         for t in ev]))
    return [{"claim": "C3_S10b_two_sided_norm_dial0", "structure": "cora-cc",
             "metric": "norm_power_one", "estimate": one, "cp_lo": "",
             "cp_hi": "", "n": 400, "batches": 1, "q_mean": ""},
            {"claim": "C3_S10b_two_sided_norm_dial0", "structure": "cora-cc",
             "metric": "norm_power_two", "estimate": two, "cp_lo": "",
             "cp_hi": "", "n": 400, "batches": 1, "q_mean": ""}]


def main():
    t0 = time.time()
    rows = []
    print("C1...", flush=True)
    rows += c1_cora_calibration()
    print("C3...", flush=True)
    rows += c3_cora_two_sided()
    print("C2 (dblp)...", flush=True)
    rows += c2_dblp_inf()
    summaries = summarize_confirmation(rows)
    rows = [dict(raw, **{key: summary[key] for key in
                        ("count", "ci_lo", "ci_hi", "interval_method", "estimand")})
            for raw, summary in zip(rows, summaries)]
    with open(RESULTS / "confirm_results_v2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r, flush=True)
    print(f"S12 done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
