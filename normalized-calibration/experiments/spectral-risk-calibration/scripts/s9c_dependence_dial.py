"""S9c: matched-marginal dependence-dial detection (review round 2).

M1/M2 mechanisms in S3/S9b change node error probabilities, so they move
E S as well - detection there measures TOTAL OPERATOR DRIFT (plan 8.3),
not variance sensitivity. This unit uses the manuscript's matched-marginal
model pair as a PURE dependence mechanism:

  edge flags a_mix(x): with probability x an edge takes its node-driven flag
  (model A, shared-node dependence); with probability 1-x it takes an
  independent Bern(p_e) draw (model B). Every edge marginal is p_e for ALL
  x; only the dependence structure changes. E S drifts between E S_A and
  E S_B only (small, measured and reported separately - S1 check4:
  0.046 at n=7, family/native values in S2 mean_shift).

Frozen center/directions/sigma/thresholds are IDENTICAL to S9b (same seeds,
same protocol): calibration at x=1 (truth = full dependence), evaluation on
a ladder x in {1.0, 0.75, 0.5, 0.25, 0.0}. Power(x) is a
power-severity curve in DEPENDENCE LOSS for four statistics:
norm / inf / var / rand. If inf declines slowest toward x=0 (or rises
toward x=1 steepest), the influence structure is doing detection work at a
matched-marginal mechanism, unlike the mean-shift reading of S9b.

Also reports ||E S_mix(x) - E S_A|| via a small MC to document the residual
mean drift along the dial.

Writes results/dependence_dial_v2.csv after the reference-direction repair.
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
import s9b_directional_detection as s9b  # reuse frozen-direction machinery

RESULTS = HERE / "results"
ALPHA = 0.05
X_LADDER = (1.0, 0.75, 0.5, 0.25, 0.0)


def mix_flags(op, P, p, x, rng):
    """Matched-marginal dependence dial: fraction x of edges node-driven."""
    a_node, Y = ho.node_flags(op, P, rng)
    a_iid = ho.edge_flags_iid(p, rng)
    pick = rng.random(op.m) < x
    return np.where(pick, a_node, a_iid)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", default="cora",
                    choices=["cora", "dblp", "familyA", "familyBmax"])
    args = ap.parse_args()
    tag = {"cora": "cora-cc", "dblp": "dblp-ca",
           "familyA": "familyA_coherent",
           "familyBmax": "familyB_flat"}[args.unit]

    if args.unit == "familyA":
        import families as fam
        edges, P, _ = fam.build_family_A(t=2, d=120, pool=5, seed=11)
        n = P.shape[0]
        labels = None
    elif args.unit == "familyBmax":
        import families as fam
        edges, P, _ = fam.build_family_B(d=120, gamma=1 - fam.EPS_WEAK,
                                         seed=12)
        n = P.shape[0]
        labels = None
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

    # budgets identical to S9b for the same frozen objects (same seeds)
    if args.unit == "cora":
        N_SIG, N_CAL, N_EVAL = 400, 800, 1200
        N_ALT = 600
    elif args.unit == "dblp":
        N_SIG, N_CAL, N_EVAL = 150, 400, 500
        N_ALT = 250
    else:
        N_SIG, N_CAL, N_EVAL = 400, 800, 1000
        N_ALT = 500

    t0 = time.time()
    M_hat = ho.estimate_center(op, sampler, N_SIG, np.random.default_rng(11001))
    d_inf, sel = s9b.influence_directions(op, P, deg, 2201, 800)
    groups = {"inf": d_inf,
              "var": s9b.fluctuation_directions(op, P, M_hat, 2202),
              "rand": s9b.random_directions(op, 2203)}
    T0 = s9b.dir_tvals(op, groups, M_hat, sampler, min(N_SIG, 300),
                       np.random.default_rng(11002))
    sigma = {g: np.std(T0[g], axis=0, ddof=1) for g in groups}

    rng_c = np.random.default_rng(11003)
    Tc = s9b.dir_tvals(op, groups, M_hat, sampler, N_CAL, rng_c)
    cal = {g: s9b.standardized(Tc[g], sigma[g]) for g in groups}
    cal["norm"], _ = ho.draw_scores(op, sampler, N_CAL,
                                    np.random.default_rng(11004), M_hat)
    q = {g: ho.rank_threshold(cal[g], ALPHA) for g in cal}
    print(f"{tag}: frozen q={ {k: round(v,4) for k,v in q.items()} } "
          f"{time.time()-t0:.0f}s", flush=True)

    # normal-mechanism achieved FPR sanity on fresh draws
    def eval_mech(x, N, seed):
        rng = np.random.default_rng(seed)
        Mc = op.csr(M_hat)
        const = {g: np.array([float(v @ (Mc @ v)) for v in vs])
                 for g, vs in groups.items()}
        stats = {g: np.empty(N) for g in groups}
        stats["norm"] = np.empty(N)
        mean_drift = np.zeros(op.n_entries)
        for b in range(N):
            a = mix_flags(op, P, p, x, rng)
            s_data = op.S_data(a)
            Sc = op.csr(s_data)
            mean_drift += s_data - M_hat
            for g, vs in groups.items():
                stats[g][b] = np.max(np.abs(
                    np.array([float(v @ (Sc @ v)) for v in vs])
                    - const[g]) / (sigma[g] + s9b.EPS_SIG))
            stats["norm"][b] = ho.spectral_norm_entries(
                op, s_data, M_hat)[0]
        drift = mean_drift / N
        return stats, ho.spectral_norm_matrix(op, drift)[0]

    rows = []
    def add_dir(label, rng_cal, rng_norm_cal, seed0):
        # calibration under `label` baseline; evaluation over the ladder
        rng = np.random.default_rng(seed0)
        Tc2 = dir_tvals_mixed(op, groups, M_hat, p, P,
                              BASE_X[label], N_CAL, rng)
        cal2 = {g: s9b.standardized(Tc2[g], sigma[g]) for g in groups}
        # norm calibration via mixed draws
        rng2 = np.random.default_rng(seed0 + 1)
        cal2["norm"] = np.array([
            ho.spectral_norm_entries(
                op, op.S_data(mix_flags(op, P, p, BASE_X[label], rng2)),
                M_hat)[0] for _ in range(N_CAL)])
        q2 = {g: ho.rank_threshold(cal2[g], ALPHA) for g in cal2}
        for x in X_LADDER:
            N = N_ALT
            stats, drift = eval_mech(x, N, seed0 + int(x * 100) + 7)
            for g in ("norm", "inf", "var", "rand"):
                ex = float(np.mean(stats[g] > q2[g]))
                lo, hi = ho.clopper_pearson(int(round(ex * N)), N)
                rows.append({"structure": tag,
                             "mechanism": f"dep_x{x:.2f}",
                             "statistic": g, "alpha": ALPHA, "q": q2[g],
                             "power": ex, "cp_lo": lo, "cp_hi": hi, "n": N,
                             "mean_drift_vs_A": float(drift),
                             "baseline": label})

    def dir_tvals_mixed(op, groups, M_hat, p, P, xb, N, rng):
        Mc = op.csr(M_hat)
        const = {g: np.array([float(v @ (Mc @ v)) for v in vs])
                 for g, vs in groups.items()}
        out = {g: np.empty((N, len(vs))) for g, vs in groups.items()}
        for b in range(N):
            a = mix_flags(op, P, p, xb, rng)
            Sc = op.csr(op.S_data(a))
            for g, vs in groups.items():
                out[g][b] = np.array([float(v @ (Sc @ v))
                                      for v in vs]) - const[g]
        return out

    BASE_X = {"cal_B_x0.00": 0.0, "cal_mix_x0.50": 0.5}
    for label, seed0 in (("cal_B_x0.00", 33001), ("cal_mix_x0.50", 33101)):
        print(f"{tag}: {label}", flush=True)
        add_dir(label, None, None, seed0)

    for x in X_LADDER:
        N = N_EVAL if x == 1.0 else N_ALT
        stats, drift = eval_mech(x, N, 21000 + int(x * 100))
        for g in ("norm", "inf", "var", "rand"):
            ex = float(np.mean(stats[g] > q[g]))
            lo, hi = ho.clopper_pearson(int(round(ex * N)), N)
            rows.append({"structure": tag, "mechanism": f"dep_x{x:.2f}",
                         "statistic": g, "alpha": ALPHA, "q": q[g],
                         "power": ex, "cp_lo": lo, "cp_hi": hi, "n": N,
                         "mean_drift_vs_A": float(drift),
                         "baseline": "cal_A_x1.00"})
            print(f"x={x:.2f} {g}: {ex:.3f} [{lo:.3f},{hi:.3f}] "
                  f"drift={drift:.3f}", flush=True)

    outp = RESULTS / "dependence_dial_v2.csv"
    exists = outp.exists()
    with open(outp, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(rows)
    print(f"S9c {tag} done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
