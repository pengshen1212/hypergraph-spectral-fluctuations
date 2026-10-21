"""S10a: perturbation amplitude scan A(t) = A0 + tE (review round 3).

For paired random perturbations (node-driven A arm, iid-edge B arm around
the same reference p), continuously interpolate a_t = (1-t) p + t a so that
B(a_t) = B(p) + t E and scan t in [0,1]:

  - true(t)   = ||S(a_t) - S0||_2
  - lin(t)    = ||t J_A(E)||_2          (first-order prediction)
  - resid(t)  = ||S(a_t) - S0 - t J||_2

Questions answered:
  (1) where does resid/true cross 0.5 -> the valid range of the linear
      approximation for REAL random perturbations;
  (2) ratio(t) = mean_A true(t) / mean_B true(t): at which amplitude does
      the coherent family's reversal appear (ratio crossing 1), i.e. is
      the reversal a small-t (local) phenomenon or only at full t=1
      (finite-perturbation) phenomenon.

Writes results/amplitude_scan_v2.csv. Historical amplitude_scan.csv is kept
unchanged: its resid_frac_node was a difference-of-norms proxy, not a matrix
remainder. Version 2 records both quantities and their numerical residuals.
"""

from __future__ import annotations

import csv
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
T_GRID = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                   0.65, 0.80, 1.00])


def amplitude_metrics(op, basis, a, t):
    """Return true, linear, actual matrix-remainder norms and scalar norm gap."""
    _, _, j, _, _ = op.linear_response(a, basis)
    at = basis["p"] + t * (a - basis["p"])
    delta = op.S_data(at) - basis["S0"]
    zero = np.zeros(op.n_entries)
    true = ho.spectral_norm_entries(op, delta, zero)[0]
    linear = ho.spectral_norm_entries(op, t * j, zero)[0]
    remainder = ho.spectral_norm_entries(op, delta, t * j)[0]
    return true, linear, remainder, abs(true - linear)


def scan(name, profile, op, P, n_pairs, seed):
    t0 = time.time()
    p = op.edge_marginals(P)
    basis = op.response_basis(p)
    rng = np.random.default_rng(seed)
    node_s = ho.bound_node_sampler(op, P)
    edge_s = ho.bound_edge_sampler(op, p)
    zero = np.zeros(op.n_entries)

    per_arm = {}
    for arm, sampler in (("node", node_s), ("edge", edge_s)):
        truths = np.zeros((n_pairs, len(T_GRID)))
        lins = np.zeros_like(truths)
        remainders = np.zeros_like(truths)
        numerical_resids = np.zeros_like(truths)
        for i in range(n_pairs):
            a, _ = sampler(rng)
            _, _, j, _, _ = op.linear_response(a, basis)
            for k, t in enumerate(T_GRID):
                a_t = p + t * (a - p)
                st = op.S_data(a_t) - basis["S0"]
                truths[i, k], er_true = ho.spectral_norm_entries(op, st, zero)
                lins[i, k], er_lin = ho.spectral_norm_entries(op, t * j, zero)
                remainders[i, k], er_rem = ho.spectral_norm_entries(op, st, t * j)
                numerical_resids[i, k] = max(er_true, er_lin, er_rem)
        per_arm[arm] = (truths, lins, remainders, numerical_resids)

    rows = []
    nA, lA, rA, erA = per_arm["node"]
    nB, lB, rB, erB = per_arm["edge"]
    for k, t in enumerate(T_GRID):
        trueA, trueB = float(nA[:, k].mean()), float(nB[:, k].mean())
        linA = float(lA[:, k].mean())
        # Reverse triangle inequality: norm-gap <= actual matrix remainder.
        gapA = float(np.mean(np.abs(nA[:, k] - lA[:, k])) / max(trueA, 1e-12))
        rows.append({
            "structure": name, "profile": profile, "t": float(t),
            "true_node": trueA, "lin_node": linA,
            "true_edge": trueB,
            "resid_frac_node": float(rA[:, k].mean() / max(trueA, 1e-12)),
            "norm_gap_frac_node": gapA,
            "max_solver_resid": float(max(erA[:, k].max(), erB[:, k].max())),
            "ratio_t": trueA / max(trueB, 1e-12),
            "n_pairs": n_pairs,
        })
    # crossing estimates
    def cross_fracs(col, target):
        xs = [r["t"] for r in rows]
        ys = [r[col] for r in rows]
        for x1, y1, x0, y0 in zip(xs[1:], ys[1:], xs[:-1], ys[:-1]):
            if (y0 < target) != (y1 < target):
                return x1
        return float("nan")
    t_resid50 = cross_fracs("resid_frac_node", 0.5)
    rows_out = rows + [{"structure": name, "profile": profile,
                        "t": "T50_resid", "true_node": "", "lin_node": "",
                        "true_edge": "", "resid_frac_node": t_resid50,
                        "ratio_t": cross_fracs("ratio_t", 1.0),
                        "n_pairs": n_pairs,
                        "note": "resid50 col = t at which resid/true "
                                "exceeds 0.5; ratio col = t at which "
                                "node/edge ratio crosses 1"}]
    print(f"{name} {profile}: t_resid50={t_resid50} "
          f"ratio_cross_1="
          f"{cross_fracs('ratio_t', 1.0)} ({time.time()-t0:.0f}s)", flush=True)
    return rows_out


UNITS = {
    "familyA": ("familyA_coherent", "fixed", "familyA"),
    "familyB": ("familyB_flat", "gamma=0.95_max", "familyB"),
    "cora": ("cora-cc", "binary-eps0.05", "cora"),
    "dblp": ("dblp-ca", "binary-eps0.05", "dblp"),
}
FIELDS = ["structure", "profile", "t", "true_node", "lin_node", "true_edge",
          "resid_frac_node", "norm_gap_frac_node", "max_solver_resid",
          "ratio_t", "n_pairs", "note"]


def build_op(unit):
    if unit == "familyA":
        e, P, _ = fam.build_family_A(t=2, d=120, pool=5, seed=11)
        return ho.RelationOperator(P.shape[0], e), P, 150, 61
    if unit == "familyB":
        e, P, _ = fam.build_family_B(d=120, gamma=1 - fam.EPS_WEAK, seed=12)
        return ho.RelationOperator(P.shape[0], e), P, 150, 62
    tag = {"cora": "cora-cc", "dblp": "dblp-ca"}[unit]
    n, edges, labels = data_io.load_graph(tag)
    P = data_io.majority_binary_profile(labels, 0.05)
    npairs = 120 if unit == "cora" else 60
    seed = 63 if unit == "cora" else 64
    return ho.RelationOperator(n, edges), P, npairs, seed


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", default="all",
                    choices=["all"] + list(UNITS))
    args = ap.parse_args()
    units = list(UNITS) if args.unit == "all" else [args.unit]
    outp = RESULTS / "amplitude_scan_v2.csv"
    for unit in units:
        name, profile, _ = UNITS[unit]
        op, P, npairs, seed = build_op(unit)
        rows = scan(name, profile, op, P, npairs, seed)
        exists = outp.exists()
        with open(outp, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if not exists:
                w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in FIELDS})
    print("S10a appended done")


if __name__ == "__main__":
    main()
