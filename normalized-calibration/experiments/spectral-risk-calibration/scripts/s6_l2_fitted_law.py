"""S6: L2 fitted-law calibration pilot (plan 3.4-L2, 5.T5).

True mechanism: binary-eps0.05 on Cora (known law, grade 1).
Fitted law: class frequencies P_hat[u,c] from an INDEPENDENT reference batch
of node-driven draws (n_fit=2000; no test labels used).

Procedure: M_hat, calibration scores, q_alpha all under P_hat (the fitted
model's own node sampler); then evaluate exceedance on
  (a) a fresh P_hat batch (within fitted model: rank guarantee expected), and
  (b) a fresh TRUE-law batch (mismatch cost of estimating r_u).

Writes results/l2_fitted_law.csv. Guarantee labels: (a) grade 2 under fitted
model; (b) grade 4/5 empirical mismatch diagnostic.
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
import data_io  # noqa: E402

RESULTS = HERE / "results"

N_FIT = 2000
N_REF = 500
N_CAL = 500
N_EVAL = 1000
ALPHAS = (0.01, 0.05, 0.10)


def fit_node_law(op, P_true, n_fit, seed):
    rng = np.random.default_rng(seed)
    sampler = ho.bound_node_sampler(op, P_true)
    C = P_true.shape[1]
    counts = np.zeros((op.n, C))
    for _ in range(n_fit):
        _, Y = sampler(rng)
        counts[np.arange(op.n), Y] += 1
    return (counts + 0.5) / (n_fit + 0.5 * C)  # Jeffreys smoothing


def tv_per_node(P, Q):
    return 0.5 * np.abs(P - Q).sum(axis=1)


def main():
    t0 = time.time()
    n, edges, labels = data_io.load_graph("cora-cc")
    op = ho.RelationOperator(n, edges)
    P_true = data_io.majority_binary_profile(labels, 0.05)

    rows = []
    for rep in range(3):
        sd = 6100 + 31 * rep
        P_hat = fit_node_law(op, P_true, N_FIT, sd + 1)
        tv = tv_per_node(P_true, P_hat)
        sampler_hat = ho.bound_node_sampler(op, P_hat)
        sampler_true = ho.bound_node_sampler(op, P_true)

        M_hat = ho.estimate_center(op, sampler_hat, N_REF,
                                   np.random.default_rng(sd + 2))
        cal, _ = ho.draw_scores(op, sampler_hat, N_CAL,
                                np.random.default_rng(sd + 3), M_hat)
        ev_hat, _ = ho.draw_scores(op, sampler_hat, N_EVAL,
                                   np.random.default_rng(sd + 4), M_hat)
        ev_true, _ = ho.draw_scores(op, sampler_true, N_EVAL,
                                    np.random.default_rng(sd + 5), M_hat)
        for alpha in ALPHAS:
            q = ho.rank_threshold(cal, alpha)
            for name, ev in (("fitted-model", ev_hat), ("true-law", ev_true)):
                ex = float(np.mean(ev > q))
                lo, hi = ho.clopper_pearson(int(round(ex * len(ev))), len(ev))
                rows.append({
                    "rep": rep, "alpha": alpha, "eval": name, "q": q,
                    "exceed": ex, "cp_lo": lo, "cp_hi": hi,
                    "mean_tv": float(tv.mean()),
                    "max_tv": float(tv.max()),
                    "n_fit": N_FIT, "n_cal": N_CAL, "n_eval": len(ev),
                })
                print(rows[-1], flush=True)

    with open(RESULTS / "l2_fitted_law.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"S6 done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
