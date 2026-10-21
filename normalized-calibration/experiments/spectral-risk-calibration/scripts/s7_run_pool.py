"""S7 / A3-S8: real-predictor run-pool calibration (plan 3.4-L3, 9-A3).

Mechanism: the manuscript's LP predictor (paper/paper1/.../
multi_dataset_pipeline.py: label_propagation, alpha=0.9, n_iter=50,
n_lab=max(n_cls, n//5), propagation on the candidate hypergraph's
symmetrically normalized clique expansion) is RE-RUN with independent seeds.
Each training run yields a full prediction vector; statistical unit = one run.
Training protocol fixed; only the seed changes (plan: split changes are a
separate mechanism, not mixed in).

Three calibrators, shared frozen center M_hat from the REFERENCE run pool:
  run-pool : calibration scores from calibration-pool runs (grade 3,
             exchangeability under the same repeat-run mechanism)
  node-fit : r_hat_u = class frequencies in reference runs; independent node
             resampling -> S scores (grade 4: fitted product law)
  edge-iid : p_e from r_hat_u marginals (grade 5 misspecified control)
Exceedance always evaluated on INDEPENDENT evaluation-pool runs (the truth).

Writes results/l3_run_pool.csv. Also reports cost (#runs, seconds) per
plan 7: "必须同时报告真实训练次数、模拟次数和总计算时间".
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

import hyperops as ho  # noqa: E402
import data_io  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
ALPHAS = (0.01, 0.05, 0.10)


def build_Snorm(n, edges):
    rows, cols, base = [], [], []
    for e in edges:
        for a in e:
            for b in e:
                rows.append(a); cols.append(b)
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = A + sp.eye(n, format="csr")
    deg = np.asarray(A.sum(axis=1)).ravel()
    d_is = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    return (sp.diags(d_is) @ A @ sp.diags(d_is)).tocsr()


def label_propagation(A, labels, seed, alpha=0.9, n_iter=50):
    """Verbatim protocol from the manuscript (source cited above)."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    n_cls = len(np.unique(labels))
    n_lab = max(n_cls, n // 5)
    idx = rng.choice(n, n_lab, replace=False)
    Y0 = np.zeros((n, n_cls))
    Y0[idx, labels[idx]] = 1.0
    Y = Y0.copy()
    for _ in range(n_iter):
        Y = alpha * (A @ Y) + (1 - alpha) * Y0
    return np.argmax(Y, axis=1)


def make_pool(A, labels, seeds):
    return [label_propagation(A, labels, s) for s in seeds]


def fit_node_law(runs, n_cls, n):
    counts = np.zeros((n, n_cls))
    for Y in runs:
        counts[np.arange(n), Y] += 1
    return (counts + 0.5) / (len(runs) + 0.5 * n_cls)


def main():
    budgets = {"cora-cc": (50, 100, 100), "dblp-ca": (40, 80, 80)}
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cora-cc")
    args = ap.parse_args()
    tag = args.tag
    n_ref, n_cal, n_eval = budgets[tag]

    n, edges, labels = data_io.load_graph(tag)
    op = ho.RelationOperator(n, edges)
    t0 = time.time()
    A = build_Snorm(n, edges)
    seeds_ref = list(range(900001, 900001 + n_ref))
    seeds_cal = list(range(900001 + n_ref, 900001 + n_ref + n_cal))
    seeds_eval = list(range(900001 + n_ref + n_cal,
                           900001 + n_ref + n_cal + n_eval))
    pool_ref = make_pool(A, labels, seeds_ref)
    t_train_ref = time.time() - t0
    pool_cal = make_pool(A, labels, seeds_cal)
    pool_eval = make_pool(A, labels, seeds_eval)
    t_train = time.time() - t0
    print(f"{tag}: {len(pool_ref)+len(pool_cal)+len(pool_eval)} LP runs in "
          f"{t_train:.0f}s", flush=True)

    # frozen center from reference runs
    acc = np.zeros(op.n_entries)
    ref_scores_data = []
    for Y in pool_ref:
        s = op.S_data(op.flags_from_Y(Y))
        acc += s
    M_hat = acc / len(pool_ref)

    # calibration scores: run-pool arm
    cal_run = np.empty(n_cal)
    res_run = np.empty(n_cal)
    for i, Y in enumerate(pool_cal):
        s = op.S_data(op.flags_from_Y(Y))
        cal_run[i], res_run[i] = ho.spectral_norm_entries(op, s, M_hat)

    # node-fitted and edge-iid arms
    P_hat = fit_node_law(pool_ref, P_hat_cls := len(np.unique(labels)), n)
    p_hat = op.edge_marginals(P_hat)
    node_s = ho.bound_node_sampler(op, P_hat)
    edge_s = ho.bound_edge_sampler(op, p_hat)
    t1 = time.time()
    cal_node, _ = ho.draw_scores(op, node_s, n_cal,
                                 np.random.default_rng(7701), M_hat)
    cal_edge, _ = ho.draw_scores(op, edge_s, n_cal,
                                 np.random.default_rng(7702), M_hat)
    t_sim = time.time() - t1
    print(f"{tag}: sim scores done ({t_sim:.0f}s), "
          f"train {t_train:.0f}s", flush=True)

    # evaluation: independent runs
    ev = np.empty(n_eval)
    ev_res = np.empty(n_eval)
    for i, Y in enumerate(pool_eval):
        s = op.S_data(op.flags_from_Y(Y))
        ev[i], ev_res[i] = ho.spectral_norm_entries(op, s, M_hat)

    rows = []
    for arm, cal in (("run-pool", cal_run), ("node-fit", cal_node),
                     ("edge-iid", cal_edge)):
        for alpha in ALPHAS:
            q = ho.rank_threshold(cal, alpha)
            ex = float(np.mean(ev > q))
            lo, hi = ho.clopper_pearson(int(round(ex * n_eval)), n_eval)
            rows.append({
                "tag": tag, "predictor": "LP", "arm": arm, "alpha": alpha,
                "q": q, "exceed_on_eval_runs": ex, "cp_lo": lo, "cp_hi": hi,
                "n_runs_train": n_ref + n_cal + n_eval,
                "n_sim_cal": n_cal, "secs_train": round(t_train, 1),
                "secs_sim": round(t_sim, 1),
                "approx_frac": float(np.mean(ev_res > 1e-5)),
            })
            print(rows[-1], flush=True)

    outp = RESULTS / "l3_run_pool.csv"
    header = list(rows[0].keys())
    write_header = not outp.exists()
    with open(outp, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"S7 {tag} done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
