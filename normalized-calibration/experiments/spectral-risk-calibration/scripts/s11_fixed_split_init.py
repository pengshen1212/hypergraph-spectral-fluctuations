"""S11: fixed training split, initialization-only randomness (review round 4).

Addresses the A3/S8 gap explicitly requested in the review: prior predictors
mixed THREE different randomness mechanisms —
  LP: random labeled set (split-like), prototy: random half split,
  GCN5f: cross-fit fold randomization;
none isolates "fixed split, only initialization".

Protocol: choose a FIXED labeled subset (20% of nodes, seed 7 once),
train the manuscript's semi-supervised GCN (hidden 64, 500 epochs, lr 0.2,
train_gcn_masked) on exactly that subset, predict all nodes; each RUN uses
only a new initialization seed. Pool ref/cal/eval -> same three calibrators
as S7/S8 (run-pool / node-fit / edge-iid), evaluation on independent runs.

Writes/append results/l3_fixed_split.csv.
"""

from __future__ import annotations

import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hyperops as ho  # noqa: E402
import data_io  # noqa: E402
import s7_run_pool as s7  # noqa: E402
import s8_predictor_matrix as s8  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def build_fixed_split(n, seed=7, frac=0.20):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)[: int(frac * n)]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cora-cc", choices=["cora-cc", "dblp-ca"])
    args = ap.parse_args()
    tag = args.tag
    if tag == "cora-cc":
        n_ref, n_cal, n_eval, epochs = 20, 40, 60, 500
    else:
        n_ref, n_cal, n_eval, epochs = 15, 25, 40, 120

    n, edges, labels = data_io.load_graph(tag)
    feats = pickle.load(open(f"{data_io.ROOT}/data/{tag}/features.pickle", "rb"))
    op = ho.RelationOperator(n, edges)
    A = s7.build_Snorm(n, edges)
    n_cls = len(np.unique(labels))
    mask = build_fixed_split(n)
    X = feats.copy().astype(np.float64)
    norm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    inv = np.where(norm > 1e-12, 1.0 / np.maximum(norm, 1e-12), 0.0)
    X = X.multiply(inv[:, None]).tocsr()

    def run(seed):
        W1, W2 = s8.train_gcn_masked(X, A, labels, n_cls, seed, mask,
                                     epochs=epochs)
        return s8.predict_gcn(X, A, W1, W2)

    t0 = time.time()
    base = 9700001
    pools = []
    acc = 0
    for role, cnt in (("ref", n_ref), ("cal", n_cal), ("eval", n_eval)):
        rng_t = time.time()
        ys = [run(base + acc + s) for s in range(cnt)]
        acc += cnt
        pools.append(ys)
        print(f"{tag} fixed-split {role}: {cnt} runs "
              f"{time.time()-rng_t:.0f}s "
              f"acc={np.mean([np.mean(y == labels) for y in ys]):.3f}",
              flush=True)
    t_train = time.time() - t0
    pool_ref, pool_cal, pool_eval = pools

    s = np.zeros(op.n_entries)
    for Y in pool_ref:
        s += op.S_data(op.flags_from_Y(Y))
    M_hat = s / len(pool_ref)

    def run_scores(pool):
        out = np.empty(len(pool))
        for i, Y in enumerate(pool):
            out[i] = ho.spectral_norm_entries(
                op, op.S_data(op.flags_from_Y(Y)), M_hat)[0]
        return out

    t1 = time.time()
    cal_run = run_scores(pool_cal)
    ev = run_scores(pool_eval)
    P_hat = s7.fit_node_law(pool_ref, n_cls, n)
    p_hat = op.edge_marginals(P_hat)
    cal_node, _ = ho.draw_scores(op, ho.bound_node_sampler(op, P_hat),
                                 n_cal, np.random.default_rng(9901), M_hat)
    cal_edge, _ = ho.draw_scores(op, ho.bound_edge_sampler(op, p_hat),
                                 n_cal, np.random.default_rng(9902), M_hat)
    t_sim = time.time() - t1

    rows = []
    for arm, cal in (("run-pool", cal_run), ("node-fit", cal_node),
                     ("edge-iid", cal_edge)):
        for alpha in (0.05, 0.10):
            q = ho.rank_threshold(cal, alpha)
            ex = float(np.mean(ev > q))
            lo, hi = ho.clopper_pearson(int(round(ex * len(ev))), len(ev))
            rows.append({"tag": tag, "predictor": "gcn-fixedsplit-init",
                         "arm": arm, "alpha": alpha, "q": q,
                         "exceed_on_eval_runs": ex, "cp_lo": lo, "cp_hi": hi,
                         "n_runs_train": n_ref + n_cal + n_eval,
                         "n_sim_cal": n_cal, "secs_train": round(t_train, 1),
                         "secs_sim": round(t_sim, 1),
                         "label_frac": 0.20, "epochs": epochs})
            print(rows[-1], flush=True)

    outp = RESULTS / "l3_fixed_split.csv"
    exists = outp.exists()
    with open(outp, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(rows)
    print(f"S11 {tag} done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
