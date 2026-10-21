"""S8: L3 run-pool for GCN and nearest-prototype predictors (plan A3/S8).

Protocol inheritance:
  GCN: gcn_crossfit from paper/paper1/.../gcn_diagnose.py — the manuscript's
        tab:real-main GCN protocol (NOT the degenerate 16-hidden in-sample
        variant found in diagnostics): row-normalized features, 5-fold
        cross-fitting, train_gcn_masked(hidden=64, epochs=500, lr=0.2),
        held-out prediction per node. Run randomness = fold-split seeds +
        init seeds (split-randomization mechanism family).
  Proto: realdata_fixed.py Arm-2: seed -> permutation, train_idx = first
        n//2 nodes, class-mean prototypes, nearest prototype prediction;
        run randomness = TRAIN SPLIT randomization (separate mechanism,
        separate rows; not mixed with seed-only).

DBLP GCN uses epochs=150 (documented reduced protocol for cost; record in
README when results are published).

Calibration arms (shared frozen center from reference run pool):
  run-pool / node-fit (r_hat from ref runs) / edge-iid (p_e from r_hat).
Evaluation: independent evaluation-run pool (the truth). Appends rows to
results/l3_run_pool.csv with the predictor column naming the mechanism.
Cost columns report real training count and seconds (plan section 7).
"""

from __future__ import annotations

import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

import hyperops as ho  # noqa: E402
import data_io  # noqa: E402
import s7_run_pool as s7  # for build_Snorm / fit_node_law reuse

RESULTS = HERE / "results"
HIDDEN, EPOCHS, LR = 64, 500, 0.2   # manuscript gcn_diagnose values


def train_gcn_masked(X, A, labels, n_cls, seed, train_mask,
                     hidden=HIDDEN, epochs=EPOCHS, lr=LR):
    """Verbatim manuscript gcn_diagnose.train_gcn_masked protocol."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W1 = rng.standard_normal((d, hidden)) * np.sqrt(2.0 / d)
    W2 = rng.standard_normal((hidden, n_cls)) * np.sqrt(2.0 / hidden)
    Y = np.eye(n_cls)[labels]
    ntr = int(train_mask.sum())
    for ep in range(epochs):
        Z1 = X @ W1
        H1 = np.maximum(A @ Z1, 0)
        H2 = A @ (H1 @ W2)
        P = np.exp(H2 - H2.max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        dP = (P - Y) * train_mask[:, None] / ntr
        dW2 = H1.T @ (A @ dP)
        dH1 = (A @ (dP @ W2.T)) * (H1 > 0)
        dW1 = X.T @ (A @ dH1)
        W2 -= lr * dW2
        W1 -= lr * dW1
    return W1, W2


def predict_gcn(X, A, W1, W2):
    H1 = np.maximum(A @ (X @ W1), 0)
    return np.argmax(A @ (H1 @ W2), axis=1)


def gcn_crossfit(feats, labels, A, n_cls, seed, epochs=EPOCHS):
    """Manuscript gcn_crossfit protocol, sparse-X equivalent.

    The manuscript densifies row-normalized features; here the row
    normalization keeps the csr structure (X entries scaled by row norm)
    and all matvecs use spmm — mathematically identical products, ~40x
    faster on these sparse feature matrices. Numerically verified against
    the dense path on Cora (tests/test_s8_equiv.py).
    """
    X = feats.copy() if sp.issparse(feats) else sp.csr_matrix(feats)
    X = X.astype(np.float64)
    norm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    inv = np.where(norm > 1e-12, 1.0 / np.maximum(norm, 1e-12), 0.0)
    data = X.data * np.repeat(inv, np.diff(X.indptr))
    X = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)
    n = len(labels)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), 5)
    pred = np.zeros(n, dtype=int)
    val_acc = []
    for f in range(5):
        test = folds[f]
        train = np.concatenate([folds[g] for g in range(5) if g != f])
        mask = np.zeros(n, dtype=bool)
        mask[train] = True
        W1, W2 = train_gcn_masked(X, A, labels, n_cls, seed * 10 + f, mask,
                                  epochs=epochs)
        p = predict_gcn(X, A, W1, W2)
        pred[test] = p[test]
        val_acc.append(float(np.mean(p[test] == labels[test])))
    return pred.astype(int), float(np.mean(val_acc))


def proto_predict(feats, labels, seed, n_cls):
    """Verbatim manuscript Arm-2: random half-split prototypes."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    idx = rng.permutation(n)
    train_idx = idx[: n // 2]
    protos = np.zeros((n_cls, feats.shape[1]))
    for c in range(n_cls):
        mask = labels[train_idx] == c
        if mask.sum() > 0:
            protos[c] = np.asarray(
                feats[train_idx][mask].mean(axis=0)).ravel()
    # nearest prototype, blocked
    pred = np.zeros(n, dtype=int)
    pnorm = (protos ** 2).sum(axis=1)
    for a in range(0, n, 5000):
        b = min(n, a + 5000)
        block = np.asarray(feats[a:b].todense()) if hasattr(
            feats[a], "todense") else feats[a:b]
        dist = (block ** 2).sum(axis=1, keepdims=True) - 2 * block @ protos.T \
            + pnorm
        pred[a:b] = dist.argmin(axis=1)
    return pred


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cora-cc")
    ap.add_argument("--predictor", default="gcn", choices=["gcn", "proto"])
    args = ap.parse_args()
    tag, pred_name = args.tag, args.predictor

    if tag == "cora-cc":
        n_ref, n_cal, n_eval = 30, 100, 150
    else:
        n_ref, n_cal, n_eval = 20, 50, 80
    if pred_name == "gcn" and tag == "cora-cc":
        n_ref, n_cal, n_eval = 20, 40, 60   # 28 s/run, protocol-faithful

    n, edges, labels = data_io.load_graph(tag)
    feats = pickle.load(open(f"{data_io.ROOT}/data/{tag}/features.pickle",
                             "rb"))
    op = ho.RelationOperator(n, edges)
    A = s7.build_Snorm(n, edges)
    n_cls = len(np.unique(labels))
    gcn_epochs = 500 if tag == "cora-cc" else 120  # reduced dblp protocol
    pred_label = {"gcn": ("gcn5f" if tag == "cora-cc" else "gcn5f-e120"),
                  "proto": "proto"}[pred_name]

    def predictor(seed):
        if pred_name == "gcn":
            return gcn_crossfit(feats, labels, A, n_cls, seed,
                                epochs=gcn_epochs)[0]
        return proto_predict(feats, labels, seed, n_cls)

    t0 = time.time()
    base = {"gcn": 9000001, "proto": 700001}[pred_name]
    pools = []
    acc = 0
    for role, cnt in (("ref", n_ref), ("cal", n_cal), ("eval", n_eval)):
        rng_t = time.time()
        ys = [predictor(base + acc + s) for s in range(cnt)]
        acc += cnt
        pools.append(ys)
        print(f"{tag}/{pred_name} {role}: {cnt} runs "
              f"{time.time()-rng_t:.0f}s acc={np.mean([float(np.mean(y==labels)) for y in ys]):.3f}", flush=True)
    t_train = time.time() - t0
    pool_ref, pool_cal, pool_eval = pools

    s = np.zeros(op.n_entries)
    for Y in pool_ref:
        s += op.S_data(op.flags_from_Y(Y))
    M_hat = s / len(pool_ref)

    def run_scores(pool):
        out = np.empty(len(pool))
        out_r = np.empty(len(pool))
        for i, Y in enumerate(pool):
            out[i], out_r[i] = ho.spectral_norm_entries(
                op, op.S_data(op.flags_from_Y(Y)), M_hat)
        return out

    t1 = time.time()
    cal_run = run_scores(pool_cal)
    ev = run_scores(pool_eval)

    P_hat = s7.fit_node_law(pool_ref, n_cls, n)
    p_hat = op.edge_marginals(P_hat)
    cal_node, _ = ho.draw_scores(op, ho.bound_node_sampler(op, P_hat),
                                 n_cal, np.random.default_rng(8801), M_hat)
    cal_edge, _ = ho.draw_scores(op, ho.bound_edge_sampler(op, p_hat),
                                 n_cal, np.random.default_rng(8802), M_hat)
    t_sim = time.time() - t1

    rows = []
    for arm, cal in (("run-pool", cal_run), ("node-fit", cal_node),
                     ("edge-iid", cal_edge)):
        for alpha in (0.01, 0.05, 0.10):
            q = ho.rank_threshold(cal, alpha)
            ex = float(np.mean(ev > q))
            lo, hi = ho.clopper_pearson(int(round(ex * len(ev))), len(ev))
            rows.append({
                "tag": tag, "predictor": pred_label, "arm": arm, "alpha": alpha,
                "q": q, "exceed_on_eval_runs": ex, "cp_lo": lo, "cp_hi": hi,
                "n_runs_train": n_ref + n_cal + n_eval,
                "n_sim_cal": n_cal, "secs_train": round(t_train, 1),
                "secs_sim": round(t_sim, 1),
                "approx_frac": 0.0,
            })
            print(rows[-1], flush=True)

    outp = RESULTS / "l3_run_pool.csv"
    header = list(rows[0].keys())
    exists = outp.exists()
    with open(outp, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerows(rows)
    print(f"S8 {tag}/{pred_name} done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
