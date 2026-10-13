"""GCN 训练诊断 + 正常训练配置下的中心化谱比较。

诊断项：loss 曲线、训练精度、预测类别分布、标签映射。
修复配置：5-fold 交叉拟合（held-out 评估）、hidden 64、300 epochs、
lr 0.01，输出每折验证精度。然后用修复后的 GCN 重跑中心化 A/B。
"""

from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

sys.stdout.reconfigure(encoding="utf-8")

CORA = "C:/Users/Peng1/OneDrive/文档/ChatGPT/hypergraph/data/cora-cc"
N_CAL, N_EVAL, N_B, N_BOOT = 20, 20, 20, 200


def load():
    with open(f"{CORA}/features.pickle", "rb") as f:
        feats = pickle.load(f)
    with open(f"{CORA}/hypergraph.pickle", "rb") as f:
        hg = pickle.load(f)
    with open(f"{CORA}/labels.pickle", "rb") as f:
        labels = np.array(pickle.load(f))
    edges = [sorted(int(u) for u in s) for s in hg.values()]
    return feats, edges, labels


def build_adj(n, edges):
    rows, cols = [], []
    for e in edges:
        for a in e:
            for b in e:
                rows.append(a); cols.append(b)
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = A + sp.eye(n, format="csr")
    deg = np.asarray(A.sum(axis=1)).ravel()
    d_is = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    return (sp.diags(d_is) @ A @ sp.diags(d_is)).tocsr()


def train_gcn(X, A, labels, n_cls, seed, hidden=64, epochs=500, lr=0.2):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W1 = rng.standard_normal((d, hidden)) * np.sqrt(2.0 / d)
    W2 = rng.standard_normal((hidden, n_cls)) * np.sqrt(2.0 / hidden)
    Y = np.eye(n_cls)[labels]
    losses = []
    for ep in range(epochs):
        Z1 = X @ W1
        H1 = np.maximum(A @ Z1, 0)
        H2 = A @ (H1 @ W2)
        P = np.exp(H2 - H2.max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        loss = -np.mean(np.log(P[np.arange(n), labels] + 1e-12))
        if ep % 50 == 0 or ep == epochs - 1:
            losses.append((ep, float(loss)))
        dP = (P - Y) / n
        dW2 = H1.T @ (A @ dP)
        dH1 = (A @ (dP @ W2.T)) * (H1 > 0)
        dW1 = X.T @ (A @ dH1)
        W2 -= lr * dW2
        W1 -= lr * dW1
    return W1, W2, losses


def predict(X, A, W1, W2):
    H1 = np.maximum(A @ (X @ W1), 0)
    return np.argmax(A @ (H1 @ W2), axis=1)


def gcn_crossfit(feats, labels, A, n_cls, seed):
    """5-fold 交叉拟合：每折训练 4/5，预测 held-out 1/5。"""
    X = feats.toarray().astype(np.float64) \
        if sp.issparse(feats) else np.asarray(feats, dtype=np.float64)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    n = len(labels)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), 5)
    pred = np.zeros(n, dtype=int)
    val_acc = []
    for f in range(5):
        test = folds[f]
        train = np.concatenate([folds[g] for g in range(5) if g != f])
        Xtr_mask = np.zeros(n, dtype=bool)
        Xtr_mask[train] = True
        # 训练时只允许用训练折的损失（屏蔽测试折）
        W1, W2, _ = train_gcn_masked(X, A, labels, n_cls, seed * 10 + f,
                                     train_mask=Xtr_mask)
        p = predict(X, A, W1, W2)
        pred[test] = p[test]
        val_acc.append(float(np.mean(p[test] == labels[test])))
    return pred, float(np.mean(val_acc))


def train_gcn_masked(X, A, labels, n_cls, seed, train_mask,
                     hidden=64, epochs=500, lr=0.2):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W1 = rng.standard_normal((d, hidden)) * np.sqrt(2.0 / d)
    W2 = rng.standard_normal((hidden, n_cls)) * np.sqrt(2.0 / hidden)
    Y = np.eye(n_cls)[labels]
    ntr = train_mask.sum()
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
    return W1, W2, None


def build_op(n, edges, weights):
    rows, cols, vals = [], [], []
    for e, w in zip(edges, weights):
        if not w:
            continue
        k = len(e)
        for a in e:
            for b in e:
                rows.append(a); cols.append(b); vals.append(w / k)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def snorm(M):
    if M.nnz == 0:
        return 0.0
    la = float(eigsh(M, k=1, which="LA", return_eigenvectors=False,
                     tol=1e-5, maxiter=5000)[0])
    li = float(eigsh(M, k=1, which="SA", return_eigenvectors=False,
                     tol=1e-5, maxiter=5000)[0])
    return max(abs(la), abs(li))


def main():
    feats, edges, labels = load()
    n = len(labels)
    n_cls = len(np.unique(labels))
    a_star = np.array([1.0 if len(set(labels[u] for u in e)) == 1 else 0.0
                       for e in edges])
    A_hat = build_adj(n, edges)
    X = feats.toarray().astype(np.float64) \
        if sp.issparse(feats) else np.asarray(feats)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    # ---- 诊断 1：旧配置（hidden16/100ep/lr0.05 全监督）----
    print("=== 旧配置诊断 ===")
    W1o = np.random.default_rng(0).standard_normal((X.shape[1], 16)) * 0.1
    rng_o = np.random.default_rng(0)
    W1o = rng_o.standard_normal((X.shape[1], 16)) * 0.1
    W2o = rng_o.standard_normal((16, n_cls)) * 0.1
    Yo = np.eye(n_cls)[labels]
    for ep in range(101):
        Z1 = X @ W1o
        H1 = np.maximum(A_hat @ Z1, 0)
        H2 = A_hat @ (H1 @ W2o)
        P = np.exp(H2 - H2.max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        if ep % 20 == 0:
            loss = -np.mean(np.log(P[np.arange(n), labels] + 1e-12))
            acc = float(np.mean(np.argmax(H2, axis=1) == labels))
            print(f"  ep{ep}: loss={loss:.4f} train-acc={acc:.3f}")
        dP = (P - Yo) / n
        dW2 = H1.T @ (A_hat @ dP)
        dH1 = (A_hat @ (dP @ W2o.T)) * (H1 > 0)
        dW1 = X.T @ (A_hat @ dH1)
        W2o -= 0.05 * dW2
        W1o -= 0.05 * dW1
    pred_o = np.argmax(A_hat @ (np.maximum(A_hat @ (X @ W1o), 0) @ W2o),
                       axis=1)
    dist_o = np.bincount(pred_o, minlength=n_cls) / n
    print(f"  最终 train-acc={np.mean(pred_o == labels):.3f}")
    print(f"  预测类分布: {dist_o.round(3)}")
    print(f"  真实类分布: {(np.bincount(labels, minlength=n_cls) / n).round(3)}")

    # ---- 诊断 2 + 修复配置：交叉拟合 held-out ----
    print("\n=== 修复配置（5-fold 交叉拟合, hidden64, 300ep, lr0.01） ===")
    W1, W2, losses = train_gcn(Xn, A_hat, labels, n_cls, seed=1)
    print(f"  全监督 loss 曲线: "
          + ", ".join(f"ep{e}:{l:.3f}" for e, l in losses))
    pred_full = predict(Xn, A_hat, W1, W2)
    print(f"  全监督 train-acc={np.mean(pred_full == labels):.3f}")

    preds, accs = [], []
    for s in range(5):
        pred, va = gcn_crossfit(feats, labels, A_hat, n_cls, 100 + s)
        preds.append(pred)
        accs.append(va)
        print(f"  seed {100 + s}: held-out acc={va:.3f}")

    # ---- 修复后 GCN 的中心化 A/B ----
    print("\n=== 修复后 GCN 中心化 A/B ===")
    D_cal, D_eval = [], []
    for base, dst in ((41000, D_cal), (42000, D_eval)):
        for s in range(N_CAL if dst is D_cal else N_EVAL):
            pred, _ = gcn_crossfit(feats, labels, A_hat, n_cls, base + s)
            dst.append(np.array(
                [1.0 if len(set(pred[u] for u in e)) == 1 else 0.0
                 for e in edges]) - a_star)
    D_cal = np.array(D_cal, dtype=float)
    D_eval = np.array(D_eval, dtype=float)
    d_bar = D_cal.mean(axis=0)
    q_hat = (D_cal != 0).mean(axis=0)
    sgn = np.where(a_star == 1, -1.0, 1.0)
    rng = np.random.default_rng(43000)
    cenA = [snorm(build_op(n, edges, D_eval[j] - d_bar))
            for j in range(N_EVAL)]
    cenB = []
    for _ in range(N_B):
        dB = np.where(rng.random(len(edges)) < q_hat, sgn, 0.0)
        cenB.append(snorm(build_op(n, edges, dB - sgn * q_hat)))
    ratio = float(np.mean(cenA) / np.mean(cenB))
    rng_bs = np.random.default_rng(99)
    ie = rng_bs.integers(0, N_EVAL, (N_BOOT, N_EVAL))
    ib = rng_bs.integers(0, N_B, (N_BOOT, N_B))
    ic = rng_bs.integers(0, N_CAL, (N_BOOT, N_CAL))
    boots = []
    for b in range(N_BOOT):
        d_b = D_cal[ic[b]].mean(axis=0)
        q_b = (D_cal[ic[b]] != 0).mean(axis=0)
        ca = [snorm(build_op(n, edges, D_eval[j] - d_b)) for j in ie[b]]
        cb = [snorm(build_op(n, edges,
                             np.where(rng.random(len(edges)) < q_b,
                                      sgn, 0.0) - sgn * q_b))
              for _ in range(5)]
        boots.append(np.mean(ca) / max(np.mean(cb), 1e-12))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    print(f"  held-out acc={np.mean(accs):.3f}")
    print(f"  cenA={np.mean(cenA):.3f} cenB={np.mean(cenB):.3f} "
          f"ratio={ratio:.3f} CI[{lo:.3f},{hi:.3f}]")
    json.dump(dict(heldout_acc=float(np.mean(accs)),
                   cenA=float(np.mean(cenA)), cenB=float(np.mean(cenB)),
                   ratio=ratio, ci=[float(lo), float(hi)]),
              open("results/gcn_fixed.json", "w"), indent=1)


if __name__ == "__main__":
    main()
