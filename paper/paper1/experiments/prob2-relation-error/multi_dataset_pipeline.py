"""多数据集 pipeline 验证：DBLP (GCN+LP) + Walmart-trips (LP)。

Cora 已验证（GCN 3.74×, LP 2.11×）。本脚本补齐：
  1. GCN on DBLP-CA（41K 节点，有特征，需高效稀疏运算）
  2. LP on DBLP-CA
  3. LP on Walmart-trips（89K 节点，无特征，边大小 2-25）
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

sys.stdout.reconfigure(encoding="utf-8")

DATA_ROOT = "C:/Users/Peng1/OneDrive/文档/ChatGPT/hypergraph/data"
DBLP = f"{DATA_ROOT}/dblp-ca"
WALMART = f"{DATA_ROOT}/web-hypergraphs/walmart-trips"
N_CAL, N_EVAL, N_B, N_BOOT = 20, 20, 20, 200
HIDDEN, EPOCHS, LR = 16, 100, 0.05


# ---------- 数据加载 ----------

def load_dblp():
    with open(f"{DBLP}/features.pickle", "rb") as f:
        feats = pickle.load(f)
    with open(f"{DBLP}/hypergraph.pickle", "rb") as f:
        hg = pickle.load(f)
    with open(f"{DBLP}/labels.pickle", "rb") as f:
        labels = np.array(pickle.load(f))
    edges = [sorted(int(u) for u in s) for s in hg.values()]
    return feats, edges, labels


def load_walmart(max_nodes=20000):
    """Walmart-trips：取前 max_nodes 个节点的导出子图（控制规模）。"""
    edges_raw = []
    with open(f"{WALMART}/hyperedges-walmart-trips.txt") as f:
        for line in f:
            e = sorted(set(int(x) - 1 for x in line.strip().split(",")))
            if all(u < max_nodes for u in e):
                edges_raw.append(e)
    labels = []
    with open(f"{WALMART}/node-labels-walmart-trips.txt") as f:
        for i, line in enumerate(f):
            if i >= max_nodes:
                break
            labels.append(int(line.strip()) - 1)
    labels = np.array(labels)
    # 重新编号（去掉孤立节点）
    used = sorted(set(u for e in edges_raw for u in e))
    remap = {u: i for i, u in enumerate(used)}
    edges = [tuple(remap[u] for u in e) for e in edges_raw]
    labels = labels[used]
    # 只保留有边的节点的标签
    return edges, labels


# ---------- 图构造 ----------

def build_adj_sparse(n, edges):
    """团展开 → 对称归一化 + 自环（稀疏）。"""
    rows, cols = [], []
    for e in edges:
        for a in e:
            for b in e:
                rows.append(a); cols.append(b)
    data = np.ones(len(rows))
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    A = A + sp.eye(n, format="csr")
    deg = np.asarray(A.sum(axis=1)).ravel()
    d_is = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    return (sp.diags(d_is) @ A @ sp.diags(d_is)).tocsr()


# ---------- 预测器 ----------

def gcn_predict_sparse(feats, labels, A, n_cls, seed):
    """2 层 GCN，稀疏特征 + 稀疏邻接。"""
    rng = np.random.default_rng(seed)
    n, d = feats.shape
    W1 = rng.standard_normal((d, HIDDEN)) * 0.1
    W2 = rng.standard_normal((HIDDEN, n_cls)) * 0.1
    Y = np.zeros((n, n_cls))
    Y[np.arange(n), labels] = 1.0
    X = sp.csr_matrix(feats) if not sp.issparse(feats) else feats
    for ep in range(EPOCHS):
        Z1 = (X @ W1).toarray() if sp.issparse(X @ W1) else (X @ W1)
        H1 = np.maximum(A @ Z1, 0)
        H2 = A @ (H1 @ W2)
        expH = np.exp(H2 - H2.max(axis=1, keepdims=True))
        P = expH / expH.sum(axis=1, keepdims=True)
        dP = (P - Y) / n
        dW2 = H1.T @ (A @ dP)
        dH1 = (A @ (dP @ W2.T)) * (H1 > 0)
        dW1 = (X.T @ (A @ dH1)).toarray() if sp.issparse(X.T @ (A @ dH1)) \
            else X.T @ (A @ dH1)
        W2 -= LR * dW2
        W1 -= LR * dW1
    H1 = np.maximum(A @ np.asarray(X @ W1), 0)
    H2 = A @ (H1 @ W2)
    return np.argmax(H2, axis=1)


def label_propagation(A, labels, seed, alpha=0.9, n_iter=50):
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


# ---------- 中心化 A/B ----------

def centered_ab(name, edges, labels, predictor, feats=None):
    n = len(labels)
    a_star = np.array([1.0 if len(set(labels[u] for u in e)) == 1 else 0.0
                       for e in edges])
    m = len(edges)
    sgn = np.where(a_star == 1, -1.0, 1.0)
    A_hat = build_adj_sparse(n, edges)
    n_cls = len(np.unique(labels))

    D_cal, D_eval = [], []
    accs = []
    for group, D_list, base in [("cal", D_cal, 41000), ("eval", D_eval, 42000)]:
        for s in range(N_CAL if group == "cal" else N_EVAL):
            if feats is not None:
                pred = predictor(feats, labels, A_hat, n_cls, base + s)
            else:
                pred = predictor(A_hat, labels, base + s)
            accs.append(np.mean(pred == labels))
            d = np.array([1.0 if len(set(pred[u] for u in e)) == 1
                          else 0.0 for e in edges]) - a_star
            D_list.append(d)

    D_cal = np.array(D_cal).astype(float)
    D_eval = np.array(D_eval).astype(float)
    q_hat = (D_cal != 0).mean(axis=0)
    d_bar = D_cal.mean(axis=0)

    rng_B = np.random.default_rng(43000)
    dB = np.array([np.where(rng_B.random(m) < q_hat, sgn, 0.0)
                   for _ in range(N_B)])

    def build_op(w):
        rows, cols, vals = [], [], []
        for e, wt in zip(edges, w):
            if abs(wt) < 1e-15:
                continue
            k = len(e)
            for a in e:
                for b in e:
                    rows.append(a); cols.append(b); vals.append(wt / k)
        return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    def snorm(M):
        if M.nnz == 0:
            return 0.0
        try:
            la = float(eigsh(M, k=1, which="LA", tol=1e-4,
                             maxiter=2000, return_eigenvectors=False)[0])
        except Exception:
            la = 0.0
        try:
            li = float(eigsh(M, k=1, which="SA", tol=1e-4,
                             maxiter=2000, return_eigenvectors=False)[0])
        except Exception:
            li = 0.0
        return max(abs(la), abs(li))

    cenA = [snorm(build_op(D_eval[j] - d_bar)) for j in range(N_EVAL)]
    cenB = [snorm(build_op(dB[k] - sgn * q_hat)) for k in range(N_B)]
    ratio = float(np.mean(cenA) / np.mean(cenB))

    rng = np.random.default_rng(99)
    ratios = []
    for _ in range(N_BOOT):
        ie = rng.integers(0, N_EVAL, 5)
        ib = rng.integers(0, N_B, 5)
        ic = rng.integers(0, N_CAL, N_CAL)
        d_b = D_cal[ic].mean(axis=0)
        q_b = (D_cal[ic] != 0).mean(axis=0)
        ca = [snorm(build_op(D_eval[j] - d_b)) for j in ie]
        cb = [snorm(build_op(np.where(rng.random(m) < q_b, sgn, 0.0)
                             - sgn * q_b)) for _ in range(5)]
        ratios.append(np.mean(ca) / max(np.mean(cb), 1e-12))
    lo, hi = np.quantile(ratios, [0.025, 0.975])
    return dict(name=name, ratio=ratio, ci=[float(lo), float(hi)],
                acc=float(np.mean(accs)),
                cenA=float(np.mean(cenA)), cenB=float(np.mean(cenB)))


def main():
    results = []
    print("=" * 90)

    # --- DBLP: GCN + LP ---
    print("\n[DBLP-CA] 加载...")
    feats, edges, labels = load_dblp()
    print(f"  n={len(labels)}, m={len(edges)}")

    print("[DBLP GCN] 训练中...")
    r = centered_ab("DBLP-GCN", edges, labels,
                    lambda f, l, A, c, s: gcn_predict_sparse(f, l, A, c, s),
                    feats=feats)
    results.append(r)
    print(f"  acc={r['acc']:.3f}, ratio={r['ratio']:.3f}, "
          f"CI=[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]")

    print("[DBLP LP] 运行中...")
    r = centered_ab("DBLP-LP", edges, labels,
                    lambda A, l, s: label_propagation(A, l, s))
    results.append(r)
    print(f"  acc={r['acc']:.3f}, ratio={r['ratio']:.3f}, "
          f"CI=[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]")

    # --- Walmart: LP ---
    print("\n[Walmart-trips] 加载（前 20K 节点）...")
    edges_w, labels_w = load_walmart(max_nodes=20000)
    print(f"  n={len(labels_w)}, m={len(edges_w)}")

    print("[Walmart LP] 运行中...")
    r = centered_ab("Walmart-LP", edges_w, labels_w,
                    lambda A, l, s: label_propagation(A, l, s))
    results.append(r)
    print(f"  acc={r['acc']:.3f}, ratio={r['ratio']:.3f}, "
          f"CI=[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]")

    with open("results/multi_dataset_pipeline.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("\n" + "=" * 90)
    print("汇总（含已有 Cora 结果）：")
    print(f"{'数据集':>15} {'预测器':>10} | {'准确率':>6} {'A/B':>6} "
          f"{'95% CI':>17} | CI>1?")
    known = [
        ("Cora", "GCN", 0.249, 3.741, [2.695, 5.125]),
        ("Cora", "LP", 0.574, 2.106, [1.569, 3.168]),
        ("Cora", "Proto", 0.69, 2.36, [2.02, 3.20]),
    ]
    for ds, pred, acc, ratio, ci in known:
        print(f"{ds:>15} {pred:>10} | {acc:>6.3f} {ratio:>6.3f} "
              f"[{ci[0]:.3f},{ci[1]:.3f}] | {'✓' if ci[0] > 1 else '✗'}")
    for r in results:
        print(f"{r['name']:>15} {'':>10} | {r['acc']:>6.3f} "
              f"{r['ratio']:>6.3f} [{r['ci'][0]:.3f},{r['ci'][1]:.3f}] | "
              f"{'✓' if r['ci'][0] > 1 else '✗'}")
    print("\n已写出 results/multi_dataset_pipeline.json")


if __name__ == "__main__":
    main()
