"""归因实验扩展：GCN 与 LP 预测器（图耦合预测，通道 (c) 检验）。

原型预测器的预测只依赖特征（结构性无通道 (c)）。GCN/LP 的预测依赖图
本身——若它们的不相交边对协方差仍为噪声地板，则通道 (c) 也不贡献。
协议同 attribution_fixed.py：真实臂 J 个种子 vs 节点边际臂 vs 边臂。
"""

from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, ".")
from multi_dataset_pipeline import (build_adj_sparse, gcn_predict_sparse,
                                    label_propagation)

CORA = "C:/Users/Peng1/OneDrive/文档/ChatGPT/hypergraph/data/cora-cc"
N_NODE_DRAWS = 200
N_EDGE_DRAWS = 200
N_BOOT = 2000


def load():
    with open(f"{CORA}/features.pickle", "rb") as f:
        feats = pickle.load(f)
    with open(f"{CORA}/hypergraph.pickle", "rb") as f:
        hg = pickle.load(f)
    with open(f"{CORA}/labels.pickle", "rb") as f:
        labels = np.array(pickle.load(f))
    edges = [sorted(int(u) for u in s) for s in hg.values()]
    return feats, edges, labels


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
    m = len(edges)
    k_max = max(len(e) for e in edges)
    edge_arr = np.full((m, k_max), -1, dtype=int)
    for i, e in enumerate(edges):
        edge_arr[i, :len(e)] = e

    def all_hom(pred):
        first = edge_arr[:, 0]
        same = pred[edge_arr] == pred[first][:, None]
        same[edge_arr < 0] = True
        return same.all(axis=1).astype(float)

    a_star = all_hom(labels)
    A_hat = build_adj_sparse(n, edges)

    results = {}
    for pname, gen in (
            ("GCN", lambda s: gcn_predict_sparse(feats, labels, A_hat,
                                                 n_cls, s)),
            ("LP", lambda s: label_propagation(A_hat, labels, s))):
        n_eval = 100
        print(f"\n===== Cora-{pname}: J={n_eval} 真实种子 =====", flush=True)
        P = np.zeros((n_eval, n), dtype=int)
        for j in range(n_eval):
            P[j] = gen(900 + j)
        acc = float(np.mean(P == labels))
        print(f"  acc={acc:.3f}", flush=True)
        D_real = np.array([all_hom(P[j]) - a_star for j in range(n_eval)])
        D_bar = D_real.mean(axis=0)

        cnt = np.zeros((n, n_cls))
        for c in range(n_cls):
            cnt[:, c] = (P == c).sum(axis=0)
        p_node = cnt / n_eval
        cum = np.cumsum(p_node, axis=1)
        cum[:, -1] = 1.0
        M0_node = np.array([p_node[np.asarray(e)].prod(axis=0).sum()
                            - a_star[i] for i, e in enumerate(edges)])
        rng = np.random.default_rng(700)
        D_node = np.zeros((N_NODE_DRAWS, m))
        for s in range(N_NODE_DRAWS):
            pred = (rng.random(n)[:, None] > cum).sum(axis=1)
            D_node[s] = all_hom(pred) - a_star
        q_hat = (D_real != 0).mean(axis=0)
        sgn = np.where(a_star == 1, -1.0, 1.0)
        D_edge = np.where(rng.random((N_EDGE_DRAWS, m)) < q_hat, sgn, 0.0)

        cen_real = [snorm(build_op(n, edges, D_real[j] - D_bar))
                    for j in range(n_eval)]
        cen_node = [snorm(build_op(n, edges, D_node[s] - M0_node))
                    for s in range(N_NODE_DRAWS)]
        cen_edge = [snorm(build_op(n, edges, D_edge[s] - sgn * q_hat))
                    for s in range(N_EDGE_DRAWS)]
        r_real, r_node, r_edge = (float(np.mean(cen_real)),
                                  float(np.mean(cen_node)),
                                  float(np.mean(cen_edge)))
        rng_bs = np.random.default_rng(99)
        ir = rng_bs.integers(0, n_eval, (N_BOOT, n_eval))
        inode = rng_bs.integers(0, N_NODE_DRAWS, (N_BOOT, N_NODE_DRAWS))
        iedge = rng_bs.integers(0, N_EDGE_DRAWS, (N_BOOT, N_EDGE_DRAWS))
        ar = np.array(cen_real)[ir].mean(1)
        an = np.array(cen_node)[inode].mean(1)
        ae = np.array(cen_edge)[iedge].mean(1)
        ci_rn = np.quantile(ar / an, [0.025, 0.975])
        print(f"  real={r_real:.3f} node={r_node:.3f} edge={r_edge:.3f} | "
              f"real/node={r_real / r_node:.3f} "
              f"CI[{ci_rn[0]:.3f},{ci_rn[1]:.3f}] "
              f"real/edge={r_real / r_edge:.3f}")

        # 二阶分解（top-eig 探针）
        B_star = build_op(n, edges, a_star)
        _, v0 = eigsh(B_star, k=1, which="LA")
        x = np.asarray(v0[:, 0]).ravel()
        x = x / np.linalg.norm(x)
        g = np.array([x[np.asarray(e)].sum() for e in edges]) ** 2 \
            / np.array([len(e) for e in edges])
        I = np.zeros((m, n))
        for i, e in enumerate(edges):
            I[i, np.asarray(e)] = 1.0
        share = (I @ I.T) > 0
        np.fill_diagonal(share, False)
        decomp = {}
        for arm, Dmat, ctr in (("real", D_real, D_bar),
                               ("node", D_node, M0_node)):
            Dc = Dmat - ctr
            Cov = np.cov(Dc, rowvar=False, bias=True)
            off = Cov.copy()
            np.fill_diagonal(off, 0.0)
            contrib = off * np.outer(g, g)
            decomp[arm] = dict(
                var=float(np.var(Dc @ g)),
                within=float(np.sum(np.diag(Cov) * g ** 2)),
                shared=float(contrib[share].sum()),
                disjoint=float(contrib[~share].sum()))
        print(f"  分解 real: var={decomp['real']['var']:.4f} "
              f"within={decomp['real']['within']:.4f} "
              f"shared={decomp['real']['shared']:+.4f} "
              f"disjoint={decomp['real']['disjoint']:+.4f}; "
              f"node disjoint={decomp['node']['disjoint']:+.4f}")

        results[pname] = dict(acc=acc, cen_real=r_real, cen_node=r_node,
                              cen_edge=r_edge,
                              ratio_real_node=float(r_real / r_node),
                              ratio_real_node_ci=[float(v) for v in ci_rn],
                              ratio_real_edge=float(r_real / r_edge),
                              ratio_node_edge=float(r_node / r_edge),
                              decomp=decomp)

    with open("results/attribution_gcn_lp.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("\n已写出 results/attribution_gcn_lp.json")


if __name__ == "__main__":
    main()
