"""机制归因：固定预测器 + 固定结构，比较三臂的中心化谱波动。

审稿意见 5：真实预测器的谱放大可能来自 (a) 共享节点的局部依赖（理论
机制）、(b) 共享训练状态、(c) 图平滑造成的跨节点预测相关。

设计（Cora-CC 全边，原型 5-fold 交叉拟合预测器固定）：
  真实臂  ：J=100 个不重叠种子的真实交叉拟合错误
  节点臂  ：每个节点从其 J 次真实预测的经验分布独立采样——精确匹配
            逐节点边际、重建共享节点机制 (a)、摧毁 (b)(c)
  边臂    ：模型 B，逐边经验翻转率 q̂_e

读数：
  1. 中心化谱范数比值 real/node：≈1 ⟹ 机制 (a) 解释全部放大（理论闭环）；
     >1 ⟹ 存在超出节点边际的依赖 (b)(c)。
  2. 二阶分解（探针方向 x）：Var(x^TΔBx) = Σ_{e,f} Cov(d_e,d_f)g_e g_f
     按边对拆分：e=f / 共享节点 / 不相交。节点臂的不相交份额是噪声地板；
     真实臂超出噪声地板的部分 = (b)(c) 的直接测量。

DBLP-CA 同样跑比值（无分解，m 太大）。
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
DBLP = "C:/Users/Peng1/OneDrive/文档/ChatGPT/hypergraph/data/dblp-ca"
N_NODE_DRAWS = 200
N_EDGE_DRAWS = 200
N_BOOT = 2000


def load(data_dir):
    with open(f"{data_dir}/features.pickle", "rb") as f:
        feats = pickle.load(f)
    with open(f"{data_dir}/hypergraph.pickle", "rb") as f:
        hg = pickle.load(f)
    with open(f"{data_dir}/labels.pickle", "rb") as f:
        labels = np.array(pickle.load(f))
    edges = [sorted(int(u) for u in s) for s in hg.values()]
    return feats, edges, labels


def crossfit_predict(feats, labels, seed, n_folds=5):
    n = len(labels)
    n_cls = len(np.unique(labels))
    X = feats.toarray().astype(np.float64)
    rng = np.random.default_rng(seed)
    fold_idx = rng.permutation(n)
    folds = np.array_split(fold_idx, n_folds)
    pred = np.zeros(n, dtype=int)
    for f in range(n_folds):
        test = folds[f]
        train = np.concatenate([folds[g] for g in range(n_folds) if g != f])
        protos = np.zeros((n_cls, X.shape[1]))
        for c in range(n_cls):
            mask = labels[train] == c
            if mask.sum() > 0:
                protos[c] = X[train][mask].mean(axis=0)
        blk = X[test]
        pred[test] = np.argmin(((blk[:, None, :] - protos[None, :, :]) ** 2)
                               .sum(axis=2), axis=1)
    return pred


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


def all_hom(pred, edges, edge_arr):
    """edge_arr: (m, k_max) 填充的成员索引（-1 填充）。"""
    first = edge_arr[:, 0]
    same = pred[edge_arr] == pred[first][:, None]
    same[edge_arr < 0] = True
    return same.all(axis=1).astype(float)


def main():
    results = {}
    for name, path, n_eval, do_decomp in (
            ("Cora", CORA, 100, True), ("DBLP", DBLP, 60, False)):
        feats, edges, labels = load(path)
        n = len(labels)
        n_cls = len(np.unique(labels))
        m = len(edges)
        k_max = max(len(e) for e in edges)
        edge_arr = np.full((m, k_max), -1, dtype=int)
        for i, e in enumerate(edges):
            edge_arr[i, :len(e)] = e

        a_star = all_hom(labels, edges, edge_arr)
        B_star = build_op(n, edges, a_star)

        # ---- 真实臂：J 个不重叠种子的交叉拟合 ----
        print(f"\n===== {name}: n={n}, m={m}, J={n_eval} 真实种子 =====")
        P = np.zeros((n_eval, n), dtype=int)
        accs = []
        for j in range(n_eval):
            P[j] = crossfit_predict(feats, labels, 500 + j)
            accs.append(float(np.mean(P[j] == labels)))
        print(f"  预测精度: mean={np.mean(accs):.3f} "
              f"sd={np.std(accs):.3f}")
        D_real = np.array([all_hom(P[j], edges, edge_arr) - a_star
                           for j in range(n_eval)])
        D_bar = D_real.mean(axis=0)

        # ---- 节点臂：逐节点经验边际独立采样 ----
        cnt = np.zeros((n, n_cls))
        for c in range(n_cls):
            cnt[:, c] = (P == c).sum(axis=0)
        p_node = cnt / n_eval
        cum = np.cumsum(p_node, axis=1)
        cum[:, -1] = 1.0
        # 节点臂期望（解析）：E[â_e] = Σ_c ∏_u p_u(c)
        M0_node = np.zeros(m)
        for i, e in enumerate(edges):
            pe = p_node[np.asarray(e)]
            M0_node[i] = float(pe.prod(axis=0).sum() - a_star[i])
        chk = np.abs(M0_node - D_bar).max()
        print(f"  节点边际校验: max|E_node[d]−D̄| = {chk:.4f}, "
              f"mean|·| = {np.abs(M0_node - D_bar).mean():.5f}")

        rng = np.random.default_rng(700)
        D_node = np.zeros((N_NODE_DRAWS, m))
        for s in range(N_NODE_DRAWS):
            u01 = rng.random(n)
            pred = (u01[:, None] > cum).sum(axis=1)
            D_node[s] = all_hom(pred, edges, edge_arr) - a_star

        # ---- 边臂：逐边经验率 ----
        q_hat = (D_real != 0).mean(axis=0)
        sgn = np.where(a_star == 1, -1.0, 1.0)
        D_edge = np.where(rng.random((N_EDGE_DRAWS, m)) < q_hat, sgn, 0.0)

        # ---- 中心化谱范数 ----
        cen_real = [snorm(build_op(n, edges, D_real[j] - D_bar))
                    for j in range(n_eval)]
        cen_node = [snorm(build_op(n, edges, D_node[s] - M0_node))
                    for s in range(N_NODE_DRAWS)]
        cen_edge = [snorm(build_op(
            n, edges, D_edge[s] - sgn * q_hat)) for s in range(N_EDGE_DRAWS)]

        r_real = float(np.mean(cen_real))
        r_node = float(np.mean(cen_node))
        r_edge = float(np.mean(cen_edge))
        rng_bs = np.random.default_rng(99)
        ir = rng_bs.integers(0, n_eval, (N_BOOT, n_eval))
        inode = rng_bs.integers(0, N_NODE_DRAWS, (N_BOOT, N_NODE_DRAWS))
        iedge = rng_bs.integers(0, N_EDGE_DRAWS, (N_BOOT, N_EDGE_DRAWS))
        ar = np.array(cen_real)[ir].mean(1)
        an = np.array(cen_node)[inode].mean(1)
        ae = np.array(cen_edge)[iedge].mean(1)
        ci_rn = np.quantile(ar / an, [0.025, 0.975])
        ci_re = np.quantile(ar / ae, [0.025, 0.975])
        print(f"  中心化谱范数: real={r_real:.3f} node={r_node:.3f} "
              f"edge={r_edge:.3f}")
        print(f"  real/node = {r_real / r_node:.3f} "
              f"CI[{ci_rn[0]:.3f},{ci_rn[1]:.3f}]"
              f"{'  ← 机制(a)解释全部' if ci_rn[0] <= 1.0 else ''}")
        print(f"  real/edge = {r_real / r_edge:.3f} "
              f"CI[{ci_re[0]:.3f},{ci_re[1]:.3f}]")
        print(f"  node/edge = {r_node / r_edge:.3f}")

        out = dict(acc=float(np.mean(accs)),
                   cen_real=r_real, cen_node=r_node, cen_edge=r_edge,
                   ratio_real_node=float(r_real / r_node),
                   ratio_real_node_ci=[float(x) for x in ci_rn],
                   ratio_real_edge=float(r_real / r_edge),
                   ratio_real_edge_ci=[float(x) for x in ci_re],
                   ratio_node_edge=float(r_node / r_edge),
                   marginal_check=float(chk))

        # ---- 二阶分解（仅 Cora） ----
        if do_decomp:
            print("\n  二阶分解 Var(x^TΔBx)：按 e=f / 共享节点 / 不相交")
            lam0, v0 = eigsh(B_star, k=1, which="LA")
            probes = {"top-eig(B*)": np.asarray(v0[:, 0]).ravel()}
            Y = np.zeros((n, n_cls))
            Y[np.arange(n), labels] = 1.0
            Yc = Y - Y.mean(axis=0)
            U, S, _ = np.linalg.svd(Yc, full_matrices=False)
            probes["class-readout"] = U[:, 0]
            for r in range(2):
                probes[f"random-{r}"] = np.random.default_rng(r).standard_normal(n)
            # 共享节点掩码
            I = np.zeros((m, n))
            for i, e in enumerate(edges):
                I[i, np.asarray(e)] = 1.0
            share = (I @ I.T) > 0
            np.fill_diagonal(share, False)

            decomp = {}
            for pname, x in probes.items():
                x = x / np.linalg.norm(x)
                sums = np.array([x[np.asarray(e)].sum() for e in edges])
                g = sums ** 2 / np.array([len(e) for e in edges])
                row = {}
                for arm, Dmat, ctr in (("real", D_real, D_bar),
                                       ("node", D_node, M0_node)):
                    Dc = Dmat - ctr
                    Cov = np.cov(Dc, rowvar=False, bias=True)
                    var_direct = float(np.var(Dc @ g))
                    within = float(np.trace(Cov) * 0 +
                                   np.sum(np.diag(Cov) * g ** 2))
                    off = Cov.copy()
                    np.fill_diagonal(off, 0.0)
                    contrib = off * np.outer(g, g)
                    shared = float(contrib[share].sum())
                    disj = float(contrib[~share].sum())
                    row[arm] = dict(var=var_direct, within=within,
                                    shared=shared, disjoint=disj,
                                    identity_err=abs(
                                        within + shared + disj - var_direct))
                decomp[pname] = row
                rr, nn = row["real"], row["node"]
                print(f"    {pname:>12}: real var={rr['var']:.4f} "
                      f"(within {rr['within']:.4f} / shared "
                      f"{rr['shared']:+.4f} / disjoint {rr['disjoint']:+.4f});"
                      f" node disjoint={nn['disjoint']:+.4f} (噪声地板)")
            out["decomp"] = decomp

        results[name] = out

    with open("results/attribution_fixed.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("\n已写出 results/attribution_fixed.json")


if __name__ == "__main__":
    main()
