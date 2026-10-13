"""完整传播 bootstrap：每次重采样校准→重算 q̂→重新生成 B→全量评估组。

审稿要求：B 样本按重采样后的率重新生成；评估组按实际样本量重采样。
目标统计量：平均范数之比 mean‖A−M‖/mean‖B−M‖，与主表一致。
四设置 × 校准 {20, 40, 80}；报告各划分独立区间。

效率优化：
  A 中心化：ΔB_A^(j) − M̂* = Σ_e (d_e^(j) − d̄*) H_e。
  B 中心化：ΔB_B^(k) − M̂* = Σ_e (ξ_k − q̂*) s_e H_e。
  两者都是加权团展开；用 Gram 技巧 O(nnz) 计算 Frobenius，
  谱范数用 eigsh。
"""

from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

sys.stdout.reconfigure(encoding="utf-8")
from realdata_fixed import all_hom, load

CORA = "C:/Users/Peng1/OneDrive/文档/ChatGPT/hypergraph/data/cora-cc"
DBLP = "C:/Users/Peng1/OneDrive/文档/ChatGPT/hypergraph/data/dblp-ca"
CAL_SIZES = (20, 40, 80)
N_EVAL = 20
N_B_PER_BS = 20          # 每次 bootstrap 重生成的 B 抽样数
N_BOOT = 1000
SEED_CAL_BASE = 31000
SEED_EVAL = 32000


def sym_norm_sparse(M):
    if M.nnz == 0:
        return 0.0
    try:
        la = float(eigsh(M, k=1, which="LA", tol=1e-5,
                         maxiter=5000, return_eigenvectors=False)[0])
    except Exception:
        la = 0.0
    try:
        li = float(eigsh(M, k=1, which="SA", tol=1e-5,
                         maxiter=5000, return_eigenvectors=False)[0])
    except Exception:
        li = 0.0
    return max(abs(la), abs(li))


def build_weighted(n, edges, w):
    rows, cols, vals = [], [], []
    for e, wt in zip(edges, w):
        if wt == 0:
            continue
        k = len(e)
        for a in e:
            for b in e:
                rows.append(a); cols.append(b); vals.append(wt / k)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def run_setting(name, feats, edges_all, labels, tri_only):
    n = len(labels)
    edges = [e for e in edges_all if len(e) == 3] if tri_only else edges_all
    a_star = all_hom(labels, edges)
    m = len(edges)
    sgn = np.where(a_star == 1, -1.0, 1.0)

    from crossfit_v2 import crossfit_predict_sparse
    # 评估组（固定）
    D_eval = []
    for s in range(N_EVAL):
        pred = crossfit_predict_sparse(feats, labels, SEED_EVAL + s)
        D_eval.append(all_hom(pred, edges) - a_star)
    D_eval = np.array(D_eval).astype(float)

    # 预算校准运行（最大 80）
    D_cal_pool = []
    for s in range(max(CAL_SIZES)):
        pred = crossfit_predict_sparse(feats, labels, SEED_CAL_BASE + s)
        D_cal_pool.append(all_hom(pred, edges) - a_star)
    D_cal_pool = np.array(D_cal_pool).astype(float)

    results = []
    for n_cal in CAL_SIZES:
        D_cal = D_cal_pool[:n_cal]
        # 点估计
        q0 = (D_cal != 0).mean(axis=0)
        d_bar0 = D_cal.mean(axis=0)
        M0 = build_weighted(n, edges, d_bar0)
        cenA0 = [sym_norm_sparse(build_weighted(
            n, edges, D_eval[j] - d_bar0)) for j in range(N_EVAL)]
        rng0 = np.random.default_rng(33000)
        cenB0 = [sym_norm_sparse(build_weighted(
            n, edges,
            np.where(rng0.random(m) < q0, sgn, 0.0) - sgn * q0))
            for _ in range(N_B_PER_BS)]
        ratio0 = float(np.mean(cenA0) / np.mean(cenB0))

        # 完整传播 bootstrap
        rng = np.random.default_rng(99 + n_cal)
        ratios = []
        for b in range(N_BOOT):
            ic = rng.integers(0, n_cal, n_cal)
            q_b = (D_cal[ic] != 0).mean(axis=0)
            d_bar_b = D_cal[ic].mean(axis=0)
            ie = rng.integers(0, N_EVAL, N_EVAL)
            # A 中心化（用 bootstrap 校准均值）
            ca = [sym_norm_sparse(build_weighted(
                n, edges, D_eval[j] - d_bar_b)) for j in ie]
            # B 重新生成（用 bootstrap 率）
            cb = [sym_norm_sparse(build_weighted(
                n, edges,
                np.where(rng.random(m) < q_b, sgn, 0.0) - sgn * q_b))
                for _ in range(N_B_PER_BS)]
            ratios.append(float(np.mean(ca) / max(np.mean(cb), 1e-12)))
        lo, hi = np.quantile(ratios, [0.025, 0.975])
        results.append(dict(n_cal=n_cal, ratio=ratio0,
                            ci=[float(lo), float(hi)]))
        print(f"    N_cal={n_cal:>3}: ratio={ratio0:.4f} "
              f"CI=[{lo:.4f},{hi:.4f}] {'✓>1' if lo > 1 else ''}")
    return dict(name=name, results=results)


def main():
    print("=" * 90)
    print(f"完整传播 bootstrap：N_BOOT={N_BOOT}，每次 {N_EVAL} A + "
          f"{N_B_PER_BS} B（重新生成），B 率逐 replicate 重算")
    print("=" * 90)
    rows = []
    feats, edges, labels = load(CORA)
    print(f"\n[Cora-all]")
    rows.append(run_setting("Cora-all", feats, edges, labels, False))
    print(f"\n[Cora-tri]")
    rows.append(run_setting("Cora-tri", feats, edges, labels, True))
    feats, edges, labels = load(DBLP)
    print(f"\n[DBLP-all]")
    rows.append(run_setting("DBLP-all", feats, edges, labels, False))
    print(f"\n[DBLP-tri]")
    rows.append(run_setting("DBLP-tri", feats, edges, labels, True))
    print("\n" + "=" * 90)
    for r in rows:
        for res in r["results"]:
            print(f"  {r['name']} & {res['n_cal']} & "
                  f"{res['ratio']:.3f} & "
                  f"[{res['ci'][0]:.4f},{res['ci'][1]:.4f}] "
                  f"{'✓' if res['ci'][0] > 1 else ''} \\\\")
    with open("results/full_bootstrap.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("\n已写出 results/full_bootstrap.json")


if __name__ == "__main__":
    main()
