"""谱嵌入子空间后果实验（Paper1 后续任务，2026-10-09）。

问题：中心化波动被低估 1.3–2.7 倍的"校准警示"之外，能否给一个下游
数学后果？Davis–Kahan：同一特征间隔下，算子误差的范数差按比例传导到
特征子空间角误差。本实验在真实预测器构型上直接测量：

  对每个实现 r：pred（全新种子 60000+，5 折交叉拟合原型）→ 估计算子
  B(â^A)；匹配逐边边际的独立基准 B(â^B)（q̂_e 来自 20 个校准运行，
  61000+ 块，与评估实现不相交；ξ_e ~ Bern(q̂_e)）；
  V_* = B(a*) 的 top-k（k=8）特征子空间。
  量：sinΘ 误差 ‖sinΘ(V_arm, V_*)‖_F = sqrt(k − ‖V_armᵀV_*‖_F²)，
  以及算子误差范数 ‖B(â)−B(a*)‖₂。
  报告：A/B 的 sinΘ 比值与范数比值（应一致），附 oracle 特征间隔。

冻结：k=8、单位团展开权重（与 H_e=𝟙𝟙ᵀ/r 的协议一致，用 1/r 权重建
算子）；Cora-all R=20、DBLP-all R=10；不与旧评估种子重叠。诚实报告：
若比值不接近，如实写。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as spsp
from scipy.sparse.linalg import eigsh

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from realdata_fixed import load, all_hom
from crossfit_v2 import crossfit_predict_sparse

CORA = "C:/Users/Peng1/OneDrive/文档/ChatGPT/hypergraph/data/cora-cc"
DBLP = "C:/Users/Peng1/OneDrive/文档/ChatGPT/hypergraph/data/dblp-ca"
K = 8
N_CAL = 20
SEED_EVAL = 60000
SEED_CAL = 61000
R = {"Cora-all": 20, "DBLP-all": 10}
RESULTS = Path(__file__).resolve().parent / "results"


def build_op(n, edges, w):
    """B(w) = Σ_e w_e H_e，H_e = 𝟙_e𝟙_eᵀ/|e|（与论文式 (B) 一致）。"""
    rows, cols, vals = [], [], []
    for e, wt in zip(edges, w):
        if wt == 0:
            continue
        k = len(e)
        for a in e:
            for b in e:
                rows.append(a); cols.append(b); vals.append(wt / k)
    return spsp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def topk(M, k):
    if spsp.issparse(M) and M.nnz == 0:
        return np.zeros((M.shape[0], k))
    lam, V = eigsh(M, k=k, which="LM", tol=1e-8, maxiter=10000)
    return V


def sintheta(Va, Vb):
    s = np.linalg.svd(Va.T @ Vb, compute_uv=False)
    return float(np.sqrt(max(len(s) - (s ** 2).sum(), 0.0)))


def main():
    doc = dict(protocol=(f"spectral subspace consequence of relation-"
                         f"estimation errors; k={K}; unit clique expansion "
                         f"weights 1/|e|; fresh eval seeds {SEED_EVAL}+ and "
                         f"calibration seeds {SEED_CAL}+ (disjoint from all "
                         f"earlier blocks); B matched per-edge rates from "
                         f"{N_CAL} calibration runs"),
               settings=[])
    for name, path in (("Cora-all", CORA), ("DBLP-all", DBLP)):
        t0 = time.time()
        feats, edges, labels = load(path)
        n = len(labels)
        a_star = all_hom(labels, edges)
        m = len(edges)
        sgn = np.where(a_star == 1, -1.0, 1.0)
        Bstar = build_op(n, edges, a_star)
        Vs = topk(Bstar, K)
        lam = eigsh(Bstar, k=K + 1, which="LM", tol=1e-8,
                    return_eigenvectors=False)
        gap = float(lam[-K] - lam[-K - 1])   # λ_k − λ_{k+1}

        D_cal = []
        for s in range(N_CAL):
            pred = crossfit_predict_sparse(feats, labels, SEED_CAL + s)
            D_cal.append((all_hom(pred, edges) - a_star) != 0)
        q_hat = np.asarray(D_cal).mean(axis=0)

        sinA, sinB, nrmA, nrmB, VA_list, VB_list = [], [], [], [], [], []
        rng = np.random.default_rng(62000)
        for r in range(R[name]):
            pred = crossfit_predict_sparse(feats, labels, SEED_EVAL + r)
            wA = all_hom(pred, edges)
            flip = rng.random(m) < q_hat
            wB = a_star + np.where(flip, sgn, 0.0)
            BA, BB = build_op(n, edges, wA), build_op(n, edges, wB)
            VA, VB = topk(BA, K), topk(BB, K)
            VA_list.append(VA); VB_list.append(VB)
            sinA.append(sintheta(VA, Vs))
            sinB.append(sintheta(VB, Vs))
            dA = BA - Bstar
            dB = BB - Bstar
            nrmA.append(float(abs(eigsh(dA, k=1, which="LM", tol=1e-6,
                                         return_eigenvectors=False)[0])))
            nrmB.append(float(abs(eigsh(dB, k=1, which="LM", tol=1e-6,
                                         return_eigenvectors=False)[0])))
        sinA, sinB = np.array(sinA), np.array(sinB)
        nrmA, nrmB = np.array(nrmA), np.array(nrmB)

        def dispersion(Vlist):
            """跨实现分散度：各实现子空间对该臂平均子空间的 sinΘ 均值。

            平均子空间 = Σ_r V_r V_rᵀ = W Wᵀ（W = hstack(V_r)）的 top-k；
            LinearOperator 避免在 DBLP 上构造 n×n 稠密矩阵。"""
            from scipy.sparse.linalg import LinearOperator
            W = np.hstack(Vlist)                     # n × (R*k)
            M = LinearOperator((W.shape[0], W.shape[0]),
                               matvec=lambda x: W @ (W.T @ x))
            _, Vbar = eigsh(M, k=K, which="LM", tol=1e-8, maxiter=10000)
            return float(np.mean([sintheta(V, Vbar) for V in Vlist]))

        dispA, dispB = dispersion(VA_list), dispersion(VB_list)
        row = dict(
            name=name, n_realizations=R[name], k=K, oracle_gap_k=gap,
            sinA_mean=float(sinA.mean()), sinB_mean=float(sinB.mean()),
            sin_ratio=float(sinA.mean() / sinB.mean()),
            sin_ratio_ci=_boot_ratio(sinA, sinB),
            dispersionA=dispA, dispersionB=dispB,
            dispersion_ratio=float(dispA / dispB),
            normA_mean=float(nrmA.mean()), normB_mean=float(nrmB.mean()),
            norm_ratio=float(nrmA.mean() / nrmB.mean()),
            sinA_minmax=[float(sinA.min()), float(sinA.max())],
            sinB_minmax=[float(sinB.min()), float(sinB.max())])
        doc["settings"].append(row)
        print(f"[{name}] gap(λ{K}−λ{K+1})={gap:.4f}  "
              f"对oracle sinΘ A/B = {sinA.mean():.4f}/{sinB.mean():.4f} "
              f"比 {row['sin_ratio']:.2f}  |  "
              f"跨实现分散度 A/B = {dispA:.4f}/{dispB:.4f} "
              f"比 {row['dispersion_ratio']:.2f}  |  "
              f"原始误差范数比 {row['norm_ratio']:.2f}  ({time.time()-t0:.0f}s)",
              flush=True)
    dest = RESULTS / "spectral_subspace_consequence.json"
    dest.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"已写出 {dest}")


def _boot_ratio(a, b, n_boot=2000):
    rng = np.random.default_rng(99)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    ra = a[idx].mean(axis=1)
    rb = b[idx].mean(axis=1)
    q = np.quantile(ra / rb, [0.025, 0.975]).tolist()
    return [float(x) for x in q]


if __name__ == "__main__":
    main()
