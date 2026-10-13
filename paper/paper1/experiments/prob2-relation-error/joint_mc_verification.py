"""独立联合蒙特卡洛验证 Theorem centered-sep（P1-A 补证据，2026-10-08）。

与 theorem_verification.py 的区别：那里先造单星范数池再有放回抽 max，
500 个最大值复用同一批池样本，只能作尺度示意。本脚本每次 draw 独立
生成全部 t 个星的节点错误与全部 t·d 条 B 边错误，在真正的联合图上计算
中心化谱范数，逐 draw 分类计数：
  - no_wrong_center：t 个中心全部正确（定理 (i) 的事件前提不成立）；
  - a_lower_fail：β̃_A < L_A（定理 (i) 允许失败率 η）；
  - b_upper_fail：β̃_B > U_B（定理 (ii) 允许失败率 δ_B）；
  - sep_fail：β̃_A <= β̃_B（定理 (iii) 允许失败率 (1-p)^t+η+δ_B）。
报告各频率的 Wilson 95% 区间与理论容许值对比。

采样等价性（自检覆盖）：星内叶子跨边不相交，给定中心对/错时，
  broken_j | 中心错 ~ Bern(1-p^{r-1})，broken_j | 中心对 ~ Bern(p^{r-1})
与逐节点 Bernoulli(p) 误差模型完全同分布（叶集不相交 ⟹ 边指示独立）。
谱计算：不相交叶星形 M = Σ_j w_j C_j 的非零谱 = d×d 归约 Gram 矩阵
G_{jk} = u_j^T M u_k 的谱（补空间恰为零空间），比 (1+d(r-1)) 阶全矩阵快。

网格与定理设定一致：r=3, t=100, p ∈ {0.05,0.10}, d ∈ {20,50,100,200}，
η=δ_B=0.01，每格 DRAWS 次独立联合实现。
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

DRAWS = 5000
R, T = 3, 100
P_LIST = (0.05, 0.10)
D_LIST = (20, 50, 100, 200)
ETA = DELTA_B = 0.01
SEED = 20261008


def star_weights_from_node_errors(rng, t, d, r, p):
    """直接逐节点采样（仅自检用）：返回每星边权 w = d_edge + q_r。"""
    q_r = 1 - (1 - p) ** r - p ** r
    n_star = 1 + d * (r - 1)
    W = rng.random((t, n_star)) < p
    broken = np.zeros((t, d), dtype=bool)
    for i in range(t):
        for j in range(d):
            leaves = W[i, 1 + j * (r - 1): 1 + (j + 1) * (r - 1)]
            preds = np.concatenate(([W[i, 0]], leaves))
            broken[i, j] = len(set(preds.tolist())) > 1
    return np.where(broken, -1.0, 0.0) + q_r


def star_weights_conditional(rng, t, d, r, p):
    """条件采样（主路径）：中心对/错分别采边 broken 指示。"""
    q_r = 1 - (1 - p) ** r - p ** r
    wrong = rng.random((t, 1)) < p
    u = rng.random((t, d))
    keep_given_wrong = p ** (r - 1)         # 边不破 ⟺ 全部叶子也错
    keep_given_clean = (1 - p) ** (r - 1)   # 边不破 ⟺ 全部叶子也对
    keep = np.where(wrong, u < keep_given_wrong, u < keep_given_clean)
    return np.where(~keep, -1.0, 0.0) + q_r


def star_weights_b(rng, t, d, q_r):
    """B 模型：全部 t·d 条边独立 Bernoulli(q_r) 翻转。"""
    flip = rng.random((t, d)) < q_r
    return np.where(flip, -1.0, 0.0) + q_r


def star_norms_batched(w, r):
    """批量星形中心化谱范数（两值权重闭式解，O(td)）。

    归约：range(M) ⊆ span{𝟙_j} 且不变，非零谱 = 限制矩阵
      A_{jk} = w_j/r (j≠k)，A_{jj} = w_j   （c = 1−1/r）
    的谱。两值权重 w ∈ {q_r, q_r−1}（kept 取较大值 q_r）时
    A = c·diag(w) + (w/r)𝟙ᵀ 是两值对角 + 秩一，特征值为
      c·q_r   （重数 k−1，k = kept 边数，k≥2 时存在）
      c(q_r−1)（重数 b−1，b = broken 边数，b≥2 时存在）
      λ_±     （二次 rλ² − Sλ + T = 0 的两根：
                a1 = c q_r, a2 = c(q_r−1)，
                S = r(a1+a2) + k q_r + b(q_r−1)，
                T = r a1 a2 + k q_r a2 + b(q_r−1) a1）
    自检：与全矩阵 eigvalsh 及一般非对称 eig 逐星对账。
    """
    t, d = w.shape
    q_r = float(w.max())               # kept 权重（q_r < 1 恒成立）
    ok = np.isclose(w, q_r) | np.isclose(w, q_r - 1.0)
    assert ok.all() and q_r < 1.0, "两值权重假设不成立"
    k = (w == q_r).sum(axis=1).astype(float)
    b = d - k
    c = 1.0 - 1.0 / r
    a1 = c * q_r
    a2 = c * (q_r - 1.0)
    S = r * (a1 + a2) + k * q_r + b * (q_r - 1.0)
    T = r * a1 * a2 + k * q_r * a2 + b * (q_r - 1.0) * a1
    disc = np.maximum(S * S - 4.0 * r * T, 0.0)
    sq = np.sqrt(disc)
    lam_p = (S + sq) / (2.0 * r)
    lam_m = (S - sq) / (2.0 * r)
    # k=0/b=0 时二次方程含一个恰为空极点值的伪根（乘 (a1-λ)(a2-λ) 引入）
    tol = 1e-7
    valid_p = ((k > 0) | (np.abs(lam_p - a1) > tol)) & \
              ((b > 0) | (np.abs(lam_p - a2) > tol))
    valid_m = ((k > 0) | (np.abs(lam_m - a1) > tol)) & \
              ((b > 0) | (np.abs(lam_m - a2) > tol))
    cand = [np.where(valid_p, np.abs(lam_p), 0.0),
            np.where(valid_m, np.abs(lam_m), 0.0)]
    cand.append(np.where(k >= 2, abs(a1), 0.0))
    cand.append(np.where(b >= 2, abs(a2), 0.0))
    return np.maximum.reduce(cand)


def star_norms_eig(w, r):
    """一般路径（自检交叉对账用）：d×d 非对称限制矩阵 eig。"""
    t, d = w.shape
    A = np.empty((t, d, d))
    A[:] = w[:, :, None] / r
    idx = np.arange(d)
    A[:, idx, idx] += w * (1.0 - 1.0 / r)
    ev = np.linalg.eig(A)[0]
    return np.abs(ev).max(axis=1)


def star_norm_full(w, r):
    """(1+d(r-1)) 阶全矩阵参考实现（自检用）。"""
    d = len(w)
    nn = 1 + d * (r - 1)
    M = np.zeros((nn, nn))
    for j in range(d):
        if abs(w[j]) < 1e-15:
            continue
        e = [0] + list(range(1 + j * (r - 1), 1 + (j + 1) * (r - 1)))
        for a in e:
            for b in e:
                M[a, b] += w[j] / r
    eig = np.linalg.eigvalsh(M)
    return max(abs(eig[0]), abs(eig[-1]))


def wilson(k, n, z=1.959963984540054):
    """Wilson 95% 区间。"""
    if n == 0:
        return [0.0, 1.0]
    ph = k / n
    den = 1 + z * z / n
    ctr = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [max(0.0, ctr - half), min(1.0, ctr + half)]


def self_test():
    rng = np.random.default_rng(7)
    # 1) 闭式解 = 一般 eig 限制矩阵 = 全矩阵 eigvalsh（两值权重）
    for d in (3, 8, 20, 50):
        for kk in (0, 1, d // 2, d - 1, d):
            w = np.full((6, d), Q_R_TEST - 1.0)
            for i in range(6):
                keep = rng.random(d) < kk / d
                w[i, keep] = Q_R_TEST
            closed = star_norms_batched(w, R)
            gen = star_norms_eig(w, R)
            full = np.array([star_norm_full(w[i], R) for i in range(6)])
            assert np.allclose(closed, gen, atol=1e-8), (d, kk, closed, gen)
            assert np.allclose(closed, full, atol=1e-8), (d, kk, closed, full)
    # 2) 条件采样与逐节点采样的边权分布一致
    p = 0.1
    q_r = 1 - (1 - p) ** R - p ** R
    r2 = np.random.default_rng(11)
    w_nodes = star_weights_from_node_errors(r2, 200, 6, R, p)
    r4 = np.random.default_rng(12)
    w_cond = star_weights_conditional(r4, 20000, 6, R, p)
    fr_nodes = (w_nodes == q_r - 1).mean()
    fr_cond = (w_cond == q_r - 1).mean()
    # 边际破边率 = q_r = 1-(1-p)^r - p^r
    target = 1 - (1 - p) ** R - p ** R
    assert abs(fr_cond - target) < 0.01, (fr_cond, target)
    assert abs(fr_nodes - target) < 0.02, (fr_nodes, target)
    # 3) 定理验证配置复现（论文数字）
    r_, t_, d_, p_ = 3, 100, 100, 0.05
    c_rp = (1 - p_) * ((1 - p_) ** (r_ - 1) - p_ ** (r_ - 1))
    n_ = t_ * (1 + d_ * (r_ - 1))
    L_A = (d_ * c_rp - np.sqrt(d_ / 2 * np.log(1 / ETA))) / r_
    q_r0 = 1 - (1 - p_) ** r_ - p_ ** r_
    v_B = q_r0 * (1 - q_r0) * (d_ + r_ - 1) / r_
    Rm = max(q_r0, 1 - q_r0)
    L_log = np.log(2 * n_ / DELTA_B)
    U_B = np.sqrt(2 * v_B * L_log) + 2 * Rm * L_log / 3
    assert abs(L_A - 23.44) < 0.01 and abs(U_B - 19.93) < 0.01, (L_A, U_B)
    print("自检通过：闭式解 = 一般 eig = 全矩阵谱；条件采样分布一致；"
          "定理数字复现")


Q_R_TEST = 0.27   # 仅自检用的两值权重 kept 值（p=0.1, r=3）


def config2(r, t, d, p, draws, rng):
    """主路径：显式跟踪每星中心状态，避免 config 的信息丢失。"""
    q_r = 1 - (1 - p) ** r - p ** r
    c_rp = (1 - p) * ((1 - p) ** (r - 1) - p ** (r - 1))
    n = t * (1 + d * (r - 1))

    L_A = (d * c_rp - np.sqrt(d / 2 * np.log(1 / ETA))) / r
    v_B = q_r * (1 - q_r) * (d + r - 1) / r
    Rm = max(q_r, 1 - q_r)
    L_log = np.log(2 * n / DELTA_B)
    U_B = np.sqrt(2 * v_B * L_log) + 2 * Rm * L_log / 3

    cnt = dict(no_wrong_center=0, a_lower_fail=0, a_lower_fail_cond=0,
               b_upper_fail=0, sep_fail=0, sep_fail_given_bounded=0)
    cenA = np.empty(draws)
    cenB = np.empty(draws)
    keep_given_wrong = p ** (r - 1)
    keep_given_clean = (1 - p) ** (r - 1)
    for k in range(draws):
        wrong = rng.random((t, 1)) < p
        u = rng.random((t, d))
        keep = np.where(wrong, u < keep_given_wrong, u < keep_given_clean)
        wA = np.where(~keep, -1.0, 0.0) + q_r
        wB = star_weights_b(rng, t, d, q_r)
        bA = float(star_norms_batched(wA, r).max())
        bB = float(star_norms_batched(wB, r).max())
        cenA[k], cenB[k] = bA, bB
        any_wrong = bool(wrong.any())
        if not any_wrong:
            cnt["no_wrong_center"] += 1
        else:
            if bA < L_A:
                cnt["a_lower_fail"] += 1
                cnt["a_lower_fail_cond"] += 1
        if bB > U_B:
            cnt["b_upper_fail"] += 1
        if bA <= bB:
            cnt["sep_fail"] += 1
        if any_wrong and bB <= U_B and bA < L_A:
            cnt["sep_fail_given_bounded"] += 1
    return cnt, cenA, cenB, dict(L_A=float(L_A), U_B=float(U_B),
                                 allowed=dict(
                                     no_center=float((1 - p) ** t),
                                     a_fail=ETA, b_fail=DELTA_B,
                                     sep_fail=float((1 - p) ** t
                                                    + ETA + DELTA_B)),
                                 q_r=float(q_r))


def main():
    self_test()
    rng = np.random.default_rng(SEED)
    print("=" * 104)
    print(f"独立联合 MC 验证：r={R}, t={T}, p ∈ {P_LIST}, "
          f"d ∈ {D_LIST}，每格 {DRAWS} 次独立联合实现，seed={SEED}")
    print("=" * 104)
    rows = []
    for p in P_LIST:
        for d in D_LIST:
            cnt, cenA, cenB, th = config2(R, T, d, p, DRAWS, rng)
            row = dict(r=R, t=T, d=d, p=p, draws=DRAWS,
                       L_A=th["L_A"], U_B=th["U_B"], q_r=th["q_r"],
                       allowed=th["allowed"], counts=cnt,
                       cenA_mean=float(cenA.mean()),
                       cenB_mean=float(cenB.mean()),
                       freq={})
            n_wc = DRAWS - cnt["no_wrong_center"]
            row["freq"] = dict(
                no_wrong_center=(cnt["no_wrong_center"], DRAWS,
                                 wilson(cnt["no_wrong_center"], DRAWS)),
                a_lower_fail_cond=(cnt["a_lower_fail_cond"], n_wc,
                                   wilson(cnt["a_lower_fail_cond"], n_wc)),
                b_upper_fail=(cnt["b_upper_fail"], DRAWS,
                              wilson(cnt["b_upper_fail"], DRAWS)),
                sep_fail=(cnt["sep_fail"], DRAWS,
                          wilson(cnt["sep_fail"], DRAWS)))
            rows.append(row)
            f = row["freq"]
            print(f"p={p:.2f} d={d:>3} | L_A={th['L_A']:6.2f} "
                  f"U_B={th['U_B']:6.2f} | β̃A={cenA.mean():7.2f} "
                  f"β̃B={cenB.mean():6.2f} | 无错中心 "
                  f"{cnt['no_wrong_center']}/{DRAWS} | A下界败 "
                  f"{cnt['a_lower_fail_cond']}/{n_wc} | B上界败 "
                  f"{cnt['b_upper_fail']}/{DRAWS} | 分离败 "
                  f"{cnt['sep_fail']}/{DRAWS} "
                  f"CI[{f['sep_fail'][2][0]:.4f},{f['sep_fail'][2][1]:.4f}]"
                  f" 允许 {th['allowed']['sep_fail']:.4f}")
    with open("results/joint_mc_verification.json", "w",
              encoding="utf-8") as fh:
        json.dump(dict(
            protocol=("independent joint realizations: per draw all node "
                      "errors and all t*d B-edge errors are freshly "
                      "generated; star spectra computed on the true joint "
                      "graph (disjoint blocks); r=3, eta=delta_B=0.01"),
            seed=SEED, draws=DRAWS, rows=rows), fh, ensure_ascii=False,
            indent=1)
    print("\n已写出 results/joint_mc_verification.json")


if __name__ == "__main__":
    main()
