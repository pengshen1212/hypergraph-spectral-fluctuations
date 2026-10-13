"""归因的可识别性边界：不可识别反例 + 已知特征律下的可识别性
（用户裁决 2026-10-03：先做边界定理，不重开谱尾攻关）。

【反例构造】在固定观测特征 X₀ 的条件设计下，机制 A：
ŷ_u = sign(x_u − τ)，重训随机性 τ ~ N(0, s²)；
机制 B（X₀ 冻结仿真器）：对任意输入特征，节点 u 恒输出
sign(x⁰_u − τ)——即 A 的规则在已观测 X₀ 上的值。两机制在 X₀ 上的
观测过程**逐点相同**（同为 τ 的同一函数），但：
  给定 τ，B 对新特征恒定 ⟹ 内层方差 ≡ 0；
  给定 τ，A 随新特征波动 ⟹ 内层方差 = E_τ[q(τ)(1−q(τ))] > 0。
⟹ 仅凭 X₀ 上的重训观测无法在不限制预测机制时识别内层方差。
这不声称两机制在重新抽取 X₀ 的实验设计下也观测等价。

【可识别侧（ii）】若节点独立的条件特征律 L(X|y) 已知可信、预测器可
在任意特征处求值，且重训状态分布已知，则各分量可由两层积分识别。
仅有有限次独立重训时，外层总体方差只能估计。下面对机制 A 用已知
高斯律积分 q_e(τ_k)，与嵌套抽样的对角部分估计对账。

输出：两机制在 X₀ 上的观测过程一致性（按构造恒等，验证打印）、
内层方差 A>0 / B≡0、外层项对照、(ii) 的积分 vs 嵌套对账。
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
from propagation_chain_checked import sample_edges

MU, SIGMA = 2.0, 1.0
TAU_SD = 0.5
N, C_DENS, RHO = 400, 3.0, 0.7
K_OUT, M_IN = 24, 200
SEED = 700


def main():
    rng = np.random.default_rng(123)
    y = np.tile([1, -1], N // 2)
    rng2 = np.random.default_rng(SEED)
    rng2.shuffle(y)
    E = [tuple(int(v) for v in e)
         for e in sample_edges(rng2, y, int(C_DENS * N), RHO)]
    a_star = np.array([1.0 if len({int(y[u]) for u in e}) == 1 else 0.0
                       for e in E])
    signs = np.where(a_star == 1.0, -1.0, 1.0)
    x_dir = y / np.sqrt(N)
    h_e = np.array([sum(x_dir[list(e)]) ** 2 / len(e) for e in E])

    def fresh_X(rng_):
        X = SIGMA * rng_.standard_normal(N)
        X += y * MU
        return X

    def indicators(pred):
        return np.array([1.0 if len({int(pred[u]) for u in e}) == 1
                         else 0.0 for e in E]) != a_star

    def Y_of(I):
        return float((signs * I) @ h_e)

    X0 = fresh_X(rng)

    def pred_A(X, tau):
        return np.where(X - tau >= 0, 1, -1)

    def pred_B(X, tau):           # 忽略输入，输出 A 在 X0 上的值
        return np.where(X0 - tau >= 0, 1, -1)

    taus = TAU_SD * rng.standard_normal(K_OUT)
    # (a) 观测过程一致性（按构造恒等；数值确认）
    same = all(np.array_equal(pred_A(X0, tau), pred_B(X0, tau))
               for tau in taus)
    observed_Y = np.array([Y_of(indicators(pred_A(X0, tau))) for tau in taus])
    print(f"(a) 两机制在 X₀ 上的预测向量：{'逐点相同 ✓' if same else '不一致 ✗'}"
          f"（{K_OUT} 次重训全部核对）")
    # (b) 内层方差：A > 0，B ≡ 0
    inner_A, inner_B, outer_terms_A, outer_terms_B = [], [], [], []
    for tau in taus:
        YA, YB = [], []
        for _ in range(M_IN):
            Xf = fresh_X(rng)
            YA.append(Y_of(indicators(pred_A(Xf, tau))))
            YB.append(Y_of(indicators(pred_B(Xf, tau))))
        inner_A.append(np.var(YA, ddof=1))
        inner_B.append(np.var(YB, ddof=1))
        outer_terms_A.append(np.mean(YA))
        outer_terms_B.append(np.mean(YB))
    print(f"(b) 内层方差：A 均值 {np.mean(inner_A):.5f}（>0），"
          f"B 均值 {np.mean(inner_B):.3e}（应 ≡0，最大 "
          f"{np.max(inner_B):.1e}）")
    observed_outer = float(np.var(observed_Y, ddof=1))
    outer_B = float(np.var(outer_terms_B, ddof=1))
    assert np.isclose(observed_outer, outer_B)
    print(f"    固定 X₀ 下可观测的跨重训方差：A = B = "
          f"{observed_outer:.5f}；A 在新鲜特征律下的外层方差 "
          f"{np.var(outer_terms_A, ddof=1):.5f} 是额外模型量")
    # (c) 已知特征律下的可识别性：q_e(τ) 数值积分 vs 嵌套估计
    from scipy.stats import norm
    grid = np.linspace(MU - 6, MU + 6, 4001)   # x 的边缘（两类混合）
    # 逐边 q_e(τ)：齐次边破坏率 / 混合边翻转率（二类阈值预测器）
    def q_edge(e, tau):
        # 节点 u 预测 +1 的概率 = Φ((μ_u − τ)/σ)
        p_pos = norm.cdf((y[list(e)] * MU - tau) / SIGMA)
        # 全同预测概率 = Π p⁺ + Π p⁻
        allp = np.prod(p_pos) + np.prod(1 - p_pos)
        return (1 - allp) if a_star[E.index(e)] == 1.0 else allp

    inner_int = []
    for tau in taus:
        qs = np.array([q_edge(e, tau) for e in E])
        # Var(Y|τ) = Σ q(1−q)h² + 重叠协方差项；只对账对角部分
        inner_int.append(float(np.sum(qs * (1 - qs) * h_e ** 2)))
    emp_diag = []
    for k, tau in enumerate(taus):
        cnt = np.zeros(len(E))
        for _ in range(M_IN * 2):
            Xf = fresh_X(rng)
            cnt += indicators(pred_A(Xf, tau))
        qh = cnt / (M_IN * 2)
        emp_diag.append(float(np.sum(qh * (1 - qh) * h_e ** 2)))
    r = np.corrcoef(inner_int, emp_diag)[0, 1]
    print(f"(c) 已知特征律可识别性（对角部分对账，{K_OUT} 个 τ）："
          f"积分均值 {np.mean(inner_int):.5f} vs 经验均值 "
          f"{np.mean(emp_diag):.5f}，相关 {r:.3f}")
    print("⟹ (i) 仅固定 X₀ 的重训观测不能识别内层方差；"
          "(ii) 特征律、预测器及重训状态律已知时各项可积分，"
          "有限重训只能估计外层项")
    with open("results/identifiability_boundary.json", "w",
              encoding="utf-8") as f:
        json.dump(dict(same_on_X0=bool(same),
                       inner_A_mean=float(np.mean(inner_A)),
                       inner_B_max=float(np.max(inner_B)),
                       outer_A=float(np.var(outer_terms_A, ddof=1)),
                       outer_B=outer_B,
                       observed_outer=observed_outer,
                       diag_integral=float(np.mean(inner_int)),
                       diag_empirical=float(np.mean(emp_diag)),
                       corr=float(r)), f, ensure_ascii=False, indent=1)
    print("已写出 results/identifiability_boundary.json")


if __name__ == "__main__":
    main()
