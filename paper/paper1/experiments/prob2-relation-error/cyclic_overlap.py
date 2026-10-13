"""循环重叠超图上的中心化分离验证（审稿人指定构造）。

构造：t 个中心 v_0..v_{t-1}，td 个叶 u_0..u_{td-1} 排成环。
边 e_j = {v_{⌊j/d⌋}, u_j, u_{(j+1) mod td}}，j=0..td-1。
所有节点真标签相同。连通，相邻边共享叶，相邻中心共享边界叶。

理论量（审稿人推导，已验证）：
  n = t(d+1), m = td, Q = 4d-2, ‖ΣH_e‖ = (d+4)/3.

验证：r=3, t=100, d∈{20,50,100,150,200}, p=0.05。
d=200 时 L_A=42.71 > U_B=24.59，分离概率 ≥ 0.9741。
d≤100 时理论证书不成立，但经验分离预期仍 100%（保守性展示）。
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

DRAWS = 500
R, T, P = 3, 100, 0.05
D_LIST = (20, 50, 100, 150, 200)


def build_cyclic(t, d, r=3):
    """循环超图：td 个叶排成环，边连接相邻叶和所属中心。"""
    n = t * (d + 1)  # t centers + td leaves
    n += n % 2
    labels = np.ones(n)  # 全同质
    centers = list(range(t))  # v_0..v_{t-1}
    leaves = list(range(t, t + t * d))  # u_0..u_{td-1}
    edges = []
    for j in range(t * d):
        c = centers[j // d]
        u1 = leaves[j]
        u2 = leaves[(j + 1) % (t * d)]
        edges.append(tuple(sorted([c, u1, u2])))
    return labels, edges, centers, n


def edge_homogeneous(labels, e):
    return len(set(labels[u] for u in e)) == 1


def sym_norm_weighted(n, edges, w):
    M = np.zeros((n, n))
    for e, wt in zip(edges, w):
        if abs(wt) < 1e-15:
            continue
        k = len(e)
        for a in e:
            for b in e:
                M[a, b] += wt / k
    eig = np.linalg.eigvalsh(M)
    return max(abs(eig[0]), abs(eig[-1]))


def config(d, draws=DRAWS):
    r, t, p = R, T, P
    q_r = 1 - (1 - p) ** r - p ** r
    c_rp = (1 - p) * ((1 - p) ** (r - 1) - p ** (r - 1))
    labels, edges, centers, n = build_cyclic(t, d, r)

    # 结构量
    from collections import Counter
    Q_theory = 4 * d - 2
    H_theory = (d + 4) / 3
    # 实际计算验证
    Q_actual = 0
    for c in centers:
        cooccur = Counter()
        for e in edges:
            if c in e:
                for u in e:
                    if u != c:
                        cooccur[u] += 1
        Q_actual = max(Q_actual, sum(v * v for v in cooccur.values()))
    # ‖ΣH_e‖
    M_H = np.zeros((n, n))
    for e in edges:
        for a in e:
            for b in e:
                M_H[a, b] += 1.0 / r
    H_actual = float(max(abs(np.linalg.eigvalsh(M_H)[0]),
                         abs(np.linalg.eigvalsh(M_H)[-1])))

    # 理论界
    eta = delta_B = 0.01
    L_A = (d * c_rp - np.sqrt(Q_theory / 2 * np.log(1 / eta))) / r
    v_B = q_r * (1 - q_r) * H_theory
    R_b = max(q_r, 1 - q_r)     # Bernstein 范数界（避免与全局 R=3 冲突）
    L_log = np.log(2 * n / delta_B)
    U_B = np.sqrt(2 * v_B * L_log) + 2 * R_b * L_log / 3
    sep_theory = L_A > U_B
    conf = 1 - (1 - p) ** t - eta - delta_B if sep_theory else 0.0

    # 经验模拟
    rng = np.random.default_rng(42)
    m = len(edges)
    # 逐中心分边
    edge_by_center = {c: [] for c in centers}
    center_set = set(centers)
    for ei, e in enumerate(edges):
        for u in e:
            if u in center_set:
                edge_by_center[u].append(ei)
                break

    # 池：模拟单星分布，再取 t 个 max
    pool_size = draws * t // 5
    c0 = centers[0]
    c0_idx = edge_by_center[c0]
    c0_edges = [edges[i] for i in c0_idx]

    pool_A, pool_B = [], []
    for _ in range(pool_size):
        W = rng.random(n) < p
        dA = np.zeros(m)
        for ei, e in enumerate(edges):
            preds = [labels[u] * (1 - 2 * int(W[u])) for u in e]
            if len(set(preds)) > 1:
                dA[ei] = -1.0
        flips = rng.random(m) < q_r
        dB = np.where(flips, -1.0, 0.0)
        wA = dA[c0_idx] + q_r
        wB = dB[c0_idx] + q_r
        # 单星范数
        nodes_set = sorted(set(u for e in c0_edges for u in e))
        idx_map = {u: i for i, u in enumerate(nodes_set)}
        nn = len(nodes_set)
        MA = np.zeros((nn, nn))
        MB = np.zeros((nn, nn))
        for e, wa, wb in zip(c0_edges, wA, wB):
            ei2 = [idx_map[u] for u in e]
            for a2 in ei2:
                for b2 in ei2:
                    MA[a2, b2] += wa / r
                    MB[a2, b2] += wb / r
        ea = np.linalg.eigvalsh(MA)
        eb = np.linalg.eigvalsh(MB)
        pool_A.append(max(abs(ea[0]), abs(ea[-1])))
        pool_B.append(max(abs(eb[0]), abs(eb[-1])))

    pool_A = np.array(pool_A)
    pool_B = np.array(pool_B)
    cenA = np.zeros(draws)
    cenB = np.zeros(draws)
    for k in range(draws):
        ia = rng.integers(0, pool_size, t)
        ib = rng.integers(0, pool_size, t)
        cenA[k] = pool_A[ia].max()
        cenB[k] = pool_B[ib].max()

    sep_freq = float(np.mean(cenA > cenB))
    sep_se = float(np.sqrt(max(sep_freq * (1 - sep_freq), 1e-10) / draws))

    return dict(d=d, Q_theory=Q_theory, Q_actual=Q_actual,
                H_theory=H_theory, H_actual=H_actual,
                L_A=float(L_A), U_B=float(U_B),
                sep_theory=bool(sep_theory), conf=float(conf),
                cenA=float(cenA.mean()), cenB=float(cenB.mean()),
                sep_freq=sep_freq, sep_se=sep_se)


def main():
    print("=" * 100)
    print(f"循环重叠超图验证：r={R}, t={T}, p={P}, d∈{D_LIST}，{DRAWS} 次抽样")
    print("=" * 100)
    print(f"{'d':>5} | {'Q':>6} {'‖ΣH‖':>6} | {'L_A':>6} {'U_B':>6} "
          f"{'理论':>4} | {'β̃_A':>7} {'β̃_B':>7} | {'分离频率':>8} ± {'MC se':>5}")
    rows = []
    for d in D_LIST:
        r = config(d)
        rows.append(r)
        print(f"{d:>5} | {r['Q_actual']:>6.0f} {r['H_actual']:>6.1f} | "
              f"{r['L_A']:>6.2f} {r['U_B']:>6.2f} "
              f"{'✓' if r['sep_theory'] else '✗':>4} | "
              f"{r['cenA']:>7.2f} {r['cenB']:>7.2f} | "
              f"{r['sep_freq']:>8.1%} ± {r['sep_se']:>5.1%}")
    print(f"\n结构量验证（d=200）：Q_theory={4*200-2}, "
          f"Q_actual={rows[-1]['Q_actual']}, "
          f"H_theory={(200+4)/3:.1f}, H_actual={rows[-1]['H_actual']:.1f}")
    with open("results/cyclic_overlap.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("\n已写出 results/cyclic_overlap.json")


if __name__ == "__main__":
    main()
