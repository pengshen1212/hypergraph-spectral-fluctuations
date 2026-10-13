"""混合边环境的中心化分离：真实结构非空验证 + MC（Paper1，2026-10-09）。

定理背景（二元标签，p<1/2）：
- 星形块（d 条同质 r 边、叶子不相交）：首个错误中心 v* 给出
  S|W=1 ~ Bin(d, 1-p^{r-1})，星块范数 >= (S - d q_r)/r（同 thm 3.5）。
- 混合环境 F（其余全部边，同质或混合，可与星支撑重叠）中只有
  **正中心化质量**能抵消下特征值深度：
    hom 边 kept: w=+q_e；mixed 边 inserted: w=1-π_e ≈ +0.9（罕见）；
  负质量只会加深 |λ_min|。用 Perron 证据向量 x*（支撑=星节点）：
    x*ᵀ(全图)x* <= -(S-dq_r)/r + P_v，  P_v = 环境正重叠质量，
  P_v 的期望 μ_v 与有界差 Q̃_v 用 McDiarmid 控制（同重叠扩展机制）。
- B 上界（全图逐边率 Bernstein）：v_B = ‖Σ_e q_e(1-q_e)H_e‖₂，
  U_B = √(2 v_B L) + 2RL/3，R<=1。
- 证明分离条件：L_A - (μ_v + √(Q̃_v log(1/η₂)/2)) > U_B。

本脚本在真实混合结构（Cora/DBLP 二元化：多数类 vs 其余）上：
 1 构造候选星（同质 3-边、叶子不相交、跨星节点不相交）；
 2 计算逐边 q_e/π_e、μ_v、Q̃_v、v_B、U_B、L_A，报告认证是否非空；
 3 MC（K 次）：真实混合图上独立采样节点错误（二元翻转）与 B 边翻转，
   计算全图 β̃_A、β̃_B（按逐边真实均值中心化），报告经验分离频率；
   并数值验证星块范数 >= (S-dq_r)/r 的引理。
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as spsp
from scipy.sparse.linalg import eigsh

sys.stdout.reconfigure(encoding="utf-8")

DATA = {"cora": "C:/Users/Peng1/OneDrive/文档/ChatGPT/hypergraph/data/cora-cc",
        "dblp": "C:/Users/Peng1/OneDrive/文档/ChatGPT/hypergraph/data/dblp-ca"}
P_LIST = (0.05, 0.10)
R_STAR = 3
ETA = ETA2 = DELTA_B = 0.01
K_MC = 200
RESULTS = Path(__file__).resolve().parent / "results"


def load(tag):
    with open(f"{DATA[tag]}/hypergraph.pickle", "rb") as f:
        hg = pickle.load(f)
    with open(f"{DATA[tag]}/labels.pickle", "rb") as f:
        labels = np.array(pickle.load(f))
    edges = [sorted(int(u) for u in s) for s in hg.values()]
    return edges, labels


def opnorm(M, psd=False):
    if M.nnz == 0:
        return 0.0

    def one(which, tol):
        try:
            return float(eigsh(M, k=1, which=which, tol=tol,
                               maxiter=50000,
                               return_eigenvectors=False)[0])
        except Exception:
            return None
    la = one("LA", 1e-6)
    if psd:
        return max(la, 0.0)
    li = one("SA", 1e-6)
    if li is None:
        li = one("SA", 1e-4) or 0.0
    if la is None:
        la = one("LA", 1e-4) or 0.0
    return max(abs(la), abs(li))


def edge_rates(edges, y2, p):
    """逐边 k、a*、q（hom 破坏率）或 π（mixed 插入率）、d̄、s。"""
    m = len(edges)
    k = np.array([len(e) for e in edges])
    a_star = np.array([1.0 if len(set(y2[list(e)].tolist())) == 1 else 0.0
                       for e in edges])
    # 逐边：hom 用 q=1-(1-p)^k-p^k；mixed 用 π=(1-p)^m p^{k-m}+p^m (1-p)^{k-m}
    q = np.zeros(m)
    for i, e in enumerate(edges):
        kk = k[i]
        if a_star[i] == 1.0:
            q[i] = 1 - (1 - p) ** kk - p ** kk
        else:
            mm = int(y2[list(e)].sum())
            q[i] = ((1 - p) ** mm * p ** (kk - mm)
                    + p ** mm * (1 - p) ** (kk - mm))
    d_bar = np.where(a_star == 1.0, -q, q)
    sgn = np.where(a_star == 1.0, -1.0, 1.0)
    return k, a_star, q, d_bar, sgn


INC = {}

def incidence(n, edges, k):
    key = (n, len(edges))
    if key not in INC:
        rows, cols, invk = [], [], []
        for i, e in enumerate(edges):
            kk = k[i] if hasattr(k, "__len__") else len(e)
            rows.extend(np.repeat(e, kk).tolist())
            cols.extend(np.tile(e, kk).tolist())
            invk.extend([1.0 / kk] * (kk * kk))
        INC[key] = (np.array(rows, dtype=np.int32),
                    np.array(cols, dtype=np.int32), np.array(invk))
    return INC[key]


def build_op(n, edges, w):
    rows, cols, invk = incidence(n, edges, None)
    vals = invk * w[np.repeat(np.arange(len(edges)),
                              [len(e) ** 2 for e in edges])]
    return spsp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def build_stars(edges, y2, a_star, k, n, want=8, d_target=100):
    """贪心构造节点不相交的候选星（同质 3-边、叶不相交）。"""
    hom3 = [i for i in range(len(edges))
            if a_star[i] == 1.0 and k[i] == R_STAR]
    deg = np.zeros(n, dtype=int)
    for i in hom3:
        for u in edges[i]:
            deg[u] += 1
    order = np.argsort(-deg)
    used_nodes = set()
    stars = []
    for v in order:
        if v in used_nodes or deg[v] < 10:
            continue
        star_edges, leaves = [], set()
        for i in hom3:
            if i in star_edges:
                continue
            e_set = set(edges[i])
            if v not in e_set or (e_set - {v}) & leaves:
                continue
            star_edges.append(i)
            leaves |= (e_set - {v})
            if len(star_edges) >= d_target:
                break
        if len(star_edges) >= 20:
            stars.append((int(v), star_edges, leaves | {int(v)}))
            used_nodes |= (leaves | {int(v)})
        if len(stars) >= want:
            break
    return stars


def gamma_budget(edges, k, q, a_star, star_nodes, star_edge_ids,
                 center=None):
    """μ_v 与 Q̃_v（Cauchy–Schwarz 预算；center 给定时排除中心关联边，
    对应第十轮修复后定理的中心隔离条件）。"""
    supp = star_nodes
    mu = 0.0
    for i, e in enumerate(edges):
        if i in star_edge_ids:
            continue
        if center is not None and center in e:
            continue
        ov = sum(1 for u in e if u in supp)
        if ov == 0:
            continue
        mu += q[i] * ov / k[i]    # hom: kept 正质量率 q；mixed: 插入率 π
    # 有界差：翻转 u 的误差至多改变其重叠边正质量 ≤ Σ_{e∋u} 1·ov/k
    ell2 = {}
    for i, e in enumerate(edges):
        if i in star_edge_ids:
            continue
        ov = sum(1 for u in e if u in supp)
        if ov == 0:
            continue
        w_pos = q[i] if a_star[i] == 1.0 else (1 - q[i])
        for u in e:
            if u in supp:
                ell2[u] = ell2.get(u, 0.0) + w_pos * ov / k[i]
    Q = sum(v * v for v in ell2.values())
    return mu, Q




def part_D(doc):
    """定位认证保守性：星块底特征向量的真实环境重叠质量 vs 两种预算。"""
    rng = np.random.default_rng(31337)
    for tag in ("cora",):
        edges, labels = load(tag)
        n = len(labels)
        c_star = np.bincount(labels).argmax()
        y2 = (labels == c_star).astype(int)
        p = 0.05
        k_arr, a_star, q_arr, d_bar, sgn = edge_rates(edges, y2, p)
        node_deg = np.zeros(n, dtype=int)
        for e in edges:
            for u in e:
                node_deg[u] += 1
        pool = np.flatnonzero(node_deg > 0)
        v0 = 2430
        rng2 = np.random.default_rng(777)
        pool_same = [u for u in pool
                     if y2[u] == y2[v0] and u != v0]
        leaves = [int(u) for u in rng2.choice(
            np.array(pool_same), size=min(400, len(pool_same)),
            replace=False)][:380]
        d = 190
        star = [tuple(sorted((v0, leaves[2 * j], leaves[2 * j + 1])))
                for j in range(d)]
        star_set = set(range(len(edges), len(edges) + d))
        edges_i = edges + star
        k_i = np.concatenate([k_arr, np.full(d, R_STAR)])
        a_i = np.concatenate([a_star, np.ones(d)])
        q_i = np.concatenate([q_arr, np.full(d,
                                             1 - (1 - p) ** 3 - p ** 3)])
        d_i = np.concatenate([d_bar, np.full(d, -(1 - (1-p)**3 - p**3))])
        supp = set([v0]) | set(leaves[:2 * d])
        idx = {u: i for i, u in enumerate(sorted(supp))}
        N = len(supp)
        rows_rec = []
        for it in range(20):
            flip = rng.random(n) < p
            pred2 = np.where(flip, 1 - y2, y2)
            # 星块（小矩阵）
            w_star = []
            for e in star:
                same = all(pred2[u] == pred2[e[0]] for u in e)
                w_star.append((0.0 if same else -1.0) + (1 - (1-p)**3 - p**3))
            w_star = np.array(w_star)
            M = np.zeros((N, N))
            for j, e in enumerate(star):
                ids = [idx[u] for u in e]
                for a_ in ids:
                    for b_ in ids:
                        M[a_, b_] += w_star[j] / 3
            lam, V = np.linalg.eigh(M)
            x = V[:, 0] if abs(lam[0]) >= abs(lam[-1]) else V[:, -1]
            S = int((w_star < 0).sum())
            # 真实环境正重叠质量
            Pv = 0.0
            Pv_center = 0.0
            for i, e in enumerate(edges_i):
                if i in star_set:
                    continue
                ov = [u for u in e if u in supp]
                if not ov:
                    continue
                w_pos = (q_i[i] if a_i[i] == 1.0 else 1 - q_i[i])
                w_e = (w_pos if (a_i[i] == 1.0) ==
                       (len(set(pred2[list(e)].tolist())) == 1) else 0.0)
                # 中心化正质量：hom kept → +q；mixed inserted → 1-π
                if a_i[i] == 1.0:
                    kept = len(set(pred2[list(e)].tolist())) == 1
                    w_e = q_i[i] if kept else 0.0
                else:
                    ins = len(set(pred2[list(e)].tolist())) == 1
                    w_e = (1 - q_i[i]) if ins else 0.0
                qf = (sum(x[idx[u]] for u in ov)) ** 2 / len(e)
                Pv += w_e * qf
                if v0 in e:
                    Pv_center += w_e * qf
            rows_rec.append(dict(S=S, lam_min=float(lam[0]),
                                 Pv=float(Pv), Pv_center=float(Pv_center)))
        Pvs = [r["Pv"] for r in rows_rec]
        Pcs = [r["Pv_center"] for r in rows_rec]
        print(f"[{tag} p={p} d={d}] 20 draws: 实际环境正重叠质量 "
              f"P_v 均值 {np.mean(Pvs):.3f}（其中中心相邻边 "
              f"{np.mean(Pcs):.3f}）；S 均值 {np.mean([r['S'] for r in rows_rec]):.0f}"
              f"；λ_min(star) 均值 {np.mean([r['lam_min'] for r in rows_rec]):.2f}")
        doc["cancellation_locality"] = dict(
            tag=tag, p=p, d=d, draws=rows_rec,
            Pv_mean=float(np.mean(Pvs)), Pv_center_mean=float(np.mean(Pcs)))


def gamma_budget_local(edges, k, q, a_star, star_nodes, center,
                      star_edge_ids, S_min):
    """中心局域预算（Corollary，中心隔离条件下 F 无中心边）。"""
    supp = star_nodes
    mu = 0.0
    ell2 = {}
    for i, e in enumerate(edges):
        if i in star_edge_ids or center in e:
            continue
        ov = sum(1 for u in e if u in supp)
        if ov == 0:
            continue
        b = (2.0 if center in e else 0.0) + 2.0 * ov * ov / S_min ** 2
        w_bar = q[i] if a_star[i] == 1.0 else q[i]
        w_hat = q[i] if a_star[i] == 1.0 else (1 - q[i])
        mu += w_bar * b / k[i]
        for u in e:
            if u in supp:
                ell2[u] = ell2.get(u, 0.0) + w_hat * b / k[i]
    Q = sum(v * v for v in ell2.values())
    return mu, Q


def main():
    doc = dict(protocol=("binary-majority binarization; candidate stars of "
                         "homogeneous 3-edges with disjoint leaves; "
                         "per-edge rates from real label compositions; "
                         "eta=eta2=delta_B=0.01; MC K=200 with per-edge "
                         "centering on the full mixed graph"),
               settings=[])
    for tag in ("cora", "dblp"):
        edges, labels = load(tag)
        n = len(labels)
        c_star = np.bincount(labels).argmax()
        y2 = (labels == c_star).astype(int)
        t0 = time.time()
        for p in P_LIST:
            k_arr, a_star, q_arr, d_bar, sgn = edge_rates(edges, y2, p)
            stars = build_stars(edges, y2, a_star, k_arr, n)
            if not stars:
                print(f"[{tag} p={p}] 原生星：二元化同质 3-边最大不相交 "
                      f"度 <10，无候选星（如实记录）", flush=True)
                doc["settings"].append(dict(tag=tag, p=p, n=n, m=len(edges),
                                            n_stars=0, stars=[]))
                continue
            # B 上界（全图逐边率）
            Vmat = build_op(n, edges, q_arr * (1 - q_arr))
            v_B = opnorm(Vmat, psd=True)
            L = np.log(2 * n / DELTA_B)
            U_B = np.sqrt(2 * v_B * L) + 2 * 1.0 * L / 3
            q_r = 1 - (1 - p) ** R_STAR - p ** R_STAR
            c_rp = (1 - p) * ((1 - p) ** (R_STAR - 1) - p ** (R_STAR - 1))
            cert = []
            for v, eids, supp in stars:
                d = len(eids)
                L_A = (d * c_rp - np.sqrt(d / 2 * np.log(1 / ETA))) / R_STAR
                mu, Q = gamma_budget(edges, k_arr, q_arr, a_star, supp,
                                     set(eids), center=v)
                Gam = mu + np.sqrt(Q / 2 * np.log(1 / ETA2))
                cert.append(dict(v=v, d=d, L_A=L_A, mu=mu, Q=Q, Gamma=Gam,
                                 certified=bool(L_A - Gam > U_B)))
            n_cert = sum(c["certified"] for c in cert)
            ds = [c["d"] for c in cert]
            print(f"[{tag} p={p}] t={len(stars)} d(min/med/max)="
                  f"{min(ds)}/{int(np.median(ds))}/{max(ds)}  "
                  f"U_B={U_B:.2f}  认证星 {n_cert}/{len(stars)}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            for c in cert[:4]:
                print(f"    v={c['v']} d={c['d']} L_A={c['L_A']:.2f} "
                      f"μ={c['mu']:.2f} Γ={c['Gamma']:.2f} "
                      f"{'✓认证' if c['certified'] else '未认证'} "
                      f"(需 L_A-Γ > U_B={U_B:.2f})")
            doc["settings"].append(dict(tag=tag, p=p, n=n, m=len(edges),
                                        n_stars=len(stars),
                                        stars=cert, U_B=U_B, v_B=v_B))
    # Part B：植入星核到真实环境的非空性检验（环境全真实）
    rng0 = np.random.default_rng(777)
    for tag in ("cora", "dblp"):
        edges, labels = load(tag)
        n = len(labels)
        c_star = np.bincount(labels).argmax()
        y2 = (labels == c_star).astype(int)
        node_deg = np.zeros(n, dtype=int)
        for e in edges:
            for u in e:
                node_deg[u] += 1
        pool = np.flatnonzero(node_deg > 0)
        v0 = int(pool[rng0.integers(len(pool))])
        pool_same = [u for u in pool
                     if y2[u] == y2[v0] and u != v0]
        # 叶子与中心真标签同类 ⟹ 植入星边是真同质边（定理对象）
        leaves = [int(u) for u in rng0.choice(
            np.array(pool_same), size=min(400, len(pool_same)),
            replace=False)][:380]
        for p in P_LIST:
            k_arr, a_star0, q_arr0, d_bar0, sgn0 = edge_rates(edges, y2, p)
            q_r = 1 - (1 - p) ** R_STAR - p ** R_STAR
            c_rp = (1 - p) * ((1 - p) ** (R_STAR - 1) - p ** (R_STAR - 1))
            L = np.log(2 * n / DELTA_B)
            rowsB = []
            for d in (50, 100, 150, 190):
                star_eids = []
                star_edges = []
                for j in range(d):
                    e = tuple(sorted((v0, leaves[2 * j], leaves[2 * j + 1])))
                    star_eids.append(len(edges) + j)
                    star_edges.append(e)
                edges_i = edges + star_edges
                k_i = np.concatenate([k_arr, np.full(d, R_STAR)])
                a_i = np.concatenate([a_star0, np.ones(d)])
                q_i = np.concatenate([q_arr0, np.full(d, q_r)])
                d_i = np.concatenate([d_bar0, np.full(d, -q_r)])
                s_i = np.concatenate([sgn0, np.full(d, -1.0)])
                v_B = opnorm(build_op(n, edges_i, q_i * (1 - q_i)), psd=True)
                U_B = np.sqrt(2 * v_B * L) + 2 * L / 3
                supp = set([v0]) | set(leaves[:2 * d])
                mu, Q = gamma_budget(edges_i, k_i, q_i, a_i, supp,
                                 set(star_eids), center=v0)
                Gam = mu + np.sqrt(Q / 2 * np.log(1 / ETA2))
                S_min = d * (1 - p ** (R_STAR - 1)) - np.sqrt(
                    d / 2 * np.log(1 / ETA))
                mu_l, Q_l = gamma_budget_local(
                    edges_i, k_i, q_i, a_i, supp, v0, set(star_eids), S_min)
                Gam_l = mu_l + np.sqrt(Q_l / 2 * np.log(1 / ETA2))
                L_A = (d * c_rp - np.sqrt(d / 2 * np.log(1 / ETA))) / R_STAR
                rowsB.append(dict(d=d, v_B=v_B, U_B=U_B, mu=mu, Gamma=Gam,
                                  mu_loc=mu_l, Gamma_loc=Gam_l, L_A=L_A,
                                  certified=bool(L_A - Gam > U_B),
                                  certified_loc=bool(L_A - Gam_l > U_B)))
            dmin = next((r["d"] for r in rowsB if r["certified"]), None)
            dmin_l = next((r["d"] for r in rowsB
                           if r["certified_loc"]), None)
            print(f"[{tag} p={p}] 植入检验：v0={v0}，"
                  + " ".join(f"d={r['d']}:L_A-Γ={r['L_A']-r['Gamma']:.1f}"
                             f"/U_B={r['U_B']:.1f}"
                             f"{'✓' if r['certified'] else '✗'}"
                             for r in rowsB)
                  + " | 局域: "
                  + " ".join(f"d={r['d']}:Γloc={r['Gamma_loc']:.1f}"
                             f"{'✓' if r['certified_loc'] else '✗'}"
                             for r in rowsB)
                  + ((f" → 阈值 CS:{dmin}/局域:{dmin_l}")
                     if (dmin or dmin_l) else " → 全部未认证"),
                  flush=True)
            for s in doc["settings"]:
                if s["tag"] == tag and s["p"] == p:
                    s["implant"] = dict(center=v0, rows=rowsB,
                                        certified_min_d=dmin,
                                        certified_min_d_local=dmin_l)

    # Part C：真实混合图（无植入）MC 分离频率（两数据集 × 两 p）
    rng = np.random.default_rng(20261009)
    K_HERE = 100
    rng = np.random.default_rng(20261009)
    for tag in ("cora", "dblp"):
        edges, labels = load(tag)
        n = len(labels)
        c_star = np.bincount(labels).argmax()
        y2 = (labels == c_star).astype(int)
        for p in P_LIST:
            k_arr, a_star, q_arr, d_bar, sgn = edge_rates(edges, y2, p)
            t0 = time.time()
            sep = 0
            bAs, bBs = [], []
            for it in range(K_HERE):
                flip = rng.random(n) < p
                pred2 = np.where(flip, 1 - y2, y2)
                pmn = np.array([pred2[e[0]] for e in edges])
                pmx = np.array([pred2[e[-1]] for e in edges])
                pad = np.full((len(edges), max(k_arr)), -1, dtype=np.int64)
                for i, e in enumerate(edges):
                    pad[i, :len(e)] = e
                pv = pred2[pad]
                valid = pad >= 0
                pmn2 = np.min(np.where(valid, pv, 1 << 30), axis=1)
                pmx2 = np.max(np.where(valid, pv, -1), axis=1)
                wA = (pmn2 == pmx2).astype(float)
                xi = rng.random(len(edges)) < q_arr
                wB = a_star + np.where(xi, sgn, 0.0)
                bA = opnorm(build_op(n, edges,
                                      (wA - a_star) - d_bar))
                bB = opnorm(build_op(n, edges,
                                      (wB - a_star) - d_bar))
                bAs.append(bA); bBs.append(bB)
                sep += int(bA > bB)
            print(f"[{tag} p={p}] 纯真实图 MC {K_HERE} 次：分离 "
                  f"{sep}/{K_HERE} ({sep/K_HERE:.1%})，"
                  f"β̃A 均值 {np.mean(bAs):.2f} β̃B 均值 {np.mean(bBs):.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            for s in doc["settings"]:
                if s["tag"] == tag and s["p"] == p:
                    s["mc"] = dict(K=K_HERE, sep_freq=sep / K_HERE,
                                   betaA_mean=float(np.mean(bAs)),
                                   betaB_mean=float(np.mean(bBs)))
    part_D(doc)
    dest = RESULTS / "mixed_separation_check.json"
    dest.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"\n已写出 {dest}")


if __name__ == "__main__":
    main()
