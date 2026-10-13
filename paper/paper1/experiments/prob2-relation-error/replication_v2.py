"""修正版独立重复（用户审计后，2026-10-01）。

修正（相对 replication.py）：
  1. 逐边解析概率按实际标签组成计算（非通用公式）。
  2. C 模型去掉双重过滤：每对独立以 q_e 抽，无整边跳过。
  3. 汇总统计改为 β̄A/β̄C（均值之比），bootstrap CI 对均值之比做。
  4. 使用全新种子（BASE_SEED=7777）。
"""

from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

sys.stdout.reconfigure(encoding="utf-8")

CORA = "C:/Users/Peng1/OneDrive/文档/ChatGPT/hypergraph/data/cora-cc"
DBLP = "C:/Users/Peng1/OneDrive/文档/ChatGPT/hypergraph/data/dblp-ca"

P_LIST = (0.05, 0.10, 0.20, 0.30)
N_BATCHES = 4
N_DRAWS = 50
BASE_SEED = 7777    # 全新种子（与 replication.py 的 2020 不同）


def load(data_dir):
    with open(f"{data_dir}/features.pickle", "rb") as f:
        feats = pickle.load(f)
    with open(f"{data_dir}/hypergraph.pickle", "rb") as f:
        hg = pickle.load(f)
    with open(f"{data_dir}/labels.pickle", "rb") as f:
        labels = np.array(pickle.load(f))
    edges = [sorted(s) for s in hg.values()]
    return feats, edges, labels


def node_pred_prob(y_i, c, p, C):
    return (1 - p) + p / C if c == y_i else p / C


def edge_error_prob(labels_e, p, C, is_hom):
    """逐边解析错误概率（按实际标签组成）。"""
    p_all_same = sum(np.prod([node_pred_prob(y, c, p, C) for y in labels_e])
                     for c in range(C))
    return 1.0 - p_all_same if is_hom else p_all_same


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


def clique_op(n, pair_entries):
    rows, cols, vals = [], [], []
    for i, j, k, sgn in pair_entries:
        w_off = 1.0 / k
        w_diag = 1.0 / (k * (k - 1)) if k > 1 else 0.0
        rows.extend([i, j, i, j]); cols.extend([j, i, i, j])
        vals.extend([sgn * w_off, sgn * w_off, sgn * w_diag, sgn * w_diag])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def snorm(B):
    if B.nnz == 0:
        return 0.0
    try:
        la = eigsh(B, k=1, which='LA', return_eigenvectors=False,
                   tol=1e-5, maxiter=5000)[0]
        li = eigsh(B, k=1, which='SA', return_eigenvectors=False,
                   tol=1e-5, maxiter=5000)[0]
        return float(max(abs(la), abs(li)))
    except ArpackNoConvergence:
        raise RuntimeError(f"Lanczos no convergence on nnz={B.nnz}")


def all_hom(pred, edges):
    return np.array([1.0 if len(set(pred[i] for i in e)) == 1 else 0.0
                     for e in edges])


def run_setting(name, edges_all, labels, tri_only):
    n = len(labels)
    C = len(np.unique(labels))
    edges = [e for e in edges_all if len(e) == 3] if tri_only else edges_all
    a_star = all_hom(labels, edges)
    # 每条边的真实标签
    edge_labels = [tuple(labels[i] for i in e) for e in edges]
    results = {}

    for p in P_LIST:
        # 逐边解析概率
        flip_probs = np.array([
            edge_error_prob(edge_labels[i], p, C, a_star[i] == 1.0)
            for i in range(len(edges))])
        all_bAs, all_bBs, all_bCs, all_errs = [], [], [], []
        for batch in range(N_BATCHES):
            rng = np.random.default_rng(BASE_SEED + batch * 10000 + int(p * 1000))
            for draw in range(N_DRAWS):
                # A：受控注错（从全部类均匀重抽）
                W = (rng.random(n) < p).astype(int)
                pred = labels.copy()
                flip = W == 1
                pred[flip] = rng.integers(0, C, size=flip.sum())
                a_hat = all_hom(pred, edges)
                dA = a_hat - a_star
                bA = snorm(build_op(n, edges, dA))
                err = float(np.mean(pred != labels))
                # B：独立超边，按逐边概率
                dB = np.zeros(len(edges))
                for i in range(len(edges)):
                    if rng.random() < flip_probs[i]:
                        dB[i] = -1.0 if a_star[i] == 1.0 else 1.0
                bB = snorm(build_op(n, edges, dB))
                # C：独立位置，正确权重，无双重过滤
                ent = []
                for i, e in enumerate(edges):
                    k = len(e)
                    sgn = -1 if a_star[i] == 1.0 else 1
                    for a_ in range(k):
                        for b_ in range(a_ + 1, k):
                            if rng.random() < flip_probs[i]:
                                ent.append((e[a_], e[b_], k, sgn))
                bC = snorm(clique_op(n, ent)) if ent else 0.0
                all_bAs.append(bA); all_bBs.append(bB); all_bCs.append(bC)
                all_errs.append(err)

        bAs = np.array(all_bAs); bBs = np.array(all_bBs)
        bCs = np.array(all_bCs)
        # 均值之比
        ac_ratio = bAs.mean() / max(bCs.mean(), 1e-9)
        ab_ratio = bAs.mean() / max(bBs.mean(), 1e-9)
        # Bootstrap CI 对均值之比
        boot = np.random.default_rng(999)
        n_s = len(bAs)
        ac_boots = []
        for _ in range(2000):
            idx = boot.choice(n_s, n_s, replace=True)
            ac_boots.append(bAs[idx].mean() / max(bCs[idx].mean(), 1e-9))
        ci_lo, ci_hi = np.percentile(ac_boots, [2.5, 97.5])
        results[p] = dict(
            bA_mean=float(bAs.mean()), bB_mean=float(bBs.mean()),
            bC_mean=float(bCs.mean()),
            ac_ratio=float(ac_ratio), ac_lo=float(ci_lo), ac_hi=float(ci_hi),
            ab_ratio=float(ab_ratio),
            err_mean=float(np.mean(all_errs)), n=n_s)
    return results


def main():
    all_results = {}
    for name, path in (("Cora", CORA), ("DBLP", DBLP)):
        feats, edges, labels = load(path)
        for tri in (False, True):
            tag = "三元" if tri else "全边"
            key = f"{name}-{tag}"
            print(f"\n{'='*84}")
            print(f"[{key}] {N_BATCHES} 批 × {N_DRAWS} 抽样（修正版，新种子 {BASE_SEED}）")
            print(f"{'='*84}")
            r = run_setting(name, edges, labels, tri)
            all_results[key] = {str(k): v for k, v in r.items()}
            print(f"{'p':>6} | {'βA':>7} {'βB':>7} {'βC':>7} | "
                  f"{'β̄A/β̄C':>7} {'95%CI':>14} | {'β̄A/β̄B':>7} | {'err':>5}")
            for p in P_LIST:
                d = r[p]
                print(f"{p:>6.2f} | {d['bA_mean']:>7.2f} {d['bB_mean']:>7.2f} "
                      f"{d['bC_mean']:>7.2f} | {d['ac_ratio']:>7.2f} "
                      f"[{d['ac_lo']:.2f},{d['ac_hi']:.2f}] | "
                      f"{d['ab_ratio']:>7.2f} | {d['err_mean']:>5.3f}")

    # 预注册检验
    print("\n" + "=" * 84)
    print("预注册检验（修正版）")
    print("=" * 84)
    h1_ok = all(all_results[f"{d}-{t}"]["0.05"]["ac_lo"] > 1.0
                for d in ("Cora", "DBLP") for t in ("全边", "三元"))
    cora_full = all_results["Cora-全边"]
    p02 = cora_full.get("0.2") or cora_full.get("0.20")
    h2_ok = p02["ac_hi"] < 1.0
    print(f"  H1（p=0.05 四格 CI 下界全 > 1）: {'✓ 通过' if h1_ok else '✗ 不通过'}")
    print(f"  H2（Cora 全边 p=0.20 CI 上界 < 1）: "
          f"{'✓ 反例稳定' if h2_ok else '✗ 反例不稳定'}")

    with open("results/replication_v2.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=1)
    print("\n已写出 results/replication_v2.json")


if __name__ == "__main__":
    main()
