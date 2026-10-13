"""Real-hypergraph A/B/C comparison with matched conditional edge marginals.

Arm 1 computes exact per-edge flip rates under independent node corruption.
Arm 2 uses type-pooled empirical rates because one realized predictor cannot
identify each edge's error probability.
"""

from __future__ import annotations

import pickle
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

sys.stdout.reconfigure(encoding="utf-8")

CORA = "C:/Users/Peng1/OneDrive/文档/ChatGPT/hypergraph/data/cora-cc"
DBLP = "C:/Users/Peng1/OneDrive/文档/ChatGPT/hypergraph/data/dblp-ca"


def load(data_dir):
    with open(f"{data_dir}/features.pickle", "rb") as f:
        feats = pickle.load(f)
    with open(f"{data_dir}/hypergraph.pickle", "rb") as f:
        hg = pickle.load(f)
    with open(f"{data_dir}/labels.pickle", "rb") as f:
        labels = np.array(pickle.load(f))
    edges = [sorted(s) for s in hg.values()]
    return feats, edges, labels


def build_op_sparse(n, edges, weights):
    rows, cols, vals = [], [], []
    for e, w in zip(edges, weights):
        if not w:
            continue
        k = len(e)
        for a in e:
            for b in e:
                rows.append(a); cols.append(b); vals.append(w / k)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def snorm_checked(B, label=""):
    """Spectral norm of a symmetric sparse matrix with residual validation."""
    if B.nnz == 0:
        return 0.0
    try:
        values, vectors = eigsh(B, k=1, which="LM", tol=1e-6,
                                maxiter=10000)
    except ArpackNoConvergence as exc:
        raise RuntimeError(f"Lanczos did not converge ({label})") from exc
    value = float(values[0])
    residual = np.linalg.norm(B @ vectors[:, 0] - value * vectors[:, 0])
    if residual > 1e-4 * max(1.0, abs(value)):
        raise RuntimeError(f"Lanczos residual too large ({label}): {residual}")
    return abs(value)


def all_hom(pred, edges):
    return np.array([1.0 if len(set(pred[i] for i in e)) == 1 else 0.0
                     for e in edges])


def clique_op_correct(n, pair_entries):
    """正确权重的团展开位置算子。pair_entries = [(i, j, k, sgn)]。

    每对 (i,j) 在 k-边内：非对角各 1/k，两对角各 1/[k(k-1)]。
    """
    rows, cols, vals = [], [], []
    for i, j, k, sgn in pair_entries:
        w_off = 1.0 / k
        w_diag = 1.0 / (k * (k - 1)) if k > 1 else 0.0
        rows.extend([i, j, i, j])
        cols.extend([j, i, i, j])
        vals.extend([sgn * w_off, sgn * w_off, sgn * w_diag, sgn * w_diag])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def edge_flip_probabilities(labels, edges, p, n_cls):
    """Exact per-edge flip rates for independent uniform-label redraws."""
    if not 0 <= p <= 1 or n_cls < 2:
        raise ValueError("Expected 0 <= p <= 1 and at least two classes")
    correct = 1 - p + p / n_cls
    other = p / n_cls
    rates = np.empty(len(edges), dtype=float)
    cache = {}
    for i, edge in enumerate(edges):
        if len(edge) < 2:
            raise ValueError("Position control requires edges of size at least two")
        _, counts = np.unique(labels[edge], return_counts=True)
        key = tuple(sorted(counts.tolist()))
        if key not in cache:
            size = len(edge)
            p_hom = sum(correct ** count * other ** (size - count)
                        for count in counts)
            p_hom += (n_cls - len(counts)) * other ** size
            cache[key] = p_hom
        rates[i] = 1 - cache[key] if len(counts) == 1 else cache[key]
    return np.clip(rates, 0.0, 1.0)


def sample_independent_edges(rng, a_star, rates):
    """Sample an independent Bernoulli relation error on each edge."""
    flips = (rng.random(len(rates)) < rates).astype(float)
    return np.where(a_star == 1, -flips, flips)


def sample_independent_positions(rng, n, edges, a_star, rates):
    """Sample independent clique positions with each edge's matched rate."""
    pairs = []
    for edge, true_hom, rate in zip(edges, a_star, rates):
        if rate == 0:
            continue
        size = len(edge)
        sign = -1 if true_hom else 1
        for a in range(size):
            for b in range(a + 1, size):
                if rng.random() < rate:
                    pairs.append((edge[a], edge[b], size, sign))
    return clique_op_correct(n, pairs)


def empirical_type_rates(a_star, delta):
    """Descriptive type-pooled rates from one predictor realization."""
    rates = np.zeros(len(delta), dtype=float)
    for true_hom in (0, 1):
        mask = a_star == true_hom
        if mask.any():
            rates[mask] = np.mean(delta[mask] != 0)
    return rates


def ratio_interval(numerators, denominators, seed, bootstrap_draws=2000):
    """Paired bootstrap interval for the displayed ratio of means."""
    numerator = np.asarray(numerators)
    denominator = np.asarray(denominators)
    if len(numerator) != len(denominator) or len(numerator) < 2:
        raise ValueError("Need at least two paired draws")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(numerator), size=(bootstrap_draws, len(numerator)))
    ratios = numerator[indices].mean(axis=1) / denominator[indices].mean(axis=1)
    return tuple(np.quantile(ratios, [0.025, 0.975]))


def nearest_prototypes(features, prototypes, block_size=5000):
    """Predict from sparse features without materializing a dense feature matrix."""
    pred = np.empty(features.shape[0], dtype=int)
    prototype_norms = np.sum(prototypes ** 2, axis=1)
    for lo in range(0, features.shape[0], block_size):
        hi = min(lo + block_size, features.shape[0])
        block = features[lo:hi]
        distances = np.asarray(block.multiply(block).sum(axis=1))
        distances = distances - 2 * (block @ prototypes.T) + prototype_norms
        pred[lo:hi] = np.argmin(distances, axis=1)
    return pred


def run_abc(name, feats, edges_all, labels, p_list, n_draws=20, seed=42,
            tri_only=False, arm2_seeds=()):
    n = len(labels)
    n_cls = len(np.unique(labels))
    edges = [e for e in edges_all if len(e) == 3] if tri_only else edges_all
    a_star = all_hom(labels, edges)
    n_hom = int(a_star.sum())
    tag = "三元子集" if tri_only else "全数据"
    print(f"\n{'='*88}")
    print(f"[{name}] {tag}：{len(edges)}/{len(edges_all)} 边"
          f"（{len(edges)/len(edges_all)*100:.0f}%），同质 {n_hom}/{len(edges)}"
          f"（{n_hom/len(edges)*100:.0f}%）")
    print(f"{'='*88}")

    # ---- Arm-1: 受控注错 ----
    print(f"\nArm-1（受控注错，n_draws={n_draws}）：")
    print(f"{'p':>6} | {'βA':>7} {'βB':>7} {'βC':>7} | {'A/C':>6} {'A/B':>6} | "
          f"{'95% CI(A/C)':>17}")
    for p in p_list:
        seed_seq = np.random.SeedSequence(seed + int(p * 1000))
        rng_a, rng_b, rng_c = [np.random.default_rng(child)
                               for child in seed_seq.spawn(3)]
        rates = edge_flip_probabilities(labels, edges, p, n_cls)
        bAs, bBs, bCs = [], [], []
        for _ in range(n_draws):
            W = rng_a.random(n) < p
            pred = labels.copy()
            pred[W] = rng_a.integers(0, n_cls, size=W.sum())
            a_hat = all_hom(pred, edges)
            dA = a_hat - a_star
            bAs.append(snorm_checked(build_op_sparse(n, edges, dA), f"A p={p}"))
            dB = sample_independent_edges(rng_b, a_star, rates)
            bBs.append(snorm_checked(build_op_sparse(n, edges, dB), f"B p={p}"))
            B_C = sample_independent_positions(rng_c, n, edges, a_star, rates)
            bCs.append(snorm_checked(B_C, f"C p={p}"))
        bA, bB, bC = np.mean(bAs), np.mean(bBs), np.mean(bCs)
        ci_low, ci_high = ratio_interval(bAs, bCs, seed + int(p * 1000) + 1)
        print(f"{p:>6.2f} | {bA:>7.2f} {bB:>7.2f} {bC:>7.2f} | {bA/bC:>6.2f} "
              f"{bA/bB:>6.2f} | [{ci_low:.2f}, {ci_high:.2f}]")

    # ---- Arm-2: 真实预测器 ----
    if arm2_seeds:
        print(f"\nArm-2（真实预测器；B/C 为类型汇总率对照，非逐边精确匹配；seeds={arm2_seeds}）：")
        print(f"{'seed':>5} | {'acc':>5} | {'βA':>7} {'βB':>7} {'βC':>7} | "
              f"{'A/C':>6} {'A/B':>6}")
        bAs2, bBs2, bCs2 = [], [], []
        for s in arm2_seeds:
            rng = np.random.default_rng(s)
            idx = rng.permutation(n)
            train_idx = idx[:n // 2]
            protos = np.zeros((n_cls, feats.shape[1]))
            for c in range(n_cls):
                mask = labels[train_idx] == c
                if mask.sum() > 0:
                    protos[c] = np.asarray(feats[train_idx][mask].mean(axis=0)).ravel()
            pred = nearest_prototypes(feats, protos)
            acc = float(np.mean(pred == labels))
            a_hat = all_hom(pred, edges)
            dA = a_hat - a_star
            bA = snorm_checked(build_op_sparse(n, edges, dA), f"A2 s={s}")
            rates = empirical_type_rates(a_star, dA)
            rng_b, rng_c = [np.random.default_rng(child)
                            for child in np.random.SeedSequence(s + 1000).spawn(2)]
            dB = sample_independent_edges(rng_b, a_star, rates)
            bB = snorm_checked(build_op_sparse(n, edges, dB), f"B2 s={s}")
            B_C = sample_independent_positions(rng_c, n, edges, a_star, rates)
            bC = snorm_checked(B_C, f"C2 s={s}")
            bAs2.append(bA); bBs2.append(bB); bCs2.append(bC)
            print(f"{s:>5} | {acc:>5.3f} | {bA:>7.2f} {bB:>7.2f} {bC:>7.2f} | "
                  f"{bA/bC:>6.2f} {bA/bB:>6.2f}")
        print(f"  均值: A/C={np.mean(bAs2)/np.mean(bCs2):.2f}  "
              f"A/B={np.mean(bAs2)/np.mean(bBs2):.2f}")
    return


if __name__ == "__main__":
    # Cora
    feats, edges, labels = load(CORA)
    run_abc("Cora-CC", feats, edges, labels,
            (0.05, 0.10, 0.20, 0.30), n_draws=20, arm2_seeds=(100, 101, 102))
    run_abc("Cora-CC", feats, edges, labels,
            (0.05, 0.10, 0.20, 0.30), n_draws=20, tri_only=True)

    # DBLP
    feats, edges, labels = load(DBLP)
    run_abc("DBLP-CA", feats, edges, labels,
            (0.05, 0.10, 0.20), n_draws=15, arm2_seeds=(100,))
    run_abc("DBLP-CA", feats, edges, labels,
            (0.05, 0.10, 0.20, 0.30), n_draws=15, tri_only=True)
