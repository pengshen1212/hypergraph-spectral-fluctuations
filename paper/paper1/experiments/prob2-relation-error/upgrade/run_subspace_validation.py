"""Single spectral-subspace readout for the influence upgrade (G5).

For each native (graph, p) setting: regenerate the G4 controlled draws with
the registered seeds, compute the top-k (k=8 fixed) eigenspace of each A/B
centered-operator realization... readout definition follows the existing
spectral-subspace experiment: eigenspaces of the ESTIMATED operator
B(a) = sum_e a_e H_e (unit clique-expansion weights 1/|e|); the reference
subspace is that of the common expected operator sum_e p_e H_e (never
recomputed from test draws). Metric: ||Q Q^T - Q_ref Q_ref^T||_F.
Eigengap (lambda_8 - lambda_9 of the reference) recorded per setting.

Usage:
    python run_subspace_validation.py --protocol protocol.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as spsp
from scipy.sparse.linalg import eigsh

from influence_families import sample_weights_A, sample_weights_B

sys.stdout.reconfigure(encoding="utf-8")
UP = Path(__file__).resolve().parent
DATA = {"cora": "C:/Users/Peng1/OneDrive/文档/ChatGPT/hypergraph/data/cora-cc",
        "dblp": "C:/Users/Peng1/OneDrive/文档/ChatGPT/hypergraph/data/dblp-ca"}
K = 8


def protocol_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def load_native(tag):
    with open(f"{DATA[tag]}/hypergraph.pickle", "rb") as f:
        hg = pickle.load(f)
    with open(f"{DATA[tag]}/labels.pickle", "rb") as f:
        labels = np.array(pickle.load(f))
    edges = [tuple(sorted(int(u) for u in s)) for s in hg.values()]
    return edges, np.array(labels)


def build_ops(n, edges):
    rr, cc, lens = [], [], []
    for e in edges:
        for a_ in e:
            for b_ in e:
                rr.append(a_)
                cc.append(b_)
        lens.append(len(e))
    rows = np.array(rr, dtype=np.int32)
    cols = np.array(cc, dtype=np.int32)
    lens = np.array(lens)
    entry_edge = np.repeat(np.arange(len(edges)), lens ** 2)

    def mat(w):
        vals = (np.asarray(w, dtype=float) / lens)[entry_edge]
        return spsp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return mat


def topk_projector(M, k):
    vals, vecs = eigsh(M, k=k + 1, which="LA", tol=1e-8, maxiter=100000)
    order = np.argsort(-vals)
    vals, vecs = vals[order], vecs[:, order]
    Q = vecs[:, :k]
    gap = float(vals[k - 1] - vals[k])
    return Q, gap


def proj_dist(QA, QB):
    """||P_A - P_B||_F via 2k - 2||QA^T QB||_F^2 (both k-dim)."""
    k = QA.shape[1]
    S = QA.T @ QB
    return float(np.sqrt(max(2 * k - 2 * float((S * S).sum()), 0.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default=str(UP / "protocol.json"))
    ap.add_argument("--draws", type=int, default=30)
    args = ap.parse_args()
    ph = protocol_hash(args.protocol)
    out = dict(protocol_hash=ph, k=K, draws=args.draws, settings=[])
    for tag in ("cora", "dblp"):
        edges, labels = load_native(tag)
        n = len(labels)
        y2 = (labels == np.bincount(labels).argmax()).astype(int)
        mat = build_ops(n, edges)
        for p in (0.05, 0.15):
            P = np.zeros((n, 2))
            P[:, 1] = np.where(y2 == 1, 1 - p, p)
            P[:, 0] = 1 - P[:, 1]
            t0 = time.time()
            pe = np.array([np.prod(P[list(e), 0]) + np.prod(P[list(e), 1])
                           for e in edges])
            Mref = mat(pe)
            Qref, gap = topk_projector(Mref, K)
            rng = np.random.default_rng(7200001 + int(p * 100))
            dA, dB = [], []
            fails = 0
            for _ in range(args.draws):
                wA_all = sample_weights_A(edges, P, rng)
                wB_all = sample_weights_B(edges, P, rng)
                # readout uses the estimated operator B(a) = sum a_e H_e
                aA = wA_all + pe
                aB = wB_all + pe
                try:
                    QA, _ = topk_projector(mat(aA), K)
                    QB, _ = topk_projector(mat(aB), K)
                except Exception:
                    fails += 1
                    continue
                dA.append(proj_dist(QA, Qref))
                dB.append(proj_dist(QB, Qref))
            dA, dB = np.array(dA), np.array(dB)
            rng2 = np.random.default_rng(7201001 + int(p * 100))
            boots = []
            for _ in range(1000):
                idx = rng2.integers(0, len(dA), len(dA))
                boots.append(dA[idx].mean() - dB[idx].mean())
            boots = np.array(boots)
            res = dict(tag=tag, p=p, n=n, m=len(edges), eigengap=gap,
                       mean_distA=float(dA.mean()),
                       mean_distB=float(dB.mean()),
                       diff=float(dA.mean() - dB.mean()),
                       diff_ci=[float(np.percentile(boots, 2.5)),
                                float(np.percentile(boots, 97.5))],
                       dispersionA=float(dA.std()),
                       dispersionB=float(dB.std()),
                       eigensolver_failures=fails,
                       wall_seconds=time.time() - t0)
            out["settings"].append(res)
            print(f"[{tag} p={p}] gap={gap:.4f} distA={res['mean_distA']:.4f}"
                  f" distB={res['mean_distB']:.4f} "
                  f"diff={res['diff']:+.4f} "
                  f"[{res['diff_ci'][0]:+.4f},{res['diff_ci'][1]:+.4f}] "
                  f"dispA={res['dispersionA']:.4f} "
                  f"dispB={res['dispersionB']:.4f} "
                  f"({res['wall_seconds']:.0f}s, {fails} fails)", flush=True)
    dest = UP / "results" / "subspace_validation.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
