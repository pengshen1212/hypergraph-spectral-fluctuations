"""Formal structure + native validation for the influence theory (G4).

Runs the frozen protocol: family A (6 d-points) and family B (6 gamma
-points) at 256 draws each; native Cora/DBLP at p in {0.05, 0.15} with
30 draws each. Every registered unit (including failures) is recorded.

Usage:
    python run_influence_validation.py --protocol protocol.json
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

from influence_families import (
    beta_centered,
    build_family_A,
    build_family_B,
    edge_probabilities,
    influence_at_witness,
    quadratic_at,
    sample_weights_A,
    sample_weights_B,
    theory_bounds_family_A,
)

sys.stdout.reconfigure(encoding="utf-8")
UP = Path(__file__).resolve().parent
DATA = {"cora": "C:/Users/Peng1/OneDrive/文档/ChatGPT/hypergraph/data/cora-cc",
        "dblp": "C:/Users/Peng1/OneDrive/文档/ChatGPT/hypergraph/data/dblp-ca"}


def protocol_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


class FastOps:
    """Precomputed COO structure; per-draw value update only."""

    def __init__(self, n, edges):
        self.n = n
        self.edges = edges
        lens = np.array([len(e) for e in edges])
        self.lens = lens
        rr, cc = [], []
        for e in edges:
            for a_ in e:
                for b_ in e:
                    rr.append(a_)
                    cc.append(b_)
        self.rows = np.array(rr, dtype=np.int32)
        self.cols = np.array(cc, dtype=np.int32)
        self.edge_of_entry = np.repeat(np.arange(len(edges)), lens ** 2)
        self.edge_pos = {e: j for j, e in enumerate(edges)}
        # connected components of the hypergraph (for blockwise spectra)
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for e in edges:
            for v in list(e)[1:]:
                ra, rb = find(list(e)[0]), find(v)
                if ra != rb:
                    parent[ra] = rb
        comp = {}
        for v in range(n):
            comp.setdefault(find(v), []).append(v)
        self.blocks = [np.array(vs) for vs in comp.values()]
        self.block_edges = []
        for vs in self.blocks:
            vs_set = set(vs.tolist())
            self.block_edges.append(
                [edges[j] for j in range(len(edges))
                 if edges[j][0] in vs_set])

    def beta(self, w):
        """Exact operator norm; blockwise dense for small components."""
        vals = np.asarray(w, dtype=float)
        best = 0.0
        for vs, bedges in zip(self.blocks, self.block_edges):
            m = len(vs)
            if m == 0:
                continue
            if m <= 1500:
                idx = {int(v): i for i, v in enumerate(vs)}
                M = np.zeros((m, m))
                for j, e in enumerate(bedges):
                    ii = [idx[int(v)] for v in e]
                    M[np.ix_(ii, ii)] += vals[self.edge_pos[j]] / len(e)
                ev = np.linalg.eigvalsh(M)
                best = max(best, float(np.abs(ev).max()))
            else:
                M = self._sparse_block(vs, bedges, vals)
                if M.nnz == 0:
                    continue
                from scipy.sparse.linalg import eigsh
                la = eigsh(M, k=1, which="LA", tol=1e-6,
                           maxiter=100000,
                           return_eigenvectors=False)[0]
                li = eigsh(M, k=1, which="SA", tol=1e-6,
                           maxiter=100000,
                           return_eigenvectors=False)[0]
                best = max(best, abs(float(la)), abs(float(li)))
        return best

    def _sparse_block(self, vs, bedges, vals):
        idx = {int(v): i for i, v in enumerate(vs)}
        rows, cols, vv = [], [], []
        for j, e in enumerate(bedges):
            ii = [idx[int(v)] for v in e]
            w_e = vals[self.edge_pos[j]] / len(e)
            for a_ in ii:
                for b_ in ii:
                    rows.append(a_)
                    cols.append(b_)
                    vv.append(w_e)
        return spsp.csr_matrix((vv, (rows, cols)), shape=(len(vs),) * 2)

    def edge_positions(self):
        pos = {}
        for j, e in enumerate(self.edges):
            pos[e] = j
        return pos

    def matrix(self, w):
        vals = (np.asarray(w) / self.lens)[self.edge_of_entry]
        return spsp.csr_matrix(
            (vals, (self.rows, self.cols)), shape=(self.n, self.n))

    def beta(self, w):
        M = self.matrix(w)
        if M.nnz == 0:
            return 0.0
        from scipy.sparse.linalg import eigsh
        la = eigsh(M, k=1, which="LA", tol=1e-6, maxiter=100000,
                   return_eigenvectors=False)[0]
        li = eigsh(M, k=1, which="SA", tol=1e-6, maxiter=100000,
                   return_eigenvectors=False)[0]
        return float(max(abs(la), abs(li)))

    def quadratic(self, w, x):
        s = 0.0
        for j, e in enumerate(self.edges):
            idx = list(e)
            s += w[j] * np.sum(x[idx]) ** 2 / len(e)
        return float(s)


def boot_ci(vals, n_boot=1000, seed=0, stat=np.mean):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    s = stat(vals[idx], axis=1)
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def run_point(n, edges, P, witness, draws, seed, extra=None):
    x = np.zeros(n)
    x[witness] = 1.0
    Phi, V1 = influence_at_witness(edges, P, x)
    ops = FastOps(n, edges)
    rng = np.random.default_rng(seed)
    recs = []
    missing = 0
    t0 = time.time()
    for _ in range(draws):
        wA = sample_weights_A(edges, P, rng)
        wB = sample_weights_B(edges, P, rng)
        try:
            bA = ops.beta(wA)
        except Exception:
            missing += 1
            bA = None
        try:
            bB = ops.beta(wB)
        except Exception:
            missing += 1
            bB = None
        recs.append(dict(betaA=bA, betaB=bB,
                         qA=ops.quadratic(wA, x),
                         qB=ops.quadratic(wB, x)))
    wall = time.time() - t0
    ok = [r for r in recs if r["betaA"] is not None
          and r["betaB"] is not None]
    bAs = np.array([r["betaA"] for r in ok])
    bBs = np.array([r["betaB"] for r in ok])
    qAs = np.array([r["qA"] for r in ok])
    qBs = np.array([r["qB"] for r in ok])
    out = dict(draws=draws, seed=seed, wall_seconds=wall,
               missing_units=missing, records=recs,
               Phi_witness=float(Phi[witness]), V1=float(V1),
               mean_betaA=float(bAs.mean()) if len(ok) else None,
               mean_betaB=float(bBs.mean()) if len(ok) else None,
               betaA_ci=boot_ci(bAs, seed=seed) if len(ok) else None,
               betaB_ci=boot_ci(bBs, seed=seed + 1) if len(ok) else None,
               sep_freq=float((bAs > bBs).mean()) if len(ok) else None,
               varA_q=float(np.var(qAs)) if len(ok) else None,
               varB_q=float(np.var(qBs)) if len(ok) else None,
               var_ratio_ci=None)
    if len(ok):
        rng = np.random.default_rng(seed + 3)
        idx = rng.integers(0, len(ok), size=(1000, len(ok)))
        va = np.var(qAs[idx], axis=1)
        vb = np.var(qBs[idx], axis=1)
        ratio = va / np.maximum(vb, 1e-15)
        out["var_ratio_ci"] = [float(np.percentile(ratio, 2.5)),
                               float(np.percentile(ratio, 97.5))]
    if extra:
        out.update(extra)
    return out


def native_point(tag, p, draws, seed):
    with open(f"{DATA[tag]}/hypergraph.pickle", "rb") as f:
        hg = pickle.load(f)
    with open(f"{DATA[tag]}/labels.pickle", "rb") as f:
        labels = np.array(pickle.load(f))
    edges = [tuple(sorted(int(u) for u in s)) for s in hg.values()]
    n = len(labels)
    y2 = (labels == np.bincount(labels).argmax()).astype(int)
    # controlled error model: independent flips at rate p (binarized)
    P = np.zeros((n, 2))
    P[:, 1] = np.where(y2 == 1, 1 - p, p)
    P[:, 0] = 1 - P[:, 1]
    # witness from degrees only (fixed before any sampling)
    deg = np.zeros(n, dtype=int)
    for e in edges:
        for u in e:
            deg[u] += 1
    witness = int(np.argmax(deg))
    Phi_arr, V1_exact = influence_witness_native(edges, P, witness)
    res = run_point(n, edges, P, witness, draws, seed,
                    extra=dict(tag=tag, p=p, witness=witness,
                               witness_degree=int(deg[witness])))
    res["V1"] = V1_exact
    res["top_nodes"] = top_v1_nodes(edges, P, witness, k=5)
    return res


def influence_witness_native(edges, P, witness):
    n = len(P)
    Phi = np.zeros(n)
    V1 = 0.0
    for e in edges:
        if witness not in e:
            continue
        he = 1.0 / len(e)
        idx = list(e)
        p1 = float(np.prod(P[idx, 1]))
        p0 = float(np.prod(P[idx, 0]))
        for u in idx:
            others = [v for v in idx if v != u]
            g1 = float(np.prod(P[others, 1])) if others else 1.0
            g0 = float(np.prod(P[others, 0])) if others else 1.0
            Phi[u] += (g1 - g0) * he
    V1 = float(np.sum(P[:, 0] * P[:, 1] * Phi ** 2))
    return Phi, V1


def top_v1_nodes(edges, P, witness, k=5):
    Phi, V1 = influence_witness_native(edges, P, witness)
    order = np.argsort(-np.abs(Phi))[:k]
    return [(int(u), float(Phi[u])) for u in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default=str(UP / "protocol.json"))
    ap.add_argument("--draws", type=int, default=256)
    ap.add_argument("--native-draws", type=int, default=30)
    ap.add_argument("--skip-native", action="store_true")
    args = ap.parse_args()
    proto = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    ph = protocol_hash(args.protocol)
    out = dict(protocol_hash=ph, formal=True, structure=[], native=[])
    t00 = time.time()

    fa = proto["structure_families"]["family_A_amplification"]["params"]
    print(f"== family A (t={fa['t_blocks']}, kappa={fa['kappa']}) ==",
          flush=True)
    for d in fa["d_values"]:
        edges, P, hubs = build_family_A(fa["t_blocks"], d,
                                        kappa=fa["kappa"], seed=20261009)
        th = theory_bounds_family_A(fa["t_blocks"], d, fa["kappa"])
        res = run_point(len(P), edges, P, hubs[0], args.draws,
                        7000001 + d,
                        extra=dict(family="A", d=d, theory=th))
        out["structure"].append(res)
        (UP / "results").mkdir(exist_ok=True)
        (UP / "results" / "structure_validation.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[A d={d}] cert={th['certified']} L_A={th['L_A']:.1f} "
              f"U_B={th['U_B']:.1f} | betaA={res['mean_betaA']:.2f} "
              f"[{res['betaA_ci'][0]:.1f},{res['betaA_ci'][1]:.1f}] "
              f"betaB={res['mean_betaB']:.2f} sep={res['sep_freq']:.1%} "
              f"({res['wall_seconds']:.0f}s)", flush=True)

    fb = proto["structure_families"]["family_B_competition"]["params"]
    print(f"== family B (d={fb['d_values'][-1]}, gamma sweep) ==",
          flush=True)
    for gamma in fb["gamma_values"]:
        edges, P, hubs = build_family_B(fb["d_values"][-1], gamma)
        res = run_point(len(P), edges, P, hubs[0], args.draws,
                        7100001 + int(gamma * 100),
                        extra=dict(family="B", d=fb["d_values"][-1],
                                   gamma=gamma))
        out["structure"].append(res)
        (UP / "results" / "structure_validation.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[B gamma={gamma}] Phi={res['Phi_witness']:.2f} "
              f"| betaA={res['mean_betaA']:.2f} betaB={res['mean_betaB']:.2f} "
              f"sep={res['sep_freq']:.1%} varA/varB="
              f"{res['varA_q'] / max(res['varB_q'], 1e-12):.3f} "
              f"({res['wall_seconds']:.0f}s)", flush=True)

    dest = UP / "results" / "structure_validation.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"\nwrote {dest} ({time.time() - t00:.0f}s total)")

    if not args.skip_native:
        out2 = dict(protocol_hash=ph, formal=True, settings=[])
        for tag in ("cora", "dblp"):
            for p in (0.05, 0.15):
                t0 = time.time()
                res = native_point(tag, p, args.native_draws,
                                   7200001 + int(p * 100))
                res["tag"], res["p"] = tag, p
                out2["settings"].append(res)
                print(f"[{tag} p={p}] witness={res['witness']} "
                      f"deg={res['witness_degree']} Phi={res['Phi_witness']:.3f} "
                      f"| betaA={res['mean_betaA']:.2f} "
                      f"betaB={res['mean_betaB']:.2f} "
                      f"sep={res['sep_freq']:.1%} ({time.time()-t0:.0f}s)",
                      flush=True)
        dest2 = UP / "results" / "native_validation.json"
        dest2.write_text(json.dumps(out2, ensure_ascii=False, indent=1),
                         encoding="utf-8")
        print(f"wrote {dest2}")


if __name__ == "__main__":
    main()
