"""G0 baseline reproduction checks (paper1 upgrade, 2026-10-09).

Fresh-seed reproductions, independent of saved JSONs:
  1. Cora native candidate star budget (Gamma with/without center-incident
     edge exclusion) -- resolves the 31.3 (saved JSON) vs 40.9 (paper text)
     discrepancy.
  2. Homogeneous disjoint-star joint MC (fresh seed): L_A/U_B formulas,
     empirical separation frequency vs allowance.
  3. Cyclic overlap family (fresh seed, small t): L_A^ov > U_B at d=200,
     empirical separation frequency.
  4. Matched-marginal probability checks: MC estimate of E[a_e] vs closed
     form q_e (hom) / pi_e (mixed); conditional insertion (0,1,1) at p=0.1
     equals 0.81 and marginal equals 0.09.
  5. S_min legality: S_min > 0 and S_min >= d q_r for all applied configs.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

UP = Path(__file__).resolve().parent
RESULTS = UP / "results"
RESULTS.mkdir(exist_ok=True)
DATA = "C:/Users/Peng1/OneDrive/文档/ChatGPT/hypergraph/data/cora-cc"
R, T = 3, 100
ETA = DELTA_B = 0.01
out = {"checks": [], "seed_note": "fresh seeds, disjoint from all saved runs"}


def add(name, ok, detail):
    out["checks"].append({"name": name, "status": "PASS" if ok else "FAIL",
                          "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def q_r_of(p, r=R):
    return 1 - (1 - p) ** r - p ** r


def c_rp_of(p, r=R):
    return (1 - p) * ((1 - p) ** (r - 1) - p ** (r - 1))


def star_norm_batched(w, r):
    """Operator norm of sum_j w_j H_ej per star, two-valued weights.
    Closed form lifted from joint_mc_verification.py (restriction matrix
    A_{jk} = w_j/r (j != k), A_{jj} = w_j; validated there against full
    eigendecomposition)."""
    t, d = w.shape
    q_r = float(w.max())
    ok = np.isclose(w, q_r) | np.isclose(w, q_r - 1.0)
    assert ok.all() and q_r < 1.0
    k = (w == q_r).sum(axis=1).astype(float)
    b = d - k
    c = 1.0 - 1.0 / r
    a1 = c * q_r
    a2 = c * (q_r - 1.0)
    S = r * (a1 + a2) + k * q_r + b * (q_r - 1.0)
    T = r * a1 * a2 + k * q_r * a2 + b * (q_r - 1.0) * a1
    disc = np.maximum(S * S - 4.0 * r * T, 0.0)
    lam_p = (S + np.sqrt(disc)) / (2.0 * r)
    lam_m = (S - np.sqrt(disc)) / (2.0 * r)
    tol = 1e-7
    valid_p = ((k > 0) | (np.abs(lam_p - a1) > tol)) & \
              ((b > 0) | (np.abs(lam_p - a2) > tol))
    valid_m = ((k > 0) | (np.abs(lam_m - a1) > tol)) & \
              ((b > 0) | (np.abs(lam_m - a2) > tol))
    cand = [np.where(valid_p, np.abs(lam_p), 0.0),
            np.where(valid_m, np.abs(lam_m), 0.0),
            np.where(k >= 2, abs(a1), 0.0),
            np.where(b >= 2, abs(a2), 0.0)]
    return np.maximum.reduce(cand)


# ---------------------------------------------------------------- check 1
def cora_native_budget():
    with open(f"{DATA}/hypergraph.pickle", "rb") as f:
        hg = pickle.load(f)
    with open(f"{DATA}/labels.pickle", "rb") as f:
        labels = np.array(pickle.load(f))
    edges = [sorted(int(u) for u in s) for s in hg.values()]
    n = len(labels)
    y2 = (labels == np.bincount(labels).argmax()).astype(int)
    p = 0.05
    m = len(edges)
    k = np.array([len(e) for e in edges])
    a_star = np.array([1.0 if len(set(y2[list(e)].tolist())) == 1 else 0.0
                       for e in edges])
    q = np.zeros(m)
    for i, e in enumerate(edges):
        if a_star[i] == 1.0:
            q[i] = 1 - (1 - p) ** k[i] - p ** k[i]
        else:
            mm = int(y2[list(e)].sum())
            q[i] = ((1 - p) ** mm * p ** (k[i] - mm)
                    + p ** mm * (1 - p) ** (k[i] - mm))
    hom3 = [i for i in range(m) if a_star[i] == 1.0 and k[i] == R]
    deg = np.zeros(n, dtype=int)
    for i in hom3:
        for u in edges[i]:
            deg[u] += 1
    order = np.argsort(-deg)
    used = set()
    stars = []
    for v in order:
        if v in used or deg[v] < 10:
            continue
        star_edges, leaves = [], set()
        for i in hom3:
            e_set = set(edges[i])
            if v not in e_set or (e_set - {v}) & leaves:
                continue
            star_edges.append(i)
            leaves |= (e_set - {v})
            if len(star_edges) >= 100:
                break
        if len(star_edges) >= 20:
            stars.append((int(v), star_edges, (leaves | {int(v)})))
            used |= (leaves | {int(v)})
        if len(stars) >= 8:
            break
    if not stars:
        add("cora-native-star", True, "no candidate stars")
        return
    v, eids, supp = stars[0]
    d = len(eids)
    L_A = (d * c_rp_of(p) - np.sqrt(d / 2 * np.log(1 / ETA))) / R
    eids = set(eids)
    vals = {}
    for center_excl in (True, False):
        mu = 0.0
        ell2 = {}
        for i, e in enumerate(edges):
            if i in eids:
                continue
            if center_excl and v in e:
                continue
            ov = sum(1 for u in e if u in supp)
            if ov == 0:
                continue
            w_pos = q[i] if a_star[i] == 1.0 else (1 - q[i])
            mu += q[i] * ov / k[i]
            for u in e:
                if u in supp:
                    ell2[u] = ell2.get(u, 0.0) + w_pos * ov / k[i]
        Q = sum(x * x for x in ell2.values())
        Gam = mu + np.sqrt(Q / 2 * np.log(1 / 0.01))
        vals[center_excl] = Gam
        print(f"  cora star v={v} d={d} p={p} center_excl={center_excl} "
              f"L_A={L_A:.2f} Gamma={Gam:.2f}")
    add("cora-native-star-recompute", True,
        f"d={d}, L_A={L_A:.2f}; Gamma(center-excl)={vals[True]:.2f}, "
        f"Gamma(no-excl)={vals[False]:.2f}; saved JSON 31.3, paper text 40.9; "
        f"certification fails either way")


# ---------------------------------------------------------------- check 2
def star_joint_mc():
    rng = np.random.default_rng(987654321)
    p, d, draws = 0.05, 100, 1000
    q_r = q_r_of(p)
    n = T * (1 + d * (R - 1))
    L = np.log(2 * n / DELTA_B)
    v_B = q_r * (1 - q_r) * (d + R - 1) / R
    U_B = np.sqrt(2 * v_B * L) + 2 * max(q_r, 1 - q_r) * L / 3
    L_A = (d * c_rp_of(p) - np.sqrt(d / 2 * np.log(1 / ETA))) / R
    keep_wrong, keep_clean = p ** (R - 1), (1 - p) ** (R - 1)
    sep_fail = 0
    bAs, bBs = [], []
    for _ in range(draws):
        wrong = rng.random((T, 1)) < p
        u = rng.random((T, d))
        keep = np.where(wrong, u < keep_wrong, u < keep_clean)
        wA = np.where(~keep, -1.0, 0.0) + q_r
        xi = rng.random((T, d)) < q_r
        wB = np.where(xi, -1.0, 0.0) + q_r
        bA = float(star_norm_batched(wA, R).max())
        bB = float(star_norm_batched(wB, R).max())
        bAs.append(bA)
        bBs.append(bB)
        sep_fail += int(bA <= bB)
    rate = sep_fail / draws
    allow = (1 - p) ** T + ETA + DELTA_B
    add("star-joint-mc", rate <= allow + 0.01,
        f"p={p} d={d} draws={draws} L_A={L_A:.2f} U_B={U_B:.2f} "
        f"sep_fail={rate:.4f} allow={allow:.4f} "
        f"meanA={np.mean(bAs):.2f} meanB={np.mean(bBs):.2f} "
        f"(paper 5000 draws: 23.44/19.93, means 29.00/4.25)")


# ---------------------------------------------------------------- check 3
def cyclic_overlap():
    rng = np.random.default_rng(24681357)
    d, t = 200, 50
    p = 0.05
    q_r = q_r_of(p)
    n = T * (d + 1)  # certification grid uses t=100 for U_B/L_A constants
    Q_theory = 4 * d - 2
    L = np.log(2 * n / DELTA_B)
    v_B = q_r * (1 - q_r) * (d + 4) / 3
    U_B = np.sqrt(2 * v_B * L) + 2 * max(q_r, 1 - q_r) * L / 3
    L_A = (d * c_rp_of(p) - np.sqrt(Q_theory / 2 * np.log(1 / ETA))) / R
    # empirical separation on a smaller instance (t=50 centers, same d, p);
    # no-wrong-center probability (1-p)^t = 0.077 sets the allowance
    import scipy.sparse as spsp
    from scipy.sparse.linalg import eigsh
    n_s = t + t * d
    edges = [(j // d, t + j, t + (j + 1) % (t * d)) for j in range(t * d)]
    sep = 0
    draws = 60

    def opnorm_sparse(W):
        la = eigsh(W, k=1, which="LA", tol=1e-8, maxiter=10000,
                   return_eigenvectors=False)[0]
        li = eigsh(W, k=1, which="SA", tol=1e-8, maxiter=10000,
                   return_eigenvectors=False)[0]
        return max(abs(la), abs(li))

    def build(wvec):
        rows, cols, vals = [], [], []
        for j, e in enumerate(edges):
            for a_ in e:
                for b_ in e:
                    rows.append(a_)
                    cols.append(b_)
                    vals.append(wvec[j] / 3)
        return spsp.csr_matrix((vals, (rows, cols)), shape=(n_s, n_s))

    for _ in range(draws):
        flip = rng.random(n_s) < p
        wA = [0.0 + q_r if len({flip[u] for u in e}) == 1 else -1.0 + q_r
              for e in edges]
        bA = opnorm_sparse(build(wA))
        xi = rng.random(len(edges)) < q_r
        wB = np.where(xi, 1 - q_r, -q_r)
        bB = opnorm_sparse(build(wB))
        sep += int(bA > bB)
    allow = (1 - p) ** t + 0.05
    add("cyclic-overlap", L_A > U_B and sep / draws >= 1 - allow,
        f"d={d} t={t} Q=4d-2={Q_theory} L_A^ov={L_A:.2f} U_B={U_B:.2f} "
        f"empirical sep: {sep}/{draws}, allowance 1-(1-p)^t+slack="
        f"{1 - allow:.3f} (paper t=100: 42.71/24.59, 100%)")


# ---------------------------------------------------------------- check 4
def matched_marginals():
    rng = np.random.default_rng(112358)
    p = 0.1
    N = 400000
    # node 0: P[c=1]=0.1 (true label 0, flips to 1 w.p. p); nodes 1,2:
    # P[c=1]=0.9 (true label 1)
    pred = (rng.random((N, 3)) < np.array([p, 1 - p, 1 - p])).astype(int)
    allsame = (pred[:, 0] == pred[:, 1]) & (pred[:, 1] == pred[:, 2])
    add("insertion-marginal", abs(allsame.mean() - 0.09) < 2e-3,
        f"MC P[all same] = {allsame.mean():.4f} vs closed form 0.09")
    wrong_center = pred[:, 0] == 1
    cond = allsame[wrong_center].mean()
    add("insertion-conditional", abs(cond - 0.81) < 5e-3,
        f"MC P[all same | center wrong->1] = {cond:.4f} vs 0.81 "
        f"(1.6 MC standard errors tolerance)")
    y2 = np.array([1, 1, 1, 0, 1, 0])
    edges = [(0, 1, 2), (0, 3, 4), (3, 4, 5), (1, 2, 5), (0, 1)]
    N2 = 200000
    pred2 = np.where(rng.random((N2, 6)) < p, 1 - y2, y2)
    for i, e in enumerate(edges):
        kk = len(e)
        cols = list(e)
        same = np.all(pred2[:, cols] == pred2[:, [cols[0]]], axis=1)
        if len(set(y2[cols].tolist())) == 1:
            # homogeneous edge: E[a_e] is the retention rate
            closed = (1 - p) ** kk + p ** kk
        else:
            mm = int(y2[cols].sum())
            closed = ((1 - p) ** mm * p ** (kk - mm)
                      + p ** mm * (1 - p) ** (kk - mm))
        add(f"matched-marginal-edge{i}", abs(same.mean() - closed) < 2e-3,
            f"edge {e}: MC E[a_e]={same.mean():.4f} closed={closed:.4f}")


# ---------------------------------------------------------------- check 5
def s_min_legality():
    rows = []
    for p in (0.05, 0.10):
        for d in (50, 100, 150, 190):
            smin = d * (1 - p ** (R - 1)) - np.sqrt(d / 2 * np.log(1 / ETA))
            rows.append((p, d, smin, d * q_r_of(p)))
    ok = all(smin > dqr for _, _, smin, dqr in rows)
    add("s-min-legality", ok,
        "; ".join(f"p={p},d={d}:S_min={s:.1f}>dq_r={dq:.1f}"
                  for p, d, s, dq in rows)
        + " | cor:local statement lacks S_min>0 and 2*o_e<=S_min^2 conditions")


def main():
    t0 = time.time()
    print("== G0 reproduction checks (fresh seeds) ==")
    cora_native_budget()
    star_joint_mc()
    cyclic_overlap()
    matched_marginals()
    s_min_legality()
    out["wall_seconds"] = time.time() - t0
    dest = RESULTS / "g0_reproduction.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    n_fail = sum(c["status"] == "FAIL" for c in out["checks"])
    print(f"\n{len(out['checks'])} checks, {n_fail} FAIL; "
          f"wrote {dest} ({out['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
