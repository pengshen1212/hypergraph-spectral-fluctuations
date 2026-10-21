"""S9b: influence-direction standardized detection (review round 2, task 2).

Statistic (directions, scales, counts ALL frozen in a reference stage):

    t_j(b) = v_j^T S^(b) v_j - v_j^T M_hat v_j
    T_g(b) = max_{j in g} |t_j(b)| / (sigma_j + eps)

groups: inf (principal eigen-directions of node conditional influences of
top-degree nodes), var (extreme eigen-directions of reference S - M_hat,
the normal-mechanism fluctuation directions), rand (Gaussian controls),
plus norm = ||S - M_hat||_2 (current statistic).

Protocol identical to S3 (reference/calibration/evaluation separated;
truth = node law). Calibration achieves nominal alpha for EVERY statistic;
power is then compared on frozen thresholds under the same M1/M2 mechanisms
as S3. If inf > rand and > var at matched achieved false-alarm, the
dependence structure is doing detection work.

Writes results/directional_detection_v2.csv. The repaired reference
directions are not mixed with the historical CSV used by the gate report.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

import hyperops as ho  # noqa: E402
import data_io  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
K_DIR = 8
ALPHA = 0.05
EPS_SIG = 1e-6


def _principal_pair(op, data):
    """Return up to 2 eigen-directions (|lambda| largest, then opposite end)."""
    out = []
    if op.n <= 600:
        vals, vecs = np.linalg.eigh(op.csr(data).toarray())
        order = np.argsort(-np.abs(vals))
        for i in order[:2]:
            out.append(np.asarray(vecs[:, i]))
        return out
    for which in ("LA", "SA"):
        try:
            start = np.random.default_rng(20261021).standard_normal(op.n)
            v, w = eigsh(op.csr(data), k=1, which=which, maxiter=800, tol=1e-7,
                         v0=start)
            out.append(np.asarray(w[:, 0]).ravel())
        except Exception:
            pass
    return out


def influence_directions(op, P, degrees, seed, n_batch):
    rng = np.random.default_rng(seed)
    sel = np.argsort(-degrees)[:K_DIR]
    sampler = ho.bound_node_sampler(op, P)
    acc = np.zeros((len(sel), 2, op.n_entries))
    cnt = np.zeros((len(sel), 2), dtype=np.int64)
    for _ in range(n_batch):
        a, Y = sampler(rng)
        s = op.S_data(a)
        for j, u in enumerate(sel):
            z = int(Y[u])
            acc[j, z] += s
            cnt[j, z] += 1
    dirs = []
    for j in range(len(sel)):
        if cnt[j, 0] < 20 or cnt[j, 1] < 20:
            continue
        delta = acc[j, 1] / cnt[j, 1] - acc[j, 0] / cnt[j, 0]
        got = _principal_pair(op, delta)
        if got:
            dirs.append(got[0])
        if len(dirs) >= K_DIR:
            break
    dirs = [v / np.linalg.norm(v) for v in dirs[:K_DIR]]
    return dirs, sel


def fluctuation_directions(op, P, M_hat, seed, n_batch=30):
    rng = np.random.default_rng(seed)
    sampler = ho.bound_node_sampler(op, P)
    dirs = []
    seen = 0
    while len(dirs) < K_DIR and seen < n_batch:
        a, _ = sampler(rng)
        seen += 1
        for v in _principal_pair(op, op.S_data(a) - M_hat):
            dirs.append(v / np.linalg.norm(v))
            if len(dirs) >= K_DIR:
                break
    return dirs[:K_DIR]


def random_directions(op, seed):
    rng = np.random.default_rng(seed)
    vectors = [rng.standard_normal(op.n) for _ in range(K_DIR)]
    return [vector / np.linalg.norm(vector) for vector in vectors]


def dir_tvals(op, groups, M_hat, sampler, n, rng):
    Mc = op.csr(M_hat)
    const = {g: np.array([float(v @ (Mc @ v)) for v in vs])
             for g, vs in groups.items()}
    out = {g: np.empty((n, len(vs))) for g, vs in groups.items()}
    for b in range(n):
        a, _ = sampler(rng)
        Sc = op.csr(op.S_data(a))
        for g, vs in groups.items():
            out[g][b] = np.array([float(v @ (Sc @ v)) for v in vs]) - const[g]
    return out


def standardized(T_raw, sigma):
    return np.max(np.abs(T_raw) / (sigma + EPS_SIG), axis=1)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", default="cora", choices=["cora", "dblp"])
    args = ap.parse_args()
    tag = {"cora": "cora-cc", "dblp": "dblp-ca"}[args.unit]

    n, edges, labels = data_io.load_graph(tag)
    op = ho.RelationOperator(n, edges)
    P = data_io.majority_binary_profile(labels, 0.05)
    deg = np.zeros(n)
    for e in edges:
        for u in e:
            deg[u] += 1
    sampler = ho.bound_node_sampler(op, P)

    if args.unit == "cora":
        N_SIG, N_CAL, N_EVAL = 400, 800, 1200
        N_ALT = 600
    else:
        N_SIG, N_CAL, N_EVAL = 150, 400, 500
        N_ALT = 250

    t0 = time.time()
    # ---- reference stage: center, directions, sigma (all frozen) ----
    M_hat = ho.estimate_center(op, sampler, N_SIG, np.random.default_rng(11001))
    d_inf, sel = influence_directions(op, P, deg, 2201, 800)
    groups = {"inf": d_inf,
              "var": fluctuation_directions(op, P, M_hat, 2202),
              "rand": random_directions(op, 2203)}
    T0 = dir_tvals(op, groups, M_hat, sampler, min(N_SIG, 300),
                   np.random.default_rng(11002))
    sigma = {g: np.std(T0[g], axis=0, ddof=1) for g in groups}
    print(f"ref stage {time.time()-t0:.0f}s dirs: "
          f"inf={len(groups['inf'])} var={len(groups['var'])} "
          f"rand={len(groups['rand'])}", flush=True)

    # ---- calibration ----
    Tc = dir_tvals(op, groups, M_hat, sampler, N_CAL,
                   np.random.default_rng(11003))
    cal = {g: standardized(Tc[g], sigma[g]) for g in groups}
    cal["norm"], _ = ho.draw_scores(op, sampler, N_CAL,
                                    np.random.default_rng(11004), M_hat)
    q = {g: ho.rank_threshold(cal[g], ALPHA) for g in cal}

    # ---- evaluation: normal mechanism (achieved FPR) ----
    Te = dir_tvals(op, groups, M_hat, sampler, N_EVAL,
                   np.random.default_rng(11005))
    ev = {g: standardized(Te[g], sigma[g]) for g in groups}
    ev["norm"], _ = ho.draw_scores(op, sampler, N_EVAL,
                                   np.random.default_rng(11006), M_hat)

    # ---- alternative mechanisms (same as S3, binary coding) ----
    eps_vec = np.full(n, 0.05)
    alts = []
    for f in (1.5, 2.0, 3.0):
        alts.append((f"M1_x{f}", data_io.majority_binary_profile(
            labels, 0.05 * f)))
    hub_cut = float(np.quantile(deg, 0.90))
    hub = deg >= hub_cut
    e2 = np.where(hub, np.minimum(eps_vec * 4, 0.49), eps_vec)
    alts.append(("M2_hubs_x4", s9b_binary_from_eps(labels, e2)))
    order = np.argsort(deg)
    low = np.zeros(n, dtype=bool)
    picked = [u for u in order if not hub[u]][: int(hub.sum())]
    low[picked] = True
    e3 = np.where(low, np.minimum(eps_vec * 4, 0.49), eps_vec)
    alts.append(("M2_lowdeg_x4", s9b_binary_from_eps(labels, e3)))

    rows = []
    for alab, P_alt in [("", P)] + alts:
        s_alt = ho.bound_node_sampler(op, P_alt)
        N = N_EVAL if alab == "" else N_ALT
        Ta = dir_tvals(op, groups, M_hat, s_alt, N,
                       np.random.default_rng(11007 + len(rows)))
        stat = {g: standardized(Ta[g], sigma[g]) for g in groups}
        stat["norm"], _ = ho.draw_scores(op, s_alt, N,
                                         np.random.default_rng(11008 +
                                                              len(rows)),
                                         M_hat)
        for g in ("norm", "inf", "var", "rand"):
            ex = float(np.mean(stat[g] > q[g]))
            lo, hi = ho.clopper_pearson(int(round(ex * N)), N)
            rows.append({"structure": tag, "profile": "binary-eps0.05",
                         "mechanism": alab or "normal", "statistic": g,
                         "alpha": ALPHA, "q": q[g], "rate": ex,
                         "cp_lo": lo, "cp_hi": hi, "n": N,
                         "n_dirs": len(groups.get(g, [])),
                         "secs": round(time.time() - t0, 1)})
            print(f"{alab or 'normal'} {g}: {ex:.3f} "
                  f"[{lo:.3f},{hi:.3f}]", flush=True)

    outp = RESULTS / "directional_detection_v2.csv"
    exists = outp.exists()
    with open(outp, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(rows)
    print(f"S9b {tag} done in {time.time()-t0:.0f}s")


def s9b_binary_from_eps(labels, eps_vec):
    labels = np.asarray(labels)
    classes, inverse, counts = np.unique(labels, return_inverse=True,
                                         return_counts=True)
    maj = int(np.argmax(counts))
    P = np.empty((len(labels), 2))
    is_maj = inverse == maj
    P[:, 1] = np.where(is_maj, 1.0 - eps_vec, eps_vec)
    P[:, 0] = 1.0 - P[:, 1]
    return P


if __name__ == "__main__":
    main()
