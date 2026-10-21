"""S4: normalized first-order influence, residual, coherent/flat survival.

Theory task T1 pilot (plan 5.T1, 6.4). For each structure under a known L1
binary node law, the conditional influence satisfies (binary, exact)

    F_u(0) = -P[u,1] * Delta_u,   F_u(1) = +P[u,0] * Delta_u,
    Delta_u = E[S | Y_u=1] - E[S | Y_u=0],
    v_u = E||F_u(Y_u)||_F^2 = P[u,0] P[u,1] ||Delta_u||_F^2

(the matrix form of the manuscript's V1(x) = sum P0 P1 Phi_u(x)^2; the
difference cancels any shared center error, so NO demeaning is needed and
the bias correction is per-cell unbiased:
E||Deltahat||^2 = ||Delta||^2 + trSig1/N1 + trSig0/N0).

X = S - E S ;  R^S = X - sum_u F_u^S(Y_u); exact at small scale (S1 check3).
Full-node decomposition on controlled families; top-impact + random SUBSETS
on native graphs (plan 6.4: subset energies are never called total energies).

Spectral level per draw: ||sum F_u(Y_u)||_2 vs ||X||_2 (coherent/flat
survival under dynamic normalization) and the raw-B counterparts.

Writes results/influence_residual.csv + influence_residual_summary.json.
Guarantee label: grade 4 with grade-2 bound logic; native subsets are
diagnostic, not total energy.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

import hyperops as ho  # noqa: E402
import families as fam  # noqa: E402
import data_io  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def selected_nodes(op, degrees, n_sel, rng):
    n = op.n
    hub_cut = float(np.quantile(degrees, 0.90))
    hub = np.flatnonzero(degrees >= hub_cut)
    hub = hub[np.argsort(-degrees[hub])]
    top = hub[:n_sel]
    rest = np.setdiff1d(np.arange(n), top)
    rnd = rng.choice(rest, size=min(n_sel, len(rest)), replace=False)
    sel = np.concatenate([top, rnd]).astype(int)
    role = np.array(["hub"] * len(top) + ["random"] * len(rnd))
    return sel, role


def accumulate(op, P, sel, n_batch, seed):
    """One node-driven batch: conditional sums of RAW S (and raw B)."""
    rng = np.random.default_rng(seed)
    sampler = ho.bound_node_sampler(op, P)
    k = len(sel)
    accS = np.zeros((k, 2, op.n_entries))
    accSq = np.zeros((k, 2))
    cnt = np.zeros((k, 2), dtype=np.int64)
    accB = np.zeros((k, 2, op.n_entries))
    accBSq = np.zeros((k, 2))
    s_sum = np.zeros(op.n_entries)
    x2_raw = 0.0
    B_sum = np.zeros(op.n_entries)
    for b in range(n_batch):
        a, Y = sampler(rng)
        s = op.S_data(a)
        bdat = op.raw_b_data(a)
        s_sum += s
        B_sum += bdat
        for j in range(k):
            z = int(Y[sel[j]])
            accS[j, z] += s
            accSq[j, z] += float(s @ s)
            accB[j, z] += bdat
            accBSq[j, z] += float(bdat @ bdat)
            cnt[j, z] += 1
        x2_raw += float(s @ s)
    CHat = s_sum / n_batch
    EBhat = B_sum / n_batch
    return (accS, accSq, accB, accBSq, cnt, CHat, EBhat, x2_raw, rng)


def influence_from_delta(accS, accSq, cnt, P, sel):
    """F_u(z) from unbiased corrected ||Delta_u||^2; returns F matrices, v, noise."""
    k = accS.shape[0]
    F = np.zeros_like(accS)
    v = np.zeros(k)
    noise_floor = np.zeros(k)
    for j in range(k):
        N0, N1 = int(cnt[j, 0]), int(cnt[j, 1])
        if N0 < 2 or N1 < 2:
            continue
        mu0 = accS[j, 0] / N0
        mu1 = accS[j, 1] / N1
        delM = mu1 - mu0
        d_sq = float(delM @ delM)
        tr0 = (accSq[j, 0] - N0 * float(mu0 @ mu0)) / (N0 - 1)
        tr1 = (accSq[j, 1] - N1 * float(mu1 @ mu1)) / (N1 - 1)
        noise = tr0 / N0 + tr1 / N1
        noise_floor[j] = P[sel[j], 0] * P[sel[j], 1] * noise
        d_sq_c = d_sq - noise
        p0, p1 = P[sel[j], 0], P[sel[j], 1]
        F[j, 0] = -p1 * delM
        F[j, 1] = p0 * delM
        v[j] = p0 * p1 * d_sq_c
    return F, v, noise_floor


def analyze(name, profile_label, op, P, degrees, seed,
            n_batch, k_sel, full, n_spectral):
    t0 = time.time()
    if full:
        sel, role = np.arange(op.n), np.array(["all"] * op.n)
    else:
        sel, role = selected_nodes(op, degrees, k_sel,
                                   np.random.default_rng(seed + 2))
    (accS, accSq, accB, accBSq, cnt, CHat, EBhat, x2_raw, _) = accumulate(
        op, P, sel, n_batch, seed + 3)
    k = len(sel)
    X2_frob = x2_raw / n_batch - float(CHat @ CHat)  # E||S||^2-||ES||^2
    # exact X2_Frob of B via EB analytic
    EB = op.EB_data(op.edge_marginals(P))
    # E||B-EB||^2: second moment via batch of raw data
    # recompute quickly (cheap) using stored batch? we recompute moments:
    rng = np.random.default_rng(seed + 3)
    sampler = ho.bound_node_sampler(op, P)
    B2 = 0.0
    for b in range(n_batch):
        a, Y = sampler(rng)
        d = op.raw_b_data(a) - EB
        B2 += float(d @ d)
    B2 /= n_batch

    F, vS, floorS = influence_from_delta(accS, accSq, cnt, P, sel)
    FB, vBarm, floorB = influence_from_delta(accB, accBSq, cnt, P, sel)

    # realized first-order sum (Frobenius) over a fresh shared batch
    rng2 = np.random.default_rng(seed + 4)
    realized_S = realized_B = 0.0
    Ys = np.zeros((min(n_batch, 2000), op.n), dtype=np.int8)
    nb2 = len(Ys)
    for b in range(nb2):
        a, Y = sampler(rng2)
        Ys[b] = Y
        tot = np.zeros(op.n_entries)
        totB = np.zeros(op.n_entries)
        for j in range(k):
            z = int(Y[sel[j]])
            tot += F[j, z]
            totB += FB[j, z]
        realized_S += float(tot @ tot)
        realized_B += float(totB @ totB)
    realized_S /= nb2
    realized_B /= nb2

    # spectral level
    rng3 = np.random.default_rng(seed + 5)
    sp = {"sumF": [], "X": [], "sumFB": [], "XB": []}
    zero = np.zeros(op.n_entries)
    for b in range(n_spectral):
        a, Y = sampler(rng3)
        tot = np.zeros(op.n_entries)
        totB = np.zeros(op.n_entries)
        for j in range(k):
            z = int(Y[sel[j]])
            tot += F[j, z]
            totB += FB[j, z]
        sp["sumF"].append(ho.spectral_norm_entries(op, tot, zero)[0])
        sp["sumFB"].append(ho.spectral_norm_entries(op, totB, zero)[0])
        sp["X"].append(ho.spectral_norm_entries(op, op.S_data(a), CHat)[0])
        sp["XB"].append(ho.spectral_norm_entries(op, op.raw_b_data(a), EB)[0])

    resid_maxabs = cross_maxabs = math.nan
    if full:
        rng4 = np.random.default_rng(seed + 6)
        accR = np.zeros((op.n, 2, op.n_entries))
        cntR = np.zeros((op.n, 2), dtype=np.int64)
        nr = nb2
        cross = np.zeros((min(20, k), min(20, k)))
        for b in range(nr):
            Y = Ys[b]
            a = op.flags_from_Y(Y)
            d = op.S_data(a) - CHat
            tot = np.zeros(op.n_entries)
            for j in range(k):
                tot += F[j, int(Y[sel[j]])]
            R = d - tot
            for j in range(k):
                z = int(Y[sel[j]])
                accR[j, z] += R
                cntR[j, z] += 1
            if b < 400:
                idx = np.arange(min(20, k))
                fv = np.stack([F[i, int(Y[sel[i]])] for i in idx])
                cross += fv @ fv.T
        ok = (cntR >= 50)[:, :, None]
        resid_maxabs = float(np.max(np.abs(np.where(
            ok, accR / np.maximum(cntR, 1)[:, :, None], 0.0))))
        cross /= 400
        np.fill_diagonal(cross, 0.0)
        cross_maxabs = float(np.max(np.abs(cross)))

    rho = spearmanr(np.maximum(vS, 0), degrees[sel]).statistic if k > 2 \
        else math.nan
    return {
        "structure": name, "profile": profile_label, "full_node": bool(full),
        "n": op.n, "k_sel": k, "n_batch": n_batch, "n_spectral": n_spectral,
        "secs": round(time.time() - t0, 1),
        "X2_frob_S": X2_frob, "X2_frob_B": B2,
        "sum_vS": float(vS.sum()) if full else math.nan,
        "sum_vB": float(vBarm.sum()) if full else math.nan,
        "realized_sum_frob_S": realized_S,
        "realized_sum_frob_B": realized_B,
        "resid_cond_maxabs": resid_maxabs,
        "cross_frob_maxabs": cross_maxabs,
        "spearman_v_degree": float(rho),
        "E_spec_sumF": float(np.mean(sp["sumF"])),
        "E_spec_X": float(np.mean(sp["X"])),
        "E_spec_sumFB": float(np.mean(sp["sumFB"])),
        "E_spec_XB": float(np.mean(sp["XB"])),
        "rows": [{"node": int(sel[j]), "role": str(role[j]),
                  "degree": int(degrees[sel[j]]),
                  "v_u_S": float(vS[j]), "v_u_B": float(vBarm[j]),
                  "noise_floor_S": float(floorS[j])}
                 for j in range(k)],
    }


def main():
    out = []
    eA, PA, _ = fam.build_family_A(t=2, d=120, pool=5, seed=11)
    degA = np.zeros(PA.shape[0])
    for e in eA:
        for u in e:
            degA[u] += 1
    opA = ho.RelationOperator(PA.shape[0], eA)
    out.append(analyze("familyA_coherent", "fixed", opA, PA, degA, 31,
                       n_batch=3000, k_sel=0, full=True, n_spectral=300))
    for g, gl in ((fam.EPS_WEAK, "gamma=0.05_cancel"),
                  (1 - fam.EPS_WEAK, "gamma=0.95_max")):
        eB, PB, _ = fam.build_family_B(d=120, gamma=g, seed=12)
        degB = np.zeros(PB.shape[0])
        for e in eB:
            for u in e:
                degB[u] += 1
        opB = ho.RelationOperator(PB.shape[0], eB)
        out.append(analyze("familyB_flat", gl, opB, PB, degB, 32,
                           n_batch=3000, k_sel=0, full=True, n_spectral=300))

    cora_n, cora_edges, cora_labels = data_io.load_graph("cora-cc")
    op_c = ho.RelationOperator(cora_n, cora_edges)
    deg_c = np.zeros(cora_n)
    for e in cora_edges:
        for u in e:
            deg_c[u] += 1
    Pc = data_io.majority_binary_profile(cora_labels, 0.05)
    out.append(analyze("cora-cc", "binary-eps0.05", op_c, Pc, deg_c, 33,
                       n_batch=1500, k_sel=40, full=False, n_spectral=100))

    dblp_n, dblp_edges, dblp_labels = data_io.load_graph("dblp-ca")
    op_d = ho.RelationOperator(dblp_n, dblp_edges)
    deg_d = np.zeros(dblp_n)
    for e in dblp_edges:
        for u in e:
            deg_d[u] += 1
    Pd = data_io.majority_binary_profile(dblp_labels, 0.05)
    out.append(analyze("dblp-ca", "binary-eps0.05", op_d, Pd, deg_d, 34,
                       n_batch=600, k_sel=6, full=False, n_spectral=30))

    node_rows = []
    for r in out:
        tot = max(sum(max(x["v_u_S"], 0) for x in r["rows"]), 1e-18)
        for rr in r["rows"]:
            node_rows.append({
                "structure": r["structure"], "profile": r["profile"],
                "role": rr["role"], "node": rr["node"], "degree": rr["degree"],
                "v_u_S": rr["v_u_S"], "v_u_B": rr["v_u_B"],
                "noise_floor_S": rr["noise_floor_S"],
                "share_of_selected_S": max(rr["v_u_S"], 0) / tot,
            })
    with open(RESULTS / "influence_residual.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(node_rows[0].keys()))
        w.writeheader()
        w.writerows(node_rows)
    for r in out:
        r.pop("rows")
    with open(RESULTS / "influence_residual_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
