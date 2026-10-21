"""S5-b: why the V_ES point estimate is not a certificate; heavy-tail
diagnosis and a truncated empirical-Bernstein prototype (T2b groundwork).

Z = ||S(Y) - S(Y^(u))||_F^2 / (2 p_u), p_u = structural degree share, is
unbiased for V_ES (plan 6.3) but HEAVY-TAILED: low-probability replacement
nodes carry weight 1/(2 p_u). This script:

  1. measures the tail: q99/q999/max, fraction of E Z explained by Z>t;
  2. computes the truncated empirical-Bernstein one-sided bound
     t-mean + sqrt(2 hat_var ln(2/delta)/n) + 7 b ln(2/delta)/(3(n-1)),
     reporting the bound AND the dropped tail mass E[Z 1{Z>t}]/E Z
     (truncation bias) at several thresholds t;
  3. gives the smallest t where (EB bound + tail estimate) exceeds the true
     X2_Frob by a controlled margin - the prototype UCB.

Guarantee label: this is NOT yet a certificate. The dropped-tail term is
estimated by sample proportion, not bounded; converting it into a valid UCB
(Markov-on-truncated or Catoni under a variance bound) is open task T2b.

Writes results/es_ucb.json.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

import hyperops as ho  # noqa: E402
import families as fam  # noqa: E402
import data_io  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

DELTA = 0.05


def sample_z(op, P, n, seed):
    rng = np.random.default_rng(seed)
    sampler = ho.bound_node_sampler(op, P)
    struct = np.asarray(op.incidence @ np.ones(op.m)).ravel() + 1e-12
    p_u = struct / struct.sum()
    zs = np.empty(n)
    x2 = 0.0
    CHat = None
    acc = np.zeros(op.n_entries)
    for i in range(n):
        a, Y = sampler(rng)
        s = op.S_data(a)
        acc += s
        u = int(rng.choice(op.n, p=p_u))
        P_u = P[u]
        z_new = int((rng.random() >= np.cumsum(P_u)).sum())
        if z_new == int(Y[u]):
            zs[i] = 0.0
            continue
        Yp = Y.copy()
        Yp[u] = z_new
        dd = s - op.S_data(op.flags_from_Y(Yp))
        zs[i] = float(dd @ dd) / (2.0 * p_u[u])
    CHat = acc / n
    # X2 via a second cheap pass with the frozen center
    x2s = []
    for i in range(n):
        a, _ = sampler(rng)
        d = op.S_data(a) - CHat
        x2s.append(float(d @ d))
    return zs, float(np.mean(x2s)), CHat


def eb_onesided(z_trunc, b, delta):
    n = len(z_trunc)
    mean = z_trunc.mean()
    var = z_trunc.var(ddof=1)
    ln = math.log(2.0 / delta)
    return mean + math.sqrt(2 * var * ln / n) + 7 * b * ln / (3 * (n - 1))


def analyze(name, op, P, n=4000, seed=4242):
    t0 = time.time()
    zs, x2, _ = sample_z(op, P, n, seed)
    mean_z = zs.mean()
    q = np.quantile(zs, [0.99, 0.999])
    res = {"V_hat": float(mean_z), "X2_frob": x2,
           "q99": float(q[0]), "q999": float(q[1]), "max": float(zs.max()),
           "thresholds": []}
    for t in (np.quantile(zs, 0.95), np.quantile(zs, 0.99),
              np.quantile(zs, 0.999), np.inf):
        tt = float(t) if np.isfinite(t) else float(zs.max())
        mask = zs <= t
        kept = zs[mask]
        tail_frac = 1.0 - mask.mean()
        tail_mass_est = float(zs[~mask].sum()) / n if (~mask).any() else 0.0
        bound = eb_onesided(kept, tt, DELTA) if len(kept) > 2 else math.nan
        total = bound + tail_mass_est
        res["thresholds"].append({
            "t": tt, "kept_frac": float(mask.mean()),
            "tail_mass_E": tail_mass_est,
            "tail_share_of_mean": tail_mass_est / max(mean_z, 1e-18),
            "eb_bound_trunc": bound, "total_ucb_prototype": total,
            "covers_x2": bool(total >= x2),
        })
    res["secs"] = round(time.time() - t0, 1)
    return res


def main():
    out = {}
    eA, PA, _ = fam.build_family_A(t=2, d=120, pool=5, seed=11)
    op = ho.RelationOperator(PA.shape[0], eA)
    out["familyA"] = analyze("familyA", op, PA, n=4000)
    print("familyA done", flush=True)
    cora_n, cora_edges, cora_labels = data_io.load_graph("cora-cc")
    op_c = ho.RelationOperator(cora_n, cora_edges)
    Pc = data_io.majority_binary_profile(cora_labels, 0.05)
    out["cora-cc"] = analyze("cora", op_c, Pc, n=2000)
    print("cora done", flush=True)
    dblp_n, dblp_edges, dblp_labels = data_io.load_graph("dblp-ca")
    op_d = ho.RelationOperator(dblp_n, dblp_edges)
    Pd = data_io.majority_binary_profile(dblp_labels, 0.05)
    out["dblp-ca"] = analyze("dblp", op_d, Pd, n=600)
    print("dblp done", flush=True)
    with open(RESULTS / "es_ucb.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
