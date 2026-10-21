"""S2: three-operator normalization pilot (plan sections 3.2/3.3, stage A1/A2;
upgrade-plan item 19.1).

For each structure x error-law profile x arm (node-driven A / iid-edge B):

  operator raw    B    : fluctuation ||B(a) - E B||_2      (E B exact)
  operator S0          : fluctuation ||S0(a) - E S0||_2    (exact: linear mean)
  operator S (dynamic) : fluctuation ||S(a) - ES_arm||_2   (center = independent
                         MC reference batch of the SAME arm; center error noted)

Between-arm quantities (per structure x profile):
  mean_shift   = ||E S_A - E S_B||_2          (MC centers, independent batches)
  norm_ES_A/B, norm_ES0
  ratio_*      = mean_fluctuation_A / mean_fluctuation_B per operator
  shift_scale  = mean_shift / mean fluctuation of the smaller arm

Writes results/normalization_pilot.csv and results/normalization_pilot_summary.json.
Guarantee label: fitted-model / specified-model Monte Carlo estimate (T6 grade 4
for native graphs under known L1 law = grade 1-4 boundary; centers are finite
sample -> flagged).
"""

from __future__ import annotations

import csv
import json
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

ALPHAS_UNUSED = None  # thresholds are S3; here only distribution summaries


def fluct_stats(op, sampler, count, rng, centers):
    """Mean/sd/q95/means of ||op(a)-center|| for operators raw,S0,S."""
    di0 = centers["di0"]
    vals = {k: np.empty(count) for k in ("raw", "S0", "S")}
    resids = {k: np.empty(count) for k in ("raw", "S0", "S")}
    for b in range(count):
        a, _ = sampler(rng)
        vals["raw"][b], resids["raw"][b] = ho.spectral_norm_entries(
            op, op.raw_b_data(a), centers["EB"])
        vals["S0"][b], resids["S0"][b] = ho.spectral_norm_entries(
            op, op.S0_data(a, di0), centers["ES0"])
        vals["S"][b], resids["S"][b] = ho.spectral_norm_entries(
            op, op.S_data(a), centers["S"])
    out = {}
    for k in vals:
        out[f"mean_{k}"] = float(np.mean(vals[k]))
        out[f"sd_{k}"] = float(np.std(vals[k], ddof=1))
        out[f"q95_{k}"] = float(np.quantile(vals[k], 0.95))
        out[f"approxfrac_{k}"] = float(np.mean(resids[k] > 1e-6))
    return out, vals, resids


def run_structure(name, profile_label, op, P, budget, seed_base):
    t0 = time.time()
    p = op.edge_marginals(P)
    node_s = ho.bound_node_sampler(op, P)
    edge_s = ho.bound_edge_sampler(op, p)
    n_center, n_fluct = budget

    rng = np.random.default_rng
    ES_A = ho.estimate_center(op, node_s, n_center, rng(seed_base + 1))
    ES_B = ho.estimate_center(op, edge_s, n_center, rng(seed_base + 2))
    di0 = op.S0_inverse_degrees(p)
    centers_common = {
        "EB": op.EB_data(p),
        "ES0": op.S0_data(p, di0),
        "di0": di0,
    }
    norm_ES_A = ho.spectral_norm_matrix(op, ES_A)[0]
    norm_ES_B = ho.spectral_norm_matrix(op, ES_B)[0]
    norm_ES0 = ho.spectral_norm_matrix(op, centers_common["ES0"])[0]
    mean_shift = ho.spectral_norm_entries(op, ES_A, ES_B)[0]

    rows = {}
    for arm, sampler, center in (("node", node_s, ES_A), ("edge", edge_s, ES_B)):
        cc = dict(centers_common)
        cc["S"] = center
        st, _, _ = fluct_stats(op, sampler, n_fluct, rng(seed_base + 3 +
                                 (0 if arm == "node" else 1)), cc)
        rows[arm] = st

    rec = {
        "structure": name, "profile": profile_label,
        "n": op.n, "m": op.m, "n_center": n_center, "n_fluct": n_fluct,
        "mean_shift_dyn": mean_shift,
        "norm_ES_A": norm_ES_A, "norm_ES_B": norm_ES_B, "norm_ES0": norm_ES0,
        "secs": round(time.time() - t0, 2),
    }
    for arm in ("node", "edge"):
        for k, v in rows[arm].items():
            rec[f"{arm}_{k}"] = v
    for k in ("raw", "S0", "S"):
        rec[f"ratio_{k}"] = rec[f"node_mean_{k}"] / max(
            rec[f"edge_mean_{k}"], 1e-12)
    smaller = min(rec["node_mean_S"], rec["edge_mean_S"])
    rec["shift_scale"] = mean_shift / max(smaller, 1e-12)
    return rec


def timing_probe(op, P, draws=8, seed=99):
    sampler = ho.bound_node_sampler(op, P)
    center = ho.estimate_center(op, sampler, 8, np.random.default_rng(seed))
    t = time.time()
    rng = np.random.default_rng(seed + 1)
    for _ in range(draws):
        a, _ = sampler(rng)
        ho.spectral_norm_entries(op, op.S_data(a), center)
    return (time.time() - t) / draws


def main():
    structures = []
    # controlled coherent / flat families (stage A1)
    eA, PA, _ = fam.build_family_A(t=2, d=120, pool=5, seed=11)
    structures.append(("familyA_coherent", "fixed", eA, PA, PA.shape[0]))
    for g, glabel in ((fam.EPS_WEAK, "gamma=0.05_cancel"),
                      (0.5, "gamma=0.5"),
                      (1 - fam.EPS_WEAK, "gamma=0.95_max")):
        eB, PB, _ = fam.build_family_B(d=120, gamma=g, seed=12)
        structures.append(("familyB_flat", glabel, eB, PB, PB.shape[0]))

    cora_n, cora_edges, cora_labels = data_io.load_graph("cora-cc")
    dblp_n, dblp_edges, dblp_labels = data_io.load_graph("dblp-ca")

    def cora_profile(label):
        if label == "binary-eps0.05":
            return data_io.majority_binary_profile(cora_labels, 0.05)
        if label == "binary-eps0.15":
            return data_io.majority_binary_profile(cora_labels, 0.15)
        if label == "multiclass-eps0.05":
            return data_io.multiclass_profile(cora_labels, 0.05)
        if label == "multiclass-eps0.15":
            return data_io.multiclass_profile(cora_labels, 0.15)
        if label == "hetero-0.02/0.30":
            deg = np.zeros(cora_n)
            for e in cora_edges:
                for u in e:
                    deg[u] += 1
            return data_io.hetero_binary_profile(cora_labels, deg, 0.02, 0.30)
        raise ValueError(label)

    cora_op = ho.RelationOperator(cora_n, cora_edges)
    for label in ("binary-eps0.05", "binary-eps0.15", "multiclass-eps0.05",
                  "multiclass-eps0.15", "hetero-0.02/0.30"):
        structures.append(("cora-cc", label, cora_edges, cora_profile(label),
                           cora_n))

    dblp_op = ho.RelationOperator(dblp_n, dblp_edges)
    per_draw = timing_probe(dblp_op, data_io.majority_binary_profile(
        dblp_labels, 0.05))
    print(f"dblp spectral-norm per draw: {per_draw:.3f}s", flush=True)
    if per_draw < 0.35:
        dblp_budget = (400, 300)
    elif per_draw < 1.0:
        dblp_budget = (200, 150)
    else:
        dblp_budget = (120, 80)

    rows = []

    def flush():
        if not rows:
            return
        fields = list(rows[0].keys())
        with open(RESULTS / "normalization_pilot.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    seed_base = 100
    for name, label, edges, P, n in structures:
        op = ho.RelationOperator(n, edges)
        if name == "dblp-ca":
            budget = dblp_budget
        else:
            budget = (800, 600)
        print(f"running {name} / {label} (n={n})", flush=True)
        rows.append(run_structure(name, label, op, P, budget, seed_base))
        flush()
        seed_base += 10
    for label in ("binary-eps0.05", "binary-eps0.15", "hetero-0.02/0.30"):
        op = dblp_op
        if label.startswith("binary"):
            eps = float(label.split("eps")[-1])
            P = data_io.majority_binary_profile(dblp_labels, eps)
        else:
            deg = np.zeros(dblp_n)
            for e in dblp_edges:
                for u in e:
                    deg[u] += 1
            P = data_io.hetero_binary_profile(dblp_labels, deg, 0.02, 0.30)
        print(f"running dblp-ca / {label}", flush=True)
        rows.append(run_structure("dblp-ca", label, op, P, dblp_budget,
                                  seed_base))
        flush()
        seed_base += 10

    summary = {
        "dblp_per_draw_secs": per_draw,
        "dblp_budget": list(dblp_budget),
        "rows": rows,
        "reading": {
            "coherent_survival": "ratio_S(familyA) vs ratio_raw(familyA): "
                                 "share of amplification kept after dynamic norm",
            "flat_survival": "ratio_* at gamma cancel/max on familyB",
            "mean_vs_fluct": "shift_scale >> 1 would mean differences are "
                             "explained by mean shift (Gate A stop signal)",
        },
    }
    with open(RESULTS / "normalization_pilot_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("S2 done ->", RESULTS / "normalization_pilot.csv")


if __name__ == "__main__":
    main()
