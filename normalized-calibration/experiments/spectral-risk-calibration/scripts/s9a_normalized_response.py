"""S9a: decomposition of the normalization cancellation (review round 2, task 1).

Around the reference center p (= expected flags), with base S0:

    S(a) - S0  =  J_A(E)  +  resid,      E = B(a) - B(p)
    J_A(E)     =  R1 + Rdeg
    R1         =  D0^{-1/2} E D0^{-1/2}          (relation change)
    Rdeg       = -(C S0 + S0 C)/2,  C = D0^{-1} diag(E 1)   (degree counter)

For every paired draw under BOTH arms (node-driven A, iid-edge B):

  - spectral norms of R1, Rdeg, J, true, resid (eigsh/dense pipeline,
    deterministic v0);
  - Frobenius inner products <R1, Rdeg> and <J, R1> (cancellation sign/size);
  - arm ratios: ratio_R1 (should mirror raw amplification),
    ratio_J (first-order prediction of the S amplification),
    ratio_true (common-base centered fluctuation, i.e. the shared-frozen-
    center comparison demanded in the review, separate from the per-arm
    centering used in S2/S3).

Outputs results/normalization_response.csv + summary json with the
"predict cancellation from structure" test:
    ratio_J ~ ratio_true within MC error?  (first-order explains reversal)
    resid/true fraction (second-order + kappa-fixed remainder, NOT a bug)

Guarantee label: grade 1/2 linearization with grade-4 MC measurement.
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


def unit(name, profile, op, P, n_draws, seed):
    t0 = time.time()
    p = op.edge_marginals(P)
    basis = op.response_basis(p)
    node_s = ho.bound_node_sampler(op, P)
    edge_s = ho.bound_edge_sampler(op, p)
    zero = np.zeros(op.n_entries)
    rows = {}
    per_arm = {}
    for arm, sampler in (("node", node_s), ("edge", edge_s)):
        rng = np.random.default_rng(seed + (0 if arm == "node" else 1))
        m = dict(R1=[], Rdeg=[], J=[], true=[], resid=[])
        ip_RD, ip_JR1 = [], []
        for b in range(n_draws):
            a, _ = sampler(rng)
            r1, rdeg, j, true, resid = op.linear_response(a, basis)
            m["R1"].append(ho.spectral_norm_entries(op, r1, zero)[0])
            m["Rdeg"].append(ho.spectral_norm_entries(op, rdeg, zero)[0])
            m["J"].append(ho.spectral_norm_entries(op, j, zero)[0])
            m["true"].append(ho.spectral_norm_entries(op, true, zero)[0])
            m["resid"].append(ho.spectral_norm_entries(op, resid, zero)[0])
            ip_RD.append(float(r1 @ rdeg))
            ip_JR1.append(float(j @ r1))
        per_arm[arm] = {k: (float(np.mean(v)), float(np.median(v)))
                        for k, v in m.items()}
        per_arm[arm + "_ipRD"] = (float(np.mean(ip_RD)), float(np.std(ip_RD)))
        per_arm[arm + "_ipJR1"] = float(np.mean(ip_JR1))
    rec = {"structure": name, "profile": profile, "n": op.n, "m": op.m,
           "n_draws": n_draws, "secs": round(time.time() - t0, 1)}
    for arm in ("node", "edge"):
        for k in ("R1", "Rdeg", "J", "true", "resid"):
            rec[f"{arm}_{k}_mean"] = per_arm[arm][k][0]
            rec[f"{arm}_{k}_median"] = per_arm[arm][k][1]
        rec[f"{arm}_cos_R1_Rdeg"] = per_arm[arm + "_ipRD"][0] / max(
            per_arm[arm]["R1"][0] * per_arm[arm]["Rdeg"][0], 1e-12)
    for k in ("R1", "J", "true", "resid", "Rdeg"):
        rec[f"ratio_{k}"] = rec[f"node_{k}_mean"] / max(
            rec[f"edge_{k}_mean"], 1e-12)
    for arm in ("node", "edge"):
        rec[f"{arm}_canc_frac"] = 1.0 - rec[f"{arm}_J_mean"] / max(
            rec[f"{arm}_R1_mean"], 1e-12)
        rec[f"{arm}_resid_frac"] = rec[f"{arm}_resid_mean"] / max(
            rec[f"{arm}_true_mean"], 1e-12)
    # first-order explanatory power: agreement ratio_J vs ratio_true
    rec["ratio_J_over_ratio_true"] = rec["ratio_J"] / max(
        rec["ratio_true"], 1e-12)
    return rec, per_arm


def main():
    out = []
    eA, PA, _ = fam.build_family_A(t=2, d=120, pool=5, seed=11)
    opA = ho.RelationOperator(PA.shape[0], eA)
    out.append(unit("familyA_coherent", "fixed", opA, PA, 300, 51)[0])
    for g, gl in ((fam.EPS_WEAK, "gamma=0.05_cancel"),
                  (1 - fam.EPS_WEAK, "gamma=0.95_max")):
        eB, PB, _ = fam.build_family_B(d=120, gamma=g, seed=12)
        opB = ho.RelationOperator(PB.shape[0], eB)
        out.append(unit("familyB_flat", gl, opB, PB, 300, 52)[0])

    cora_n, cora_edges, cora_labels = data_io.load_graph("cora-cc")
    op_c = ho.RelationOperator(cora_n, cora_edges)
    Pc = data_io.majority_binary_profile(cora_labels, 0.05)
    out.append(unit("cora-cc", "binary-eps0.05", op_c, Pc, 250, 53)[0])

    dblp_n, dblp_edges, dblp_labels = data_io.load_graph("dblp-ca")
    op_d = ho.RelationOperator(dblp_n, dblp_edges)
    Pd = data_io.majority_binary_profile(dblp_labels, 0.05)
    out.append(unit("dblp-ca", "binary-eps0.05", op_d, Pd, 100, 54)[0])

    fields = list(out[0].keys())
    with open(RESULTS / "normalization_response.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)

    s2 = {}
    s2p = RESULTS / "normalization_pilot.csv"
    if s2p.exists():
        for r in csv.DictReader(open(s2p, newline="")):
            s2[r["structure"] + "/" + r["profile"]] = r

    def s2get(key, col, r):
        rr = s2.get(key)
        return float(rr[col]) if rr else None

    summary = {r["structure"] + "/" + r["profile"]: {
        "ratio_raw_S2": s2get(r["structure"] + "/" + r["profile"],
                              "ratio_raw", r),
        "ratio_S_S2": s2get(r["structure"] + "/" + r["profile"],
                            "ratio_S", r),
         "ratio_R1_first_order": r["ratio_R1"],
         "ratio_J": r["ratio_J"],
         "ratio_true_common_base": r["ratio_true"],
         "canc_frac_node": r["node_canc_frac"],
         "canc_frac_edge": r["edge_canc_frac"],
         "resid_frac_node": r["node_resid_frac"],
         "cos_R1_Rdeg_node": r["node_cos_R1_Rdeg"],
         "secs": r["secs"]} for r in out}
    with open(RESULTS / "normalization_response_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
