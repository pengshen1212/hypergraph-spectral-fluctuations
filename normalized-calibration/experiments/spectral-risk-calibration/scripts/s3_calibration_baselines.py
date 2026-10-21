"""S3: dependence-aware vs misspecified calibration, equal budget.

Per unit (structure x known L1 node law x batch b in {0,1,2}):
  Reference  (frozen): M_hat = mean of R_ref node-driven S draws.
  Calibration: scores from calibrator in {edge-iid, node} (M_cal draws).
  Evaluation: NEW node-law draws (truth = node mechanism), N_eval draws.

For alpha in {0.01, 0.05, 0.10}: q_alpha = rank_threshold(calibration scores);
exceedance of the FROZEN q_alpha on evaluation draws + Clopper-Pearson CI;
calibration-internal exceedance (zero-alarm sanity); threshold stability
across batches.

Detection (alpha = 0.05, algorithm and threshold frozen):
  M1 severities: all node eps x {1.5, 2, 3} (candidate graph unchanged).
  M2 severities: eps x4 on top-degree hubs vs eps x4 on matched count of
      LOW-degree nodes (mechanism 2 of plan 8.2).
  Power = fraction of alternative-mechanism eval scores > q.
  NOTE: these mechanisms also move E S -> detection is TOTAL operator drift,
  labeled per plan 8.3; marginal-matched pure-variance studies deferred.

Writes results/calibration_baselines.csv. Guarantee labels per T6:
node-calibrated exceedance = grade 2 (finite-sample rank guarantee, specified
model); edge-calibrated = grade 5 misspecified control; detection = grade 4/5.
"""

from __future__ import annotations

import csv
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
ALPHAS = (0.01, 0.05, 0.10)
N_BATCH = 3


def binary_P(labels, eps_vec):
    labels = np.asarray(labels)
    classes, inverse, counts = np.unique(labels, return_inverse=True,
                                          return_counts=True)
    maj = int(np.argmax(counts))
    P = np.empty((len(labels), 2))
    is_maj = inverse == maj
    P[:, 1] = np.where(is_maj, 1.0 - eps_vec, eps_vec)
    P[:, 0] = 1.0 - P[:, 1]
    return P


def multiclass_P(labels, eps_vec):
    labels = np.asarray(labels)
    classes, inverse, counts = np.unique(labels, return_inverse=True,
                                         return_counts=True)
    C = len(classes)
    P = np.full((len(labels), C), 0.0)
    for c in range(C):
        P[:, c] = eps_vec / (C - 1)
    P[np.arange(len(labels)), inverse] = 1.0 - eps_vec
    return P


def base_eps_vector(labels, degrees, kind, param):
    """Known per-node error probabilities under the normal mechanism."""
    labels = np.asarray(labels)
    n = len(labels)
    if kind == "binary":
        eps = np.full(n, param)
    elif kind == "multiclass":
        eps = np.full(n, param)
    elif kind == "hetero":
        cut = float(np.quantile(degrees, 0.90))
        eps = np.where(degrees >= cut, param[1], param[0])
    else:
        raise ValueError(kind)
    return eps.astype(float)


def make_P(labels, degrees, kind, eps_vec):
    if kind in ("binary",):
        return binary_P(labels, eps_vec)
    return multiclass_P(labels, np.minimum(eps_vec, 1 - 1e-9))


def hub_mask(degrees, frac=0.10):
    cut = float(np.quantile(degrees, 1 - frac))
    return degrees >= cut


def lowdeg_mask(degrees, n_target, hub):
    order = np.argsort(degrees)
    picked = [u for u in order if not hub[u]][:int(n_target)]
    m = np.zeros(len(degrees), dtype=bool)
    m[picked] = True
    return m


def run_unit(name, profile_label, op, base_eps, labels, kind, degrees,
             budgets, seed0, n_alt_eval=None):
    n_ref, n_cal, n_eval = budgets
    if n_alt_eval is None:
        n_alt_eval = n_eval
    P0 = make_P(labels, degrees, kind, base_eps)
    p0 = op.edge_marginals(P0)
    node_s = ho.bound_node_sampler(op, P0)
    edge_s = ho.bound_edge_sampler(op, p0)

    rows = []
    # alternative mechanisms: (label, P)
    alts = []
    for sev in (1.5, 2.0, 3.0):
        alts.append((f"M1_x{sev}", make_P(labels, degrees, kind,
                                          base_eps * sev)))
    hub = hub_mask(degrees)
    n_hub = int(hub.sum())
    for scale in (4.0,):
        e = base_eps.copy()
        e[hub] = np.minimum(e[hub] * scale, 0.49)
        alts.append((f"M2_hubs_x{scale}", make_P(labels, degrees, kind, e)))
        e2 = base_eps.copy()
        lm = lowdeg_mask(degrees, n_hub, hub)
        e2[lm] = np.minimum(e2[lm] * scale, 0.49)
        alts.append((f"M2_lowdeg_x{scale}", make_P(labels, degrees, kind, e2)))

    qs = {}  # (calibrator, alpha) -> list over batches for stability
    for b in range(N_BATCH):
        sd = seed0 + 97 * b
        rng_ref = np.random.default_rng(sd + 1)
        M_hat = ho.estimate_center(op, node_s, n_ref, rng_ref)

        cal_scores = {}
        for cname, sampler in (("edge-iid", edge_s), ("node", node_s)):
            rng_c = np.random.default_rng(sd + 2 + (0 if cname == "edge-iid"
                                                    else 1))
            cal_scores[cname], _ = ho.draw_scores(op, sampler, n_cal, rng_c,
                                                  M_hat)
        rng_e = np.random.default_rng(sd + 10)
        eval_scores, eval_resid = ho.draw_scores(op, node_s, n_eval, rng_e,
                                                 M_hat)
        alt_scores = {}
        for idx, (alab, P_alt) in enumerate(alts):
            s_alt = ho.bound_node_sampler(op, P_alt)
            rng_a = np.random.default_rng(sd + 20 + 13 * idx)
            alt_scores[alab], _ = ho.draw_scores(op, s_alt, n_alt_eval, rng_a,
                                                 M_hat)

        for cname in cal_scores:
            for alpha in ALPHAS:
                q = ho.rank_threshold(cal_scores[cname], alpha)
                exceed = float(np.mean(eval_scores > q))
                lo, hi = ho.clopper_pearson(int(round(exceed * n_eval)), n_eval)
                cal_exceed = float(np.mean(cal_scores[cname] > q))
                rows.append({
                    "kind": "normal", "structure": name, "profile": profile_label,
                    "mechanism": "", "batch": b, "calibrator": cname,
                    "alpha": alpha, "q": q,
                    "exceed": exceed, "cp_lo": lo, "cp_hi": hi,
                    "cal_internal_exceed": cal_exceed,
                    "approx_frac_eval": float(np.mean(eval_resid > 1e-6)),
                    "n_eval": n_eval,
                })
                qs.setdefault((cname, alpha), []).append(q)
            # detection at alpha=0.05
            q_det = ho.rank_threshold(cal_scores[cname], 0.05)
            for alab, _ in alts:
                pw = float(np.mean(alt_scores[alab] > q_det))
                lo, hi = ho.clopper_pearson(int(round(pw * n_alt_eval)),
                                            n_alt_eval)
                rows.append({
                    "kind": "detection", "structure": name,
                    "profile": profile_label, "mechanism": alab, "batch": b,
                    "calibrator": cname, "alpha": 0.05, "q": q_det,
                    "exceed": pw, "cp_lo": lo, "cp_hi": hi,
                    "cal_internal_exceed": "", "approx_frac_eval": "",
                    "n_eval": n_alt_eval,
                })
    stability = {}
    for (cname, alpha), lst in qs.items():
        stability[f"{cname}@{alpha}"] = {"q_mean": float(np.mean(lst)),
                                         "q_cv": float(np.std(lst)
                                                       / max(np.mean(lst),
                                                             1e-12))}
    return rows, stability


def main():
    t0 = time.time()
    all_rows, all_stab = [], {}
    FIELDS = ["kind", "structure", "profile", "mechanism", "batch",
              "calibrator", "alpha", "q", "exceed", "cp_lo", "cp_hi",
              "cal_internal_exceed", "approx_frac_eval", "n_eval"]

    def flush():
        if not all_rows:
            return
        with open(RESULTS / "calibration_baselines.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(all_rows)
        import json
        with open(RESULTS / "calibration_stability.json", "w") as f:
            json.dump(all_stab, f, indent=2)

    # Cora: five profiles, pilot budgets
    cora_n, cora_edges, cora_labels = data_io.load_graph("cora-cc")
    op_c = ho.RelationOperator(cora_n, cora_edges)
    deg_c = np.zeros(cora_n)
    for e in cora_edges:
        for u in e:
            deg_c[u] += 1

    cora_units = [
        ("binary-eps0.05", "binary", 0.05),
        ("binary-eps0.15", "binary", 0.15),
        ("multiclass-eps0.05", "multiclass", 0.05),
        ("multiclass-eps0.15", "multiclass", 0.15),
        ("hetero-0.02/0.30", "hetero", (0.02, 0.30)),
    ]
    budgets_c = (200, 500, 1000)
    for i, (label, kind, param) in enumerate(cora_units):
        eps = base_eps_vector(cora_labels, deg_c, kind, param)
        print(f"cora {label}", flush=True)
        r, st = run_unit("cora-cc", label, op_c, eps, cora_labels, kind,
                         deg_c, budgets_c, 5000 + 100 * i, n_alt_eval=400)
        all_rows += r
        all_stab[f"cora-cc/{label}"] = st
        flush()

    # controlled coherent family sanity (edge calibration should fail there too)
    eA, PA, _ = fam.build_family_A(t=2, d=120, pool=5, seed=11)
    op_f = ho.RelationOperator(PA.shape[0], eA)
    # family profiles are not majority-coded; use custom P path
    P0 = PA.copy()
    p0 = op_f.edge_marginals(P0)
    rng = np.random.default_rng(777)
    M_hat = ho.estimate_center(op_f, ho.bound_node_sampler(op_f, P0), 400, rng)
    cs_node, _ = ho.draw_scores(op_f, ho.bound_node_sampler(op_f, P0), 800,
                                np.random.default_rng(778), M_hat)
    cs_edge, _ = ho.draw_scores(op_f, ho.bound_edge_sampler(op_f, p0), 800,
                                np.random.default_rng(779), M_hat)
    ev, _ = ho.draw_scores(op_f, ho.bound_node_sampler(op_f, P0), 1600,
                           np.random.default_rng(780), M_hat)
    for cname, cs in (("node", cs_node), ("edge-iid", cs_edge)):
        for alpha in ALPHAS:
            q = ho.rank_threshold(cs, alpha)
            ex = float(np.mean(ev > q))
            lo, hi = ho.clopper_pearson(int(round(ex * len(ev))), len(ev))
            all_rows.append({
                "kind": "normal", "structure": "familyA_coherent",
                "profile": "fixed", "mechanism": "", "batch": 0,
                "calibrator": cname, "alpha": alpha, "q": q, "exceed": ex,
                "cp_lo": lo, "cp_hi": hi,
                "cal_internal_exceed": float(np.mean(cs > q)),
                "approx_frac_eval": float(np.mean(np.zeros(1))),
                "n_eval": len(ev),
            })

    # DBLP: three profiles, smaller budgets
    dblp_n, dblp_edges, dblp_labels = data_io.load_graph("dblp-ca")
    op_d = ho.RelationOperator(dblp_n, dblp_edges)
    deg_d = np.zeros(dblp_n)
    for e in dblp_edges:
        for u in e:
            deg_d[u] += 1
    budgets_d = (80, 150, 300)
    for i, (label, kind, param) in enumerate([
            ("binary-eps0.05", "binary", 0.05),
            ("binary-eps0.15", "binary", 0.15),
            ("hetero-0.02/0.30", "hetero", (0.02, 0.30))]):
        eps = base_eps_vector(dblp_labels, deg_d, kind, param)
        print(f"dblp {label}", flush=True)
        r, st = run_unit("dblp-ca", label, op_d, eps, dblp_labels, kind,
                         deg_d, budgets_d, 8000 + 100 * i, n_alt_eval=150)
        all_rows += r
        all_stab[f"dblp-ca/{label}"] = st
        flush()

    flush()
    print(f"S3 done in {time.time()-t0:.0f}s ->", RESULTS /
          "calibration_baselines.csv")


if __name__ == "__main__":
    main()
