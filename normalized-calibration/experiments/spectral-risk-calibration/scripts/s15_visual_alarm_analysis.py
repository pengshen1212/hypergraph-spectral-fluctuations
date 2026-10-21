"""S15: calibration, detection, task association and fixed fallback analysis.

Reads ONLY the frozen run cache and the frozen protocol.  It never retrains and
never changes a threshold using evaluation data.

Outputs (CSV/JSON under results/visual_alarm_v1/analysis):
  e1_calibration.csv / e1_calibration_aggregate.csv
  e2_detection.csv   / e2_detection_aggregate.csv
  e3_task_association.csv
  e4_application.csv / e4_main_comparison.csv
  diagnostics.json
  sim/*.npz (deterministic L2 / independent-edge calibration scores)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

import hyperops as ho  # noqa: E402
import monitoring_workflow as mw  # noqa: E402
import vision_data as vd  # noqa: E402
import vision_runs as vr  # noqa: E402

ALPHAS = (0.05, 0.10)
GRID = (0.01, 0.025, 0.05, 0.10, 0.20)


def _load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def _batches(frozen, ds, counts):
    return list(range(1, int(counts["batches"]) + 1))


def _batch_dir(frozen, ds, b):
    cfg = frozen["configs"][ds]
    cfg = {**cfg, "_root": str(vr.ROOT)}
    return vr.stage_dataset_dir(cfg, "formal", ds, b)


def load_batch(frozen, ds, b, counts):
    ddir = _batch_dir(frozen, ds, b)
    cfg = frozen["configs"][ds]
    chash = frozen["config_sha256"][ds]
    refs, cals, evals = [], [], []
    for i in range(int(counts["ref"])):
        rec = _load_npz(ddir / "ref" / f"{ds}_b{b:02d}_ref_{i:03d}.npz")
        if str(rec["config_hash"]) != chash:
            raise SystemExit("reference cache config mismatch")
        refs.append(rec)
    for i in range(int(counts["cal"])):
        rec = _load_npz(ddir / "cal" / f"{ds}_b{b:02d}_cal_{i:03d}.npz")
        if str(rec["config_hash"]) != chash:
            raise SystemExit("calibration cache config mismatch")
        cals.append(rec)
    for i in range(int(counts["eval"])):
        rec = _load_npz(ddir / "eval" / f"{ds}_b{b:02d}_eval_{i:03d}.npz")
        if str(rec["config_hash"]) != chash:
            raise SystemExit("evaluation cache config mismatch")
        evals.append(rec)
    summ = _load_npz(ddir / "reference_summary.npz")
    if str(summ["config_hash"]) != chash:
        raise SystemExit("reference summary config mismatch")
    if int(summ["n_ref"]) != int(counts["ref"]):
        raise SystemExit("reference summary budget mismatch")
    return {"refs": refs, "cals": cals, "evals": evals, "summary": summ,
            "dir": ddir, "cfg": cfg, "config_hash": chash}


def stack_eval(evals):
    keys = ["acc_base", "acc_ref", "ce_base", "ce_ref", "change_fraction",
            "T", "frob", "entropy", "active_fraction", "active_dev",
            "T_resid", "T_approx", "infer_seconds", "score_seconds",
            "refine_seconds"]
    out = {k: np.stack([np.asarray(e[k]) for e in evals]) for k in keys}
    out["sigmas"] = np.asarray(evals[0]["sigmas"], dtype=float)
    out["Y_base"] = np.stack([e["Y_base"] for e in evals])
    out["Y_ref"] = np.stack([e["Y_ref"] for e in evals])
    return out


def _sim_cache(frozen, ds, b, counts, center, arm, rebuild):
    cfg = frozen["configs"][ds]
    path = HERE / "results" / "visual_alarm_v1" / "analysis" / "sim" / \
        f"{ds}_b{b:02d}_{arm}.npz"
    center_hash = hashlib.sha256(np.asarray(center).tobytes()).hexdigest()
    if path.exists() and not rebuild:
        z = _load_npz(path)
        if str(z.get("config_hash", "")) == frozen["config_sha256"][ds] and \
                str(z.get("center_hash", "")) == center_hash and \
                int(z.get("n_cal", -1)) == int(counts["cal"]):
            return z
    return None


def simulate_scores(frozen, ds, b, batch, counts, op, rebuild=False):
    """Deterministic node-product and independent-edge calibration scores."""
    cfg = frozen["configs"][ds]
    center = np.asarray(batch["summary"]["center"], dtype=np.float64)
    Ps = [np.asarray(r["P"], dtype=np.float64) for r in batch["refs"]]
    P_hat = mw.fit_node_law(Ps, int(cfg["datasets"][ds]["n_classes"]))
    out = {}
    ds_idx = int(cfg["datasets"][ds]["dataset_index"])
    for arm in ("node", "edge"):
        cached = _sim_cache(frozen, ds, b, counts, center, arm, rebuild)
        if cached is not None:
            out[arm] = cached
            continue
        ss = np.random.SeedSequence([
            int(cfg["seed_base"]), ds_idx, int(b), 771 if arm == "node" else 772])
        rng = np.random.default_rng(ss)
        t0 = time.time()
        if arm == "node":
            scores = mw.sample_node_scores(op, P_hat, counts["cal"], rng,
                                           center)
        else:
            p_hat = mw.edge_marginals(op, P_hat)
            scores = mw.sample_edge_scores(op, p_hat, counts["cal"], rng,
                                           center)
        scores["seconds"] = np.float64(time.time() - t0)
        scores["config_hash"] = frozen["config_sha256"][ds]
        scores["center_hash"] = hashlib.sha256(center.tobytes()).hexdigest()
        scores["n_cal"] = np.int64(counts["cal"])
        path = HERE / "results" / "visual_alarm_v1" / "analysis" / "sim" / \
            f"{ds}_b{b:02d}_{arm}.npz"
        vr._save_npz_atomic(path, scores)
        out[arm] = _load_npz(path)
    return out, P_hat


# ----------------------------------------------------------------------
# bootstrap / tests
# ----------------------------------------------------------------------
def stratified_bootstrap(arr_by_batch, reps, seed):
    """Equal-weight batch mean; resample run IDs within each batch."""
    rng = np.random.default_rng(int(seed))
    out = np.empty(int(reps))
    vals = [np.asarray(v, dtype=float) for v in arr_by_batch.values()]
    for r in range(int(reps)):
        means = []
        for v in vals:
            idx = rng.integers(0, len(v), len(v))
            means.append(float(np.mean(v[idx])))
        out[r] = float(np.mean(means))
    return out


def bootstrap_summary(arr_by_batch, reps, seed):
    obs = float(np.mean([np.mean(v) for v in arr_by_batch.values()]))
    bs = stratified_bootstrap(arr_by_batch, reps, seed)
    return {"estimate": obs,
            "ci_lo": float(np.percentile(bs, 2.5)),
            "ci_hi": float(np.percentile(bs, 97.5)),
            "boot_mean": float(np.mean(bs))}


def signflip_pvalue(arr_by_batch, reps, seed):
    rng = np.random.default_rng(int(seed) + 17)
    vals = [np.asarray(v, dtype=float) for v in arr_by_batch.values()]
    obs = float(np.mean([np.mean(v) for v in vals]))
    hits = 0
    for _ in range(int(reps)):
        means = []
        for v in vals:
            signs = rng.integers(0, 2, len(v)) * 2 - 1
            means.append(float(np.mean(signs * v)))
        if abs(np.mean(means)) >= abs(obs):
            hits += 1
    return (1.0 + hits) / (1.0 + int(reps))


def holm(pvals):
    """Holm step-down adjusted p-values, preserving input order."""
    items = sorted(enumerate(pvals), key=lambda t: t[1])
    m = len(items)
    adj = [0.0] * m
    running = 0.0
    for rank, (idx, p) in enumerate(items):
        val = min(1.0, (m - rank) * p)
        running = max(running, val)
        adj[idx] = running
    return adj


def rate_diff_by_batch(exceed_a, exceed_b):
    return {b: np.asarray(exceed_a[b], dtype=float) -
            np.asarray(exceed_b[b], dtype=float)
            for b in exceed_a}


# ----------------------------------------------------------------------
# per-dataset analysis
# ----------------------------------------------------------------------
def analyze_dataset(frozen, ds, rebuild_sim=False):
    counts = frozen["budget_counts"]
    cfg = frozen["configs"][ds]
    nboot = int(cfg["application"]["bootstrap_reps"])
    bseed = int(cfg["application"]["bootstrap_seed"])
    data = vd.load_vision_dataset(ds, root=vr.ROOT, verify=True)
    op = mw.operator_for(len(data.labels), data.edges,
                         kappa=float(cfg["monitoring"]["kappa"]))
    batches = _batches(frozen, ds, counts)
    e1, e2, e3, e4 = [], [], [], []
    diag = {"dataset": ds, "batches": {}, "config_sha256":
            frozen["config_sha256"][ds]}
    per_batch = {}
    t_start = time.time()

    for b in batches:
        print(f"[{ds} b{b}] load", flush=True)
        batch = load_batch(frozen, ds, b, counts)
        ev = stack_eval(batch["evals"])
        refs = batch["refs"]
        cals = batch["cals"]
        summary = batch["summary"]
        center = np.asarray(summary["center"], dtype=np.float64)
        m_ref = float(summary["m_ref"])
        ref_y = [np.asarray(r["Y"], dtype=np.int64) for r in refs]
        center_check = mw.fit_center(op, ref_y)
        center_diff = float(np.max(np.abs(center_check - center)))
        cal_T = np.array([float(r["T"]) for r in cals])
        cal_frob = np.array([float(r["frob"]) for r in cals])
        cal_ent = np.array([float(r["entropy"]) for r in cals])
        cal_act = np.array([float(r["active_dev"]) for r in cals])
        cal_acc = np.array([float(r["test_acc"]) for r in cals])
        eval_T = ev["T"][:, 0]
        eval_acc = ev["acc_base"][:, 0]
        sims, P_hat = simulate_scores(frozen, ds, b, batch, counts, op,
                                      rebuild=rebuild_sim)
        cal_node = np.asarray(sims["node"]["T"], dtype=float)
        cal_edge = np.asarray(sims["edge"]["T"], dtype=float)

        per_batch[b] = {
            "center_diff": center_diff,
            "cal_T": cal_T, "cal_frob": cal_frob, "cal_ent": cal_ent,
            "cal_act": cal_act, "cal_node": cal_node, "cal_edge": cal_edge,
            "eval_T": eval_T, "m_ref": m_ref, "ev": ev, "cals": cals,
            "summary": summary, "P_hat_hash": hashlib.sha256(
                P_hat.tobytes()).hexdigest(),
        }

        # ---------------- E1: three calibrators ----------------
        arms = (("L3_full_run", cal_T), ("L2_node_product", cal_node),
                ("edge_iid", cal_edge))
        for arm, cal_scores in arms:
            for alpha in ALPHAS:
                s = mw.threshold_summary(cal_scores, eval_T, alpha)
                e1.append({
                    "dataset": ds, "batch": b, "arm": arm, "alpha": alpha,
                    "n_cal": int(len(cal_scores)),
                    "n_eval": int(len(eval_T)),
                    "threshold": s["threshold"],
                    "exceed_count": s["exceed_count"],
                    "rate": s["rate"], "cp_lo": s["cp_lo"],
                    "cp_hi": s["cp_hi"],
                    "tie_cal": s["tie_rate_cal"],
                    "tie_eval": s["tie_rate_eval"],
                    "eval_clean_acc": float(np.mean(eval_acc)),
                    "eval_active_fraction": float(
                        np.mean(ev["active_fraction"][:, 0])),
                    "cal_seconds": float(np.median(
                        [float(r["score_seconds"]) for r in cals])),
                    "sim_seconds": float(sims[
                        "node" if arm == "L2_node_product" else "edge"
                    ]["seconds"]) if arm != "L3_full_run" else 0.0,
                })

        # ---------------- E2: detection / baselines ----------------
        for score_name in ("T", "frob", "entropy", "active_dev"):
            cal = {"T": cal_T, "frob": cal_frob, "entropy": cal_ent,
                   "active_dev": cal_act}[score_name]
            eval_s = {"T": ev["T"], "frob": ev["frob"],
                      "entropy": ev["entropy"],
                      "active_dev": ev["active_dev"]}[score_name]
            for alpha in GRID:
                q = ho.rank_threshold(cal, alpha)
                for si, sigma in enumerate(ev["sigmas"]):
                    exceed = eval_s[:, si] > q
                    k = int(exceed.sum())
                    lo, hi = ho.clopper_pearson(k, len(exceed))
                    e2.append({
                        "dataset": ds, "batch": b, "score": score_name,
                        "alpha": alpha, "sigma": float(sigma),
                        "threshold": float(q), "count": k,
                        "n": int(len(exceed)), "rate": k / len(exceed),
                        "cp_lo": lo, "cp_hi": hi,
                        "tie_cal": float(np.mean(cal == q)),
                        "tie_eval": float(np.mean(eval_s[:, si] == q)),
                    })
        # two-sided sensitivity U = |T - median(reference T)|
        cal_U = np.abs(cal_T - m_ref)
        eval_U = np.abs(ev["T"] - m_ref)
        for alpha in GRID:
            q = ho.rank_threshold(cal_U, alpha)
            for si, sigma in enumerate(ev["sigmas"]):
                exceed = eval_U[:, si] > q
                k = int(exceed.sum())
                lo, hi = ho.clopper_pearson(k, len(exceed))
                e2.append({
                    "dataset": ds, "batch": b, "score": "U_two_sided",
                    "alpha": alpha, "sigma": float(sigma),
                    "threshold": float(q), "count": k, "n": int(len(exceed)),
                    "rate": k / len(exceed), "cp_lo": lo, "cp_hi": hi,
                    "tie_cal": float(np.mean(cal_U == q)),
                    "tie_eval": float(np.mean(eval_U[:, si] == q)),
                })

        # ---------------- E3: task association ----------------
        qT = ho.rank_threshold(cal_T, 0.05)
        qH = ho.rank_threshold(cal_ent, 0.05)
        thresh = float(cfg["application"]["task_event_drop"])
        for si, sigma in enumerate(ev["sigmas"]):
            alarm = ev["T"][:, si] > qT
            event = (ev["acc_base"][:, 0] - ev["acc_base"][:, si]) >= thresh
            conf = mw.confusion(alarm, event)
            drop = ev["acc_base"][:, 0] - ev["acc_base"][:, si]
            ce_inc = ev["ce_base"][:, si] - ev["ce_base"][:, 0]
            e3.append({
                "dataset": ds, "batch": b, "sigma": float(sigma),
                **conf,
                "event_rate": float(np.mean(event)),
                "alarm_rate": float(np.mean(alarm)),
                "alarm_no_event_fraction": float(np.mean(alarm & ~event)),
                "event_no_alarm_fraction": float(np.mean(~alarm & event)),
                "mean_acc_drop": float(np.mean(drop)),
                "mean_ce_increase": float(np.mean(ce_inc)),
                "mean_change_fraction": float(
                    np.mean(ev["change_fraction"][:, si])),
                "qT": qT, "qH": qH,
            })

        # ---------------- E4: application policies ----------------
        lam = float(cfg["application"]["lambda"])
        for si, sigma in enumerate(ev["sigmas"]):
            base = ev["acc_base"][:, si]
            ref = ev["acc_ref"][:, si]
            alarmT = ev["T"][:, si] > qT
            alarmH = ev["entropy"][:, si] > qH
            spec = np.where(alarmT, base, ref)
            ent = np.where(alarmH, base, ref)
            k = int(alarmT.sum())
            rng = np.random.default_rng(
                np.random.SeedSequence([int(cfg["seed_base"]),
                                        int(cfg["datasets"][ds]["dataset_index"]),
                                        int(b), int(si), 991]))
            rand_means = np.empty(int(cfg["application"]["random_fallback_draws"]))
            for r in range(len(rand_means)):
                if k == 0:
                    idx = np.zeros(0, dtype=int)
                elif k == len(base):
                    idx = np.arange(len(base))
                else:
                    idx = rng.choice(len(base), size=k, replace=False)
                mask = np.zeros(len(base), dtype=bool)
                mask[idx] = True
                rand_means[r] = float(np.mean(np.where(mask, base, ref)))
            e4.append({
                "dataset": ds, "batch": b, "sigma": float(sigma),
                "lambda": lam,
                "policy_base": float(np.mean(base)),
                "policy_refine": float(np.mean(ref)),
                "policy_spectral_fallback": float(np.mean(spec)),
                "policy_entropy_fallback": float(np.mean(ent)),
                "policy_random_fallback": float(np.mean(rand_means)),
                "random_fallback_lo": float(np.percentile(rand_means, 2.5)),
                "random_fallback_hi": float(np.percentile(rand_means, 97.5)),
                "fallback_fraction_spectral": float(np.mean(alarmT)),
                "fallback_fraction_entropy": float(np.mean(alarmH)),
                "delta_spec_vs_refine": float(np.mean(spec - ref)),
                "delta_spec_vs_base": float(np.mean(spec - base)),
                "delta_ent_vs_refine": float(np.mean(ent - ref)),
                "delta_refine_vs_base": float(np.mean(ref - base)),
                "damage_event_rate": float(np.mean(
                    ref <= base - float(cfg["application"]["damage_event_drop"]))),
                "alarm_damage_tp": int(np.sum(alarmT & (
                    ref <= base - float(cfg["application"]["damage_event_drop"])))),
                "alarm_damage_fp": int(np.sum(alarmT & (
                    ref > base - float(cfg["application"]["damage_event_drop"])))),
                "score_seconds_median": float(np.median(
                    ev["score_seconds"][:, si])),
                "refine_seconds_median": float(np.median(
                    ev["refine_seconds"][:, si])),
                "infer_seconds_median": float(np.median(
                    ev["infer_seconds"][:, si])),
                "_spec_minus_ref": spec - ref,
                "_spec_minus_base": spec - base,
                "_ent_minus_ref": ent - ref,
                "_ref_minus_base": ref - base,
                "_alarmT": alarmT, "_alarmH": alarmH,
                "_ref": ref, "_base": base,
            })

        diag["batches"][str(b)] = {
            "center_recompute_max_abs_diff": center_diff,
            "m_ref": m_ref,
            "n_ref": int(counts["ref"]), "n_cal": int(counts["cal"]),
            "n_eval": int(counts["eval"]),
            "cal_T_unique": int(len(np.unique(cal_T))),
            "cal_T_tie_mode_fraction": float(
                np.max(np.unique(cal_T, return_counts=True)[1]) / len(cal_T)),
            "cal_node_unique": int(len(np.unique(cal_node))),
            "cal_edge_unique": int(len(np.unique(cal_edge))),
            "solver_approx_fraction_eval": float(
                np.mean(ev["T_approx"])),
            "active_fraction_clean_mean": float(
                np.mean(ev["active_fraction"][:, 0])),
            "active_fraction_clean_min": float(
                np.min(ev["active_fraction"][:, 0])),
            "active_fraction_clean_max": float(
                np.max(ev["active_fraction"][:, 0])),
            "cal_mean_test_acc": float(np.mean(cal_acc)),
            "eval_mean_clean_acc": float(np.mean(eval_acc)),
            "min_class_count": int(np.bincount(
                data.labels, minlength=data.n_classes).min()),
            "sim_seconds_node": float(sims["node"]["seconds"]),
            "sim_seconds_edge": float(sims["edge"]["seconds"]),
            "node_law_hash": hashlib.sha256(P_hat.tobytes()).hexdigest(),
            "train_seconds_median": float(np.median(
                [float(r["train_seconds"]) for r in refs + cals + batch["evals"]])),
            "eval_run_seconds_median": float(np.median(
                [float(r["run_seconds"]) for r in batch["evals"]])),
            "cache_bytes_median": float(np.median(
                [p.stat().st_size for p in Path(batch["dir"]).rglob("*.npz")])),
            "score_seconds_median_eval": float(np.median(
                ev["score_seconds"])),
            "refine_seconds_median_eval": float(np.median(
                ev["refine_seconds"])),
            "infer_seconds_median_eval": float(np.median(
                ev["infer_seconds"])),
        }

    # ---------------- aggregate E1 across batches ----------------
    e1_agg = []
    for arm in ("L3_full_run", "L2_node_product", "edge_iid"):
        for alpha in ALPHAS:
            rows = [r for r in e1 if r["arm"] == arm and r["alpha"] == alpha]
            rates = [r["rate"] for r in rows]
            e1_agg.append({
                "dataset": ds, "arm": arm, "alpha": alpha,
                "batches": len(rows),
                "rates": ";".join(f"{v:.6g}" for v in rates),
                "mean_rate": float(np.mean(rates)),
                "min_rate": float(np.min(rates)),
                "max_rate": float(np.max(rates)),
                "endpoint_envelope_lo": float(np.min([r["cp_lo"] for r in rows])),
                "endpoint_envelope_hi": float(np.max([r["cp_hi"] for r in rows])),
                "note": ("equal-weight mean of batch rates; endpoints are not "
                         "a confidence interval for the mean"),
            })

    # L2/L3 and edge/L3 paired differences on shared evaluation runs
    q05 = {}
    for b in batches:
        q05[b] = {
            "L3_full_run": ho.rank_threshold(per_batch[b]["cal_T"], 0.05),
            "L2_node_product": ho.rank_threshold(per_batch[b]["cal_node"], 0.05),
            "edge_iid": ho.rank_threshold(per_batch[b]["cal_edge"], 0.05),
        }
    ex = {}
    for arm in ("L3_full_run", "L2_node_product", "edge_iid"):
        ex[arm] = {b: (per_batch[b]["eval_T"] > q05[b][arm]).astype(float)
                   for b in batches}
    for other in ("L2_node_product", "edge_iid"):
        diff = rate_diff_by_batch(ex[other], ex["L3_full_run"])
        bs = bootstrap_summary(diff, nboot, bseed + 5)
        e1_agg.append({
            "dataset": ds, "arm": f"{other}_minus_L3", "alpha": 0.05,
            "batches": len(batches),
            "rates": ";".join(f"{np.mean(diff[b]):.6g}" for b in batches),
            "mean_rate": bs["estimate"], "min_rate": "",
            "max_rate": "", "endpoint_envelope_lo": bs["ci_lo"],
            "endpoint_envelope_hi": bs["ci_hi"],
            "note": "paired difference on shared evaluation runs; bootstrap CI",
        })

    # ---------------- aggregate E2 across batches ----------------
    e2_agg = []
    for score in ("T", "frob", "entropy", "active_dev", "U_two_sided"):
        for alpha in GRID:
            for si, sigma in enumerate(per_batch[batches[0]]["ev"]["sigmas"]):
                rows = [r for r in e2 if r["score"] == score and
                        r["alpha"] == alpha and r["sigma"] == float(sigma)]
                rates = [r["rate"] for r in rows]
                e2_agg.append({
                    "dataset": ds, "score": score, "alpha": alpha,
                    "sigma": float(sigma), "batches": len(rows),
                    "rates": ";".join(f"{v:.6g}" for v in rates),
                    "mean_rate": float(np.mean(rates)),
                    "min_rate": float(np.min(rates)),
                    "max_rate": float(np.max(rates)),
                    "endpoint_envelope_lo": float(min(r["cp_lo"] for r in rows)),
                    "endpoint_envelope_hi": float(max(r["cp_hi"] for r in rows)),
                })

    # ---------------- aggregate E4 and main comparison ----------------
    e4_agg = []
    sigmas = per_batch[batches[0]]["ev"]["sigmas"]
    for si, sigma in enumerate(sigmas):
        for policy, key in (("base", "policy_base"),
                            ("always_refine", "policy_refine"),
                            ("spectral_fallback", "policy_spectral_fallback"),
                            ("entropy_fallback", "policy_entropy_fallback"),
                            ("random_fallback", "policy_random_fallback")):
            rows = [r for r in e4 if r["sigma"] == float(sigma)]
            vals = [r[key] for r in rows]
            e4_agg.append({
                "dataset": ds, "policy": policy, "sigma": float(sigma),
                "batches": len(rows),
                "per_batch": ";".join(f"{v:.6g}" for v in vals),
                "mean_acc": float(np.mean(vals)),
                "min_acc": float(np.min(vals)),
                "max_acc": float(np.max(vals)),
            })
    main_sigma = float(cfg["application"]["main_sigma"])
    sec_sigma = float(cfg["application"]["secondary_sigma"])
    main_rows = []
    for si, sigma in enumerate(sigmas):
        if float(sigma) not in (main_sigma, sec_sigma, 0.0):
            continue
        rows = [r for r in e4 if r["sigma"] == float(sigma)]
        for label, key in (("spec_minus_refine", "_spec_minus_ref"),
                           ("spec_minus_base", "_spec_minus_base"),
                           ("ent_minus_refine", "_ent_minus_ref"),
                           ("refine_minus_base", "_ref_minus_base")):
            arr = {r["batch"]: np.asarray(r[key], dtype=float) for r in rows}
            bs = bootstrap_summary(arr, nboot, bseed + 11)
            p = signflip_pvalue(arr, nboot, bseed + 12)
            main_rows.append({
                "dataset": ds, "sigma": float(sigma), "comparison": label,
                "estimate": bs["estimate"], "ci_lo": bs["ci_lo"],
                "ci_hi": bs["ci_hi"], "p_signflip": p,
                "n_runs": int(sum(len(v) for v in arr.values())),
                "fallback_fraction_spectral": float(np.mean(
                    [r["fallback_fraction_spectral"] for r in rows])),
                "fallback_fraction_entropy": float(np.mean(
                    [r["fallback_fraction_entropy"] for r in rows])),
            })
    # damage-event association at each strength
    damage = []
    for si, sigma in enumerate(sigmas):
        rows = [r for r in e4 if r["sigma"] == float(sigma)]
        for kind, key in (("spectral", "_alarmT"), ("entropy", "_alarmH")):
            tp = sum(int(np.sum(r[key] & (r["_ref"] <= r["_base"] -
                                          float(cfg["application"]["damage_event_drop"]))))
                     for r in rows)
            fp = sum(int(np.sum(r[key] & (r["_ref"] > r["_base"] -
                                          float(cfg["application"]["damage_event_drop"]))))
                     for r in rows)
            fn = sum(int(np.sum(~r[key] & (r["_ref"] <= r["_base"] -
                                           float(cfg["application"]["damage_event_drop"]))))
                     for r in rows)
            tn = sum(int(np.sum(~r[key] & (r["_ref"] > r["_base"] -
                                           float(cfg["application"]["damage_event_drop"]))))
                     for r in rows)
            damage.append({
                "dataset": ds, "sigma": float(sigma), "alarm": kind,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "detection_rate": tp / (tp + fn) if (tp + fn) else math.nan,
                "false_alarm_rate": fp / (fp + tn) if (fp + tn) else math.nan,
                "precision": tp / (tp + fp) if (tp + fp) else math.nan,
            })

    diag["seconds"] = time.time() - t_start
    diag["counts"] = {k: int(v) for k, v in counts.items()}
    return {
        "e1": e1, "e1_agg": e1_agg, "e2": e2, "e2_agg": e2_agg,
        "e3": e3, "e4": e4, "e4_agg": e4_agg, "main": main_rows,
        "damage": damage, "diag": diag, "per_batch": per_batch,
        "nboot": nboot, "bseed": bseed,
    }


def _strip_internal(rows):
    out = []
    for r in rows:
        out.append({k: v for k, v in r.items() if not str(k).startswith("_")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--rebuild-sim", action="store_true")
    args = ap.parse_args()
    frozen = vr.load_json(args.protocol)
    if frozen.get("budget_selected") not in ("formal_full", "formal_resource"):
        raise SystemExit("protocol has no selected budget")
    outdir = HERE / "results" / "visual_alarm_v1" / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    all_res = {}
    for ds in frozen["dataset_order"]:
        print(f"\n===== analyzing {ds} =====", flush=True)
        all_res[ds] = analyze_dataset(frozen, ds, args.rebuild_sim)
        r = all_res[ds]
        _write_csv(outdir / f"e1_calibration_{ds}.csv", r["e1"])
        _write_csv(outdir / f"e1_calibration_aggregate_{ds}.csv", r["e1_agg"])
        _write_csv(outdir / f"e2_detection_{ds}.csv", r["e2"])
        _write_csv(outdir / f"e2_detection_aggregate_{ds}.csv", r["e2_agg"])
        _write_csv(outdir / f"e3_task_association_{ds}.csv", r["e3"])
        _write_csv(outdir / f"e4_application_{ds}.csv", _strip_internal(r["e4"]))
        _write_csv(outdir / f"e4_application_aggregate_{ds}.csv", r["e4_agg"])
        _write_csv(outdir / f"e4_main_comparison_{ds}.csv", r["main"])
        _write_csv(outdir / f"e4_damage_association_{ds}.csv", r["damage"])
        vr._write_json_atomic(outdir / f"diagnostics_{ds}.json", r["diag"])

    # Holm correction for the primary application comparison (spec vs refine,
    # sigma=0.5), family = two datasets.
    main_sigma = float(frozen["configs"][frozen["dataset_order"][0]]
                       ["application"]["main_sigma"])
    pvals, labels = [], []
    for ds in frozen["dataset_order"]:
        row = next(r for r in all_res[ds]["main"]
                   if r["sigma"] == main_sigma and
                   r["comparison"] == "spec_minus_refine")
        pvals.append(row["p_signflip"])
        labels.append(ds)
    adj = holm(pvals)
    holm_rows = []
    for ds, p, a in zip(labels, pvals, adj):
        row = next(r for r in all_res[ds]["main"]
                   if r["sigma"] == main_sigma and
                   r["comparison"] == "spec_minus_refine")
        holm_rows.append({"dataset": ds, "sigma": main_sigma,
                          "comparison": "spec_minus_refine",
                          "estimate": row["estimate"],
                          "ci_lo": row["ci_lo"], "ci_hi": row["ci_hi"],
                          "p_raw": p, "p_holm": a})
    _write_csv(outdir / "e4_main_holm.csv", holm_rows)

    summary = {
        "analysis_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "protocol_sha256": vd.sha256_file(args.protocol),
        "budget_selected": frozen["budget_selected"],
        "budget_counts": frozen["budget_counts"],
        "config_sha256": frozen["config_sha256"],
        "datasets": {},
        "holm_main": holm_rows,
        "code_sha256": vr.code_hashes(),
        "notes": [
            "Batch rates keep their own center and threshold; no pooled "
            "binomial interval is reported for mixed batches.",
            "Paired bootstrap resamples run IDs within batch; nodes are never "
            "treated as independent replicates.",
            "Task association is offline and does not use test labels online.",
            "Negative and degenerate results are retained in the CSV outputs.",
        ],
    }
    for ds in frozen["dataset_order"]:
        r = all_res[ds]
        summary["datasets"][ds] = {
            "e1_aggregate": r["e1_agg"],
            "e2_primary": [x for x in r["e2_agg"]
                           if x["score"] in ("T", "frob", "entropy",
                                             "active_dev", "U_two_sided")
                           and x["alpha"] == 0.05],
            "e3": r["e3"],
            "e4_aggregate": r["e4_agg"],
            "e4_main": r["main"],
            "damage": r["damage"],
            "diagnostics": r["diag"],
        }
    vr._write_json_atomic(outdir / "analysis_summary.json", summary)
    print(f"\nanalysis written to {outdir}", flush=True)


if __name__ == "__main__":
    main()
