"""S13: vision asset audit, development selection and pipeline pilot.

Modes
-----
audit   Verify hashes/IDs/splits/operator shapes; write audit JSON.
dev     Train 10 development runs per dataset (batch 0, role dev), choose the
        epoch ceiling by validation loss, the learning rate only if the
        declared first choice diverges, and lambda by validation accuracy.
pilot   10/19/20 reference/calibration/evaluation runs for one dataset; writes
        timing, degeneracy and cache-footprint diagnostics.  Pilot results are
        NOT used to select a favourable outcome.
decide  Apply the pre-registered budget rule to the two pilot reports and
        write budget_decision.json.  Must run before formal generation.

No manuscript numbers are produced here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

import hyperops as ho  # noqa: E402
import monitoring_workflow as mw  # noqa: E402
import vision_data as vd  # noqa: E402
import vision_predictor as vp  # noqa: E402
import vision_runs as vr  # noqa: E402

RESULT_ROOT = HERE / "results" / "visual_alarm_v1"


def _out(cfg):
    return HERE / cfg["output_root"].split("spectral-risk-calibration/", 1)[-1] \
        if False else HERE / "results" / "visual_alarm_v1"


def _atomic_json(path, obj):
    vr._write_json_atomic(path, obj)


def audit(cfg, dataset):
    print(f"[audit] {dataset}", flush=True)
    data = vd.load_vision_dataset(dataset, root=vr.ROOT, verify=True)
    masks = vd.stratified_masks(data.labels, cfg["split"]["seed"],
                                tuple(cfg["split"]["fractions"]))
    std = vd.fit_standardizer(data.X, masks["train"])
    X_std = vd.standardize(data.X, std)
    op = mw.operator_for(len(data.labels), data.edges,
                         kappa=float(cfg["monitoring"]["kappa"]))
    S_all = mw.all_active_operator(op)
    t0 = time.time()
    s_norm, s_resid = ho.spectral_norm_entries(
        op, op.S_data(np.ones(op.m)), np.zeros(op.n_entries))
    op_seconds = time.time() - t0
    summary = vd.dataset_summary(data)
    summary.update({
        "split": {
            "seed": int(cfg["split"]["seed"]),
            "fingerprint": vd.split_fingerprint(masks),
            "n_train": int(masks["train"].sum()),
            "n_val": int(masks["val"].sum()),
            "n_test": int(masks["test"].sum()),
            "train_class_min": int(np.bincount(
                data.labels[masks["train"]],
                minlength=data.n_classes).min()),
            "val_class_min": int(np.bincount(
                data.labels[masks["val"]],
                minlength=data.n_classes).min()),
            "test_class_min": int(np.bincount(
                data.labels[masks["test"]],
                minlength=data.n_classes).min()),
        },
        "standardizer": {
            "zero_variance_dims": int(std["zero_var"].sum()),
            "mean_abs_max": float(np.abs(std["mean"]).max()),
            "scale_min_nonzero": float(std["scale"][~std["zero_var"]].min()),
        },
        "operator": {
            "kappa": float(cfg["monitoring"]["kappa"]),
            "n_entries": int(op.n_entries),
            "nnz_ratio": float(op.n_entries / (op.n * op.n)),
            "S_all_norm": float(s_norm),
            "S_all_resid": float(s_resid),
            "S_all_seconds": op_seconds,
            "all_active_fraction": float(np.mean(op.flags_from_Y(
                np.zeros(op.n, dtype=int)))),
        },
        "feature_shape": [int(v) for v in X_std.shape],
        "feature_finite": bool(np.isfinite(X_std).all()),
        "object_rows_unique": bool(len(np.unique(data.object_rows)) ==
                                   len(data.object_rows)),
        "extra_rows_are_not_objects": bool(data.n_extra_rows > 0),
        "seed_examples": {
            "ref_b1_run0": vr.seed_for(cfg, cfg["datasets"][dataset]
                                       ["dataset_index"], 1, "ref", 0),
            "cal_b1_run0": vr.seed_for(cfg, cfg["datasets"][dataset]
                                       ["dataset_index"], 1, "cal", 0),
            "eval_b1_run0": vr.seed_for(cfg, cfg["datasets"][dataset]
                                        ["dataset_index"], 1, "eval", 0),
        },
    })
    _atomic_json(RESULT_ROOT / "audit" / f"{dataset}.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def _train_dev(data, masks, ctx, cfg, lr, limit_max):
    n = len(data.labels)
    out = []
    for i in range(int(cfg["budgets"]["dev"]["runs_per_dataset"])):
        seed = vr.seed_for(cfg, ctx["data_index"], 0, "dev", i)
        pred_cfg = vp.PredictorConfig.from_dict(cfg["predictor"])
        pred_cfg.lr = float(lr)
        res = vp.train_predictor(
            ctx["G"], ctx["S_all"], data.labels, masks["train"], masks["val"],
            data.n_classes, seed, pred_cfg,
            epoch_limits=tuple(cfg["predictor"]["epoch_candidates"]),
            deadline=time.time() + float(cfg["timeout_seconds_per_run"]))
        out.append((seed, res))
        cks = res["checkpoints"]
        desc = {k: (v.get("status"), v.get("epoch"), v.get("val_loss"))
                for k, v in cks.items()}
        print(f"  dev {i}: {desc}", flush=True)
    return out


def development(cfg, dataset):
    print(f"[dev] {dataset}", flush=True)
    ctx = vr.build_context(cfg, dataset)
    data, masks = ctx["data"], ctx["masks"]
    base_cfg = cfg
    attempts = []
    chosen = None
    for lr in cfg["predictor"]["learning_rate_candidates"]:
        t0 = time.time()
        runs = _train_dev(data, masks, ctx, base_cfg, lr,
                          max(cfg["predictor"]["epoch_candidates"]))
        bad = [i for i, (_s, r) in enumerate(runs)
               if any(v.get("status") != "ok"
                      for v in r["checkpoints"].values())]
        attempts.append({"learning_rate": float(lr),
                         "failed_runs": bad,
                         "seconds": time.time() - t0})
        if not bad:
            chosen = (float(lr), runs)
            break
        print(f"  lr {lr}: failed dev runs {bad}; trying next", flush=True)
    if chosen is None:
        raise RuntimeError(f"{dataset}: all development learning rates failed")

    lr, runs = chosen
    limits = list(cfg["predictor"]["epoch_candidates"])
    epoch_rows = []
    for lim in limits:
        losses = [r["checkpoints"][lim]["val_loss"] for _s, r in runs]
        accs = [r["checkpoints"][lim]["val_acc"] for _s, r in runs]
        epoch_rows.append({"epoch": int(lim),
                           "mean_val_loss": float(np.mean(losses)),
                           "mean_val_acc": float(np.mean(accs)),
                           "per_run_val_loss": [float(v) for v in losses],
                           "per_run_val_acc": [float(v) for v in accs]})
    best = min(epoch_rows, key=lambda r: (r["mean_val_loss"],
                                          -r["mean_val_acc"], r["epoch"]))
    epoch = int(best["epoch"])

    P_list, S_list, dev_records = [], [], []
    for i, (seed, res) in enumerate(runs):
        ck = res["checkpoints"][epoch]
        P, Y = vp.predict_full(ctx["G"], ctx["S_all"], ck["W1"], ck["W2"])
        flags = ctx["op"].flags_from_Y(Y)
        S_Y = ctx["op"].S_csr(flags)
        P_list.append(P)
        S_list.append(S_Y)
        dev_records.append({
            "index": i, "seed": int(seed), "epoch": int(ck["epoch"]),
            "val_loss": float(ck["val_loss"]), "val_acc": float(ck["val_acc"]),
            "train_seconds": float(ck["train_seconds"]),
            "checkpoints": {
                str(lim): {"epoch": int(res["checkpoints"][lim]["epoch"]),
                           "val_loss": float(res["checkpoints"][lim]["val_loss"]),
                           "val_acc": float(res["checkpoints"][lim]["val_acc"])}
                for lim in limits},
        })
        path = RESULT_ROOT / "development" / dataset / f"run_{i:02d}.npz"
        vr._save_npz_atomic(path, {
            "seed": np.int64(seed), "W1": ck["W1"].astype(np.float32),
            "W2": ck["W2"].astype(np.float32), "Y": Y.astype(np.uint8),
            "P": P.astype(np.float16), "epoch": np.int64(ck["epoch"]),
            "val_loss": np.float64(ck["val_loss"]),
            "val_acc": np.float64(ck["val_acc"]),
            "test_acc": np.float64(mw.accuracy(
                Y, data.labels, masks["test"])),
        })

    lam, lam_rows = mw.select_lambda(
        P_list, S_list, data.labels, masks["val"],
        cfg["application"]["lambda_candidates"])
    selection = {
        "dataset": dataset,
        "learning_rate": float(lr),
        "epochs": int(epoch),
        "lambda": float(lam),
        "epoch_selection": epoch_rows,
        "lambda_selection": lam_rows,
        "learning_rate_attempts": attempts,
        "dev_runs": dev_records,
        "selected_config_sha256": None,
        "selected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _atomic_json(RESULT_ROOT / "development" / f"selection_{dataset}.json",
                 selection)
    print(f"[dev] {dataset}: lr={lr} epochs={epoch} lambda={lam}", flush=True)
    return selection


def _dir_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def _percentiles(values, qs=(50, 95)):
    if not values:
        return {f"p{q}": math.nan for q in qs}
    return {f"p{q}": float(np.percentile(values, q)) for q in qs}


def pilot(cfg, dataset, jobs):
    print(f"[pilot] {dataset}", flush=True)
    b = cfg["budgets"]["pilot"]
    ddir = vr.stage_dataset_dir(cfg, "pilot", dataset)
    config_hash = vr.config_hash(cfg)
    ref_dir = ddir / "ref"
    cal_dir = ddir / "cal"
    eval_dir = ddir / "eval"
    t0 = time.time()
    s_ref = vr.run_role(cfg, dataset, "ref", 0, b["ref"], ref_dir,
                        config_hash, center_path=None, jobs=jobs)
    ref_summary = vr.build_reference_summary(
        cfg, dataset, 0, ref_dir, b["ref"], config_hash,
        summary_path=ddir / "reference_summary.npz")
    center_path = str(ref_summary["path"])
    s_cal = vr.run_role(cfg, dataset, "cal", 0, b["cal"], cal_dir,
                        config_hash, center_path=center_path, jobs=jobs)
    s_eval = vr.run_role(cfg, dataset, "eval", 0, b["eval"], eval_dir,
                         config_hash, center_path=center_path, jobs=jobs)

    # timing / cache / degeneracy diagnostics from the pilot cache
    train_s, eval_s, score_s, infer_s, rss, cache_b = [], [], [], [], [], []
    active, T_cal, approx = [], [], []
    for role, path in (("ref", ref_dir), ("cal", cal_dir), ("eval", eval_dir)):
        for f in sorted(Path(path).glob("*.npz")):
            cache_b.append(f.stat().st_size)
            with np.load(f, allow_pickle=False) as z:
                if "train_seconds" in z:
                    train_s.append(float(z["train_seconds"]))
                if role == "eval":
                    eval_s.append(float(z["run_seconds"]))
                    score_s.extend(np.asarray(z["score_seconds"]).tolist())
                    infer_s.extend(np.asarray(z["infer_seconds"]).tolist())
                    active.extend(np.asarray(z["active_fraction"]).tolist())
                    approx.extend(np.asarray(z["T_approx"]).tolist())
                    rss.append(float(z["worker_rss_mb"])
                               if "worker_rss_mb" in z.files else np.nan)
                elif role == "cal":
                    active.append(float(z["active_fraction"]))
                    T_cal.append(float(z["T"]))
                    approx.append(bool(z["T_approx"]))
                    score_s.append(float(z["score_seconds"]))
    T_cal = np.asarray(T_cal)
    uniq, counts = np.unique(T_cal, return_counts=True)
    tie_frac = float(counts.max() / counts.sum()) if len(counts) else math.nan
    report = {
        "dataset": dataset,
        "budget": {k: int(v) for k, v in b.items()},
        "counts_ok": bool(
            s_ref["ok"] + s_ref["skipped_complete"] == b["ref"] and
            s_cal["ok"] + s_cal["skipped_complete"] == b["cal"] and
            s_eval["ok"] + s_eval["skipped_complete"] == b["eval"]),
        "failed": int(s_ref["failed"] + s_cal["failed"] + s_eval["failed"]),
        "train_seconds_median": float(np.median(train_s)) if train_s else None,
        "train_seconds": _percentiles(train_s),
        "eval_run_seconds": _percentiles(eval_s),
        "eval_non_train_seconds": _percentiles(
            [e - np.median(train_s) for e in eval_s]) if eval_s else {},
        "score_seconds": _percentiles(score_s),
        "infer_seconds": _percentiles(infer_s),
        "worker_rss_mb_median": float(np.nanmedian(rss)) if rss else None,
        "cache_bytes_total": int(np.sum(cache_b)),
        "cache_bytes_median": float(np.median(cache_b)) if cache_b else None,
        "ref_score_seconds_total": float(ref_summary["score_seconds"]),
        "active_fraction": _percentiles(active),
        "active_fraction_below_0_05": float(np.mean(np.asarray(active) < 0.05))
        if active else None,
        "T_cal_tie_fraction_at_mode": tie_frac,
        "T_cal_unique": int(len(uniq)),
        "solver_approx_fraction": float(np.mean(approx)) if approx else None,
        "wall_seconds": time.time() - t0,
        "reference_center_norm": float(np.linalg.norm(
            ref_summary["center"])),
        "reference_m_ref": float(ref_summary["m_ref"]),
        "note": "Pilot only: not used to select a positive result.",
    }
    _atomic_json(ddir / "timing.json", report)
    (RESULT_ROOT / "pilot").mkdir(parents=True, exist_ok=True)
    _atomic_json(RESULT_ROOT / "pilot" / f"timing_{dataset}.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def _load_timing(dataset):
    p = RESULT_ROOT / "pilot" / f"timing_{dataset}.json"
    if not p.exists():
        raise FileNotFoundError(f"missing pilot timing: {p}")
    return json.load(open(p, encoding="utf-8"))


def decide(cfg):
    reports = {d: _load_timing(d) for d in cfg["datasets"]}
    full = cfg["budgets"]["formal_full"]
    res = cfg["budgets"]["formal_resource"]
    margin = float(cfg["budget_decision"]["margin"])

    def project(budget, ds):
        r = reports[ds]
        train_n = (budget["ref"] + budget["cal"] + budget["eval"]) * \
            budget["batches"]
        eval_n = budget["eval"] * budget["batches"]
        ref_n = budget["ref"] * budget["batches"]
        cal_n = budget["cal"] * budget["batches"]
        t_train = float(r["train_seconds_median"])
        t_eval_extra = max(0.0, float(r["eval_run_seconds"]["p50"]) - t_train)
        t_score = float(r["score_seconds"]["p50"]
                        if math.isfinite(r["score_seconds"]["p50"]) else 0.0)
        hours = (train_n * t_train + eval_n * t_eval_extra +
                 (ref_n + cal_n) * t_score +
                 2 * cal_n * t_score) / 3600.0
        cache = ((budget["ref"] + budget["cal"] + budget["eval"]) *
                 budget["batches"] * float(r["cache_bytes_median"] or 0))
        return {"train_seconds": t_train,
                "eval_extra_seconds": t_eval_extra,
                "score_seconds": t_score,
                "projected_hours": margin * hours,
                "projected_cache_gb": cache / (1024 ** 3),
                "n_train_runs": train_n}

    proj_full = {d: project(full, d) for d in cfg["datasets"]}
    proj_res = {d: project(res, d) for d in cfg["datasets"]}
    hours_full = sum(v["projected_hours"] for v in proj_full.values())
    cache_full = sum(v["projected_cache_gb"] for v in proj_full.values())
    use_full = (hours_full <= float(cfg["budget_decision"]["max_full_hours"]) and
                cache_full <= float(cfg["budget_decision"]["max_full_cache_gb"]))
    decision = {
        "decided_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rule": cfg["budget_decision"]["rule"],
        "thresholds": {
            "max_full_hours": cfg["budget_decision"]["max_full_hours"],
            "max_full_cache_gb": cfg["budget_decision"]["max_full_cache_gb"],
            "margin": margin,
        },
        "pilot_reports": reports,
        "projection_full": proj_full,
        "projection_resource": proj_res,
        "projected_full_hours": hours_full,
        "projected_full_cache_gb": cache_full,
        "budget_selected": "formal_full" if use_full else "formal_resource",
        "reason": ("full budget within pre-registered time/cache limits"
                   if use_full else
                   "full budget exceeds a pre-registered time/cache limit"),
    }
    _atomic_json(RESULT_ROOT / "budget_decision.json", decision)
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
    return decision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["audit", "dev", "pilot", "decide"])
    ap.add_argument("--dataset", default=None,
                    choices=["ntu2012", "modelnet40"])
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    cfg = vr.load_json(args.config)
    cfg["_root"] = str(vr.ROOT)
    if args.mode == "decide":
        decide(cfg)
        return
    if args.dataset is None:
        raise SystemExit("--dataset is required for this mode")
    if args.mode == "audit":
        audit(cfg, args.dataset)
    elif args.mode == "dev":
        development(cfg, args.dataset)
    elif args.mode == "pilot":
        sel_path = RESULT_ROOT / "development" / f"selection_{args.dataset}.json"
        if not sel_path.exists():
            raise SystemExit(f"missing development selection: {sel_path}")
        sel = json.load(open(sel_path, encoding="utf-8"))
        eff = vr.selected_config(cfg, sel)
        pilot(eff, args.dataset, args.jobs)


if __name__ == "__main__":
    main()
