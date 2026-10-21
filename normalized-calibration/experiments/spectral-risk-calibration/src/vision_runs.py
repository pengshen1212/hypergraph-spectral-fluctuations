"""Shared run-cache machinery for the vision alarm protocol.

One "run" is one complete training run of the fixed predictor with a distinct
initialisation seed.  The graph, node split, preprocessing and training
protocol are frozen.  Reference / calibration / evaluation runs are disjoint
seed streams; evaluation runs additionally share one noise field Z across all
stress strengths, so the strength trajectory is paired within a run.

Storage layout (stage = pilot | formal):
    <root>/<stage>/<dataset>[/batchNN]/{ref,cal,eval}/<run_id>.npz
    <root>/<stage>/<dataset>[/batchNN]/reference_summary.npz

Every cache file carries the frozen config SHA-256; resume refuses to mix
files from a different configuration.  Saves are atomic (tmp + os.replace).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import hyperops as ho
import monitoring_workflow as mw
import vision_data as vd
import vision_predictor as vp

SRC = Path(__file__).resolve().parent
EXP = SRC.parent
ROOT = EXP.parents[1]

CODE_FILES = [
    SRC / "vision_data.py",
    SRC / "vision_predictor.py",
    SRC / "monitoring_workflow.py",
    SRC / "hyperops.py",
    SRC / "vision_runs.py",
    EXP / "scripts" / "s13_visual_asset_audit.py",
    EXP / "scripts" / "s14_visual_run_cache.py",
    EXP / "scripts" / "s15_visual_alarm_analysis.py",
    EXP / "scripts" / "s16_visual_alarm_report.py",
]


# ----------------------------------------------------------------------
# config helpers
# ----------------------------------------------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def sha256_json(obj):
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def config_hash(cfg):
    """Hash only protocol fields; runtime-only keys start with '_'."""
    clean = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    return sha256_json(clean)


def code_hashes():
    out = {}
    for p in CODE_FILES:
        if p.exists():
            out[str(p.relative_to(ROOT)).replace("\\", "/")] = \
                vd.sha256_file(p)
        else:
            out[str(p.relative_to(ROOT)).replace("\\", "/")] = "MISSING"
    return out


def environment():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "sklearn": _safe_version("sklearn"),
        "torch": _safe_version("torch"),
    }


def _safe_version(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def seed_for(cfg, dataset_index, batch, role, index):
    base = int(cfg["seed_base"])
    role_code = int(cfg["roles"][role])
    return base + int(dataset_index) * 100000 + int(batch) * 10000 + \
        role_code * 1000 + int(index)


def noise_seed(cfg, dataset_index, batch, index):
    ss = np.random.SeedSequence(
        [int(cfg.get("noise_seed_base", 20261023)),
         int(dataset_index), int(batch), int(index)])
    return int(ss.generate_state(1)[0])


def stage_dataset_dir(cfg, stage, dataset, batch=None):
    root = Path(cfg.get("_root", ROOT)) / cfg["output_root"] / stage / dataset
    if batch is not None:
        root = root / f"batch{int(batch):02d}"
    return root


def run_id(dataset, batch, role, index):
    return f"{dataset}_b{int(batch):02d}_{role}_{int(index):03d}"


def selected_config(base, selection):
    """Return a deep-ish copy with development selections applied."""
    import copy
    cfg = copy.deepcopy(base)
    pred = cfg["predictor"]
    pred["epochs"] = int(selection["epochs"])
    pred["lr"] = float(selection.get("learning_rate", pred["lr"]))
    cfg["application"]["lambda"] = float(selection["lambda"])
    return cfg


# ----------------------------------------------------------------------
# data context
# ----------------------------------------------------------------------
def build_context(cfg, dataset, center_path=None):
    data = vd.load_vision_dataset(dataset, root=ROOT, verify=True)
    masks = vd.stratified_masks(data.labels, cfg["split"]["seed"],
                                tuple(cfg["split"]["fractions"]))
    std = vd.fit_standardizer(data.X, masks["train"])
    X_std = vd.standardize(data.X, std)
    op = mw.operator_for(len(data.labels), data.edges,
                         kappa=float(cfg["monitoring"]["kappa"]))
    S_all = mw.all_active_operator(op)
    G = np.asarray(S_all @ X_std)
    center = None
    ref_active_median = None
    m_ref = None
    if center_path is not None and Path(center_path).exists():
        with np.load(center_path, allow_pickle=False) as z:
            center = np.asarray(z["center"], dtype=np.float64)
            ref_active_median = float(z["active_median"])
            m_ref = float(z["m_ref"])
    return {
        "data": data, "masks": masks, "std": std, "X_std": X_std,
        "op": op, "S_all": S_all, "G": G,
        "center": center, "ref_active_median": ref_active_median,
        "m_ref": m_ref,
        "split_hash": vd.split_fingerprint(masks),
        "n_classes": int(data.n_classes),
        "data_index": int(cfg["datasets"][dataset]["dataset_index"]),
    }


def _save_npz_atomic(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _failure(path_dir, rid, exc, extra=None):
    rec = {"run_id": rid, "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "error": f"{type(exc).__name__}: {exc}",
           "traceback": traceback.format_exc()}
    if extra:
        rec.update(extra)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    _write_json_atomic(Path(path_dir) / f"{rid}.failed.{stamp}.json", rec)
    return rec


# ----------------------------------------------------------------------
# one run
# ----------------------------------------------------------------------
def train_one(ctx, cfg, seed, timeout_seconds):
    pred_cfg = vp.PredictorConfig.from_dict(cfg["predictor"])
    deadline = time.time() + float(timeout_seconds)
    out = vp.train_predictor(
        ctx["G"], ctx["S_all"], ctx["data"].labels,
        ctx["masks"]["train"], ctx["masks"]["val"], ctx["n_classes"],
        seed, pred_cfg, epoch_limits=(pred_cfg.epochs,), deadline=deadline)
    return out["checkpoints"][pred_cfg.epochs]


def _score_y(op, Y, P, center, train_mask, ref_active_median):
    flags = op.flags_from_Y(Y)
    s = op.S_data(flags)
    T, resid = ho.spectral_norm_entries(op, s, center)
    return flags, s, {
        "T": float(T), "T_resid": float(resid),
        "T_approx": bool(not math.isfinite(resid) or resid > 1e-5),
        "frob": float(np.linalg.norm(s - center)),
        "entropy": mw.max_entropy(P, ~np.asarray(train_mask, dtype=bool)),
        "active_fraction": float(np.mean(flags)) if op.m else 0.0,
        "active_dev": abs(float(np.mean(flags)) - ref_active_median)
        if op.m else 0.0,
    }


def compute_run(ctx, cfg, role, batch, index, seed, out_dir, config_hash,
                timeout_seconds):
    pred_cfg = vp.PredictorConfig.from_dict(cfg["predictor"])
    labels = ctx["data"].labels
    test_mask = ctx["masks"]["test"]
    train_mask = ctx["masks"]["train"]
    rid = run_id(ctx["data"].name, batch, role, index)
    out_dir = Path(out_dir)
    t_start = time.time()

    ck = train_one(ctx, cfg, seed, timeout_seconds)
    if ck.get("status") != "ok" or "W1" not in ck:
        raise RuntimeError(f"training status {ck.get('status')}")

    W1, W2 = ck["W1"], ck["W2"]
    probs, Y = vp.predict_full(ctx["G"], ctx["S_all"], W1, W2)
    base = {
        "seed": np.int64(seed), "role": role, "batch": np.int64(batch),
        "index": np.int64(index), "epoch": np.int64(ck["epoch"]),
        "val_loss": np.float64(ck["val_loss"]),
        "val_acc": np.float64(ck["val_acc"]),
        "train_loss": np.float64(ck["train_loss"]),
        "train_seconds": np.float64(ck["train_seconds"]),
        "config_hash": config_hash,
        "dataset": ctx["data"].name,
        "split_hash": ctx["split_hash"],
        "W1": W1.astype(np.float32), "W2": W2.astype(np.float32),
        "Y": Y.astype(np.uint8), "P": probs.astype(np.float16),
        "test_acc": np.float64(mw.accuracy(Y, labels, test_mask)),
        "test_ce": np.float64(mw.cross_entropy(probs, labels, test_mask)),
    }

    if role in ("ref", "cal"):
        if role == "cal":
            if ctx["center"] is None:
                raise RuntimeError("calibration run requires reference center")
            t_score = time.time()
            _flags, _s, sc = _score_y(
                ctx["op"], Y, probs, ctx["center"], train_mask,
                ctx["ref_active_median"])
            base["score_seconds"] = np.float64(time.time() - t_score)
            for k, v in sc.items():
                base[k] = v
        try:
            import psutil
            base["worker_rss_mb"] = np.float64(
                psutil.Process().memory_info().rss / (1024 ** 2))
        except Exception:
            base["worker_rss_mb"] = np.float64(np.nan)
        base["run_seconds"] = np.float64(time.time() - t_start)
        _save_npz_atomic(out_dir / f"{rid}.npz", base)
        return {"run_id": rid, "status": "ok", "seconds":
                time.time() - t_start, "role": role}

    if role != "eval":
        raise ValueError(f"unknown role {role}")

    if ctx["center"] is None:
        raise RuntimeError("evaluation run requires reference center")
    sigmas = np.asarray(cfg["stress"]["sigmas"], dtype=np.float64)
    S = len(sigmas)
    n = len(labels)
    lam = float(cfg["application"]["lambda"])
    rng = np.random.default_rng(noise_seed(cfg, ctx["data_index"], batch, index))
    Z = vd.noise_matrix(n, ctx["data"].X.shape[1], train_mask,
                        ctx["std"]["zero_var"], rng)
    GZ = np.asarray(ctx["S_all"] @ Z)

    Y_base = np.empty((S, n), dtype=np.uint8)
    Y_ref = np.empty((S, n), dtype=np.uint8)
    acc_base = np.empty(S); acc_ref = np.empty(S)
    ce_base = np.empty(S); ce_ref = np.empty(S); chg = np.empty(S)
    T = np.empty(S); frob = np.empty(S); ent = np.empty(S)
    act = np.empty(S); act_dev = np.empty(S)
    resid = np.empty(S); approx = np.zeros(S, dtype=bool)
    t_inf = np.empty(S); t_score = np.empty(S); t_ref = np.empty(S)

    clean_pred = Y
    for si, sigma in enumerate(sigmas):
        t0 = time.time()
        if sigma == 0.0:
            P_s, Y_s = probs, clean_pred
        else:
            G_s = ctx["G"] + float(sigma) * GZ
            P_s, Y_s = vp.predict_full(G_s, ctx["S_all"], W1, W2)
        t_inf[si] = time.time() - t0

        t0 = time.time()
        flags, s_data, sc = _score_y(
            ctx["op"], Y_s, P_s, ctx["center"], train_mask,
            ctx["ref_active_median"])
        t_score[si] = time.time() - t0

        t0 = time.time()
        S_Y = ctx["op"].csr(s_data)      # exactly the scored operator
        Q = mw.refine_probs(P_s, S_Y, lam)
        Y_r = np.argmax(Q, axis=1).astype(np.int64)
        t_ref[si] = time.time() - t0

        Y_base[si] = Y_s.astype(np.uint8)
        Y_ref[si] = Y_r.astype(np.uint8)
        acc_base[si] = mw.accuracy(Y_s, labels, test_mask)
        acc_ref[si] = mw.accuracy(Y_r, labels, test_mask)
        ce_base[si] = mw.cross_entropy(P_s, labels, test_mask)
        ce_ref[si] = mw.cross_entropy(Q, labels, test_mask)
        chg[si] = mw.change_fraction(Y_s, clean_pred, test_mask)
        T[si] = sc["T"]; frob[si] = sc["frob"]; ent[si] = sc["entropy"]
        act[si] = sc["active_fraction"]; act_dev[si] = sc["active_dev"]
        resid[si] = sc["T_resid"]; approx[si] = sc["T_approx"]

    base.update({
        "sigmas": sigmas, "Y_base": Y_base, "Y_ref": Y_ref,
        "acc_base": acc_base, "acc_ref": acc_ref,
        "ce_base": ce_base, "ce_ref": ce_ref, "change_fraction": chg,
        "T": T, "frob": frob, "entropy": ent,
        "active_fraction": act, "active_dev": act_dev,
        "T_resid": resid, "T_approx": approx,
        "infer_seconds": t_inf, "score_seconds": t_score,
        "refine_seconds": t_ref,
        "lambda": np.float64(lam),
        "run_seconds": np.float64(time.time() - t_start),
    })
    try:
        import psutil
        base["worker_rss_mb"] = np.float64(
            psutil.Process().memory_info().rss / (1024 ** 2))
    except Exception:
        base["worker_rss_mb"] = np.float64(np.nan)
    _save_npz_atomic(out_dir / f"{rid}.npz", base)
    return {"run_id": rid, "status": "ok", "seconds": time.time() - t_start,
            "role": role}


# ----------------------------------------------------------------------
# multiprocessing worker
# ----------------------------------------------------------------------
_W = {}


def _worker_init(dataset, cfg, config_hash, center_path):
    # Single-threaded BLAS per worker; outer parallelism is process-level.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    ctx = build_context(cfg, dataset, center_path=center_path)
    ctx["data_index"] = int(cfg["datasets"][dataset]["dataset_index"])
    _W.update({"ctx": ctx, "cfg": cfg, "config_hash": config_hash})


def _worker_task(task):
    role, batch, index, seed, out_dir = task
    try:
        return compute_run(
            _W["ctx"], _W["cfg"], role, batch, index, seed, out_dir,
            _W["config_hash"], _W["cfg"].get("timeout_seconds_per_run", 900))
    except Exception as exc:  # noqa: BLE001 - recorded, never silently dropped
        rid = run_id(_W["ctx"]["data"].name, batch, role, index)
        rec = _failure(out_dir, rid, exc)
        return {"run_id": rid, "status": "failed", "error": rec["error"],
                "role": role}


def _existing_ok(path, config_hash):
    if not Path(path).exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            return str(z["config_hash"]) == config_hash
    except Exception:
        return False


def run_role(cfg, dataset, role, batch, count, out_dir, config_hash,
             center_path=None, jobs=1):
    """Run `count` independent runs, skipping complete matching cache files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = int(cfg["datasets"][dataset]["dataset_index"])
    tasks = []
    for i in range(int(count)):
        rid = run_id(dataset, batch, role, i)
        if _existing_ok(out_dir / f"{rid}.npz", config_hash):
            continue
        tasks.append((role, int(batch), i,
                      seed_for(cfg, idx, batch, role, i), str(out_dir)))
    done_before = int(count) - len(tasks)
    records = []
    t0 = time.time()
    if tasks:
        if jobs <= 1:
            _worker_init(dataset, cfg, config_hash, center_path)
            for task in tasks:
                records.append(_worker_task(task))
        else:
            import multiprocessing as mp
            ctxmp = mp.get_context("spawn")
            with ctxmp.Pool(processes=int(jobs),
                            initializer=_worker_init,
                            initargs=(dataset, cfg, config_hash,
                                      center_path)) as pool:
                for rec in pool.imap_unordered(_worker_task, tasks):
                    records.append(rec)
                    print(f"  {rec.get('run_id')}: {rec.get('status')} "
                          f"({rec.get('seconds', 0):.1f}s)", flush=True)
    summary = {
        "dataset": dataset, "role": role, "batch": int(batch),
        "count": int(count), "skipped_complete": done_before,
        "attempted": len(tasks),
        "ok": int(sum(r["status"] == "ok" for r in records)),
        "failed": int(sum(r["status"] == "failed" for r in records)),
        "seconds": time.time() - t0,
        "records": records,
    }
    print(f"[{dataset} b{batch} {role}] {summary['ok']} ok, "
          f"{summary['failed']} failed, {summary['skipped_complete']} cached, "
          f"{summary['seconds']:.1f}s", flush=True)
    return summary


def reference_path(cfg, stage, dataset, batch=None):
    return stage_dataset_dir(cfg, stage, dataset, batch) / \
        "reference_summary.npz"


def build_reference_summary(cfg, dataset, batch, out_dir, n_ref, config_hash,
                            summary_path=None):
    """Center, active-edge median and reference T scores from reference runs."""
    ctx = build_context(cfg, dataset)
    op = ctx["op"]
    Ys, active = [], []
    times = []
    for i in range(int(n_ref)):
        rid = run_id(dataset, batch, "ref", i)
        path = Path(out_dir) / f"{rid}.npz"
        if not _existing_ok(path, config_hash):
            raise RuntimeError(f"reference run missing/incompatible: {rid}")
        with np.load(path, allow_pickle=False) as z:
            Ys.append(np.asarray(z["Y"], dtype=np.int64))
            times.append(float(z["train_seconds"]))
    center = mw.fit_center(op, Ys)
    for Y in Ys:
        active.append(float(np.mean(op.flags_from_Y(Y))))
    active_median = float(np.median(active))
    T_ref = np.empty(len(Ys))
    t0 = time.time()
    for i, Y in enumerate(Ys):
        s = op.S_data(op.flags_from_Y(Y))
        T_ref[i], _ = ho.spectral_norm_entries(op, s, center)
    m_ref = float(np.median(T_ref))
    path = Path(summary_path) if summary_path is not None else         Path(out_dir).parent / "reference_summary.npz"
    _save_npz_atomic(path, {
        "center": center, "active_median": np.float64(active_median),
        "m_ref": np.float64(m_ref), "T_ref": T_ref,
        "n_ref": np.int64(n_ref), "config_hash": config_hash,
        "train_seconds_median": np.float64(np.median(times)),
        "score_seconds": np.float64(time.time() - t0),
    })
    return {"center": center, "active_median": active_median,
            "m_ref": m_ref, "T_ref": T_ref, "path": str(path),
            "score_seconds": _read_scalar(path, "score_seconds")}


def _read_scalar(path, key):
    with np.load(path, allow_pickle=False) as z:
        return float(z[key])


def load_reference_summary(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
