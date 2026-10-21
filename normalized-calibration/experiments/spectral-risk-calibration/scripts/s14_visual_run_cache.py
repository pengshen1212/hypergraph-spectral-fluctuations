"""S14: independent vision run cache, protocol freeze and formal execution.

Modes
-----
freeze  Combine the base protocol, both development selections, both pilot
        reports and the pre-registered budget decision into
        results/visual_alarm_v1/protocol_frozen.json (config + hashes + seeds
        + environment + code hashes).  Formal runs are refused without it.
pilot   Optional standalone pilot runner (same cache as s13 --mode pilot).
formal  Reference -> calibration -> evaluation runs for the selected budget.
        Every cache file is atomic; resume skips only complete files whose
        embedded config hash matches.  Different configs are never mixed.

Formal order is NTU2012 batch 1..B, then ModelNet40 batch 1..B.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

import vision_data as vd  # noqa: E402
import vision_runs as vr  # noqa: E402

RESULT_ROOT = HERE / "results" / "visual_alarm_v1"
FROZEN = RESULT_ROOT / "protocol_frozen.json"


def _load_selection(dataset):
    p = RESULT_ROOT / "development" / f"selection_{dataset}.json"
    if not p.exists():
        raise SystemExit(f"missing development selection: {p}")
    sel = json.load(open(p, encoding="utf-8"))
    if sel.get("dataset") != dataset:
        raise SystemExit(f"selection dataset mismatch in {p}")
    return sel


def _load_pilot(dataset):
    p = RESULT_ROOT / "pilot" / f"timing_{dataset}.json"
    if not p.exists():
        raise SystemExit(f"missing pilot timing: {p}")
    return json.load(open(p, encoding="utf-8"))


def freeze(base_cfg, base_path):
    if FROZEN.exists():
        raise SystemExit(f"protocol already frozen: {FROZEN}")
    decision = vr.load_json(RESULT_ROOT / "budget_decision.json")
    budget_name = decision["budget_selected"]
    if budget_name not in ("formal_full", "formal_resource"):
        raise SystemExit(f"invalid budget decision {budget_name}")

    datasets = list(base_cfg["datasets"])
    configs, hashes, selections, pilots = {}, {}, {}, {}
    for ds in datasets:
        sel = _load_selection(ds)
        if sel.get("epochs") is None or sel.get("lambda") is None:
            raise SystemExit(f"{ds}: development selection incomplete")
        eff = vr.selected_config(base_cfg, sel)
        configs[ds] = eff
        hashes[ds] = vr.config_hash(eff)
        selections[ds] = sel
        pilots[ds] = _load_pilot(ds)

    for ds in datasets:
        data = vd.load_vision_dataset(ds, root=vr.ROOT, verify=True)
        spec = base_cfg["datasets"][ds]
        if data.content_sha256 != spec["content_sha256"] or \
                data.edges_sha256 != spec["edges_sha256"]:
            raise SystemExit(f"{ds}: data hash changed before freeze")

    payload = {
        "protocol_version": base_cfg["protocol_version"],
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base_config_path": str(Path(base_path).resolve()),
        "base_config_sha256": vd.sha256_file(base_path),
        "configs": configs,
        "config_sha256": hashes,
        "development_selections": selections,
        "pilot_reports": pilots,
        "budget_decision": decision,
        "budget_selected": budget_name,
        "budget_counts": base_cfg["budgets"][budget_name],
        "dataset_order": datasets,
        "data_hashes": {
            ds: {"content": base_cfg["datasets"][ds]["content_sha256"],
                 "edges": base_cfg["datasets"][ds]["edges_sha256"]}
            for ds in datasets},
        "code_sha256": vr.code_hashes(),
        "environment": vr.environment(),
        "seed_base": base_cfg["seed_base"],
        "seed_formula": ("seed_base + dataset_index*100000 + batch*10000 "
                         "+ role_code*1000 + run_index"),
        "noise_seed_base": base_cfg["noise_seed_base"],
        "statistics": {
            "bootstrap_reps": base_cfg["application"]["bootstrap_reps"],
            "bootstrap_seed": base_cfg["application"]["bootstrap_seed"],
            "paired_unit": "run_id (all strengths and methods move together)",
            "batch_specific_thresholds": True,
            "mixed_batch_binomial_interval": False,
            "holm_family": base_cfg["application"]["holm_family"],
        },
        "note": ("Frozen before formal reference/calibration/evaluation "
                 "generation. Existing prior visual results are not reused."),
    }
    vr._write_json_atomic(FROZEN, payload)
    print(json.dumps({k: payload[k] for k in
                      ("frozen_at", "budget_selected", "config_sha256")},
                     indent=2), flush=True)
    return payload


def _summary_usable(path, config_hash, n_ref):
    if not Path(path).exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if str(z["config_hash"]) != config_hash:
                return None
            if int(z["n_ref"]) != int(n_ref):
                return None
            return {k: z[k] for k in z.files}
    except Exception:
        return None


def run_stage(cfg, stage, dataset, batch, counts, jobs):
    ddir = vr.stage_dataset_dir(cfg, stage, dataset, batch)
    config_hash = vr.config_hash(cfg)
    ref_dir, cal_dir, eval_dir = ddir / "ref", ddir / "cal", ddir / "eval"
    print(f"\n=== {stage} {dataset} batch {batch} ===", flush=True)
    print(f"config {config_hash[:16]}  budget {counts}", flush=True)

    s_ref = vr.run_role(cfg, dataset, "ref", batch, counts["ref"], ref_dir,
                        config_hash, center_path=None, jobs=jobs)
    summary_path = ddir / "reference_summary.npz"
    summary = _summary_usable(summary_path, config_hash, counts["ref"])
    if summary is None:
        print("  rebuilding reference summary", flush=True)
        vr.build_reference_summary(cfg, dataset, batch, ref_dir,
                                   counts["ref"], config_hash,
                                   summary_path=summary_path)
    s_cal = vr.run_role(cfg, dataset, "cal", batch, counts["cal"], cal_dir,
                        config_hash, center_path=str(summary_path), jobs=jobs)
    s_eval = vr.run_role(cfg, dataset, "eval", batch, counts["eval"], eval_dir,
                         config_hash, center_path=str(summary_path), jobs=jobs)
    status = {
        "stage": stage, "dataset": dataset, "batch": int(batch),
        "config_sha256": config_hash, "counts": counts,
        "ref": {k: s_ref[k] for k in ("ok", "failed", "skipped_complete",
                                      "seconds")},
        "cal": {k: s_cal[k] for k in ("ok", "failed", "skipped_complete",
                                      "seconds")},
        "eval": {k: s_eval[k] for k in ("ok", "failed", "skipped_complete",
                                        "seconds")},
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    vr._write_json_atomic(ddir / "status.json", status)
    return status


def _check_cache_not_mixed(ddir, config_hash, counts):
    for role, n in (("ref", counts["ref"]), ("cal", counts["cal"]),
                    ("eval", counts["eval"])):
        folder = Path(ddir) / role
        if not folder.exists():
            continue
        files = sorted(folder.glob("*.npz"))
        if len(files) > n:
            raise SystemExit(
                f"{folder}: {len(files)} cache files exceed frozen budget {n}")
        for f in files:
            with np.load(f, allow_pickle=False) as z:
                if str(z["config_hash"]) != config_hash:
                    raise SystemExit(
                        f"{f}: cache config hash differs; refusing to mix")


def formal(base_cfg, jobs, only_dataset=None, only_batch=None):
    if not FROZEN.exists():
        raise SystemExit("formal runs require a frozen protocol "
                         f"({FROZEN} does not exist)")
    frozen = vr.load_json(FROZEN)
    budget_name = frozen["budget_selected"]
    counts = frozen["budget_counts"]
    total_batches = int(counts["batches"])
    order = frozen["dataset_order"]
    if only_dataset:
        if only_dataset not in order:
            raise SystemExit(f"dataset {only_dataset} not in frozen order")
        order = [only_dataset]
    for ds in order:
        cfg = frozen["configs"][ds]
        cfg["_root"] = str(vr.ROOT)
        config_hash = frozen["config_sha256"][ds]
        if vr.config_hash(frozen["configs"][ds]) != config_hash:
            raise SystemExit(f"{ds}: frozen config hash does not verify")
        for k in ("content_sha256", "edges_sha256"):
            if cfg["datasets"][ds][k] != \
                    frozen["data_hashes"][ds][k.replace("_sha256", "")]:
                raise SystemExit(f"{ds}: frozen data hash mismatch for {k}")
        batches = [int(only_batch)] if only_batch else \
            list(range(1, total_batches + 1))
        for b in batches:
            if b < 1 or b > total_batches:
                raise SystemExit(f"batch {b} outside frozen budget")
            ddir = vr.stage_dataset_dir(cfg, "formal", ds, b)
            _check_cache_not_mixed(ddir, config_hash, counts)
            run_stage(cfg, "formal", ds, b, counts, jobs)
    print("\nformal stage complete", flush=True)


def pilot(cfg, dataset, jobs):
    sel = _load_selection(dataset)
    eff = vr.selected_config(cfg, sel)
    eff["_root"] = str(vr.ROOT)
    counts = cfg["budgets"]["pilot"]
    return run_stage(eff, "pilot", dataset, 0, counts, jobs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True,
                    choices=["freeze", "pilot", "formal"])
    ap.add_argument("--dataset", default=None,
                    choices=["ntu2012", "modelnet40"])
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    base = vr.load_json(args.config)
    base["_root"] = str(vr.ROOT)
    if args.stage == "freeze":
        freeze(base, args.config)
    elif args.stage == "pilot":
        if args.dataset is None:
            raise SystemExit("--dataset required for pilot")
        pilot(base, args.dataset, args.jobs)
    else:
        formal(base, args.jobs, args.dataset, args.batch)


if __name__ == "__main__":
    main()
