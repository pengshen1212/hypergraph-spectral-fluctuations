"""Development-only diagnostic: does any refinement lambda change predictions?

This is NOT part of the formal confirmation cache.  It loads the ten saved
development runs per dataset, applies the five frozen noise strengths to
validation/test features, and reports validation accuracy plus the fraction of
validation predictions changed by each lambda.  It exists to document why the
formal application experiment is inert at lambda=0.

Writes results/visual_alarm_v1/development/lambda_inertia_<dataset>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

import monitoring_workflow as mw  # noqa: E402
import vision_data as vd  # noqa: E402
import vision_predictor as vp  # noqa: E402
import vision_runs as vr  # noqa: E402

ROOT = HERE / "results" / "visual_alarm_v1"


def one(ds, cfg, selection):
    eff = vr.selected_config(cfg, selection)
    ctx = vr.build_context(eff, ds)
    data, masks = ctx["data"], ctx["masks"]
    val = masks["val"]
    lams = [float(v) for v in eff["application"]["lambda_candidates"]]
    sigmas = [float(v) for v in eff["stress"]["sigmas"]]
    rows = {lam: {s: {"acc": [], "change": [], "n_changed": []}
                  for s in sigmas} for lam in lams}
    base_accs = {s: [] for s in sigmas}
    for i in range(int(eff["budgets"]["dev"]["runs_per_dataset"])):
        path = ROOT / "development" / ds / f"run_{i:02d}.npz"
        z = np.load(path, allow_pickle=False)
        W1 = z["W1"].astype(np.float64)
        W2 = z["W2"].astype(np.float64)
        seed = int(z["seed"])
        rng = np.random.default_rng(vr.noise_seed(
            eff, ctx["data_index"], 0, i))
        Z = vd.noise_matrix(len(data.labels), data.X.shape[1],
                            masks["train"], ctx["std"]["zero_var"], rng)
        GZ = np.asarray(ctx["S_all"] @ Z)
        for sigma in sigmas:
            G = ctx["G"] if sigma == 0 else ctx["G"] + sigma * GZ
            P, Y = vp.predict_full(G, ctx["S_all"], W1, W2)
            S_Y = ctx["op"].S_csr(ctx["op"].flags_from_Y(Y))
            base_acc = float(np.mean(Y[val] == data.labels[val]))
            base_accs[sigma].append(base_acc)
            for lam in lams:
                Q = mw.refine_probs(P, S_Y, lam)
                Yr = np.argmax(Q, axis=1)
                rows[lam][sigma]["acc"].append(
                    float(np.mean(Yr[val] == data.labels[val])))
                changed = Yr[val] != Y[val]
                rows[lam][sigma]["change"].append(float(np.mean(changed)))
                rows[lam][sigma]["n_changed"].append(int(changed.sum()))
    out = {"dataset": ds, "seed_note": "development runs only",
           "lambdas": lams, "sigmas": sigmas,
           "n_runs": int(eff["budgets"]["dev"]["runs_per_dataset"]),
           "n_val": int(val.sum()), "rows": {}}
    for lam in lams:
        out["rows"][str(lam)] = {}
        for s in sigmas:
            a = rows[lam][s]["acc"]
            c = rows[lam][s]["change"]
            out["rows"][str(lam)][str(s)] = {
                "mean_val_acc": float(np.mean(a)),
                "mean_val_change_fraction": float(np.mean(c)),
                "max_val_change_fraction": float(np.max(c)),
                "max_val_changed_nodes": int(np.max(rows[lam][s]["n_changed"])),
                "base_val_acc": float(np.mean(base_accs[s])),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True,
                    choices=["ntu2012", "modelnet40"])
    args = ap.parse_args()
    cfg = vr.load_json(args.config)
    cfg["_root"] = str(vr.ROOT)
    sel = vr.load_json(ROOT / "development" / f"selection_{args.dataset}.json")
    out = one(args.dataset, cfg, sel)
    path = ROOT / "development" / f"lambda_inertia_{args.dataset}.json"
    vr._write_json_atomic(path, out)
    lam1 = out["rows"]["1.0"]
    print(json.dumps({
        "dataset": args.dataset, "path": str(path),
        "lambda_1_change_by_sigma": {
            s: lam1[s]["mean_val_change_fraction"] for s in lam1},
        "lambda_1_acc_by_sigma": {
            s: lam1[s]["mean_val_acc"] for s in lam1},
    }, indent=2))


if __name__ == "__main__":
    main()
