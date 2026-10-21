"""S15b: internal threshold-stability and conditional-risk diagnostics.

Read-only: uses the existing 99 calibration and 200 evaluation scores of each
dataset under the frozen center.  It does NOT create new runs and is not a
new confirmation.

1. Repeated random 99/200 splits of the pooled 299 scores: distribution of
   the rank threshold and the realized clean exceedance rate.
2. Uniform decimal rounding of calibration and evaluation scores.
3. A PAC/conditional-risk threshold with a declared binomial target
   (Vovk 2012, Proposition 2b), evaluated post hoc on the old evaluation set.
   The rule is exploratory here; it was not frozen before seeing results.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

import hyperops as ho  # noqa: E402
import vision_runs as vr  # noqa: E402

ROOT = HERE / "results" / "visual_alarm_v1"


def _load_T(ds):
    base = ROOT / "formal" / ds / "batch01"
    cal = np.array([np.load(p, allow_pickle=False)["T"]
                    for p in sorted((base / "cal").glob("*.npz"))])
    ev = np.array([np.load(p, allow_pickle=False)["T"][0]
                   for p in sorted((base / "eval").glob("*.npz"))])
    return cal, ev


def _summ(a, prefix):
    a = np.asarray(a, dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(a)),
        f"{prefix}_sd": float(np.std(a, ddof=1)),
        f"{prefix}_median": float(np.median(a)),
        f"{prefix}_p2_5": float(np.percentile(a, 2.5)),
        f"{prefix}_p97_5": float(np.percentile(a, 97.5)),
        f"{prefix}_min": float(np.min(a)),
        f"{prefix}_max": float(np.max(a)),
    }


def pac_k(M, alpha, delta):
    """Smallest k with P(Bin(M, 1-alpha) >= k) <= delta; q is the k-th smallest."""
    for k in range(1, M + 1):
        if binom.sf(k - 1, M, 1.0 - alpha) <= delta:
            return k
    return M


def analyze(ds, reps, seed, alpha, delta):
    cal, ev = _load_T(ds)
    M, N = len(cal), len(ev)
    q = ho.rank_threshold(cal, alpha)
    original_count = int(np.sum(ev > q))
    pooled = np.concatenate([cal, ev])
    rng = np.random.default_rng(seed)

    fprs = np.empty(reps)
    qs = np.empty(reps)
    for r in range(reps):
        idx = rng.permutation(len(pooled))
        q_r = ho.rank_threshold(pooled[idx[:M]], alpha)
        qs[r] = q_r
        fprs[r] = float(np.mean(pooled[idx[M:]] > q_r))

    precision = []
    for dec in (6, 8, 10, 12, 14):
        c = np.round(cal, dec)
        e = np.round(ev, dec)
        q_d = ho.rank_threshold(c, alpha)
        precision.append({
            "decimals": dec, "threshold": float(q_d),
            "count": int(np.sum(e > q_d)),
            "rate": float(np.mean(e > q_d)),
        })

    k = pac_k(M, alpha, delta)
    q_pac = float(np.sort(cal)[k - 1])
    pac_count = int(np.sum(ev > q_pac))
    k_table = []
    for m in (99, 199, 499, 999):
        kk = pac_k(m, alpha, delta)
        k_table.append({
            "M": m, "k": kk, "calibration_exceedances_allowed": m - kk,
            "tail_probability": float(binom.sf(kk - 1, m, 1.0 - alpha)),
        })

    # Pooled-score gap above the original threshold: how far a threshold jump
    # must move to change the number of calibration exceedances.
    above = np.sort(pooled[pooled > q])
    gap = float(above[0] - q) if len(above) else math.inf

    out = {
        "dataset": ds, "n_cal": M, "n_eval": N,
        "alpha": alpha, "delta": delta,
        "seed": seed, "reps": reps,
        "original": {"threshold": float(q), "count": original_count,
                     "rate": float(original_count / N)},
        "precision_sensitivity": precision,
        "repeated_split": {
            **_summ(fprs, "fpr"), **_summ(qs, "q"),
            "fpr_frac_gt_10": float(np.mean(fprs > 0.10)),
            "fpr_frac_gt_15": float(np.mean(fprs > 0.15)),
            "fpr_frac_gt_20": float(np.mean(fprs > 0.20)),
            "unique_thresholds": int(len(np.unique(qs))),
            "frac_q_below_original": float(np.mean(qs < q)),
            "frac_q_above_original": float(np.mean(qs > q)),
            "pooled_gap_above_original_threshold": gap,
            "note": ("repeated 99/200 splits of the same 299 existing runs; "
                     "internal split-sensitivity only, not new independent "
                     "runs or a new population-level confirmation"),
        },
        "pac_rule": {
            "k": k, "threshold": q_pac, "count": pac_count,
            "rate": pac_count / N,
            "calibration_exceedances": M - k,
            "tail_probability": float(binom.sf(k - 1, M, 1.0 - alpha)),
            "note": ("Vovk-type tolerance rule evaluated post hoc on the old "
                     "evaluation set; exploratory, not a pre-registered fix, "
                     "and not evidence that the old 14% result is repaired"),
        },
        "pac_k_table": k_table,
        "caveat": ("All numbers condition on the frozen reference center and "
                   "the existing 299 runs. They do not establish a new "
                   "independent confirmation or identify a unique cause for "
                   "the original 28/200."),
    }
    path = ROOT / "analysis" / f"threshold_stability_{ds}.json"
    vr._write_json_atomic(path, out)
    return out, fprs, qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    args = ap.parse_args()
    for ds in ("ntu2012", "modelnet40"):
        out, fprs, qs = analyze(ds, args.reps, args.seed, args.alpha,
                                args.delta)
        vr._save_npz_atomic(
            ROOT / "analysis" / f"threshold_stability_{ds}_splits.npz",
            {"fpr": fprs, "q": qs})
        print(json.dumps({
            "dataset": ds,
            "original": out["original"],
            "repeated_fpr": {k: out["repeated_split"][k] for k in
                             ("fpr_mean", "fpr_median", "fpr_p2_5",
                              "fpr_p97_5", "fpr_frac_gt_15")},
            "threshold_p2_5_p97_5": (out["repeated_split"]["q_p2_5"],
                                     out["repeated_split"]["q_p97_5"]),
            "pac_rule": out["pac_rule"],
            "precision_sensitivity": out["precision_sensitivity"],
        }, indent=2))


if __name__ == "__main__":
    main()
