"""Reconstruct defensible summaries without changing historical measurements.

C1 has three independently fitted centers/thresholds. Conditional on all of
them, evaluation indicators are independent but need not be identically
distributed. A Hoeffding interval covers their mean conditional probability;
a pooled binomial Clopper-Pearson interval does not have that justification.
Single-batch CP intervals concern one realized threshold, not marginal rank
coverage over a new calibration experiment.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
import hyperops as ho


def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def observed_count(rate, n):
    if n <= 0 or not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("rate and sample size must define binomial counts")
    raw = rate * n
    count = round(raw)
    if not math.isclose(raw, count, abs_tol=1e-7, rel_tol=0):
        raise ValueError("saved rate does not represent an integer count")
    return count


def summarize_confirmation(rows, confidence=0.95):
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie between zero and one")
    out = []
    for row in rows:
        rate, n = float(row["estimate"]), int(row["n"])
        count = observed_count(rate, n)
        batches = int(row["batches"])
        if not 1 <= batches <= n:
            raise ValueError("invalid number of batches")
        if batches > 1:
            radius = math.sqrt(math.log(2 / (1 - confidence)) / (2 * n))
            low, high = max(0.0, rate - radius), min(1.0, rate + radius)
            method = "conditional_hoeffding"
            estimand = "mean_conditional_exceedance_across_saved_batches"
        else:
            low, high = ho.clopper_pearson(count, n, confidence)
            method = "conditional_clopper_pearson"
            estimand = "exceedance_given_saved_reference_and_calibration"
        out.append(dict(claim=row["claim"], structure=row["structure"],
                        metric=row["metric"], count=count, n=n,
                        estimate=rate, batches=batches, ci_lo=low, ci_hi=high,
                        confidence=confidence, interval_method=method,
                        estimand=estimand, historical_cp_valid=batches == 1))
    return out


def build_audit(results):
    confirmation = summarize_confirmation(read_rows(results / "confirm_results.csv"))
    fixed = []
    for row in read_rows(results / "l3_fixed_split.csv"):
        # Historical S11 did not save n_eval, but its frozen script specifies
        # 60 evaluation runs on Cora and 40 on DBLP. Record this provenance.
        n = {"cora-cc": 60, "dblp-ca": 40}[row["tag"]]
        count = observed_count(float(row["exceed_on_eval_runs"]), n)
        low, high = ho.clopper_pearson(count, n)
        if not (math.isclose(low, float(row["cp_lo"]), abs_tol=1e-10)
                and math.isclose(high, float(row["cp_hi"]), abs_tol=1e-10)):
            raise ValueError("S11 stored interval differs from reconstructed counts")
        fixed.append(dict(structure=row["tag"], arm=row["arm"],
                          alpha=float(row["alpha"]), count=count, n=n,
                          estimate=count/n, ci_lo=low, ci_hi=high,
                          interval_method="conditional_clopper_pearson",
                          nominal_inside_interval=low <= float(row["alpha"]) <= high,
                          n_provenance="s11_fixed_split_init.py frozen budgets"))
    c2 = {r["metric"]: r for r in confirmation if r["claim"].startswith("C2_")}
    inf, norm = c2["M2_hubs_power_inf"], c2["M2_hubs_power_norm"]
    # Independent evaluation draws were used for norm and directional scores.
    # This bounded-variable interval requires no unsupported pairing.
    difference = inf["estimate"] - norm["estimate"]
    radius = math.sqrt(math.log(40) * (1/inf["n"] + 1/norm["n"]) / 2)
    return dict(
        schema_version=1,
        mainline="normalized influence -> law-specific rank calibration -> real-run mismatch diagnosis",
        confirmation=confirmation, fixed_split=fixed,
        c2_comparison=dict(difference=difference, ci_lo=max(-1, difference-radius),
                           ci_hi=min(1, difference+radius),
                           interval_method="unpaired_hoeffding_95",
                           scope="descriptive direction comparison; no new normal-FPR check",
                           confirmed_superiority=False),
        scope_limits=[
            "C1 estimates conditional rates for three saved thresholds, not exact equality to alpha.",
            "Rank validity is marginal over calibration and a new exchangeable observation, conditional on the reference stage.",
            "C2 lacks paired evaluations and a fresh normal-mechanism check; it does not confirm directional superiority.",
            "S12 contains no new L3 run-pool experiment and no timestamped pre-run protocol artifact.",
            "Historical S10a resid_frac_node is a scalar norm-gap lower bound, not the matrix remainder.",
            "Historical numerical residuals/raw score arrays were not saved comprehensively; claims of unaffected DBLP scores are unverified."
        ],
        source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(results.iterdir())
                       if p.suffix in {".csv", ".json"} and p.name != "evidence_audit.json"})


def main():
    result = build_audit(HERE / "results")
    target = HERE / "results" / "evidence_audit.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print("wrote", target)
    for row in result["confirmation"]:
        print(f"{row['metric']}: {row['count']}/{row['n']} "
              f"[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}] "
              f"{row['interval_method']}")


if __name__ == "__main__":
    main()
