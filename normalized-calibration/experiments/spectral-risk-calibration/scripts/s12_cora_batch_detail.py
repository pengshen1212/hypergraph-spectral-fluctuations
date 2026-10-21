"""Fresh three-batch Cora node-law calibration rerun with retained counts.

This is a descriptive follow-up using a new deterministic seed namespace.
It was not prospectively registered and does not constitute confirmation.
Results are written separately from the historical aggregate and S12 v2 file.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "src"))

import data_io  # noqa: E402
import hyperops as ho  # noqa: E402

ALPHA = 0.05
SEED_BASE = 20271021
COUNTS = {"reference": 200, "calibration": 500, "evaluation": 1000}
OUT = EXP / "results" / "cora_calibration_batches_20271021.csv"
SUMMARY = EXP / "results" / "cora_calibration_batches_20271021.json"


def run() -> None:
    n, edges, labels = data_io.load_graph("cora-cc")
    op = ho.RelationOperator(n, edges)
    profile = data_io.majority_binary_profile(labels, 0.05)
    sampler = ho.bound_node_sampler(op, profile)
    rows = []
    started = time.time()

    for batch in range(3):
        batch_start = time.time()
        seed = SEED_BASE + 17 * batch
        center = ho.estimate_center(
            op, sampler, COUNTS["reference"], np.random.default_rng(seed))
        calibration, _ = ho.draw_scores(
            op, sampler, COUNTS["calibration"], np.random.default_rng(seed + 1), center)
        evaluation, _ = ho.draw_scores(
            op, sampler, COUNTS["evaluation"], np.random.default_rng(seed + 2), center)
        threshold = ho.rank_threshold(calibration, ALPHA)
        count = int(np.count_nonzero(evaluation > threshold))
        lower, upper = ho.clopper_pearson(count, COUNTS["evaluation"])
        row = {
            "batch": batch + 1,
            "seed": seed,
            "alpha": ALPHA,
            "reference_n": COUNTS["reference"],
            "calibration_n": COUNTS["calibration"],
            "evaluation_n": COUNTS["evaluation"],
            "threshold": float(threshold),
            "exceedance_count": count,
            "exceedance_rate": count / COUNTS["evaluation"],
            "cp95_lower": lower,
            "cp95_upper": upper,
            "elapsed_seconds": round(time.time() - batch_start, 2),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total_n = sum(row["evaluation_n"] for row in rows)
    total_count = sum(row["exceedance_count"] for row in rows)
    summary = {
        "status": "completed",
        "estimand": "equal-weight mean of three batch-specific conditional rates",
        "prospective_registration": False,
        "seed_base": SEED_BASE,
        "batches": rows,
        "aggregate_count": total_count,
        "aggregate_n": total_n,
        "aggregate_rate": total_count / total_n,
        "batch_cp95_endpoint_envelope": [
            min(row["cp95_lower"] for row in rows),
            max(row["cp95_upper"] for row in rows),
        ],
        "elapsed_seconds": round(time.time() - started, 2),
        "csv": OUT.name,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "batches"}, indent=2))


if __name__ == "__main__":
    run()
