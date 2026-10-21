"""Generate manuscript tables from the frozen spectral-risk CSVs, without training."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
from collections import defaultdict

PAPER = Path(__file__).resolve().parents[2]
ROOT = PAPER.parent.parent
RESULTS = ROOT / "experiments/spectral-risk-calibration/results"
OUTPUT = PAPER / "tables"


def read(name):
    with (RESULTS / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def table(rows, columns, heading, caption, label):
    return ("% Generated from frozen results by notes/calibration-integration/generate_tables.py\n"
            "\\begin{table}[htbp]\n\\centering\\small\n"
            "\\setlength{\\tabcolsep}{4pt}\n"
            "\\begin{tabular}{" + columns + "}\n\\toprule\n" + heading +
            " \\\\\n\\midrule\n" + "\n".join(" & ".join(row) + r" \\" for row in rows) +
            "\n\\bottomrule\n\\end{tabular}\n\\caption{" + caption +
            "}\n\\label{" + label + "}\n\\end{table}\n")


def graph_name(value):
    return {"cora-cc": "Cora", "dblp-ca": "DBLP",
            "familyA_coherent": "Coherent", "familyB_flat": "Competition"}[value]


def law_name(profile):
    return {"fixed": "aligned", "gamma=0.05_cancel": r"$\gamma=.05$",
            "gamma=0.5": r"$\gamma=.50$", "gamma=0.95_max": r"$\gamma=.95$",
            "binary-eps0.05": "binary .05", "binary-eps0.15": "binary .15",
            "multiclass-eps0.05": "multi .05", "multiclass-eps0.15": "multi .15",
            "hetero-0.02/0.30": "hetero"}[profile]


def main():
    OUTPUT.mkdir(exist_ok=True)
    normalization = read("normalization_pilot.csv")
    rows = []
    for row in normalization:
        name = graph_name(row["structure"])
        if row["structure"] == "dblp-ca":
            name += r"$^\dagger$"
        rows.append([name, law_name(row["profile"])] +
                    [f"{float(row[key]):.2f}" for key in
                     ("ratio_raw", "ratio_S0", "ratio_S", "mean_shift_dyn")])
    text = table(rows, "llrrrr", r"Structure & Profile & $R_B$ & $R_{S_0}$ & $R_S$ & $\Delta_\mu$",
                 r"Computed mean centered-norm ratios A/B; $\Delta_\mu=\|\widehat{\mathbb E}_A S-\widehat{\mathbb E}_B S\|_2$. "
                 r"Dynamic norms use each arm's own estimated mean; raw and fixed-degree means match analytically. "
                 r"Center/evaluation budgets per arm are 800/600 for the controlled families and Cora, and 120/80 for DBLP. "
                 r"Native binary/multiclass profiles have the displayed error probability; hetero uses binary errors at .02 below the upper degree decile and .30 within it. "
                 r"$\dagger$ DBLP raw/fixed-control scores include numerical approximations; these are point estimates, not certified norm ratios.",
                 "tab:normalization")
    (OUTPUT / "normalization.tex").write_text(text, encoding="utf-8")

    cells = defaultdict(lambda: defaultdict(list))
    for row in read("calibration_baselines.csv"):
        if row["kind"] == "normal" and float(row["alpha"]) == .05:
            cells[(row["structure"], row["profile"])][row["calibrator"]].append(row)
    rows = []
    summaries = []
    for (graph, law), arms in sorted(cells.items()):
        rendered = []
        for arm in ("node", "edge-iid"):
            rs = arms[arm]
            expected_batches = 1 if graph.startswith("family") else 3
            assert len(rs) == expected_batches
            rate = sum(float(r["exceed"]) for r in rs)/len(rs)
            low = min(float(r["cp_lo"]) for r in rs)
            high = max(float(r["cp_hi"]) for r in rs)
            rendered.append(f"{100*rate:.1f} [{100*low:.1f},{100*high:.1f}]")
            summaries.append(dict(graph=graph, law=law, arm=arm, rate=rate,
                                  envelope_lo=low, envelope_hi=high))
        rows.append([graph_name(graph), law_name(law)] + rendered)
    text = table(rows, "llrr", "Structure & Profile & Node (\\%) & Edge-iid (\\%)",
                 r"Known-node-law exceedance at $\alpha=5\%$. Native entries are means over three independent threshold batches; "
                 r"brackets give the minimum lower and maximum upper endpoint of their individual 95\% binomial intervals, "
                 r"not a pooled 95\% interval. Per-batch reference/calibration/evaluation budgets are 200/500/1000 on Cora "
                 r"and 80/150/300 on DBLP. Hetero here uses multiclass errors at .02/.30. The coherent-family row is one batch with 400/800/1600 observations.",
                 "tab:node-calibration")
    (OUTPUT / "node-calibration.tex").write_text(text, encoding="utf-8")

    budget = {("cora-cc", "LP"): (50, 100, 100),
              ("dblp-ca", "LP"): (40, 80, 80),
              ("cora-cc", "proto"): (30, 100, 150),
              ("dblp-ca", "proto"): (20, 50, 80),
              ("cora-cc", "gcn5f"): (20, 40, 60),
              ("cora-cc", "gcn-fixedsplit-init"): (20, 40, 60),
              ("dblp-ca", "gcn-fixedsplit-init"): (15, 25, 40)}
    code = {"LP": "LP", "proto": "Proto", "gcn5f": "GCN-XF",
            "gcn-fixedsplit-init": "GCN-init"}
    grouped = defaultdict(dict)
    for row in read("l3_run_pool.csv") + read("l3_fixed_split.csv"):
        if float(row["alpha"]) == .05:
            key = (row["tag"], row["predictor"])
            assert row["arm"] not in grouped[key]
            grouped[key][row["arm"]] = row
    rows, run_manifest = [], []
    for key in budget:
        arms = grouped[key]
        r, c, e = budget[key]
        run = arms["run-pool"]
        assert int(run["n_runs_train"]) == r+c+e
        assert int(run["n_sim_cal"]) == c
        count = float(run["exceed_on_eval_runs"])*e
        assert abs(count-round(count)) < 1e-8
        rate = 100*float(run["exceed_on_eval_runs"])
        ci = f"[{100*float(run['cp_lo']):.1f},{100*float(run['cp_hi']):.1f}]"
        rows.append([graph_name(key[0]), code[key[1]], f"{r}/{c}/{e}",
                     f"{rate:.1f} {ci}",
                     f"{100*float(arms['node-fit']['exceed_on_eval_runs']):.1f}",
                     f"{100*float(arms['edge-iid']['exceed_on_eval_runs']):.1f}"])
        run_manifest.append(dict(graph=key[0], predictor=key[1], n_ref=r,
                                 n_cal=c, n_eval=e, run_exceedances=round(count)))
    text = table(rows, "lllrrr", r"Data & Mechanism & R/C/E & Run-pool (\%) & Node-fit & Edge-iid",
                 r"Whole-run evaluation at $\alpha=5\%$. R/C/E are the reference/calibration/evaluation run counts. "
                 r"All rates are percentages; run-pool brackets are conditional 95\% binomial intervals for the realized threshold. "
                 r"LP varies labeled subsets, Proto varies a half split, GCN-XF uses five-fold cross-fitting, "
                 r"and GCN-init holds a 20\% labeled subset fixed. Zero node-fit counts do not imply zero population probability; "
                 r"their upper 95\% endpoints range from 2.4\% to 8.8\%. Every 5\% threshold displayed is finite.",
                 "tab:run-calibration")
    (OUTPUT / "run-calibration.tex").write_text(text, encoding="utf-8")

    cora_batches = read("cora_calibration_batches_20271021.csv")
    rows = []
    for row in cora_batches:
        rows.append([str(int(row["batch"])),
                     f"{int(row['exceedance_count'])}/{int(row['evaluation_n'])}",
                     f"{100*float(row['exceedance_rate']):.1f}",
                     f"[{100*float(row['cp95_lower']):.1f},{100*float(row['cp95_upper']):.1f}]",
                     f"{float(row['threshold']):.3f}"])
    text = table(rows, "rrrrr", "Batch & Exceedances & Rate (\\%) & CP 95\\% interval & $q_{.05}$",
                 r"Fresh Cora-CC rerun under the specified binary node-product law. Each batch uses 200/500/1000 reference/calibration/evaluation draws and a distinct deterministic seed. Intervals are batch-specific conditional Clopper--Pearson intervals; the endpoint envelope is not a pooled interval. The three-batch mean exceedance rate was 5.03\%, consistent with the nominal 5\% target at this resolution.",
                 "tab:cora-rerun")
    (OUTPUT / "cora-rerun.tex").write_text(text, encoding="utf-8")

    source_names = ["normalization_pilot.csv", "calibration_baselines.csv",
                    "l3_run_pool.csv", "l3_fixed_split.csv", "confirm_results.csv",
                    "evidence_audit.json", "influence_residual.csv", "l2_fitted_law.csv",
                    "cora_calibration_batches_20271021.csv",
                    "cora_calibration_batches_20271021.json"]
    visual_dir = RESULTS / "visual_alarm_v1"
    visual_table = visual_dir / "report/table_visual_results.tex"
    if not visual_table.exists():
        raise FileNotFoundError(
            "visual result table missing; run s16_visual_alarm_report.py")
    (OUTPUT / "visual-alarm.tex").write_text(
        "% Generated from frozen visual_alarm_v1 results by "
        "notes/calibration-integration/generate_tables.py\n" +
        visual_table.read_text(encoding="utf-8"), encoding="utf-8")
    for source, target in (
            ("figure_visual_alarm_main.pdf", "visual_alarm_main.pdf"),
            ("figure_visual_alarm_roc.pdf", "visual_alarm_roc.pdf")):
        shutil.copy2(visual_dir / "report" / source, PAPER / "figures" / target)
    visual_names = ["protocol_frozen.json", "analysis/analysis_summary.json",
                    "analysis/e1_calibration_aggregate_ntu2012.csv",
                    "analysis/e1_calibration_aggregate_modelnet40.csv",
                    "analysis/e2_detection_aggregate_ntu2012.csv",
                    "analysis/e2_detection_aggregate_modelnet40.csv",
                    "analysis/e3_task_association_ntu2012.csv",
                    "analysis/e3_task_association_modelnet40.csv",
                    "analysis/e4_main_comparison_ntu2012.csv",
                    "analysis/e4_main_comparison_modelnet40.csv",
                    "analysis/threshold_stability_ntu2012.json",
                    "analysis/threshold_stability_modelnet40.json",
                    "analysis/equal_fpr_roc.csv",
                    "analysis/equal_fpr_bootstrap.json",
                    "analysis/reproducibility.json",
                    "cache_manifest.json",
                    "report/verification.json"]
    visual_hashes = {name: hashlib.sha256(
        (visual_dir / name).read_bytes()).hexdigest() for name in visual_names}
    manifest = dict(normalization_rows=len(normalization), calibration_summaries=summaries,
                    runs=run_manifest,
                    source_hashes={name: hashlib.sha256((RESULTS/name).read_bytes()).hexdigest()
                                   for name in source_names},
                    visual_source_hashes=visual_hashes,
                    generated_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted(OUTPUT.glob("*.tex"))})
    (Path(__file__).parent / "table_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print("Generated five tables: 12 normalization, 9 calibration, 7 run-mechanism, "
          "3 Cora rerun, 2 visual summary rows")


if __name__ == "__main__":
    main()
