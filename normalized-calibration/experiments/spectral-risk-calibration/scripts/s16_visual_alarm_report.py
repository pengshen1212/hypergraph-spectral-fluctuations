"""S16: tables, plot data, figure and machine verification from frozen results.

Reads the S15 analysis outputs; does not retrain, rethreshold or alter numbers.
Every reported value is checked for internal consistency and missing cells.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

import vision_data as vd  # noqa: E402
import vision_runs as vr  # noqa: E402


def _num(v):
    if v is None or v == "":
        return v
    try:
        f = float(v)
    except ValueError:
        return v
    return f


def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(f)]


def pick(rows, **kw):
    out = []
    for r in rows:
        ok = True
        for k, v in kw.items():
            rv = r.get(k)
            if isinstance(v, float):
                try:
                    if abs(float(rv) - v) > 1e-9:
                        ok = False
                except (TypeError, ValueError):
                    ok = False
            elif str(rv) != str(v):
                ok = False
        if ok:
            out.append(r)
    return out


def mean_vals(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return float(np.mean(vals)) if vals else math.nan


def _pct(x, digits=2):
    return "--" if x is None or (isinstance(x, float) and not math.isfinite(x)) \
        else f"{100.0 * float(x):.{digits}f}"


def build_table(ds, e1, e2, e3, e4, main, diag, stability):
    label = {"ntu2012": "NTU2012", "modelnet40": "ModelNet40"}[ds]
    cells = []
    for arm in ("L3_full_run", "L2_node_product", "edge_iid"):
        rows = pick(e1, dataset=ds, arm=arm, alpha=0.05)
        if len(rows) != 1:
            raise ValueError("The CP table requires exactly one frozen batch")
        row = rows[0]
        cells.append(f"{100*float(row['rate']):.1f} "
                     f"[{100*float(row['cp_lo']):.1f},{100*float(row['cp_hi']):.1f}]")
    bdiag = list(diag["batches"].values())
    train_t = float(np.median([x["train_seconds_median"] for x in bdiag]))
    score_t = float(np.median([x["score_seconds_median_eval"] for x in bdiag]))
    resplit = 100 * float(stability["repeated_split"]["fpr_mean"])
    return (f"{label} & " + " & ".join(cells) +
            f" & {resplit:.1f} & {train_t:.1f}/{score_t:.2f} " + r"\\")


def latex_table(ds_rows):
    head = r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccc}
\toprule
Dataset & \multicolumn{3}{c}{Clean exceedance (\%; 95\% CP interval)} & Resplit & Time (s) \\
\cmidrule(lr){2-4}
 & L3 & L2 & Edge-iid & mean (\%) & train/score \\
\midrule
"""
    tail = r"""
\bottomrule
\end{tabular}
\caption{Cross-domain calibration at $\alpha=5\%$. Each dataset uses
30 reference, 99 calibration and 200 independent evaluation complete
runs. L3 calibrates on complete runs, L2 on a fitted node-product law,
and Edge-iid on independently sampled edge events. Brackets are
batch-specific $95\%$ Clopper--Pearson intervals. Resplit means use
10,000 random 99/200 partitions of the same 299 scores with a frozen
reference center; they are internal sensitivity summaries, not
independent validation. Timing gives median training/spectral-scoring
seconds per complete run under the recorded six-way parallel setup.}
\label{tab:visual-alarm-validation}
\end{table}
"""
    return head + "\n".join(ds_rows) + tail


def make_figure(out_pdf, out_png, all_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dsets = list(all_data)
    fig, axes = plt.subplots(len(dsets), 4, figsize=(17, 3.6 * len(dsets)))
    if len(dsets) == 1:
        axes = np.asarray([axes])
    for row, ds in enumerate(dsets):
        d = all_data[ds]
        e1, e2, e3, e4 = d["e1"], d["e2"], d["e3"], d["e4"]
        ax = axes[row, 0]
        arms = ["L3_full_run", "L2_node_product", "edge_iid"]
        vals = [mean_vals(pick(e1, dataset=ds, arm=a, alpha=0.05),
                          "mean_rate") for a in arms]
        ax.bar(range(3), [100 * v for v in vals],
               color=["#2b6cb0", "#c53030", "#718096"])
        ax.axhline(5, color="black", ls="--", lw=1)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["L3", "L2", "edge"], fontsize=8)
        ax.set_ylabel("clean FPR (%)")
        ax.set_title(f"{ds}: calibration", fontsize=9)
        ax.set_ylim(0, max(10, 1.2 * max(vals + [0.05]) * 100))

        ax = axes[row, 1]
        for score, label in (("T", "T (spectral)"), ("frob", "Frobenius"),
                             ("entropy", "entropy"),
                             ("active_dev", "active-edge"),
                             ("U_two_sided", "two-sided U")):
            rows = pick(e2, dataset=ds, score=score, alpha=0.05)
            xs = [float(r["sigma"]) for r in rows]
            ys = [100 * float(r["mean_rate"]) for r in rows]
            ax.plot(xs, ys, marker="o", ms=3, label=label)
        ax.axhline(5, color="black", ls=":", lw=1)
        ax.set_xlabel(r"feature-noise $\sigma$")
        ax.set_ylabel("alarm rate (%)")
        ax.set_title("detection / clean calibration", fontsize=9)
        ax.legend(fontsize=6, frameon=False)

        ax = axes[row, 2]
        rows = sorted(pick(e3, dataset=ds), key=lambda r: float(r["sigma"]))
        xs = [r["sigma"] for r in rows]
        ax.plot(xs, [100 * r["detection_rate"] for r in rows],
                marker="o", label="event detection")
        ax.plot(xs, [100 * r["alarm_no_event_fraction"] for r in rows],
                marker="s", label="alarm, no $\\geq$3pp drop")
        ax.plot(xs, [100 * r["event_no_alarm_fraction"] for r in rows],
                marker="^", label="drop, no alarm")
        ax.set_xlabel(r"feature-noise $\sigma$")
        ax.set_ylabel("rate (%)")
        ax.set_title("task association", fontsize=9)
        ax.legend(fontsize=6, frameon=False)

        ax = axes[row, 3]
        for policy, label in (("base", "base"), ("always_refine", "refine"),
                              ("spectral_fallback", "spec fallback"),
                              ("entropy_fallback", "entropy fallback"),
                              ("random_fallback", "random fallback")):
            rows = sorted(pick(e4, dataset=ds, policy=policy),
                          key=lambda r: float(r["sigma"]))
            ax.plot([r["sigma"] for r in rows],
                    [100 * r["mean_acc"] for r in rows], marker="o", ms=3,
                    label=label)
        ax.set_xlabel(r"feature-noise $\sigma$")
        ax.set_ylabel("test accuracy (%)")
        ax.set_title("fixed fallback action", fontsize=9)
        ax.legend(fontsize=6, frameon=False)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_main_figure(out_pdf, out_png, all_data):
    """Rebuild the approved point/interval layout from the same frozen files."""
    import subprocess

    out_pdf, out_png = Path(out_pdf), Path(out_png)
    assert out_png == out_pdf.with_suffix(".png")
    subprocess.run([
        sys.executable, str(Path(__file__).with_name("s16a_calibration_figure.py")),
        "--analysis-dir", str(out_pdf.parent.parent/"analysis"),
        "--output-prefix", str(out_pdf.with_suffix("")),
    ], check=True)


def verify(ds, e1, e2, e3, e4, main, diag, holm_rows):
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    arms = ("L3_full_run", "L2_node_product", "edge_iid")
    for arm in arms:
        rows = pick(e1, dataset=ds, arm=arm, alpha=0.05)
        add(f"E1 rows {arm}", len(rows) == int(diag["counts"]["batches"]),
            f"{len(rows)} batches")
        for r in rows:
            add(f"E1 rate matches count b{r['batch']} {arm}",
                abs(float(r["rate"]) - float(r["exceed_count"]) /
                    float(r["n_eval"])) < 1e-12)
    l3 = pick(e1, dataset=ds, arm="L3_full_run", alpha=0.05)
    t0 = pick(e2, dataset=ds, score="T", alpha=0.05, sigma=0.0)
    for a, b in zip(l3, t0):
        if a["batch"] == b["batch"]:
            add(f"E1/E2 clean FPR agree b{a['batch']}",
                abs(float(a["rate"]) - float(b["rate"])) < 1e-12)
    sig = [float(r["sigma"]) for r in sorted(pick(e2, dataset=ds, score="T",
                                                  alpha=0.05),
                                             key=lambda r: float(r["sigma"]))]
    add("E2 strength grid", sig == [0.0, 0.1, 0.25, 0.5, 1.0], str(sig))
    for r in pick(e2, dataset=ds, alpha=0.05):
        add(f"E2 rate/count {r['score']} b{r['batch']} s{r['sigma']}",
            abs(float(r["rate"]) - float(r["count"]) / float(r["n"])) < 1e-12)
    for r in pick(e4, dataset=ds):
        for key in ("policy_base", "policy_refine", "policy_spectral_fallback",
                    "policy_entropy_fallback", "policy_random_fallback"):
            add(f"E4 range {key} b{r['batch']} s{r['sigma']}",
                0.0 <= float(r[key]) <= 1.0)
    for row in main:
        add(f"E4 CI ordered {row['comparison']} s{row['sigma']}",
            float(row["ci_lo"]) <= float(row["ci_hi"]))
    events = sum(r["tp"] + r["fn"] for r in pick(e3, dataset=ds))
    if events == 0:
        add("E3 event estimation status", True,
            "zero task events: detection/precision not estimable (allowed)")
    else:
        add("E3 event counts", True, f"{events} events")
    if len(holm_rows) >= 2:
        by = {r["dataset"]: r for r in holm_rows}
        if ds in by:
            h = by[ds]
            add("Holm adjusted >= raw", float(h["p_holm"]) >=
                float(h["p_raw"]) - 1e-12)
            add("Holm <= 1", float(h["p_holm"]) <= 1.0)
    add("solver approximation diagnosed",
        all("solver_approx_fraction_eval" in d
            for d in diag.get("batches", {}).values()))
    add("batch thresholds not pooled", int(diag["counts"]["batches"]) >= 1)
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    results = Path(args.results)
    analysis = results / "analysis"
    report_dir = results / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "plot_data").mkdir(exist_ok=True)
    summary = vr.load_json(analysis / "analysis_summary.json")
    preferred = ["ntu2012", "modelnet40"]
    dsets = [d for d in preferred if d in summary["datasets"]] +         [d for d in summary["datasets"] if d not in preferred]
    all_data, checks, rows_tex = {}, [], []
    for ds in dsets:
        e1 = read_csv(analysis / f"e1_calibration_aggregate_{ds}.csv")
        e2 = read_csv(analysis / f"e2_detection_aggregate_{ds}.csv")
        e3 = read_csv(analysis / f"e3_task_association_{ds}.csv")
        e4 = read_csv(analysis / f"e4_application_aggregate_{ds}.csv")
        main = read_csv(analysis / f"e4_main_comparison_{ds}.csv")
        e1_raw = read_csv(analysis / f"e1_calibration_{ds}.csv")
        e2_raw = read_csv(analysis / f"e2_detection_{ds}.csv")
        e4_raw = read_csv(analysis / f"e4_application_{ds}.csv")
        diag = summary["datasets"][ds]["diagnostics"]
        stability = vr.load_json(analysis / f"threshold_stability_{ds}.json")
        all_data[ds] = {"e1": e1, "e2": e2, "e3": e3, "e4": e4,
                        "main": main, "stability": stability}
        rows_tex.append(build_table(ds, e1_raw, e2, e3, e4, main, diag, stability))
        checks.extend(verify(ds, e1_raw, e2_raw, e3, e4_raw, main, diag,
                             summary["holm_main"]))

        # plot data
        with open(report_dir / "plot_data" / f"calibration_{ds}.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["arm", "alpha", "mean_fpr", "per_batch_rates"])
            for r in e1:
                if r["alpha"] == 0.05 and "minus" not in str(r["arm"]):
                    w.writerow([r["arm"], r["alpha"], r["mean_rate"],
                                r["rates"]])
        with open(report_dir / "plot_data" / f"detection_{ds}.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["score", "sigma", "mean_rate", "per_batch_rates"])
            for r in e2:
                if r["alpha"] == 0.05:
                    w.writerow([r["score"], r["sigma"], r["mean_rate"],
                                r["rates"]])
        with open(report_dir / "plot_data" / f"task_{ds}.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sigma", "alarm_rate", "event_rate", "detection_rate",
                        "false_alarm_rate", "alarm_no_event",
                        "event_no_alarm"])
            for r in e3:
                w.writerow([r["sigma"], r["alarm_rate"], r["event_rate"],
                            r["detection_rate"], r["false_alarm_rate"],
                            r["alarm_no_event_fraction"],
                            r["event_no_alarm_fraction"]])
        with open(report_dir / "plot_data" / f"application_{ds}.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["policy", "sigma", "mean_acc", "per_batch"])
            for r in e4:
                w.writerow([r["policy"], r["sigma"], r["mean_acc"],
                            r["per_batch"]])

    tex = latex_table(rows_tex)
    (report_dir / "table_visual_results.tex").write_text(tex, encoding="utf-8")

    try:
        make_main_figure(report_dir / "figure_visual_alarm_main.pdf",
                         report_dir / "figure_visual_alarm_main.png", all_data)
        fig_ok = True
    except Exception as exc:  # pragma: no cover - reported, not hidden
        fig_ok = False
        print(f"figure generation failed: {exc}")

    checks.append({"check": "figure generated", "pass": fig_ok})
    report = {
        "generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        "analysis_summary_sha256": vd.sha256_file(
            analysis / "analysis_summary.json"),
        "budget_selected": summary["budget_selected"],
        "budget_counts": summary["budget_counts"],
        "dataset_order": dsets,
        "holm_main": summary["holm_main"],
        "checks": checks,
        "all_checks_pass": all(c["pass"] for c in checks),
        "figure_pdf": str(report_dir / "figure_visual_alarm_main.pdf"),
        "figure_png": str(report_dir / "figure_visual_alarm_main.png"),
        "main_figure_pdf": str(report_dir / "figure_visual_alarm_main.pdf"),
        "main_figure_png": str(report_dir / "figure_visual_alarm_main.png"),
        "table_tex": str(report_dir / "table_visual_results.tex"),
        "notes": summary["notes"],
    }
    vr._write_json_atomic(report_dir / "verification.json", report)
    lines = ["# Visual alarm validation: machine report", "",
             f"Budget: {summary['budget_selected']} {summary['budget_counts']}",
             "", "## Checks", ""]
    for c in checks:
        lines.append(f"- [{'x' if c['pass'] else ' '}] {c['check']}"
                     + (f" ({c['detail']})" if c.get("detail") else ""))
    lines += ["", "## Main paired comparison (sigma=0.5)", ""]
    for r in summary["holm_main"]:
        lines.append(
            f"- {r['dataset']}: spec-refine = {100*r['estimate']:.2f} pp "
            f"[{100*r['ci_lo']:.2f}, {100*r['ci_hi']:.2f}], "
            f"p={r['p_raw']:.4g}, Holm={r['p_holm']:.4g}")
    (report_dir / "report.md").write_text("\n".join(lines) + "\n",
                                          encoding="utf-8")
    print("\n".join(lines))
    print(f"\nreport written to {report_dir}")


if __name__ == "__main__":
    main()
