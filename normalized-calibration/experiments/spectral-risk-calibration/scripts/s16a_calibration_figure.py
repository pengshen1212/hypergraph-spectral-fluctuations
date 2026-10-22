"""Rebuild approved Figure 2 from frozen calibration and split-sensitivity files."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
from scipy.stats import beta

HERE = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--analysis-dir", type=Path,
                    default=HERE/"results/visual_alarm_v1/analysis")
parser.add_argument("--output-prefix", type=Path,
                    default=HERE/"results/visual_alarm_v1/report/figure_visual_alarm_main")
args = parser.parse_args()
DATA = args.analysis_dir.resolve()
PREFIX = args.output_prefix.resolve()
OUT = PREFIX.parent
OUT.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8.3, "xtick.labelsize": 7.7,
    "ytick.labelsize": 8.2, "mathtext.fontset": "stixsans",
    "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.left": False,
    "axes.edgecolor": "#A3AFBD", "axes.linewidth": .6,
    "axes.labelcolor": "#253343", "text.color": "#253343",
    "xtick.color": "#566477", "ytick.color": "#253343",
    "xtick.major.size": 3, "ytick.major.size": 0,
})
COLORS = {"ntu2012": "#397A9C", "modelnet40": "#BE8248"}
NAMES = {"ntu2012": "NTU2012", "modelnet40": "ModelNet40"}
ARMS = [("L3_full_run", "Full-run (L3)"),
        ("L2_node_product", "Node-product (L2)"),
        ("edge_iid", "Edge-iid")]
source_hashes, rows, stability = {}, [], {}

for ds in COLORS:
    path = DATA/f"e1_calibration_{ds}.csv"
    source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    for arm, _ in ARMS:
        selected = [r for r in records if r["arm"] == arm and float(r["alpha"]) == .05]
        assert len(selected) == 1
        r = selected[0]
        n, k = int(r["n_eval"]), int(r["exceed_count"])
        assert n == 200 and abs(float(r["rate"])-k/n) < 1e-12
        lo = beta.ppf(.025, k, n-k+1) if k else 0.
        hi = beta.ppf(.975, k+1, n-k) if k < n else 1.
        assert abs(lo-float(r["cp_lo"])) < 1e-12
        assert abs(hi-float(r["cp_hi"])) < 1e-12
        rows.append(dict(panel="a", dataset=ds, arm=arm, count=k, n_eval=n,
                         rate_pct=100*k/n, low_pct=100*lo, high_pct=100*hi,
                         interval_type="exact 95% Clopper-Pearson confidence interval"))
    path = DATA/f"threshold_stability_{ds}.json"
    source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    st = json.loads(path.read_text())
    stability[ds] = st
    original = next(r for r in rows if r["dataset"] == ds and r["arm"] == "L3_full_run")
    assert abs(original["rate_pct"]-100*st["original"]["rate"]) < 1e-12
    split = st["repeated_split"]
    rows.append(dict(panel="b", dataset=ds, arm="L3_resplit", count="", n_eval=200,
                     rate_pct=100*split["fpr_median"], low_pct=100*split["fpr_p2_5"],
                     high_pct=100*split["fpr_p97_5"],
                     interval_type="central 95% range over internal repeated splits"))

fig = plt.figure(figsize=(183/25.4, 98/25.4), facecolor="white")
ax = fig.add_axes([.165, .28, .375, .445])
bx = fig.add_axes([.725, .28, .245, .445])

def title(x, letter, label, subtitle):
    fig.text(x, .94, letter, fontweight="bold", fontsize=10.1)
    fig.text(x+.027, .94, label, fontweight="bold", fontsize=9.3)
    fig.text(x+.027, .884, subtitle, fontsize=7.4, color="#566477")

title(.025, "a", "Calibration by prediction law",
      r"$\alpha=0.05$; 200 evaluation runs per cell")
title(.615, "b", "L3 split sensitivity",
      "10,000 repeated 99/200 partitions")

for target in (ax, bx):
    target.set_axisbelow(True)
    target.grid(axis="x", color="#E3E9EF", linewidth=.55)
    target.axvline(5, color="#768393", linestyle=(0, (3, 3)), lw=.85, zorder=1)
    target.text(5, 1.035, "5% target", transform=target.get_xaxis_transform(),
                ha="center", color="#566477", fontsize=7.1)
    target.set_xlabel("Clean exceedance (%)", labelpad=7)

for j, (arm, _) in enumerate(ARMS):
    center = 2-j
    if j % 2 == 0:
        ax.axhspan(center-.4, center+.4, color="#F5F7FA", zorder=-1)
    for i, ds in enumerate(COLORS):
        r = next(v for v in rows if v["panel"] == "a" and v["dataset"] == ds and v["arm"] == arm)
        y = center + (.14 if i == 0 else -.14)
        value, lo, hi = r["rate_pct"], r["low_pct"], r["high_pct"]
        ax.errorbar(value, y, xerr=[[value-lo], [hi-value]], fmt="o",
                    markersize=4.2, color=COLORS[ds], ecolor=COLORS[ds],
                    linewidth=1.4, capsize=3, capthick=1, zorder=4)
        ax.text(38.5, y, f"{value:.1f}%", ha="right", va="center",
                fontsize=7.7, color=COLORS[ds], fontweight="bold")
ax.set(ylim=(-.5, 2.6), xlim=(-1, 40), xticks=[0, 10, 20, 30],
       yticks=[2, 1, 0], yticklabels=[v[1] for v in ARMS])
ax.tick_params(axis="y", pad=7)
ax.legend(handles=[Line2D([], [], marker="o", color=COLORS[ds], lw=1.3,
                         markersize=4, label=NAMES[ds]) for ds in COLORS],
          frameon=False, ncol=2, columnspacing=1, handlelength=1.25,
          handletextpad=.45, loc="lower left", bbox_to_anchor=(-.39, 1.20),
          fontsize=7.5, borderaxespad=0)

for i, ds in enumerate(COLORS):
    y = 1-i
    row = next(r for r in rows if r["panel"] == "b" and r["dataset"] == ds)
    median, lo, hi = row["rate_pct"], row["low_pct"], row["high_pct"]
    original = 100*stability[ds]["original"]["rate"]
    bx.errorbar(median, y+.14, xerr=[[median-lo], [hi-median]], fmt="o",
                markersize=4.5, color=COLORS[ds], linewidth=1.6,
                capsize=4, capthick=1, zorder=4)
    bx.scatter(original, y-.14, marker="D", s=25,
               color=COLORS[ds], linewidth=.5, edgecolor="white", zorder=5)
    bx.annotate(f"{median:.1f}%", (median, y+.14), xytext=(0, 9),
                textcoords="offset points", ha="center", color=COLORS[ds], fontsize=7.5)
    bx.annotate(f"{original:.1f}%", (original, y-.14), xytext=(0, -11),
                textcoords="offset points", ha="center", color=COLORS[ds],
                fontweight="bold", fontsize=7.7)
bx.set(ylim=(-.5, 1.6), xlim=(-.5, 18), xticks=[0, 5, 10, 15],
       yticks=[1, 0], yticklabels=list(NAMES.values()))
bx.tick_params(axis="y", labelsize=7.7, pad=6)
bx.legend(handles=[Line2D([], [], marker="o", color="#65758A", lw=1.3,
                         markersize=4, label="Resplit median"),
                   Line2D([], [], marker="D", color="#65758A", lw=0,
                         markersize=4, label="Frozen split")],
          frameon=False, ncol=2, columnspacing=.8, handlelength=1.1,
          handletextpad=.35, loc="lower left", bbox_to_anchor=(-.44, 1.20),
          fontsize=7.2, borderaxespad=0)

fig.text(.025, .105, "Whiskers: exact 95% Clopper–Pearson intervals.",
         fontsize=7, color="#566477")
fig.text(.025, .059, "30 reference / 99 calibration / 200 evaluation runs.",
         fontsize=7, color="#566477")
fig.text(.615, .105, "Whiskers: central 95% resplit range.",
         fontsize=7, color="#566477")
fig.text(.615, .059, "Same 299 runs; internal sensitivity only.",
         fontsize=7, color="#566477")

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
issues = []
for artist in fig.findobj(Text):
    if artist.get_visible() and artist.get_text().strip():
        extent = artist.get_window_extent(renderer)
        if not fig.bbox.contains(extent.x0, extent.y0) or not fig.bbox.contains(extent.x1, extent.y1):
            issues.append(artist.get_text())
assert not issues, issues
for extension in ("png", "pdf", "svg"):
    kw = {"dpi": 300} if extension == "png" else {}
    if extension == "pdf":
        kw["metadata"] = {"CreationDate": None, "ModDate": None}
    fig.savefig(PREFIX.with_suffix("."+extension), facecolor="white", **kw)
with (OUT/(PREFIX.name+"_source_data.csv")).open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(OUT/(PREFIX.name+"_qa.json")).write_text(json.dumps({
    "status": "PASS", "source_hashes": source_hashes,
    "all_six_CP_intervals_independently_checked": True,
    "source_rows": len(rows), "text_outside_canvas": issues,
    "size_mm": [183, 98], "new_training_or_resampling": False,
    "usage": "approved manuscript Figure 2",
}, indent=2)+"\n")
plt.close(fig)
print("Approved Figure 2 rebuilt; six rates and exact CP intervals verified")
