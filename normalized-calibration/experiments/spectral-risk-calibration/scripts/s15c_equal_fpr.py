"""S15c: retrospective equal-FPR ROC comparison from the existing cache.

No training.  For each dataset and noise strength, clean evaluation scores are
the negative class and the paired perturbed scores are the positive class.
Thresholds are swept over the same evaluation scores, so the curve is
in-sample/optimistic; it is a score-ranking diagnostic, not a deployment
rule and not a new calibrated detector.

Paired bootstrap resamples run IDs (clean and perturbed scores of a run move
together) and reports T-minus-Frobenius differences at matched realized FPR.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, roc_curve

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

import vision_runs as vr  # noqa: E402

ROOT = HERE / "results" / "visual_alarm_v1"
SCORES = ("T", "frob", "entropy", "active_dev", "U_two_sided")
TARGETS = (0.05, 0.10, 0.15, 0.20)


def _load(ds):
    base = ROOT / "formal" / ds / "batch01"
    recs = [np.load(p, allow_pickle=False)
            for p in sorted((base / "eval").glob("*.npz"))]
    out = {k: np.stack([np.asarray(r[k]) for r in recs]) for k in
           ("T", "frob", "entropy", "active_dev")}
    m_ref = float(np.load(base / "reference_summary.npz",
                          allow_pickle=False)["m_ref"])
    out["U_two_sided"] = np.abs(out["T"] - m_ref)
    out["sigmas"] = np.asarray(recs[0]["sigmas"], dtype=float)
    return out


def curve(neg, pos):
    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    s = np.concatenate([neg, pos])
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr, float(auc(fpr, tpr))


def tpr_at(fpr, tpr, target):
    return float(np.interp(target, fpr, tpr))


def point_estimates(data, sigma_index):
    rows = []
    for name in SCORES:
        neg = data[name][:, 0]
        pos = data[name][:, sigma_index]
        fpr, tpr, a = curve(neg, pos)
        row = {"score": name, "sigma": float(data["sigmas"][sigma_index]),
               "auc": a}
        for t in TARGETS:
            row[f"tpr_at_fpr_{int(round(t * 100))}"] = tpr_at(fpr, tpr, t)
        rows.append(row)
    return rows


def observed_diff(data, sigma_index):
    """Original-sample contrasts; bootstrap means are separate diagnostics."""
    ft, tt, at = curve(data["T"][:, 0], data["T"][:, sigma_index])
    ff, tf, af = curve(data["frob"][:, 0], data["frob"][:, sigma_index])
    estimates = {"auc_diff": at - af}
    for target in TARGETS:
        key = f"tpr_diff_{int(round(target * 100))}"
        estimates[key] = tpr_at(ft, tt, target) - tpr_at(ff, tf, target)
    return estimates


def bootstrap_diff(data, sigma_index, reps, seed):
    estimates = observed_diff(data, sigma_index)
    rng = np.random.default_rng(seed)
    n = len(data["T"])
    out = {f"tpr_diff_{int(round(t * 100))}": np.empty(reps)
           for t in TARGETS}
    out["auc_diff"] = np.empty(reps)
    out["tpr_T"] = {f"{int(round(t * 100))}": np.empty(reps) for t in TARGETS}
    out["tpr_F"] = {f"{int(round(t * 100))}": np.empty(reps) for t in TARGETS}
    for r in range(reps):
        idx = rng.integers(0, n, n)
        ft = {}
        ff = {}
        for name, target in (("T", out["tpr_T"]), ("frob", out["tpr_F"])):
            fpr, tpr, a = curve(data[name][idx, 0],
                                data[name][idx, sigma_index])
            if name == "T":
                out["auc_diff"][r] = a
            else:
                out["auc_diff"][r] -= a
            for t in TARGETS:
                key = f"{int(round(t * 100))}"
                target[key][r] = tpr_at(fpr, tpr, t)
                if name == "T":
                    ft[key] = target[key][r]
                else:
                    ff[key] = target[key][r]
        for t in TARGETS:
            key = f"tpr_diff_{int(round(t * 100))}"
            out[key][r] = ft[f"{int(round(t * 100))}"] - \
                ff[f"{int(round(t * 100))}"]
    summary = {}
    for key, arr in out.items():
        if isinstance(arr, dict):
            summary[key] = {k: {"mean": float(np.mean(v)),
                                "ci_lo": float(np.percentile(v, 2.5)),
                                "ci_hi": float(np.percentile(v, 97.5))}
                            for k, v in arr.items()}
        else:
            summary[key] = {
                "estimate": float(estimates[key]),
                "mean": float(np.mean(arr)),
                "ci_lo": float(np.percentile(arr, 2.5)),
                "ci_hi": float(np.percentile(arr, 97.5)),
                "fraction_positive": float(np.mean(arr > 0)),
            }
    return summary


def make_figure(path_pdf, path_png, all_data=None):
    """Render formal labels at readable size; retain full ROC source points."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points_path = Path(path_pdf).parent / "plot_data/equal_fpr_roc_points.csv"
    if all_data is not None:
        rows = []
        for dataset in ("ntu2012", "modelnet40"):
            data = all_data[dataset]
            for score in ("T", "frob", "entropy"):
                fpr, tpr, _ = curve(data[score][:, 0], data[score][:, -1])
                rows.extend({"dataset": dataset, "sigma": float(data["sigmas"][-1]),
                             "score": score, "fpr": float(x), "tpr": float(y)}
                            for x, y in zip(fpr, tpr))
        points_path.parent.mkdir(exist_ok=True)
        with points_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["dataset", "sigma", "score", "fpr", "tpr"])
            writer.writeheader(); writer.writerows(rows)
    else:
        with points_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelsize": 11, "axes.titlesize": 11.5,
                         "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
                         "legend.fontsize": 10.5, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.35), layout="constrained")
    labels = {"ntu2012": "NTU2012", "modelnet40": "ModelNet40"}
    for ax, dataset in zip(axes, ("ntu2012", "modelnet40")):
        for score, color, label in (("T", "#3973A1", r"Spectral $T$"),
                                    ("frob", "#D58232", "Frobenius"),
                                    ("entropy", "#43865A", "Entropy")):
            pts = [r for r in rows if r["dataset"] == dataset and r["score"] == score]
            x = np.array([float(r["fpr"]) for r in pts])
            y = np.array([float(r["tpr"]) for r in pts])
            ax.plot(x, y, color=color, linewidth=1.7,
                    label=f"{label} ({auc(x, y):.3f})")
        ax.plot([0, 1], [0, 1], color="#777777", ls=":", lw=1)
        ax.set_xlim(0, .30); ax.set_ylim(0, 1.02)
        ax.set_xlabel("False-positive rate")
        ax.set_ylabel("Detection rate")
        ax.set_title(labels[dataset] + r" ($\sigma=1.0$)")
        ax.legend(loc="lower right", frameon=True, framealpha=.97,
                  facecolor="white", edgecolor="none", handlelength=1.4)
    fig.savefig(path_pdf, bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(Path(path_pdf).with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw the ROC from stored source points; no prediction cache needed")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261023)
    args = ap.parse_args()
    if args.plot_only:
        make_figure(ROOT / "report/figure_visual_alarm_roc.pdf",
                    ROOT / "report/figure_visual_alarm_roc.png")
        return
    all_data, rows, boot = {}, [], {}
    for ds in ("ntu2012", "modelnet40"):
        data = _load(ds)
        all_data[ds] = data
        for si in (3, 4):  # sigma = 0.5, 1.0
            for r in point_estimates(data, si):
                r["dataset"] = ds
                rows.append(r)
            boot[f"{ds}_sigma_{data['sigmas'][si]:.2f}"] = bootstrap_diff(
                data, si, args.reps, args.seed)
        print(f"{ds}: ROC and bootstrap done", flush=True)
    import csv
    path = ROOT / "analysis" / "equal_fpr_roc.csv"
    keys = ["dataset", "sigma", "score", "auc"] + \
        [f"tpr_at_fpr_{int(round(t*100))}" for t in TARGETS]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    vr._write_json_atomic(ROOT / "analysis" / "equal_fpr_bootstrap.json", {
        "reps": args.reps, "seed": args.seed,
        "targets": list(TARGETS),
        "comparison": "T minus Frobenius; paired run bootstrap",
        "point_estimate_note": "estimate is the original-sample contrast; mean is the bootstrap replicate mean",
        "caveat": ("retrospective in-sample ROC using evaluation scores; "
                   "thresholds are not deployment rules and this is not a "
                   "new independent confirmation"),
        "results": boot,
    })
    make_figure(ROOT / "report" / "figure_visual_alarm_roc.pdf",
                ROOT / "report" / "figure_visual_alarm_roc.png", all_data)
    for r in rows:
        if r["sigma"] == 1.0:
            print(json.dumps({k: r[k] for k in keys}, indent=0))


if __name__ == "__main__":
    main()
