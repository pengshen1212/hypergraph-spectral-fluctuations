"""Monitoring workflow: scores, three calibrators, refinement and fallback.

This module contains no model training and no manuscript/table logic.  The
monitored operator is always built from the model's PREDICTION vector:

    a_e(Y) = 1{all predictions on edge e agree}
    S(Y)   = D(Y)^-1/2 (sum_e a_e H_e + I) D(Y)^-1/2

The predictor's fixed propagation matrix S_all is a different object and is
never described as varying across runs.  The optional post-processing step

    P_refine = row_normalize((1-lam) P + lam S(Y) P)

is the only place where the monitored S(Y) participates in an action.  lam is
selected on development validation nodes only and frozen before formal runs.
"""

from __future__ import annotations

import math

import numpy as np

import hyperops as ho


# ----------------------------------------------------------------------
# operator / reference quantities
# ----------------------------------------------------------------------
def operator_for(n, edges, kappa=1.0):
    return ho.RelationOperator(n, edges, kappa=kappa)


def all_active_operator(op):
    """Fixed predictor propagation matrix S_all (every candidate edge on)."""
    return op.S_csr(np.ones(op.m))


def fit_center(op, Ys):
    """Frozen elementwise mean M_hat of S(Y) over reference runs."""
    acc = np.zeros(op.n_entries)
    for Y in Ys:
        acc += op.S_data(op.flags_from_Y(Y))
    return acc / float(len(Ys))


def fit_node_law(Ps, n_classes):
    """Reference-run class frequencies with uniform Dirichlet mass 1."""
    n = Ps[0].shape[0]
    counts = np.zeros((n, int(n_classes)), dtype=np.float64)
    for P in Ps:
        Y = np.argmax(P, axis=1)
        counts[np.arange(n), Y] += 1.0
    return (counts + 1.0 / n_classes) / (len(Ps) + 1.0)


def edge_marginals(op, P_hat):
    return op.edge_marginals(P_hat)


# ----------------------------------------------------------------------
# scores
# ----------------------------------------------------------------------
def max_entropy(P, mask):
    m = np.asarray(mask, dtype=bool)
    Q = np.clip(np.asarray(P, dtype=np.float64)[m], 1e-15, 1.0)
    return float(np.mean(-(Q * np.log(Q)).sum(axis=1)))


def run_score(op, Y, P, center, train_mask, ref_active_median):
    """All scores for one prediction vector; P only needed for entropy."""
    flags = op.flags_from_Y(Y)
    s = op.S_data(flags)
    T, resid = ho.spectral_norm_entries(op, s, center)
    frob = float(np.linalg.norm(s - center))
    active = float(np.mean(flags)) if op.m else 0.0
    return {
        "T": T,
        "T_resid": float(resid),
        "T_approx": bool(not math.isfinite(resid) or resid > 1e-5),
        "frob": frob,
        "entropy": max_entropy(P, ~np.asarray(train_mask, dtype=bool)),
        "active_fraction": active,
        "active_dev": abs(active - float(ref_active_median)),
    }


def score_from_flags(op, flags, center):
    s = op.S_data(flags)
    T, resid = ho.spectral_norm_entries(op, s, center)
    return T, float(np.linalg.norm(s - center)), float(resid), \
        float(np.mean(flags))


def sample_node_scores(op, P_hat, count, rng, center):
    """L2 node-product arm: full independent node vectors from fitted law."""
    n, C = P_hat.shape
    T = np.empty(int(count))
    F = np.empty(int(count))
    R = np.empty(int(count))
    A = np.empty(int(count))
    for i in range(int(count)):
        Y = ho.sample_Y(P_hat, rng)
        T[i], F[i], R[i], A[i] = score_from_flags(
            op, op.flags_from_Y(Y), center)
    return {"T": T, "frob": F, "resid": R, "active_fraction": A}


def sample_edge_scores(op, p_hat, count, rng, center):
    """Independent-edge counterfactual arm with matched marginals."""
    T = np.empty(int(count))
    F = np.empty(int(count))
    R = np.empty(int(count))
    A = np.empty(int(count))
    for i in range(int(count)):
        flags = ho.edge_flags_iid(p_hat, rng)
        T[i], F[i], R[i], A[i] = score_from_flags(op, flags, center)
    return {"T": T, "frob": F, "resid": R, "active_fraction": A}


# ----------------------------------------------------------------------
# thresholds / calibration summaries
# ----------------------------------------------------------------------
def rank_threshold(scores, alpha):
    return ho.rank_threshold(scores, alpha)


def threshold_summary(cal_scores, eval_scores, alpha):
    q = ho.rank_threshold(cal_scores, alpha)
    cal = np.asarray(cal_scores, dtype=float)
    ev = np.asarray(eval_scores, dtype=float)
    if math.isinf(q):
        exceed = np.zeros(len(ev), dtype=bool)
    else:
        exceed = ev > q
    k = int(exceed.sum())
    lo, hi = ho.clopper_pearson(k, len(ev))
    return {
        "threshold": float(q),
        "exceed_count": k,
        "n_eval": int(len(ev)),
        "rate": float(k / len(ev)) if len(ev) else math.nan,
        "cp_lo": lo, "cp_hi": hi,
        "tie_rate_cal": float(np.mean(cal == q)) if len(cal) else math.nan,
        "tie_rate_eval": float(np.mean(ev == q)) if len(ev) else math.nan,
    }


def confusion(alarm, event):
    """Run-level 2x2 counts for alarms and task events."""
    alarm = np.asarray(alarm, dtype=bool)
    event = np.asarray(event, dtype=bool)
    tp = int(np.sum(alarm & event))
    fp = int(np.sum(alarm & ~event))
    fn = int(np.sum(~alarm & event))
    tn = int(np.sum(~alarm & ~event))
    out = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "n": int(len(alarm))}
    out["detection_rate"] = tp / (tp + fn) if (tp + fn) else math.nan
    out["precision"] = tp / (tp + fp) if (tp + fp) else math.nan
    out["false_alarm_rate"] = fp / (fp + tn) if (fp + tn) else math.nan
    out["miss_rate"] = fn / (tp + fn) if (tp + fn) else math.nan
    return out


# ----------------------------------------------------------------------
# task metrics (test labels only here; offline evaluation)
# ----------------------------------------------------------------------
def accuracy(pred, labels, mask):
    m = np.asarray(mask, dtype=bool)
    return float(np.mean(np.asarray(pred)[m] == np.asarray(labels)[m]))


def cross_entropy(probs, labels, mask):
    m = np.asarray(mask, dtype=bool)
    Q = np.clip(np.asarray(probs, dtype=np.float64)[m], 1e-15, 1.0)
    y = np.asarray(labels)[m]
    return float(np.mean(-np.log(Q[np.arange(len(y)), y])))


def change_fraction(pred_a, pred_b, mask):
    m = np.asarray(mask, dtype=bool)
    return float(np.mean(np.asarray(pred_a)[m] != np.asarray(pred_b)[m]))


# ----------------------------------------------------------------------
# fixed refinement / fallback action
# ----------------------------------------------------------------------
def row_normalize(P):
    P = np.asarray(P, dtype=np.float64)
    row = P.sum(axis=1, keepdims=True)
    row = np.where(row > 0, row, 1.0)
    return P / row


def refine_probs(P, S_Y, lam):
    """One fixed relation-refinement step; lam is frozen before evaluation."""
    lam = float(lam)
    if lam == 0.0:
        return np.asarray(P, dtype=np.float64).copy()
    Q = (1.0 - lam) * np.asarray(P, dtype=np.float64) + lam * (S_Y @ P)
    return row_normalize(Q)


def refine_accuracy(P, S_Y, lam, labels, mask):
    Q = refine_probs(P, S_Y, lam)
    return accuracy(np.argmax(Q, axis=1), labels, mask)


def select_lambda(P_list, S_list, labels, val_mask, candidates):
    """Development-only selection: mean validation accuracy, tie -> smaller lam."""
    rows = []
    for lam in candidates:
        accs = [refine_accuracy(P, S, lam, labels, val_mask)
                for P, S in zip(P_list, S_list)]
        rows.append({"lambda": float(lam),
                     "mean_val_acc": float(np.mean(accs)),
                     "per_run_val_acc": [float(v) for v in accs]})
    best = max(rows, key=lambda r: (r["mean_val_acc"], -r["lambda"]))
    return float(best["lambda"]), rows


def apply_policy(base_pred, refined_pred, alarms):
    """Alarm -> fall back to base prediction; otherwise use refinement."""
    alarms = np.asarray(alarms, dtype=bool)
    base_pred = np.asarray(base_pred)
    refined_pred = np.asarray(refined_pred)
    return np.where(alarms[:, None], base_pred, refined_pred)
