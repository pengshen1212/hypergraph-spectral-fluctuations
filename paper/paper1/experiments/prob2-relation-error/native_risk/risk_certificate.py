"""Moment-based conditional tail bounds used by the native-risk pilot."""

from __future__ import annotations

import math


def cantelli_tail_lower(mu, variance, threshold):
    """Lower bound P(Y > threshold) from E[Y]=mu and Var(Y)=variance."""
    mu, variance, threshold = float(mu), float(variance), float(threshold)
    if not all(math.isfinite(v) for v in (mu, variance, threshold)):
        raise ValueError("moments and threshold must be finite")
    if variance < 0:
        raise ValueError("variance must be nonnegative")
    gap = mu - threshold
    if gap <= 0:
        return 0.0
    if variance == 0:
        return 1.0
    return gap * gap / (variance + gap * gap)


def frobenius_tail_upper(mean_norm_upper, residual_second_moment, threshold):
    """Upper bound P(||M+R||_2 > threshold) using ||R||_F and Markov."""
    mean_norm_upper = float(mean_norm_upper)
    residual_second_moment = float(residual_second_moment)
    threshold = float(threshold)
    if not all(math.isfinite(v) for v in (
        mean_norm_upper, residual_second_moment, threshold
    )):
        raise ValueError("bounds and threshold must be finite")
    if mean_norm_upper < 0 or residual_second_moment < 0:
        raise ValueError("norm and second-moment bounds must be nonnegative")
    if threshold <= mean_norm_upper:
        return 1.0
    return min(1.0, residual_second_moment / (threshold - mean_norm_upper) ** 2)


def mixture_tail_bounds(states, threshold):
    """Mix per-state lower/upper bounds using the state probabilities.

    Each state is a mapping with ``probability``, ``mean``, ``variance``,
    ``mean_norm_upper`` and ``residual_second_moment``. ``mean`` is the
    nonnegative signed directional mean selected for the lower bound.
    """
    total_probability = sum(float(s["probability"]) for s in states)
    if abs(total_probability - 1.0) > 1e-10:
        raise ValueError("group-state probabilities must sum to one")
    lower = 0.0
    upper = 0.0
    for state in states:
        probability = float(state["probability"])
        if probability < 0:
            raise ValueError("state probability must be nonnegative")
        lower += probability * cantelli_tail_lower(
            state["mean"], state["variance"], threshold
        )
        upper += probability * frobenius_tail_upper(
            state["mean_norm_upper"],
            state["residual_second_moment"],
            threshold,
        )
    return min(1.0, lower), min(1.0, upper)
