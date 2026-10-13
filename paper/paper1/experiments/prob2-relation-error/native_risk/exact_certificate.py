"""Exact rational directional moments and one-sided Cantelli certificates."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from itertools import combinations


def _q(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    raise TypeError("exact certificate inputs must be Fraction or int")


def _validate(P, group, state, direction, edges):
    P = tuple(tuple(_q(value) for value in row) for row in P)
    if not P or len(P[0]) < 2 or any(len(row) != len(P[0]) for row in P):
        raise ValueError("P must be a nonempty rectangular probability table")
    if any(value < 0 for row in P for value in row):
        raise ValueError("probabilities must be nonnegative")
    if any(sum(row) != 1 for row in P):
        raise ValueError("each probability row must sum exactly to one")
    group = tuple(int(u) for u in group)
    state = tuple(int(c) for c in state)
    if len(group) != len(state) or len(set(group)) != len(group):
        raise ValueError("group and state must align; group nodes must be unique")
    if any(u < 0 or u >= len(P) for u in group):
        raise ValueError("group node outside P")
    if any(c < 0 or c >= len(P[0]) for c in state):
        raise ValueError("state class outside P")
    direction = tuple(_q(value) for value in direction)
    if len(direction) != len(P):
        raise ValueError("direction must have one coordinate per node")
    if sum(value * value for value in direction) == 0:
        raise ValueError("direction must be nonzero")
    normalized_edges = []
    for raw_edge in edges:
        edge = tuple(int(u) for u in raw_edge)
        if len(edge) < 2 or len(set(edge)) != len(edge):
            raise ValueError("each edge slot must contain >=2 distinct nodes")
        if min(edge) < 0 or max(edge) >= len(P):
            raise ValueError("edge node outside P")
        normalized_edges.append(edge)
    return P, group, state, direction, tuple(normalized_edges)


def _edge_probability(edge, P):
    result = Fraction(0)
    for c in range(len(P[0])):
        term = Fraction(1)
        for u in edge:
            term *= P[u][c]
        result += term
    return result


def _conditional_edge_rate(edge, P, fixed):
    fixed_labels = {fixed[u] for u in edge if u in fixed}
    if len(fixed_labels) > 1:
        return Fraction(0)
    if fixed_labels:
        c = next(iter(fixed_labels))
        result = Fraction(1)
        for u in edge:
            if u not in fixed:
                result *= P[u][c]
        return result
    return _edge_probability(edge, P)


def _conditional_joint(edge, other, P, fixed, rate_edge, rate_other):
    if rate_edge == 0 or rate_other == 0:
        return Fraction(0)
    if set(edge) == set(other):
        return rate_edge
    free_edge = set(edge).difference(fixed)
    free_other = set(other).difference(fixed)
    if not free_edge.intersection(free_other):
        return rate_edge * rate_other
    union = set(edge).union(other)
    fixed_labels = {fixed[u] for u in union if u in fixed}
    if len(fixed_labels) > 1:
        return Fraction(0)
    free_union = union.difference(fixed)
    if fixed_labels:
        c = next(iter(fixed_labels))
        result = Fraction(1)
        for u in free_union:
            result *= P[u][c]
        return result
    result = Fraction(0)
    for c in range(len(P[0])):
        term = Fraction(1)
        for u in free_union:
            term *= P[u][c]
        result += term
    return result


def exact_group_probability(P, group, state):
    P = tuple(tuple(_q(value) for value in row) for row in P)
    if len(group) != len(state):
        raise ValueError("group and state must align")
    probability = Fraction(1)
    for u, c in zip(group, state):
        probability *= P[int(u)][int(c)]
    return probability


def exact_directional_moments(edges, P, group, state, direction):
    """Return exact unnormalized quadratic-form mean, variance, and ||x||^2.

    For q equal to ``direction``, the actual unit-direction statistic is
    q^T X q / (q^T q).  All inputs are rational, including dyadic rationals
    obtained exactly from a frozen IEEE-754 witness vector.
    """
    P, group, state, direction, edges = _validate(
        P, group, state, direction, edges
    )
    fixed = dict(zip(group, state))
    support = {u for u, value in enumerate(direction) if value}
    active = []
    for edge in edges:
        if not support.intersection(edge):
            continue
        total = sum((direction[u] for u in edge if u in support), Fraction(0))
        weight = total * total / len(edge)
        if not weight:
            continue
        rate = _conditional_edge_rate(edge, P, fixed)
        marginal = _edge_probability(edge, P)
        active.append((edge, rate, marginal, weight))

    mean = sum(((rate - marginal) * weight
                for _, rate, marginal, weight in active), Fraction(0))
    variance = sum((rate * (1 - rate) * weight * weight
                    for _, rate, _, weight in active), Fraction(0))

    by_free_node = defaultdict(list)
    pairs = set()
    for i, (edge, _, _, _) in enumerate(active):
        for u in set(edge).difference(fixed):
            for j in by_free_node[u]:
                if j != i:
                    pairs.add((j, i) if j < i else (i, j))
            by_free_node[u].append(i)
    for i, j in sorted(pairs):
        joint = _conditional_joint(
            active[i][0], active[j][0], P, fixed, active[i][1], active[j][1]
        )
        covariance = joint - active[i][1] * active[j][1]
        variance += 2 * covariance * active[i][3] * active[j][3]

    if variance < 0:
        raise ArithmeticError("exact conditional variance is negative")
    norm_squared = sum(
        (value * value for value in direction), Fraction(0)
    )
    return {
        "mean": mean,
        "variance": variance,
        "norm_squared": norm_squared,
        "overlap_pair_count": len(pairs),
    }


def exact_cantelli_lower(mean, variance, norm_squared, threshold):
    """Exact lower bound for P(||X||_2 > threshold) from one fixed direction."""
    mean = _q(mean)
    variance = _q(variance)
    norm_squared = _q(norm_squared)
    threshold = _q(threshold)
    if variance < 0 or norm_squared <= 0 or threshold < 0:
        raise ValueError("invalid variance, direction norm, or threshold")
    gap = abs(mean) - threshold * norm_squared
    if gap <= 0:
        return Fraction(0)
    if variance == 0:
        return Fraction(1)
    return gap * gap / (variance + gap * gap)
