"""Exact conditional moments for node-driven hyperedge retention.

Hyperedges are a sequence of slots. Repeated node sets are retained because
they represent repeated terms in the relation operator and the same event in
the node-driven arm; they must not be silently made independent in that arm.
"""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np
import scipy.sparse as sp


def _edge_tuple(edge, n=None):
    edge = tuple(int(u) for u in edge)
    if len(edge) < 2 or len(set(edge)) != len(edge):
        raise ValueError("each hyperedge slot must contain >=2 distinct nodes")
    if min(edge) < 0 or (n is not None and max(edge) >= n):
        raise ValueError("hyperedge node index outside P")
    return edge


def _inputs(edges, P, group, state):
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] == 0 or P.shape[1] < 2:
        raise ValueError("P must have shape (n_nodes, n_classes>=2)")
    if not np.isfinite(P).all() or np.any(P < 0):
        raise ValueError("P must contain finite, nonnegative probabilities")
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-12, rtol=0):
        raise ValueError("each row of P must sum to one")
    group = tuple(int(u) for u in group)
    state = tuple(int(c) for c in state)
    if len(group) != len(state) or len(set(group)) != len(group):
        raise ValueError("group and state must align and group nodes be unique")
    if any(u < 0 or u >= len(P) for u in group):
        raise ValueError("group node index outside P")
    if any(c < 0 or c >= P.shape[1] for c in state):
        raise ValueError("state class index outside P")
    edges = tuple(_edge_tuple(edge, len(P)) for edge in edges)
    return edges, P, group, state


def edge_retention_probability(edge, P):
    """Unconditional probability that all predictions on one edge agree."""
    edge = _edge_tuple(edge, len(P))
    return float(np.prod(np.asarray(P)[list(edge)], axis=0).sum())


def conditional_state_probability(P, group, state):
    """Probability of a node-label state under independent node predictions."""
    P = np.asarray(P, dtype=float)
    group, state = tuple(group), tuple(state)
    if len(group) != len(state):
        raise ValueError("group and state must have equal length")
    if any(u < 0 or u >= len(P) for u in group):
        raise ValueError("group node index outside P")
    if any(c < 0 or c >= P.shape[1] for c in state):
        raise ValueError("state class index outside P")
    return float(np.prod([P[u, c] for u, c in zip(group, state)]))


def enumerate_group_states(P, group):
    """Return only positive-probability states and their exact product law."""
    P = np.asarray(P, dtype=float)
    group = tuple(int(u) for u in group)
    if len(set(group)) != len(group) or any(u < 0 or u >= len(P) for u in group):
        raise ValueError("group nodes must be unique valid row indices")
    states = []
    for state in itertools.product(range(P.shape[1]), repeat=len(group)):
        probability = conditional_state_probability(P, group, state)
        if probability > 0.0:
            states.append((state, probability))
    return states


def conditional_edge_rate(edge, P, group, state):
    """E[a_e | predictions on group equal state], without probability division."""
    P = np.asarray(P, dtype=float)
    edge = _edge_tuple(edge, len(P))
    group, state = tuple(group), tuple(state)
    if len(group) != len(state):
        raise ValueError("group and state must have equal length")
    fixed = dict(zip(group, state))
    labels = {fixed[u] for u in edge if u in fixed}
    if len(labels) > 1:
        return 0.0
    if labels:
        c = next(iter(labels))
        free = [u for u in edge if u not in fixed]
        return float(np.prod(P[free, c])) if free else 1.0
    return edge_retention_probability(edge, P)


def conditional_joint_retention(edge, other, P, group, state):
    """E[a_e a_f | group state] for two edge slots.

    Identical slots on the same node set share the same node-prediction event.
    Distinct events with disjoint remaining nodes are conditionally independent.
    """
    P = np.asarray(P, dtype=float)
    edge = _edge_tuple(edge, len(P))
    other = _edge_tuple(other, len(P))
    group, state = tuple(group), tuple(state)
    fixed = dict(zip(group, state))
    ge = conditional_edge_rate(edge, P, group, state)
    gf = conditional_edge_rate(other, P, group, state)
    if ge == 0.0 or gf == 0.0:
        return 0.0
    if set(edge) == set(other):
        return ge
    free_e = set(edge).difference(fixed)
    free_f = set(other).difference(fixed)
    if not free_e.intersection(free_f):
        return ge * gf

    fixed_labels = {
        fixed[u] for u in set(edge).union(other) if u in fixed
    }
    if len(fixed_labels) > 1:
        return 0.0
    free_union = set(edge).union(other).difference(fixed)
    if fixed_labels:
        c = next(iter(fixed_labels))
        return float(np.prod(P[list(free_union), c])) if free_union else 1.0
    return float(np.prod(P[list(free_union)], axis=0).sum())


def conditional_edge_moments(edges, P, group, state):
    """Return group-state probability, edge rates and conditional M_U(z)."""
    edges, P, group, state = _inputs(edges, P, group, state)
    rates = np.asarray([
        conditional_edge_rate(edge, P, group, state) for edge in edges
    ])
    marginals = np.asarray([edge_retention_probability(edge, P) for edge in edges])
    rows, cols, values = [], [], []
    for edge, weight in zip(edges, rates - marginals):
        if weight == 0.0:
            continue
        scale = weight / len(edge)
        for u in edge:
            for v in edge:
                rows.append(u)
                cols.append(v)
                values.append(scale)
    matrix = sp.coo_matrix(
        (values, (rows, cols)), shape=(len(P), len(P)), dtype=float
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return {
        "state_probability": conditional_state_probability(P, group, state),
        "rates": rates,
        "marginals": marginals,
        "mean": matrix,
    }


def _free_overlap_pairs(edges, group, active_ids):
    fixed = set(group)
    by_free_node = defaultdict(list)
    active = set(active_ids)
    pairs = set()
    for i in active_ids:
        for u in set(edges[i]).difference(fixed):
            for j in by_free_node[u]:
                if j in active:
                    pairs.add((j, i) if j < i else (i, j))
            by_free_node[u].append(i)
    return pairs


def _edge_weights(edges, x):
    weights = np.zeros(len(edges), dtype=float)
    for i, edge in enumerate(edges):
        total = float(np.sum(x[list(edge)]))
        weights[i] = total * total / len(edge)
    return weights


def directional_conditional_moments(edges, P, group, state, x):
    """Mean and variance of x^T X_A x conditional on one group state.

    Every edge with nonzero quadratic weight is included, including edges
    that do not touch the conditioned group. Remaining-node overlap indexes
    identify all potentially nonzero conditional covariances.
    """
    edges, P, group, state = _inputs(edges, P, group, state)
    x = np.asarray(x, dtype=float)
    if x.shape != (len(P),) or not np.isfinite(x).all():
        raise ValueError("x must be a finite vector with one entry per node")
    rates = np.asarray([
        conditional_edge_rate(edge, P, group, state) for edge in edges
    ])
    marginals = np.asarray([edge_retention_probability(edge, P) for edge in edges])
    weights = _edge_weights(edges, x)
    mean = float(np.dot(rates - marginals, weights))
    active_ids = np.flatnonzero(weights != 0.0).tolist()
    variance = float(np.dot(rates * (1.0 - rates), weights * weights))
    for i, j in _free_overlap_pairs(edges, group, active_ids):
        joint = conditional_joint_retention(edges[i], edges[j], P, group, state)
        covariance = joint - rates[i] * rates[j]
        variance += 2.0 * covariance * weights[i] * weights[j]
    return mean, float(variance)


def conditional_frobenius_residual_second_moment(edges, P, group, state):
    """E[||X_A-M_U(z)||_F^2 | z] from exact conditional edge covariances."""
    edges, P, group, state = _inputs(edges, P, group, state)
    rates = np.asarray([
        conditional_edge_rate(edge, P, group, state) for edge in edges
    ])
    value = float(np.sum(rates * (1.0 - rates)))
    active_ids = list(range(len(edges)))
    for i, j in _free_overlap_pairs(edges, group, active_ids):
        joint = conditional_joint_retention(edges[i], edges[j], P, group, state)
        covariance = joint - rates[i] * rates[j]
        overlap = len(set(edges[i]).intersection(edges[j]))
        trace_product = overlap * overlap / (len(edges[i]) * len(edges[j]))
        value += 2.0 * covariance * trace_product
    return float(value)


def conditional_mean_matrix(edges, P, group, state):
    """Convenience wrapper returning only M_U(z) as a sparse matrix."""
    return conditional_edge_moments(edges, P, group, state)["mean"]
