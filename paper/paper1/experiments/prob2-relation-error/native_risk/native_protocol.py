"""Frozen, data-only choices used by the native-risk feasibility stage."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def pilot_seed_key(tag, epsilon):
    """Map dataset/config labels to the exact keys frozen in protocol.json."""
    prefix = {"cora-cc": "cora", "dblp-ca": "dblp"}.get(tag)
    if prefix is None:
        raise ValueError(f"unknown native-risk pilot tag: {tag}")
    level = int(round(100 * float(epsilon)))
    return f"{prefix}-epsilon-{level:03d}"


def select_structural_groups(edges, n_nodes):
    """Choose the maximum-degree node and its weighted co-edge partner.

    Hyperedge slots are counted as provided, including repeated node sets.
    Ties use the smallest node id. A pair is omitted when its overlap score is
    zero, so a graph with no co-edge partner only receives the singleton.
    """
    degree = np.zeros(int(n_nodes), dtype=np.int64)
    overlap = defaultdict(float)
    normalized = []
    for raw_edge in edges:
        edge = tuple(int(u) for u in raw_edge)
        if len(edge) < 2 or len(set(edge)) != len(edge):
            raise ValueError("each hyperedge slot needs >=2 distinct nodes")
        if min(edge) < 0 or max(edge) >= n_nodes:
            raise ValueError("hyperedge node outside graph")
        normalized.append(edge)
        for u in edge:
            degree[u] += 1
    if not normalized or degree.max() == 0:
        raise ValueError("cannot select a hub from an empty hypergraph")
    hub = int(np.flatnonzero(degree == degree.max())[0])
    for edge in normalized:
        if hub not in edge:
            continue
        for node in edge:
            if node != hub:
                overlap[node] += 1.0 / len(edge)
    candidates = [node for node, score in overlap.items() if score > 0]
    partner = min(candidates, key=lambda node: (-overlap[node], node)) \
        if candidates else None
    return {
        "hub": hub,
        "partner": int(partner) if partner is not None else None,
        "degree": degree,
        "partner_score": float(overlap[partner]) if partner is not None else 0.0,
        "overlap_scores": dict(overlap),
        "groups": [(hub,), (hub, partner)] if partner is not None else [(hub,)],
    }


def _labels_and_majority(labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a nonempty vector")
    classes, inverse, counts = np.unique(
        labels, return_inverse=True, return_counts=True
    )
    majority = classes[int(np.argmax(counts))]
    return labels, classes, inverse, majority


def majority_binary_profile(labels, epsilon):
    """The legacy majority-versus-rest flip profile, in its original coding."""
    labels, _, _, majority = _labels_and_majority(labels)
    epsilon = float(epsilon)
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0,1]")
    P = np.empty((len(labels), 2), dtype=float)
    is_majority = labels == majority
    P[:, 1] = np.where(is_majority, 1.0 - epsilon, epsilon)
    P[:, 0] = 1.0 - P[:, 1]
    return P, majority


def multiclass_profile(labels, epsilon):
    """Symmetric error profile over original labels, without relabelling data."""
    labels, classes, inverse, _ = _labels_and_majority(labels)
    if len(classes) < 2:
        raise ValueError("multiclass profile requires at least two classes")
    epsilon = float(epsilon)
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0,1]")
    P = np.full((len(labels), len(classes)), epsilon / (len(classes) - 1))
    P[np.arange(len(labels)), inverse] = 1.0 - epsilon
    return P, classes.tolist()
