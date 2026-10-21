"""Controlled coherent/flat structure families (ported from the manuscript's
upgrade/influence_families.py; `pool` renamed from the original `kappa` to
avoid collision with the operator self-loop kappa=1).

Family A (coherent amplification, non-star overlapping blocks):
    t node-disjoint blocks; block i: weak hub u_i (P[u_i,1]=EPS_WEAK),
    a pool of d/pool strong co-members w (each w reused in ~pool edges),
    fresh strong leaves. Edges (u_i, w_j, leaf). phi_{u_i,e} = 1-2*eps on
    every incident edge -> coherent alignment.

Family B (flat competition sweep, fresh co-members):
    one hub u with d edges (u, w_j, leaf_j), all co-members fresh;
    w strongly class-1; leaf swept P[leaf,1]=gamma.
    phi_{u,e} = gamma - eps: zero at gamma=eps (second-order cancellation),
    max at gamma=1-eps.
"""

from __future__ import annotations

import numpy as np

EPS_WEAK = 0.05
EPS_STRONG = 0.95


def build_family_A(t, d, pool=5, seed=0):
    rng = np.random.default_rng(seed)
    edges, P, hubs = [], [], []
    base = 0
    for _ in range(t):
        u = base
        nw = max(1, d // pool)
        w_pool = list(range(base + 1, base + 1 + nw))
        leaf0 = base + 1 + nw
        block_edges = []
        for j in range(d):
            w = w_pool[int(rng.integers(nw))]
            leaf = leaf0 + j
            block_edges.append((u, int(w), leaf))
        edges.extend(block_edges)
        hubs.append(u)
        base = leaf0 + d
    n = base
    P = np.zeros((n, 2))
    P[:, 1] = EPS_STRONG
    for u in hubs:
        P[u] = [1 - EPS_WEAK, EPS_WEAK]
    return edges, P, hubs


def build_family_B(d, gamma, seed=0):
    n = 1 + 2 * d
    edges = [(0, 1 + 2 * j, 2 + 2 * j) for j in range(d)]
    P = np.zeros((n, 2))
    P[0] = [1 - EPS_WEAK, EPS_WEAK]
    for j in range(d):
        P[1 + 2 * j] = [1 - EPS_STRONG, EPS_STRONG]
        P[2 + 2 * j] = [1 - gamma, gamma]
    return edges, P, [0]
