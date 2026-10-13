"""Pilot run for the influence validation (cost + debugging only).

Runs the two structure families at pilot scale (64 draws), reports raw
records, running cost, and the protocol hash. Pilot records never enter
formal intervals.

Usage:
    python run_influence_pilot.py --protocol protocol.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from influence_families import (
    beta_centered,
    build_family_A,
    build_family_B,
    edge_probabilities,
    influence_at_witness,
    quadratic_at,
    sample_weights_A,
    sample_weights_B,
    theory_bounds_family_A,
)

sys.stdout.reconfigure(encoding="utf-8")
UP = Path(__file__).resolve().parent


def protocol_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def run_point(edges, P, hubs, draws, seed):
    n = len(P)
    x = np.zeros(n)
    x[hubs[0]] = 1.0
    Phi, V1 = influence_at_witness(edges, P, x)
    pe = edge_probabilities(edges, P)
    rng = np.random.default_rng(seed)
    recs = []
    t0 = time.time()
    for _ in range(draws):
        wA = sample_weights_A(edges, P, rng)
        wB = sample_weights_B(edges, P, rng)
        bA = beta_centered(n, edges, wA)
        bB = beta_centered(n, edges, wB)
        qA = quadratic_at(edges, wA, x)
        qB = quadratic_at(edges, wB, x)
        recs.append(dict(betaA=bA, betaB=bB, qA=qA, qB=qB))
    wall = time.time() - t0
    bAs = np.array([r["betaA"] for r in recs])
    bBs = np.array([r["betaB"] for r in recs])
    qAs = np.array([r["qA"] for r in recs])
    qBs = np.array([r["qB"] for r in recs])
    return dict(
        draws=draws, seed=seed, wall_seconds=wall,
        sec_per_draw=wall / draws,
        Phi_hub=float(Phi[hubs[0]]), V1=float(V1),
        varA_q=float(np.var(qAs)), varB_q=float(np.var(qBs)),
        mean_betaA=float(bAs.mean()), mean_betaB=float(bBs.mean()),
        sep_freq=float((bAs > bBs).mean()),
        records=recs,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default=str(UP / "protocol.json"))
    ap.add_argument("--draws", type=int, default=64)
    args = ap.parse_args()
    proto = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    ph = protocol_hash(args.protocol)
    out = dict(protocol_hash=ph, pilot=True, points=[])

    print(f"== influence pilot (protocol {ph}, {args.draws} draws/point) ==")
    for d in (100, 200):
        edges, P, hubs = build_family_A(20, d, kappa=5, seed=20261009)
        th = theory_bounds_family_A(20, d, 5)
        res = run_point(edges, P, hubs, args.draws, 7000001 + d)
        res.update(family="A", d=d, kappa=5, theory=th)
        out["points"].append(res)
        print(f"[A d={d}] Phi={res['Phi_hub']:.1f} V1={res['V1']:.2f} "
              f"cert={th['certified']} L_A={th['L_A']:.1f} U_B={th['U_B']:.1f} "
              f"| betaA={res['mean_betaA']:.2f} betaB={res['mean_betaB']:.2f} "
              f"sep={res['sep_freq']:.0%} varA/varB(q)="
              f"{res['varA_q'] / max(res['varB_q'], 1e-12):.2f} "
              f"({res['sec_per_draw']:.2f}s/draw)")
    for gamma in (0.95, 0.05):
        edges, P, hubs = build_family_B(200, gamma)
        res = run_point(edges, P, hubs, args.draws, 7100001 + int(gamma * 100))
        phi_u = res["Phi_hub"]
        res.update(family="B", d=200, gamma=gamma)
        out["points"].append(res)
        print(f"[B gamma={gamma}] Phi_u={phi_u:.3f} V1={res['V1']:.4f} "
              f"| betaA={res['mean_betaA']:.2f} betaB={res['mean_betaB']:.2f} "
              f"sep={res['sep_freq']:.0%} varA/varB(q)="
              f"{res['varA_q'] / max(res['varB_q'], 1e-12):.3f} "
              f"({res['sec_per_draw']:.2f}s/draw)")

    dest = UP / "results" / "pilot_influence.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
