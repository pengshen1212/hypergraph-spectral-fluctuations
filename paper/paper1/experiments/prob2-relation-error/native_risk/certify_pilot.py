"""Recompute the N2 binary pilot bounds with exact rational arithmetic."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from conditional_moments import conditional_edge_moments
from exact_certificate import (
    exact_cantelli_lower,
    exact_directional_moments,
    exact_group_probability,
)
from native_protocol import majority_binary_profile


HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[2]
ROOT = PAPER.parent.parent
DATA = ROOT / "data"
NOTES = PAPER / "notes" / "native-risk"
PILOT_PATH = NOTES / "feasibility_pilot.json"
OUTPUT_PATH = NOTES / "exact_certificate_pilot.json"
sys.path.insert(0, str(HERE))
from run_feasibility import spectral_witness  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fraction_record(value):
    return f"{value.numerator}/{value.denominator}"


def rational_probability_profile(labels, majority, epsilon):
    e = Fraction(str(float(epsilon)))
    return [
        [e, 1 - e] if int(label) == int(majority) else [1 - e, e]
        for label in labels
    ]


def load_edges(tag):
    with (DATA / tag / "hypergraph.pickle").open("rb") as stream:
        hypergraph = pickle.load(stream)
    return [tuple(sorted(int(u) for u in edge)) for edge in hypergraph.values()]


def witness_for(edges, P_float, group, state):
    mean = conditional_edge_moments(edges, P_float, group, state)["mean"]
    witness = spectral_witness(mean)
    support = np.flatnonzero(witness["x"])
    return witness["eigenvalue"], witness["x"], support


def main():
    pilot = json.loads(PILOT_PATH.read_text(encoding="utf-8"))
    output = {
        "protocol_id": "paper1-native-risk-exact-certificate-n3-20261012-v1",
        "source_protocol_id": pilot["protocol_id"],
        "status": "exact rational moments for frozen IEEE-754 witness directions; pilot thresholds only",
        "arithmetic": "Python Fraction exact rational operations; P probabilities interpreted as exact decimal protocol values; witness float64 coordinates interpreted as exact dyadic rationals",
        "started_unix": time.time(),
        "inputs": {},
        "settings": [],
    }
    for setting in pilot["settings"]:
        tag, epsilon = setting["tag"], setting["epsilon"]
        data_dir = DATA / tag
        with (data_dir / "labels.pickle").open("rb") as stream:
            labels = np.asarray(pickle.load(stream))
        edges = load_edges(tag)
        P_float, majority = majority_binary_profile(labels, epsilon)
        P_exact = rational_probability_profile(labels, majority, epsilon)
        tau = Fraction.from_float(float(setting["pilot_B_threshold"]["value"]))
        cell = {
            "tag": tag,
            "epsilon": epsilon,
            "threshold_float_hex": float(tau).hex(),
            "threshold_exact_fraction": fraction_record(tau),
            "groups": [],
        }
        for group_record in setting["groups"]:
            group = tuple(group_record["group"])
            group_out = {
                "group": list(group),
                "states": [],
                "weighted_lower_exact": "0/1",
                "direction_encoding": "sorted nonzero node ids and Python float.hex values; each float is decoded exactly as a binary rational",
            }
            mixture_lower = Fraction(0)
            for state_record in group_record["states"]:
                state = tuple(state_record["state"])
                eigenvalue, direction_float, support = witness_for(
                    edges, P_float, group, state
                )
                direction = [Fraction.from_float(float(x)) for x in direction_float]
                moments = exact_directional_moments(
                    edges, P_exact, group, state, direction
                )
                probability = exact_group_probability(P_exact, group, state)
                lower = exact_cantelli_lower(
                    moments["mean"], moments["variance"],
                    moments["norm_squared"], tau
                )
                mixture_lower += probability * lower
                nz = np.flatnonzero(direction_float)
                group_out["states"].append({
                    "state": list(state),
                    "probability_exact": fraction_record(probability),
                    "rayleigh_eigenvalue_float": eigenvalue,
                    "direction_support": int(len(support)),
                    "direction_sha256": hashlib.sha256(
                        direction_float.tobytes()
                    ).hexdigest(),
                    "direction_nodes": nz.astype(int).tolist(),
                    "direction_float_hex": [float(direction_float[i]).hex() for i in nz],
                    "raw_mean_exact": fraction_record(moments["mean"]),
                    "raw_variance_exact": fraction_record(moments["variance"]),
                    "norm_squared_exact": fraction_record(moments["norm_squared"]),
                    "overlap_pair_count": moments["overlap_pair_count"],
                    "cantelli_lower_exact": fraction_record(lower),
                    "cantelli_lower_approx": float(lower),
                    "n2_float_mean_abs_difference": abs(
                        abs(float(moments["mean"] / moments["norm_squared"]))
                        - float(state_record["mu_abs"])
                    ),
                    "n2_float_variance_abs_difference": abs(
                        float(moments["variance"] / moments["norm_squared"] ** 2)
                        - float(state_record["conditional_direction_variance"])
                    ),
                })
                print(
                    f"{tag} eps={epsilon} group={group} state={state}: "
                    f"pairs={moments['overlap_pair_count']} L={float(lower):.6g}",
                    flush=True,
                )
            group_out["weighted_lower_exact"] = fraction_record(mixture_lower)
            group_out["weighted_lower_approx"] = float(mixture_lower)
            group_out["pilot_threshold_only"] = True
            cell["groups"].append(group_out)
        output["settings"].append(cell)
        output["inputs"][tag] = {
            "hypergraph_sha256": sha256(data_dir / "hypergraph.pickle"),
            "labels_sha256": sha256(data_dir / "labels.pickle"),
        }
        OUTPUT_PATH.write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    output["finished_unix"] = time.time()
    output["wall_seconds"] = output["finished_unix"] - output["started_unix"]
    output["code_sha256"] = {
        path.name: sha256(path)
        for path in (HERE / "exact_certificate.py", HERE / "certify_pilot.py")
    }
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {OUTPUT_PATH}; runtime={output['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
