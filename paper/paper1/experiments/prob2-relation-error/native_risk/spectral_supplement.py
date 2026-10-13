"""Retrospective exact certificates with supported conditional-mean witnesses.

Original N4 streams, thresholds, code, and gate decisions are immutable inputs.
Binary witnesses are reused from N3. Multiclass witnesses are computed after N4
from the model and graph only, then saved before evaluating their exact bounds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

from conditional_moments import conditional_edge_moments
from exact_certificate import (
    exact_cantelli_lower, exact_directional_moments, exact_group_probability,
)
from n4_runner import (
    DATA, _float_profile, _load_native, _verify_execution_amendment,
    group_states, rational_profile,
)
from native_protocol import select_structural_groups
from run_feasibility import spectral_witness


HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[2]
NOTES = PAPER / "notes" / "native-risk"
PROTOCOL = NOTES / "n4_spectral_supplement_protocol.json"
DIRECTIONS = NOTES / "n4_spectral_supplement_directions.json"
RESULTS = NOTES / "n4_spectral_supplement_certificate.json"
CALIBRATION = NOTES / "n4_calibration.json"
PRIOR = NOTES / "exact_certificate_pilot.json"
SOURCE_PROTOCOL = HERE / "formal_protocol_n4.json"
CORE_FILES = [Path(__file__), HERE / "exact_certificate.py",
              HERE / "run_feasibility.py", HERE / "conditional_moments.py",
              HERE / "native_protocol.py", HERE / "n4_runner.py"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def fraction_text(value):
    return f"{value.numerator}/{value.denominator}"


def frozen_inputs():
    files = CORE_FILES + [CALIBRATION, PRIOR, SOURCE_PROTOCOL,
                         NOTES / "n4_directions.json",
                         NOTES / "n4_precommit_certificate.json",
                         NOTES / "n4_evaluation.json",
                         HERE / "formal_protocol_n4_amendment_v1.json"]
    hashes = {str(path.relative_to(PAPER)): sha256(path) for path in files}
    for tag in ("cora-cc", "dblp-ca"):
        for name in ("hypergraph.pickle", "labels.pickle"):
            hashes[f"source-data/{tag}/{name}"] = sha256(DATA / tag / name)
    return hashes


def supported_witness(mean):
    # The existing pilot solver works on the nonzero principal submatrix.
    # Its full-size result has exact zero coordinates outside that support.
    rows, columns = mean.nonzero()
    allowed = set(rows).union(columns)
    if len(allowed) > 256:
        raise ValueError("native supplement supports only the deterministic dense local solver")
    witness = spectral_witness(mean)
    if not np.any(witness["x"]):
        witness["x"][0] = 1.0
    if allowed and not set(np.flatnonzero(witness["x"])).issubset(allowed):
        raise ArithmeticError("witness has a nonzero coordinate outside mean support")
    return witness


def freeze():
    if PROTOCOL.exists():
        raise FileExistsError("supplement protocol already exists; do not overwrite it")
    source = read(SOURCE_PROTOCOL)
    _verify_execution_amendment(source)
    value = {
        "id": "paper1-n4-retrospective-spectral-supplement-20261012-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inferential_status": "retrospective exact-model calculation; not a registered N4 pass",
        "scope": "all eight original native cells, same groups and calibrated thresholds; no random draws",
        "binary_rule": "reuse every saved N3 state witness without reselection",
        "multiclass_rule": "existing supported spectral_witness at largest absolute eigenvalue for every positive-probability state; exact zeros outside conditional-mean support",
        "zero_mean_rule": "unit vector at node zero if the conditional mean is identically zero",
        "ordering": "save and hash all directions before computing any new threshold bounds",
        "stop": "one fixed witness rule only; retain zero and unavailable results; no tuning after bounds",
        "source_hashes": frozen_inputs(),
        "data_sha256": source["data_sha256"],
        "cells": [{key: cell[key] for key in ("id", "graph", "profile", "epsilon")}
                  for cell in source["native_cells"]],
    }
    save(PROTOCOL, value)
    print(f"FROZEN supplementary calculation: {PROTOCOL}", flush=True)


def load_protocol():
    protocol = read(PROTOCOL)
    if protocol["source_hashes"] != frozen_inputs():
        raise RuntimeError("supplement input/code hashes changed after freeze")
    _verify_execution_amendment(read(SOURCE_PROTOCOL))
    return protocol


def prepare():
    if DIRECTIONS.exists():
        raise FileExistsError("supplement directions already exist")
    protocol = load_protocol()
    prior = {(cell["tag"], cell["epsilon"]): cell for cell in read(PRIOR)["settings"]}
    output = {"protocol_sha256": sha256(PROTOCOL), "status": "all retrospective directions saved before bound evaluation", "cells": []}
    for cell in protocol["cells"]:
        edges, labels = _load_native(cell["graph"])
        probabilities = rational_profile(labels, cell["epsilon"], cell["profile"])
        groups = select_structural_groups(edges, len(labels))["groups"]
        float_probabilities = _float_profile(probabilities)
        saved = {tuple(g["group"]): g for g in prior.get((cell["graph"], cell["epsilon"]), {}).get("groups", [])}
        cell_out = dict(cell, groups=[])
        for group in groups:
            states_out = []
            prior_states = {tuple(s["state"]): s for s in saved.get(group, {}).get("states", [])}
            for state, probability in group_states(probabilities, group):
                if cell["profile"] == "binary_majority_vs_rest":
                    old = prior_states[state]
                    nodes, hexes = old["direction_nodes"], old["direction_float_hex"]
                    source = "N3 prior saved direction"
                    residual = None
                else:
                    mean = conditional_edge_moments(edges, float_probabilities, group, state)["mean"]
                    witness = supported_witness(mean)
                    nodes = np.flatnonzero(witness["x"]).tolist()
                    hexes = [float(witness["x"][u]).hex() for u in nodes]
                    source, residual = "post-N4 supported conditional-mean eigenvector", witness["residual"]
                states_out.append({"state": list(state), "probability_exact": fraction_text(probability),
                                   "direction_nodes": nodes, "direction_float_hex": hexes,
                                   "source": source, "eigen_residual_float": residual})
            cell_out["groups"].append({"nodes": list(group), "states": states_out})
        output["cells"].append(cell_out)
        print(f"prepared {cell['id']}: {sum(len(g['states']) for g in cell_out['groups'])} directions", flush=True)
    save(DIRECTIONS, output)
    print(f"ALL directions saved; sha256={sha256(DIRECTIONS)}", flush=True)


def certify():
    if RESULTS.exists():
        raise FileExistsError("supplement certificate already exists")
    protocol = load_protocol()
    directions = read(DIRECTIONS)
    if directions["protocol_sha256"] != sha256(PROTOCOL):
        raise RuntimeError("direction protocol mismatch")
    thresholds = {cell["id"]: Fraction.from_float(cell["threshold"])
                  for cell in read(CALIBRATION)["cells"]}
    prior = {(c["tag"], c["epsilon"]): c for c in read(PRIOR)["settings"]}
    output = {"protocol_sha256": sha256(PROTOCOL), "directions_sha256": sha256(DIRECTIONS),
              "calibration_sha256": sha256(CALIBRATION),
              "status": "retrospective exact-model certificates; original N4 gates unchanged", "cells": []}
    started = time.perf_counter()
    for cell in directions["cells"]:
        edges, labels = _load_native(cell["graph"])
        probabilities = rational_profile(labels, cell["epsilon"], cell["profile"])
        threshold = thresholds[cell["id"]]
        cell_out = dict(id=cell["id"], graph=cell["graph"], profile=cell["profile"],
                        epsilon=cell["epsilon"], threshold=float(threshold), groups=[])
        old_groups = {tuple(g["group"]): g for g in prior[(cell["graph"], cell["epsilon"])]["groups"]}
        for group in cell["groups"]:
            state_out, mixture, mass = [], Fraction(0), Fraction(0)
            old_states = {tuple(s["state"]): s for s in old_groups[tuple(group["nodes"])]["states"]}
            for state in group["states"]:
                before = time.perf_counter()
                q = [Fraction(0)] * len(labels)
                for node, value in zip(state["direction_nodes"], state["direction_float_hex"]):
                    q[node] = Fraction.from_float(float.fromhex(value))
                moments = exact_directional_moments(edges, probabilities, group["nodes"], state["state"], q)
                probability = exact_group_probability(probabilities, group["nodes"], state["state"])
                if probability != Fraction(state["probability_exact"]):
                    raise ArithmeticError("state probability mismatch")
                if cell["profile"] == "binary_majority_vs_rest":
                    old = old_states[tuple(state["state"])]
                    for key, source_key in (("mean", "raw_mean_exact"), ("variance", "raw_variance_exact"), ("norm_squared", "norm_squared_exact")):
                        if moments[key] != Fraction(old[source_key]):
                            raise ArithmeticError(f"binary prior {key} mismatch")
                lower = exact_cantelli_lower(moments["mean"], moments["variance"], moments["norm_squared"], threshold)
                mixture += probability * lower
                mass += probability
                state_out.append({"state": state["state"], "probability_exact": fraction_text(probability),
                                  "mean_exact": fraction_text(moments["mean"]),
                                  "variance_exact": fraction_text(moments["variance"]),
                                  "norm_squared_exact": fraction_text(moments["norm_squared"]),
                                  "lower_exact": fraction_text(lower), "lower": float(lower),
                                  "overlap_pair_count": moments["overlap_pair_count"]})
                print(f"{cell['id']} group={group['nodes']} state={state['state']} support={len(state['direction_nodes'])} L={float(lower):.6g} seconds={time.perf_counter()-before:.2f}", flush=True)
            if mass != 1:
                raise ArithmeticError("group states do not exhaust probability one")
            cell_out["groups"].append({"nodes": group["nodes"], "lower_exact": fraction_text(mixture), "lower": float(mixture), "states": state_out})
        output["cells"].append(cell_out)
        output["wall_seconds"] = time.perf_counter() - started
        save(RESULTS, output)
    output["complete"] = True
    save(RESULTS, output)
    print("COMPLETE: original N4 gate results remain unchanged", flush=True)


if __name__ == "__main__":
    # Exact sums of many state bounds can legitimately have large integer encodings.
    sys.set_int_max_str_digits(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "directions", "certify"))
    args = parser.parse_args()
    {"freeze": freeze, "directions": prepare, "certify": certify}[args.stage]()
