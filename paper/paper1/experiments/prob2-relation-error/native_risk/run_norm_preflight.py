"""Numerical enclosure preflight; never reads or creates N4 streams."""

import json
import time
import platform
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy

from run_feasibility import load_native, sha256, NOTES, DATA
from verified_norm import ExactOperatorBuilder, enclose_norm


def profiles(labels, epsilon, profile):
    eps = Fraction(str(epsilon))
    classes, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if profile == "binary":
        true = (labels == classes[int(np.argmax(counts))]).astype(int)
        c = 2
    else:
        true = inverse
        c = len(classes)
    other = eps / (c - 1)
    rows = [[1 - eps if j == label else other for j in range(c)] for label in true]
    return rows


def marginals(edges, rows):
    values = []
    cache = {}
    for e in edges:
        key = tuple(sorted(tuple(rows[u]) for u in e))
        if key not in cache:
            total = Fraction(0)
            for c in range(len(rows[0])):
                value = Fraction(1)
                for row in key:
                    value *= row[c]
                total += value
            cache[key] = total
        values.append(cache[key])
    return values


def main():
    start = time.time()
    output = {
        "id": "native-risk-norm-preflight-20261012-v1",
        "scope": "numerical width/cost preflight only; no formal calibration or risk inference",
        "seed_namespace": "2026101201 + 100*cell_index + arm_index",
        "draws_per_arm_per_cell": 2,
        "solver_parameters": {"square_path_cap": 20000000, "relative_goal": 0.002,
                              "weight_iterations": 32, "weight_floor": 1e-6},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__},
        "cells": [], "data_sha256": {},
    }
    index = 0
    for graph in ("cora-cc", "dblp-ca"):
        edges, labels, _ = load_native(graph)
        for name in ("hypergraph", "labels"):
            output["data_sha256"][f"{graph}/{name}.pickle"] = sha256(DATA / graph / f"{name}.pickle")
        for profile in ("binary", "multiclass"):
            for epsilon in (.05, .15):
                begin = time.time()
                rows = profiles(labels, epsilon, profile)
                pe = marginals(edges, rows)
                builder = ExactOperatorBuilder(len(labels), edges, pe)
                P = np.array(rows, dtype=float)
                p_float = np.array(pe, dtype=float)
                cell = {"graph": graph, "profile": profile, "epsilon": epsilon,
                        "build_seconds": time.time() - begin, "draws": []}
                for arm_index, arm in enumerate(("A", "B")):
                    seed = 2026101201 + 100 * index + arm_index
                    rng = np.random.default_rng(seed)
                    for draw in range(2):
                        if arm == "A":
                            uniforms = rng.random(len(labels))
                            predictions = (uniforms[:, None] > np.cumsum(P, axis=1)).sum(axis=1)
                            indicators = [int(np.all(predictions[list(e)] == predictions[e[0]])) for e in edges]
                        else:
                            indicators = rng.random(len(edges)) < p_float
                        begin = time.time()
                        matrix, error = builder.build(indicators)
                        result = enclose_norm(matrix, assembly_error=error)
                        result.update({"arm": arm, "seed": seed, "draw": draw,
                                       "seconds": time.time() - begin,
                                       "relative_width": (result["hi"] - result["lo"]) / result["lo"]
                                       if result["lo"] else None})
                        cell["draws"].append(result)
                        print(graph, profile, epsilon, arm, draw,
                              f'[{result["lo"]:.6g}, {result["hi"]:.6g}]',
                              f'gap={result["relative_width"]:.3g}',
                              f'seconds={result["seconds"]:.3g}', flush=True)
                output["cells"].append(cell)
                index += 1
    output["wall_seconds"] = time.time() - start
    output["code_sha256"] = {p.name: sha256(p) for p in (
        Path(__file__), Path(__file__).with_name("verified_norm.py"),
        Path(__file__).with_name("test_verified_norm.py"))}
    path = NOTES / "norm_preflight.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("SAVED", path, "seconds", output["wall_seconds"], flush=True)


if __name__ == "__main__":
    main()
