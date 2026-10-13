"""Frozen N4 preparation, conservative calibration, and held-out evaluation."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import pickle
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.stats import beta

from conditional_moments import conditional_edge_moments
from exact_certificate import (exact_cantelli_lower, exact_directional_moments,
                               exact_group_probability)
from finite_sample import tolerance_order_statistic_rank
from native_protocol import select_structural_groups
from verified_norm import ExactOperatorBuilder, enclose_norm


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from upgrade.influence_families import build_family_A, build_family_B

PAPER = HERE.parents[2]
ROOT = PAPER.parent.parent
DATA = ROOT / "data"
NOTES = PAPER / "notes" / "native-risk"
PROTOCOL_PATH = HERE / "formal_protocol_n4.json"
PREPARED_PATH = NOTES / "n4_directions.json"
CALIBRATION_PATH = NOTES / "n4_calibration.json"
CERTIFICATE_PATH = NOTES / "n4_precommit_certificate.json"
EVALUATION_PATH = NOTES / "n4_evaluation.json"
AMENDMENT_PATH = HERE / "formal_protocol_n4_amendment_v1.json"
AMENDMENT_CODE = {
    "n4_runner.py": HERE / "n4_runner.py",
    "verified_norm.py": HERE / "verified_norm.py",
    "exact_certificate.py": HERE / "exact_certificate.py",
    "conditional_moments.py": HERE / "conditional_moments.py",
    "native_protocol.py": HERE / "native_protocol.py",
    "finite_sample.py": HERE / "finite_sample.py",
    "influence_families.py": HERE.parent / "upgrade" / "influence_families.py",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _frac(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float):
        result = Fraction(value).limit_denominator(1_000_000)
        if abs(float(result) - value) > 1e-12:
            raise ValueError("float is not a registered rational probability")
        return result
    return Fraction(str(value))


def rational_profile(labels, epsilon, profile):
    eps = _frac(epsilon)
    classes, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if len(classes) < 2:
        raise ValueError("at least two observed labels are required")
    if profile == "binary_majority_vs_rest":
        majority = int(np.argmax(counts))
        true_binary = (inverse == majority).astype(int)
        return [[eps, 1 - eps] if y == 1 else [1 - eps, eps]
                for y in true_binary]
    if profile == "symmetric_multiclass":
        other = eps / (len(classes) - 1)
        return [[1 - eps if c == y else other for c in range(len(classes))]
                for y in inverse]
    raise ValueError(f"unsupported profile {profile}")


def controlled_profile(spec):
    generator = spec["generator"]
    params = spec["params"]
    if generator == "existing family_A_amplification":
        edges, raw, _ = build_family_A(seed=params["structure_seed"], **{
            key: value for key, value in params.items() if key != "structure_seed"})
        raw[:, 0] = 1.0 - raw[:, 1]
    elif generator == "existing family_B_competition":
        edges, raw, _ = build_family_B(seed=0, **params)
    elif generator == "60 pairwise node-disjoint triples":
        edges = [(3 * j, 3 * j + 1, 3 * j + 2) for j in range(60)]
        raw = np.tile(np.array([[.95, .05], [.95, .05], [.05, .95]]), (60, 1))
    else:
        raise ValueError(f"unsupported controlled generator: {generator}")
    profile = [[_frac(x) for x in row] for row in raw]
    if any(sum(row) != 1 or any(x < 0 or x > 1 for x in row) for row in profile):
        raise ValueError("controlled profile must be exact stochastic rows")
    return [tuple(map(int, edge)) for edge in edges], profile


def edge_marginals(edges, probabilities):
    cache = {}
    output = []
    for edge in edges:
        pattern = tuple(sorted(tuple(probabilities[u]) for u in edge))
        if pattern not in cache:
            value = Fraction(0)
            for c in range(len(probabilities[0])):
                term = Fraction(1)
                for row in pattern:
                    term *= row[c]
                value += term
            cache[pattern] = value
        output.append(cache[pattern])
    return output


def group_states(probabilities, group):
    for state in itertools.product(range(len(probabilities[0])), repeat=len(group)):
        pi = exact_group_probability(probabilities, group, state)
        if pi:
            yield state, pi


def _deterministic_witness(mean, _token, node_degrees=None):
    diagonal = np.asarray(mean.diagonal(), dtype=float)
    if not len(diagonal):
        return np.zeros(mean.shape[0]), 0.0
    degrees = np.ones(len(diagonal), dtype=np.int64) if node_degrees is None else np.asarray(node_degrees)
    if degrees.shape != diagonal.shape:
        raise ValueError("one structural degree per node is required")
    candidates = np.flatnonzero(diagonal != 0)
    if not len(candidates):
        rows, cols = mean.nonzero()
        candidates = np.unique(np.concatenate((rows, cols)))
    if not len(candidates):
        candidates = np.arange(len(diagonal))
    coordinate = min(candidates, key=lambda u: (degrees[u], -abs(diagonal[u]), int(u)))
    coordinate = int(coordinate)
    result = np.zeros(mean.shape[0])
    result[coordinate] = 1.0
    eigenvalue = float(diagonal[coordinate])
    return result, eigenvalue


def classify_enclosure(lo, hi, threshold):
    if lo > threshold:
        return "alarm"
    if hi <= threshold:
        return "nonalarm"
    return "ambiguous"


def count_envelope(intervals, threshold):
    states = [classify_enclosure(lo, hi, threshold) for lo, hi in intervals]
    n = len(states)
    if n == 0:
        raise ValueError("cannot summarize an empty batch")
    alarms, nonalarms = states.count("alarm"), states.count("nonalarm")
    ambiguous = n - alarms - nonalarms
    return {"draws": n, "definite_alarm": alarms,
            "definite_nonalarm": nonalarms, "ambiguous": ambiguous,
            "rate_envelope": [alarms / n, (alarms + ambiguous) / n]}


def exact_binomial_interval(successes, n, confidence):
    if n < 1 or not 0 <= successes <= n or not 0 < confidence < 1:
        raise ValueError("invalid binomial interval inputs")
    tail = (1.0 - confidence) / 2.0
    low = 0.0 if successes == 0 else float(beta.ppf(tail, successes, n - successes + 1))
    high = 1.0 if successes == n else float(beta.ppf(1.0 - tail, successes + 1,
                                                        n - successes))
    return low, high


def order_statistic_threshold(scores, rank):
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite one-dimensional scores required")
    if not 1 <= int(rank) <= len(values):
        raise ValueError("rank outside sample")
    return float(np.partition(values, int(rank) - 1)[int(rank) - 1])


def retry_enclosure(builder, indicators, solve=enclose_norm):
    """Retry one unchanged draw, using a coordinate witness on retry."""
    last_error = None
    matrix = None
    assembly_error = 0.0
    for attempt in range(2):
        try:
            matrix, assembly_error = builder.build(indicators)
            witness = None
            if attempt and matrix.shape[0]:
                witness = np.zeros(matrix.shape[0])
                witness[int(np.argmax(abs(matrix.diagonal())))] = 1.0
            result = solve(matrix, assembly_error=assembly_error, witness=witness)
            return {"lo": result["lo"], "hi": result["hi"],
                    "power": result.get("power", 1)}
        except Exception as error:
            last_error = error
    return {"lo": 0.0, "hi": np.finfo(float).max,
            "failure": type(last_error).__name__ if last_error else "UnknownError",
            "status": "ambiguous"}


def primary_task_consequence_gate(batches, certificate_lower, pooled_lower,
                                  alpha=.05, target=.10):
    return (Fraction(certificate_lower) >= Fraction(7, 100) and
            float(pooled_lower) > target and len(batches) == 2 and
            all(row["rate_envelope"][0] > alpha for row in batches))


def validate_seed_plan(protocol):
    seeds = []
    for cell in protocol["native_cells"]:
        seeds.extend(cell[key] for key in ("B_calibration_seed", "A_batch_1_seed",
                                            "A_batch_2_seed", "B_batch_1_seed",
                                            "B_batch_2_seed"))
    for cell in protocol["controlled_boundary"]["cells"]:
        seeds.extend(cell[key] for key in ("B_calibration_seed", "A_seed",
                                            "B_validation_seed"))
    if len(seeds) != 55 or len(set(seeds)) != 55:
        raise RuntimeError("frozen N4 seed plan must contain 55 unique streams")


def _verify_execution_amendment(protocol):
    if not AMENDMENT_PATH.exists():
        raise RuntimeError("formal N4 execution amendment must be frozen before sampling")
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    if (amendment.get("parent_protocol_sha256") != sha256(PROTOCOL_PATH) or
            amendment.get("status") != "frozen; N4 execution authorized" or
            amendment.get("runner_code_sha256") != {
                name: sha256(path) for name, path in AMENDMENT_CODE.items()}):
        raise RuntimeError("N4 execution amendment/code hashes do not match")
    data_files = {
        "cora_hypergraph": DATA / "cora-cc" / "hypergraph.pickle",
        "cora_labels": DATA / "cora-cc" / "labels.pickle",
        "dblp_hypergraph": DATA / "dblp-ca" / "hypergraph.pickle",
        "dblp_labels": DATA / "dblp-ca" / "labels.pickle",
    }
    if {name: sha256(path) for name, path in data_files.items()} != protocol["data_sha256"]:
        raise RuntimeError("native input data do not match the frozen protocol hashes")
    validate_seed_plan(protocol)
    return sha256(AMENDMENT_PATH)


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")
    temp.replace(path)


def _load_native(tag):
    with (DATA / tag / "hypergraph.pickle").open("rb") as stream:
        raw = pickle.load(stream)
    with (DATA / tag / "labels.pickle").open("rb") as stream:
        labels = np.asarray(pickle.load(stream))
    edges = [tuple(sorted(int(u) for u in edge)) for edge in raw.values()]
    return edges, labels


def _float_profile(profile):
    p = np.asarray([[float(x) for x in row] for row in profile], dtype=np.float64)
    p[:, -1] = 1.0 - p[:, :-1].sum(axis=1)
    return p


def prepare_directions(protocol):
    started = time.time()
    output = {"protocol_id": protocol["protocol_id"], "status": "directions prepared before any N4 sampling",
              "protocol_sha256": sha256(PROTOCOL_PATH), "data_sha256": {},
              "amendment_sha256": _verify_execution_amendment(protocol),
              "code_sha256": {}, "cells": [], "wall_seconds": None}
    cells = list(protocol["native_cells"])
    for cell_index, cell in enumerate(cells):
        graph, epsilon = cell["graph"], cell["epsilon"]
        edges, labels = _load_native(graph)
        node_degrees = np.bincount([u for edge in edges for u in edge],
                                   minlength=len(labels))
        for filename in ("hypergraph.pickle", "labels.pickle"):
            output["data_sha256"][f"{graph}/{filename}"] = sha256(DATA / graph / filename)
        profile = rational_profile(labels, epsilon, cell["profile"])
        P = _float_profile(profile)
        groups = select_structural_groups(edges, len(labels))["groups"]
        pe = edge_marginals(edges, profile)
        cell_out = {"id": cell["id"], "profile": cell["profile"], "epsilon": epsilon,
                    "groups": []}
        for group_index, group in enumerate(groups):
            group_out = {"nodes": list(group), "states": []}
            for state, probability in group_states(profile, group):
                token = f"{cell['id']}|{group}|{state}"
                mean = conditional_edge_moments(edges, P, group, state)["mean"]
                direction, ritz = _deterministic_witness(mean, token, node_degrees)
                nz = np.flatnonzero(direction)
                exact_direction = [Fraction.from_float(float(direction[i])) for i in nz]
                moments = exact_directional_moments(edges, profile, group, state,
                                                    [Fraction.from_float(float(x)) for x in direction])
                raw_mean = moments["mean"]
                if raw_mean < 0:
                    direction = -direction
                    exact_direction = [-x for x in exact_direction]
                    raw_mean = -raw_mean
                digest = hashlib.sha256(direction.tobytes()).hexdigest()
                group_out["states"].append({
                    "state": list(state), "probability_exact": f"{probability.numerator}/{probability.denominator}",
                    "direction_nodes": nz.astype(int).tolist(),
                    "direction_float_hex": [float(direction[i]).hex() for i in nz],
                    "direction_sha256": digest, "conditional_mean_exact": f"{raw_mean.numerator}/{raw_mean.denominator}",
                    "conditional_variance_exact": f"{moments['variance'].numerator}/{moments['variance'].denominator}",
                    "direction_norm_squared_exact": f"{moments['norm_squared'].numerator}/{moments['norm_squared'].denominator}",
                    "coordinate_diagonal_proposal": ritz,
                    "overlap_pair_count": moments["overlap_pair_count"]})
            cell_out["groups"].append(group_out)
        output["cells"].append(cell_out)
        print(f"prepared {cell['id']}: {sum(len(g['states']) for g in cell_out['groups'])} state directions",
              flush=True)
    for path in (Path(__file__), HERE / "verified_norm.py", HERE / "exact_certificate.py",
                 HERE / "conditional_moments.py", HERE / "native_protocol.py",
                 HERE / "finite_sample.py", HERE.parent / "upgrade" / "influence_families.py"):
        output["code_sha256"][path.name] = sha256(path)
    output["wall_seconds"] = time.time() - started
    _atomic_json(PREPARED_PATH, output)
    print(f"saved frozen directions: {PREPARED_PATH}; {output['wall_seconds']:.1f}s", flush=True)
    return output


def _native_draw(generator, arm, edges, P, pe):
    if arm == "A":
        uniforms = generator.random(len(P))
        predictions = np.fromiter((np.searchsorted(np.cumsum(row), u, side="right")
                                   for row, u in zip(P, uniforms)), dtype=np.int16,
                                  count=len(P))
        predictions = np.minimum(predictions, P.shape[1] - 1)
        return np.fromiter((int(np.all(predictions[list(edge)] == predictions[edge[0]]))
                            for edge in edges), dtype=np.int8, count=len(edges))
    return (generator.random(len(edges)) < pe).astype(np.int8)


def _interval_for_draw(builder, indicators):
    return retry_enclosure(builder, indicators)


def _native_cell_inputs(cell):
    edges, labels = _load_native(cell["graph"])
    profile = rational_profile(labels, cell["epsilon"], cell["profile"])
    P = _float_profile(profile)
    marginals = edge_marginals(edges, profile)
    builder = ExactOperatorBuilder(len(labels), edges, marginals)
    return edges, profile, P, np.asarray([float(x) for x in marginals]), builder


def _generate_stream(cell, arm, seed, count):
    edges, profile, P, pe, builder = _native_cell_inputs(cell)
    generator = np.random.default_rng(int(seed))
    intervals = []
    for draw in range(count):
        indicators = _native_draw(generator, arm, edges, P, pe)
        intervals.append(_interval_for_draw(builder, indicators))
        if (draw + 1) % 100 == 0 or draw + 1 == count:
            print(f"{cell['id']} {arm} seed={seed}: {draw+1}/{count}", flush=True)
    return intervals


def _controlled_inputs(spec):
    edges, profile = controlled_profile(spec)
    marginals = edge_marginals(edges, profile)
    builder = ExactOperatorBuilder(len(profile), edges, marginals)
    return edges, _float_profile(profile), np.asarray([float(x) for x in marginals]), builder


def _generate_control_stream(spec, arm, seed, count):
    edges, P, pe, builder = _controlled_inputs(spec)
    generator = np.random.default_rng(int(seed))
    intervals = []
    for draw in range(count):
        indicators = _native_draw(generator, arm, edges, P, pe)
        intervals.append(_interval_for_draw(builder, indicators))
        if (draw + 1) % 100 == 0 or draw + 1 == count:
            print(f"{spec['id']} {arm} seed={seed}: {draw+1}/{count}", flush=True)
    return intervals


def run_calibration(protocol):
    amendment_hash = _verify_execution_amendment(protocol)
    if not PREPARED_PATH.exists():
        raise RuntimeError("directions must be prepared before generating formal samples")
    prepared = json.loads(PREPARED_PATH.read_text(encoding="utf-8"))
    if prepared["protocol_sha256"] != sha256(PROTOCOL_PATH):
        raise RuntimeError("prepared directions are for a different protocol")
    if prepared.get("amendment_sha256") != amendment_hash:
        raise RuntimeError("prepared directions are for a different execution amendment")
    n = protocol["finite_sample_calibration"]["calibration_draws_per_cell"]
    alpha = Fraction(protocol["finite_sample_calibration"]["alpha"])
    delta = Fraction(protocol["finite_sample_calibration"]["per_cell_failure_probability"])
    rank = tolerance_order_statistic_rank(n, alpha, delta)
    if rank != protocol["finite_sample_calibration"]["order_statistic_rank_one_based"]:
        raise RuntimeError("native calibration rank differs from frozen protocol")
    output = {"protocol_sha256": sha256(PROTOCOL_PATH), "directions_sha256": sha256(PREPARED_PATH),
              "amendment_sha256": amendment_hash,
              "status": "formal B calibration only", "cells": [], "wall_seconds": None}
    start = time.time()
    for cell in protocol["native_cells"]:
        records = _generate_stream(cell, "B", cell["B_calibration_seed"], n)
        scores = [r["hi"] for r in records]
        threshold = order_statistic_threshold(scores, rank)
        output["cells"].append({"id": cell["id"], "rank": rank, "threshold": threshold,
                                "upper_scores": scores,
                                "failed_draws": [i for i, r in enumerate(records) if "failure" in r],
                                "maximum_interval_width": max(r["hi"] - r["lo"] for r in records)})
        output["wall_seconds"] = time.time() - start
        _atomic_json(CALIBRATION_PATH, output)
        print(f"CALIBRATED {cell['id']}: tau={threshold:.9g}, failures={len(output['cells'][-1]['failed_draws'])}",
              flush=True)
    controls = protocol["controlled_boundary"]
    control_n = controls["B_calibration_draws"]
    control_rank = tolerance_order_statistic_rank(
        control_n, alpha, Fraction(controls["per_cell_calibration_failure_probability"]))
    if control_rank != controls["order_statistic_rank_one_based"]:
        raise RuntimeError("controlled-boundary calibration rank differs from protocol")
    output["controlled_boundary"] = []
    for spec in controls["cells"]:
        records = _generate_control_stream(spec, "B", spec["B_calibration_seed"], control_n)
        scores = [record["hi"] for record in records]
        threshold = order_statistic_threshold(scores, control_rank)
        output["controlled_boundary"].append({
            "id": spec["id"], "rank": control_rank, "threshold": threshold,
            "upper_scores": scores,
            "failed_draws": [i for i, row in enumerate(records) if "failure" in row],
            "maximum_interval_width": max(row["hi"] - row["lo"] for row in records)})
        output["wall_seconds"] = time.time() - start
        _atomic_json(CALIBRATION_PATH, output)
        print(f"CALIBRATED CONTROL {spec['id']}: tau={threshold:.9g}", flush=True)
    output["wall_seconds"] = time.time() - start
    _atomic_json(CALIBRATION_PATH, output)
    by_id = {c["id"]: c for c in prepared["cells"]}
    cert_out = {"protocol_sha256": sha256(PROTOCOL_PATH),
                "amendment_sha256": amendment_hash,
                "directions_sha256": sha256(PREPARED_PATH),
                "calibration_sha256": sha256(CALIBRATION_PATH),
                "status": "precommit certificates locked before opening any A/B validation draws",
                "cells": [], "controlled_boundary": [
                    {"id": row["id"], "threshold": row["threshold"], "rank": row["rank"]}
                    for row in output["controlled_boundary"]]}
    pass_graphs = set()
    for calibration_cell in output["cells"]:
        source = next(c for c in protocol["native_cells"] if c["id"] == calibration_cell["id"])
        threshold = Fraction.from_float(calibration_cell["threshold"])
        settings = by_id[source["id"]]
        group_results = []
        for group in settings["groups"]:
            lower = Fraction(0)
            states = []
            for state in group["states"]:
                pi = Fraction(state["probability_exact"])
                value = exact_cantelli_lower(Fraction(state["conditional_mean_exact"]),
                                             Fraction(state["conditional_variance_exact"]),
                                             Fraction(state["direction_norm_squared_exact"]),
                                             threshold)
                lower += pi * value
                states.append({"state": state["state"], "lower_exact": f"{value.numerator}/{value.denominator}"})
            group_results.append({"nodes": group["nodes"], "lower_exact": f"{lower.numerator}/{lower.denominator}",
                                  "lower": float(lower), "states": states})
            if lower >= Fraction(7, 100):
                pass_graphs.add(source["graph"])
        cert_out["cells"].append({"id": source["id"], "graph": source["graph"],
                                  "threshold": calibration_cell["threshold"], "groups": group_results})
    cert_out["primary_certificate_gate_met"] = len(pass_graphs) == 2
    cert_out["graphs_with_registered_group_pass"] = sorted(pass_graphs)
    _atomic_json(CERTIFICATE_PATH, cert_out)
    print(f"PRECOMMIT CERTIFICATE LOCKED: graph gates={sorted(pass_graphs)}; {CERTIFICATE_PATH}", flush=True)
    return cert_out


def _evaluate_native(protocol):
    amendment_hash = _verify_execution_amendment(protocol)
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    certificate = json.loads(CERTIFICATE_PATH.read_text(encoding="utf-8"))
    expected = sha256(PROTOCOL_PATH)
    for record in (calibration, certificate):
        if record["protocol_sha256"] != expected:
            raise RuntimeError("formal artifacts do not match the frozen protocol")
        if record.get("amendment_sha256") != amendment_hash:
            raise RuntimeError("formal artifacts do not match the execution amendment")
    if certificate["calibration_sha256"] != sha256(CALIBRATION_PATH):
        raise RuntimeError("precommit certificate does not match calibration artifact")
    alpha = float(Fraction(protocol["finite_sample_calibration"]["alpha"]))
    family_conf = 1.0 - 0.05 / len(protocol["native_cells"])
    threshold_by_id = {c["id"]: c["threshold"] for c in calibration["cells"]}
    # Multiple preselected groups may occur in a cell; retain the strongest
    # registered lower bound, without using validation outcomes.
    lower_by_id = {row["id"]: max((Fraction(group["lower_exact"])
                                   for group in row["groups"]), default=Fraction(0))
                   for row in certificate["cells"]}
    output = {"protocol_sha256": expected, "calibration_sha256": sha256(CALIBRATION_PATH),
              "amendment_sha256": amendment_hash,
              "certificate_sha256": sha256(CERTIFICATE_PATH),
              "status": "formal native evaluation; all draws retained", "cells": []}
    gate_rows = {}
    for cell in protocol["native_cells"]:
        tau = threshold_by_id[cell["id"]]
        batch_records = {}
        for arm, key in (("A", "A_batch_1_seed"), ("A", "A_batch_2_seed"),
                         ("B", "B_batch_1_seed"), ("B", "B_batch_2_seed")):
            intervals = _generate_stream(cell, arm, cell[key], 1000)
            summary = count_envelope([(r["lo"], r["hi"]) for r in intervals], tau)
            s, f, n = summary["definite_alarm"], summary["ambiguous"], summary["draws"]
            low_ci = exact_binomial_interval(s, n, family_conf)
            high_ci = exact_binomial_interval(s + f, n, family_conf)
            summary["simultaneous_adjusted_ci_envelope"] = [low_ci[0], high_ci[1]]
            summary["descriptive_95pct_ci_envelope"] = [
                exact_binomial_interval(s, n, .95)[0], exact_binomial_interval(s + f, n, .95)[1]]
            summary["max_interval_width"] = max(r["hi"] - r["lo"] for r in intervals)
            summary["failed_draw_indices"] = [i for i, r in enumerate(intervals) if "failure" in r]
            batch_records[f"{arm}_{key}"] = summary
            if arm == "A":
                gate_rows.setdefault(cell["id"], []).append(summary)
        a_batches = [batch_records["A_A_batch_1_seed"], batch_records["A_A_batch_2_seed"]]
        pooled_s = sum(row["definite_alarm"] for row in a_batches)
        pooled_f = sum(row["ambiguous"] for row in a_batches)
        pooled_n = sum(row["draws"] for row in a_batches)
        pooled_ci_low = exact_binomial_interval(pooled_s, pooled_n, family_conf)[0]
        pooled_ci_high = exact_binomial_interval(pooled_s + pooled_f, pooled_n, family_conf)[1]
        pooled = {"draws": pooled_n, "definite_alarm": pooled_s,
                  "ambiguous": pooled_f, "rate_envelope": [pooled_s / pooled_n,
                                                                  (pooled_s + pooled_f) / pooled_n],
                  "simultaneous_adjusted_ci_envelope": [pooled_ci_low, pooled_ci_high]}
        output["cells"].append({"id": cell["id"], "threshold": tau,
                                "batches": batch_records, "A_pooled": pooled,
                                "precommitted_L": float(lower_by_id[cell["id"]])})
        _atomic_json(EVALUATION_PATH, output)
        print(f"EVALUATED {cell['id']}: tau={tau:.7g}", flush=True)
    output["certificate_gate_met"] = certificate["primary_certificate_gate_met"]
    output["task_consequence_gate_met"] = any(
        primary_task_consequence_gate(gate_rows[cell["id"]],
                                      lower_by_id[cell["id"]],
                                      next(row["A_pooled"]["simultaneous_adjusted_ci_envelope"][0]
                                           for row in output["cells"] if row["id"] == cell["id"]),
                                      alpha)
        for cell in protocol["native_cells"])
    output["multiclass_nonempty_certificate_gate_met"] = any(
        cell["profile"] == "symmetric_multiclass" and lower_by_id[cell["id"]] > 0
        for cell in protocol["native_cells"])
    output["all_native_cells_reported"] = len(output["cells"]) == len(protocol["native_cells"])
    output["controlled_boundary"] = []
    control_calibration = {row["id"]: row["threshold"]
                           for row in calibration["controlled_boundary"]}
    for spec in protocol["controlled_boundary"]["cells"]:
        rows = {}
        for arm, seed_key, count_key in (
                ("A", "A_seed", "A_draws_per_cell"),
                ("B", "B_validation_seed", "B_validation_draws_per_cell")):
            intervals = _generate_control_stream(spec, arm, spec[seed_key],
                                                 protocol["controlled_boundary"][count_key])
            summary = count_envelope([(r["lo"], r["hi"]) for r in intervals],
                                     control_calibration[spec["id"]])
            summary["descriptive_95pct_ci_envelope"] = [
                exact_binomial_interval(summary["definite_alarm"], summary["draws"], .95)[0],
                exact_binomial_interval(summary["definite_alarm"] + summary["ambiguous"],
                                        summary["draws"], .95)[1]]
            summary["failed_draw_indices"] = [i for i, r in enumerate(intervals)
                                               if "failure" in r]
            rows[arm] = summary
        output["controlled_boundary"].append({
            "id": spec["id"], "threshold": control_calibration[spec["id"]],
            "A": rows["A"], "B": rows["B"]})
        _atomic_json(EVALUATION_PATH, output)
        print(f"EVALUATED CONTROL {spec['id']}", flush=True)
    output["interpretation"] = "Controlled probability profiles on two fixed native hypergraphs only."
    _atomic_json(EVALUATION_PATH, output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "calibrate", "evaluate"), required=True)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if args.stage == "prepare":
        prepare_directions(protocol)
    elif args.stage == "calibrate":
        run_calibration(protocol)
    else:
        _evaluate_native(protocol)


if __name__ == "__main__":
    main()
