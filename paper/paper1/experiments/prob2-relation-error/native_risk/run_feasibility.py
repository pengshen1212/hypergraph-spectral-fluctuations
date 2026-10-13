"""Run the frozen native-risk feasibility pilot (N0-N2 only)."""

from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.stats import binomtest

from conditional_moments import (
    conditional_edge_moments,
    directional_conditional_moments,
    enumerate_group_states,
)
from native_protocol import (
    majority_binary_profile,
    pilot_seed_key,
    select_structural_groups,
)
from risk_certificate import cantelli_tail_lower


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PAPER = ROOT / "paper" / "paper1"
UPGRADE = PAPER / "experiments" / "prob2-relation-error" / "upgrade"
DATA = ROOT / "data"
NOTES = PAPER / "notes" / "native-risk"
PROTOCOL_PATH = HERE / "protocol.json"
sys.path.insert(0, str(UPGRADE))

from influence_families import beta_centered, sample_weights_A, sample_weights_B  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_native(tag):
    directory = DATA / tag
    with (directory / "hypergraph.pickle").open("rb") as stream:
        hypergraph = pickle.load(stream)
    with (directory / "labels.pickle").open("rb") as stream:
        labels = np.asarray(pickle.load(stream))
    edges = [tuple(sorted(int(u) for u in edge)) for edge in hypergraph.values()]
    selection = select_structural_groups(edges, len(labels))
    return edges, labels, selection


def spectral_witness(matrix):
    """Compute both spectral extremes on the nonzero support of a sparse mean."""
    rows, cols = matrix.nonzero()
    support = np.unique(np.concatenate((rows, cols)))
    if len(support) == 0:
        return {
            "x": np.zeros(matrix.shape[0]),
            "eigenvalue": 0.0,
            "rayleigh": 0.0,
            "residual": 0.0,
            "support_size": 0,
            "eigen_gap_estimate": None,
        }
    local = matrix[support][:, support].tocsr()
    m = len(support)
    if m <= 256:
        vals, vecs = np.linalg.eigh(local.toarray())
        order = np.argsort(np.abs(vals))
        index = int(order[-1])
        eigenvalue = float(vals[index])
        vector = vecs[:, index]
        gap = float(abs(vals[order[-1]]) - abs(vals[order[-2]])) if m > 1 else None
    else:
        largest, largest_vec = eigsh(local, k=1, which="LA", tol=1e-9,
                                     maxiter=100000)
        smallest, smallest_vec = eigsh(local, k=1, which="SA", tol=1e-9,
                                       maxiter=100000)
        if abs(largest[0]) >= abs(smallest[0]):
            eigenvalue, vector = float(largest[0]), largest_vec[:, 0]
        else:
            eigenvalue, vector = float(smallest[0]), smallest_vec[:, 0]
        gap = None
    x = np.zeros(matrix.shape[0])
    x[support] = vector
    norm = np.linalg.norm(x)
    x /= norm
    rayleigh = float(x @ (matrix @ x))
    residual = float(np.linalg.norm(matrix @ x - eigenvalue * x))
    return {
        "x": x,
        "eigenvalue": eigenvalue,
        "rayleigh": rayleigh,
        "residual": residual,
        "support_size": int(m),
        "eigen_gap_estimate": gap,
    }


def state_moments(edges, P, group):
    records = []
    mixture_mean = None
    for state, probability in enumerate_group_states(P, group):
        info = conditional_edge_moments(edges, P, group, state)
        mean_matrix = info["mean"]
        witness = spectral_witness(mean_matrix)
        direction_mean, direction_variance = directional_conditional_moments(
            edges, P, group, state, witness["x"]
        )
        rayleigh_mismatch = abs(direction_mean - witness["rayleigh"])
        tolerance = 1e-8 * (1.0 + abs(direction_mean))
        if rayleigh_mismatch > tolerance:
            raise ArithmeticError(
                f"directional mean mismatch for group={group}, state={state}: "
                f"{rayleigh_mismatch:g}"
            )
        variance_correction = 0.0
        if direction_variance < 0:
            if direction_variance < -1e-10 * (1 + direction_mean * direction_mean):
                raise ArithmeticError(
                    f"negative conditional variance {direction_variance:g}"
                )
            variance_correction = -direction_variance
            direction_variance = 0.0
        if mixture_mean is None:
            mixture_mean = probability * mean_matrix
        else:
            mixture_mean = mixture_mean + probability * mean_matrix
        records.append({
            "state": list(state),
            "probability": float(probability),
            "conditional_mean_direction": float(direction_mean),
            "mu_abs": float(abs(direction_mean)),
            "conditional_direction_variance": float(direction_variance),
            "variance_roundoff_clipped": float(variance_correction),
            "mean_operator_norm_eigen_estimate": float(abs(witness["eigenvalue"])),
            "rayleigh_value": float(witness["rayleigh"]),
            "eigen_residual": float(witness["residual"]),
            "eigen_gap_estimate": witness["eigen_gap_estimate"],
            "witness_support_size": witness["support_size"],
            "witness_x": witness["x"],
        })
    mixture_residual = float(np.max(np.abs(mixture_mean.data))) if mixture_mean.nnz else 0.0
    for record in records:
        record["cantelli_at_tau"] = None
        record["active_for_tail"] = None
        record.pop("witness_x")
    return records, mixture_residual


def simulate_norms(edges, P, n_draws, seed, arm):
    rng = np.random.default_rng(seed)
    output = np.empty(n_draws, dtype=float)
    for i in range(n_draws):
        weights = (sample_weights_A(edges, P, rng) if arm == "A"
                   else sample_weights_B(edges, P, rng))
        output[i] = beta_centered(len(P), edges, weights)
        if (i + 1) % 64 == 0 or i + 1 == n_draws:
            print(f"    {arm}: {i + 1}/{n_draws}", flush=True)
    return output


def proportion_record(values, threshold):
    successes = int(np.count_nonzero(values > threshold))
    interval = binomtest(successes, len(values)).proportion_ci(
        confidence_level=0.95, method="exact"
    )
    return {
        "draws": int(len(values)),
        "exceedances": successes,
        "rate": successes / len(values),
        "exact_95_ci": [float(interval.low), float(interval.high)],
        "mean_norm": float(np.mean(values)),
    }


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def main():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    draws = int(protocol["pilot"]["draws_per_arm_per_cell"])
    seed_settings = protocol["pilot"]["seeds"]
    output = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "status": "exploratory N2 pilot; float64 analytic moments and independent pilot draws",
        "started_unix": time.time(),
        "inputs": {},
        "settings": [],
    }
    for tag in ("cora-cc", "dblp-ca"):
        edges, labels, selection = load_native(tag)
        data_dir = DATA / tag
        output["inputs"][tag] = {
            "hypergraph_sha256": sha256(data_dir / "hypergraph.pickle"),
            "labels_sha256": sha256(data_dir / "labels.pickle"),
            "n_nodes": int(len(labels)),
            "n_edge_slots": int(len(edges)),
            "selected_hub": selection["hub"],
            "selected_degree": int(selection["degree"][selection["hub"]]),
            "selected_partner": selection["partner"],
            "partner_score": selection["partner_score"],
            "groups": [list(group) for group in selection["groups"]],
        }
        for epsilon in (0.05, 0.15):
            P, majority = majority_binary_profile(labels, epsilon)
            key = pilot_seed_key(tag, epsilon)
            seeds = seed_settings[key]
            print(f"== {key}: groups={selection['groups']} ==", flush=True)
            setting = {
                "tag": tag,
                "epsilon": epsilon,
                "majority_label": int(majority),
                "draws_per_arm": draws,
                "groups": [],
                "pilot_B_threshold": None,
                "pilot_A_false_alarm": None,
                "pilot_B_false_alarm": None,
            }
            for group in selection["groups"]:
                state_rows, mixture_residual = state_moments(edges, P, group)
                setting["groups"].append({
                    "group": list(group),
                    "state_probability_sum": sum(row["probability"] for row in state_rows),
                    "mean_mixture_max_abs_entry": mixture_residual,
                    "states": state_rows,
                })
            print("  analytic states computed; generating B calibration", flush=True)
            b_cal = simulate_norms(edges, P, draws,
                                   int(seeds["B_calibration"]), "B")
            tau = float(np.quantile(b_cal, 0.95, method="higher"))
            setting["pilot_B_threshold"] = {
                "value": tau,
                "method": "empirical 0.95 quantile, higher order statistic",
                "draws": draws,
                "explicitly_not_a_finite_sample_guarantee": True,
            }
            for group_row in setting["groups"]:
                lower = 0.0
                active_mass = 0.0
                for row in group_row["states"]:
                    mu = row["mu_abs"]
                    active = mu > tau
                    row["active_for_tail"] = bool(active)
                    row["cantelli_at_tau"] = cantelli_tail_lower(
                        mu, row["conditional_direction_variance"], tau
                    )
                    lower += row["probability"] * row["cantelli_at_tau"]
                    if active:
                        active_mass += row["probability"]
                group_row["active_state_mass_upper_cap"] = active_mass
                group_row["weighted_Cantelli_lower_at_pilot_tau"] = lower
            print("  generating independent A evaluation", flush=True)
            A = simulate_norms(edges, P, draws, int(seeds["A_evaluation"]), "A")
            print("  generating independent B evaluation", flush=True)
            B = simulate_norms(edges, P, draws, int(seeds["B_evaluation"]), "B")
            setting["pilot_A_false_alarm"] = proportion_record(A, tau)
            setting["pilot_B_false_alarm"] = proportion_record(B, tau)
            output["settings"].append(setting)
            atomic_json(NOTES / "feasibility_pilot.json", output)
            print(
                f"[{key}] tau={tau:.4g} "
                f"A={setting['pilot_A_false_alarm']['rate']:.3f} "
                f"B_eval={setting['pilot_B_false_alarm']['rate']:.3f} "
                f"L_single/pair=" + ",".join(
                    f"{r['weighted_Cantelli_lower_at_pilot_tau']:.4g}"
                    for r in setting["groups"]
                ),
                flush=True,
            )
    output["finished_unix"] = time.time()
    output["wall_seconds"] = output["finished_unix"] - output["started_unix"]
    output["code_sha256"] = {
        path.name: sha256(path)
        for path in (HERE / "conditional_moments.py",
                     HERE / "native_protocol.py",
                     HERE / "risk_certificate.py",
                     HERE / "run_feasibility.py")
    }
    try:
        output["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        output["git_head"] = None
    atomic_json(NOTES / "feasibility_pilot.json", output)
    print(f"Wrote {NOTES / 'feasibility_pilot.json'}", flush=True)


if __name__ == "__main__":
    main()
