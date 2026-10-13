"""Independently verify saved hashes, exact risk bounds, and validation counts."""

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTES = ROOT / "paper/paper1/notes/native-risk"
UPGRADE = ROOT / "paper/paper1/experiments/prob2-relation-error/upgrade"
sys.set_int_max_str_digits(100000)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    manifest = read(ROOT / "artifact_manifest.json")
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        assert path.stat().st_size == entry["bytes"], entry["path"]
        assert digest(path) == entry["sha256"], entry["path"]

    protocol = read(NOTES / "n4r_protocol.json")
    moments = read(NOTES / "n4_spectral_supplement_certificate.json")
    certificates = read(NOTES / "n4r_precommit_certificate.json")
    calibration = read(NOTES / "n4r_calibration.json")
    evaluation = read(NOTES / "n4r_evaluation.json")
    for value in (certificates, calibration, evaluation):
        assert value["protocol_sha256"] == digest(NOTES / "n4r_protocol.json")
    assert certificates["calibration_sha256"] == digest(NOTES / "n4r_calibration.json")
    assert certificates["moments_sha256"] == digest(NOTES / "n4_spectral_supplement_certificate.json")
    assert evaluation["certificate_sha256"] == digest(NOTES / "n4r_precommit_certificate.json")
    for name, expected in protocol["source_hashes"].items():
        assert digest(ROOT / "paper/paper1" / name) == expected, name
    supplement = read(NOTES / "n4_spectral_supplement_protocol.json")
    for name, expected in supplement["source_hashes"].items():
        if not name.startswith("source-data/"):
            assert digest(ROOT / "paper/paper1" / name) == expected, name

    by_moments = {cell["id"]: cell for cell in moments["cells"]}
    by_certificate = {cell["id"]: cell for cell in certificates["cells"]}
    states_checked = 0
    for cell in certificates["cells"]:
        tau = Fraction.from_float(cell["threshold"])
        for original, certified in zip(by_moments[cell["id"]]["groups"], cell["groups"]):
            assert original["nodes"] == certified["nodes"]
            assert len(original["states"]) == len(certified["states"])
            mass, total = Fraction(0), Fraction(0)
            for state, bound in zip(original["states"], certified["states"]):
                assert state["state"] == bound["state"]
                probability = Fraction(state["probability_exact"])
                mean = Fraction(state["mean_exact"])
                variance = Fraction(state["variance_exact"])
                norm2 = Fraction(state["norm_squared_exact"])
                assert variance >= 0 and norm2 > 0
                delta = abs(mean) - tau * norm2
                lower = delta**2 / (variance + delta**2) if delta > 0 else Fraction(0)
                assert lower == Fraction(bound["lower_exact"])
                total += probability * lower
                mass += probability
                states_checked += 1
            assert mass == 1 and total == Fraction(certified["lower_exact"])
    assert states_checked == 220

    all_draws, ambiguous = 0, 0
    for cell, design in zip(evaluation["cells"], protocol["cells"]):
        assert cell["id"] == design["id"]
        threshold = by_certificate[cell["id"]]["threshold"]
        for arm in ("A", "B"):
            records = []
            for index in (1, 2):
                seed = design[f"{arm}_batch_{index}_seed"]
                raw = read(NOTES / f"n4r_raw_{cell['id']}_{arm}_{seed}.json")
                assert raw["protocol_sha256"] == digest(NOTES / "n4r_protocol.json")
                assert len(raw["records"]) == raw["count"] == 1000
                records.extend(raw["records"])
            definite = sum(record["lo"] > threshold for record in records)
            uncertain = sum(record["lo"] <= threshold < record["hi"] for record in records)
            expected = cell[f"{arm}_pooled"]
            assert definite == expected["definite_alarm"]
            assert uncertain == expected["ambiguous"]
            assert expected["rate_envelope"] == [definite / 2000, (definite + uncertain) / 2000]
            all_draws += len(records)
            ambiguous += uncertain
        calibration_raw = read(NOTES / f"n4r_raw_{cell['id']}_B_{design['B_calibration_seed']}.json")
        assert len(calibration_raw["records"]) == calibration_raw["count"] == 2000
        all_draws += len(calibration_raw["records"])
    assert all_draws == 48000 and ambiguous == 49
    original = read(NOTES / "n4_evaluation.json")
    assert not any(original[key] for key in ("certificate_gate_met", "task_consequence_gate_met",
                                            "multiclass_nonempty_certificate_gate_met"))

    for name in ("structure_validation", "native_validation", "subspace_validation"):
        result = read(UPGRADE / f"results/{name}.json")
        assert result["protocol_hash"] == digest(UPGRADE / "protocol.json")[:16]
    tables = read(ROOT / "paper_tables.json")["tables"]
    assert len(tables) == 7
    input_status = []
    for item in read(ROOT / "input_hashes.json")["inputs"]:
        path = ROOT / item["path"]
        if path.exists():
            assert digest(path) == item["sha256"], item["path"]
        else:
            input_status.append(item["path"])
    report = {"status": "passed", "artifact_files": len(manifest["files"]),
              "exact_states": states_checked, "native_cells": len(evaluation["cells"]),
              "saved_draws": all_draws, "ambiguous_intervals": ambiguous,
              "paper_tables": len(tables), "third_party_inputs_not_present": input_status}
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    verify()
