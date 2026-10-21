"""Statistical and numerical regressions for the Paper1 evidence repair."""
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "src"))


def test_multibatch_confirmation_does_not_claim_pooled_binomial_coverage():
    import audit_evidence as audit
    row = dict(claim="C1", structure="cora-cc", metric="node", n="3000",
               estimate=str(112 / 3000), batches="3", cp_lo="0.03084",
               cp_hi="0.04475", q_mean="0.78")
    result = audit.summarize_confirmation([row])[0]
    assert result["count"] == 112
    assert result["interval_method"] == "conditional_hoeffding"
    assert result["ci_lo"] <= 0.05 <= result["ci_hi"]
    assert result["historical_cp_valid"] is False
    assert result["estimand"] == "mean_conditional_exceedance_across_saved_batches"


def test_single_batch_extremes_have_intervals_even_when_csv_omits_them():
    import audit_evidence as audit
    rows = [dict(claim="C3", structure="cora-cc", metric=str(k), n="400",
                 estimate=str(k / 400), batches="1", cp_lo="", cp_hi="",
                 q_mean="") for k in (0, 394)]
    one, two = audit.summarize_confirmation(rows)
    assert one["ci_lo"] == 0.0 and 0.0 < one["ci_hi"] < 0.02
    assert 0.95 < two["ci_lo"] < 0.985 < two["ci_hi"] < 1.0


def test_saved_rate_must_represent_an_integer_count():
    import audit_evidence as audit
    row = dict(claim="bad", structure="toy", metric="bad", n="3",
               estimate="0.5", batches="1", cp_lo="", cp_hi="", q_mean="")
    with pytest.raises(ValueError, match="integer"):
        audit.summarize_confirmation([row])


def test_amplitude_scan_measures_matrix_remainder_not_difference_of_norms():
    import hyperops as ho
    import s10a_amplitude_scan as scan
    op = ho.RelationOperator(5, [(0, 1, 2), (0, 3), (2, 3, 4)])
    baseline = np.array([0.3, 0.6, 0.4])
    draw = np.array([1.0, 0.0, 1.0])
    t = 0.7
    basis = op.response_basis(baseline)
    true, linear, remainder, scalar_gap = scan.amplitude_metrics(op, basis, draw, t)
    _, _, j, _, _ = op.linear_response(draw, basis)
    actual_matrix = op.csr(op.S_data(baseline + t * (draw - baseline))
                          - basis["S0"]).toarray()
    linear_matrix = op.csr(t * j).toarray()
    expected = np.linalg.norm(actual_matrix - linear_matrix, 2)
    assert remainder == pytest.approx(expected, abs=1e-11)
    assert scalar_gap == pytest.approx(abs(true - linear), abs=1e-11)
    assert remainder > scalar_gap + 1e-4


def test_report_distinguishes_guarantee_and_historical_diagnostics(tmp_path, monkeypatch):
    import build_gate_report as report
    (tmp_path / "docs").mkdir()
    monkeypatch.setattr(report, "HERE", tmp_path)
    report.main()
    rendered = (tmp_path / "docs" / "gate_ab_report.md").read_text(encoding="utf-8")
    assert "Hoeffding" in rendered
    assert "矩阵剩余项" in rendered
    assert "S12 未新增 L3" in rendered
    assert "唯一有效校准" not in rendered
    assert "各臂分别中心化（S2、S9a）" not in rendered
    assert "不受该 bug 影响" not in rendered


def test_random_direction_control_uses_its_own_vector_norm():
    import hyperops as ho
    import s9b_directional_detection as directions
    op = ho.RelationOperator(7, [(0, 1), (1, 2)])
    vectors = directions.random_directions(op, 711)
    assert all(np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-13)
               for vector in vectors)


def test_reference_eigendirections_have_an_explicit_deterministic_start(monkeypatch):
    import hyperops as ho
    import s9b_directional_detection as directions
    op = ho.RelationOperator(601, [(0, 1), (1, 2)])
    starts = []
    def inspect_start(matrix, **kwargs):
        assert "v0" in kwargs
        starts.append(kwargs["v0"].copy())
        return np.array([1.0]), np.ones((601, 1)) / np.sqrt(601)
    monkeypatch.setattr(directions, "eigsh", inspect_start)
    got = directions._principal_pair(op, op.S_data(np.ones(2)))
    assert len(got) == 2
    assert len(starts) == 2 and np.array_equal(starts[0], starts[1])


def test_scan_exports_actual_remainder_and_lower_bound_separately(monkeypatch):
    import hyperops as ho
    import s10a_amplitude_scan as scan
    op = ho.RelationOperator(5, [(0, 1, 2), (0, 3), (2, 3, 4)])
    probabilities = np.tile([0.6, 0.4], (5, 1))
    monkeypatch.setattr(scan, "T_GRID", np.array([0.3, 0.7]))
    rows = scan.scan("toy", "known", op, probabilities, 5, 311)
    for row in rows[:-1]:
        assert row["resid_frac_node"] >= row["norm_gap_frac_node"] - 1e-12
        assert row["max_solver_resid"] == 0.0
    assert "norm_gap_frac_node" in scan.FIELDS


def test_data_root_follows_a_relocated_reproduction_package(tmp_path):
    import importlib.util
    import shutil
    root = tmp_path / "relocated-research"
    target = root / "experiments/spectral-risk-calibration/src/data_io.py"
    target.parent.mkdir(parents=True)
    shutil.copyfile(HERE / "src/data_io.py", target)
    spec = importlib.util.spec_from_file_location("relocated_data_io", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.ROOT) == root
