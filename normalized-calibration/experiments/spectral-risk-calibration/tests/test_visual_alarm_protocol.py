"""Protocol tests for the visual alarm validation.

These tests check the failure modes named in the plan: object/auxiliary-row
confusion, missing or duplicate IDs, split leakage, rank-threshold boundaries,
tie conservatism, monitored-operator/refinement consistency, test-label
isolation, seed determinism, cache config mixing and noise placement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "scripts"))

import hyperops as ho  # noqa: E402
import monitoring_workflow as mw  # noqa: E402
import vision_data as vd  # noqa: E402
import vision_predictor as vp  # noqa: E402
import vision_runs as vr  # noqa: E402


def test_equal_fpr_reports_observed_difference_separately_from_bootstrap_mean():
    from s15c_equal_fpr import bootstrap_diff

    clean = np.arange(20, dtype=float)
    data = {"T": np.column_stack((clean, clean + 5)),
            "frob": np.column_stack((clean, clean + 10))}
    summary = bootstrap_diff(data, sigma_index=1, reps=50, seed=9)
    # At 10% FPR the original ROC has TPRs 35% and 60%.
    assert summary["tpr_diff_10"]["estimate"] == pytest.approx(-0.25)
    assert abs(summary["tpr_diff_10"]["mean"] + 0.25) > 1e-5


N_OBJECTS, N_CLASSES, DIM, EDGE_SIZE = 6, 2, 3, 3
LABELS = np.array([0, 0, 0, 1, 1, 1])
FEATURES = np.arange(N_OBJECTS * DIM, dtype=float).reshape(N_OBJECTS, DIM) / 10.0
EDGES = {
    6: (0, 1, 2), 7: (0, 3, 4), 8: (1, 3, 5),
    9: (2, 4, 5), 10: (0, 2, 5), 11: (1, 3, 4),
}


def _write_fixture(tmp_path, *, drop_id=None, duplicate=False, shuffle=True):
    rows = []
    for u in range(N_OBJECTS):
        vals = "\t".join(f"{v:.6f}" for v in FEATURES[u])
        rows.append(f"{u}\t{vals}\t{LABELS[u]}")
    for e in EDGES:
        vals = "\t".join("1.0" for _ in range(DIM))
        rows.append(f"{e}.0\t{vals}\t0")
    if drop_id is not None:
        rows = [r for r in rows if int(float(r.split("\t")[0])) != drop_id]
    if duplicate:
        rows = rows + [rows[1]]
    if shuffle:
        rows = list(reversed(rows))
    content = tmp_path / "toy.content"
    content.write_text("\n".join(rows) + "\n", encoding="utf-8")
    lines = []
    for e, members in EDGES.items():
        for u in members:
            lines.append(f"{u}\t{e}")
    if shuffle:
        lines = list(reversed(lines))
    edges = tmp_path / "toy.edges"
    edges.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return content, edges


def _load(tmp_path, **kw):
    content, edges = _write_fixture(tmp_path, **kw)
    return vd.parse_vision_files(
        content, edges, n_objects=N_OBJECTS, n_classes=N_CLASSES,
        label_offset=0, feature_dim=DIM, edge_size=EDGE_SIZE,
        source_name="toy")


def test_shuffled_rows_and_auxiliary_nodes_parse_by_id(tmp_path):
    data = _load(tmp_path)
    assert data.X.shape == (N_OBJECTS, DIM)
    assert data.n_extra_rows == N_OBJECTS
    assert np.array_equal(data.labels, LABELS)
    assert len(data.edges) == N_OBJECTS
    assert all(len(e) == EDGE_SIZE for e in data.edges)
    assert len(set(data.object_rows.tolist())) == N_OBJECTS
    assert min(data.object_rows) >= 0
    assert max(data.object_rows) < 2 * N_OBJECTS


def test_missing_object_id_fails(tmp_path):
    with pytest.raises(vd.VisionDataError, match="missing object IDs"):
        _load(tmp_path, drop_id=5)


def test_duplicate_content_id_fails(tmp_path):
    with pytest.raises(vd.VisionDataError, match="duplicate content IDs"):
        _load(tmp_path, duplicate=True)


def test_edge_column_orientation_is_object_then_hyperedge(tmp_path):
    data = _load(tmp_path)
    assert set(data.edges[0]) == set(EDGES[6])
    with pytest.raises(vd.VisionDataError):
        content, edges = _write_fixture(tmp_path)
        lines = edges.read_text().splitlines()
        bad = "\n".join(" ".join(reversed(line.split())) for line in lines)
        edges.write_text(bad + "\n", encoding="utf-8")
        vd.parse_vision_files(
            content, edges, n_objects=N_OBJECTS, n_classes=N_CLASSES,
            label_offset=0, feature_dim=DIM, edge_size=EDGE_SIZE,
            source_name="toy")


def test_split_is_disjoint_covering_and_seed_fixed():
    m1 = vd.stratified_masks(LABELS, 202610210)
    m2 = vd.stratified_masks(LABELS, 202610210)
    for k in ("train", "val", "test"):
        assert np.array_equal(m1[k], m2[k])
    assert not (m1["train"] & m1["val"]).any()
    assert not (m1["train"] & m1["test"]).any()
    assert not (m1["val"] & m1["test"]).any()
    assert (m1["train"] | m1["val"] | m1["test"]).all()
    for c in range(N_CLASSES):
        for k in ("train", "val", "test"):
            assert (m1[k] & (LABELS == c)).any()
    assert vd.split_fingerprint(m1) == vd.split_fingerprint(m2)


def test_rank_threshold_boundary_and_ties():
    scores = np.arange(1, 200, dtype=float)
    q = ho.rank_threshold(scores, 0.05)
    assert q == 190.0
    # ceil((M+1)(1-alpha)) = 190; strict exceedance is 9/199 (conservative).
    assert int(np.sum(scores > q)) == 9
    assert ho.rank_threshold(np.ones(199), 0.05) == 1.0
    assert int(np.sum(np.ones(199) > 1.0)) == 0
    assert math_is_inf(ho.rank_threshold(np.ones(10), 0.01))


def math_is_inf(x):
    return np.isinf(x)


def test_operator_and_refinement_use_the_same_S(tmp_path):
    data = _load(tmp_path)
    op = mw.operator_for(N_OBJECTS, data.edges, kappa=1.0)
    rng = np.random.default_rng(3)
    Y = rng.integers(0, N_CLASSES, N_OBJECTS)
    flags = op.flags_from_Y(Y)
    s_data = op.S_data(flags)
    S_csr = op.S_csr(flags)
    assert np.allclose(S_csr.toarray(), op.csr(s_data).toarray())
    assert np.allclose(S_csr.toarray(), S_csr.toarray().T)
    P = np.full((N_OBJECTS, N_CLASSES), 0.5)
    Q = mw.refine_probs(P, S_csr, 0.5)
    assert np.allclose(Q.sum(axis=1), 1.0)
    assert np.allclose(mw.refine_probs(P, S_csr, 0.0), P)


def test_node_law_smoothing_has_unit_mass(tmp_path):
    data = _load(tmp_path)
    runs = [np.eye(N_CLASSES)[data.labels],
            np.eye(N_CLASSES)[np.roll(data.labels, 1)]]
    P_hat = mw.fit_node_law(runs, N_CLASSES)
    assert np.allclose(P_hat.sum(axis=1), 1.0)
    assert (P_hat > 0).all()


def test_noise_only_on_non_train_and_zero_variance_columns():
    rng = np.random.default_rng(0)
    n, d = 10, 4
    train = np.zeros(n, dtype=bool)
    train[:4] = True
    zero_var = np.array([False, True, False, False])
    Z = vd.noise_matrix(n, d, train, zero_var, rng)
    assert np.all(Z[train] == 0.0)
    assert np.all(Z[:, 1] == 0.0)
    assert np.any(Z[~train][:, [0, 2, 3]] != 0.0)


def _toy_graph(tmp_path):
    data = _load(tmp_path)
    masks = vd.stratified_masks(data.labels, 7)
    op = mw.operator_for(N_OBJECTS, data.edges)
    S = mw.all_active_operator(op)
    G = np.asarray(S @ data.X)
    return data, masks, op, S, G


def test_same_seed_reproduces_and_roles_have_disjoint_seeds(tmp_path):
    data, masks, op, S, G = _toy_graph(tmp_path)
    cfg = vp.PredictorConfig(hidden=4, epochs=3, val_every=1,
                             epoch_candidates=(3,))
    a = vp.train_predictor(G, S, data.labels, masks["train"], masks["val"],
                           N_CLASSES, 123, cfg, epoch_limits=(3,))
    b = vp.train_predictor(G, S, data.labels, masks["train"], masks["val"],
                           N_CLASSES, 123, cfg, epoch_limits=(3,))
    assert np.array_equal(a["checkpoints"][3]["W1"],
                          b["checkpoints"][3]["W1"])
    pcfg = {"seed_base": 100, "roles": {"ref": 1, "cal": 2, "eval": 3,
                                        "dev": 8, "pilot": 9}}
    seeds = {r: vr.seed_for(pcfg, 1, 1, r, 0)
             for r in ("ref", "cal", "eval", "dev", "pilot")}
    assert len(set(seeds.values())) == len(seeds)


def test_test_labels_do_not_change_training_predictions_or_center(tmp_path):
    data, masks, op, S, G = _toy_graph(tmp_path)
    cfg = vp.PredictorConfig(hidden=4, epochs=5, val_every=1,
                             epoch_candidates=(5,))
    y_a = data.labels.copy()
    y_b = data.labels.copy()
    test = masks["test"]
    y_b[test] = 1 - y_b[test]          # flips only held-out labels
    ra = vp.train_predictor(G, S, y_a, masks["train"], masks["val"],
                            N_CLASSES, 77, cfg, epoch_limits=(5,))
    rb = vp.train_predictor(G, S, y_b, masks["train"], masks["val"],
                            N_CLASSES, 77, cfg, epoch_limits=(5,))
    ca, cb = ra["checkpoints"][5], rb["checkpoints"][5]
    assert np.array_equal(ca["W1"], cb["W1"])
    pa, Ya = vp.predict_full(G, S, ca["W1"], ca["W2"])
    pb, Yb = vp.predict_full(G, S, cb["W1"], cb["W2"])
    assert np.array_equal(Ya, Yb)
    assert np.allclose(mw.fit_center(op, [Ya]), mw.fit_center(op, [Yb]))


def test_cache_config_hash_mismatch_is_rejected(tmp_path):
    p = tmp_path / "run.npz"
    vr._save_npz_atomic(p, {"config_hash": "abc"})
    assert vr._existing_ok(p, "abc")
    assert not vr._existing_ok(p, "xyz")


def test_bootstrap_and_holm_are_well_formed():
    arr = {1: np.array([1.0, 0.0, 1.0, -1.0]),
           2: np.array([0.0, 1.0, 0.0, 1.0])}
    from s15_visual_alarm_analysis import holm, signflip_pvalue
    p = signflip_pvalue(arr, 200, 5)
    assert 0.0 <= p <= 1.0
    adj = holm([0.01, 0.04])
    assert adj[0] <= adj[1] and adj[1] <= 1.0


def test_real_ntu_audit_counts_and_hashes():
    content = Path(vr.ROOT) / "data/vision-hypergraphs/NTU2012.content"
    if not content.exists():
        pytest.skip("released vision data are not redistributed in this package")
    data = vd.load_vision_dataset("ntu2012", root=vr.ROOT, verify=True)
    s = vd.dataset_summary(data)
    assert s["n_objects"] == 2012
    assert s["n_edges"] == 2012
    assert s["n_classes"] == 67
    assert s["feature_dim"] == 100
    assert s["n_extra_content_rows"] == 2012
    assert s["hyperedge_size_unique"] == [5]
    assert s["content_sha256"].startswith("f59d6f62")
    assert s["edges_sha256"].startswith("3de34493")
