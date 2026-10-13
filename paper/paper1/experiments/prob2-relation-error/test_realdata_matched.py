"""Regression checks for the real-data matched-error controls."""

import itertools
import math

import numpy as np
import pytest
import scipy.sparse as sp

import realdata_fixed as experiment


def test_edge_marginals_match_independent_node_corruption_exactly():
    labels = np.array([0, 0, 1, 2])
    edges = [[0, 1, 2], [0, 1, 2, 3], [0, 1], [0, 2]]
    p = 0.2
    rates = experiment.edge_flip_probabilities(labels, edges, p, n_cls=3)

    for edge, rate in zip(edges, rates):
        true_hom = len({labels[i] for i in edge}) == 1
        predicted_hom = 0.0
        for predictions in itertools.product(range(3), repeat=len(edge)):
            probability = math.prod(
                (1 - p + p / 3) if pred == labels[node] else p / 3
                for node, pred in zip(edge, predictions)
            )
            if len(set(predictions)) == 1:
                predicted_hom += probability
        expected_rate = 1 - predicted_hom if true_hom else predicted_hom
        assert rate == pytest.approx(expected_rate)


def test_independent_edge_control_samples_its_own_bernoulli_draws():
    a_star = np.array([1, 1, 0, 1, 0], dtype=float)
    rates = np.full(5, 0.5)
    d_b = experiment.sample_independent_edges(np.random.default_rng(0), a_star, rates)

    np.testing.assert_array_equal(d_b, [0, -1, 1, -1, 0])
    assert not np.array_equal(d_b, np.array([-1, -1, 1, -1, 1]))


@pytest.mark.parametrize("size", [2, 3, 4, 10])
def test_all_pair_positions_reconstruct_one_hyperedge(size):
    edge = list(range(size))
    pairs = [(i, j, size, -1) for i in edge for j in edge if i < j]
    actual = experiment.clique_op_correct(size, pairs).toarray()
    expected = experiment.build_op_sparse(size, [edge], [-1]).toarray()
    np.testing.assert_allclose(actual, expected)


def test_analytic_a_b_c_expected_operators_match_on_variable_edges():
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0])
    edges = [[0, 1], [0, 1, 3], [2, 3, 4, 5], list(range(10))]
    a_star = experiment.all_hom(labels, edges)
    rates = experiment.edge_flip_probabilities(labels, edges, 0.15, n_cls=3)
    signed_rates = np.where(a_star == 1, -rates, rates)
    expected_b = experiment.build_op_sparse(len(labels), edges, signed_rates)
    expected_c_pairs = [
        (edge[i], edge[j], len(edge), signed_rates[index])
        for index, edge in enumerate(edges)
        for i in range(len(edge))
        for j in range(i + 1, len(edge))
    ]
    expected_c = experiment.clique_op_correct(len(labels), expected_c_pairs)
    np.testing.assert_allclose(expected_c.toarray(), expected_b.toarray())


def test_position_control_is_independently_sampled_from_edge_rates():
    edges = [[0, 1, 2], [1, 2, 3]]
    a_star = np.array([1.0, 0.0])
    actual = experiment.sample_independent_positions(
        np.random.default_rng(0), 4, edges, a_star, np.array([1.0, 0.0])
    )
    expected = experiment.build_op_sparse(4, [edges[0]], [-1.0])
    np.testing.assert_allclose(actual.toarray(), expected.toarray())


def test_sparse_spectral_norm_handles_both_signs():
    matrix = sp.diags([2.0, -3.0], format="csr")
    assert experiment.snorm_checked(matrix) == pytest.approx(3.0)


def test_sparse_nearest_prototypes_matches_dense_distances():
    features = sp.csr_matrix([[1, 0, 2], [0, 2, 1], [3, 0, 0]])
    prototypes = np.array([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]])
    expected = np.argmin(
        ((features.toarray()[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2),
        axis=1,
    )
    np.testing.assert_array_equal(
        experiment.nearest_prototypes(features, prototypes, block_size=2), expected
    )


def test_lanczos_failure_never_allocates_a_dense_dblp_matrix(monkeypatch):
    def fail(*_args, **_kwargs):
        raise experiment.ArpackNoConvergence("no convergence", np.array([]), np.array([]))

    monkeypatch.setattr(experiment, "eigsh", fail)
    with pytest.raises(RuntimeError, match="Lanczos"):
        experiment.snorm_checked(sp.eye(3, format="csr"), label="DBLP")
