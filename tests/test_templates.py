"""Integration tests for star-domain optimizer templates and example graph helpers."""

import numpy as np
import pytest

from vizopt.base import OptimConfig
from vizopt.components.stars import BSpline, Discrete, Fourier
from vizopt.examples.sets import (
    get_enclosing_sets,
    get_leaf_circles,
    graph_to_optimizer_inputs,
    make_british_islands_graph,
)
from vizopt.templates.stars_vs_circles import optimize_multiple_radially_convex_sets

_FAST = OptimConfig(n_iters=5, learning_rate=1e-2)
_NO_PRINT = lambda *_: None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_circle_problem(representation=None):
    """Minimal two-circle, two-set problem for smoke-testing optimizers."""
    circles = np.array([[0.0, 0.0, 0.5], [2.0, 0.0, 0.5]], dtype=np.float32)
    sets = [[0], [1]]
    return optimize_multiple_radially_convex_sets(
        circles=circles,
        sets=sets,
        representation=representation,
        optim_config=_FAST,
        callback=_NO_PRINT,
    )


# ---------------------------------------------------------------------------
# optimize_multiple_radially_convex_sets — result structure
# ---------------------------------------------------------------------------


def test_optimize_returns_three_tuple():
    results, history, problem = _two_circle_problem()
    assert isinstance(results, list)
    assert isinstance(history, list)


def test_optimize_result_length():
    results, _, _ = _two_circle_problem()
    assert len(results) == 2


def test_optimize_result_keys():
    results, _, _ = _two_circle_problem()
    for r in results:
        assert "center" in r
        assert "radii" in r
        assert "angles" in r


def test_optimize_result_shapes():
    results, _, _ = _two_circle_problem(Discrete(k_angles=16))
    for r in results:
        assert r["center"].shape == (2,)
        assert r["radii"].shape == (16,)
        assert r["angles"].shape == (16,)


def test_optimize_radii_positive():
    results, _, _ = _two_circle_problem()
    for r in results:
        assert np.all(r["radii"] > 0)


def test_optimize_history_has_term_keys():
    _, history, _ = _two_circle_problem()
    assert len(history) > 0
    for record in history:
        assert "iteration" in record
        assert "total" in record


# ---------------------------------------------------------------------------
# Representation variants
# ---------------------------------------------------------------------------


def test_optimize_fourier_representation():
    results, _, _ = _two_circle_problem(Fourier(k_angles=16, n_harmonics=4))
    for r in results:
        assert "fourier_coeffs" in r
        assert r["fourier_coeffs"].shape == (2 * 4 + 1,)
        assert r["radii"].shape == (16,)


def test_optimize_bspline_representation():
    results, _, _ = _two_circle_problem(BSpline(k_angles=16, n_ctrl_pts=8))
    for r in results:
        assert "bspline_ctrl" in r
        assert r["bspline_ctrl"].shape == (8,)
        assert r["radii"].shape == (16,)


def test_optimize_discrete_no_extra_results():
    results, _, _ = _two_circle_problem(Discrete(k_angles=16))
    for r in results:
        assert "fourier_coeffs" not in r
        assert "bspline_ctrl" not in r


# ---------------------------------------------------------------------------
# make_british_islands_graph
# ---------------------------------------------------------------------------


def test_graph_has_expected_nodes():
    G = make_british_islands_graph()
    assert "England" in G.nodes
    assert "Scotland" in G.nodes
    assert "Great Britain" in G.nodes
    assert "British Islands" in G.nodes


def test_graph_leaf_nodes_have_target_area():
    import networkx as nx
    G = make_british_islands_graph()
    for n in G.nodes:
        if G.in_degree(n) == 0:
            assert G.nodes[n]["target_area"] is not None
            assert G.nodes[n]["target_area"] > 0


def test_graph_enclosing_nodes_have_none_target_area():
    import networkx as nx
    G = make_british_islands_graph()
    for n in G.nodes:
        if G.in_degree(n) > 0:
            assert G.nodes[n]["target_area"] is None


def test_graph_england_in_great_britain():
    G = make_british_islands_graph()
    assert G.has_edge("England", "Great Britain")


def test_graph_without_ireland_island():
    G = make_british_islands_graph(include_ireland_island=False)
    assert "Ireland island" not in G.nodes
    assert G.has_edge("Republic of Ireland", "British Islands")


def test_graph_with_ireland_island():
    G = make_british_islands_graph(include_ireland_island=True)
    assert "Ireland island" in G.nodes
    assert G.has_edge("Republic of Ireland", "Ireland island")
    assert G.has_edge("Northern Ireland", "Ireland island")


# ---------------------------------------------------------------------------
# get_leaf_circles
# ---------------------------------------------------------------------------


def test_get_leaf_circles_all_in_degree_zero():
    import networkx as nx
    G = make_british_islands_graph()
    names, circles, name_to_idx = get_leaf_circles(G)
    for name in names:
        assert G.in_degree(name) == 0


def test_get_leaf_circles_radius_from_area():
    G = make_british_islands_graph()
    names, circles, name_to_idx = get_leaf_circles(G)
    for name, (cx, cy, r) in zip(names, circles):
        area = G.nodes[name]["target_area"]
        assert r == pytest.approx(np.sqrt(area / np.pi), rel=1e-5)


def test_get_leaf_circles_index_map():
    G = make_british_islands_graph()
    names, circles, name_to_idx = get_leaf_circles(G)
    assert len(name_to_idx) == len(names)
    for i, name in enumerate(names):
        assert name_to_idx[name] == i


# ---------------------------------------------------------------------------
# get_enclosing_sets
# ---------------------------------------------------------------------------


def test_get_enclosing_sets_all_have_children():
    import networkx as nx
    G = make_british_islands_graph()
    leaf_names, _, _ = get_leaf_circles(G)
    set_names, sets_idx = get_enclosing_sets(G, leaf_names)
    for name in set_names:
        assert G.in_degree(name) > 0


def test_get_enclosing_sets_indices_in_range():
    G = make_british_islands_graph()
    leaf_names, _, _ = get_leaf_circles(G)
    set_names, sets_idx = get_enclosing_sets(G, leaf_names)
    for idx_list in sets_idx:
        for i in idx_list:
            assert 0 <= i < len(leaf_names)


def test_get_enclosing_sets_great_britain_contains_england():
    G = make_british_islands_graph()
    leaf_names, _, name_to_idx = get_leaf_circles(G)
    set_names, sets_idx = get_enclosing_sets(G, leaf_names)
    gb_pos = set_names.index("Great Britain")
    england_idx = name_to_idx["England"]
    assert england_idx in sets_idx[gb_pos]


# ---------------------------------------------------------------------------
# graph_to_optimizer_inputs
# ---------------------------------------------------------------------------


def test_graph_to_optimizer_inputs_lengths():
    G = make_british_islands_graph()
    set_names, idx, enclosures, target_areas, initial_centers = (
        graph_to_optimizer_inputs(G)
    )
    n = len(G.nodes)
    assert len(set_names) == n
    assert len(idx) == n
    assert len(target_areas) == n
    assert initial_centers.shape == (n, 2)


def test_graph_to_optimizer_inputs_enclosures_valid():
    G = make_british_islands_graph()
    set_names, idx, enclosures, target_areas, initial_centers = (
        graph_to_optimizer_inputs(G)
    )
    n = len(G.nodes)
    for u, v in enclosures:
        assert 0 <= u < n
        assert 0 <= v < n
        assert u != v


def test_graph_to_optimizer_inputs_centers_dtype():
    G = make_british_islands_graph()
    _, _, _, _, initial_centers = graph_to_optimizer_inputs(G)
    assert initial_centers.dtype == np.float32
