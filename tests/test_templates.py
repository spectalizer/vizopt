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
from vizopt.templates.euler.stars_vs_circles import EulerDiagram

_FAST = OptimConfig(n_iters=5, learning_rate=1e-2)


def _NO_PRINT(*_):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_circle_problem(representation=None) -> EulerDiagram:
    """Minimal two-circle, two-set problem for smoke-testing optimizers."""
    circles = np.array([[0.0, 0.0, 0.5], [2.0, 0.0, 0.5]], dtype=np.float32)
    sets = [[0], [1]]
    diagram = EulerDiagram(circles, sets, representation=representation)
    diagram.optimize(optim_config=_FAST, callback=_NO_PRINT)
    return diagram


# ---------------------------------------------------------------------------
# optimize_multiple_radially_convex_sets_with_movable_circles — result structure
# ---------------------------------------------------------------------------


def test_optimize_returns_euler_diagram():
    diagram = _two_circle_problem()
    assert isinstance(diagram.sets_, list)
    assert isinstance(diagram.result_.history, list)


def test_optimize_result_length():
    assert len(_two_circle_problem().sets_) == 2


def test_optimize_result_keys():
    for r in _two_circle_problem().sets_:
        assert "center" in r
        assert "radii" in r
        assert "angles" in r


def test_optimize_result_shapes():
    for r in _two_circle_problem(Discrete(k_angles=16)).sets_:
        assert r["center"].shape == (2,)
        assert r["radii"].shape == (16,)
        assert r["angles"].shape == (16,)


def test_optimize_radii_positive():
    for r in _two_circle_problem().sets_:
        assert np.all(r["radii"] > 0)


def test_optimize_history_has_term_keys():
    history = _two_circle_problem().result_.history
    assert len(history) > 0
    for record in history:
        assert "iteration" in record
        assert "total" in record


# ---------------------------------------------------------------------------
# Representation variants
# ---------------------------------------------------------------------------


def test_optimize_fourier_representation():
    for r in _two_circle_problem(Fourier(k_angles=16, n_harmonics=4)).sets_:
        assert "fourier_coeffs" in r
        assert r["fourier_coeffs"].shape == (2 * 4 + 1,)
        assert r["radii"].shape == (16,)


def test_optimize_bspline_representation():
    for r in _two_circle_problem(BSpline(k_angles=16, n_ctrl_pts=8)).sets_:
        assert "bspline_ctrl" in r
        assert r["bspline_ctrl"].shape == (8,)
        assert r["radii"].shape == (16,)


def test_optimize_discrete_no_extra_results():
    for r in _two_circle_problem(Discrete(k_angles=16)).sets_:
        assert "fourier_coeffs" not in r
        assert "bspline_ctrl" not in r


def test_convexity_alpha_reduces_concavity():
    """convexity_alpha=1 should produce a more convex boundary than alpha=0."""
    circles = np.array(
        [[-2.0, 0.0, 0.4], [0.0, 2.0, 0.4], [2.0, 0.0, 0.4]], dtype=np.float32
    )
    sets = [[0, 1, 2]]

    def _run(alpha):
        import jax.numpy as jnp

        diagram = EulerDiagram(
            circles,
            sets,
            weight_convexity=5.0,
            convexity_alpha=alpha,
            weight_area=0.5,
            weight_perimeter=0.5,
            weight_smoothness=0.1,
            representation=Discrete(k_angles=32),
        )
        diagram.optimize(
            OptimConfig(n_iters=200, learning_rate=5e-3), callback=_NO_PRINT
        )
        radii = jnp.array(diagram.sets_[0]["radii"])
        angles = jnp.array(diagram.sets_[0]["angles"])
        center = jnp.array(diagram.sets_[0]["center"])
        directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
        points = center[None, :] + radii[:, None] * directions
        edges = jnp.roll(points, -1, axis=0) - points
        edges_next = jnp.roll(edges, -1, axis=0)
        cross = edges[:, 0] * edges_next[:, 1] - edges[:, 1] * edges_next[:, 0]
        return float(jnp.sum(jnp.maximum(0.0, -cross)))

    concavity_no_alpha = _run(0.0)
    concavity_with_alpha = _run(1.0)
    assert concavity_with_alpha < concavity_no_alpha


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
    G = make_british_islands_graph()
    for n in G.nodes:
        if G.out_degree(n) == 0:
            assert G.nodes[n]["target_area"] is not None
            assert G.nodes[n]["target_area"] > 0


def test_graph_enclosing_nodes_have_none_target_area():
    G = make_british_islands_graph()
    for n in G.nodes:
        if G.out_degree(n) > 0:
            assert G.nodes[n]["target_area"] is None


def test_graph_england_in_great_britain():
    G = make_british_islands_graph()
    assert G.has_edge("Great Britain", "England")


def test_graph_without_ireland_island():
    G = make_british_islands_graph(include_ireland_island=False)
    assert "Ireland island" not in G.nodes
    assert G.has_edge("British Islands", "Republic of Ireland")


def test_graph_with_ireland_island():
    G = make_british_islands_graph(include_ireland_island=True)
    assert "Ireland island" in G.nodes
    assert G.has_edge("Ireland island", "Republic of Ireland")
    assert G.has_edge("Ireland island", "Northern Ireland")


# ---------------------------------------------------------------------------
# get_leaf_circles
# ---------------------------------------------------------------------------


def test_get_leaf_circles_all_out_degree_zero():
    G = make_british_islands_graph()
    names, circles, name_to_idx = get_leaf_circles(G)
    for name in names:
        assert G.out_degree(name) == 0


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
    G = make_british_islands_graph()
    leaf_names, _, _ = get_leaf_circles(G)
    set_names, sets_idx = get_enclosing_sets(G, leaf_names)
    for name in set_names:
        assert G.out_degree(name) > 0


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
