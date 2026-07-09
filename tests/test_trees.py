"""Tests for vizopt.templates.trees.tree_layout"""

import networkx as nx
import numpy as np
import pytest

from vizopt.base import OptimConfig
from vizopt.templates.trees.tree_layout import (
    TreeLayoutOptimizer,
    make_tree_input_params,
)

_FAST = OptimConfig(n_iters=5, learning_rate=1e-2)


def _NO_PRINT(*_):
    pass


def _small_tree() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from(
        [
            ("root", "a"),
            ("root", "b"),
            ("root", "c"),
            ("a", "a1"),
            ("a", "a2"),
            ("b", "b1"),
            ("c", "c1"),
            ("c", "c2"),
            ("c", "c3"),
        ]
    )
    return g


def _single_node_tree() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node("root")
    return g


# ---------------------------------------------------------------------------
# make_tree_input_params — validation
# ---------------------------------------------------------------------------


def test_rejects_multiple_roots():
    g = nx.DiGraph()
    g.add_edges_from([("root1", "a"), ("root2", "a")])
    with pytest.raises(ValueError):
        make_tree_input_params(g)


def test_rejects_cycle():
    g = nx.DiGraph()
    g.add_edges_from([("a", "b"), ("b", "a")])
    with pytest.raises(ValueError):
        make_tree_input_params(g)


def test_rejects_disconnected_graph():
    g = nx.DiGraph()
    g.add_edges_from([("root", "a")])
    g.add_node("isolated")
    with pytest.raises(ValueError):
        make_tree_input_params(g)


# ---------------------------------------------------------------------------
# make_tree_input_params — structure
# ---------------------------------------------------------------------------


def test_depths_match_tree_structure():
    params = make_tree_input_params(_small_tree())
    depths = dict(zip(params["node_names"], params["depths"]))
    assert depths["root"] == 0
    assert depths["a"] == 1
    assert depths["a1"] == 2
    assert depths["c3"] == 2


def test_root_detected():
    params = make_tree_input_params(_small_tree())
    assert params["root"] == "root"


def test_order_pairs_only_within_same_depth():
    params = make_tree_input_params(_small_tree())
    depths = params["depths"]
    for i, j in params["order_pairs"]:
        assert depths[i] == depths[j]


def test_order_pairs_preserve_child_order():
    g = nx.DiGraph()
    g.add_edges_from([("root", "x"), ("root", "y"), ("root", "z")])
    params = make_tree_input_params(g)
    name_to_id = {n: i for i, n in enumerate(params["node_names"])}
    pairs = {tuple(p) for p in params["order_pairs"].tolist()}
    assert (name_to_id["x"], name_to_id["y"]) in pairs
    assert (name_to_id["y"], name_to_id["z"]) in pairs


def test_order_pairs_respect_order_attribute():
    g = nx.DiGraph()
    g.add_node("root")
    g.add_node("x", order=1)
    g.add_node("y", order=0)
    g.add_edges_from([("root", "x"), ("root", "y")])
    params = make_tree_input_params(g)
    name_to_id = {n: i for i, n in enumerate(params["node_names"])}
    pairs = {tuple(p) for p in params["order_pairs"].tolist()}
    assert (name_to_id["y"], name_to_id["x"]) in pairs


def test_single_node_tree_has_no_edges_or_pairs():
    params = make_tree_input_params(_single_node_tree())
    assert params["edge_indices"].shape == (0, 2)
    assert params["order_pairs"].shape == (0, 2)
    assert params["initial_node_xys"].shape == (1, 2)


def test_preferred_edge_length_defaults_to_layer_spacing():
    params = make_tree_input_params(_small_tree(), layer_spacing=2.5)
    assert params["preferred_edge_length"] == pytest.approx(2.5)


def test_growth_direction_is_unit_vector():
    params = make_tree_input_params(_small_tree(), growth_direction=(3.0, 4.0))
    assert np.linalg.norm(params["growth_direction"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TreeLayoutOptimizer — optimize() smoke tests
# ---------------------------------------------------------------------------


def test_optimize_returns_node_positions_for_every_node():
    optimizer = TreeLayoutOptimizer(_small_tree())
    optimizer.optimize(optim_config=_FAST, callback=_NO_PRINT)
    assert set(optimizer.node_positions_) == set(_small_tree().nodes)


def test_node_positions_raises_before_optimize():
    optimizer = TreeLayoutOptimizer(_small_tree())
    with pytest.raises(ValueError):
        _ = optimizer.node_positions_


def test_optimize_history_has_term_keys():
    optimizer = TreeLayoutOptimizer(_small_tree())
    result = optimizer.optimize(optim_config=_FAST, callback=_NO_PRINT)
    assert len(result.history) > 0
    for record in result.history:
        assert "depth_alignment" in record
        assert "parent_centering" in record
        assert "order_separation" in record
        assert "edge_length" in record


def test_optimize_root_stays_at_origin_growth_coordinate():
    g = nx.DiGraph()
    g.add_edges_from([("root", "a"), ("root", "b")])
    optimizer = TreeLayoutOptimizer(g, growth_direction=(0.0, -1.0))
    optimizer.optimize(
        optim_config=OptimConfig(n_iters=500, learning_rate=1e-2), callback=_NO_PRINT
    )
    root_y = optimizer.node_positions_["root"][1]
    child_y = optimizer.node_positions_["a"][1]
    assert root_y > child_y  # growth direction (0, -1): depth increases downward


def test_optimize_preserves_sibling_order():
    g = nx.DiGraph()
    g.add_edges_from([("root", "x"), ("root", "y"), ("root", "z")])
    optimizer = TreeLayoutOptimizer(g, growth_direction=(0.0, -1.0), min_distance=1.0)
    optimizer.optimize(
        optim_config=OptimConfig(n_iters=500, learning_rate=1e-2), callback=_NO_PRINT
    )
    x_pos = optimizer.node_positions_["x"][0]
    y_pos = optimizer.node_positions_["y"][0]
    z_pos = optimizer.node_positions_["z"][0]
    assert x_pos < y_pos < z_pos
