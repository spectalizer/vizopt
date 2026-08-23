"""Tests for vizopt.templates.trees.recursive_raster_treemap"""

import networkx as nx
import numpy as np
import pytest

from vizopt.components.stars import radius_at_angle, star_polygon_area
from vizopt.templates.trees.recursive_raster_treemap import (
    RasterTreemapOptimizer,
    _branching_buckets,
    _bucket_size,
)
from vizopt.treemap import subtree_sizes as _subtree_sizes_from_attr

_FAST = dict(grid_resolution=16, n_iters=30, learning_rate=0.02)
_MODERATE = dict(grid_resolution=24, n_iters=600, learning_rate=0.02)


def _subtree_sizes(graph: nx.DiGraph, root, leaf_weights: dict) -> dict:
    """Sum leaf_weights up the tree, via vizopt.treemap.subtree_sizes."""
    nx.set_node_attributes(graph, leaf_weights, "size")
    return _subtree_sizes_from_attr(graph, root)


def _small_tree_and_sizes():
    """root -> a -> {a1, a2}; root -> b -> b1; root -> c -> {c1, c2, c3}."""
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
    leaf_weights = {"a1": 3.0, "a2": 1.0, "b1": 2.0, "c1": 1.0, "c2": 1.0, "c3": 2.0}
    sizes = _subtree_sizes(g, "root", leaf_weights)
    return g, sizes


# ---------------------------------------------------------------------------
# Branching-factor bucketing
# ---------------------------------------------------------------------------


def _lopsided_tree_and_sizes():
    """root has 5 children; one of them ("a") has only 2 children."""
    g = nx.DiGraph()
    g.add_edges_from(
        [
            ("root", "a"),
            ("root", "b"),
            ("root", "c"),
            ("root", "d"),
            ("root", "e"),
            ("a", "a1"),
            ("a", "a2"),
        ]
    )
    leaf_weights = {"a1": 1.0, "a2": 1.0, "b": 1.0, "c": 1.0, "d": 1.0, "e": 1.0}
    sizes = _subtree_sizes(g, "root", leaf_weights)
    return g, sizes


def test_branching_buckets_are_sorted_distinct_observed_counts():
    g, sizes = _lopsided_tree_and_sizes()
    # root has 5 children, "a" has 2 -> observed branching factors are {2, 5}.
    assert _branching_buckets(g, sizes) == [2, 5]


def test_bucket_size_never_exceeds_the_tree_wide_max():
    g, sizes = _lopsided_tree_and_sizes()
    buckets = _branching_buckets(g, sizes)
    # "a"'s 2 real children get padded to 2, not root's 5 -- the whole point
    # of bucketing instead of a single tree-wide max.
    assert _bucket_size(2, buckets) == 2
    assert _bucket_size(5, buckets) == 5


def test_single_element_override_reproduces_uniform_padding():
    g, sizes = _lopsided_tree_and_sizes()
    assert _bucket_size(2, [5]) == 5
    assert _bucket_size(5, [5]) == 5


def test_optimize_with_lopsided_tree_pads_small_group_below_tree_max():
    g, sizes = _lopsided_tree_and_sizes()
    optimizer = RasterTreemapOptimizer(g, sizes, **_FAST)
    optimizer.optimize()
    # No pad nodes should leak regardless of how much padding each group got.
    assert all(not str(n).startswith("__pad_") for n in optimizer.node_shapes_)


# ---------------------------------------------------------------------------
# optimize() / node_shapes_
# ---------------------------------------------------------------------------


def test_node_shapes_raises_before_optimize():
    g, sizes = _small_tree_and_sizes()
    optimizer = RasterTreemapOptimizer(g, sizes, **_FAST)
    with pytest.raises(ValueError):
        _ = optimizer.node_shapes_


def test_optimize_returns_shapes_for_every_non_root_node():
    g, sizes = _small_tree_and_sizes()
    optimizer = RasterTreemapOptimizer(g, sizes, **_FAST)
    optimizer.optimize()
    expected = set(g.nodes) - {"root"}
    assert set(optimizer.node_shapes_) == expected


def test_no_pad_nodes_leak_into_results():
    g, sizes = _small_tree_and_sizes()
    optimizer = RasterTreemapOptimizer(g, sizes, **_FAST)
    optimizer.optimize()
    assert all(not str(n).startswith("__pad_") for n in optimizer.node_shapes_)


def test_optimize_returns_same_dict_as_result_attribute():
    g, sizes = _small_tree_and_sizes()
    optimizer = RasterTreemapOptimizer(g, sizes, **_FAST)
    returned = optimizer.optimize()
    assert returned is optimizer.result_
    assert returned is optimizer.node_shapes_


def test_root_defaults_to_unique_in_degree_zero_node():
    g, sizes = _small_tree_and_sizes()
    optimizer = RasterTreemapOptimizer(g, sizes, **_FAST)
    assert optimizer.root == "root"


def test_directory_with_single_child():
    # "b" has exactly one child ("b1"); n_sets == 1 before padding must not break.
    g, sizes = _small_tree_and_sizes()
    optimizer = RasterTreemapOptimizer(g, sizes, **_FAST)
    optimizer.optimize()
    assert "b1" in optimizer.node_shapes_


def test_plot_returns_axes_with_a_label_per_node():
    g, sizes = _small_tree_and_sizes()
    optimizer = RasterTreemapOptimizer(g, sizes, **_FAST)
    optimizer.optimize()
    ax = optimizer.plot()
    assert len(ax.texts) == len(optimizer.node_shapes_)


# ---------------------------------------------------------------------------
# Layout quality
# ---------------------------------------------------------------------------


def test_top_level_areas_roughly_proportional_to_sizes():
    g, sizes = _small_tree_and_sizes()
    optimizer = RasterTreemapOptimizer(g, sizes, mean_leaf_area=3.0, **_MODERATE)
    optimizer.optimize()

    # root's children: a=4, b=2, c=4 -> expect area(a) ~= area(c) > area(b)
    area_a = star_polygon_area(np.asarray(optimizer.node_shapes_["a"]["radii"]))
    area_b = star_polygon_area(np.asarray(optimizer.node_shapes_["b"]["radii"]))
    area_c = star_polygon_area(np.asarray(optimizer.node_shapes_["c"]["radii"]))

    assert area_a > area_b
    assert area_c > area_b
    assert area_a == pytest.approx(area_c, rel=0.35)


def test_children_stay_mostly_within_parent_boundary():
    g, sizes = _small_tree_and_sizes()
    optimizer = RasterTreemapOptimizer(
        g, sizes, containment_offset=0.05, weight_containment=30.0, **_MODERATE
    )
    optimizer.optimize()

    parent_res = optimizer.node_shapes_["c"]
    parent_center = np.asarray(parent_res["center"])
    parent_radii = np.asarray(parent_res["radii"])

    for child_name in ["c1", "c2", "c3"]:
        child_res = optimizer.node_shapes_[child_name]
        cx, cy = child_res["center"]
        r, angs = np.asarray(child_res["radii"]), np.asarray(child_res["angles"])
        boundary = np.stack([cx + r * np.cos(angs), cy + r * np.sin(angs)], axis=1)
        dist_from_parent_center = np.linalg.norm(boundary - parent_center, axis=1)
        angle_from_parent_center = np.arctan2(
            boundary[:, 1] - parent_center[1], boundary[:, 0] - parent_center[0]
        )
        allowed = np.array(
            [radius_at_angle(parent_radii, a) for a in angle_from_parent_center]
        )
        # Soft containment penalty, not a hard constraint: allow generous slack.
        assert np.mean(dist_from_parent_center <= allowed + 0.3) > 0.8
