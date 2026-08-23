"""Tests for vizopt.treemap"""

import networkx as nx

from vizopt.treemap import subtree_sizes


def test_subtree_sizes_leaf_is_its_own_attribute():
    g = nx.DiGraph()
    g.add_node("leaf", size=5.0)
    assert subtree_sizes(g, "leaf") == {"leaf": 5.0}


def test_subtree_sizes_sums_descendant_leaves():
    g = nx.DiGraph()
    g.add_edges_from([("root", "a"), ("root", "b"), ("a", "a1"), ("a", "a2")])
    nx.set_node_attributes(g, {"a1": 3.0, "a2": 1.0, "b": 2.0}, "size")
    sizes = subtree_sizes(g, "root")
    assert sizes["a1"] == 3.0
    assert sizes["a"] == 4.0
    assert sizes["b"] == 2.0
    assert sizes["root"] == 6.0


def test_subtree_sizes_custom_weight_key():
    g = nx.DiGraph()
    g.add_edges_from([("root", "leaf")])
    nx.set_node_attributes(g, {"leaf": 7.0}, "population")
    assert subtree_sizes(g, "root", weight_key="population") == {
        "root": 7.0,
        "leaf": 7.0,
    }


def test_subtree_sizes_only_includes_reachable_nodes():
    g = nx.DiGraph()
    g.add_edges_from([("root", "a")])
    g.add_node("unrelated", size=99.0)
    nx.set_node_attributes(g, {"a": 1.0}, "size")
    sizes = subtree_sizes(g, "root")
    assert "unrelated" not in sizes
