"""Tests for vizopt.introspection"""

from pathlib import Path

import pytest

from vizopt.introspection import build_file_tree, compute_subtree_sizes, treemap_layout


def _make_tree(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("world!")
    ignored = tmp_path / "__pycache__"
    ignored.mkdir()
    (ignored / "c.pyc").write_text("noise")
    return tmp_path


def test_build_file_tree_structure(tmp_path):
    root = _make_tree(tmp_path)
    graph = build_file_tree(root)

    assert graph.nodes[Path(".")]["is_dir"] is True
    assert set(graph.successors(Path("."))) == {Path("a.txt"), Path("sub")}
    assert list(graph.successors(Path("sub"))) == [Path("sub/b.txt")]


def test_build_file_tree_node_attributes(tmp_path):
    root = _make_tree(tmp_path)
    graph = build_file_tree(root)

    assert graph.nodes[Path("a.txt")] == {"is_dir": False, "size": 5}
    assert graph.nodes[Path("sub")] == {"is_dir": True, "size": 0}


def test_build_file_tree_ignores_default_noise_dirs(tmp_path):
    root = _make_tree(tmp_path)
    graph = build_file_tree(root)

    assert Path("__pycache__") not in graph.nodes


def test_build_file_tree_custom_ignore(tmp_path):
    root = _make_tree(tmp_path)
    graph = build_file_tree(root, ignore={"sub"})

    assert Path("sub") not in graph.nodes
    assert Path("__pycache__") in graph.nodes


def test_build_file_tree_rejects_non_directory(tmp_path):
    file_path = tmp_path / "a.txt"
    file_path.write_text("hello")

    with pytest.raises(NotADirectoryError):
        build_file_tree(file_path)


def test_compute_subtree_sizes(tmp_path):
    root = _make_tree(tmp_path)
    graph = build_file_tree(root)

    sizes = compute_subtree_sizes(graph)

    assert sizes[Path("a.txt")] == 5
    assert sizes[Path("sub/b.txt")] == 6
    assert sizes[Path("sub")] == 6
    assert sizes[Path(".")] == 11


def test_treemap_layout_covers_bounding_rect(tmp_path):
    root = _make_tree(tmp_path)
    graph = build_file_tree(root)

    rects = treemap_layout(graph)

    assert rects[Path(".")] == (0.0, 0.0, 1.0, 1.0)
    assert set(rects.keys()) == {
        Path("."),
        Path("a.txt"),
        Path("sub"),
        Path("sub/b.txt"),
    }
    # sub/b.txt's rect must be nested within sub's rect.
    sx0, sy0, sx1, sy1 = rects[Path("sub")]
    bx0, by0, bx1, by1 = rects[Path("sub/b.txt")]
    assert sx0 <= bx0 and bx1 <= sx1
    assert sy0 <= by0 and by1 <= sy1


def test_treemap_layout_empty_tree_yields_no_rects(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    graph = build_file_tree(empty_dir)

    assert treemap_layout(graph) == {}


def test_treemap_layout_skips_zero_size_subdirectories(tmp_path):
    root = _make_tree(tmp_path)
    (root / "sub" / "empty_subdir").mkdir()
    graph = build_file_tree(root)

    rects = treemap_layout(graph)

    assert Path("sub/empty_subdir") not in rects
    assert set(rects.keys()) == {
        Path("."),
        Path("a.txt"),
        Path("sub"),
        Path("sub/b.txt"),
    }
