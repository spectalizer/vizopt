"""Tests for vizopt.introspection"""

from pathlib import Path

import pytest

from vizopt.introspection import build_file_tree


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
