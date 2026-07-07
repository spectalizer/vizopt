"""Tests for vizopt.introspection"""

import ast
from pathlib import Path

import pytest

from vizopt.introspection import (
    build_class_hierarchy,
    build_file_tree,
    compute_subtree_sizes,
    parse_module_ast,
    treemap_layout,
)


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


def test_treemap_layout_padding_insets_children(tmp_path):
    root = _make_tree(tmp_path)
    graph = build_file_tree(root)

    rects = treemap_layout(graph, padding=0.05)

    # Root itself keeps the full rect; only its children are inset.
    assert rects[Path(".")] == (0.0, 0.0, 1.0, 1.0)
    for node in (Path("a.txt"), Path("sub")):
        x0, y0, x1, y1 = rects[node]
        assert x0 >= 0.05 - 1e-9
        assert y0 >= 0.05 - 1e-9
        assert x1 <= 0.95 + 1e-9
        assert y1 <= 0.95 + 1e-9
    # sub/b.txt must be strictly inside sub's (also inset) rect.
    sx0, sy0, sx1, sy1 = rects[Path("sub")]
    bx0, by0, bx1, by1 = rects[Path("sub/b.txt")]
    assert sx0 < bx0 and bx1 < sx1
    assert sy0 < by0 and by1 < sy1


def test_treemap_layout_padding_clamped_for_tiny_rects(tmp_path):
    root = _make_tree(tmp_path)
    graph = build_file_tree(root)

    # padding far larger than the rect must not invert x0/x1 or y0/y1.
    rects = treemap_layout(graph, padding=1000.0)

    for x0, y0, x1, y1 in rects.values():
        assert x0 <= x1
        assert y0 <= y1


def test_parse_module_ast_returns_module(tmp_path):
    module_path = tmp_path / "example.py"
    module_path.write_text("class Base:\n    pass\n\n\nclass Child(Base):\n    pass\n")

    tree = parse_module_ast(module_path)

    assert isinstance(tree, ast.Module)
    class_names = [
        n.name for n in ast.iter_child_nodes(tree) if isinstance(n, ast.ClassDef)
    ]
    assert class_names == ["Base", "Child"]


def test_parse_module_ast_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_module_ast(tmp_path / "does_not_exist.py")


def test_parse_module_ast_invalid_syntax(tmp_path):
    module_path = tmp_path / "broken.py"
    module_path.write_text("def broken(:\n")

    with pytest.raises(SyntaxError):
        parse_module_ast(module_path)


def test_parse_module_ast_reads_non_ascii_as_utf8(tmp_path):
    module_path = tmp_path / "unicode_docstring.py"
    module_path.write_bytes(
        'def f():\n    """Uses subscripts a₀, a₁."""\n'.encode("utf-8")
    )

    tree = parse_module_ast(module_path)

    docstring = ast.get_docstring(tree.body[0])
    assert docstring == "Uses subscripts a₀, a₁."


def test_build_class_hierarchy_simple_inheritance(tmp_path):
    module_path = tmp_path / "example.py"
    module_path.write_text("class Base:\n    pass\n\n\nclass Child(Base):\n    pass\n")

    graph = build_class_hierarchy(parse_module_ast(module_path))

    assert set(graph.nodes) == {"Base", "Child"}
    assert list(graph.successors("Base")) == ["Child"]


def test_build_class_hierarchy_multiple_inheritance(tmp_path):
    module_path = tmp_path / "example.py"
    module_path.write_text(
        "class A:\n"
        "    pass\n\n\n"
        "class B:\n"
        "    pass\n\n\n"
        "class C(A, B):\n"
        "    pass\n"
    )

    graph = build_class_hierarchy(parse_module_ast(module_path))

    assert set(graph.predecessors("C")) == {"A", "B"}


def test_build_class_hierarchy_external_base_added_as_node(tmp_path):
    module_path = tmp_path / "example.py"
    module_path.write_text(
        "import base\n\n\nclass Child(base.VizOptimizer):\n    pass\n"
    )

    graph = build_class_hierarchy(parse_module_ast(module_path))

    assert set(graph.nodes) == {"base.VizOptimizer", "Child"}
    assert list(graph.successors("base.VizOptimizer")) == ["Child"]


def test_build_class_hierarchy_no_inheritance_isolated_node(tmp_path):
    module_path = tmp_path / "example.py"
    module_path.write_text("class Standalone:\n    pass\n")

    graph = build_class_hierarchy(parse_module_ast(module_path))

    assert list(graph.nodes) == ["Standalone"]
    assert list(graph.edges) == []


def test_build_class_hierarchy_subscripted_base_falls_back_to_unparse(tmp_path):
    module_path = tmp_path / "example.py"
    module_path.write_text("class Child(Generic[int]):\n    pass\n")

    graph = build_class_hierarchy(parse_module_ast(module_path))

    assert set(graph.nodes) == {"Generic[int]", "Child"}
    assert list(graph.successors("Generic[int]")) == ["Child"]
