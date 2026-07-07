"""Introspection utilities for visualizing the structure of this project."""

import ast
from pathlib import Path

import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from .treemap import squarify_layout

_DEFAULT_IGNORE = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".venv"}

Rect = tuple[float, float, float, float]


def build_file_tree(
    root: str | Path,
    *,
    ignore: set[str] | None = None,
) -> nx.DiGraph:
    """Build a directory/file containment tree as a NetworkX digraph.

    Walks root recursively and adds one node per file or directory
    encountered, with edges directed from each directory to its direct
    children (parent -> child), following the project's convention for
    containment hierarchies.

    Args:
        root: Directory to walk. Accepts a string path or a pathlib.Path;
            relative paths are resolved against the current working
            directory.
        ignore: File and directory names to skip entirely, matched against
            Path.name. Defaults to a small set of common noise directories
            (.git, __pycache__, .mypy_cache, .pytest_cache, .venv).

    Returns:
        Directed graph whose nodes are pathlib.Path objects relative to
        root (the root itself is Path(".")). Each node carries an is_dir
        (bool) and a size (int, bytes; 0 for directories) attribute.

    Raises:
        NotADirectoryError: If root is not an existing directory.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    if ignore is None:
        ignore = _DEFAULT_IGNORE

    graph: nx.DiGraph = nx.DiGraph()
    root_rel = Path(".")
    graph.add_node(root_rel, is_dir=True, size=0)

    def _walk(directory: Path, rel_directory: Path) -> None:
        for child in sorted(directory.iterdir()):
            if child.name in ignore:
                continue
            rel_child = rel_directory / child.name
            is_dir = child.is_dir()
            size = 0 if is_dir else child.stat().st_size
            graph.add_node(rel_child, is_dir=is_dir, size=size)
            graph.add_edge(rel_directory, rel_child)
            if is_dir:
                _walk(child, rel_child)

    _walk(root, root_rel)
    return graph


def compute_subtree_sizes(graph: nx.DiGraph, root: Path = Path(".")) -> dict[Path, int]:
    """Compute the total file size of every node's subtree.

    A file's size is its own size attribute; a directory's size is the
    sum of the sizes of all files reachable from it.

    Args:
        graph: Directed graph as returned by build_file_tree.
        root: Node to start from. Only nodes reachable from root are
            included in the result.

    Returns:
        Dict mapping every node in the subtree rooted at root to its
        total size in bytes.
    """
    sizes: dict[Path, int] = {}
    for node in nx.dfs_postorder_nodes(graph, source=root):
        if graph.nodes[node]["is_dir"]:
            sizes[node] = sum(sizes[child] for child in graph.successors(node))
        else:
            sizes[node] = graph.nodes[node]["size"]
    return sizes


def treemap_layout(
    graph: nx.DiGraph,
    root: Path = Path("."),
    rect: Rect = (0.0, 0.0, 1.0, 1.0),
    *,
    padding: float = 0.0,
) -> dict[Path, Rect]:
    """Recursive squarified treemap layout of a file tree.

    Each directory's children are laid out via squarify_layout with
    weights proportional to their subtree size, then every child
    directory is recursively laid out within its assigned rectangle.
    Empty subtrees (size 0) are omitted, since squarify_layout requires
    positive weights.

    Args:
        graph: Directed graph as returned by build_file_tree.
        root: Node to lay out the subtree of.
        rect: Bounding rectangle (x0, y0, x1, y1) assigned to root.
        padding: Inset applied to a directory's rectangle, in the same
            units as rect, before laying out its children. Leaves a
            margin between a directory's border and its children so
            nested rectangles stay visually distinguishable. Clamped so
            it never exceeds half of either side of the directory's
            rectangle.

    Returns:
        Dict mapping every node reachable from root (including root
        itself) to its (x0, y0, x1, y1) rectangle, except nodes in
        zero-size subtrees.
    """
    sizes = compute_subtree_sizes(graph, root)
    out: dict[Path, Rect] = {}
    if sizes[root] == 0:
        return out
    out[root] = rect

    def _inset(r: Rect) -> Rect:
        x0, y0, x1, y1 = r
        pad = min(padding, (x1 - x0) / 2, (y1 - y0) / 2)
        return (x0 + pad, y0 + pad, x1 - pad, y1 - pad)

    def _layout(node: Path, node_rect: Rect) -> None:
        items = [(c, sizes[c]) for c in graph.successors(node) if sizes[c] > 0]
        for child, child_rect in squarify_layout(items, node_rect).items():
            out[child] = child_rect
            if graph.nodes[child]["is_dir"]:
                _layout(child, _inset(child_rect))

    _layout(root, _inset(rect))
    return out


def plot_treemap(
    graph: nx.DiGraph,
    root: Path = Path("."),
    *,
    padding: float = 0.0,
    ax: Axes | None = None,
) -> Axes:
    """Plot a squarified treemap of a file tree.

    Directories are drawn as unfilled, labeled outlines (so nested
    children remain visible) and files as filled, labeled rectangles.

    Args:
        graph: Directed graph as returned by build_file_tree.
        root: Node to plot the subtree of.
        padding: Inset left between a directory's border and its
            children; see treemap_layout. Also gives directory name
            labels room to sit clear of their children.
        ax: Axes to draw on. A new figure and axes are created if None.

    Returns:
        The axes the treemap was drawn on.
    """
    rects = treemap_layout(graph, root, padding=padding)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))
    assert ax is not None

    for node, (x0, y0, x1, y1) in rects.items():
        is_dir = graph.nodes[node]["is_dir"]
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor="none" if is_dir else "#4472c4",
                edgecolor="black",
                linewidth=1.5 if is_dir else 0.5,
            )
        )
        if is_dir:
            if node != root:
                ax.annotate(
                    node.name,
                    xy=(x0, y1),
                    xytext=(3, -3),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    fontsize=8,
                    fontweight="bold",
                    clip_on=True,
                )
        else:
            ax.text(
                (x0 + x1) / 2,
                (y0 + y1) / 2,
                node.name,
                ha="center",
                va="center",
                fontsize=7,
                clip_on=True,
            )

    x0, y0, x1, y1 = rects[root]
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return ax


def parse_module_ast(path: str | Path) -> ast.Module:
    """Parse a Python module file into its abstract syntax tree.

    Args:
        path: Path to a .py file. Accepts a string path or a
            pathlib.Path.

    Returns:
        The parsed module. Its filename is attached via ast.parse's
        filename argument, so downstream tools (e.g. ast.walk error
        messages) can report locations against the original file.

    Raises:
        FileNotFoundError: If path does not point to an existing file.
        SyntaxError: If the file's contents are not valid Python.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is not a file")
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _base_name(node: ast.expr) -> str:
    """Best-effort dotted name for a base class expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_base_name(node.value)}.{node.attr}"
    return ast.unparse(node)


def build_class_hierarchy(tree: ast.Module) -> nx.DiGraph:
    """Build a class-inheritance graph from a module's AST.

    Walks every ast.ClassDef in tree, at any nesting level, and adds a
    parent -> child edge from each base class to the class that inherits
    from it, following the project's parent -> child edge convention.
    Base classes are added as nodes even when they are not themselves
    defined in tree (e.g. imported classes), so cross-module inheritance
    edges remain visible.

    Classes are identified by their simple name (dotted, for bases
    accessed through attribute access, e.g. module.Class); same-named
    classes defined in different scopes are not distinguished and will
    collide as a single node.

    Args:
        tree: Module AST, as returned by parse_module_ast.

    Returns:
        Directed graph whose edges point from base class to subclass.
    """
    graph: nx.DiGraph = nx.DiGraph()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        graph.add_node(node.name)
        for base in node.bases:
            graph.add_edge(_base_name(base), node.name)
    return graph
