"""Introspection utilities for visualizing the structure of this project."""

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

    def _layout(node: Path, node_rect: Rect) -> None:
        items = [(c, sizes[c]) for c in graph.successors(node) if sizes[c] > 0]
        for child, child_rect in squarify_layout(items, node_rect).items():
            out[child] = child_rect
            if graph.nodes[child]["is_dir"]:
                _layout(child, child_rect)

    _layout(root, rect)
    return out


def plot_treemap(
    graph: nx.DiGraph,
    root: Path = Path("."),
    *,
    ax: Axes | None = None,
) -> Axes:
    """Plot a squarified treemap of a file tree.

    Directories are drawn as unfilled outlines (so nested children remain
    visible) and files as filled, labeled rectangles.

    Args:
        graph: Directed graph as returned by build_file_tree.
        root: Node to plot the subtree of.
        ax: Axes to draw on. A new figure and axes are created if None.

    Returns:
        The axes the treemap was drawn on.
    """
    rects = treemap_layout(graph, root)
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
        if not is_dir:
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
