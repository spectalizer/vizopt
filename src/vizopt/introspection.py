"""Introspection utilities for visualizing the structure of this project."""

import ast
from pathlib import Path

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from .templates.euler.stars_vs_rectangles import EulerDiagramRect
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


_DEFAULT_ARROW_PROPS = {
    "arrowstyle": "->",
    "color": "#8f52e0",
    "lw": 1.0,
    "alpha": 0.6,
    "shrinkA": 5,
    "shrinkB": 5,
    "connectionstyle": "arc3,rad=0.15",
}


def plot_treemap_with_imports(
    file_graph: nx.DiGraph,
    import_graph: nx.DiGraph,
    root: Path = Path("."),
    *,
    padding: float = 0.0,
    ax: Axes | None = None,
    arrow_props: dict | None = None,
) -> Axes:
    """Plot a file-tree treemap with import edges overlaid.

    Draws the treemap via plot_treemap, then, for every edge in
    import_graph whose endpoints both have a rectangle in the treemap,
    draws a curved arrow between the centers of those rectangles
    (importer -> imported). Edges with an endpoint outside the plotted
    subtree (e.g. filtered out by root, or resolving to a node absent
    from file_graph) are silently skipped, as are self-loops.

    Args:
        file_graph: Directed graph as returned by build_file_tree.
        import_graph: Directed graph as returned by build_import_graph,
            built from the same root directory as file_graph so node
            paths line up with the treemap rectangles.
        root: Node to plot the subtree of.
        padding: Forwarded to plot_treemap and treemap_layout.
        ax: Axes to draw on. A new figure and axes are created if None.
        arrow_props: Keyword arguments forwarded to matplotlib's
            Axes.annotate as arrowprops, merged over the default style
            (a semi-transparent red curved arrow). Keys given here
            override the corresponding default.

    Returns:
        The axes the treemap and import edges were drawn on.
    """
    ax = plot_treemap(file_graph, root, padding=padding, ax=ax)
    rects = treemap_layout(file_graph, root, padding=padding)
    centers = {
        node: ((x0 + x1) / 2, (y0 + y1) / 2) for node, (x0, y0, x1, y1) in rects.items()
    }
    resolved_arrow_props = {**_DEFAULT_ARROW_PROPS, **(arrow_props or {})}

    for source, target in import_graph.edges:
        if source == target or source not in centers or target not in centers:
            continue
        ax.annotate(
            "",
            xy=centers[target],
            xytext=centers[source],
            arrowprops=resolved_arrow_props,
        )
    return ax


def build_euler_diagram(
    file_graph: nx.DiGraph,
    root: Path = Path("."),
    *,
    padding: float = 0.0,
    **kwargs,
) -> EulerDiagramRect:
    """Build a star-shaped Euler diagram optimizer for a file tree.

    Seeds each file's rectangle from a squarified treemap layout of
    file_graph (see treemap_layout) and wraps every directory in a convex
    star-shaped boundary via EulerDiagramRect.from_graph, so nested
    directories become nested/overlapping blobs rather than nested
    rectangles. Files whose subtree has zero size (e.g. empty files) have
    no treemap rectangle and are silently omitted, mirroring
    treemap_layout's own convention.

    Args:
        file_graph: Directed graph as returned by build_file_tree.
        root: Node to build the subtree of.
        padding: Forwarded to treemap_layout; only affects the initial
            (pre-optimization) seed positions and sizes.
        **kwargs: Forwarded to EulerDiagramRect.from_graph (e.g. weight_*
            terms, offsets, k_angles, term_schedules).

    Returns:
        An unfitted EulerDiagramRect. Call .optimize() before plotting it
        with plot_euler_diagram_with_imports.
    """
    rects = treemap_layout(file_graph, root, padding=padding)
    sub_nodes = set(rects)

    inclusion_graph: nx.DiGraph = nx.DiGraph()
    inclusion_graph.add_nodes_from(sub_nodes)
    inclusion_graph.add_edges_from(
        (u, v) for u, v in file_graph.edges if u in sub_nodes and v in sub_nodes
    )
    for node in sub_nodes:
        if file_graph.nodes[node]["is_dir"]:
            continue
        x0, y0, x1, y1 = rects[node]
        inclusion_graph.nodes[node]["center"] = [(x0 + x1) / 2, (y0 + y1) / 2]
        inclusion_graph.nodes[node]["hw"] = (x1 - x0) / 2
        inclusion_graph.nodes[node]["hh"] = (y1 - y0) / 2

    return EulerDiagramRect.from_graph(inclusion_graph, **kwargs)


def plot_euler_diagram_with_imports(
    optim: EulerDiagramRect,
    import_graph: nx.DiGraph,
    *,
    ax: Axes | None = None,
    arrow_props: dict | None = None,
) -> Axes:
    """Plot a fitted file-tree Euler diagram with import edges overlaid.

    Draws each directory's star-shaped boundary (optim.sets_) and each
    file's rectangle (optim.rects_), then, for every edge in import_graph
    whose endpoints both correspond to a leaf rectangle in optim, draws a
    curved arrow between rectangle centers (importer -> imported). Edges
    with an endpoint outside optim.leaf_names (e.g. a zero-size file
    dropped by build_euler_diagram, or a node outside the plotted
    subtree) are silently skipped, as are self-loops.

    Args:
        optim: An EulerDiagramRect built via build_euler_diagram, with
            .optimize() already called.
        import_graph: Directed graph as returned by build_import_graph,
            built from the same root directory as the file tree optim
            was seeded from.
        ax: Axes to draw on. A new figure and axes are created if None.
        arrow_props: Keyword arguments forwarded to matplotlib's
            Axes.annotate as arrowprops, merged over the same default
            style used by plot_treemap_with_imports. Keys given here
            override the corresponding default.

    Returns:
        The axes the diagram and import edges were drawn on.

    Raises:
        ValueError: If optim.optimize() has not been called yet.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))
    assert ax is not None

    for res in reversed(optim.sets_):
        cx, cy = res["center"]
        xs = cx + res["radii"] * np.cos(res["angles"])
        ys = cy + res["radii"] * np.sin(res["angles"])
        xs, ys = np.append(xs, xs[0]), np.append(ys, ys[0])
        ax.fill(xs, ys, alpha=0.12, color="#4472c4")
        ax.plot(xs, ys, color="#4472c4", lw=1.5)

    centers: dict[Path, tuple[float, float]] = {}
    for name, (cx, cy, hw, hh) in zip(optim.leaf_names, optim.rects_):
        ax.add_patch(
            Rectangle(
                (cx - hw, cy - hh),
                2 * hw,
                2 * hh,
                facecolor="#4472c4",
                edgecolor="white",
                alpha=0.85,
                linewidth=1,
            )
        )
        ax.text(
            cx,
            cy,
            name.name,
            ha="center",
            va="center",
            fontsize=6.5,
            color="white",
            fontweight="bold",
            clip_on=True,
        )
        centers[name] = (cx, cy)

    resolved_arrow_props = {**_DEFAULT_ARROW_PROPS, **(arrow_props or {})}
    for source, target in import_graph.edges:
        if source == target or source not in centers or target not in centers:
            continue
        ax.annotate(
            "",
            xy=centers[target],
            xytext=centers[source],
            arrowprops=resolved_arrow_props,
        )

    ax.set_aspect("equal")
    ax.autoscale()
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


def _climb(directory: Path, levels: int) -> Path | None:
    """Directory reached by climbing levels - 1 steps above directory.

    Mirrors Python's relative-import level convention: level 1 (a
    single leading dot) leaves directory unchanged, level 2 goes up one
    directory, and so on. Returns None if climbing would go above the
    project root.
    """
    if levels == 1:
        return directory
    ancestors = directory.parents
    index = levels - 2
    if index >= len(ancestors):
        return None
    return ancestors[index]


def build_import_graph(
    root: str | Path,
    *,
    ignore: set[str] | None = None,
) -> nx.DiGraph:
    """Build a directed graph of import dependencies between modules.

    Parses every .py file under root and adds an edge from an importing
    module to each internal module or package it imports (edges point
    importer -> imported). Imports of external and standard-library
    packages are dropped, since they resolve to nothing under root.

    Absolute imports (e.g. from vizopt.base import X) are resolved by
    matching root.name as the top-level package name; if root is not
    itself that package's directory, absolute imports will not resolve.
    Relative imports (from .base import X, from ..templates import
    color, etc.) are resolved directly against root's directory
    structure and are unaffected by root.name.

    When an imported name refers to a further submodule or subpackage
    (e.g. from ..components import common, where common is
    components/common.py), the edge points to that submodule rather
    than to the enclosing package; otherwise it points to the module or
    package the name was imported from.

    Args:
        root: Package directory to walk. Accepts a string path or a
            pathlib.Path; relative paths are resolved against the
            current working directory.
        ignore: Forwarded to build_file_tree.

    Returns:
        Directed graph whose nodes are the same pathlib.Path keys as
        build_file_tree (relative to root, including every .py file
        even if it has no internal imports) and whose edges point from
        an importing module to each internal module or package it
        imports.

    Raises:
        NotADirectoryError: If root is not an existing directory.
    """
    root = Path(root).resolve()
    file_tree = build_file_tree(root, ignore=ignore)
    package_name = root.name

    file_nodes = {n for n in file_tree.nodes if not file_tree.nodes[n]["is_dir"]}
    dir_nodes = {n for n in file_tree.nodes if file_tree.nodes[n]["is_dir"]}

    def _resolve_module(module_dir: Path) -> Path | None:
        if module_dir == Path("."):
            return Path(".")
        module_file = module_dir.with_name(module_dir.name + ".py")
        if module_file in file_nodes:
            return module_file
        if module_dir in dir_nodes:
            return module_dir
        return None

    def _add_from_edges(source: Path, base_dir: Path, names: list[str]) -> None:
        for name in names:
            if base_dir / f"{name}.py" in file_nodes:
                graph.add_edge(source, base_dir / f"{name}.py")
                continue
            if base_dir / name in dir_nodes:
                graph.add_edge(source, base_dir / name)
                continue
            target = _resolve_module(base_dir)
            if target is not None:
                graph.add_edge(source, target)

    def _absolute_base_dir(dotted: str) -> Path | None:
        if not (dotted == package_name or dotted.startswith(package_name + ".")):
            return None
        remainder = dotted[len(package_name) :].lstrip(".")
        return Path(*remainder.split(".")) if remainder else Path(".")

    graph: nx.DiGraph = nx.DiGraph()
    graph.add_nodes_from(n for n in file_nodes if n.suffix == ".py")

    for rel_path in file_nodes:
        if rel_path.suffix != ".py":
            continue
        tree = parse_module_ast(root / rel_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_dir = _absolute_base_dir(alias.name)
                    if module_dir is None:
                        continue
                    target = _resolve_module(module_dir)
                    if target is not None:
                        graph.add_edge(rel_path, target)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    # Python's grammar guarantees module is set when level
                    # is 0 (only "from . import x" style syntax, which
                    # always has level >= 1, allows module to be None).
                    assert node.module is not None
                    base_dir = _absolute_base_dir(node.module)
                    if base_dir is None:
                        continue
                else:
                    climbed = _climb(rel_path.parent, node.level)
                    if climbed is None:
                        continue
                    base_dir = (
                        climbed / Path(*node.module.split("."))
                        if node.module
                        else climbed
                    )
                _add_from_edges(
                    rel_path, base_dir, [alias.name for alias in node.names]
                )

    return graph
