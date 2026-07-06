"""Introspection utilities for visualizing the structure of this project."""

from pathlib import Path

import networkx as nx

_DEFAULT_IGNORE = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".venv"}


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
