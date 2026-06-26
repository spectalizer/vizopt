"""Graph-based utilities shared across Euler diagram templates."""

import networkx as nx
import numpy as np


def offsets_from_graph(
    inclusion_graph: nx.DiGraph,
    set_names: list[str],
    leaf_names: list[str],
    offset_step: float = 0.15,
    sub_step: float = 0.04,
    min_offset: float = 0.05,
    exclusion_offset: float | None = None,
) -> np.ndarray:
    """Compute per-(set, leaf) boundary offsets from a set-hierarchy graph.

    Assigns larger offsets to shallower (outer) sets so that nested set
    boundaries are drawn at visibly different sizes. Within the same depth
    level, larger sets (more leaf members) receive a slightly larger offset.
    All offsets are at least `min_offset`.

    The safety invariant `(max_same_depth_count - 1) * sub_step < offset_step`
    ensures no same-depth set ever overshoots its parent's offset.

    Non-members get `exclusion_offset` instead of the depth-based value,
    which controls how far each boundary stays from elements it must exclude.
    Increasing it creates visible spacing between sibling set boundaries where
    their elements are adjacent. Defaults to `offset_step * (max_depth + 1)`.

    Args:
        inclusion_graph: DiGraph with parent→child edges.
        set_names: Ordered list of internal node names, in the same order
            used by the optimizer (topological, as from `_sets_from_graph`).
        leaf_names: List of leaf node names (out-degree 0).
        offset_step: Offset increment per depth level.
        sub_step: Additional offset per size-rank within the same depth level.
        min_offset: Floor applied to every set's offset.
        exclusion_offset: Offset applied to non-member leaves. Defaults to
            `offset_step * (max_depth + 1)` when `None`.

    Returns:
        Array of shape `(S, N)` with one offset per (set, leaf) pair.
    """
    roots = [n for n in inclusion_graph.nodes if inclusion_graph.in_degree(n) == 0]
    depth = {}
    for root in roots:
        for node, d in nx.single_source_shortest_path_length(inclusion_graph, root).items():
            if node not in depth or d < depth[node]:
                depth[node] = d

    max_set_depth = max(depth[s] for s in set_names)
    leaf_set = set(leaf_names)
    n_leaves = {
        s: sum(1 for n in nx.descendants(inclusion_graph, s) if n in leaf_set)
        for s in set_names
    }

    offset_dict: dict[str, float] = {}
    for d in set(depth[s] for s in set_names):
        group = sorted(
            [s for s in set_names if depth[s] == d], key=lambda s: n_leaves[s]
        )
        for rank, s in enumerate(group):
            offset_dict[s] = (max_set_depth - d) * offset_step + rank * sub_step + min_offset

    if exclusion_offset is None:
        exclusion_offset = offset_step * (max_set_depth + 1)

    leaf_idx = {name: i for i, name in enumerate(leaf_names)}
    result = np.empty((len(set_names), len(leaf_names)), dtype=np.float32)
    for si, s in enumerate(set_names):
        members = {n for n in nx.descendants(inclusion_graph, s) if n in leaf_set}
        for leaf, ni in leaf_idx.items():
            result[si, ni] = offset_dict[s] if leaf in members else exclusion_offset
    return result
