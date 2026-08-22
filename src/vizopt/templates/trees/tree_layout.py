"""Node-link tree layout optimization.

Lays out a rooted tree (a NetworkX `DiGraph` arborescence, parent → child
edges) so that: nodes at the same depth align along a growth axis, parents
are centered over their children, siblings and unrelated subtrees keep a
minimum separation in a fixed left-to-right order, and edges tend toward a
preferred length.
"""

import networkx as nx
import numpy as np
from jax import numpy as jnp
from matplotlib import pyplot as plt

from ...base import (
    ObjectiveTerm,
    OptimizationProblem,
    OptimizationProblemTemplate,
    VizOptimizer,
)
from ...components.common import (
    calculate_total_width_penalty_ignoring_radii,
    should_be_positive_activation,
)

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _children_in_order(graph: nx.DiGraph, node) -> list:
    """Return a node's children in a fixed left-to-right order.

    Uses the `"order"` node attribute when every child carries one,
    otherwise falls back to the successor iteration order of the graph.
    """
    children = list(graph.successors(node))
    if children and all("order" in graph.nodes[c] for c in children):
        children = sorted(children, key=lambda c: graph.nodes[c]["order"])
    return children


def _dfs_preorder(graph: nx.DiGraph, root) -> list:
    """DFS preorder traversal visiting children in `_children_in_order`.

    For a tree, this also lists nodes at any given depth in left-to-right
    order, since a subtree is fully traversed before its next sibling.
    """
    order: list = []

    def visit(node):
        order.append(node)
        for child in _children_in_order(graph, node):
            visit(child)

    visit(root)
    return order


def _initial_cross_coordinates(
    graph: nx.DiGraph, root, node_name_to_id: dict, min_distance: float
) -> np.ndarray:
    """Bottom-up initial cross-axis coordinates: leaves evenly spaced,
    each internal node at the mean of its children's coordinates.
    """
    n = len(node_name_to_id)
    cross_coord = np.zeros(n, dtype=np.float32)
    leaf_counter = 0

    def assign(node):
        nonlocal leaf_counter
        children = _children_in_order(graph, node)
        if not children:
            cross_coord[node_name_to_id[node]] = leaf_counter * min_distance
            leaf_counter += 1
            return
        for child in children:
            assign(child)
        child_ids = [node_name_to_id[c] for c in children]
        cross_coord[node_name_to_id[node]] = np.mean(cross_coord[child_ids])

    assign(root)
    cross_coord -= cross_coord.mean()
    return cross_coord


# ---------------------------------------------------------------------------
# JAX loss components
#
# optim_vars keys: "node_xys"
# input_params keys: "edge_indices", "order_pairs", "depths", "min_distance",
#                    "layer_spacing", "growth_direction", "cross_direction",
#                    "preferred_edge_length", "initial_node_xys"
# ---------------------------------------------------------------------------


def _term_depth_alignment(optim_vars, input_params):
    """Penalize deviation of each node's growth-axis coordinate from its
    depth-level target (`depth * layer_spacing`).
    """
    node_xys = optim_vars["node_xys"]
    growth_dir = jnp.array(input_params["growth_direction"])
    depths = input_params["depths"]
    layer_spacing = input_params["layer_spacing"]

    growth_coord = node_xys @ growth_dir
    target = depths * layer_spacing
    return jnp.sum((growth_coord - target) ** 2)


def _term_parent_centering(optim_vars, input_params):
    """Penalize a parent's cross-axis coordinate deviating from the mean
    cross-axis coordinate of its children.
    """
    edge_indices = input_params["edge_indices"]
    if len(edge_indices) == 0:
        return jnp.array(0.0)

    node_xys = optim_vars["node_xys"]
    cross_dir = jnp.array(input_params["cross_direction"])
    cross_coord = node_xys @ cross_dir  # (N,)

    parent_idx = edge_indices[:, 0]
    child_idx = edge_indices[:, 1]
    n = node_xys.shape[0]
    child_cross_sum = jnp.zeros(n).at[parent_idx].add(cross_coord[child_idx])
    child_count = jnp.zeros(n).at[parent_idx].add(1.0)
    has_children = child_count > 0
    mean_cross = jnp.where(
        has_children, child_cross_sum / jnp.maximum(child_count, 1.0), cross_coord
    )
    diff = jnp.where(has_children, cross_coord - mean_cross, 0.0)
    return jnp.sum(diff**2)


def _term_order_separation(optim_vars, input_params):
    """Penalize order_pairs (nodes adjacent in fixed left-to-right order,
    within the same depth level) whose cross-axis gap is below min_distance.

    One-directional: only the later node (in fixed order) is required to be
    ahead of the earlier one, so this both separates and preserves order.
    """
    order_pairs = input_params["order_pairs"]
    if len(order_pairs) == 0:
        return jnp.array(0.0)

    node_xys = optim_vars["node_xys"]
    cross_dir = jnp.array(input_params["cross_direction"])
    min_distance = input_params["min_distance"]

    cross_coord = node_xys @ cross_dir
    gap = cross_coord[order_pairs[:, 1]] - cross_coord[order_pairs[:, 0]]
    return jnp.sum(should_be_positive_activation(gap - min_distance))


def _term_edge_length(optim_vars, input_params):
    """Penalize parent-child edge lengths deviating from preferred_edge_length."""
    edge_indices = input_params["edge_indices"]
    if len(edge_indices) == 0:
        return jnp.array(0.0)

    node_xys = optim_vars["node_xys"]
    preferred_length = input_params["preferred_edge_length"]

    start = node_xys[edge_indices[:, 0]]
    end = node_xys[edge_indices[:, 1]]
    lengths = jnp.sqrt(jnp.sum((end - start) ** 2, axis=1) + 1e-12)
    return jnp.sum((lengths - preferred_length) ** 2)


def _term_compactness(optim_vars, _input_params):
    """Penalize the overall bounding-box extent of the drawing."""
    return calculate_total_width_penalty_ignoring_radii(optim_vars["node_xys"])


def _initialize(input_params, seed):
    initial = input_params["initial_node_xys"]
    rng = np.random.default_rng(seed)
    noise = (
        rng.standard_normal(initial.shape).astype(np.float32)
        * float(input_params["min_distance"])
        * 0.1
    )
    return {"node_xys": initial + noise}


# ---------------------------------------------------------------------------
# Plotting / animation
# ---------------------------------------------------------------------------


def _plot_configuration(optim_vars, input_params):
    node_xys = optim_vars["node_xys"]
    edge_indices = input_params["edge_indices"]
    node_names = input_params.get("node_names", None)

    _, ax = plt.subplots(figsize=(6, 5))

    for i, j in edge_indices:
        xi, yi = node_xys[i]
        xj, yj = node_xys[j]
        ax.plot([xi, xj], [yi, yj], color="gray", lw=1.5, zorder=1)

    ax.scatter(node_xys[:, 0], node_xys[:, 1], s=80, zorder=3, color="steelblue")

    if node_names is not None:
        for k, (x, y) in enumerate(node_xys):
            ax.annotate(
                str(node_names[k]), (x, y), textcoords="offset points", xytext=(6, 4)
            )

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def _svg_configuration(snapshots, input_params, size):
    all_xys = np.stack([s["node_xys"] for _, s in snapshots])  # (frames, n, 2)
    edge_indices = input_params["edge_indices"]
    node_names = input_params.get("node_names", None)

    margin = 0.5
    x_min = all_xys[:, :, 0].min() - margin
    x_max = all_xys[:, :, 0].max() + margin
    y_min = all_xys[:, :, 1].min() - margin
    y_max = all_xys[:, :, 1].max() + margin
    span = max(x_max - x_min, y_max - y_min)

    def to_x(x):
        return float((x - x_min) / span * size)

    def to_y(y):
        return float((1 - (y - y_min) / span) * size)

    node_r = 6  # node circle radius in SVG pixels

    elements = []

    for i, j in edge_indices:
        elements.append(
            {
                "tag": "line",
                "stroke": "gray",
                "stroke-width": "1.5",
                "x1": [f"{to_x(s['node_xys'][i, 0]):.2f}" for _, s in snapshots],
                "y1": [f"{to_y(s['node_xys'][i, 1]):.2f}" for _, s in snapshots],
                "x2": [f"{to_x(s['node_xys'][j, 0]):.2f}" for _, s in snapshots],
                "y2": [f"{to_y(s['node_xys'][j, 1]):.2f}" for _, s in snapshots],
            }
        )

    n = all_xys.shape[1]
    for k in range(n):
        elements.append(
            {
                "tag": "circle",
                "r": str(node_r),
                "fill": "steelblue",
                "cx": [f"{to_x(s['node_xys'][k, 0]):.2f}" for _, s in snapshots],
                "cy": [f"{to_y(s['node_xys'][k, 1]):.2f}" for _, s in snapshots],
            }
        )

    if node_names is not None:
        for k, name in enumerate(node_names):
            elements.append(
                {
                    "tag": "text",
                    "font-size": "12",
                    "font-family": "sans-serif",
                    "fill": "black",
                    "_text": str(name),
                    "x": [
                        f"{to_x(s['node_xys'][k, 0]) + node_r + 2:.2f}"
                        for _, s in snapshots
                    ],
                    "y": [f"{to_y(s['node_xys'][k, 1]) - 4:.2f}" for _, s in snapshots],
                }
            )

    return elements


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_tree_input_params(
    graph: nx.DiGraph,
    min_distance: float = 1.0,
    layer_spacing: float = 1.0,
    growth_direction: tuple[float, float] = (0.0, -1.0),
    preferred_edge_length: float | None = None,
) -> dict:
    """Pre-process a rooted-tree DiGraph into input_parameters for tree layout.

    Args:
        graph: A tree, i.e. a NetworkX arborescence with parent → child edges:
            weakly connected, acyclic, exactly one node with in-degree 0
            (the root), every other node with in-degree 1. Child order for
            crossing-avoidance is taken from each node's `"order"` attribute
            if every child of that node has one, otherwise from successor
            iteration order.
        min_distance: Minimum required cross-axis gap between order-adjacent
            nodes at the same depth.
        layer_spacing: Growth-axis distance between consecutive depth levels.
        growth_direction: Unit vector (as `(dx, dy)`) along which depth
            increases. Default `(0, -1)` draws the root at the top with
            depth increasing downward; use `(1, 0)` for a left-to-right tree.
        preferred_edge_length: Target length for parent-child edges. Defaults
            to `layer_spacing` when `None`.

    Returns:
        Dict suitable for `TreeLayoutOptimizer` or for direct use with
        `~vizopt.base.OptimizationProblemTemplate`. Includes `"node_names"`
        for post-processing (not used in loss computation).

    Raises:
        ValueError: If `graph` is not a rooted tree (single root, acyclic,
            one parent per non-root node).
    """
    if not nx.is_arborescence(graph):
        raise ValueError(
            "graph must be a rooted tree: weakly connected, acyclic, with "
            "exactly one root (in-degree 0) and every other node having "
            "exactly one parent."
        )
    root = next(n for n in graph.nodes if graph.in_degree(n) == 0)

    node_names = _dfs_preorder(graph, root)
    n = len(node_names)
    node_name_to_id = {name: i for i, name in enumerate(node_names)}

    depths_dict = nx.single_source_shortest_path_length(graph, root)
    depths = np.array([depths_dict[name] for name in node_names], dtype=np.float32)

    edges_list = [(node_name_to_id[u], node_name_to_id[v]) for u, v in graph.edges]
    edge_indices = (
        np.array(edges_list, dtype=np.int32)
        if edges_list
        else np.zeros((0, 2), dtype=np.int32)
    )

    by_depth: dict[int, list[int]] = {}
    for idx, name in enumerate(node_names):
        by_depth.setdefault(int(depths_dict[name]), []).append(idx)
    order_pairs_list = [
        (a, b) for idx_list in by_depth.values() for a, b in zip(idx_list, idx_list[1:])
    ]
    order_pairs = (
        np.array(order_pairs_list, dtype=np.int32)
        if order_pairs_list
        else np.zeros((0, 2), dtype=np.int32)
    )

    growth_dir = np.array(growth_direction, dtype=np.float32)
    growth_dir = growth_dir / (np.linalg.norm(growth_dir) + 1e-8)
    cross_dir = np.array([-growth_dir[1], growth_dir[0]], dtype=np.float32)

    if preferred_edge_length is None:
        preferred_edge_length = layer_spacing

    cross_coord = _initial_cross_coordinates(graph, root, node_name_to_id, min_distance)
    growth_coord = depths * layer_spacing
    initial_node_xys = (
        growth_coord[:, None] * growth_dir[None, :]
        + cross_coord[:, None] * cross_dir[None, :]
    ).astype(np.float32)

    return {
        "initial_node_xys": initial_node_xys,
        "edge_indices": edge_indices,
        "order_pairs": order_pairs,
        "depths": depths,
        "min_distance": np.float32(min_distance),
        "layer_spacing": np.float32(layer_spacing),
        "growth_direction": growth_dir,
        "cross_direction": cross_dir,
        "preferred_edge_length": np.float32(preferred_edge_length),
        "node_names": node_names,
        "root": root,
    }


class TreeLayoutOptimizer(VizOptimizer):
    """Optimize node positions in a rooted tree to produce a node-link layout.

    Minimizes a weighted sum of:

    - `depth_alignment`: penalizes deviation of each node's growth-axis
      coordinate from its depth-level target.
    - `parent_centering`: penalizes a parent's cross-axis coordinate
      deviating from the mean of its children's.
    - `order_separation`: penalizes nodes adjacent in fixed left-to-right
      order (within the same depth) that are closer than `min_distance`.
      Order is fixed from the graph, so this prevents overlap and edge
      crossings by construction rather than by search.
    - `edge_length`: penalizes parent-child edges deviating from
      `preferred_edge_length`.
    - `compactness`: penalizes the overall bounding-box extent (disabled
      by default).

    Args:
        graph: A tree, i.e. a NetworkX arborescence with parent → child
            edges. See `make_tree_input_params` for the exact requirement
            and how child order is determined.
        min_distance: Minimum required cross-axis gap between order-adjacent
            nodes at the same depth.
        layer_spacing: Growth-axis distance between consecutive depth levels.
        growth_direction: Unit vector `(dx, dy)` along which depth increases.
            Default `(0, -1)` draws the root at the top, depth increasing
            downward.
        preferred_edge_length: Target length for parent-child edges. Defaults
            to `layer_spacing` when `None`.
        weight_depth_alignment: Weight for the depth alignment term.
        weight_parent_centering: Weight for the parent centering term.
        weight_order_separation: Weight for the order separation term.
        weight_edge_length: Weight for the edge length term.
        weight_compactness: Weight for the compactness term. Default 0.0
            (disabled).
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        *,
        min_distance: float = 1.0,
        layer_spacing: float = 1.0,
        growth_direction: tuple[float, float] = (0.0, -1.0),
        preferred_edge_length: float | None = None,
        weight_depth_alignment: float = 1.0,
        weight_parent_centering: float = 1.0,
        weight_order_separation: float = 10.0,
        weight_edge_length: float = 1.0,
        weight_compactness: float = 0.0,
    ):
        self.graph = graph
        self.min_distance = min_distance
        self.layer_spacing = layer_spacing
        self.growth_direction = growth_direction
        self.preferred_edge_length = preferred_edge_length
        self.weight_depth_alignment = weight_depth_alignment
        self.weight_parent_centering = weight_parent_centering
        self.weight_order_separation = weight_order_separation
        self.weight_edge_length = weight_edge_length
        self.weight_compactness = weight_compactness

    def _build_problem(self) -> OptimizationProblem:
        input_parameters = make_tree_input_params(
            self.graph,
            min_distance=self.min_distance,
            layer_spacing=self.layer_spacing,
            growth_direction=self.growth_direction,
            preferred_edge_length=self.preferred_edge_length,
        )
        return OptimizationProblemTemplate(
            terms=[
                ObjectiveTerm(
                    "depth_alignment",
                    _term_depth_alignment,
                    self.weight_depth_alignment,
                ),
                ObjectiveTerm(
                    "parent_centering",
                    _term_parent_centering,
                    self.weight_parent_centering,
                ),
                ObjectiveTerm(
                    "order_separation",
                    _term_order_separation,
                    self.weight_order_separation,
                ),
                ObjectiveTerm(
                    "edge_length", _term_edge_length, self.weight_edge_length
                ),
                ObjectiveTerm(
                    "compactness", _term_compactness, self.weight_compactness
                ),
            ],
            initialize=_initialize,
            plot_configuration=_plot_configuration,
            svg_configuration=_svg_configuration,
        ).instantiate(input_parameters)

    @property
    def node_positions_(self) -> dict:
        """Optimized node positions as a dict mapping node name to `(x, y)`.

        Raises:
            ValueError: If `optimize` has not been called yet.
        """
        if not hasattr(self, "result_"):
            raise ValueError("No result yet — call optimize() first.")
        node_names = self.problem_.input_parameters["node_names"]
        node_xys = np.array(self.result_.optim_vars["node_xys"])
        return {
            name: tuple(float(c) for c in xy) for name, xy in zip(node_names, node_xys)
        }
