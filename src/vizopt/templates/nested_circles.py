"""Optimization templates for circle-based Euler diagrams with inclusion constraints.

Provides two high-level optimizers for laying out graphs where some nodes are
visually enclosed by others (e.g. set membership diagrams, containment hierarchies):

- :class:`NestedCirclesOptimizer`: pure inclusion tree, no edges.
- :class:`LinkedNestedCirclesOptimizer`: inclusion tree plus an ordinary graph
  whose edges should be drawn short.

Both optimizers store fitted positions and radii in ``positions_`` and ``radii_``
after :meth:`optimize` is called. Leaf node radii are fixed (from ``"size"``
attributes); non-leaf / enclosing node radii are optimization variables.

Initialization helpers (:func:`treemap_node_positions`,
:func:`greedy_bottomup_node_positions`) are also public and can be used standalone to
seed custom optimizers.
"""

from collections import deque

import networkx as nx
import numpy as np
from jax import numpy as jnp

from ..base import ObjectiveTerm, OptimConfig, OptimizationProblem, OptimizationProblemTemplate, VizOptimizer
from ..components.common import (
    calculate_collision_penalty,
    calculate_total_width_penalty_for_circular_layout,
    should_be_positive_activation,
)
from ..treemap import squarify_layout


def get_random_node_positions(graph, scale=1.0):
    """Generate random initial positions for graph nodes.

    Args:
        graph: NetworkX graph whose nodes to position.
        scale: Coordinate range for positions.

    Returns:
        Dict mapping node names to (x, y) tuples.
    """
    pos = {}
    for node in graph.nodes:
        pos[node] = (scale * np.random.rand(), scale * np.random.rand())
    return pos


def treemap_node_positions(
    inclusion_graph: nx.DiGraph,
    canvas_size: float = 1.0,
) -> tuple[dict, dict]:
    """Treemap-based initial positions for a circle Euler diagram.

    Uses a squarified treemap to place nodes so that containment is encoded
    constructively: each node's rectangle is nested inside its parent's
    rectangle. Gradient descent then only needs to tune margins and overlap
    boundaries rather than discover the topology.

    Each non-leaf node's initial radius is estimated as the circumradius of
    its treemap rectangle, which is large enough to enclose the rectangle
    (and therefore all children placed inside it).

    For DAG cases where one node has two incomparable parents, the first
    parent encountered in topological order determines the node's position.

    Args:
        inclusion_graph: DiGraph with parent→child edges (edge (u, v) means
            v ⊂ u). Leaf nodes (out-degree 0) must have a ``"size"`` attribute.
        canvas_size: Side length of the square canvas for the top-level layout.

    Returns:
        Tuple ``(pos, variable_radii)`` where:

        - ``pos``: dict mapping every node name → ``(x, y)`` center
        - ``variable_radii``: dict mapping non-leaf node name → initial radius
          estimate (circumradius of the treemap rectangle)
    """
    # Bottom-up weight: each node's weight = sum of its leaf descendants' sizes.
    weights: dict = {}
    for node in reversed(list(nx.topological_sort(inclusion_graph))):
        if inclusion_graph.out_degree(node) == 0:
            weights[node] = float(inclusion_graph.nodes[node].get("size", 1.0))
        else:
            child_sum = sum(weights[c] for c in inclusion_graph.successors(node))
            weights[node] = child_sum if child_sum > 0 else 1.0

    roots = [n for n in inclusion_graph.nodes if inclusion_graph.in_degree(n) == 0]
    node_rects: dict = {}

    def _layout_children(parent_rect, children):
        if not children:
            return
        rects = squarify_layout([(c, weights[c]) for c in children], parent_rect)
        for child, child_rect in rects.items():
            if child not in node_rects:  # first-seen wins for DAG nodes
                node_rects[child] = child_rect
            _layout_children(child_rect, list(inclusion_graph.successors(child)))

    canvas_rect = (0.0, 0.0, float(canvas_size), float(canvas_size))
    if len(roots) == 1:
        node_rects[roots[0]] = canvas_rect
        _layout_children(canvas_rect, list(inclusion_graph.successors(roots[0])))
    else:
        root_rects = squarify_layout([(r, weights[r]) for r in roots], canvas_rect)
        for root, rect in root_rects.items():
            node_rects[root] = rect
            _layout_children(rect, list(inclusion_graph.successors(root)))

    pos: dict = {}
    variable_radii: dict = {}
    for node in inclusion_graph.nodes:
        x0, y0, x1, y1 = node_rects.get(node, canvas_rect)
        pos[node] = ((x0 + x1) / 2, (y0 + y1) / 2)
        if inclusion_graph.out_degree(node) > 0:
            variable_radii[node] = float(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / 2)

    return pos, variable_radii


def _greedy_place_circle(r, placed_circles, n_angles=36):
    """Find a non-overlapping position for a new circle near the current centroid.

    Tries positions tangent to each already-placed circle at ``n_angles``
    evenly-spaced angles, picks the overlap-free candidate closest to the
    centroid of all placed circle centres. Falls back to placing just outside
    the cluster along the +x direction if no tangent position is overlap-free.

    Args:
        r: Radius of the circle to place.
        placed_circles: List of ``(center_xy, radius)`` for already-placed circles.
        n_angles: Number of candidate angles to sample per placed circle.

    Returns:
        ``(x, y)`` position for the new circle.
    """
    if not placed_circles:
        return (0.0, 0.0)

    centers = np.array([c for c, _ in placed_circles])
    radii = np.array([rv for _, rv in placed_circles])
    centroid = centers.mean(axis=0)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    best_pos = None
    best_dist = float("inf")

    for (cx, cy), ri in placed_circles:
        for a in angles:
            px = cx + (ri + r) * np.cos(a)
            py = cy + (ri + r) * np.sin(a)
            dists = np.sqrt((centers[:, 0] - px) ** 2 + (centers[:, 1] - py) ** 2)
            if np.all(dists >= radii + r - 1e-9):
                d = float(np.sqrt((px - centroid[0]) ** 2 + (py - centroid[1]) ** 2))
                if d < best_dist:
                    best_dist = d
                    best_pos = (float(px), float(py))

    if best_pos is not None:
        return best_pos

    enc_r = float(np.max(np.sqrt(np.sum((centers - centroid) ** 2, axis=1)) + radii))
    return (float(centroid[0]) + enc_r + r, float(centroid[1]))


def greedy_bottomup_node_positions(
    inclusion_graph: nx.DiGraph,
    canvas_size: float = 1.0,
    margin_factor: float = 0.1,
    n_angles: int = 36,
) -> tuple[dict, dict]:
    """Greedy bottom-up packing initialization for a circle Euler diagram.

    Places nodes one at a time without any top-down pass:

    - **Leaf nodes**: placed greedily at the overlap-free position closest to
      the centroid of all currently placed circles. Leaves are initially queued
      in decreasing-radius order so large circles pack first.
    - **Non-leaf nodes**: enqueued at the *front* (taking priority over remaining
      leaves) as soon as all direct children are placed. Position = unweighted
      centroid of children; radius = enclosing circle of children + margin.
      Added to the placement list so subsequent leaves avoid the enclosing circle.

    For DAG nodes with multiple parents, each parent independently computes its
    enclosing circle from the shared child's already-fixed position.

    Args:
        inclusion_graph: DiGraph with parent->child edges (edge (u, v) means
            v is in u). Leaf nodes (out-degree 0) must have a ``"size"`` attribute.
        canvas_size: Side length of the square canvas for the final layout.
        margin_factor: Relative extra margin added to each non-leaf radius.
        n_angles: Number of candidate angles to try per placed circle when
            placing a new leaf.

    Returns:
        Tuple ``(pos, variable_radii)`` where:

        - ``pos``: dict mapping every node name -> ``(x, y)`` center
        - ``variable_radii``: dict mapping non-leaf node name -> initial radius
          estimate
    """
    node_pos: dict = {}
    node_r: dict = {}
    placed: set = set()
    placed_list: list = []  # list of (np.ndarray center, float radius)

    for node in inclusion_graph.nodes:
        if inclusion_graph.out_degree(node) == 0:
            node_r[node] = float(inclusion_graph.nodes[node].get("size", 1.0))

    children_placed_count = {n: 0 for n in inclusion_graph.nodes}
    n_children = {n: inclusion_graph.out_degree(n) for n in inclusion_graph.nodes}

    leaves_sorted = sorted(
        [n for n in inclusion_graph.nodes if inclusion_graph.out_degree(n) == 0],
        key=lambda n: -node_r[n],
    )
    queue: deque = deque(leaves_sorted)

    while queue:
        node = queue.popleft()
        if node in placed:
            continue

        if inclusion_graph.out_degree(node) == 0:
            pos = _greedy_place_circle(node_r[node], placed_list, n_angles)
            node_pos[node] = pos
            placed_list.append((np.array(pos), node_r[node]))
        else:
            children = list(inclusion_graph.successors(node))
            child_centers = np.array([node_pos[c] for c in children])
            child_radii_arr = np.array([node_r[c] for c in children])
            centroid = child_centers.mean(axis=0)
            enc_r = float(
                np.max(
                    np.sqrt(np.sum((child_centers - centroid) ** 2, axis=1))
                    + child_radii_arr
                )
            ) * (1.0 + margin_factor)
            node_r[node] = enc_r
            node_pos[node] = (float(centroid[0]), float(centroid[1]))
            placed_list.append((centroid, enc_r))

        placed.add(node)
        for parent in inclusion_graph.predecessors(node):
            children_placed_count[parent] += 1
            if children_placed_count[parent] == n_children[parent]:
                queue.appendleft(parent)  # process non-leaf before remaining leaves

    # Scale and centre on canvas
    all_nodes = list(inclusion_graph.nodes)
    all_xy = np.array([node_pos[n] for n in all_nodes])
    all_r_arr = np.array([node_r[n] for n in all_nodes])
    x_min = float(np.min(all_xy[:, 0] - all_r_arr))
    x_max = float(np.max(all_xy[:, 0] + all_r_arr))
    y_min = float(np.min(all_xy[:, 1] - all_r_arr))
    y_max = float(np.max(all_xy[:, 1] + all_r_arr))

    span = max(x_max - x_min, y_max - y_min, 1e-9)
    scale = float(canvas_size) / span
    mid_x = (x_min + x_max) / 2
    mid_y = (y_min + y_max) / 2

    pos_out: dict = {}
    variable_radii: dict = {}
    for node in inclusion_graph.nodes:
        gx, gy = node_pos[node]
        pos_out[node] = (
            (gx - mid_x) * scale + canvas_size / 2,
            (gy - mid_y) * scale + canvas_size / 2,
        )
        if inclusion_graph.out_degree(node) > 0:
            variable_radii[node] = node_r[node] * scale

    return pos_out, variable_radii


# ---------------------------------------------------------------------------
# Low-level JAX loss components
# ---------------------------------------------------------------------------


def _non_inclusion_penalty(node_xys, node_radii, inclusion_edge_indices, offset=1.0):
    """Vectorized inclusion constraint penalty.

    Convention: edge (u, v) means v is contained in u.
    """
    if len(inclusion_edge_indices) == 0:
        return jnp.array(0.0)

    including_nodes = inclusion_edge_indices[:, 0]
    included_nodes = inclusion_edge_indices[:, 1]
    including_pos = node_xys[including_nodes]
    included_pos = node_xys[included_nodes]
    dists = jnp.sqrt(jnp.sum((including_pos - included_pos) ** 2, axis=1))
    including_radii = node_radii[including_nodes]
    included_radii = node_radii[included_nodes]
    radius_diff_minus_dist = including_radii - offset - included_radii - dists
    return jnp.sum(should_be_positive_activation(radius_diff_minus_dist))


def _calculate_edge_lengths(node_xys, edge_indices):
    """Vectorized sum of edge lengths."""
    if len(edge_indices) == 0:
        return jnp.array(0.0)
    start_points = node_xys[edge_indices[:, 0]]
    end_points = node_xys[edge_indices[:, 1]]
    return jnp.sum(jnp.sqrt(jnp.sum((start_points - end_points) ** 2, axis=1)))


def _compute_collision_pairs(all_node_names, inclusion_tree):
    """Pre-compute which node pairs should be checked for collisions.

    Excludes two categories of pairs:

    - **Ancestor/descendant pairs**: one node is geometrically contained inside
      the other, so they are expected to overlap.
    - **Partial-overlap pairs**: both nodes share at least one common leaf
      descendant but neither contains the other. These pairs must be allowed to
      intersect so that their shared element(s) can satisfy both enclosure
      constraints simultaneously.

    Args:
        all_node_names: Ordered list of all node names.
        inclusion_tree: NetworkX DiGraph encoding inclusion relationships.

    Returns:
        Integer array of shape (n_pairs, 2) with index pairs to check.
    """
    leaf_set = {n for n in inclusion_tree.nodes if inclusion_tree.out_degree(n) == 0}
    all_descendants = {
        node: (
            nx.descendants(inclusion_tree, node)
            if node in inclusion_tree.nodes
            else set()
        )
        for node in all_node_names
    }
    leaf_descendants = {
        node: all_descendants[node] & leaf_set for node in all_node_names
    }
    collision_pairs = []
    for i, node_a in enumerate(all_node_names):
        for j, node_b in enumerate(all_node_names[:i]):
            if node_b in all_descendants[node_a] or node_a in all_descendants[node_b]:
                continue
            if not leaf_descendants[node_a].isdisjoint(leaf_descendants[node_b]):
                continue
            collision_pairs.append([i, j])

    return (
        np.array(collision_pairs, dtype=np.int32)
        if collision_pairs
        else np.zeros((0, 2), dtype=np.int32)
    )


# ---------------------------------------------------------------------------
# ObjectiveTerm compute functions
#
# optim_vars keys: "node_xys", "variable_node_radii"
# input_params keys: "fixed_node_radii", "collision_pairs", "inclusion_edge_indices"
#                    and optionally "edge_indices"
# ---------------------------------------------------------------------------


def _get_node_radii(optim_vars, input_params):
    """Concatenate fixed radii (input_params) and optimizable radii (optim_vars)."""
    return jnp.concatenate(
        [
            jnp.array(input_params["fixed_node_radii"]),
            optim_vars["variable_node_radii"],
        ],
        axis=0,
    )


def _term_total_size(optim_vars, input_params):
    node_radii = _get_node_radii(optim_vars, input_params)
    return calculate_total_width_penalty_for_circular_layout(
        optim_vars["node_xys"], node_radii
    )


def _term_collision(optim_vars, input_params):
    node_radii = _get_node_radii(optim_vars, input_params)
    return calculate_collision_penalty(
        optim_vars["node_xys"], node_radii, input_params["collision_pairs"]
    )


def _term_non_inclusion(optim_vars, input_params):
    node_radii = _get_node_radii(optim_vars, input_params)
    return _non_inclusion_penalty(
        optim_vars["node_xys"], node_radii, input_params["inclusion_edge_indices"]
    )


def _term_edge_length(optim_vars, input_params):
    return _calculate_edge_lengths(optim_vars["node_xys"], input_params["edge_indices"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _build_nested_circles_input(inclusion_tree, leaf_nodes, non_leaf_nodes, all_node_names):
    """Shared preprocessing for both nested circles optimizers."""
    def _node_size(node):
        if "size" in inclusion_tree.nodes[node]:
            return inclusion_tree.nodes[node]["size"]
        print(f"leaf node {node} has no size")
        return 1.0

    fixed_node_radii = np.array([_node_size(n) for n in leaf_nodes])
    total_scale = float(sum(fixed_node_radii)) if len(fixed_node_radii) > 0 else 10.0
    initial_pos = get_random_node_positions(inclusion_tree, scale=total_scale)
    initial_node_xys = np.stack([initial_pos[n] for n in all_node_names])
    initial_variable_radii = np.full(
        len(non_leaf_nodes),
        float(fixed_node_radii.max()) if len(fixed_node_radii) > 0 else 1.0,
    )
    node_name_to_id = {name: i for i, name in enumerate(all_node_names)}
    edges_list = [
        (node_name_to_id[u], node_name_to_id[v]) for u, v in inclusion_tree.edges
    ]
    inclusion_edge_indices = (
        np.array(edges_list, dtype=np.int32)
        if edges_list
        else np.zeros((0, 2), dtype=np.int32)
    )
    collision_pairs = _compute_collision_pairs(all_node_names, inclusion_tree)
    return (
        fixed_node_radii,
        initial_node_xys,
        initial_variable_radii,
        inclusion_edge_indices,
        collision_pairs,
    )


class NestedCirclesOptimizer(VizOptimizer):
    """Optimize drawing of an inclusion tree with circular nodes.

    Minimizes a weighted sum of:

    - ``total_size``: compact layouts with low overall dimensions.
    - ``collision``: non-related nodes should not overlap.
    - ``non_inclusion``: child nodes must stay inside parent nodes.

    Leaf nodes (out-degree 0) have fixed radii from their ``"size"`` attribute.
    Non-leaf nodes have optimizable radii.

    Args:
        inclusion_tree: DiGraph with an edge ``(u, v)`` if v is contained in u.
            Leaf nodes must have a ``"size"`` attribute (fixed radius).
        weight_total_size: Weight for the total width/height objective.
        weight_collision: Weight for the collision penalty.
        weight_non_inclusion: Weight for the non-inclusion penalty.
        optim_config: Optimizer settings. Uses :class:`~vizopt.base.OptimConfig`
            defaults when ``None``.
    """

    def __init__(
        self,
        inclusion_tree: nx.DiGraph,
        *,
        weight_total_size: float = 2.0,
        weight_collision: float = 1.0,
        weight_non_inclusion: float = 1.0,
    ):
        self.inclusion_tree = inclusion_tree
        self.weight_total_size = weight_total_size
        self.weight_collision = weight_collision
        self.weight_non_inclusion = weight_non_inclusion

    def _build_problem(self) -> OptimizationProblem:
        leaf_nodes = [n for n in self.inclusion_tree.nodes if self.inclusion_tree.out_degree(n) == 0]
        non_leaf_nodes = [n for n in self.inclusion_tree.nodes if self.inclusion_tree.out_degree(n) > 0]
        all_node_names = leaf_nodes + non_leaf_nodes

        (
            fixed_node_radii,
            initial_node_xys,
            initial_variable_radii,
            inclusion_edge_indices,
            collision_pairs,
        ) = _build_nested_circles_input(
            self.inclusion_tree, leaf_nodes, non_leaf_nodes, all_node_names
        )

        input_parameters = {
            "fixed_node_radii": fixed_node_radii,
            "collision_pairs": collision_pairs,
            "inclusion_edge_indices": inclusion_edge_indices,
        }

        def initialize(_, _seed):
            return {
                "node_xys": initial_node_xys,
                "variable_node_radii": initial_variable_radii,
            }

        return OptimizationProblemTemplate(
            terms=[
                ObjectiveTerm("total_size", _term_total_size, self.weight_total_size),
                ObjectiveTerm("collision", _term_collision, self.weight_collision),
                ObjectiveTerm("non_inclusion", _term_non_inclusion, self.weight_non_inclusion),
            ],
            initialize=initialize,
        ).instantiate(input_parameters)

    @property
    def positions_(self) -> dict:
        """Optimized node positions as a dict mapping node name to ``(x, y)``.

        Raises:
            ValueError: If :meth:`optimize` has not been called yet.
        """
        if not hasattr(self, "result_"):
            raise ValueError("No result yet — call optimize() first.")
        leaf_nodes = [n for n in self.inclusion_tree.nodes if self.inclusion_tree.out_degree(n) == 0]
        non_leaf_nodes = [n for n in self.inclusion_tree.nodes if self.inclusion_tree.out_degree(n) > 0]
        all_node_names = leaf_nodes + non_leaf_nodes
        node_xys = np.array(self.result_.optim_vars["node_xys"])
        return {node: tuple(float(c) for c in xy) for node, xy in zip(all_node_names, node_xys)}

    @property
    def radii_(self) -> dict:
        """Optimized radii for non-leaf nodes as a dict mapping node name to float.

        Raises:
            ValueError: If :meth:`optimize` has not been called yet.
        """
        if not hasattr(self, "result_"):
            raise ValueError("No result yet — call optimize() first.")
        non_leaf_nodes = [n for n in self.inclusion_tree.nodes if self.inclusion_tree.out_degree(n) > 0]
        return dict(zip(non_leaf_nodes, self.result_.optim_vars["variable_node_radii"]))


class LinkedNestedCirclesOptimizer(VizOptimizer):
    """Optimize drawing of a graph with circular nodes and inclusion constraints.

    Minimizes a weighted sum of:

    - ``edge_length``: shorter edges make the graph more readable.
    - ``total_size``: compact layouts with low overall dimensions.
    - ``collision``: non-related nodes should not overlap.
    - ``non_inclusion``: child nodes must stay inside parent nodes.

    Graph nodes have fixed radii from their ``"size"`` attribute. Enclosing
    nodes (present in ``inclusion_tree`` but not in ``graph``) have optimizable radii.

    Args:
        graph: NetworkX Graph with node ``"size"`` attributes.
        inclusion_tree: DiGraph with an edge ``(u, v)`` if v is contained in u.
        weight_edge_length: Weight for the edge length objective.
        weight_total_size: Weight for the total width/height objective.
        weight_collision: Weight for the collision penalty.
        weight_non_inclusion: Weight for the non-inclusion penalty.
    """

    def __init__(
        self,
        graph: nx.Graph,
        inclusion_tree: nx.DiGraph,
        *,
        weight_edge_length: float = 1.0,
        weight_total_size: float = 2.0,
        weight_collision: float = 1.0,
        weight_non_inclusion: float = 1.0,
    ):
        self.graph = graph
        self.inclusion_tree = inclusion_tree
        self.weight_edge_length = weight_edge_length
        self.weight_total_size = weight_total_size
        self.weight_collision = weight_collision
        self.weight_non_inclusion = weight_non_inclusion

    def _build_problem(self) -> OptimizationProblem:
        def _node_size(node):
            if "size" in self.graph.nodes[node]:
                return self.graph.nodes[node]["size"]
            print(f"node {node} has no size")
            return 1.0

        graph_node_names = list(self.graph.nodes)
        enclosing_node_names = sorted(list(set(self.inclusion_tree.nodes) - set(self.graph.nodes)))
        all_node_names = graph_node_names + enclosing_node_names
        node_name_to_id = {name: i for i, name in enumerate(all_node_names)}

        fixed_node_radii = np.array([_node_size(n) for n in graph_node_names])
        total_scale = float(sum(fixed_node_radii)) if len(fixed_node_radii) > 0 else 10.0

        initial_pos = get_random_node_positions(self.graph, scale=total_scale)
        for n in enclosing_node_names:
            initial_pos[n] = (
                total_scale * np.random.rand(),
                total_scale * np.random.rand(),
            )
        initial_node_xys = np.stack([initial_pos[n] for n in all_node_names])
        initial_variable_radii = np.full(
            len(enclosing_node_names),
            float(fixed_node_radii.max()) if len(fixed_node_radii) > 0 else 1.0,
        )

        edges_list = [(node_name_to_id[u], node_name_to_id[v]) for u, v in self.graph.edges]
        edge_indices = (
            np.array(edges_list, dtype=np.int32)
            if edges_list
            else np.zeros((0, 2), dtype=np.int32)
        )

        inclusion_edges_list = [
            (node_name_to_id[u], node_name_to_id[v]) for u, v in self.inclusion_tree.edges
        ]
        inclusion_edge_indices = (
            np.array(inclusion_edges_list, dtype=np.int32)
            if inclusion_edges_list
            else np.zeros((0, 2), dtype=np.int32)
        )

        collision_pairs = _compute_collision_pairs(all_node_names, self.inclusion_tree)

        input_parameters = {
            "fixed_node_radii": fixed_node_radii,
            "edge_indices": edge_indices,
            "collision_pairs": collision_pairs,
            "inclusion_edge_indices": inclusion_edge_indices,
        }

        def initialize(_, _seed):
            return {
                "node_xys": initial_node_xys,
                "variable_node_radii": initial_variable_radii,
            }

        return OptimizationProblemTemplate(
            terms=[
                ObjectiveTerm("edge_length", _term_edge_length, self.weight_edge_length),
                ObjectiveTerm("total_size", _term_total_size, self.weight_total_size),
                ObjectiveTerm("collision", _term_collision, self.weight_collision),
                ObjectiveTerm("non_inclusion", _term_non_inclusion, self.weight_non_inclusion),
            ],
            initialize=initialize,
        ).instantiate(input_parameters)

    @property
    def positions_(self) -> dict:
        """Optimized node positions as a dict mapping node name to ``(x, y)``.

        Raises:
            ValueError: If :meth:`optimize` has not been called yet.
        """
        if not hasattr(self, "result_"):
            raise ValueError("No result yet — call optimize() first.")
        graph_node_names = list(self.graph.nodes)
        enclosing_node_names = sorted(list(set(self.inclusion_tree.nodes) - set(self.graph.nodes)))
        all_node_names = graph_node_names + enclosing_node_names
        node_xys = np.array(self.result_.optim_vars["node_xys"])
        return {node: tuple(float(c) for c in xy) for node, xy in zip(all_node_names, node_xys)}

    @property
    def radii_(self) -> dict:
        """Optimized radii for enclosing nodes as a dict mapping node name to float.

        Raises:
            ValueError: If :meth:`optimize` has not been called yet.
        """
        if not hasattr(self, "result_"):
            raise ValueError("No result yet — call optimize() first.")
        enclosing_node_names = sorted(list(set(self.inclusion_tree.nodes) - set(self.graph.nodes)))
        return dict(zip(enclosing_node_names, self.result_.optim_vars["variable_node_radii"]))
