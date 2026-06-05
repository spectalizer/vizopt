import jax
import networkx as nx
import numpy as np
from jax import numpy as jnp

from ..base import ObjectiveTerm, OptimizationProblemTemplate, OptimConfig
from ..components.common import (
    calculate_collision_penalty,
    calculate_total_width_penalty_for_circular_layout,
    calculate_total_width_penalty_ignoring_radii,
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
    leaf_set = {
        n for n in inclusion_tree.nodes if inclusion_tree.out_degree(n) == 0
    }
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
    # return calculate_total_width_penalty_ignoring_radii(optim_vars["node_xys"])
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


def optimize_circular_layout_with_enclosed_nodes(
    inclusion_tree: nx.DiGraph,
    weight_total_size=2.0,
    optim_config: OptimConfig | None = None,
):
    """Optimize drawing of a tree with circular nodes and inclusion constraints.

    Minimizes a weighted sum of:
        - total width/height: compact layouts with low overall dimensions are better
        - collision penalty: nodes not in inclusion relationships shouldn't overlap
        - non-inclusion penalty: child nodes must stay inside parent nodes

    Args:
        inclusion_tree: a networkx DiGraph with an edge (u, v) if v is contained in u.
            Leaf nodes must have a "size" attribute (fixed radius).
            Non-leaf nodes will have optimizable radii.
        weight_total_size: weight for the total width/height objective.
        optim_config: Optimizer settings (iterations, learning rate, seeds,
            restarts). Uses :class:`~vizopt.base.OptimConfig` defaults when ``None``.

    Returns:
        Tuple of (pos, non_leaf_node_radius_dict) where pos maps node names to
        (x, y) tuples and non_leaf_node_radius_dict maps non-leaf node names to
        their optimized radii.
    """
    leaf_nodes = [n for n in inclusion_tree.nodes if inclusion_tree.out_degree(n) == 0]
    non_leaf_nodes = [
        n for n in inclusion_tree.nodes if inclusion_tree.out_degree(n) > 0
    ]
    all_node_names = leaf_nodes + non_leaf_nodes

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

    terms = [
        ObjectiveTerm("total_size", _term_total_size, weight_total_size),
        ObjectiveTerm("collision", _term_collision),
        ObjectiveTerm("non_inclusion", _term_non_inclusion),
    ]

    problem = OptimizationProblemTemplate(
        terms=terms, initialize=initialize
    ).instantiate(input_parameters)
    optim_vars, _ = problem.optimize(optim_config)

    pos = {
        node: tuple(float(c) for c in xy)
        for node, xy in zip(all_node_names, optim_vars["node_xys"])
    }
    non_leaf_node_radius_dict = dict(
        zip(non_leaf_nodes, optim_vars["variable_node_radii"])
    )
    return pos, non_leaf_node_radius_dict


def optimize_circular_layout_with_enclosed_and_linked_nodes(
    graph: nx.Graph,
    inclusion_tree: nx.DiGraph,
    weight_edge_length=1.0,
    weight_total_size=2.0,
    optim_config: OptimConfig | None = None,
):
    """Optimize drawing of a graph with circular nodes and inclusion constraints.

    Minimizes a weighted sum of:
        - edge lengths: shorter edges make the graph more readable
        - total width/height: compact layouts with low overall dimensions are better
        - collision penalty: nodes not in inclusion relationships shouldn't overlap
        - non-inclusion penalty: child nodes must stay inside parent nodes

    Args:
        graph: a networkx Graph with node "size" attributes.
        inclusion_tree: a networkx DiGraph with an edge (u, v) if v is contained in u.
        weight_edge_length: weight for the edge length objective.
        weight_total_size: weight for the total width/height objective.
        optim_config: Optimizer settings (iterations, learning rate, seeds,
            restarts). Uses :class:`~vizopt.base.OptimConfig` defaults when ``None``.
            (e.g. n_iters, learning_rate).

    Returns:
        Tuple of (pos, enclosing_node_radius_dict) where pos maps node names to
        (x, y) tuples and enclosing_node_radius_dict maps enclosing node names to
        their optimized radii.
    """

    def _node_size(node):
        if "size" in graph.nodes[node]:
            return graph.nodes[node]["size"]
        print(f"node {node} has no size")
        return 1.0

    graph_node_names = list(graph.nodes)
    enclosing_node_names = sorted(list(set(inclusion_tree.nodes) - set(graph.nodes)))
    all_node_names = graph_node_names + enclosing_node_names
    node_name_to_id = {name: i for i, name in enumerate(all_node_names)}

    fixed_node_radii = np.array([_node_size(n) for n in graph_node_names])
    total_scale = float(sum(fixed_node_radii)) if len(fixed_node_radii) > 0 else 10.0

    # Generate positions for graph nodes and enclosing nodes separately to avoid
    # graph nodes having their positions overwritten by the inclusion tree positions.
    initial_pos = get_random_node_positions(graph, scale=total_scale)
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

    edges_list = [(node_name_to_id[u], node_name_to_id[v]) for u, v in graph.edges]
    edge_indices = (
        np.array(edges_list, dtype=np.int32)
        if edges_list
        else np.zeros((0, 2), dtype=np.int32)
    )

    inclusion_edges_list = [
        (node_name_to_id[u], node_name_to_id[v]) for u, v in inclusion_tree.edges
    ]
    inclusion_edge_indices = (
        np.array(inclusion_edges_list, dtype=np.int32)
        if inclusion_edges_list
        else np.zeros((0, 2), dtype=np.int32)
    )

    collision_pairs = _compute_collision_pairs(all_node_names, inclusion_tree)

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

    terms = [
        ObjectiveTerm("edge_length", _term_edge_length, weight_edge_length),
        ObjectiveTerm("total_size", _term_total_size, weight_total_size),
        ObjectiveTerm("collision", _term_collision),
        ObjectiveTerm("non_inclusion", _term_non_inclusion),
    ]

    problem = OptimizationProblemTemplate(
        terms=terms, initialize=initialize
    ).instantiate(input_parameters)
    optim_vars, _ = problem.optimize(optim_config)

    pos = {
        node: tuple(float(c) for c in xy)
        for node, xy in zip(all_node_names, optim_vars["node_xys"])
    }
    enclosing_node_radius_dict = dict(
        zip(enclosing_node_names, optim_vars["variable_node_radii"])
    )
    return pos, enclosing_node_radius_dict
