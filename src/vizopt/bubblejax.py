import jax
import networkx as nx
import numpy as np
import pandas as pd
from jax import numpy as jnp
from matplotlib import pyplot as plt

from . import jaxopt


def get_random_node_positions(graph, scale=1.0):
    pos = {}
    for node in graph.nodes:
        pos[node] = (scale * np.random.rand(), scale * np.random.rand())
    return pos


def _should_be_positive_activation(x_value, factor=100.0):
    """A penalty for negative values"""
    return factor * jax.nn.relu(-x_value)


def _calculate_total_width_penalty(node_xys):
    """A penalty for the overall width and height of the drawing"""
    return jnp.sum(jnp.max(node_xys, axis=0) - jnp.min(node_xys, axis=0))


def _compute_collision_pairs(all_node_names, inclusion_tree):
    """Pre-compute which node pairs should be checked for collisions"""
    collision_pairs = []
    for i, node_a in enumerate(all_node_names):
        for j, node_b in enumerate(all_node_names[:i]):
            a_in_b = node_a in inclusion_tree.nodes and node_b in nx.descendants(
                inclusion_tree, node_a
            )
            b_in_a = node_b in inclusion_tree.nodes and node_a in nx.descendants(
                inclusion_tree, node_b
            )
            if not (a_in_b or b_in_a):
                collision_pairs.append([i, j])

    return (
        np.array(collision_pairs)
        if collision_pairs
        else np.zeros((0, 2), dtype=np.int32)
    )


def _calculate_collision_penalty(node_xys, node_radii, collision_pairs, offset=1.0):
    """Vectorized collision penalty calculation"""
    if len(collision_pairs) == 0:
        return jnp.array(0.0)

    pos_a = node_xys[collision_pairs[:, 0]]
    pos_b = node_xys[collision_pairs[:, 1]]
    dists = jnp.sqrt(jnp.sum((pos_a - pos_b) ** 2, axis=1))

    radii_a = node_radii[collision_pairs[:, 0]]
    radii_b = node_radii[collision_pairs[:, 1]]

    d_minus_radiuses = dists - offset - radii_a - radii_b
    penalties = _should_be_positive_activation(d_minus_radiuses)

    return jnp.sum(penalties)


def _non_inclusion_penalty(node_xys, node_radii, inclusion_edge_indices, offset=1.0):
    """Vectorized inclusion constraint penalty.

    Uses the convention: edge (u, v) means v is contained in u.
    """
    if len(inclusion_edge_indices) == 0:
        return jnp.array(0.0)

    # Convention: in edge (u, v), v is contained in u
    including_nodes = inclusion_edge_indices[:, 0]
    included_nodes = inclusion_edge_indices[:, 1]

    including_pos = node_xys[including_nodes]
    included_pos = node_xys[included_nodes]
    dists = jnp.sqrt(jnp.sum((including_pos - included_pos) ** 2, axis=1))

    including_radii = node_radii[including_nodes]
    included_radii = node_radii[included_nodes]

    radius_diff_minus_dist = including_radii - offset - included_radii - dists
    penalties = _should_be_positive_activation(radius_diff_minus_dist)

    return jnp.sum(penalties)


def _calculate_edge_lengths(node_xys, edge_indices):
    """Vectorized edge length calculation"""
    if len(edge_indices) == 0:
        return jnp.array(0.0)

    start_points = node_xys[edge_indices[:, 0]]
    end_points = node_xys[edge_indices[:, 1]]
    return jnp.sum(jnp.sqrt(jnp.sum((start_points - end_points) ** 2, axis=1)))


def _plot_optimization_history(history):
    """Plot optimization loss history on log scale"""
    _, ax = plt.subplots()
    pd.DataFrame(history).set_index("iteration").iloc[::50].plot(ax=ax)
    ax.set_yscale("log")


def optimize_circular_layout_with_enclosed_nodes(
    inclusion_tree: nx.DiGraph,
    weight_total_size=2.0,
    optim_kwargs=None,
):
    """Optimize drawing of a tree with circular nodes and inclusion constraints

    Minimizing a weighted sum of the following objectives:
        - total width/height: compact layouts with low overall dimensions are better
        - collision penalty: nodes not in inclusion relationships shouldn't overlap
        - non-inclusion penalty: child nodes must stay inside parent nodes

    Args:
        inclusion_tree: a networkx DiGraph with an edge (u, v) if v is contained in u
            Leaf nodes must have a "size" attribute (fixed radius)
            Non-leaf nodes will have optimizable radii
        weight_total_size: weight for the total width/height objective
        optim_kwargs: optional keyword arguments for the optimizer
    """
    # Identify leaf and non-leaf nodes
    leaf_nodes = [node for node in inclusion_tree.nodes if inclusion_tree.out_degree(node) == 0]
    non_leaf_nodes = [node for node in inclusion_tree.nodes if inclusion_tree.out_degree(node) > 0]

    # Get fixed radii for leaf nodes
    def _node_size(node):
        if "size" in inclusion_tree.nodes[node]:
            return inclusion_tree.nodes[node]["size"]
        print(f"leaf node {node} has no size")
        return 1.0

    leaf_node_radii = np.array([_node_size(node) for node in leaf_nodes])

    # Initialize positions for all nodes
    all_node_names = leaf_nodes + non_leaf_nodes
    total_scale = sum(leaf_node_radii) if len(leaf_node_radii) > 0 else 10.0
    pos = get_random_node_positions(inclusion_tree, scale=total_scale)

    node_name_to_id = {node_name: i for i, node_name in enumerate(all_node_names)}

    # Prepare inclusion tree relationships for vectorized operations
    inclusion_edge_indices = np.array(
        [[node_name_to_id[u], node_name_to_id[v]] for u, v in inclusion_tree.edges]
    )

    # Pre-compute collision exclusion mask
    collision_pairs = _compute_collision_pairs(all_node_names, inclusion_tree)

    # Initialize non-leaf node radii
    non_leaf_node_radii = np.full(
        len(non_leaf_nodes), float(leaf_node_radii.max()) if len(leaf_node_radii) > 0 else 1.0
    )

    # Prepare initial parameters
    node_xys = np.stack([pos[node] for node in all_node_names])
    params = {"node_xys": node_xys, "non_leaf_node_radii": non_leaf_node_radii}

    # JAX-optimized functions
    def get_node_radii(params):
        """Get all node radii from parameters"""
        return jnp.concatenate(
            [jnp.array(leaf_node_radii), params["non_leaf_node_radii"]], axis=0
        )

    @jax.jit
    def function_to_minimize(params):
        """The function to minimize, taking params as argument"""
        node_xys = params["node_xys"]
        node_radii = get_node_radii(params)

        width_penalty = weight_total_size * calculate_total_width_penalty(node_xys)
        coll_penalty = calculate_collision_penalty(node_xys, node_radii)
        incl_penalty = non_inclusion_penalty(node_xys, node_radii)

        return width_penalty + coll_penalty + incl_penalty

    initial_loss = function_to_minimize(params)

    if optim_kwargs is None:
        optim_kwargs = {}

    history = []

    def optim_callback(i_iter, loss_value, params, grads):
        if i_iter % 100 == 0:
            print(i_iter, loss_value)
        history.append({"iteration": i_iter, "loss": float(loss_value)})

    params_opt, _ = jaxopt.optimize_gradient_descent(
        params, function_to_minimize, **optim_kwargs, callback=optim_callback
    )

    node_xys_opt = params_opt["node_xys"]
    non_leaf_node_radius_dict = dict(
        zip(non_leaf_nodes, params_opt["non_leaf_node_radii"])
    )

    for node, node_xy in zip(all_node_names, node_xys_opt):
        pos[node] = tuple(float(x_or_y) for x_or_y in node_xy)

    _, ax = plt.subplots()
    pd.DataFrame(history).set_index("iteration").iloc[::50].plot(ax=ax)
    ax.set_yscale("log")
    return pos, non_leaf_node_radius_dict


def optimize_circular_layout_with_enclosed_and_linked_nodes(
    graph: nx.Graph,
    inclusion_tree: nx.DiGraph,
    weight_edge_length=1.0,
    weight_total_size=2.0,
    optim_kwargs=None,
):
    """Optimize drawing of a graph with circular nodes and inclusion constraints

    Minimizing a weighted sum of the following objectives:
        - edge lengths: all things being equal, shorter edge lengths make the graph more readable.
        - total width: compact layouts with low overall width are better
        - collision penalty
        - non-inclusion penalty

    Args:
        graph: a networkx DiGraph
        inclusion_tree: a networkx DiGraph
            with an edge (u, v) if u is in v
    """

    # Pre-process all the graph data to avoid Python loops in JAX functions
    def _node_size(node):
        if "size" in graph.nodes[node]:
            return graph.nodes[node]["size"]
        print(f"node {node} has no size")
        return 1.0

    all_node_names = sorted(list(set(graph.nodes).union(set(inclusion_tree.nodes))))
    pos = get_random_node_positions(graph)
    node_names = list(pos.keys())
    fixed_node_radii = np.array([_node_size(node) for node in node_names])
    pos = get_random_node_positions(graph, scale=sum(fixed_node_radii))

    enclosing_node_names = sorted(list(set(inclusion_tree.nodes) - set(graph.nodes)))
    enclosing_pos = get_random_node_positions(
        inclusion_tree, scale=sum(fixed_node_radii)
    )
    pos = {**pos, **enclosing_pos}

    all_node_names = node_names + enclosing_node_names
    node_name_to_id = {node_name: i for i, node_name in enumerate(all_node_names)}

    # Convert NetworkX graph structures to JAX-friendly arrays
    # Create edge index arrays for vectorized operations
    edge_indices = np.array(
        [[node_name_to_id[u], node_name_to_id[v]] for u, v in graph.edges]
    )

    # Prepare inclusion tree relationships for vectorized operations
    inclusion_edge_indices = np.array(
        [[node_name_to_id[u], node_name_to_id[v]] for u, v in inclusion_tree.edges]
    )

    # Pre-compute collision exclusion mask
    collision_pairs = _compute_collision_pairs(all_node_names, inclusion_tree)

    # Initialize enclosing node radii
    enclosing_node_radii = np.full(
        len(enclosing_node_names), float(fixed_node_radii.max())
    )

    # Prepare initial parameters
    node_xys = np.stack([pos[node] for node in all_node_names])
    params = {"node_xys": node_xys, "enclosing_node_radii": enclosing_node_radii}

    # JAX-optimized functions
    # @jax.jit
    def get_node_radii(params):
        """Get all node radii from parameters"""
        return jnp.concatenate(
            [jnp.array(fixed_node_radii), params["enclosing_node_radii"]], axis=0
        )

    # @jax.jit
    def edge_lengths(node_xys):
        """Vectorized edge length calculation"""
        if len(edge_indices) == 0:
            return jnp.array(0.0)

        start_points = node_xys[edge_indices[:, 0]]
        end_points = node_xys[edge_indices[:, 1]]
        return jnp.sum(jnp.sqrt(jnp.sum((start_points - end_points) ** 2, axis=1)))

    def should_be_positive_activation(x_value, factor=100.0):
        """A penalty for negative values"""
        return factor * jax.nn.relu(-x_value)

    def calculate_total_width_penalty(node_xys):
        """A penalty for the overall width of the drawing"""
        return jnp.sum(jnp.max(node_xys, axis=0) - jnp.min(node_xys, axis=0))

    def calculate_collision_penalty(node_xys, node_radii, offset=1.0):
        """Vectorized collision penalty calculation"""
        if len(collision_pairs) == 0:
            return jnp.array(0.0)

        # Extract node positions for all collision pairs
        pos_a = node_xys[collision_pairs[:, 0]]
        pos_b = node_xys[collision_pairs[:, 1]]

        # Calculate distances between nodes
        dists = jnp.sqrt(jnp.sum((pos_a - pos_b) ** 2, axis=1))

        # Get radii for collision pairs
        radii_a = node_radii[collision_pairs[:, 0]]
        radii_b = node_radii[collision_pairs[:, 1]]

        # Calculate penalties
        d_minus_radiuses = dists - offset - radii_a - radii_b
        penalties = should_be_positive_activation(d_minus_radiuses)

        return jnp.sum(penalties)

    # @jax.jit
    def non_inclusion_penalty(node_xys, node_radii, offset=1.0):
        """Vectorized inclusion constraint penalty"""
        if len(inclusion_edge_indices) == 0:
            return jnp.array(0.0)

        # Using the convention v in (u,v) if u is in v
        including_nodes = inclusion_edge_indices[:, 1]
        included_nodes = inclusion_edge_indices[:, 0]

        # Extract positions
        including_pos = node_xys[including_nodes]
        included_pos = node_xys[included_nodes]

        # Calculate distances
        dists = jnp.sqrt(jnp.sum((including_pos - included_pos) ** 2, axis=1))

        # Get radii
        including_radii = node_radii[including_nodes]
        included_radii = node_radii[included_nodes]

        # Calculate penalties
        radius_diff_minus_dist = including_radii - offset - included_radii - dists
        penalties = should_be_positive_activation(radius_diff_minus_dist)

        return jnp.sum(penalties)

    @jax.jit
    def function_to_minimize(params):
        """The function to minimize, taking params as argument"""
        node_xys = params["node_xys"]
        node_radii = get_node_radii(params)

        edge_len_penalty = weight_edge_length * edge_lengths(node_xys)
        width_penalty = weight_total_size * calculate_total_width_penalty(node_xys)
        coll_penalty = calculate_collision_penalty(node_xys, node_radii)
        incl_penalty = non_inclusion_penalty(node_xys, node_radii)

        return edge_len_penalty + width_penalty + coll_penalty + incl_penalty

    initial_loss = function_to_minimize(params)

    if optim_kwargs is None:
        optim_kwargs = {}

    history = []

    def optim_callback(i_iter, loss_value, params, grads):
        if i_iter % 100 == 0:
            print(i_iter, loss_value)
        history.append({"iteration": i_iter, "loss": float(loss_value)})

    params_opt, _ = jaxopt.optimize_gradient_descent(
        params, function_to_minimize, **optim_kwargs, callback=optim_callback
    )

    node_xys_opt = params_opt["node_xys"]
    enclosing_node_radius_dict = dict(
        zip(enclosing_node_names, params_opt["enclosing_node_radii"])
    )

    for node, node_xy in zip(all_node_names, node_xys_opt):
        pos[node] = tuple(float(x_or_y) for x_or_y in node_xy)

    _, ax = plt.subplots()
    pd.DataFrame(history).set_index("iteration").iloc[::50].plot(ax=ax)
    ax.set_yscale("log")
    return pos, enclosing_node_radius_dict
