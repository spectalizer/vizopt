"""Layered graph layout optimization."""

import networkx as nx
import numpy as np
from jax import numpy as jnp
from matplotlib import pyplot as plt

from ..base import ObjectiveTerm, OptimizationProblemTemplate
from ..components import should_be_positive_activation


# ---------------------------------------------------------------------------
# JAX loss components
#
# optim_vars keys: "node_xys"
# input_params keys: "edge_indices", "node_pairs", "min_distance",
#                    "standard_direction", "initial_node_xys"
# ---------------------------------------------------------------------------


def _term_edge_direction(optim_vars, input_params):
    """Penalize deviation of edge directions from the standard direction.

    For each directed edge (u, v), normalizes the edge vector and penalizes
    its squared difference from the normalized standard direction.
    """
    edge_indices = input_params["edge_indices"]
    if len(edge_indices) == 0:
        return jnp.array(0.0)

    node_xys = optim_vars["node_xys"]
    std_dir = jnp.array(input_params["standard_direction"])  # (2,)

    start = node_xys[edge_indices[:, 0]]
    end = node_xys[edge_indices[:, 1]]
    edge_vecs = end - start  # (E, 2)

    lengths = jnp.sqrt(jnp.sum(edge_vecs**2, axis=1, keepdims=True)) + 1e-8
    edge_dirs = edge_vecs / lengths  # unit vectors (E, 2)

    std_norm = std_dir / (jnp.sqrt(jnp.sum(std_dir**2)) + 1e-8)  # unit vector (2,)

    return jnp.sum((edge_dirs - std_norm) ** 2)


def _term_node_separation(optim_vars, input_params):
    """Penalize node pairs closer than min_distance."""
    node_pairs = input_params["node_pairs"]
    if len(node_pairs) == 0:
        return jnp.array(0.0)

    node_xys = optim_vars["node_xys"]
    min_distance = input_params["min_distance"]

    pos_a = node_xys[node_pairs[:, 0]]
    pos_b = node_xys[node_pairs[:, 1]]
    dists = jnp.sqrt(jnp.sum((pos_a - pos_b) ** 2, axis=1))
    return jnp.sum(should_be_positive_activation(dists - min_distance))


def _initialize(input_params):
    return {"node_xys": input_params["initial_node_xys"].copy()}


def _plot_configuration(optim_vars, input_params):
    node_xys = optim_vars["node_xys"]
    edge_indices = input_params["edge_indices"]
    node_names = input_params.get("node_names", None)

    _, ax = plt.subplots(figsize=(6, 5))

    for i, j in edge_indices:
        ax.annotate(
            "",
            xy=node_xys[j],
            xytext=node_xys[i],
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
        )

    ax.scatter(node_xys[:, 0], node_xys[:, 1], s=80, zorder=3)

    if node_names is not None:
        for k, (x, y) in enumerate(node_xys):
            ax.annotate(
                str(node_names[k]), (x, y), textcoords="offset points", xytext=(6, 4)
            )

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

layered_graph_template = OptimizationProblemTemplate(
    terms=[
        ObjectiveTerm("edge_direction", _term_edge_direction, 1.0),
        ObjectiveTerm("node_separation", _term_node_separation, 1.0),
    ],
    initialize=_initialize,
    plot_configuration=_plot_configuration,
)


def make_layered_graph_input_params(
    graph: nx.DiGraph,
    min_distance: float = 1.0,
    standard_direction: tuple[float, float] = (1.0, 0.0),
) -> dict:
    """Pre-process a NetworkX digraph into input_parameters for layered_graph_template.

    Converts graph topology into pre-computed numpy arrays suitable for JAX
    JIT compilation. All pairs of nodes are pre-computed for the separation term.

    Args:
        graph: Directed graph to lay out. Edge direction defines the "from → to"
            orientation that will be aligned with standard_direction.
        min_distance: Minimum required Euclidean distance between any two nodes.
        standard_direction: Target direction for edges as (dx, dy).
            Default (1, 0) produces a left-to-right layout.

    Returns:
        Dict suitable for layered_graph_template.instantiate(). Includes
        "node_names" for post-processing (not used in loss computation).
    """
    node_names = list(graph.nodes)
    n = len(node_names)
    node_name_to_id = {name: i for i, name in enumerate(node_names)}

    scale = min_distance * max(n**0.5, 1.0)
    initial_layout = nx.spring_layout(graph, seed=0)
    initial_node_xys = (
        np.stack([initial_layout[name] for name in node_names], axis=0) * scale
    )

    edges_list = [(node_name_to_id[u], node_name_to_id[v]) for u, v in graph.edges]
    edge_indices = (
        np.array(edges_list, dtype=np.int32)
        if edges_list
        else np.zeros((0, 2), dtype=np.int32)
    )

    i_idx, j_idx = np.triu_indices(n, k=1)
    node_pairs = np.stack([i_idx, j_idx], axis=1).astype(np.int32)

    return {
        "initial_node_xys": initial_node_xys.astype(np.float32),
        "edge_indices": edge_indices,
        "node_pairs": node_pairs,
        "min_distance": float(min_distance),
        "standard_direction": np.array(standard_direction, dtype=np.float32),
        "node_names": node_names,
    }
