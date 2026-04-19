"""Layered graph layout optimization."""

import networkx as nx
import numpy as np
from jax import numpy as jnp
from matplotlib import pyplot as plt

from ..base import ObjectiveTerm, OptimizationProblemTemplate
from ..components.common import should_be_positive_activation

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _compute_sibling_pairs(graph: nx.DiGraph, node_name_to_id: dict) -> np.ndarray:
    """Pre-compute pairs of nodes that share a common predecessor.

    For each node, all pairs of its direct successors are recorded so the
    sibling-separation term can push them apart in the perpendicular direction.

    Args:
        graph: Directed graph.
        node_name_to_id: Mapping from node name to integer index.

    Returns:
        Integer array of shape (n_pairs, 2).
    """
    sibling_pairs: set[tuple[int, int]] = set()
    for node in graph.nodes:
        children = list(graph.successors(node))
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                a = node_name_to_id[children[i]]
                b = node_name_to_id[children[j]]
                sibling_pairs.add((min(a, b), max(a, b)))
    pairs = sorted(sibling_pairs)
    return (
        np.array(pairs, dtype=np.int32) if pairs else np.zeros((0, 2), dtype=np.int32)
    )


# ---------------------------------------------------------------------------
# JAX loss components
#
# optim_vars keys: "node_xys"
# input_params keys: "edge_indices", "node_pairs", "sibling_pairs",
#                    "min_distance", "preferred_edge_vector", "standard_direction",
#                    "initial_node_xys"
# ---------------------------------------------------------------------------


def _term_edge_direction(optim_vars, input_params):
    """Penalize deviation of edge directions from the standard direction.

    Scale-invariant: normalizes each edge vector before comparing to the
    unit standard direction. Use together with _term_edge_vector when a
    specific edge length is also desired.
    """
    edge_indices = input_params["edge_indices"]
    if len(edge_indices) == 0:
        return jnp.array(0.0)

    node_xys = optim_vars["node_xys"]
    std_dir = jnp.array(input_params["standard_direction"])  # (2,) unit vector

    start = node_xys[edge_indices[:, 0]]
    end = node_xys[edge_indices[:, 1]]
    edge_vecs = end - start  # (E, 2)

    lengths = jnp.sqrt(jnp.sum(edge_vecs**2, axis=1, keepdims=True)) + 1e-8
    edge_dirs = edge_vecs / lengths  # unit vectors (E, 2)

    return jnp.sum((edge_dirs - std_dir) ** 2)


def _term_edge_vector(optim_vars, input_params):
    """Penalize deviation of (non-normalized) edge vectors from preferred_edge_vector.

    Unlike _term_edge_direction, this term is sensitive to both direction and
    magnitude, pulling edges toward a specific length as well as orientation.
    """
    edge_indices = input_params["edge_indices"]
    if len(edge_indices) == 0:
        return jnp.array(0.0)

    node_xys = optim_vars["node_xys"]
    preferred = jnp.array(input_params["preferred_edge_vector"])  # (2,)

    start = node_xys[edge_indices[:, 0]]
    end = node_xys[edge_indices[:, 1]]
    edge_vecs = end - start  # (E, 2)

    return jnp.sum((edge_vecs - preferred) ** 2)


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


def _term_sibling_separation(optim_vars, input_params):
    """Penalize siblings (nodes sharing a common parent) that are too close
    in the direction perpendicular to the standard direction.

    Euclidean separation alone is insufficient because the edge_direction term
    pulls siblings to the same position along the standard direction, leaving
    the optimizer free to collapse them in the perpendicular axis.
    """
    sibling_pairs = input_params["sibling_pairs"]
    if len(sibling_pairs) == 0:
        return jnp.array(0.0)

    node_xys = optim_vars["node_xys"]
    min_distance = input_params["min_distance"]
    std_dir = jnp.array(input_params["standard_direction"])  # (2,)

    # Perpendicular direction: rotate std_dir by 90°
    perp_dir = jnp.array([-std_dir[1], std_dir[0]])
    perp_dir = perp_dir / (jnp.sqrt(jnp.sum(perp_dir**2)) + 1e-8)

    pos_a = node_xys[sibling_pairs[:, 0]]
    pos_b = node_xys[sibling_pairs[:, 1]]
    perp_dists = jnp.abs(jnp.sum((pos_b - pos_a) * perp_dir, axis=1))
    return jnp.sum(should_be_positive_activation(perp_dists - min_distance))


def _initialize(input_params, seed):
    initial = input_params["initial_node_xys"]
    rng = np.random.default_rng(seed)
    noise = (
        rng.standard_normal(initial.shape).astype(np.float32)
        * float(input_params["min_distance"])
        * 0.1
    )
    return {"node_xys": initial + noise}


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

    elements = [
        {
            "tag": "defs",
            "_text": (
                '<marker id="arrow" markerWidth="8" markerHeight="6"'
                ' refX="7" refY="3" orient="auto">'
                '<path d="M0,0 L0,6 L8,3 z" fill="gray"/>'
                "</marker>"
            ),
        }
    ]

    # Edges
    for i, j in edge_indices:
        x1_vals, y1_vals, x2_vals, y2_vals = [], [], [], []
        for _, s in snapshots:
            xi = to_x(s["node_xys"][i, 0])
            yi = to_y(s["node_xys"][i, 1])
            xj = to_x(s["node_xys"][j, 0])
            yj = to_y(s["node_xys"][j, 1])
            dx, dy = xj - xi, yj - yi
            dist = (dx**2 + dy**2) ** 0.5 + 1e-8
            dx_n, dy_n = dx / dist, dy / dist
            x1_vals.append(f"{xi + dx_n * node_r:.2f}")
            y1_vals.append(f"{yi + dy_n * node_r:.2f}")
            x2_vals.append(f"{xj - dx_n * node_r:.2f}")
            y2_vals.append(f"{yj - dy_n * node_r:.2f}")
        elements.append(
            {
                "tag": "line",
                "stroke": "gray",
                "stroke-width": "1.5",
                "marker-end": "url(#arrow)",
                "x1": x1_vals,
                "y1": y1_vals,
                "x2": x2_vals,
                "y2": y2_vals,
            }
        )

    # Nodes
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

    # Labels
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
        ObjectiveTerm("edge_vector", _term_edge_vector, 1.0),
        ObjectiveTerm("node_separation", _term_node_separation, 1.0),
        ObjectiveTerm("sibling_separation", _term_sibling_separation, 1.0),
    ],
    initialize=_initialize,
    plot_configuration=_plot_configuration,
    svg_configuration=_svg_configuration,
)


def make_layered_graph_input_params(
    graph: nx.DiGraph,
    min_distance: float = 1.0,
    preferred_edge_vector: tuple[float, float] = (1.0, 0.0),
) -> dict:
    """Pre-process a NetworkX digraph into input_parameters for layered_graph_template.

    Converts graph topology into pre-computed numpy arrays suitable for JAX
    JIT compilation. All pairs of nodes are pre-computed for the separation term.

    Args:
        graph: Directed graph to lay out. Edge direction defines the "from → to"
            orientation that will be aligned with preferred_edge_vector.
        min_distance: Minimum required Euclidean distance between any two nodes.
        preferred_edge_vector: Target edge vector as (dx, dy). Encodes both the
            preferred direction and the preferred edge length. Default (1, 0)
            produces a left-to-right layout with unit-length edges.
            ``standard_direction`` (used by the direction and sibling terms) is
            derived as its unit vector.

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

    sibling_pairs = _compute_sibling_pairs(graph, node_name_to_id)

    pev = np.array(preferred_edge_vector, dtype=np.float32)
    standard_direction = pev / (np.linalg.norm(pev) + 1e-8)

    return {
        "initial_node_xys": initial_node_xys.astype(np.float32),
        "edge_indices": edge_indices,
        "node_pairs": node_pairs,
        "sibling_pairs": sibling_pairs,
        "min_distance": float(min_distance),
        "preferred_edge_vector": pev,
        "standard_direction": standard_direction,
        "node_names": node_names,
    }
