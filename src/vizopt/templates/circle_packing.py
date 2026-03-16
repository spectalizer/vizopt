import numpy as np
from jax import numpy as jnp

from ..base import ObjectiveTerm, OptimizationProblem, OptimizationProblemTemplate
from ..components import (
    calculate_collision_penalty,
    calculate_total_width_penalty,
)


# ---------------------------------------------------------------------------
# ObjectiveTerm compute functions
#
# optim_vars keys: "node_xys"
# input_params keys: "node_radii", "collision_pairs", "collision_offset"
# ---------------------------------------------------------------------------


def _term_total_size(optim_vars, input_params):
    return calculate_total_width_penalty(
        optim_vars["node_xys"], jnp.array(input_params["node_radii"])
    )


def _term_collision(optim_vars, input_params):
    return calculate_collision_penalty(
        optim_vars["node_xys"],
        jnp.array(input_params["node_radii"]),
        input_params["collision_pairs"],
        offset=input_params["collision_offset"],
    )


# ---------------------------------------------------------------------------
# Plot configuration
# ---------------------------------------------------------------------------


def _plot_configuration(optim_vars, input_params):
    import matplotlib.patches as mpatches
    from matplotlib import pyplot as plt

    positions = optim_vars["node_xys"]
    radii = input_params["node_radii"]
    n = len(radii)
    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(5, 5))
    for i in range(n):
        ax.add_patch(
            mpatches.Circle(
                positions[i],
                radius=float(radii[i]),
                color=colors[i % len(colors)],
                alpha=0.6,
                ec="black",
                linewidth=0.5,
            )
        )
    margin = float(max(radii))
    ax.set_xlim(float(positions[:, 0].min()) - margin, float(positions[:, 0].max()) + margin)
    ax.set_ylim(float(positions[:, 1].min()) - margin, float(positions[:, 1].max()) + margin)
    ax.set_aspect("equal")
    ax.set_title(f"Circle packing: {n} circles")
    plt.axis("off")
    plt.tight_layout()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_circle_packing_problem(
    radii: list[float],
    weight_total_size=2.0,
    collision_offset=1.0,
    initial_node_xys=None,
) -> OptimizationProblem:
    """Build an :class:`OptimizationProblem` for circle packing without running it.

    Useful for animation: pass a :class:`~vizopt.animation.SnapshotCallback`
    to :meth:`~vizopt.base.OptimizationProblem.optimize` and then call
    :func:`~vizopt.animation.animate`.

    Args:
        radii: List of circle radii.
        weight_total_size: Weight for the total width/height objective.
        collision_offset: Minimum required gap between circle boundaries.
        initial_node_xys: Optional initial (x, y) positions, shape (n, 2).

    Returns:
        An :class:`OptimizationProblem` with ``plot_configuration`` set.
    """
    node_radii = np.array(radii, dtype=np.float32)
    n = len(node_radii)
    if initial_node_xys is None:
        total_scale = float(node_radii.sum()) if n > 0 else 1.0
        initial_node_xys = np.random.rand(n, 2).astype(np.float32) * total_scale
    else:
        initial_node_xys = np.array(initial_node_xys, dtype=np.float32)

    collision_pairs = (
        np.array([[i, j] for i in range(n) for j in range(i)], dtype=np.int32)
        if n > 1
        else np.zeros((0, 2), dtype=np.int32)
    )

    input_parameters = {
        "node_radii": node_radii,
        "collision_pairs": collision_pairs,
        "collision_offset": collision_offset,
    }

    _initial = initial_node_xys

    def initialize(_):
        return {"node_xys": _initial}

    terms = [
        ObjectiveTerm("total_size", _term_total_size, weight_total_size),
        ObjectiveTerm("collision", _term_collision),
    ]

    return OptimizationProblemTemplate(
        terms=terms, initialize=initialize, plot_configuration=_plot_configuration
    ).instantiate(input_parameters)


def optimize_circle_packing(
    radii: list[float],
    weight_total_size=2.0,
    collision_offset=1.0,
    initial_node_xys=None,
    optim_kwargs=None,
):
    """Pack circles of given radii to minimize overlap and overall bounding box.

    Minimizes a weighted sum of:
        - total width/height: compact bounding box of all circles
        - collision penalty: circles should not overlap

    Args:
        radii: List of circle radii.
        weight_total_size: Weight for the total width/height objective.
        collision_offset: Minimum required gap between circle boundaries.
        initial_node_xys: Optional initial (x, y) positions as an array of
            shape (n, 2). If None, positions are randomly initialized.
        optim_kwargs: Optional keyword arguments forwarded to problem.optimize()
            (e.g. n_iters, learning_rate).

    Returns:
        List of (x, y) tuples, one per circle, in the same order as radii.
    """
    problem = build_circle_packing_problem(
        radii=radii,
        weight_total_size=weight_total_size,
        collision_offset=collision_offset,
        initial_node_xys=initial_node_xys,
    )
    optim_vars, _ = problem.optimize(**(optim_kwargs or {}))

    return [tuple(float(c) for c in xy) for xy in optim_vars["node_xys"]]
