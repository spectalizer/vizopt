import numpy as np
from jax import numpy as jnp

from ..base import ObjectiveTerm, OptimizationProblemTemplate
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
# Public API
# ---------------------------------------------------------------------------


def optimize_circle_packing(
    radii: list[float],
    weight_total_size=2.0,
    collision_offset=1.0,
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
        optim_kwargs: Optional keyword arguments forwarded to problem.optimize()
            (e.g. n_iters, learning_rate).

    Returns:
        List of (x, y) tuples, one per circle, in the same order as radii.
    """
    node_radii = np.array(radii, dtype=np.float32)
    n = len(node_radii)
    total_scale = float(node_radii.sum()) if n > 0 else 1.0
    initial_node_xys = np.random.rand(n, 2).astype(np.float32) * total_scale

    collision_pairs = np.array(
        [[i, j] for i in range(n) for j in range(i)], dtype=np.int32
    ) if n > 1 else np.zeros((0, 2), dtype=np.int32)

    input_parameters = {
        "node_radii": node_radii,
        "collision_pairs": collision_pairs,
        "collision_offset": collision_offset,
    }

    def initialize(_):
        return {"node_xys": initial_node_xys}

    terms = [
        ObjectiveTerm("total_size", _term_total_size, weight_total_size),
        ObjectiveTerm("collision", _term_collision),
    ]

    problem = OptimizationProblemTemplate(
        terms=terms, initialize=initialize
    ).instantiate(input_parameters)
    optim_vars, _ = problem.optimize(**(optim_kwargs or {}))

    return [tuple(float(c) for c in xy) for xy in optim_vars["node_xys"]]
