"""Radially convex (star-shaped) set optimization.

A star-shaped region is represented by its center (cx, cy) and K radii
at uniformly-spaced angles θ_k = 2πk/K. The boundary point at angle k is:

    p_k = center + radii[k] * [cos(θ_k), sin(θ_k)]

This module provides:
  - JAX loss terms for enclosure, area, and perimeter
  - optimize_enclosing_radially_convex_set(): finds a star-shaped region
    that encloses a given set of circles while minimizing area and perimeter
"""

import numpy as np
from jax import numpy as jnp

from .base import ObjectiveTerm, OptimizationProblemTemplate

# ---------------------------------------------------------------------------
# ObjectiveTerm compute functions
#
# optim_vars keys: "center" (2,), "radii" (K,)
# input_params keys: "circles" (N, 3): [cx, cy, r], "angles" (K,)
# ---------------------------------------------------------------------------


def _projections(center, circles, angles):
    """Compute projection[k, i] = support of circle i in direction θ_k from center."""
    dx = circles[:, 0] - center[0]
    dy = circles[:, 1] - center[1]
    return (
        jnp.cos(angles)[:, None] * dx[None, :]
        + jnp.sin(angles)[:, None] * dy[None, :]
        + circles[:, 2][None, :]
    )  # (K, N)


def _term_enclosure(optim_vars, input_params):
    """Penalty for not enclosing all input circles.

    For each angle θ_k, the farthest extent of circle i in direction θ_k
    from the enclosing center is:

        projection[k, i] = (cx_i - cx)*cos(θ_k) + (cy_i - cy)*sin(θ_k) + r_i

    Penalizes squared violations of radii[k] >= projection[k, i].
    """
    center = optim_vars["center"]  # (2,)
    radii = optim_vars["radii"]  # (K,)
    projs = _projections(center, input_params["circles"], input_params["angles"])
    violations = projs - radii[:, None]  # (K, N)
    return jnp.sum(jnp.maximum(0.0, violations) ** 2)


def _term_slack(optim_vars, input_params):
    """Penalty for radii exceeding the minimum required to cover each circle.

    required[k] = max_i projection[k, i] is the tightest feasible radius at
    angle θ_k. Penalising radii[k] > required[k] gives the centre a gradient
    even when the enclosure constraint is already satisfied, allowing the
    optimiser to relocate the centre to reduce coverage slack.
    """
    center = optim_vars["center"]  # (2,)
    radii = optim_vars["radii"]  # (K,)
    projs = _projections(center, input_params["circles"], input_params["angles"])
    required = jnp.max(projs, axis=1)  # (K,)
    excess = jnp.maximum(0.0, radii - required)
    return jnp.sum(excess**2)


def _term_area(optim_vars, input_params):
    """Approximate area of the star polygon via the shoelace formula.

    Area = (1/2) * sin(2π/K) * Σ_k radii[k] * radii[k+1]
    """
    radii = optim_vars["radii"]  # (K,)
    K = radii.shape[0]
    delta_theta = 2 * jnp.pi / K
    return 0.5 * jnp.sin(delta_theta) * jnp.sum(radii * jnp.roll(radii, -1))


def _term_perimeter(optim_vars, input_params):
    """Approximate perimeter as the sum of chord lengths between adjacent boundary points."""
    center = optim_vars["center"]  # (2,)
    radii = optim_vars["radii"]  # (K,)
    angles = input_params["angles"]  # (K,)

    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)  # (K, 2)
    points = center[None, :] + radii[:, None] * directions  # (K, 2)
    points_next = jnp.roll(points, -1, axis=0)
    return jnp.sum(jnp.sqrt(jnp.sum((points_next - points) ** 2, axis=1)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def optimize_enclosing_radially_convex_set(
    circles,
    K=32,
    weight_area=1.0,
    weight_perimeter=1.0,
    optim_kwargs=None,
    weight_slack=1.0,
):
    """Find a star-shaped region that encloses all given circles.

    Minimizes a weighted sum of:
        - enclosure penalty: all circles must lie within the star-shaped boundary
        - area: area of the star polygon
        - perimeter: perimeter of the star polygon

    Args:
        circles: sequence of (cx, cy, r) triples, or array of shape (N, 3).
        K: number of angular samples defining the boundary polygon.
        weight_area: weight for the area objective.
        weight_perimeter: weight for the perimeter objective.
        optim_kwargs: keyword arguments forwarded to problem.optimize()
            (e.g. n_iters, learning_rate).

    Returns:
        Tuple of (result, history) where result is a dict with:
            "center": array of shape (2,), the center of the star-shaped region
            "radii": array of shape (K,), boundary radii at each angle
            "angles": array of shape (K,), uniformly spaced in [0, 2π)
        and history is the list of per-iteration loss dicts.
    """
    circles_array = np.asarray(circles, dtype=np.float32)
    if circles_array.ndim == 1:
        circles_array = circles_array[None, :]
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False).astype(np.float32)

    # Initialize center at centroid of circle centers.
    initial_center = np.mean(circles_array[:, :2], axis=0).astype(np.float32)

    # Initialize radii to just cover each circle at each sampled angle.
    dx = circles_array[:, 0] - initial_center[0]
    dy = circles_array[:, 1] - initial_center[1]
    projections = (
        np.cos(angles)[:, None] * dx[None, :]
        + np.sin(angles)[:, None] * dy[None, :]
        + circles_array[:, 2][None, :]
    )  # (K, N)
    initial_radii = np.maximum(projections.max(axis=1), 1.0).astype(np.float32)

    input_parameters = {
        "circles": circles_array,
        "angles": angles,
    }

    def initialize(_):
        return {
            "center": initial_center,
            "radii": initial_radii,
        }

    terms = [
        ObjectiveTerm("enclosure", _term_enclosure, 10.0),
        ObjectiveTerm("slack", _term_slack, weight_slack),
        ObjectiveTerm("area", _term_area, weight_area),
        ObjectiveTerm("perimeter", _term_perimeter, weight_perimeter),
    ]

    problem = OptimizationProblemTemplate(
        terms=terms, initialize=initialize
    ).instantiate(input_parameters)
    optim_vars, history = problem.optimize(**(optim_kwargs or {}))

    return {
        "center": np.array(optim_vars["center"]),
        "radii": np.array(optim_vars["radii"]),
        "angles": angles,
    }, history
