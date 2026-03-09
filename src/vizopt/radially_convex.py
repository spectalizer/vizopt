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


_MIN_RADIUS = 0.1


# ---------------------------------------------------------------------------
# Multi-set objective term compute functions
#
# optim_vars keys: "centers" (S, 2), "radii" (S, K)
# input_params keys: "circles" (N, 3): [cx, cy, r], "angles" (K,),
#                    "membership" (S, N): True where circle n is in set s
# ---------------------------------------------------------------------------


def _enclosure_penalty(centers, radii, circle_xy, r, angles, membership):
    """Core enclosure computation shared across fixed and movable variants.

    Args:
        centers: (S, 2) set centers.
        radii: (S, K) boundary radii.
        circle_xy: (N, 2) circle positions.
        r: (S, N) effective radii (with any offsets already applied).
        angles: (K,) ray angles.
        membership: (S, N) boolean mask, True where circle n belongs to set s.

    Returns:
        Scalar penalty.
    """
    dx = circle_xy[None, :, 0] - centers[:, None, 0]  # (S, N)
    dy = circle_xy[None, :, 1] - centers[:, None, 1]  # (S, N)

    cos_a = jnp.cos(angles)  # (K,)
    sin_a = jnp.sin(angles)  # (K,)

    tang = (
        cos_a[None, :, None] * dx[:, None, :] + sin_a[None, :, None] * dy[:, None, :]
    )  # (S, K, N)
    perp = (
        -sin_a[None, :, None] * dx[:, None, :] + cos_a[None, :, None] * dy[:, None, :]
    )  # (S, K, N)

    r_sq = r[:, None, :] ** 2  # (S, 1, N)
    hits = perp**2 <= r_sq  # (S, K, N)
    r_required = tang + jnp.sqrt(jnp.maximum(0.0, r_sq - perp**2) + 1e-12)  # (S, K, N)

    in_set = membership[:, None, :]  # (S, 1, N)
    violations = jnp.where(
        hits & in_set,
        jnp.maximum(0.0, r_required - radii[:, :, None]),
        0.0,
    )
    return jnp.sum(violations**2)


def _exclusion_penalty(centers, radii, circle_xy, circle_r, angles, membership):
    """Core exclusion computation shared across fixed and movable variants.

    Args:
        centers: (S, 2) set centers.
        radii: (S, K) boundary radii.
        circle_xy: (N, 2) circle positions.
        circle_r: (N,) circle radii (no offset).
        angles: (K,) ray angles.
        membership: (S, N) boolean mask.

    Returns:
        Scalar penalty.
    """
    dx = circle_xy[None, :, 0] - centers[:, None, 0]  # (S, N)
    dy = circle_xy[None, :, 1] - centers[:, None, 1]  # (S, N)

    cos_a = jnp.cos(angles)  # (K,)
    sin_a = jnp.sin(angles)  # (K,)

    tang = (
        cos_a[None, :, None] * dx[:, None, :] + sin_a[None, :, None] * dy[:, None, :]
    )  # (S, K, N)
    perp = (
        -sin_a[None, :, None] * dx[:, None, :] + cos_a[None, :, None] * dy[:, None, :]
    )  # (S, K, N)

    r_sq = circle_r[None, None, :] ** 2  # (1, 1, N)
    hits = perp**2 <= r_sq  # (S, K, N)
    near_edge = tang - jnp.sqrt(jnp.maximum(0.0, r_sq - perp**2) + 1e-12)  # (S, K, N)

    # Only penalize when the obstacle's near face is in the forward direction of
    # the ray (near_edge > 0). When near_edge <= 0 the obstacle is behind the
    # center and the forward ray never reaches it, so no constraint is needed.
    not_in_set = ~membership[:, None, :]  # (S, 1, N)
    violations = jnp.where(
        hits & not_in_set & (near_edge > 0),
        jnp.maximum(0.0, radii[:, :, None] - near_edge),
        0.0,
    )
    return jnp.sum(violations**2)


def _multi_term_enclosure(optim_vars, input_params):
    """Enclosure penalty summed over all sets.

    For each set s and its member circles, penalizes squared violations of
    radii[s, k] >= required radius at angle k for each member circle.
    The effective radius used is r + offset, where offset is per (set, circle).
    """
    circles = input_params["circles"]  # (N, 3)
    r = circles[None, :, 2] + input_params["offsets"]  # (S, N)
    return _enclosure_penalty(
        optim_vars["centers"], optim_vars["radii"],
        circles[:, :2], r,
        input_params["angles"], input_params["membership"],
    )


def _multi_term_exclusion(optim_vars, input_params):
    """Exclusion penalty: boundary must not overlap circles outside the set.

    For each set s and circle n not in set s, penalizes squared violations of
    radii[s, k] <= near_edge[s, k, n], where near_edge is the distance along
    ray k from center s to the near face of circle n.
    """
    circles = input_params["circles"]  # (N, 3)
    return _exclusion_penalty(
        optim_vars["centers"], optim_vars["radii"],
        circles[:, :2], circles[:, 2],
        input_params["angles"], input_params["membership"],
    )


def _multi_term_min_radius(optim_vars, input_params):
    """Penalty for any radius falling below the minimum allowed value."""
    radii = optim_vars["radii"]  # (S, K)
    return jnp.sum(jnp.maximum(0.0, _MIN_RADIUS - radii) ** 2)


def _multi_term_smoothness(optim_vars, input_params):
    """Penalty for sharp radius changes between adjacent angles.

    Penalizes the squared difference between each radius and its neighbour,
    summed over all sets and all angles (wrapping around).
    """
    radii = optim_vars["radii"]  # (S, K)
    return jnp.sum((radii - jnp.roll(radii, -1, axis=1)) ** 2)


def _multi_term_area(optim_vars, input_params):
    """Sum of star-polygon areas over all sets."""
    radii = optim_vars["radii"]  # (S, K)
    K = radii.shape[1]
    delta_theta = 2 * jnp.pi / K
    return 0.5 * jnp.sin(delta_theta) * jnp.sum(radii * jnp.roll(radii, -1, axis=1))


def _multi_term_perimeter(optim_vars, input_params):
    """Sum of star-polygon perimeters over all sets."""
    centers = optim_vars["centers"]  # (S, 2)
    radii = optim_vars["radii"]  # (S, K)
    angles = input_params["angles"]  # (K,)

    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)  # (K, 2)
    points = (
        centers[:, None, :] + radii[:, :, None] * directions[None, :, :]
    )  # (S, K, 2)
    points_next = jnp.roll(points, -1, axis=1)
    return jnp.sum(jnp.sqrt(jnp.sum((points_next - points) ** 2, axis=2) + 1e-12))


# ---------------------------------------------------------------------------
# Multi-set terms with movable circle positions
#
# optim_vars keys: "centers" (S, 2), "radii" (S, K), "circle_positions" (N, 2)
# input_params keys: "circle_radii" (N,), "angles" (K,),
#                    "membership" (S, N), "offsets" (S, N),
#                    "initial_circle_positions" (N, 2)
# ---------------------------------------------------------------------------


def _multi_term_enclosure_movable(optim_vars, input_params):
    """Enclosure penalty summed over all sets (circle positions are variables)."""
    r = input_params["circle_radii"][None, :] + input_params["offsets"]  # (S, N)
    return _enclosure_penalty(
        optim_vars["centers"], optim_vars["radii"],
        optim_vars["circle_positions"], r,
        input_params["angles"], input_params["membership"],
    )


def _multi_term_exclusion_movable(optim_vars, input_params):
    """Exclusion penalty (circle positions are variables)."""
    return _exclusion_penalty(
        optim_vars["centers"], optim_vars["radii"],
        optim_vars["circle_positions"], input_params["circle_radii"],
        input_params["angles"], input_params["membership"],
    )


def _multi_term_position_anchor(optim_vars, input_params):
    """Penalty for circle positions deviating from their initial positions."""
    circle_positions = optim_vars["circle_positions"]  # (N, 2)
    initial = input_params["initial_circle_positions"]  # (N, 2)
    return jnp.sum((circle_positions - initial) ** 2)


def _multi_term_circle_collision(optim_vars, input_params):
    """Penalty for overlapping circles.

    For each pair (i, j), penalizes squared overlap:
        max(0, r_i + r_j - dist(p_i, p_j))^2
    """
    positions = optim_vars["circle_positions"]  # (N, 2)
    radii = input_params["circle_radii"]  # (N,)

    diff = positions[:, None, :] - positions[None, :, :]  # (N, N, 2)
    dist = jnp.sqrt(jnp.sum(diff**2, axis=2) + 1e-12)  # (N, N)
    min_dist = radii[:, None] + radii[None, :]  # (N, N)
    overlap = jnp.maximum(0.0, min_dist - dist)  # (N, N)
    # Sum upper triangle only to count each pair once
    mask = jnp.triu(jnp.ones((radii.shape[0], radii.shape[0]), dtype=bool), k=1)
    return jnp.sum(jnp.where(mask, overlap**2, 0.0))


# ---------------------------------------------------------------------------
# Shared initialization helpers
# ---------------------------------------------------------------------------


def _build_membership(S, N, sets):
    """Build a boolean membership matrix of shape (S, N)."""
    membership = np.zeros((S, N), dtype=bool)
    for s, subset in enumerate(sets):
        for i in subset:
            membership[s, i] = True
    return membership


def _init_centers_and_radii(circles_array, sets, angles):
    """Initialize per-set centers and radii from member circles.

    Args:
        circles_array: (N, 3) array of [cx, cy, r].
        sets: list of S subsets (index collections into circles_array).
        angles: (K,) array of ray angles.

    Returns:
        Tuple of (initial_centers, initial_radii) with shapes (S, 2) and (S, K).
    """
    S = len(sets)
    K = len(angles)
    initial_centers = np.zeros((S, 2), dtype=np.float32)
    initial_radii = np.ones((S, K), dtype=np.float32)
    cos_a = np.cos(angles)[:, None]  # (K, 1)
    sin_a = np.sin(angles)[:, None]  # (K, 1)

    for s, subset in enumerate(sets):
        subset_circles = circles_array[list(subset)]
        center = np.mean(subset_circles[:, :2], axis=0).astype(np.float32)
        initial_centers[s] = center

        dx = subset_circles[:, 0] - center[0]
        dy = subset_circles[:, 1] - center[1]
        r = subset_circles[:, 2]
        tang = cos_a * dx[None, :] + sin_a * dy[None, :]  # (K, M)
        perp = -sin_a * dx[None, :] + cos_a * dy[None, :]  # (K, M)
        hits = perp**2 <= r[None, :] ** 2
        r_required = tang + np.sqrt(np.maximum(0.0, r[None, :] ** 2 - perp**2))
        r_required_masked = np.where(hits, r_required, 0.0)
        initial_radii[s] = np.maximum(r_required_masked.max(axis=1), 1.0).astype(
            np.float32
        )

    return initial_centers, initial_radii


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def optimize_multiple_radially_convex_sets(
    circles,
    sets,
    k_angles=32,
    weight_area=1.0,
    weight_perimeter=1.0,
    weight_exclusion=10.0,
    weight_smoothness=1.0,
    offsets=0.1,
    optim_kwargs=None,
):
    """Find star-shaped regions enclosing each set of circles without overlapping others.

    Each set gets its own star-shaped boundary that encloses its member circles
    and does not collide with circles belonging to other sets.

    Args:
        circles: array of shape (N, 3) with columns [cx, cy, r], or a sequence
            of (cx, cy, r) triples.
        sets: list of S subsets, each a collection of integer indices into circles.
            A circle may appear in multiple sets; circles absent from a set are
            treated as obstacles for that set's boundary.
        k_angles: number of angular samples defining each boundary polygon.
        weight_area: weight for the area objective (summed over sets).
        weight_perimeter: weight for the perimeter objective (summed over sets).
        weight_exclusion: weight for the exclusion penalty.
        weight_smoothness: weight for the smoothness penalty (squared radius
            differences between adjacent angles). Default 1.0.
        offsets: padding added to each circle's radius in the enclosure term,
            per (set, circle) pair. Scalar, shape (N,), or shape (S, N).
            Broadcast to (S, N). Default 0.1.
        optim_kwargs: keyword arguments forwarded to problem.optimize()
            (e.g. n_iters, learning_rate).

    Returns:
        Tuple of (results, history) where results is a list of S dicts, each with:
            "center": array of shape (2,), the center of the star-shaped region
            "radii": array of shape (K,), boundary radii at each angle
            "angles": array of shape (K,), uniformly spaced in [0, 2π)
        and history is the list of per-iteration loss dicts.
    """
    circles_array = np.asarray(circles, dtype=np.float32)
    if circles_array.ndim == 1:
        circles_array = circles_array[None, :]
    N = len(circles_array)
    S = len(sets)
    angles = np.linspace(0, 2 * np.pi, k_angles, endpoint=False).astype(np.float32)

    membership = _build_membership(S, N, sets)
    initial_centers, initial_radii = _init_centers_and_radii(circles_array, sets, angles)
    offsets_array = np.broadcast_to(
        np.asarray(offsets, dtype=np.float32), (S, N)
    ).copy()

    input_parameters = {
        "circles": circles_array,
        "angles": angles,
        "membership": membership,
        "offsets": offsets_array,
    }

    def initialize(_):
        return {
            "centers": initial_centers,
            "radii": initial_radii,
        }

    terms = [
        ObjectiveTerm("enclosure", _multi_term_enclosure, 10.0),
        ObjectiveTerm("exclusion", _multi_term_exclusion, weight_exclusion),
        ObjectiveTerm("min_radius", _multi_term_min_radius, 10.0),
        ObjectiveTerm("smoothness", _multi_term_smoothness, weight_smoothness),
        ObjectiveTerm("area", _multi_term_area, weight_area),
        ObjectiveTerm("perimeter", _multi_term_perimeter, weight_perimeter),
    ]

    problem = OptimizationProblemTemplate(
        terms=terms, initialize=initialize
    ).instantiate(input_parameters)
    optim_vars, history = problem.optimize(**(optim_kwargs or {}))

    return [
        {
            "center": np.array(optim_vars["centers"][s]),
            "radii": np.array(optim_vars["radii"][s]),
            "angles": angles,
        }
        for s in range(S)
    ], history


def optimize_multiple_radially_convex_sets_with_movable_circles(
    circles,
    sets,
    k_angles=32,
    weight_area=1.0,
    weight_perimeter=1.0,
    weight_exclusion=10.0,
    weight_smoothness=1.0,
    weight_position_anchor=1.0,
    weight_circle_collision=10.0,
    offsets=0.1,
    optim_kwargs=None,
):
    """Like optimize_multiple_radially_convex_sets, but circle positions are also optimized.

    Circle positions (cx, cy) become optimization variables. Their radii remain
    fixed. A position anchor term penalizes deviation from the initial positions.

    Args:
        circles: array of shape (N, 3) with columns [cx, cy, r], or a sequence
            of (cx, cy, r) triples.
        sets: list of S subsets, each a collection of integer indices into circles.
        k_angles: number of angular samples defining each boundary polygon.
        weight_area: weight for the area objective.
        weight_perimeter: weight for the perimeter objective.
        weight_exclusion: weight for the exclusion penalty.
        weight_smoothness: weight for the smoothness penalty.
        weight_position_anchor: weight for penalizing circle positions deviating
            from their initial positions. Higher values keep circles closer to
            their starting positions.
        weight_circle_collision: weight for penalizing overlapping circles.
        offsets: padding added to each circle's radius in the enclosure term,
            per (set, circle) pair. Scalar, shape (N,), or shape (S, N).
        optim_kwargs: keyword arguments forwarded to problem.optimize()
            (e.g. n_iters, learning_rate).

    Returns:
        Tuple of (results, circles_out, history) where results is a list of S
        dicts each with "center", "radii", "angles"; circles_out is an array of
        shape (N, 3) with optimized [cx, cy, r]; and history is the list of
        per-iteration loss dicts.
    """
    circles_array = np.asarray(circles, dtype=np.float32)
    if circles_array.ndim == 1:
        circles_array = circles_array[None, :]
    N = len(circles_array)
    S = len(sets)
    angles = np.linspace(0, 2 * np.pi, k_angles, endpoint=False).astype(np.float32)

    initial_circle_positions = circles_array[:, :2].copy()  # (N, 2)
    circle_radii = circles_array[:, 2].copy()  # (N,)

    membership = _build_membership(S, N, sets)
    initial_centers, initial_radii = _init_centers_and_radii(circles_array, sets, angles)
    offsets_array = np.broadcast_to(
        np.asarray(offsets, dtype=np.float32), (S, N)
    ).copy()

    input_parameters = {
        "circle_radii": circle_radii,
        "initial_circle_positions": initial_circle_positions,
        "angles": angles,
        "membership": membership,
        "offsets": offsets_array,
    }

    def initialize(_):
        return {
            "centers": initial_centers,
            "radii": initial_radii,
            "circle_positions": initial_circle_positions.copy(),
        }

    terms = [
        ObjectiveTerm("enclosure", _multi_term_enclosure_movable, 10.0),
        ObjectiveTerm("exclusion", _multi_term_exclusion_movable, weight_exclusion),
        ObjectiveTerm("min_radius", _multi_term_min_radius, 10.0),
        ObjectiveTerm("smoothness", _multi_term_smoothness, weight_smoothness),
        ObjectiveTerm("area", _multi_term_area, weight_area),
        ObjectiveTerm("perimeter", _multi_term_perimeter, weight_perimeter),
        ObjectiveTerm("position_anchor", _multi_term_position_anchor, weight_position_anchor),
        ObjectiveTerm("circle_collision", _multi_term_circle_collision, weight_circle_collision),
    ]

    problem = OptimizationProblemTemplate(
        terms=terms, initialize=initialize
    ).instantiate(input_parameters)
    optim_vars, history = problem.optimize(**(optim_kwargs or {}))

    circles_out = np.concatenate(
        [np.array(optim_vars["circle_positions"]), circle_radii[:, None]], axis=1
    )

    return [
        {
            "center": np.array(optim_vars["centers"][s]),
            "radii": np.array(optim_vars["radii"][s]),
            "angles": angles,
        }
        for s in range(S)
    ], circles_out, history
