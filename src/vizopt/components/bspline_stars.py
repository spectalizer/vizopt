"""B-spline boundary representation for star-shaped regions.

Uniform periodic cubic B-splines give C² smooth boundaries at O(1) per-pixel
evaluation cost (no trig). This module provides:

- Core math: ``_interp_bspline``, ``bspline_to_radii``, ``_wrap_bspline_term``
- Raster soft-membership: ``soft_rasterize_star_bspline``,
  ``raster_collision_loss_bspline``
- SVG helpers: ``_svg_configuration_bspline_star_only``,
  ``_svg_configuration_bspline_fixed``, ``_svg_configuration_bspline_movable``
- Optimizer: ``optimize_multiple_radially_convex_sets_bspline``
"""

import jax
import jax.numpy as jnp
import numpy as np

from vizopt.base import Callback, ObjectiveTerm, OptimConfig, OptimizationProblemTemplate
from vizopt.components.stars import (
    _build_membership,
    _init_centers_and_radii,
    _multi_term_area,
    _multi_term_circle_collision,
    _multi_term_enclosure,
    _multi_term_enclosure_movable,
    _multi_term_exclusion,
    _multi_term_exclusion_movable,
    _multi_term_min_radius,
    _multi_term_perimeter,
    _multi_term_position_anchor,
    _multi_term_set_attraction,
    _multi_term_smoothness,
    _multi_term_total_bounding_box,
    _svg_configuration_fixed,
    _svg_configuration_movable,
)


# ---------------------------------------------------------------------------
# Core B-spline math
# ---------------------------------------------------------------------------


def _interp_bspline(ctrl_s, alpha_s):
    """Evaluate a uniform periodic cubic B-spline at angles alpha_s.

    Args:
        ctrl_s: (P,) control points.
        alpha_s: (...) evaluation angles in radians.

    Returns:
        (...) interpolated radius values.
    """
    P = ctrl_s.shape[0]
    dt = 2.0 * jnp.pi / P
    raw = (alpha_s % (2.0 * jnp.pi)) / dt  # fractional index in [0, P)
    i = jnp.floor(raw).astype(jnp.int32) % P
    t = raw - jnp.floor(raw)
    t2 = t * t
    t3 = t2 * t
    b0 = (1.0 - t) ** 3 / 6.0
    b1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0
    b2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0
    b3 = t3 / 6.0
    return (
        b0 * ctrl_s[(i - 1) % P]
        + b1 * ctrl_s[i]
        + b2 * ctrl_s[(i + 1) % P]
        + b3 * ctrl_s[(i + 2) % P]
    )


def bspline_to_radii(ctrl_pts, angles):
    """Evaluate uniform periodic cubic B-splines at given angles.

    Args:
        ctrl_pts: (n_sets, P) B-spline control points.
        angles: (K,) evaluation angles in radians.

    Returns:
        (n_sets, K) radius values.
    """
    return jax.vmap(lambda c: _interp_bspline(c, angles))(ctrl_pts)


def _wrap_bspline_term(fn, angles):
    """Adapt a loss term that reads optim_vars["radii"] to accept bspline_ctrl.

    Args:
        fn: loss function with signature ``(optim_vars, input_params) -> scalar``
            that reads ``optim_vars["radii"]``.
        angles: (K,) evaluation angles used to convert ctrl pts → radii.

    Returns:
        Wrapped function with the same signature that reads
        ``optim_vars["bspline_ctrl"]`` instead.
    """
    def wrapped(optim_vars, input_params):
        radii = bspline_to_radii(optim_vars["bspline_ctrl"], angles)
        return fn({**optim_vars, "radii": radii}, input_params)
    return wrapped


# ---------------------------------------------------------------------------
# Raster soft-membership (for raster-collision exclusion)
# ---------------------------------------------------------------------------


def soft_rasterize_star_bspline(
    centers,
    ctrl_pts,
    grid_xy,
    temperature=0.05,
    offset=0.0,
):
    """Soft-rasterize n_sets star domains using a B-spline boundary.

    Like ``soft_rasterize_star`` but r(θ) is a uniform periodic cubic B-spline,
    giving C² smooth boundaries at O(1) per-pixel cost (no trig evaluations).

    Args:
        centers: (n_sets, 2) domain centers.
        ctrl_pts: (n_sets, P) B-spline control points.
        grid_xy: (H, W, 2) pixel centre coordinates.
        temperature: sigmoid sharpness; smaller → harder boundary.
        offset: outward boundary shift (positive values inflate the domain).

    Returns:
        masks: (n_sets, H, W) soft membership values in (0, 1).
    """
    diff = grid_xy[None, :, :, :] - centers[:, None, None, :]
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-12)

    rho2 = jnp.sum(diff**2, axis=-1)
    dx_safe = jnp.where(rho2 > 0, diff[..., 0], 1.0)
    dy_safe = jnp.where(rho2 > 0, diff[..., 1], 0.0)
    alpha = jnp.arctan2(dy_safe, dx_safe)  # (n_sets, H, W)

    r_interp = jax.vmap(_interp_bspline)(ctrl_pts, alpha)  # (n_sets, H, W)
    return jax.nn.sigmoid((r_interp + offset - dist) / temperature)


def raster_collision_loss_bspline(optim_vars, input_params):
    """Raster-based pairwise collision loss using a B-spline boundary.

    Same semantics as ``raster_collision_loss`` but reads ``bspline_ctrl`` from
    *optim_vars* instead of ``radii``.

    optim_vars keys:
        "centers":      (n_sets, 2)
        "bspline_ctrl": (n_sets, P)
    input_params keys: same as ``raster_collision_loss``.
    """
    centers = optim_vars["centers"]
    ctrl_pts = optim_vars["bspline_ctrl"]
    grid_xy = input_params["grid_xy"]
    pixel_area = input_params["pixel_area"]
    mask = input_params["exclusion_mask"]
    temperature = input_params.get("temperature", 0.05)
    exclusion_offset = input_params.get("exclusion_offset", 0.0)

    masks = soft_rasterize_star_bspline(centers, ctrl_pts, grid_xy, temperature, exclusion_offset)
    HW = grid_xy.shape[0] * grid_xy.shape[1]
    masks_flat = masks.reshape(masks.shape[0], HW)
    overlap = pixel_area * (masks_flat @ masks_flat.T)
    return jnp.sum(jnp.where(mask, overlap**2, 0.0))


# ---------------------------------------------------------------------------
# SVG configuration helpers
# ---------------------------------------------------------------------------


def _svg_configuration_bspline_star_only(snapshots, input_params, size):
    """SVG configuration for B-spline star domains (no underlying circles)."""
    from vizopt.templates.star_vs_star import _svg_configuration_star_only

    angles_jnp = jnp.array(input_params["angles"])
    converted = [
        (i, {**v, "radii": np.array(bspline_to_radii(v["bspline_ctrl"], angles_jnp))})
        for i, v in snapshots
    ]
    return _svg_configuration_star_only(converted, input_params, size)


def _svg_configuration_bspline_fixed(snapshots, input_params, size):
    """SVG configuration for B-spline star boundaries with fixed circles."""
    angles_jnp = jnp.array(input_params["angles"])
    converted = [
        (i, {**v, "radii": np.array(bspline_to_radii(v["bspline_ctrl"], angles_jnp))})
        for i, v in snapshots
    ]
    return _svg_configuration_fixed(converted, input_params, size)


def _svg_configuration_bspline_movable(snapshots, input_params, size):
    """SVG configuration for B-spline star boundaries with movable circles."""
    angles_jnp = jnp.array(input_params["angles"])
    converted = [
        (i, {**v, "radii": np.array(bspline_to_radii(v["bspline_ctrl"], angles_jnp))})
        for i, v in snapshots
    ]
    return _svg_configuration_movable(converted, input_params, size)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def optimize_multiple_radially_convex_sets_bspline(
    circles,
    sets,
    k_angles=32,
    n_ctrl_pts=16,
    weight_area=1.0,
    weight_perimeter=1.0,
    weight_exclusion=10.0,
    weight_smoothness=1.0,
    offsets=0.1,
    term_schedules=None,
    optim_config: OptimConfig | None = None,
    callback: Callback | None = None,
):
    """Like optimize_multiple_radially_convex_sets but with B-spline boundaries.

    Each star-shaped boundary is parameterised as a uniform periodic cubic
    B-spline with *n_ctrl_pts* control points, giving C² smooth boundaries.

    Args:
        circles: array of shape (N, 3) with columns [cx, cy, r], or a sequence
            of (cx, cy, r) triples.
        sets: list of S subsets, each a collection of integer indices into circles.
        k_angles: number of angular samples used to evaluate the B-spline for
            analytical loss terms (enclosure, exclusion, area, perimeter).
        n_ctrl_pts: number of B-spline control points per boundary.
        weight_area: weight for the area objective (summed over sets).
        weight_perimeter: weight for the perimeter objective (summed over sets).
        weight_exclusion: weight for the exclusion penalty.
        weight_smoothness: weight for the smoothness penalty.
        offsets: padding added to each circle's radius in the enclosure term,
            per (set, circle) pair. Scalar, shape (N,), or shape (S, N).
        term_schedules: optional dict mapping term name to a JAX-compatible
            callable ``(step: Array) -> Array`` that scales the term's weight
            over iterations. Valid keys: "enclosure", "exclusion", "min_radius",
            "smoothness", "area", "perimeter".
        optim_config: Optimizer settings. Uses :class:`OptimConfig` defaults
            when ``None``.
        callback: Optional iteration callback.

    Returns:
        Tuple of (results, history, problem) where results is a list of S dicts,
        each with:
            "center": array of shape (2,), the center of the star-shaped region
            "radii": array of shape (K,), boundary radii evaluated at k_angles
            "angles": array of shape (K,), uniformly spaced in [0, 2π)
            "bspline_ctrl": array of shape (n_ctrl_pts,), B-spline control points
        history is the list of per-iteration loss dicts, and problem is the
        :class:`~vizopt.base.OptimizationProblem` instance.
    """
    circles_array = np.asarray(circles, dtype=np.float32)
    if circles_array.ndim == 1:
        circles_array = circles_array[None, :]
    N = len(circles_array)
    S = len(sets)
    angles = np.linspace(0, 2 * np.pi, k_angles, endpoint=False).astype(np.float32)
    angles_jnp = jnp.array(angles)

    membership = _build_membership(S, N, sets)
    initial_centers, initial_radii = _init_centers_and_radii(circles_array, sets, angles)
    offsets_array = np.broadcast_to(
        np.asarray(offsets, dtype=np.float32), (S, N)
    ).copy()

    initial_ctrl_pts = np.zeros((S, n_ctrl_pts), dtype=np.float32)
    for s in range(S):
        initial_ctrl_pts[s] = initial_radii[s].mean()

    input_parameters = {
        "circles": circles_array,
        "angles": angles,
        "membership": membership,
        "offsets": offsets_array,
    }

    def initialize(_, seed):
        return {
            "centers": initial_centers,
            "bspline_ctrl": initial_ctrl_pts,
        }

    wrap = lambda fn: _wrap_bspline_term(fn, angles_jnp)
    schedules = term_schedules or {}
    terms = [
        ObjectiveTerm("enclosure", wrap(_multi_term_enclosure), 10.0, schedules.get("enclosure")),
        ObjectiveTerm("exclusion", wrap(_multi_term_exclusion), weight_exclusion, schedules.get("exclusion")),
        ObjectiveTerm("min_radius", wrap(_multi_term_min_radius), 10.0, schedules.get("min_radius")),
        ObjectiveTerm("smoothness", wrap(_multi_term_smoothness), weight_smoothness, schedules.get("smoothness")),
        ObjectiveTerm("area", wrap(_multi_term_area), weight_area, schedules.get("area")),
        ObjectiveTerm("perimeter", wrap(_multi_term_perimeter), weight_perimeter, schedules.get("perimeter")),
    ]

    problem = OptimizationProblemTemplate(
        terms=terms,
        initialize=initialize,
        svg_configuration=_svg_configuration_bspline_fixed,
    ).instantiate(input_parameters)
    optim_vars, history = problem.optimize(optim_config, callback=callback)

    radii_arr = np.array(bspline_to_radii(optim_vars["bspline_ctrl"], angles_jnp))
    results = [
        {
            "center": np.array(optim_vars["centers"][s]),
            "radii": radii_arr[s],
            "angles": angles,
            "bspline_ctrl": np.array(optim_vars["bspline_ctrl"][s]),
        }
        for s in range(S)
    ]
    return results, history, problem


def optimize_multiple_radially_convex_sets_bspline_with_movable_circles(
    circles,
    sets,
    k_angles=32,
    n_ctrl_pts=16,
    weight_area=1.0,
    weight_perimeter=1.0,
    weight_exclusion=10.0,
    weight_smoothness=1.0,
    weight_position_anchor=1.0,
    weight_circle_collision=10.0,
    weight_bounding_box=0.0,
    weight_set_attraction=0.0,
    circle_collision_alpha=0.0,
    offsets=0.1,
    term_schedules=None,
    optim_config: OptimConfig | None = None,
    callback: Callback | None = None,
):
    """Like optimize_multiple_radially_convex_sets_bspline, but circle positions are also optimized.

    Args:
        circles: array of shape (N, 3) with columns [cx, cy, r].
        sets: list of S subsets, each a collection of integer indices into circles.
        k_angles: number of angular samples for evaluating B-spline boundaries.
        n_ctrl_pts: number of B-spline control points per boundary.
        weight_area: weight for the area objective.
        weight_perimeter: weight for the perimeter objective.
        weight_exclusion: weight for the exclusion penalty.
        weight_smoothness: weight for the smoothness penalty.
        weight_position_anchor: weight for penalizing circle positions deviating
            from their initial positions.
        weight_circle_collision: weight for penalizing overlapping circles.
        weight_bounding_box: weight for minimizing total width + total height.
            Default 0.0 (disabled).
        weight_set_attraction: weight for pulling each circle toward the center
            of every set it belongs to. Default 0.0 (disabled).
        circle_collision_alpha: coefficient for the linear term in the circle
            collision penalty. Default 0.0 (pure quadratic).
        offsets: padding added to each circle's radius in the enclosure term.
        term_schedules: optional dict mapping term name to a JAX-compatible
            callable. Valid keys: "enclosure", "exclusion", "min_radius",
            "smoothness", "area", "perimeter", "position_anchor",
            "circle_collision", "bounding_box", "set_attraction".
        optim_config: Optimizer settings.
        callback: Optional iteration callback.

    Returns:
        Tuple of (results, circles_out, history, problem) where results is a list
        of S dicts each with "center", "radii", "angles", "bspline_ctrl";
        circles_out is an array of shape (N, 3) with optimized [cx, cy, r];
        history is the list of per-iteration loss dicts; and problem is the
        :class:`~vizopt.base.OptimizationProblem` instance.
    """
    circles_array = np.asarray(circles, dtype=np.float32)
    if circles_array.ndim == 1:
        circles_array = circles_array[None, :]
    N = len(circles_array)
    S = len(sets)
    angles = np.linspace(0, 2 * np.pi, k_angles, endpoint=False).astype(np.float32)
    angles_jnp = jnp.array(angles)

    initial_circle_positions = circles_array[:, :2].copy()
    circle_radii = circles_array[:, 2].copy()

    membership = _build_membership(S, N, sets)
    initial_centers, initial_radii = _init_centers_and_radii(circles_array, sets, angles)
    offsets_array = np.broadcast_to(
        np.asarray(offsets, dtype=np.float32), (S, N)
    ).copy()

    initial_ctrl_pts = np.zeros((S, n_ctrl_pts), dtype=np.float32)
    for s in range(S):
        initial_ctrl_pts[s] = initial_radii[s].mean()

    input_parameters = {
        "circle_radii": circle_radii,
        "initial_circle_positions": initial_circle_positions,
        "angles": angles,
        "membership": membership,
        "offsets": offsets_array,
        "circle_collision_alpha": np.float32(circle_collision_alpha),
    }

    def initialize(_, seed):
        return {
            "centers": initial_centers,
            "bspline_ctrl": initial_ctrl_pts,
            "circle_positions": initial_circle_positions.copy(),
        }

    wrap = lambda fn: _wrap_bspline_term(fn, angles_jnp)
    schedules = term_schedules or {}
    terms = [
        ObjectiveTerm("enclosure", wrap(_multi_term_enclosure_movable), 10.0, schedules.get("enclosure")),
        ObjectiveTerm("exclusion", wrap(_multi_term_exclusion_movable), weight_exclusion, schedules.get("exclusion")),
        ObjectiveTerm("min_radius", wrap(_multi_term_min_radius), 10.0, schedules.get("min_radius")),
        ObjectiveTerm("smoothness", wrap(_multi_term_smoothness), weight_smoothness, schedules.get("smoothness")),
        ObjectiveTerm("area", wrap(_multi_term_area), weight_area, schedules.get("area")),
        ObjectiveTerm("perimeter", wrap(_multi_term_perimeter), weight_perimeter, schedules.get("perimeter")),
        ObjectiveTerm("position_anchor", _multi_term_position_anchor, weight_position_anchor, schedules.get("position_anchor")),
        ObjectiveTerm("circle_collision", _multi_term_circle_collision, weight_circle_collision, schedules.get("circle_collision")),
        ObjectiveTerm("bounding_box", wrap(_multi_term_total_bounding_box), weight_bounding_box, schedules.get("bounding_box")),
        ObjectiveTerm("set_attraction", _multi_term_set_attraction, weight_set_attraction, schedules.get("set_attraction")),
    ]

    problem = OptimizationProblemTemplate(
        terms=terms,
        initialize=initialize,
        svg_configuration=_svg_configuration_bspline_movable,
    ).instantiate(input_parameters)
    optim_vars, history = problem.optimize(optim_config, callback=callback)

    circles_out = np.concatenate(
        [np.array(optim_vars["circle_positions"]), circle_radii[:, None]], axis=1
    )
    radii_arr = np.array(bspline_to_radii(optim_vars["bspline_ctrl"], angles_jnp))
    results = [
        {
            "center": np.array(optim_vars["centers"][s]),
            "radii": radii_arr[s],
            "angles": angles,
            "bspline_ctrl": np.array(optim_vars["bspline_ctrl"][s]),
        }
        for s in range(S)
    ]
    return results, circles_out, history, problem
