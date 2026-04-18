"""Star-shaped boundary optimization for sets of circles.

Finds star-shaped (radially convex) regions enclosing each set of circles
while minimizing area/perimeter and avoiding overlap with other sets.

Each boundary is parametrized as a center + K radii at uniformly-spaced angles.
General star-domain loss terms and helpers live in
:mod:`vizopt.components.stars`.
"""

import jax.numpy as jnp
import numpy as np

from ..base import Callback, ObjectiveTerm, OptimizationProblemTemplate, OptimConfig
from ..components.stars import Discrete, StarRepresentation
from ..components.stars import (
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
)


def optimize_multiple_radially_convex_sets(
    circles,
    sets,
    k_angles=32,
    weight_area=1.0,
    weight_perimeter=1.0,
    weight_exclusion=10.0,
    weight_smoothness=1.0,
    offsets=0.1,
    representation: StarRepresentation | None = None,
    term_schedules=None,
    optim_config: OptimConfig | None = None,
    callback: Callback | None = None,
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
        representation: star domain parametrization. One of :class:`Discrete`
            (default), :class:`Fourier`, or :class:`BSpline`. The ``k_angles``
            attribute on the representation is used when provided; the
            ``k_angles`` function argument is a fallback for ``Discrete``.
        term_schedules: optional dict mapping term name to a JAX-compatible
            callable ``(step: Array) -> Array`` that scales the term's weight
            over iterations. Valid keys: "enclosure", "exclusion", "min_radius",
            "smoothness", "area", "perimeter". The effective weight at step t is
            ``base_weight * schedule(t)``. Schedules must use JAX ops so they
            can be traced through without recompilation.
        optim_config: Optimizer settings (iterations, learning rate, seeds,
            restarts). Uses :class:`OptimConfig` defaults when ``None``.
        callback: Optional callback called after each iteration with
            ``(iteration, loss, optim_vars, grads)``. Pass a
            :class:`~vizopt.animation.SnapshotCallback` to capture frames for
            :func:`~vizopt.animation.snapshots_to_animated_svg`.

    Returns:
        Tuple of (results, history, problem) where results is a list of S dicts,
        each with:
            "center": array of shape (2,), the center of the star-shaped region
            "radii": array of shape (K,), boundary radii at each angle
            "angles": array of shape (K,), uniformly spaced in [0, 2π)
        history is the list of per-iteration loss dicts, and problem is the
        :class:`~vizopt.base.OptimizationProblem` instance (needed for
        :func:`~vizopt.animation.snapshots_to_animated_svg`).
    """
    if representation is None:
        representation = Discrete(k_angles=k_angles)

    circles_array = np.asarray(circles, dtype=np.float32)
    if circles_array.ndim == 1:
        circles_array = circles_array[None, :]
    N = len(circles_array)
    S = len(sets)
    angles = np.linspace(0, 2 * np.pi, representation.k_angles, endpoint=False).astype(np.float32)
    angles_jnp = jnp.array(angles)

    membership = _build_membership(S, N, sets)
    initial_centers, initial_radii = _init_centers_and_radii(
        circles_array, sets, angles
    )
    offsets_array = np.broadcast_to(
        np.asarray(offsets, dtype=np.float32), (S, N)
    ).copy()

    input_parameters = {
        "circles": circles_array,
        "angles": angles,
        "membership": membership,
        "offsets": offsets_array,
    }

    init_vars = representation.initialize_vars(S, initial_radii, initial_centers)

    def initialize(_, seed):
        return {k: v.copy() for k, v in init_vars.items()}

    def wrap(fn):
        return representation.wrap(fn, angles_jnp)

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
        terms=terms, initialize=initialize, svg_configuration=representation.make_svg_configuration()
    ).instantiate(input_parameters)
    optim_vars, history = problem.optimize(optim_config, callback=callback)

    radii_arr = np.array(representation.to_radii(optim_vars, angles_jnp))
    return (
        [
            {
                "center": np.array(optim_vars["centers"][s]),
                "radii": radii_arr[s],
                "angles": angles,
                **representation.extra_results(s, optim_vars),
            }
            for s in range(S)
        ],
        history,
        problem,
    )


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
    weight_bounding_box=0.0,
    weight_set_attraction=0.0,
    circle_collision_alpha=0.0,
    offsets=0.1,
    representation: StarRepresentation | None = None,
    term_schedules=None,
    optim_config: OptimConfig | None = None,
    callback: Callback | None = None,
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
        weight_bounding_box: weight for minimizing total width + total height of
            all set boundaries. Default 0.0 (disabled).
        weight_set_attraction: weight for pulling each circle toward the center
            of every set it belongs to (and pulling set centers toward their
            member circles). Default 0.0 (disabled).
        circle_collision_alpha: coefficient for the linear term in the circle
            collision penalty: ``overlap² + alpha * overlap``. The linear term
            gives a constant non-zero gradient for any overlap, preventing tiny
            violations from persisting. Default 0.0 (pure quadratic).
        offsets: padding added to each circle's radius in the enclosure term,
            per (set, circle) pair. Scalar, shape (N,), or shape (S, N).
        representation: star domain parametrization. One of :class:`Discrete`
            (default), :class:`Fourier`, or :class:`BSpline`.
        term_schedules: optional dict mapping term name to a JAX-compatible
            callable ``(step: Array) -> Array`` that scales the term's weight
            over iterations. Valid keys: "enclosure", "exclusion", "min_radius",
            "smoothness", "area", "perimeter", "position_anchor",
            "circle_collision", "bounding_box", "set_attraction". The effective
            weight at step t
            is ``base_weight * schedule(t)``. Schedules must use JAX ops so
            they can be traced through without recompilation.
        optim_config: Optimizer settings (iterations, learning rate, seeds,
            restarts). Uses :class:`OptimConfig` defaults when ``None``.
        callback: Optional callback called after each iteration with
            ``(iteration, loss, optim_vars, grads)``. Pass a
            :class:`~vizopt.animation.SnapshotCallback` to capture frames for
            :func:`~vizopt.animation.snapshots_to_animated_svg`.

    Returns:
        Tuple of (results, circles_out, history, problem) where results is a
        list of S dicts each with "center", "radii", "angles"; circles_out is
        an array of shape (N, 3) with optimized [cx, cy, r]; history is the
        list of per-iteration loss dicts; and problem is the
        :class:`~vizopt.base.OptimizationProblem` instance (needed for
        :func:`~vizopt.animation.snapshots_to_animated_svg`).
    """
    if representation is None:
        representation = Discrete(k_angles=k_angles)

    circles_array = np.asarray(circles, dtype=np.float32)
    if circles_array.ndim == 1:
        circles_array = circles_array[None, :]
    N = len(circles_array)
    S = len(sets)
    angles = np.linspace(0, 2 * np.pi, representation.k_angles, endpoint=False).astype(np.float32)
    angles_jnp = jnp.array(angles)

    initial_circle_positions = circles_array[:, :2].copy()  # (N, 2)
    circle_radii = circles_array[:, 2].copy()  # (N,)

    membership = _build_membership(S, N, sets)
    initial_centers, initial_radii = _init_centers_and_radii(
        circles_array, sets, angles
    )
    offsets_array = np.broadcast_to(
        np.asarray(offsets, dtype=np.float32), (S, N)
    ).copy()

    input_parameters = {
        "circle_radii": circle_radii,
        "initial_circle_positions": initial_circle_positions,
        "angles": angles,
        "membership": membership,
        "offsets": offsets_array,
        "circle_collision_alpha": np.float32(circle_collision_alpha),
    }

    init_vars = representation.initialize_vars(S, initial_radii, initial_centers)

    def initialize(_, seed):
        return {**{k: v.copy() for k, v in init_vars.items()}, "circle_positions": initial_circle_positions.copy()}

    def wrap(fn):
        return representation.wrap(fn, angles_jnp)

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
        terms=terms, initialize=initialize, svg_configuration=representation.make_svg_configuration()
    ).instantiate(input_parameters)
    optim_vars, history = problem.optimize(optim_config, callback=callback)

    circles_out = np.concatenate(
        [np.array(optim_vars["circle_positions"]), circle_radii[:, None]], axis=1
    )

    radii_arr = np.array(representation.to_radii(optim_vars, angles_jnp))
    return (
        [
            {
                "center": np.array(optim_vars["centers"][s]),
                "radii": radii_arr[s],
                "angles": angles,
                **representation.extra_results(s, optim_vars),
            }
            for s in range(S)
        ],
        circles_out,
        history,
        problem,
    )
