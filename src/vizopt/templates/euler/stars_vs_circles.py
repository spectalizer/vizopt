"""Star-shaped boundary optimization for sets of circles.

Finds star-shaped (radially convex) regions enclosing each set of circles
while minimizing area/perimeter and avoiding overlap with other sets.

Each boundary is parametrized as a center + K radii at uniformly-spaced angles.
General star-domain loss terms and helpers live in
:mod:`vizopt.components.stars`.
"""

import jax.numpy as jnp
import networkx as nx
import numpy as np

from ...base import Callback, ObjectiveTerm, OptimConfig, OptimizationProblemTemplate
from ...components.stars import (
    Discrete,
    StarRepresentation,
    _build_membership,
    _init_centers_and_radii,
    _multi_term_area,
    _multi_term_circle_collision,
    _multi_term_convexity,
    _multi_term_enclosure_movable,
    _multi_term_exclusion_movable,
    _multi_term_label_element_exclusion,
    _multi_term_label_enclosure,
    _multi_term_label_label_collision,
    _multi_term_label_top_attraction,
    _multi_term_min_radius,
    _multi_term_perimeter,
    _multi_term_position_anchor,
    _multi_term_set_attraction,
    _multi_term_smoothness,
    _multi_term_total_bounding_box,
    _svg_configuration_movable,
)
from ...schedules import TermSchedules
from .graph_utils import offsets_from_graph


def _leaf_circles_from_graph(inclusion_graph: nx.DiGraph):
    names = [n for n in inclusion_graph.nodes if inclusion_graph.out_degree(n) == 0]
    circles = np.array(
        [
            [*inclusion_graph.nodes[n]["center"], inclusion_graph.nodes[n]["r"]]
            for n in names
        ],
        dtype=np.float32,
    )
    name_to_idx = {name: i for i, name in enumerate(names)}
    return names, circles, name_to_idx


def _sets_from_graph(inclusion_graph: nx.DiGraph, leaf_names, name_to_idx):
    leaf_set = set(leaf_names)
    set_names = [
        n
        for n in nx.topological_sort(inclusion_graph)
        if inclusion_graph.out_degree(n) > 0
    ]
    sets_idx = [
        sorted(
            name_to_idx[n]
            for n in nx.descendants(inclusion_graph, sname)
            if n in leaf_set
        )
        for sname in set_names
    ]
    return set_names, sets_idx


def optimize_radially_convex_sets_and_circles(
    circles,
    sets,
    weight_area=1.0,
    weight_perimeter=1.0,
    weight_enclosure=20.0,
    weight_exclusion=10.0,
    weight_smoothness=1.0,
    weight_convexity=0.0,
    weight_position_anchor=1.0,
    weight_circle_collision=10.0,
    weight_bounding_box=0.0,
    weight_set_attraction=0.0,
    circle_collision_alpha=0.0,
    convexity_alpha=1.0,
    offsets: float | np.ndarray = 0.1,
    label_rect_size: tuple[float, float] | None = None,
    weight_label_enclosure: float = 10.0,
    weight_label_element_exclusion: float = 10.0,
    weight_label_collision: float = 10.0,
    weight_label_top: float = 1.0,
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
        weight_area: weight for the area objective.
        weight_perimeter: weight for the perimeter objective.
        weight_enclosure: weight for the enclosure penalty (each circle must lie
            inside its set boundary). Default 10.0.
        weight_exclusion: weight for the exclusion penalty.
        weight_smoothness: weight for the smoothness penalty.
        weight_convexity: weight for the convexity penalty, which penalizes
            concave turns in the star polygon boundary. Default 0.0 (disabled).
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
        convexity_alpha: coefficient for the linear term in the convexity
            penalty: ``violation² + alpha * violation``, where violation is
            ``max(0, -cross)``.  The linear term gives a non-zero gradient for
            any concavity, preventing mild violations from stalling when the
            pure-quadratic gradient is too small to overcome competing terms.
            Default 0.5.
        offsets: padding added to each circle's radius in the enclosure term,
            per (set, circle) pair. Scalar, shape (N,), or shape (S, N).
        label_rect_size: ``(hw, hh)`` half-extents of the label rectangle to
            reserve at the top of each set boundary.  When ``None`` (default)
            no label rectangle is used.  When set, each star boundary is
            forced to enclose a floating label rect whose position
            (``label_positions``, one per set) is an optimization variable.
            The optimizer pushes each label rect above the leaf circles while
            keeping it inside the star boundary.
        weight_label_enclosure: weight for the label enclosure term (star must
            cover the label rect).  Default 10.0.
        weight_label_element_exclusion: weight for keeping label rects away
            from leaf circles.  Default 10.0.
        weight_label_collision: weight for keeping label rects from overlapping
            each other.  Default 10.0.
        weight_label_top: weight for the upward-attraction term that pulls
            label rects toward the top of each set.  Default 1.0.
        representation: star domain parametrization. One of :class:`Discrete`
            (default), :class:`Fourier`, or :class:`BSpline`.
        term_schedules: optional dict mapping term name to a JAX-compatible
            callable ``(step: Array) -> Array`` that scales the term's weight
            over iterations. Valid keys: "enclosure", "exclusion", "min_radius",
            "smoothness", "convexity", "area", "perimeter", "position_anchor",
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
        list of S dicts each with "center", "radii", "angles", and
        (when ``label_rect_size`` is set) "label_center" giving the optimized
        ``[cx, cy]`` of the label rect; circles_out is an array of shape
        (N, 3) with optimized [cx, cy, r]; history is the list of
        per-iteration loss dicts; and problem is the
        :class:`~vizopt.base.OptimizationProblem` instance (needed for
        :func:`~vizopt.animation.snapshots_to_animated_svg`).
    """
    if representation is None:
        representation = Discrete()

    circles_array = np.asarray(circles, dtype=np.float32)
    if circles_array.ndim == 1:
        circles_array = circles_array[None, :]
    N = len(circles_array)
    S = len(sets)
    angles = np.linspace(0, 2 * np.pi, representation.k_angles, endpoint=False).astype(
        np.float32
    )
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

    has_label = label_rect_size is not None
    if has_label:
        label_hw = np.full(S, label_rect_size[0], dtype=np.float32)
        label_hh = np.full(S, label_rect_size[1], dtype=np.float32)
        # Start each label just above the upward-pointing boundary point
        k_top = int(np.argmin(np.abs(angles - np.pi / 2)))
    else:
        label_hw = label_hh = np.empty(S, dtype=np.float32)
        k_top = 0

    input_parameters = {
        "circle_radii": circle_radii,
        "initial_circle_positions": initial_circle_positions,
        "angles": angles,
        "membership": membership,
        "offsets": offsets_array,
        "circle_collision_alpha": np.float32(circle_collision_alpha),
        "convexity_alpha": np.float32(convexity_alpha),
    }
    if has_label:
        input_parameters["label_rect_hw"] = label_hw
        input_parameters["label_rect_hh"] = label_hh

    init_vars = representation.initialize_vars(S, initial_radii, initial_centers)

    pos_scale_x = max(float(np.std(circles_array[:, 0])), float(circle_radii.mean()))
    pos_scale_y = max(float(np.std(circles_array[:, 1])), float(circle_radii.mean()))
    rad_scale = float(initial_radii.mean())
    pos_scale_arr = np.array([pos_scale_x, pos_scale_y], dtype=np.float32)
    var_scales = {"centers": pos_scale_arr, "circle_positions": pos_scale_arr}
    for key in init_vars:
        if key != "centers":
            var_scales[key] = np.float32(rad_scale)
    if has_label:
        var_scales["label_positions"] = pos_scale_arr

    if has_label:
        initial_label_positions = initial_centers.copy()
        initial_label_positions[:, 1] += initial_radii[:, k_top] - label_hh

    def initialize(_, seed):
        d = {
            **{k: v.copy() for k, v in init_vars.items()},
            "circle_positions": initial_circle_positions.copy(),
        }
        if has_label:
            d["label_positions"] = initial_label_positions.copy()
        return d

    def wrap(fn):
        return representation.wrap(fn, angles_jnp)

    schedules = (
        term_schedules.schedules
        if isinstance(term_schedules, TermSchedules)
        else term_schedules
    ) or {}
    terms = [
        ObjectiveTerm(
            "enclosure",
            wrap(_multi_term_enclosure_movable),
            weight_enclosure,
            schedules.get("enclosure"),
        ),
        ObjectiveTerm(
            "exclusion",
            wrap(_multi_term_exclusion_movable),
            weight_exclusion,
            schedules.get("exclusion"),
        ),
        ObjectiveTerm(
            "min_radius",
            wrap(_multi_term_min_radius),
            10.0,
            schedules.get("min_radius"),
        ),
        ObjectiveTerm(
            "smoothness",
            wrap(_multi_term_smoothness),
            weight_smoothness,
            schedules.get("smoothness"),
        ),
        ObjectiveTerm(
            "convexity",
            wrap(_multi_term_convexity),
            weight_convexity,
            schedules.get("convexity"),
        ),
        ObjectiveTerm(
            "area", wrap(_multi_term_area), weight_area, schedules.get("area")
        ),
        ObjectiveTerm(
            "perimeter",
            wrap(_multi_term_perimeter),
            weight_perimeter,
            schedules.get("perimeter"),
        ),
        ObjectiveTerm(
            "position_anchor",
            _multi_term_position_anchor,
            weight_position_anchor,
            schedules.get("position_anchor"),
        ),
        ObjectiveTerm(
            "circle_collision",
            _multi_term_circle_collision,
            weight_circle_collision,
            schedules.get("circle_collision"),
        ),
        ObjectiveTerm(
            "bounding_box",
            wrap(_multi_term_total_bounding_box),
            weight_bounding_box,
            schedules.get("bounding_box"),
        ),
        ObjectiveTerm(
            "set_attraction",
            _multi_term_set_attraction,
            weight_set_attraction,
            schedules.get("set_attraction"),
        ),
    ]
    if has_label:
        terms += [
            ObjectiveTerm(
                "label_enclosure",
                wrap(_multi_term_label_enclosure),
                weight_label_enclosure,
                schedules.get("label_enclosure"),
            ),
            ObjectiveTerm(
                "label_element_exclusion",
                _multi_term_label_element_exclusion,
                weight_label_element_exclusion,
                schedules.get("label_element_exclusion"),
            ),
            ObjectiveTerm(
                "label_collision",
                _multi_term_label_label_collision,
                weight_label_collision,
                schedules.get("label_collision"),
            ),
            ObjectiveTerm(
                "label_top",
                _multi_term_label_top_attraction,
                weight_label_top,
                schedules.get("label_top"),
            ),
        ]

    problem = OptimizationProblemTemplate(
        terms=terms,
        initialize=initialize,
        svg_configuration=representation.make_svg_configuration(
            _svg_configuration_movable
        ),
    ).instantiate(input_parameters, var_scales=var_scales)
    result = problem.optimize(optim_config, callback=callback)

    circles_out = np.concatenate(
        [np.array(result.optim_vars["circle_positions"]), circle_radii[:, None]], axis=1
    )

    radii_arr = np.array(representation.to_radii(result.optim_vars, angles_jnp))
    return (
        [
            {
                "center": np.array(result.optim_vars["centers"][s]),
                "radii": radii_arr[s],
                "angles": angles,
                **(
                    {"label_center": np.array(result.optim_vars["label_positions"][s])}
                    if has_label
                    else {}
                ),
                **representation.extra_results(s, result.optim_vars),
            }
            for s in range(S)
        ],
        circles_out,
        result.history,
        problem,
    )


def optimize_radially_convex_sets_and_circles_from_graph(
    inclusion_graph: nx.DiGraph,
    weight_area=1.0,
    weight_perimeter=1.0,
    weight_enclosure=10.0,
    weight_exclusion=10.0,
    weight_smoothness=1.0,
    weight_convexity=0.0,
    weight_position_anchor=1.0,
    weight_circle_collision=10.0,
    weight_bounding_box=0.0,
    weight_set_attraction=0.0,
    circle_collision_alpha=1.0,
    convexity_alpha=1.0,
    offsets=None,
    label_rect_size: tuple[float, float] | None = None,
    weight_label_enclosure: float = 10.0,
    weight_label_element_exclusion: float = 10.0,
    weight_label_collision: float = 10.0,
    weight_label_top: float = 1.0,
    representation: StarRepresentation | None = None,
    term_schedules=None,
    optim_config: OptimConfig | None = None,
    callback: Callback | None = None,
):
    """Like optimize_radially_convex_sets_and_circles, but takes a DiGraph.

    Leaf nodes (out-degree 0) become circles; internal nodes (out-degree > 0) become
    sets. A leaf belongs to a set if it is a descendant of that set.

    Args:
        inclusion_graph: DiGraph with parent→child edges (edge (u, v) means v ⊂ u).
            Leaf nodes must carry ``center`` ([x, y]) and ``r`` (float) attributes.
        weight_area: weight for the area objective.
        weight_perimeter: weight for the perimeter objective.
        weight_enclosure: weight for the enclosure penalty. Default 10.0.
        weight_exclusion: weight for the exclusion penalty.
        weight_smoothness: weight for the smoothness penalty.
        weight_convexity: weight for the convexity penalty. Default 0.0 (disabled).
        weight_position_anchor: weight for penalizing circle positions deviating
            from their initial positions.
        weight_circle_collision: weight for penalizing overlapping circles.
        weight_bounding_box: weight for minimizing total width + total height of
            all set boundaries. Default 0.0 (disabled).
        weight_set_attraction: weight for pulling each circle toward the center
            of every set it belongs to. Default 0.0 (disabled).
        circle_collision_alpha: coefficient for the linear term in the circle
            collision penalty. Default 0.0 (pure quadratic).
        convexity_alpha: coefficient for the linear term in the convexity
            penalty. Default 0.5.
        offsets: padding added to each circle's radius in the enclosure term,
            per ``(set, circle)`` pair. Scalar, shape ``(N,)``, or shape ``(S, N)``.
            When ``None`` (default), computed automatically from the graph hierarchy
            via :func:`offsets_from_graph`: outer sets get larger offsets so that
            nested set boundaries are visually distinguishable.
        label_rect_size: ``(hw, hh)`` half-extents of the label rectangle.
            See :func:`optimize_radially_convex_sets_and_circles`.
        weight_label_enclosure: weight for the label enclosure term.
        weight_label_element_exclusion: weight for label-circle exclusion.
        weight_label_collision: weight for label-label collision.
        weight_label_top: weight for the upward-attraction term.
        representation: star domain parametrization.
        term_schedules: optional dict mapping term name to a schedule callable.
        optim_config: Optimizer settings. Uses :class:`OptimConfig` defaults when ``None``.
        callback: Optional callback called after each iteration.

    Returns:
        Tuple of (results, circles_out, history, problem) where results maps
        set node name → dict with "center", "radii", "angles", and (when
        ``label_rect_size`` is set) "label_center"; circles_out maps
        leaf node name → array of shape (3,) with optimized [cx, cy, r]; history
        is the list of per-iteration loss dicts; and problem is the
        :class:`~vizopt.base.OptimizationProblem` instance.
    """
    leaf_names, circles, name_to_idx = _leaf_circles_from_graph(inclusion_graph)
    set_names, sets = _sets_from_graph(inclusion_graph, leaf_names, name_to_idx)

    if offsets is None:
        offsets = offsets_from_graph(inclusion_graph, set_names, leaf_names)

    results_list, circles_out_arr, history, problem = (
        optimize_radially_convex_sets_and_circles(
            circles=circles,
            sets=sets,
            weight_area=weight_area,
            weight_perimeter=weight_perimeter,
            weight_enclosure=weight_enclosure,
            weight_exclusion=weight_exclusion,
            weight_smoothness=weight_smoothness,
            weight_convexity=weight_convexity,
            weight_position_anchor=weight_position_anchor,
            weight_circle_collision=weight_circle_collision,
            weight_bounding_box=weight_bounding_box,
            weight_set_attraction=weight_set_attraction,
            circle_collision_alpha=circle_collision_alpha,
            convexity_alpha=convexity_alpha,
            offsets=offsets,
            label_rect_size=label_rect_size,
            weight_label_enclosure=weight_label_enclosure,
            weight_label_element_exclusion=weight_label_element_exclusion,
            weight_label_collision=weight_label_collision,
            weight_label_top=weight_label_top,
            representation=representation,
            term_schedules=term_schedules,
            optim_config=optim_config,
            callback=callback,
        )
    )

    named_results = {set_names[s]: results_list[s] for s in range(len(set_names))}
    named_circles_out = {
        leaf_names[i]: circles_out_arr[i] for i in range(len(leaf_names))
    }
    return named_results, named_circles_out, history, problem
