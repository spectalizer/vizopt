"""Star-shaped (convex) boundary optimization for sets of axis-aligned rectangles.

Finds convex star-shaped regions enclosing each set of axis-aligned rectangles
while minimizing area/perimeter and avoiding overlap with other sets.

Each boundary is parametrised as a center + K radii at uniformly-spaced angles.
Convexity is enforced via a penalty on negative cross products of consecutive
polygon edges.  For a convex polygon, a rectangle is enclosed iff all 4 corners
are inside — the ray/slab intersection method used here is the vectorised
equivalent of that check.
"""

import jax.numpy as jnp
import networkx as nx
import numpy as np

from ...base import Callback, ObjectiveTerm, OptimConfig, OptimizationProblemTemplate
from ...components.stars import (
    _build_membership,
    _multi_term_area,
    _multi_term_convexity,
    _multi_term_label_element_exclusion_rect,
    _multi_term_label_enclosure,
    _multi_term_label_label_collision,
    _multi_term_label_set_exclusion,
    _multi_term_label_top_attraction,
    _multi_term_min_radius,
    _multi_term_perimeter,
    _multi_term_smoothness,
    _multi_term_total_bounding_box,
)
from ...schedules import TermSchedules
from ...utils import _SVG_SET_COLORS
from .graph_utils import offsets_from_graph

_EPS = 1e-7
_INF = 1e9


# ---------------------------------------------------------------------------
# Slab intersection helper
# ---------------------------------------------------------------------------


def _slab_t(centers, rect_xy, hw_sn, hh_sn, angles):
    """Ray-slab intersection for all (set, angle, rect) triples.

    Args:
        centers: (S, 2) set centers.
        rect_xy: (N, 2) rectangle centers.
        hw_sn: (S, N) effective half-widths (may include offset).
        hh_sn: (S, N) effective half-heights.
        angles: (K,) ray angles.

    Returns:
        t_enter, t_exit, hits — each (S, K, N).  ``hits`` is True when the ray
        intersects the rectangle and the intersection is in the forward
        direction (t_exit >= 0).
    """
    dx = rect_xy[None, :, 0] - centers[:, None, 0]  # (S, N)
    dy = rect_xy[None, :, 1] - centers[:, None, 1]

    cos_a = jnp.cos(angles)  # (K,)
    sin_a = jnp.sin(angles)

    dx_ = dx[:, None, :]        # (S, 1, N)
    dy_ = dy[:, None, :]
    ca_ = cos_a[None, :, None]  # (1, K, 1)
    sa_ = sin_a[None, :, None]
    hw_ = hw_sn[:, None, :]     # (S, 1, N)
    hh_ = hh_sn[:, None, :]

    # When cos ≈ 0 the ray is axis-aligned in y; the x-slab is a boolean
    # condition (center inside or outside) rather than a range of t.
    near_cos = jnp.abs(ca_) < _EPS
    near_sin = jnp.abs(sa_) < _EPS

    safe_cos = jnp.where(near_cos, _EPS, ca_)
    t_x1 = (dx_ - hw_) / safe_cos
    t_x2 = (dx_ + hw_) / safe_cos
    x_in_range = jnp.abs(dx_) <= hw_
    tx_lo = jnp.where(near_cos, jnp.where(x_in_range, -_INF, _INF), jnp.minimum(t_x1, t_x2))
    tx_hi = jnp.where(near_cos, jnp.where(x_in_range,  _INF, -_INF), jnp.maximum(t_x1, t_x2))

    safe_sin = jnp.where(near_sin, _EPS, sa_)
    t_y1 = (dy_ - hh_) / safe_sin
    t_y2 = (dy_ + hh_) / safe_sin
    y_in_range = jnp.abs(dy_) <= hh_
    ty_lo = jnp.where(near_sin, jnp.where(y_in_range, -_INF, _INF), jnp.minimum(t_y1, t_y2))
    ty_hi = jnp.where(near_sin, jnp.where(y_in_range,  _INF, -_INF), jnp.maximum(t_y1, t_y2))

    t_enter = jnp.maximum(tx_lo, ty_lo)  # (S, K, N)
    t_exit = jnp.minimum(tx_hi, ty_hi)
    hits = (t_enter <= t_exit) & (t_exit >= 0)
    return t_enter, t_exit, hits


# ---------------------------------------------------------------------------
# Enclosure / exclusion core penalties
# ---------------------------------------------------------------------------


def _enclosure_penalty_rect(centers, radii, rect_xy, rect_hw, rect_hh, offsets, angles, membership):
    """Enclosure penalty: boundary must reach the far edge of each member rect.

    Args:
        centers: (S, 2).
        radii: (S, K).
        rect_xy: (N, 2) rectangle centers.
        rect_hw: (N,) half-widths.
        rect_hh: (N,) half-heights.
        offsets: (S, N) padding per (set, rect) pair.
        angles: (K,).
        membership: (S, N) bool.

    Returns:
        Scalar penalty.
    """
    hw_eff = rect_hw[None, :] + offsets  # (S, N)
    hh_eff = rect_hh[None, :] + offsets
    _, t_exit, hits = _slab_t(centers, rect_xy, hw_eff, hh_eff, angles)
    in_set = membership[:, None, :]  # (S, 1, N)
    violations = jnp.where(
        hits & in_set,
        jnp.maximum(0.0, t_exit - radii[:, :, None]),
        0.0,
    )
    return jnp.sum(violations**2)


def _exclusion_penalty_rect(centers, radii, rect_xy, rect_hw, rect_hh, offsets, angles, membership):
    """Exclusion penalty: boundary must not reach the near edge of excluded rects.

    Args:
        centers: (S, 2).
        radii: (S, K).
        rect_xy: (N, 2) rectangle centers.
        rect_hw: (N,) half-widths.
        rect_hh: (N,) half-heights.
        offsets: (S, N) padding per (set, rect) pair.
        angles: (K,).
        membership: (S, N) bool.

    Returns:
        Scalar penalty.
    """
    hw_eff = rect_hw[None, :] + offsets  # (S, N)
    hh_eff = rect_hh[None, :] + offsets
    t_enter, _, hits = _slab_t(centers, rect_xy, hw_eff, hh_eff, angles)
    not_in_set = ~membership[:, None, :]  # (S, 1, N)
    violations = jnp.where(
        hits & not_in_set & (t_enter > 0),
        jnp.maximum(0.0, radii[:, :, None] - t_enter),
        0.0,
    )
    return jnp.sum(violations**2)


# ---------------------------------------------------------------------------
# ObjectiveTerm compute functions
#
# optim_vars: "centers" (S, 2), "radii" (S, K), "rect_positions" (N, 2)
# input_params: "rect_hw" (N,), "rect_hh" (N,), "angles" (K,),
#               "membership" (S, N), "offsets" (S, N),
#               "initial_rect_positions" (N, 2)
# ---------------------------------------------------------------------------


def _term_enclosure_rect(optim_vars, input_params):
    return _enclosure_penalty_rect(
        optim_vars["centers"],
        optim_vars["radii"],
        optim_vars["rect_positions"],
        input_params["rect_hw"],
        input_params["rect_hh"],
        input_params["offsets"],
        input_params["angles"],
        input_params["membership"],
    )


def _term_exclusion_rect(optim_vars, input_params):
    return _exclusion_penalty_rect(
        optim_vars["centers"],
        optim_vars["radii"],
        optim_vars["rect_positions"],
        input_params["rect_hw"],
        input_params["rect_hh"],
        input_params["offsets"],
        input_params["angles"],
        input_params["membership"],
    )



def _term_position_anchor_rect(optim_vars, input_params):
    return jnp.sum((optim_vars["rect_positions"] - input_params["initial_rect_positions"]) ** 2)


def _term_rect_collision(optim_vars, input_params):
    """Penalty for overlapping rectangles.

    Uses the minimum axis-wise penetration depth: two AABBs overlap iff both
    x- and y-gaps are negative, so violation = max(0, min(overlap_x, overlap_y)).

    Penalty is ``overlap² + alpha * overlap``. The linear term gives a constant
    non-zero gradient for any overlap, preventing tiny violations from persisting
    due to vanishing gradients. alpha=0 recovers the pure quadratic penalty.
    """
    positions = optim_vars["rect_positions"]  # (N, 2)
    hw = input_params["rect_hw"]  # (N,)
    hh = input_params["rect_hh"]
    alpha = input_params["rect_collision_alpha"]
    N = hw.shape[0]

    dx = jnp.abs(positions[:, None, 0] - positions[None, :, 0])  # (N, N)
    dy = jnp.abs(positions[:, None, 1] - positions[None, :, 1])
    overlap_x = jnp.maximum(0.0, hw[:, None] + hw[None, :] - dx)
    overlap_y = jnp.maximum(0.0, hh[:, None] + hh[None, :] - dy)
    overlap = jnp.minimum(overlap_x, overlap_y)
    mask = jnp.triu(jnp.ones((N, N), dtype=bool), k=1)
    return jnp.sum(jnp.where(mask, overlap**2 + alpha * overlap, 0.0))


def _term_set_attraction_rect(optim_vars, input_params):
    """Pulls each rectangle toward the center of its set(s) (and vice versa)."""
    rect_positions = optim_vars["rect_positions"]  # (N, 2)
    centers = optim_vars["centers"]                # (S, 2)
    membership = input_params["membership"]        # (S, N)
    diff = rect_positions[None, :, :] - centers[:, None, :]  # (S, N, 2)
    dist_sq = jnp.sum(diff**2, axis=2)
    return jnp.sum(jnp.where(membership, dist_sq, 0.0))


# ---------------------------------------------------------------------------
# SVG animation helper
# ---------------------------------------------------------------------------


def _svg_configuration_rect(snapshots, input_params, size):
    """SVG elements for animated convex star boundaries + movable rectangles."""
    rect_hw = input_params["rect_hw"]
    rect_hh = input_params["rect_hh"]
    angles = input_params["angles"]
    S = input_params["membership"].shape[0]
    N = len(rect_hw)
    has_labels = "label_positions" in snapshots[0][1]

    all_x, all_y = [], []
    for _, v in snapshots:
        centers = np.array(v["centers"])
        radii = np.array(v["radii"])
        for s in range(S):
            bx = centers[s, 0] + radii[s] * np.cos(angles)
            by = centers[s, 1] + radii[s] * np.sin(angles)
            all_x.extend(bx.tolist())
            all_y.extend(by.tolist())
        pos = np.array(v["rect_positions"])
        for i in range(N):
            cx, cy = pos[i, 0], pos[i, 1]
            all_x.extend([cx - rect_hw[i], cx + rect_hw[i]])
            all_y.extend([cy - rect_hh[i], cy + rect_hh[i]])
        if has_labels:
            label_hw = input_params["label_rect_hw"]
            label_hh = input_params["label_rect_hh"]
            lpos = np.array(v["label_positions"])
            for s in range(S):
                lx, ly = lpos[s, 0], lpos[s, 1]
                all_x.extend([lx - label_hw[s], lx + label_hw[s]])
                all_y.extend([ly - label_hh[s], ly + label_hh[s]])

    x_min, y_min = min(all_x), min(all_y)
    span = max(max(all_x) - x_min, max(all_y) - y_min)
    margin = span * 0.05
    x_min -= margin
    y_max = y_min + span + 2 * margin
    span += 2 * margin

    def to_svg(x, y):
        return (x - x_min) / span * size, (y_max - y) / span * size

    def dim_to_svg(d):
        return d / span * size

    elements = []

    for s in range(S):
        color = _SVG_SET_COLORS[s % len(_SVG_SET_COLORS)]
        points_frames = []
        for _, v in snapshots:
            cx, cy = float(v["centers"][s, 0]), float(v["centers"][s, 1])
            s_radii = np.array(v["radii"][s])
            pts = []
            for k in range(len(angles)):
                bx = cx + s_radii[k] * np.cos(angles[k])
                by = cy + s_radii[k] * np.sin(angles[k])
                px, py = to_svg(bx, by)
                pts.append(f"{px:.1f},{py:.1f}")
            points_frames.append(" ".join(pts))
        elements.append({
            "tag": "polygon",
            "fill": color,
            "fill-opacity": "0.12",
            "stroke": color,
            "stroke-width": "1.5",
            "stroke-linejoin": "round",
            "points": points_frames,
        })

    for i in range(N):
        w_svg = dim_to_svg(2 * float(rect_hw[i]))
        h_svg = dim_to_svg(2 * float(rect_hh[i]))
        x_frames, y_frames = [], []
        for _, v in snapshots:
            px, py = to_svg(float(v["rect_positions"][i, 0]), float(v["rect_positions"][i, 1]))
            x_frames.append(f"{px - w_svg / 2:.1f}")
            y_frames.append(f"{py - h_svg / 2:.1f}")
        elements.append({
            "tag": "rect",
            "width": f"{w_svg:.1f}",
            "height": f"{h_svg:.1f}",
            "fill": "#4472c4",
            "fill-opacity": "0.45",
            "stroke": "#2a52a0",
            "stroke-width": "1",
            "x": x_frames,
            "y": y_frames,
        })

    if has_labels:
        label_hw = input_params["label_rect_hw"]
        label_hh = input_params["label_rect_hh"]
        for s in range(S):
            color = _SVG_SET_COLORS[s % len(_SVG_SET_COLORS)]
            w_svg = dim_to_svg(2 * float(label_hw[s]))
            h_svg = dim_to_svg(2 * float(label_hh[s]))
            x_frames, y_frames = [], []
            for _, v in snapshots:
                lx, ly = float(v["label_positions"][s, 0]), float(v["label_positions"][s, 1])
                px, py = to_svg(lx, ly)
                x_frames.append(f"{px - w_svg / 2:.1f}")
                y_frames.append(f"{py - h_svg / 2:.1f}")
            elements.append({
                "tag": "rect",
                "width": f"{w_svg:.1f}",
                "height": f"{h_svg:.1f}",
                "fill": color,
                "fill-opacity": "0.35",
                "stroke": color,
                "stroke-width": "1",
                "x": x_frames,
                "y": y_frames,
            })

    return elements


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------


def _init_centers_and_radii_from_rects(rects_array, sets, angles, radius_multiplier=1.05):
    """Initialize per-set centers and radii from member rectangles.

    Args:
        rects_array: (N, 4) array of [cx, cy, hw, hh].
        sets: list of S index subsets.
        angles: (K,) array of ray angles.

    Returns:
        Tuple of (initial_centers, initial_radii) with shapes (S, 2) and (S, K).
    """
    S = len(sets)
    K = len(angles)
    cos_a = np.cos(angles)[:, None]  # (K, 1)
    sin_a = np.sin(angles)[:, None]
    initial_centers = np.zeros((S, 2), dtype=np.float32)
    initial_radii = np.ones((S, K), dtype=np.float32)

    near_cos = np.abs(cos_a) < _EPS  # (K, 1)
    near_sin = np.abs(sin_a) < _EPS
    safe_cos = np.where(near_cos, _EPS, cos_a)
    safe_sin = np.where(near_sin, _EPS, sin_a)

    for s, subset in enumerate(sets):
        sub = rects_array[list(subset)]  # (M, 4)
        center = np.mean(sub[:, :2], axis=0).astype(np.float32)
        initial_centers[s] = center

        dx = (sub[:, 0] - center[0])[None, :]  # (1, M)
        dy = (sub[:, 1] - center[1])[None, :]
        hw = sub[:, 2][None, :]  # (1, M)
        hh = sub[:, 3][None, :]

        # x-slab: (K, M)
        t_x1 = (dx - hw) / safe_cos
        t_x2 = (dx + hw) / safe_cos
        x_in_range = np.abs(dx) <= hw
        tx_lo = np.where(near_cos, np.where(x_in_range, -_INF, _INF), np.minimum(t_x1, t_x2))
        tx_hi = np.where(near_cos, np.where(x_in_range,  _INF, -_INF), np.maximum(t_x1, t_x2))

        # y-slab: (K, M)
        t_y1 = (dy - hh) / safe_sin
        t_y2 = (dy + hh) / safe_sin
        y_in_range = np.abs(dy) <= hh
        ty_lo = np.where(near_sin, np.where(y_in_range, -_INF, _INF), np.minimum(t_y1, t_y2))
        ty_hi = np.where(near_sin, np.where(y_in_range,  _INF, -_INF), np.maximum(t_y1, t_y2))

        t_enter = np.maximum(tx_lo, ty_lo)   # (K, M)
        t_exit = np.minimum(tx_hi, ty_hi)
        hits = (t_enter <= t_exit) & (t_exit >= 0)
        t_exits_k = np.where(hits, t_exit, 0.0).max(axis=1)  # (K,)

        initial_radii[s] = np.maximum(radius_multiplier * t_exits_k, 1.0).astype(np.float32)

    return initial_centers, initial_radii


def _leaf_rects_from_graph(inclusion_graph: nx.DiGraph):
    names = [n for n in inclusion_graph.nodes if inclusion_graph.out_degree(n) == 0]
    rects = np.array(
        [
            [
                *inclusion_graph.nodes[n]["center"],
                inclusion_graph.nodes[n]["hw"],
                inclusion_graph.nodes[n]["hh"],
            ]
            for n in names
        ],
        dtype=np.float32,
    )
    name_to_idx = {name: i for i, name in enumerate(names)}
    return names, rects, name_to_idx


def _sets_from_graph(inclusion_graph, leaf_names, name_to_idx):
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


# ---------------------------------------------------------------------------
# Main public functions
# ---------------------------------------------------------------------------


def optimize_radially_convex_sets_and_rectangles(
    rectangles,
    sets,
    weight_area=1.0,
    weight_perimeter=1.0,
    weight_exclusion=10.0,
    weight_smoothness=1.0,
    weight_convexity=10.0,
    weight_position_anchor=1.0,
    weight_rect_collision=10.0,
    weight_bounding_box=0.0,
    weight_set_attraction=0.0,
    rect_collision_alpha=0.1,
    convexity_alpha=1.0,
    k_angles: int = 64,
    offsets: float | np.ndarray = 0.1,
    label_rect_size: tuple[float, float] | None = None,
    label_membership: np.ndarray | None = None,
    weight_label_enclosure: float = 10.0,
    weight_label_element_exclusion: float = 10.0,
    weight_label_set_exclusion: float = 10.0,
    weight_label_collision: float = 10.0,
    weight_label_top: float = 1.0,
    term_schedules=None,
    optim_config: OptimConfig | None = None,
    callback: Callback | None = None,
):
    """Optimize convex star-shaped boundaries enclosing sets of axis-aligned rectangles.

    Rectangle positions (cx, cy) are optimization variables; half-widths and
    half-heights remain fixed.  A convexity penalty keeps the star polygon convex,
    which lets enclosure be checked via slab (ray-AABB) intersection.

    Args:
        rectangles: array of shape (N, 4) with columns [cx, cy, hw, hh], or a
            sequence of (cx, cy, hw, hh) tuples.
        sets: list of S subsets, each a collection of integer indices into
            ``rectangles``.
        weight_area: weight for the area objective.
        weight_perimeter: weight for the perimeter objective.
        weight_exclusion: weight for the exclusion penalty.
        weight_smoothness: weight for the smoothness penalty.
        weight_convexity: weight for the convexity penalty (penalises non-convex
            turns in the star polygon).  Set to 0 to disable.
        weight_position_anchor: weight for penalising rectangle positions
            deviating from their initial positions.
        weight_rect_collision: weight for penalising overlapping rectangles.
        weight_bounding_box: weight for minimising total width + height of all
            set boundaries.  Default 0.0 (disabled).
        weight_set_attraction: weight for pulling rectangles toward their set
            centers.  Default 0.0 (disabled).
        rect_collision_alpha: coefficient for the linear term in the rectangle
            collision penalty: ``overlap² + alpha * overlap``.  The linear term
            gives a constant non-zero gradient for any overlap, preventing tiny
            violations from persisting.  Default 0.0 (pure quadratic).
        convexity_alpha: coefficient for the linear term in the convexity
            penalty: ``violation² + alpha * violation``.  The linear term gives
            a non-zero gradient for any concavity, preventing mild violations
            from stalling.  Default 0.5.
        k_angles: angular resolution (number of uniformly-spaced rays).
        offsets: padding added to each rectangle's half-extents in the enclosure
            and exclusion terms.  Scalar, shape (N,), or shape (S, N).
        label_rect_size: ``(hw, hh)`` half-extents of the label rectangle to
            reserve at the top of each set boundary.  When ``None`` (default)
            no label rectangle is used.  When set, each star boundary is forced
            to enclose a floating label rect whose position (``label_positions``,
            one per set) is an optimization variable.
        label_membership: bool array of shape (S, S) where entry [s, l] is True
            when boundary ``s`` must enclose label rect ``l``.  When ``None``
            (default) each boundary encloses only its own label rect.  Pass a
            matrix derived from the set hierarchy to enforce that outer set
            boundaries also enclose the label rects of all nested sets.
        weight_label_enclosure: weight for the label enclosure term.  Default 10.0.
        weight_label_element_exclusion: weight for keeping label rects away from
            leaf rectangles.  Default 10.0.
        weight_label_collision: weight for keeping label rects from overlapping
            each other.  Default 10.0.
        weight_label_top: weight for the upward-attraction term.  Default 1.0.
        term_schedules: optional dict mapping term name to a JAX-compatible
            callable ``(step: Array) -> Array`` that scales the term's weight.
            Valid keys: ``"enclosure"``, ``"exclusion"``, ``"min_radius"``,
            ``"smoothness"``, ``"convexity"``, ``"area"``, ``"perimeter"``,
            ``"position_anchor"``, ``"rect_collision"``, ``"bounding_box"``,
            ``"set_attraction"``, ``"label_enclosure"``,
            ``"label_element_exclusion"``, ``"label_collision"``,
            ``"label_top"``.
        optim_config: optimizer settings.  Uses :class:`OptimConfig` defaults
            when ``None``.
        callback: optional callback called after each iteration with
            ``(iteration, loss, optim_vars, grads)``.

    Returns:
        Tuple of ``(results, rects_out, history, problem)`` where ``results``
        is a list of S dicts each with keys ``"center"``, ``"radii"``,
        ``"angles"``, and (when ``label_rect_size`` is set) ``"label_center"``;
        ``rects_out`` is an array of shape (N, 4) with optimised
        ``[cx, cy, hw, hh]``; ``history`` is the list of per-iteration loss
        dicts; and ``problem`` is the
        :class:`~vizopt.base.OptimizationProblem` instance.
    """
    rects_array = np.asarray(rectangles, dtype=np.float32)
    if rects_array.ndim == 1:
        rects_array = rects_array[None, :]
    N = len(rects_array)
    S = len(sets)

    angles = np.linspace(0, 2 * np.pi, k_angles, endpoint=False).astype(np.float32)

    rect_hw = rects_array[:, 2].copy()
    rect_hh = rects_array[:, 3].copy()
    initial_rect_positions = rects_array[:, :2].copy()

    membership = _build_membership(S, N, sets)
    initial_centers, initial_radii = _init_centers_and_radii_from_rects(
        rects_array, sets, angles
    )
    offsets_array = np.broadcast_to(
        np.asarray(offsets, dtype=np.float32), (S, N)
    ).copy()

    has_label = label_rect_size is not None
    if has_label:
        label_hw = np.full(S, label_rect_size[0], dtype=np.float32)
        label_hh = np.full(S, label_rect_size[1], dtype=np.float32)
        initial_label_positions = initial_centers.copy()
        initial_label_positions[:, 1] += np.max(initial_radii, axis=1) - label_hh
        label_membership_arr = (
            np.eye(S, dtype=bool)
            if label_membership is None
            else np.asarray(label_membership, dtype=bool)
        )

    input_parameters = {
        "rect_hw": rect_hw,
        "rect_hh": rect_hh,
        "initial_rect_positions": initial_rect_positions,
        "angles": angles,
        "membership": membership,
        "offsets": offsets_array,
        "rect_collision_alpha": np.float32(rect_collision_alpha),
        "convexity_alpha": np.float32(convexity_alpha),
    }
    if has_label:
        input_parameters["label_rect_hw"] = label_hw
        input_parameters["label_rect_hh"] = label_hh
        input_parameters["label_membership"] = label_membership_arr

    mean_size = float(np.mean(np.concatenate([rect_hw, rect_hh])))
    pos_scale_x = max(float(np.std(rects_array[:, 0])), mean_size)
    pos_scale_y = max(float(np.std(rects_array[:, 1])), mean_size)
    rad_scale = float(initial_radii.mean())
    pos_scale_arr = np.array([pos_scale_x, pos_scale_y], dtype=np.float32)
    var_scales = {
        "centers": pos_scale_arr,
        "rect_positions": pos_scale_arr,
        "radii": np.float32(rad_scale),
    }
    if has_label:
        var_scales["label_positions"] = pos_scale_arr

    def initialize(_, seed):
        d = {
            "centers": initial_centers.copy(),
            "radii": initial_radii.copy(),
            "rect_positions": initial_rect_positions.copy(),
        }
        if has_label:
            d["label_positions"] = initial_label_positions.copy()
        return d

    schedules = (
        term_schedules.schedules
        if isinstance(term_schedules, TermSchedules)
        else term_schedules
    ) or {}

    terms = [
        ObjectiveTerm("enclosure", _term_enclosure_rect, 10.0, schedules.get("enclosure")),
        ObjectiveTerm("exclusion", _term_exclusion_rect, weight_exclusion, schedules.get("exclusion")),
        ObjectiveTerm("min_radius", _multi_term_min_radius, 10.0, schedules.get("min_radius")),
        ObjectiveTerm("smoothness", _multi_term_smoothness, weight_smoothness, schedules.get("smoothness")),
        ObjectiveTerm("convexity", _multi_term_convexity, weight_convexity, schedules.get("convexity")),
        ObjectiveTerm("area", _multi_term_area, weight_area, schedules.get("area")),
        ObjectiveTerm("perimeter", _multi_term_perimeter, weight_perimeter, schedules.get("perimeter")),
        ObjectiveTerm("position_anchor", _term_position_anchor_rect, weight_position_anchor, schedules.get("position_anchor")),
        ObjectiveTerm("rect_collision", _term_rect_collision, weight_rect_collision, schedules.get("rect_collision")),
        ObjectiveTerm("bounding_box", _multi_term_total_bounding_box, weight_bounding_box, schedules.get("bounding_box")),
        ObjectiveTerm("set_attraction", _term_set_attraction_rect, weight_set_attraction, schedules.get("set_attraction")),
    ]
    if has_label:
        terms += [
            ObjectiveTerm("label_enclosure", _multi_term_label_enclosure, weight_label_enclosure, schedules.get("label_enclosure")),
            ObjectiveTerm("label_element_exclusion", _multi_term_label_element_exclusion_rect, weight_label_element_exclusion, schedules.get("label_element_exclusion")),
            ObjectiveTerm("label_set_exclusion", _multi_term_label_set_exclusion, weight_label_set_exclusion, schedules.get("label_set_exclusion")),
            ObjectiveTerm("label_collision", _multi_term_label_label_collision, weight_label_collision, schedules.get("label_collision")),
            ObjectiveTerm("label_top", _multi_term_label_top_attraction, weight_label_top, schedules.get("label_top")),
        ]

    problem = OptimizationProblemTemplate(
        terms=terms,
        initialize=initialize,
        svg_configuration=_svg_configuration_rect,
    ).instantiate(input_parameters, var_scales=var_scales)
    optim_vars, history = problem.optimize(optim_config, callback=callback)

    rects_out = np.concatenate(
        [np.array(optim_vars["rect_positions"]), rect_hw[:, None], rect_hh[:, None]],
        axis=1,
    )
    radii_arr = np.array(optim_vars["radii"])

    results = [
        {
            "center": np.array(optim_vars["centers"][s]),
            "radii": radii_arr[s],
            "angles": angles,
            **({"label_center": np.array(optim_vars["label_positions"][s])} if has_label else {}),
        }
        for s in range(S)
    ]
    return results, rects_out, history, problem


def optimize_radially_convex_sets_and_rectangles_from_graph(
    inclusion_graph: nx.DiGraph,
    weight_area=1.0,
    weight_perimeter=1.0,
    weight_exclusion=10.0,
    weight_smoothness=1.0,
    weight_convexity=10.0,
    weight_position_anchor=1.0,
    weight_rect_collision=10.0,
    weight_bounding_box=0.0,
    weight_set_attraction=0.0,
    rect_collision_alpha=0.1,
    convexity_alpha=1.0,
    k_angles: int = 64,
    offsets=None,
    label_rect_size: tuple[float, float] | None = None,
    weight_label_enclosure: float = 10.0,
    weight_label_element_exclusion: float = 10.0,
    weight_label_set_exclusion: float = 10.0,
    weight_label_collision: float = 10.0,
    weight_label_top: float = 1.0,
    term_schedules=None,
    optim_config: OptimConfig | None = None,
    callback: Callback | None = None,
):
    """Like optimize_radially_convex_sets_and_rectangles, but takes a DiGraph.

    Leaf nodes (out-degree 0) become rectangles; internal nodes become sets.
    A leaf belongs to a set if it is a descendant of that set.

    Args:
        inclusion_graph: DiGraph with parent→child edges (edge (u, v) means
            v ⊂ u).  Leaf nodes must carry ``center`` ([x, y]), ``hw``
            (half-width), and ``hh`` (half-height) attributes.
        weight_area: weight for the area objective.
        weight_perimeter: weight for the perimeter objective.
        weight_exclusion: weight for the exclusion penalty.
        weight_smoothness: weight for the smoothness penalty.
        weight_convexity: weight for the convexity penalty.
        weight_position_anchor: weight for the position anchor penalty.
        weight_rect_collision: weight for the rectangle collision penalty.
        weight_bounding_box: weight for the bounding-box objective.
        weight_set_attraction: weight for the set-attraction term.
        rect_collision_alpha: coefficient for the linear term in the rectangle
            collision penalty.  Default 0.0 (pure quadratic).
        convexity_alpha: coefficient for the linear term in the convexity
            penalty.  Default 0.5.
        k_angles: angular resolution.
        offsets: padding added to each rectangle's half-extents in the enclosure
            and exclusion terms.  Scalar, shape ``(N,)``, or shape ``(S, N)``.
            When ``None`` (default), computed automatically from the graph
            hierarchy via :func:`~vizopt.templates.euler.graph_utils.offsets_from_graph`:
            outer sets get larger offsets so that nested boundaries are visually
            distinguishable, scaled to mean rectangle half-size.
        label_rect_size: ``(hw, hh)`` half-extents of the label rectangle.
            See :func:`optimize_radially_convex_sets_and_rectangles`.
        weight_label_enclosure: weight for the label enclosure term.
        weight_label_element_exclusion: weight for label-rect exclusion.
        weight_label_collision: weight for label-label collision.
        weight_label_top: weight for the upward-attraction term.
        term_schedules: optional term schedule dict.
        optim_config: optimizer settings.
        callback: optional per-iteration callback.

    Returns:
        Tuple of ``(results, rects_out, history, problem)`` where ``results``
        maps set node name → dict with ``"center"``, ``"radii"``, ``"angles"``,
        and (when ``label_rect_size`` is set) ``"label_center"``;
        ``rects_out`` maps leaf node name → array of shape (4,) with optimised
        ``[cx, cy, hw, hh]``; ``history`` is the per-iteration loss list; and
        ``problem`` is the :class:`~vizopt.base.OptimizationProblem` instance.
    """
    leaf_names, rects, name_to_idx = _leaf_rects_from_graph(inclusion_graph)
    set_names, sets = _sets_from_graph(inclusion_graph, leaf_names, name_to_idx)

    if offsets is None:
        mean_halfsize = float(np.mean(rects[:, 2:4]))
        offsets = offsets_from_graph(
            inclusion_graph, set_names, leaf_names,
            offset_step=mean_halfsize * 0.3,
            sub_step=mean_halfsize * 0.1,
            min_offset=mean_halfsize * 0.1,
        )

    label_membership = None
    if label_rect_size is not None:
        S = len(set_names)
        label_membership = np.eye(S, dtype=bool)
        for i, s1 in enumerate(set_names):
            descendants = nx.descendants(inclusion_graph, s1)
            for j, s2 in enumerate(set_names):
                if s2 in descendants:
                    label_membership[i, j] = True

    results_list, rects_out_arr, history, problem = optimize_radially_convex_sets_and_rectangles(
        rectangles=rects,
        sets=sets,
        weight_area=weight_area,
        weight_perimeter=weight_perimeter,
        weight_exclusion=weight_exclusion,
        weight_smoothness=weight_smoothness,
        weight_convexity=weight_convexity,
        weight_position_anchor=weight_position_anchor,
        weight_rect_collision=weight_rect_collision,
        weight_bounding_box=weight_bounding_box,
        weight_set_attraction=weight_set_attraction,
        rect_collision_alpha=rect_collision_alpha,
        convexity_alpha=convexity_alpha,
        k_angles=k_angles,
        offsets=offsets,
        label_rect_size=label_rect_size,
        label_membership=label_membership,
        weight_label_enclosure=weight_label_enclosure,
        weight_label_element_exclusion=weight_label_element_exclusion,
        weight_label_set_exclusion=weight_label_set_exclusion,
        weight_label_collision=weight_label_collision,
        weight_label_top=weight_label_top,
        term_schedules=term_schedules,
        optim_config=optim_config,
        callback=callback,
    )

    named_results = {set_names[s]: results_list[s] for s in range(len(set_names))}
    named_rects_out = {leaf_names[i]: rects_out_arr[i] for i in range(len(leaf_names))}
    return named_results, named_rects_out, history, problem
