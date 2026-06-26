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

from ...base import ObjectiveTerm, OptimizationProblem, OptimizationProblemTemplate, VizOptimizer
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
        t_enter, t_exit, hits — each (S, K, N).  `hits` is True when the ray
        intersects the rectangle and the intersection is in the forward
        direction (t_exit >= 0).
    """
    dx = rect_xy[None, :, 0] - centers[:, None, 0]  # (S, N)
    dy = rect_xy[None, :, 1] - centers[:, None, 1]

    cos_a = jnp.cos(angles)  # (K,)
    sin_a = jnp.sin(angles)

    dx_ = dx[:, None, :]  # (S, 1, N)
    dy_ = dy[:, None, :]
    ca_ = cos_a[None, :, None]  # (1, K, 1)
    sa_ = sin_a[None, :, None]
    hw_ = hw_sn[:, None, :]  # (S, 1, N)
    hh_ = hh_sn[:, None, :]

    # When cos ≈ 0 the ray is axis-aligned in y; the x-slab is a boolean
    # condition (center inside or outside) rather than a range of t.
    near_cos = jnp.abs(ca_) < _EPS
    near_sin = jnp.abs(sa_) < _EPS

    safe_cos = jnp.where(near_cos, _EPS, ca_)
    t_x1 = (dx_ - hw_) / safe_cos
    t_x2 = (dx_ + hw_) / safe_cos
    x_in_range = jnp.abs(dx_) <= hw_
    tx_lo = jnp.where(
        near_cos, jnp.where(x_in_range, -_INF, _INF), jnp.minimum(t_x1, t_x2)
    )
    tx_hi = jnp.where(
        near_cos, jnp.where(x_in_range, _INF, -_INF), jnp.maximum(t_x1, t_x2)
    )

    safe_sin = jnp.where(near_sin, _EPS, sa_)
    t_y1 = (dy_ - hh_) / safe_sin
    t_y2 = (dy_ + hh_) / safe_sin
    y_in_range = jnp.abs(dy_) <= hh_
    ty_lo = jnp.where(
        near_sin, jnp.where(y_in_range, -_INF, _INF), jnp.minimum(t_y1, t_y2)
    )
    ty_hi = jnp.where(
        near_sin, jnp.where(y_in_range, _INF, -_INF), jnp.maximum(t_y1, t_y2)
    )

    t_enter = jnp.maximum(tx_lo, ty_lo)  # (S, K, N)
    t_exit = jnp.minimum(tx_hi, ty_hi)
    hits = (t_enter <= t_exit) & (t_exit >= 0)
    return t_enter, t_exit, hits


# ---------------------------------------------------------------------------
# Enclosure / exclusion core penalties
# ---------------------------------------------------------------------------


def _enclosure_penalty_rect(
    centers, radii, rect_xy, rect_hw, rect_hh, offsets, angles, membership
):
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


def _exclusion_penalty_rect(
    centers, radii, rect_xy, rect_hw, rect_hh, offsets, angles, membership
):
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
    return jnp.sum(
        (optim_vars["rect_positions"] - input_params["initial_rect_positions"]) ** 2
    )


def _term_rect_collision(optim_vars, input_params):
    """Penalty for overlapping rectangles.

    Uses the minimum axis-wise penetration depth: two AABBs overlap iff both
    x- and y-gaps are negative, so violation = max(0, min(overlap_x, overlap_y)).

    Penalty is `overlap² + alpha * overlap`. The linear term gives a constant
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
    centers = optim_vars["centers"]  # (S, 2)
    membership = input_params["membership"]  # (S, N)
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
        elements.append(
            {
                "tag": "polygon",
                "fill": color,
                "fill-opacity": "0.12",
                "stroke": color,
                "stroke-width": "1.5",
                "stroke-linejoin": "round",
                "points": points_frames,
            }
        )

    for i in range(N):
        w_svg = dim_to_svg(2 * float(rect_hw[i]))
        h_svg = dim_to_svg(2 * float(rect_hh[i]))
        x_frames, y_frames = [], []
        for _, v in snapshots:
            px, py = to_svg(
                float(v["rect_positions"][i, 0]), float(v["rect_positions"][i, 1])
            )
            x_frames.append(f"{px - w_svg / 2:.1f}")
            y_frames.append(f"{py - h_svg / 2:.1f}")
        elements.append(
            {
                "tag": "rect",
                "width": f"{w_svg:.1f}",
                "height": f"{h_svg:.1f}",
                "fill": "#4472c4",
                "fill-opacity": "0.45",
                "stroke": "#2a52a0",
                "stroke-width": "1",
                "x": x_frames,
                "y": y_frames,
            }
        )

    if has_labels:
        label_hw = input_params["label_rect_hw"]
        label_hh = input_params["label_rect_hh"]
        for s in range(S):
            color = _SVG_SET_COLORS[s % len(_SVG_SET_COLORS)]
            w_svg = dim_to_svg(2 * float(label_hw[s]))
            h_svg = dim_to_svg(2 * float(label_hh[s]))
            x_frames, y_frames = [], []
            for _, v in snapshots:
                lx, ly = float(v["label_positions"][s, 0]), float(
                    v["label_positions"][s, 1]
                )
                px, py = to_svg(lx, ly)
                x_frames.append(f"{px - w_svg / 2:.1f}")
                y_frames.append(f"{py - h_svg / 2:.1f}")
            elements.append(
                {
                    "tag": "rect",
                    "width": f"{w_svg:.1f}",
                    "height": f"{h_svg:.1f}",
                    "fill": color,
                    "fill-opacity": "0.35",
                    "stroke": color,
                    "stroke-width": "1",
                    "x": x_frames,
                    "y": y_frames,
                }
            )

    return elements


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------


def _init_centers_and_radii_from_rects(
    rects_array, sets, angles, radius_multiplier=1.05
):
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
        tx_lo = np.where(
            near_cos, np.where(x_in_range, -_INF, _INF), np.minimum(t_x1, t_x2)
        )
        tx_hi = np.where(
            near_cos, np.where(x_in_range, _INF, -_INF), np.maximum(t_x1, t_x2)
        )

        # y-slab: (K, M)
        t_y1 = (dy - hh) / safe_sin
        t_y2 = (dy + hh) / safe_sin
        y_in_range = np.abs(dy) <= hh
        ty_lo = np.where(
            near_sin, np.where(y_in_range, -_INF, _INF), np.minimum(t_y1, t_y2)
        )
        ty_hi = np.where(
            near_sin, np.where(y_in_range, _INF, -_INF), np.maximum(t_y1, t_y2)
        )

        t_enter = np.maximum(tx_lo, ty_lo)  # (K, M)
        t_exit = np.minimum(tx_hi, ty_hi)
        hits = (t_enter <= t_exit) & (t_exit >= 0)
        t_exits_k = np.where(hits, t_exit, 0.0).max(axis=1)  # (K,)

        initial_radii[s] = np.maximum(radius_multiplier * t_exits_k, 1.0).astype(
            np.float32
        )

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
# Public API
# ---------------------------------------------------------------------------


class EulerDiagramRect(VizOptimizer):
    """Star-shaped boundary optimizer for sets of axis-aligned rectangles.

    Rectangle positions are optimization variables alongside the star boundaries.
    Construct directly from arrays or via :meth:`from_graph`.

    Args:
        rectangles: Array of shape `(N, 4)` with columns `[cx, cy, hw, hh]`.
        sets: List of S subsets, each a collection of integer indices into
            `rectangles`.
        weight_area: Weight for the area objective.
        weight_perimeter: Weight for the perimeter objective.
        weight_enclosure: Weight for the enclosure penalty.
        weight_exclusion: Weight for the exclusion penalty.
        weight_smoothness: Weight for the smoothness penalty.
        weight_convexity: Weight for the convexity penalty (penalises non-convex
            turns). Set to 0 to disable.
        weight_position_anchor: Weight for penalising rectangle positions
            deviating from initial positions.
        weight_rect_collision: Weight for penalising overlapping rectangles.
        weight_bounding_box: Weight for minimising total bounding box extent.
            Default 0.0 (disabled).
        weight_set_attraction: Weight for pulling rectangles toward set centers.
            Default 0.0 (disabled).
        rect_collision_alpha: Coefficient for the linear term in the rectangle
            collision penalty. Default 0.1.
        convexity_alpha: Coefficient for the linear term in the convexity
            penalty. Default 1.0.
        k_angles: Angular resolution (number of uniformly-spaced rays).
        offsets: Padding added to each rectangle's half-extents. Scalar,
            shape `(N,)`, or shape `(S, N)`.
        label_rect_size: `(hw, hh)` half-extents of a label rectangle to
            reserve at the top of each set boundary. When set, each star
            boundary encloses a floating label rect whose position is an
            optimization variable.
        label_membership: Bool array of shape `(S, S)` where `[s, l]` is
            True when boundary `s` must enclose label rect `l`. When
            `None`, each boundary encloses only its own label rect.
        weight_label_enclosure: Weight for the label enclosure term.
        weight_label_element_exclusion: Weight for keeping label rects away
            from leaf rectangles.
        weight_label_set_exclusion: Weight for keeping label rects away from
            other set boundaries.
        weight_label_collision: Weight for keeping label rects from overlapping.
        weight_label_top: Weight for the upward-attraction term on labels.
        term_schedules: Optional dict or :class:`~vizopt.schedules.TermSchedules`
            mapping term name to a JAX-compatible schedule callable.
        set_names: Display names for the S sets.
        leaf_names: Display names for the N rectangles.
    """

    def __init__(
        self,
        rectangles,
        sets,
        *,
        weight_area: float = 1.0,
        weight_perimeter: float = 1.0,
        weight_enclosure: float = 10.0,
        weight_exclusion: float = 10.0,
        weight_smoothness: float = 1.0,
        weight_convexity: float = 10.0,
        weight_position_anchor: float = 1.0,
        weight_rect_collision: float = 10.0,
        weight_bounding_box: float = 0.0,
        weight_set_attraction: float = 0.0,
        rect_collision_alpha: float = 0.1,
        convexity_alpha: float = 1.0,
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
        set_names: list[str] | None = None,
        leaf_names: list | None = None,
    ):
        self.rectangles = np.asarray(rectangles, dtype=np.float32)
        if self.rectangles.ndim == 1:
            self.rectangles = self.rectangles[None, :]
        self.sets = sets
        self.weight_area = weight_area
        self.weight_perimeter = weight_perimeter
        self.weight_enclosure = weight_enclosure
        self.weight_exclusion = weight_exclusion
        self.weight_smoothness = weight_smoothness
        self.weight_convexity = weight_convexity
        self.weight_position_anchor = weight_position_anchor
        self.weight_rect_collision = weight_rect_collision
        self.weight_bounding_box = weight_bounding_box
        self.weight_set_attraction = weight_set_attraction
        self.rect_collision_alpha = rect_collision_alpha
        self.convexity_alpha = convexity_alpha
        self.k_angles = k_angles
        self.offsets = offsets
        self.label_rect_size = label_rect_size
        self.label_membership = label_membership
        self.weight_label_enclosure = weight_label_enclosure
        self.weight_label_element_exclusion = weight_label_element_exclusion
        self.weight_label_set_exclusion = weight_label_set_exclusion
        self.weight_label_collision = weight_label_collision
        self.weight_label_top = weight_label_top
        self.term_schedules = term_schedules
        S, N = len(sets), len(self.rectangles)
        self.set_names = set_names if set_names is not None else [f"Set {s}" for s in range(S)]
        self.leaf_names = leaf_names if leaf_names is not None else list(range(N))

    @classmethod
    def from_graph(
        cls,
        inclusion_graph: nx.DiGraph,
        *,
        offsets=None,
        label_rect_size=None,
        **kwargs,
    ) -> "EulerDiagramRect":
        """Construct from a DiGraph where leaves are rectangles and internal nodes are sets.

        Leaf nodes (out-degree 0) become rectangles; internal nodes (out-degree > 0)
        become sets. A leaf belongs to a set if it is a descendant of that set.

        Args:
            inclusion_graph: DiGraph with parent→child edges (edge `(u, v)` means
                `v ⊂ u`). Leaf nodes must carry `center` (`[x, y]`), `hw`
                (half-width), and `hh` (half-height) attributes.
            offsets: Padding added to each rectangle's half-extents. When `None`
                (default), computed automatically from the graph hierarchy via
                :func:`~vizopt.templates.euler.graph_utils.offsets_from_graph`.
            label_rect_size: `(hw, hh)` half-extents of the label rectangle.
                When set, `label_membership` is derived from the hierarchy so
                that outer set boundaries also enclose nested labels.
            **kwargs: Forwarded to :meth:`__init__`.

        Returns:
            A configured :class:`EulerDiagramRect` ready to call :meth:`optimize`.
        """
        leaf_names, rects, name_to_idx = _leaf_rects_from_graph(inclusion_graph)
        set_names, sets = _sets_from_graph(inclusion_graph, leaf_names, name_to_idx)

        if offsets is None:
            mean_halfsize = float(np.mean(rects[:, 2:4]))
            offsets = offsets_from_graph(
                inclusion_graph,
                set_names,
                leaf_names,
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

        return cls(
            rects,
            sets,
            offsets=offsets,
            label_rect_size=label_rect_size,
            label_membership=label_membership,
            set_names=set_names,
            leaf_names=leaf_names,
            **kwargs,
        )

    def _build_problem(self) -> OptimizationProblem:
        rects_array = self.rectangles
        N = len(rects_array)
        S = len(self.sets)

        angles = np.linspace(0, 2 * np.pi, self.k_angles, endpoint=False).astype(np.float32)

        rect_hw = rects_array[:, 2].copy()
        rect_hh = rects_array[:, 3].copy()
        initial_rect_positions = rects_array[:, :2].copy()

        membership = _build_membership(S, N, self.sets)
        initial_centers, initial_radii = _init_centers_and_radii_from_rects(
            rects_array, self.sets, angles
        )
        offsets_array = np.broadcast_to(
            np.asarray(self.offsets, dtype=np.float32), (S, N)
        ).copy()

        label_rect_size = self.label_rect_size  # local alias for type narrowing
        label_hw: np.ndarray = np.empty(0, dtype=np.float32)
        label_hh: np.ndarray = np.empty(0, dtype=np.float32)
        label_membership_arr: np.ndarray = np.empty(0, dtype=bool)
        initial_label_positions: np.ndarray = np.empty(0, dtype=np.float32)
        if label_rect_size is not None:
            label_hw = np.full(S, label_rect_size[0], dtype=np.float32)
            label_hh = np.full(S, label_rect_size[1], dtype=np.float32)
            initial_label_positions = initial_centers.copy()
            initial_label_positions[:, 1] += np.max(initial_radii, axis=1) - label_hh
            label_membership_arr = (
                np.eye(S, dtype=bool)
                if self.label_membership is None
                else np.asarray(self.label_membership, dtype=bool)
            )

        input_parameters = {
            "rect_hw": rect_hw,
            "rect_hh": rect_hh,
            "initial_rect_positions": initial_rect_positions,
            "angles": angles,
            "membership": membership,
            "offsets": offsets_array,
            "rect_collision_alpha": np.float32(self.rect_collision_alpha),
            "convexity_alpha": np.float32(self.convexity_alpha),
        }
        if label_rect_size is not None:
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
        if label_rect_size is not None:
            var_scales["label_positions"] = pos_scale_arr

        def initialize(_, _seed):
            d = {
                "centers": initial_centers.copy(),
                "radii": initial_radii.copy(),
                "rect_positions": initial_rect_positions.copy(),
            }
            if label_rect_size is not None:
                d["label_positions"] = initial_label_positions.copy()
            return d

        schedules = (
            self.term_schedules.schedules
            if isinstance(self.term_schedules, TermSchedules)
            else self.term_schedules
        ) or {}

        terms = [
            ObjectiveTerm("enclosure", _term_enclosure_rect, self.weight_enclosure, schedules.get("enclosure")),
            ObjectiveTerm("exclusion", _term_exclusion_rect, self.weight_exclusion, schedules.get("exclusion")),
            ObjectiveTerm("min_radius", _multi_term_min_radius, 10.0, schedules.get("min_radius")),
            ObjectiveTerm("smoothness", _multi_term_smoothness, self.weight_smoothness, schedules.get("smoothness")),
            ObjectiveTerm("convexity", _multi_term_convexity, self.weight_convexity, schedules.get("convexity")),
            ObjectiveTerm("area", _multi_term_area, self.weight_area, schedules.get("area")),
            ObjectiveTerm("perimeter", _multi_term_perimeter, self.weight_perimeter, schedules.get("perimeter")),
            ObjectiveTerm("position_anchor", _term_position_anchor_rect, self.weight_position_anchor, schedules.get("position_anchor")),
            ObjectiveTerm("rect_collision", _term_rect_collision, self.weight_rect_collision, schedules.get("rect_collision")),
            ObjectiveTerm("bounding_box", _multi_term_total_bounding_box, self.weight_bounding_box, schedules.get("bounding_box")),
            ObjectiveTerm("set_attraction", _term_set_attraction_rect, self.weight_set_attraction, schedules.get("set_attraction")),
        ]
        if label_rect_size is not None:
            terms += [
                ObjectiveTerm("label_enclosure", _multi_term_label_enclosure, self.weight_label_enclosure, schedules.get("label_enclosure")),
                ObjectiveTerm("label_element_exclusion", _multi_term_label_element_exclusion_rect, self.weight_label_element_exclusion, schedules.get("label_element_exclusion")),
                ObjectiveTerm("label_set_exclusion", _multi_term_label_set_exclusion, self.weight_label_set_exclusion, schedules.get("label_set_exclusion")),
                ObjectiveTerm("label_collision", _multi_term_label_label_collision, self.weight_label_collision, schedules.get("label_collision")),
                ObjectiveTerm("label_top", _multi_term_label_top_attraction, self.weight_label_top, schedules.get("label_top")),
            ]

        return OptimizationProblemTemplate(
            terms=terms,
            initialize=initialize,
            svg_configuration=_svg_configuration_rect,
        ).instantiate(input_parameters, var_scales=var_scales)

    @property
    def sets_(self) -> list[dict]:
        """Star boundary dicts from the last optimization result.

        Each dict has `"center"`, `"radii"`, `"angles"`, and (when a label
        rect was used) `"label_center"`.

        Raises:
            ValueError: If :meth:`optimize` has not been called yet.
        """
        if not hasattr(self, "result_"):
            raise ValueError("No result yet — call optimize() first.")
        optim_vars = self.result_.optim_vars
        angles = self.problem_.input_parameters["angles"]
        radii_arr = np.array(optim_vars["radii"])
        has_label = "label_positions" in optim_vars
        return [
            {
                "center": np.array(optim_vars["centers"][s]),
                "radii": radii_arr[s],
                "angles": angles,
                **({"label_center": np.array(optim_vars["label_positions"][s])} if has_label else {}),
            }
            for s in range(len(self.set_names))
        ]

    @property
    def rects_(self) -> np.ndarray:
        """Optimized rectangle positions as an `(N, 4)` array of `[cx, cy, hw, hh]`.

        Raises:
            ValueError: If :meth:`optimize` has not been called yet.
        """
        if not hasattr(self, "result_"):
            raise ValueError("No result yet — call optimize() first.")
        rect_hw = self.problem_.input_parameters["rect_hw"]
        rect_hh = self.problem_.input_parameters["rect_hh"]
        return np.concatenate(
            [np.array(self.result_.optim_vars["rect_positions"]), rect_hw[:, None], rect_hh[:, None]],
            axis=1,
        )
