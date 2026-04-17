"""Reusable JAX loss terms and helpers for star-shaped (radially convex) domains.

A star-shaped region is represented by its center (cx, cy) and K radii at
uniformly-spaced angles θ_k = 2πk/K.  The boundary point at angle k is:

    p_k = center + radii[k] * [cos(θ_k), sin(θ_k)]
"""

import numpy as np
from jax import numpy as jnp

from ..utils import _SVG_SET_COLORS

_MIN_RADIUS = 0.1


# ---------------------------------------------------------------------------
# Core enclosure / exclusion primitives
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
        circle_r: (N,) or (S, N) circle radii (with any offsets already applied).
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

    # circle_r may be (N,) → (1, 1, N) or (S, N) → (S, 1, N)
    if circle_r.ndim == 1:
        r_sq = circle_r[None, None, :] ** 2  # (1, 1, N)
    else:
        r_sq = circle_r[:, None, :] ** 2  # (S, 1, N)
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


# ---------------------------------------------------------------------------
# ObjectiveTerm compute functions — fixed circle positions
#
# optim_vars keys: "centers" (S, 2), "radii" (S, K)
# input_params keys: "circles" (N, 3): [cx, cy, r], "angles" (K,),
#                    "membership" (S, N), "offsets" (S, N)
# ---------------------------------------------------------------------------


def _multi_term_enclosure(optim_vars, input_params):
    """Enclosure penalty summed over all sets.

    For each set s and its member circles, penalizes squared violations of
    radii[s, k] >= required radius at angle k for each member circle.
    The effective radius used is r + offset, where offset is per (set, circle).
    """
    circles = input_params["circles"]  # (N, 3)
    r = circles[None, :, 2] + input_params["offsets"]  # (S, N)
    return _enclosure_penalty(
        optim_vars["centers"],
        optim_vars["radii"],
        circles[:, :2],
        r,
        input_params["angles"],
        input_params["membership"],
    )


def _multi_term_exclusion(optim_vars, input_params):
    """Exclusion penalty: boundary must not overlap circles outside the set.

    For each set s and circle n not in set s, penalizes squared violations of
    radii[s, k] <= near_edge[s, k, n], where near_edge is the distance along
    ray k from center s to the near face of circle n.
    """
    circles = input_params["circles"]  # (N, 3)
    return _exclusion_penalty(
        optim_vars["centers"],
        optim_vars["radii"],
        circles[:, :2],
        circles[:, 2],
        input_params["angles"],
        input_params["membership"],
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
# ObjectiveTerm compute functions — movable circle positions
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
        optim_vars["centers"],
        optim_vars["radii"],
        optim_vars["circle_positions"],
        r,
        input_params["angles"],
        input_params["membership"],
    )


def _multi_term_exclusion_movable(optim_vars, input_params):
    """Exclusion penalty (circle positions are variables).

    Applies per-(set, circle) offsets to circle radii so that the boundary
    must stay at least offset away from each excluded circle.
    """
    r = input_params["circle_radii"][None, :] + input_params["offsets"]  # (S, N)
    return _exclusion_penalty(
        optim_vars["centers"],
        optim_vars["radii"],
        optim_vars["circle_positions"],
        r,
        input_params["angles"],
        input_params["membership"],
    )


def _multi_term_position_anchor(optim_vars, input_params):
    """Penalty for circle positions deviating from their initial positions."""
    circle_positions = optim_vars["circle_positions"]  # (N, 2)
    initial = input_params["initial_circle_positions"]  # (N, 2)
    return jnp.sum((circle_positions - initial) ** 2)


def _multi_term_total_bounding_box(optim_vars, input_params):
    """Total width plus total height of all set boundaries combined."""
    centers = optim_vars["centers"]  # (S, 2)
    radii = optim_vars["radii"]  # (S, K)
    angles = input_params["angles"]  # (K,)
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)  # (K, 2)
    points = (
        centers[:, None, :] + radii[:, :, None] * directions[None, :, :]
    )  # (S, K, 2)
    return (jnp.max(points[:, :, 0]) - jnp.min(points[:, :, 0])) + (
        jnp.max(points[:, :, 1]) - jnp.min(points[:, :, 1])
    )


def _multi_term_set_attraction(optim_vars, input_params):
    """Pulls circles toward the centers of sets they belong to (and vice versa).

    For each set s and each member circle n, penalizes squared distance from
    circle_positions[n] to centers[s]. The gradient acts on both circle
    positions and set centers, attracting them toward each other.
    """
    circle_positions = optim_vars["circle_positions"]  # (N, 2)
    centers = optim_vars["centers"]  # (S, 2)
    membership = input_params["membership"]  # (S, N) bool

    diff = circle_positions[None, :, :] - centers[:, None, :]  # (S, N, 2)
    dist_sq = jnp.sum(diff**2, axis=2)  # (S, N)
    return jnp.sum(jnp.where(membership, dist_sq, 0.0))


def _multi_term_circle_collision(optim_vars, input_params):
    """Penalty for overlapping circles.

    For each pair (i, j), penalizes overlap with a quadratic + linear term:
        max(0, r_i + r_j - dist(p_i, p_j))^2 + alpha * max(0, r_i + r_j - dist(p_i, p_j))

    The linear term gives a constant non-zero gradient for any overlap, preventing
    tiny violations from persisting due to vanishing gradients. alpha=0 recovers
    the pure quadratic penalty.
    """
    positions = optim_vars["circle_positions"]  # (N, 2)
    radii = input_params["circle_radii"]  # (N,)
    alpha = input_params["circle_collision_alpha"]

    diff = positions[:, None, :] - positions[None, :, :]  # (N, N, 2)
    dist = jnp.sqrt(jnp.sum(diff**2, axis=2) + 1e-12)  # (N, N)
    min_dist = radii[:, None] + radii[None, :]  # (N, N)
    overlap = jnp.maximum(0.0, min_dist - dist)  # (N, N)
    # Sum upper triangle only to count each pair once
    mask = jnp.triu(jnp.ones((radii.shape[0], radii.shape[0]), dtype=bool), k=1)
    return jnp.sum(jnp.where(mask, overlap**2 + alpha * overlap, 0.0))


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------


def _build_membership(S, N, sets):
    """Build a boolean membership matrix of shape (S, N)."""
    membership = np.zeros((S, N), dtype=bool)
    for s, subset in enumerate(sets):
        for i in subset:
            membership[s, i] = True
    return membership


def _init_centers_and_radii(
    circles_array,
    sets,
    angles,
    radius_smoothness: float = 0.0,
    radius_multiplier: float = 1.05,
):
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
        initial_radii[s] = radius_multiplier * np.maximum(
            r_required_masked.max(axis=1), 1.0
        ).astype(np.float32)
        max_radius = initial_radii[s].max()
        # for smoothness 0.0, each radius only as large as possible
        # for smoothness 1.0, each radius the same for all angles
        initial_radii[s] = (1 - radius_smoothness) * initial_radii[
            s
        ] + radius_smoothness * max_radius

    return initial_centers, initial_radii


# ---------------------------------------------------------------------------
# SVG animation helpers
# ---------------------------------------------------------------------------


def _compute_svg_transform(snapshots, circles_array, has_movable_circles, size):
    """Compute a world→SVG coordinate transform from the bounding box of all frames."""
    all_x, all_y = [], []
    for _, v in snapshots:
        centers = np.array(v["centers"])
        radii = np.array(v["radii"])
        angles_arr = np.linspace(0, 2 * np.pi, radii.shape[1], endpoint=False)
        for s in range(len(centers)):
            bx = centers[s, 0] + radii[s] * np.cos(angles_arr)
            by = centers[s, 1] + radii[s] * np.sin(angles_arr)
            all_x.extend(bx.tolist())
            all_y.extend(by.tolist())
        if has_movable_circles:
            pos = np.array(v["circle_positions"])
            all_x.extend(pos[:, 0].tolist())
            all_y.extend(pos[:, 1].tolist())
    # Include input circle extents
    r_col = circles_array[:, 2]
    all_x.extend((circles_array[:, 0] + r_col).tolist())
    all_x.extend((circles_array[:, 0] - r_col).tolist())
    all_y.extend((circles_array[:, 1] + r_col).tolist())
    all_y.extend((circles_array[:, 1] - r_col).tolist())

    x_min, y_min = min(all_x), min(all_y)
    span = max(max(all_x) - x_min, max(all_y) - y_min)
    margin = span * 0.05
    x_min -= margin
    y_max = y_min + span + 2 * margin
    span += 2 * margin

    def to_svg(x, y):
        sx = (x - x_min) / span * size
        sy = (y_max - y) / span * size
        return sx, sy

    def r_to_svg(r):
        return r / span * size

    return to_svg, r_to_svg


def _svg_configuration_fixed(snapshots, input_params, size):
    """SVG configuration for fixed-circles radially convex sets."""
    circles = input_params["circles"]
    angles = input_params["angles"]
    S = input_params["membership"].shape[0]
    N = len(circles)
    to_svg, r_to_svg = _compute_svg_transform(snapshots, circles, False, size)

    elements = []

    # Static input circles (drawn first, behind boundaries)
    for i in range(N):
        cx, cy, r = float(circles[i, 0]), float(circles[i, 1]), float(circles[i, 2])
        sx, sy = to_svg(cx, cy)
        elements.append({
            "tag": "circle",
            "cx": f"{sx:.1f}",
            "cy": f"{sy:.1f}",
            "r": f"{r_to_svg(r):.1f}",
            "fill": "#4472c4",
            "fill-opacity": "0.45",
            "stroke": "#2a52a0",
            "stroke-width": "1",
        })

    # Animated star-shaped boundaries (one polygon per set)
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

    return elements


def _svg_configuration_movable(snapshots, input_params, size):
    """SVG configuration for movable-circles radially convex sets."""
    circle_radii = input_params["circle_radii"]
    circles_array = np.column_stack([
        input_params["initial_circle_positions"],
        circle_radii,
    ])
    angles = input_params["angles"]
    S = input_params["membership"].shape[0]
    N = len(circle_radii)
    to_svg, r_to_svg = _compute_svg_transform(snapshots, circles_array, True, size)

    elements = []

    # Animated star-shaped boundaries (drawn first, behind circles)
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

    # Animated circles
    for i in range(N):
        r = float(circle_radii[i])
        cx_frames, cy_frames = [], []
        for _, v in snapshots:
            px, py = float(v["circle_positions"][i, 0]), float(v["circle_positions"][i, 1])
            sx, sy = to_svg(px, py)
            cx_frames.append(f"{sx:.1f}")
            cy_frames.append(f"{sy:.1f}")
        elements.append({
            "tag": "circle",
            "r": f"{r_to_svg(r):.1f}",
            "fill": "#4472c4",
            "fill-opacity": "0.45",
            "stroke": "#2a52a0",
            "stroke-width": "1",
            "cx": cx_frames,
            "cy": cy_frames,
        })

    return elements
