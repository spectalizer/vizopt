import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import SVG
from jax import numpy as jnp

from vizopt.animation import SnapshotCallback, snapshots_to_animated_svg
from vizopt.base import ObjectiveTerm, OptimConfig, OptimizationProblemTemplate
from vizopt.radially_convex import (
    _build_membership,
    _init_centers_and_radii,
    _multi_term_area,
    _multi_term_enclosure,
    _multi_term_min_radius,
    _multi_term_perimeter,
    _multi_term_smoothness,
    _svg_configuration_fixed,
)
from vizopt.utils import _SVG_SET_COLORS


def _dist_and_angle(diff):
    """Compute distance and angle for an array of 2-D displacement vectors.

    Uses the "double where" trick so that both the value and the gradient of
    arctan2 are finite even when diff = (0, 0).  At the origin the angle is
    set to 0 (an arbitrary but finite fallback); the caller is responsible for
    ensuring the origin case does not affect the final loss value.

    Args:
        diff: (..., 2) array of displacement vectors.

    Returns:
        dist: (...) Euclidean distances (with a small epsilon for safety).
        alpha: (...) angles in (-π, π].
    """
    rho2 = jnp.sum(diff**2, axis=-1)  # (...)
    dist = jnp.sqrt(rho2 + 1e-12)  # (...)
    safe = rho2 > 0
    dx_safe = jnp.where(safe, diff[..., 0], 1.0)
    dy_safe = jnp.where(safe, diff[..., 1], 0.0)
    alpha = jnp.arctan2(dy_safe, dx_safe)  # (...)
    return dist, alpha


def _multi_term_star_exclusion(optim_vars, input_params):
    """Star-vs-star exclusion: boundary and interior of each set must not enter any other set's interior.

    For each pair (s, t) where exclusion_mask[s, t] is True, penalises boundary
    and interior points of s that lie inside the star domain of t.  Interior
    points are sampled at uniform fractions of each radial direction (0.5 by
    default), in addition to the boundary (fraction 1.0).  This ensures a
    nonzero penalty — and nonzero gradient — even when the two domains are
    identical or one completely encloses the other.

    optim_vars keys: "centers" (S, 2), "radii" (S, K)
    input_params keys: "angles" (K,), "exclusion_mask" (S, S) bool
    Optional input_params keys:
        "exclusion_offset" (float): minimum gap enforced between domains.
        "exclusion_interior_fracs" (list[float]): fractions in (0, 1) at which
            to sample interior points along each radial direction.
            Defaults to [0.5].
    """
    centers = optim_vars["centers"]  # (S, 2)
    radii = optim_vars["radii"]  # (S, K)
    angles = input_params["angles"]  # (K,)
    mask = input_params["exclusion_mask"]  # (S, S) bool
    S, K = radii.shape

    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)  # (K, 2)

    # Fractions along each radial direction to sample.
    # 1.0 = boundary; values in (0, 1) sample the interior.
    interior_fracs = input_params.get("exclusion_interior_fracs", [0.5])
    fracs = jnp.array(interior_fracs + [1.0])  # (F,)
    F = fracs.shape[0]

    # points[s, f, k] = center_s + frac_f * radii[s, k] * direction_k
    points = (
        centers[:, None, None, :]  # (S, 1, 1, 2)
        + fracs[None, :, None, None]  # (1, F, 1, 1)
        * radii[:, None, :, None]  # (S, 1, K, 1)
        * directions[None, None, :, :]  # (1, 1, K, 2)
    )  # (S, F, K, 2)

    # Flatten the F*K sample dimension for joint processing
    points_flat = points.reshape(S, F * K, 2)  # (S, F*K, 2)

    # diff[s, t, p] = points[s, p] - centers[t]
    diff = points_flat[:, None, :, :] - centers[None, :, None, :]  # (S, S, F*K, 2)

    # Distance and angle from center_t to each sample point of s
    dist, alpha = _dist_and_angle(diff)  # (S, S, F*K)

    # Convert angle to fractional index in [0, K) and linearly interpolate t's radii
    delta_theta = 2 * jnp.pi / K
    frac_idx = (alpha % (2 * jnp.pi)) / delta_theta  # (S, S, F*K)

    idx_lo = jnp.floor(frac_idx).astype(jnp.int32) % K  # (S, S, F*K)
    idx_hi = (idx_lo + 1) % K  # (S, S, F*K)
    w_hi = frac_idx - jnp.floor(frac_idx)  # (S, S, F*K)

    t_range = jnp.arange(S)  # (S,)
    r_lo = radii[t_range[None, :, None], idx_lo]  # (S, S, F*K)
    r_hi = radii[t_range[None, :, None], idx_hi]  # (S, S, F*K)
    r_interp = (1.0 - w_hi) * r_lo + w_hi * r_hi  # (S, S, F*K)

    offset = input_params.get("exclusion_offset", 0.0)
    # Penalise when a sample point of s lies inside t  (dist < r_interp + offset)
    violations = jnp.where(
        mask[:, :, None], jnp.maximum(0.0, r_interp + offset - dist), 0.0
    )
    return jnp.sum(violations**2)


def _multi_term_star_enclosure(optim_vars, input_params):
    """Star-vs-star enclosure: boundary of inner sets must stay inside outer sets.

    For each (inner, outer) pair indicated by input_params["enclosure_mask"],
    penalises boundary points of the inner set that lie outside the star domain
    of the outer set — i.e. when their distance from center_outer exceeds the
    outer set's radius interpolated at that angle.

    optim_vars keys: "centers" (S, 2), "radii" (S, K)
    input_params keys: "angles" (K,), "enclosure_mask" (S, S) bool
      enclosure_mask[inner, outer] = True  →  inner must be inside outer
    """
    centers = optim_vars["centers"]  # (S, 2)
    radii = optim_vars["radii"]  # (S, K)
    angles = input_params["angles"]  # (K,)
    mask = input_params["enclosure_mask"]  # (S, S) bool

    S, K = radii.shape

    # Boundary points of every set: (S, K, 2)
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)  # (K, 2)
    points = (
        centers[:, None, :] + radii[:, :, None] * directions[None, :, :]
    )  # (S, K, 2)

    # diff[inner, outer, k] = points[inner, k] - centers[outer]
    diff = points[:, None, :, :] - centers[None, :, None, :]  # (S, S, K, 2)

    # Distance and angle from center_outer to each boundary point of inner
    dist, alpha = _dist_and_angle(diff)  # (S, S, K) each

    # Linearly interpolate outer's radius at that angle
    delta_theta = 2 * jnp.pi / K
    frac_idx = (alpha % (2 * jnp.pi)) / delta_theta  # (S, S, K) in [0, K)

    idx_lo = jnp.floor(frac_idx).astype(jnp.int32) % K  # (S, S, K)
    idx_hi = (idx_lo + 1) % K
    w_hi = frac_idx - jnp.floor(frac_idx)

    # r_lo[inner, outer, k] = radii[outer, idx_lo[inner, outer, k]]
    outer_range = jnp.arange(S)  # (S,)
    r_lo = radii[outer_range[None, :, None], idx_lo]  # (S, S, K)
    r_hi = radii[outer_range[None, :, None], idx_hi]
    r_interp = (1.0 - w_hi) * r_lo + w_hi * r_hi  # (S, S, K)

    offset = input_params.get("enclosure_offset", 0.0)
    # Violation: boundary point of inner is OUTSIDE outer minus offset  →  dist > r_interp - offset
    violations = jnp.where(
        mask[:, :, None],
        jnp.maximum(0.0, dist - (r_interp - offset)),
        0.0,
    )
    return jnp.sum(violations**2)


def _multi_term_target_area(optim_vars, input_params):
    """Penalises deviation of each set's area from its target.

    Sets with target_areas[s] = nan are ignored.

    The area of a star polygon with K uniformly-spaced radii r_k is:
        A = (1/2) sin(2π/K) Σ_k r_k · r_{k+1}

    optim_vars keys: "radii" (S, K)
    input_params keys: "target_areas" (S,)  — nan where unspecified
    """
    radii = optim_vars["radii"]  # (S, K)
    target = input_params["target_areas"]  # (S,)
    K = radii.shape[1]
    delta_theta = 2 * jnp.pi / K
    areas = (
        0.5
        * jnp.sin(delta_theta)
        * jnp.sum(radii * jnp.roll(radii, -1, axis=1), axis=1)
    )  # (S,)
    has_target = jnp.isfinite(target)
    # Replace nan targets with current area so the true branch stays finite;
    # without this, (areas - nan)**2 = nan and 0*nan = nan in the backward pass.
    safe_target = jnp.where(has_target, target, areas)
    return jnp.sum(jnp.where(has_target, (areas - safe_target) ** 2, 0.0))


def _build_exclusion_mask(S, enclosures):
    """Build a boolean (S, S) exclusion mask.

    Exclusion is disabled for (A, B) if and only if A and B have a common
    descendant in the directed enclosure graph (edges go from outer to inner;
    a node is considered a descendant of itself).
    """
    import networkx as nx

    mask = np.ones((S, S), dtype=bool)
    np.fill_diagonal(mask, False)

    if not enclosures:
        return mask

    G = nx.DiGraph()
    G.add_nodes_from(range(S))
    for inner, outer in enclosures:
        G.add_edge(outer, inner)

    desc = {s: nx.descendants(G, s) | {s} for s in range(S)}

    for a in range(S):
        for b in range(a + 1, S):
            if desc[a] & desc[b]:
                mask[a, b] = False
                mask[b, a] = False

    return mask


def _svg_configuration_star_only(snapshots, input_params, size):
    """SVG configuration for pure star domains (no underlying circles)."""
    angles = input_params["angles"]
    S = snapshots[0][1]["centers"].shape[0]

    # Compute bounding box from all boundary points across all frames
    all_x, all_y = [], []
    for _, v in snapshots:
        centers = np.array(v["centers"])
        radii = np.array(v["radii"])
        for s in range(len(centers)):
            bx = centers[s, 0] + radii[s] * np.cos(angles)
            by = centers[s, 1] + radii[s] * np.sin(angles)
            all_x.extend(bx.tolist())
            all_y.extend(by.tolist())

    x_min, y_min = min(all_x), min(all_y)
    span = max(max(all_x) - x_min, max(all_y) - y_min)
    margin = span * 0.05
    x_min -= margin
    y_max = y_min + span + 2 * margin
    span += 2 * margin

    def to_svg(x, y):
        return (x - x_min) / span * size, (y_max - y) / span * size

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
    return elements


def _radius_from_target_area(target_area, K):
    """Uniform radius r such that a regular K-gon star polygon has the given area."""
    factor = 0.5 * K * np.sin(2 * np.pi / K)  # ≈ π for large K
    return float(np.sqrt(target_area / factor))


def optimize_star_domains(
    S,
    initial_centers,
    k_angles=64,
    target_areas=None,
    initial_radius=1.0,
    weight_target_area=20.0,
    weight_area=1.0,
    weight_perimeter=0.5,
    weight_exclusion=10.0,
    weight_enclosure=20.0,
    weight_smoothness=1.0,
    enclosures=None,
    exclusion_offset=0.1,
    enclosure_offset=0.1,
    exclusion_interior_fracs=None,
    optim_config=None,
    callback=None,
):
    """Optimise pure star domains with no underlying circles.

    Unlike optimize_star_vs_star, there are no member circles — the loss
    is driven entirely by geometric constraints (enclosure, exclusion) and
    optional per-set area targets.

    For sets with a target area the area term pulls toward that value; for
    sets without one the plain area-minimisation term acts as a regulariser
    (making unconstrained outer sets hug their contents as tightly as the
    enclosure constraints allow).

    Args:
        S: number of star domains.
        initial_centers: (S, 2) starting centers for each domain.
        k_angles: angular resolution of each boundary polygon.
        target_areas: list of S values (float or None). None means no area
            target for that set; its area is then minimised by weight_area.
        initial_radius: fallback starting radius for sets without a target area.
        weight_target_area: weight for the target-area penalty.
        weight_area: weight for the area-minimisation regulariser (applied to
            all sets; dominated by target_area term for sets with targets).
        weight_perimeter: weight for the perimeter-minimisation regulariser.
        weight_exclusion: weight for star-vs-star exclusion (all pairs s≠t).
            Set to 0 for purely nested layouts.
        weight_enclosure: weight for star-vs-star enclosure constraints.
        weight_smoothness: weight for adjacent-radii smoothness penalty.
        enclosures: list of (inner_idx, outer_idx) pairs.
        exclusion_offset: minimum gap enforced between non-nested boundaries.
            A positive value pushes boundaries apart beyond mere non-overlap.
        enclosure_offset: minimum inset enforced for enclosure constraints.
            A positive value requires the inner boundary to stay at least this
            far inside the outer boundary.
        optim_config: OptimConfig; uses defaults when None.
        callback: optional iteration callback.

    Returns:
        (results, history, problem) where results is a list of S dicts with
        "center" (2,), "radii" (K,), "angles" (K,).
    """

    if exclusion_interior_fracs is None:
        exclusion_interior_fracs = [0.5]
    angles = np.linspace(0, 2 * np.pi, k_angles, endpoint=False).astype(np.float32)
    initial_centers = np.asarray(initial_centers, dtype=np.float32)  # (S, 2)

    # Build per-set initial radii from target areas where available
    targets_raw = target_areas if target_areas is not None else [None] * S
    target_arr = np.array(
        [t if t is not None else np.nan for t in targets_raw], dtype=np.float32
    )  # (S,) with nan for unspecified

    initial_radii = np.zeros((S, k_angles), dtype=np.float32)
    for s in range(S):
        r0 = (
            _radius_from_target_area(target_arr[s], k_angles)
            if np.isfinite(target_arr[s])
            else initial_radius
        )
        initial_radii[s] = r0

    # Enclosure mask
    enclosure_mask = np.zeros((S, S), dtype=bool)
    for inner, outer in enclosures or []:
        enclosure_mask[inner, outer] = True

    # Exclusion is disabled for pairs in a containment relationship
    exclusion_mask = _build_exclusion_mask(S, enclosures)

    input_parameters = {
        "angles": angles,
        "target_areas": target_arr,
        "enclosure_mask": enclosure_mask,
        "exclusion_mask": exclusion_mask,
        "exclusion_offset": float(exclusion_offset),
        "enclosure_offset": float(enclosure_offset),
        "exclusion_interior_fracs": exclusion_interior_fracs,
    }

    def initialize(_, seed):
        return {
            "centers": initial_centers.copy(),
            "radii": initial_radii.copy(),
        }

    terms = [
        ObjectiveTerm("target_area", _multi_term_target_area, weight_target_area),
        ObjectiveTerm("star_excl", _multi_term_star_exclusion, weight_exclusion),
        ObjectiveTerm("star_enclose", _multi_term_star_enclosure, weight_enclosure),
        ObjectiveTerm("min_radius", _multi_term_min_radius, 10.0),
        ObjectiveTerm("smoothness", _multi_term_smoothness, weight_smoothness),
        ObjectiveTerm("area", _multi_term_area, weight_area),
        ObjectiveTerm("perimeter", _multi_term_perimeter, weight_perimeter),
    ]
    _multi_term_star_exclusion(initialize(input_parameters, 0), input_parameters)
    problem = OptimizationProblemTemplate(
        terms=terms,
        initialize=initialize,
        svg_configuration=_svg_configuration_star_only,
    ).instantiate(input_parameters)

    optim_vars, history = problem.optimize(optim_config, callback=callback)

    results = [
        {
            "center": np.array(optim_vars["centers"][s]),
            "radii": np.array(optim_vars["radii"][s]),
            "angles": angles,
        }
        for s in range(S)
    ]
    return results, history, problem


def optimize_star_vs_star(
    circles,
    sets,
    k_angles=32,
    weight_area=1.0,
    weight_perimeter=1.0,
    weight_exclusion=10.0,
    weight_enclosure=10.0,
    weight_smoothness=1.0,
    offsets=0.1,
    enclosures=None,
    optim_config=None,
    callback=None,
):
    """Like optimize_multiple_radially_convex_sets, but with star-vs-star terms.

    Swaps the circle-based exclusion for a star-vs-star exclusion, and adds an
    optional star-vs-star enclosure term for nested-set constraints.

    Args:
        circles: (N, 3) array [cx, cy, r].
        sets: list of S index subsets into circles.
        k_angles: number of angular samples per boundary.
        weight_area, weight_perimeter, weight_smoothness: loss weights.
        weight_exclusion: weight for the star-vs-star exclusion penalty
            (all pairs (s, t) with s ≠ t).
        weight_enclosure: weight for the star-vs-star enclosure penalty.
            Only active when `enclosures` is non-empty.
        offsets: padding added to circle radii in the enclosure term.
        enclosures: list of (inner_idx, outer_idx) pairs. Each pair means
            "the boundary of sets[inner_idx] must lie inside sets[outer_idx]."
            Default None (no enclosure constraint).
        optim_config: OptimConfig (uses defaults when None).
        callback: optional iteration callback.

    Returns:
        (results, history, problem) — same shape as optimize_multiple_radially_convex_sets.
    """
    circles_array = np.asarray(circles, dtype=np.float32)
    if circles_array.ndim == 1:
        circles_array = circles_array[None, :]
    N = len(circles_array)
    S = len(sets)
    angles = np.linspace(0, 2 * np.pi, k_angles, endpoint=False).astype(np.float32)

    membership = _build_membership(S, N, sets)
    initial_centers, initial_radii = _init_centers_and_radii(
        circles_array, sets, angles
    )
    offsets_array = np.broadcast_to(
        np.asarray(offsets, dtype=np.float32), (S, N)
    ).copy()

    # Build enclosure mask (S, S): [inner, outer]
    enclosure_mask = np.zeros((S, S), dtype=bool)
    for inner, outer in enclosures or []:
        enclosure_mask[inner, outer] = True

    # Exclusion is disabled for pairs in a containment relationship
    exclusion_mask = _build_exclusion_mask(S, enclosures)

    input_parameters = {
        "circles": circles_array,
        "angles": angles,
        "membership": membership,
        "offsets": offsets_array,
        "enclosure_mask": enclosure_mask,
        "exclusion_mask": exclusion_mask,
    }

    def initialize(_, seed):
        return {"centers": initial_centers, "radii": initial_radii}

    terms = [
        ObjectiveTerm("enclosure", _multi_term_enclosure, 10.0),
        ObjectiveTerm("star_excl", _multi_term_star_exclusion, weight_exclusion),
        ObjectiveTerm("star_enclose", _multi_term_star_enclosure, weight_enclosure),
        ObjectiveTerm("min_radius", _multi_term_min_radius, 10.0),
        ObjectiveTerm("smoothness", _multi_term_smoothness, weight_smoothness),
        ObjectiveTerm("area", _multi_term_area, weight_area),
        ObjectiveTerm("perimeter", _multi_term_perimeter, weight_perimeter),
    ]

    problem = OptimizationProblemTemplate(
        terms=terms,
        initialize=initialize,
        svg_configuration=_svg_configuration_fixed,
    ).instantiate(input_parameters)

    optim_vars, history = problem.optimize(optim_config, callback=callback)

    results = [
        {
            "center": np.array(optim_vars["centers"][s]),
            "radii": np.array(optim_vars["radii"][s]),
            "angles": angles,
        }
        for s in range(S)
    ]
    return results, history, problem
