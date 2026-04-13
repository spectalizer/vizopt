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


def _multi_term_star_exclusion(optim_vars, input_params):
    """Star-vs-star exclusion: boundary of each set must not enter any other set's interior.

    For each pair (s, t) where exclusion_mask[s, t] is True, penalises boundary
    points of s that lie inside the star domain of t.

    optim_vars keys: "centers" (S, 2), "radii" (S, K)
    input_params keys: "angles" (K,), "exclusion_mask" (S, S) bool
    """
    centers = optim_vars["centers"]  # (S, 2)
    radii = optim_vars["radii"]  # (S, K)
    angles = input_params["angles"]  # (K,)
    mask = input_params["exclusion_mask"]  # (S, S) bool
    S, K = radii.shape

    # Boundary points for every set: (S, K, 2)
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)  # (K, 2)
    points = (
        centers[:, None, :] + radii[:, :, None] * directions[None, :, :]
    )  # (S, K, 2)

    # Vector from center_t to each boundary point of s:
    #   diff[s, t, k] = points[s, k] - centers[t]
    diff = points[:, None, :, :] - centers[None, :, None, :]  # (S, S, K, 2)

    # Distance from center_t to each boundary point of s
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-12)  # (S, S, K)

    # Normalise diff before atan2 to bound gradients when diff ≈ 0.
    # Adding 1e-7 to the x-component prevents arctan2(0, 0) from producing NaN
    # gradients: when diff=(0,0) the upstream gradient is 0 (no violation when
    # dist=0 < r_interp), so 0 * finite = 0 rather than 0 * NaN = NaN.
    diff_unit = diff / dist[..., None]  # (S, S, K, 2)
    alpha = jnp.arctan2(diff_unit[..., 1], diff_unit[..., 0] + 1e-7)  # (S, S, K)

    # Convert angle to fractional index in [0, K) and linearly interpolate t's radii
    delta_theta = 2 * jnp.pi / K
    frac_idx = (alpha % (2 * jnp.pi)) / delta_theta  # (S, S, K)

    idx_lo = jnp.floor(frac_idx).astype(jnp.int32) % K  # (S, S, K)
    idx_hi = (idx_lo + 1) % K  # (S, S, K)
    w_hi = frac_idx - jnp.floor(frac_idx)  # (S, S, K)

    # r_lo[s, t, k] = radii[t, idx_lo[s, t, k]]
    t_range = jnp.arange(S)  # (S,)
    r_lo = radii[t_range[None, :, None], idx_lo]  # (S, S, K)
    r_hi = radii[t_range[None, :, None], idx_hi]  # (S, S, K)
    r_interp = (1.0 - w_hi) * r_lo + w_hi * r_hi  # (S, S, K)

    # Penalise when boundary point of s is inside t  (dist < r_interp)
    violations = jnp.where(mask[:, :, None], jnp.maximum(0.0, r_interp - dist), 0.0)
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

    # Distance from center_outer to each boundary point of inner
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-12)  # (S, S, K)

    # Normalise diff before atan2; add 1e-7 to x to prevent arctan2(0,0) NaN
    # gradient (same reasoning as in _multi_term_star_exclusion above).
    diff_unit = diff / dist[..., None]  # (S, S, K, 2)
    alpha = jnp.arctan2(diff_unit[..., 1], diff_unit[..., 0] + 1e-7)  # (S, S, K)

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

    # Violation: boundary point of inner is OUTSIDE outer  →  dist > r_interp
    violations = jnp.where(
        mask[:, :, None],
        jnp.maximum(0.0, dist - r_interp),
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
    """Build a boolean (S, S) exclusion mask. ..."""
    mask = np.ones((S, S), dtype=bool)
    np.fill_diagonal(mask, False)
    for inner, outer in enclosures or []:
        mask[inner, outer] = False
        mask[outer, inner] = False
    return mask


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
        optim_config: OptimConfig; uses defaults when None.
        callback: optional iteration callback.

    Returns:
        (results, history, problem) where results is a list of S dicts with
        "center" (2,), "radii" (K,), "angles" (K,).
    """
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
