"""B-spline boundary representation for star-shaped regions.

Uniform periodic cubic B-splines give C² smooth boundaries at O(1) per-pixel
evaluation cost (no trig). This module provides the math behind the
`BSpline` star representation (`components/stars.py`):

- Core math: `_interp_bspline`, `bspline_to_radii`, `_wrap_bspline_term`
- Raster soft-membership: `soft_rasterize_star_bspline`,
  `raster_collision_loss_bspline` (used by `RasterStarOptimizer`)
"""

import jax
import jax.numpy as jnp

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
        fn: loss function with signature `(optim_vars, input_params) -> scalar`
            that reads `optim_vars["radii"]`.
        angles: (K,) evaluation angles used to convert ctrl pts → radii.

    Returns:
        Wrapped function with the same signature that reads
        `optim_vars["bspline_ctrl"]` instead.
    """

    def wrapped(optim_vars, input_params):
        radii = bspline_to_radii(optim_vars["bspline_ctrl"], angles)
        return fn({**optim_vars, "radii": radii}, input_params)

    return wrapped


# ---------------------------------------------------------------------------
# Raster soft-membership (for raster-collision exclusion)
# ---------------------------------------------------------------------------


def soft_rasterize_star_bspline(
    centers: jnp.ndarray,
    ctrl_pts,
    grid_xy: jnp.ndarray,
    temperature=0.05,
    offset=0.0,
):
    """Soft-rasterize n_sets star domains using a B-spline boundary.

    Like `soft_rasterize_star` but r(θ) is a uniform periodic cubic B-spline,
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
    rho2 = jnp.sum(diff**2, axis=-1)
    dist = jnp.sqrt(rho2 + 1e-12)
    dx_safe = jnp.where(rho2 > 0, diff[..., 0], 1.0)
    dy_safe = jnp.where(rho2 > 0, diff[..., 1], 0.0)
    alpha = jnp.arctan2(dy_safe, dx_safe)  # (n_sets, H, W)

    r_interp = jax.vmap(_interp_bspline)(ctrl_pts, alpha)  # (n_sets, H, W)
    return jax.nn.sigmoid((r_interp + offset - dist) / temperature)


def raster_collision_loss_bspline(optim_vars, input_params):
    """Raster-based pairwise collision loss using a B-spline boundary.

    Same semantics as `raster_collision_loss` but reads `bspline_ctrl` from
    *optim_vars* instead of `radii`.

    optim_vars keys:
        "centers":      (n_sets, 2)
        "bspline_ctrl": (n_sets, P)
    input_params keys: same as `raster_collision_loss`.
    """
    centers = optim_vars["centers"]
    ctrl_pts = optim_vars["bspline_ctrl"]
    grid_xy = input_params["grid_xy"]
    pixel_area = jnp.asarray(input_params["pixel_area"])
    mask = jnp.asarray(input_params["exclusion_mask"])
    temperature = input_params.get("temperature", 0.05)
    exclusion_offset = input_params.get("exclusion_offset", 0.0)

    masks = soft_rasterize_star_bspline(
        centers, ctrl_pts, grid_xy, temperature, exclusion_offset
    )
    HW = grid_xy.shape[0] * grid_xy.shape[1]
    masks_flat = masks.reshape(masks.shape[0], HW)
    overlap = pixel_area * (masks_flat @ masks_flat.T)
    return jnp.sum(jnp.where(mask, overlap**2, 0.0))
