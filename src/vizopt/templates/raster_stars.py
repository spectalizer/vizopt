import jax
import jax.numpy as jnp
import numpy as np

from vizopt.base import ObjectiveTerm, OptimConfig, OptimizationProblemTemplate
from vizopt.radially_convex import (
    _multi_term_area,
    _multi_term_min_radius,
    _multi_term_perimeter,
    _multi_term_smoothness,
)
from vizopt.templates.star_vs_star import (
    _build_exclusion_mask,
    _multi_term_star_enclosure,
    _multi_term_target_area,
    _radius_from_target_area,
    _svg_configuration_star_only,
)


def make_pixel_grid(x_min, x_max, y_min, y_max, resolution):
    """Build a (H, W, 2) grid of pixel-centre coordinates.

    Args:
        x_min, x_max, y_min, y_max: scene bounding box.
        resolution: number of pixels along the longer axis.

    Returns:
        grid_xy: (H, W, 2) float32 array.
        pixel_area: scalar area of one pixel.
    """
    span_x = x_max - x_min
    span_y = y_max - y_min
    if span_x >= span_y:
        W = resolution
        H = max(1, int(round(resolution * span_y / span_x)))
    else:
        H = resolution
        W = max(1, int(round(resolution * span_x / span_y)))

    xs = np.linspace(x_min, x_max, W, endpoint=False) + (x_max - x_min) / (2 * W)
    ys = np.linspace(y_min, y_max, H, endpoint=False) + (y_max - y_min) / (2 * H)
    gx, gy = np.meshgrid(xs, ys)
    grid_xy = np.stack([gx, gy], axis=-1).astype(np.float32)  # (H, W, 2)
    pixel_area = float((x_max - x_min) * (y_max - y_min) / (H * W))
    return grid_xy, pixel_area


def soft_rasterize_star(
    centers,
    radii,
    grid_xy,
    temperature=0.05,
    offset=0.0,
):
    """Soft-rasterize S star domains onto a pixel grid.

    For each pixel and each domain, computes a soft membership value in [0, 1]
    using a sigmoid of (r_interp + offset - dist) / temperature.

    Args:
        centers: (S, 2) domain centers.
        radii: (S, K) domain radii at K uniformly-spaced angles.
        grid_xy: (H, W, 2) pixel centre coordinates.
        temperature: sigmoid sharpness; smaller → harder boundary.
        offset: outward shift applied to the boundary before the sigmoid;
            positive values inflate the effective domain, enforcing a gap
            between non-overlapping domains.

    Returns:
        masks: (S, H, W) soft membership values in (0, 1).
    """
    S, K = radii.shape

    # diff[s, h, w] = grid_xy[h, w] - centers[s]  →  (S, H, W, 2)
    diff = grid_xy[None, :, :, :] - centers[:, None, None, :]

    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-12)  # (S, H, W)

    # Double-where trick: safe for autodiff even at the origin
    rho2 = jnp.sum(diff**2, axis=-1)
    dx_safe = jnp.where(rho2 > 0, diff[..., 0], 1.0)
    dy_safe = jnp.where(rho2 > 0, diff[..., 1], 0.0)
    alpha = jnp.arctan2(dy_safe, dx_safe)  # (S, H, W)

    def _interp_one(r_s, alpha_s):
        delta_theta = 2 * jnp.pi / K
        frac_idx = (alpha_s % (2 * jnp.pi)) / delta_theta
        idx_lo = jnp.floor(frac_idx).astype(jnp.int32) % K
        idx_hi = (idx_lo + 1) % K
        w_hi = frac_idx - jnp.floor(frac_idx)
        return (1.0 - w_hi) * r_s[idx_lo] + w_hi * r_s[idx_hi]

    r_interp = jax.vmap(_interp_one)(radii, alpha)  # (S, H, W)

    return jax.nn.sigmoid((r_interp + offset - dist) / temperature)


def raster_collision_loss(optim_vars, input_params):
    """Raster-based pairwise collision loss for star domains.

    For each pair (s, t) where exclusion_mask[s, t] is True, penalises the
    overlap area computed by soft rasterization.

    optim_vars keys:
        "centers": (S, 2)
        "radii":   (S, K)
    input_params keys:
        "grid_xy":        (H, W, 2) pixel-centre coordinates
        "pixel_area":     scalar area of one pixel
        "exclusion_mask": (S, S) bool — True where (s, t) must not overlap
    Optional input_params keys:
        "temperature":      sigmoid temperature (default 0.05)
        "exclusion_offset": outward boundary shift enforcing a minimum gap
                            between domains (default 0.0)
    """
    centers = optim_vars["centers"]
    radii = optim_vars["radii"]
    grid_xy = input_params["grid_xy"]
    pixel_area = input_params["pixel_area"]
    mask = input_params["exclusion_mask"]
    temperature = input_params.get("temperature", 0.05)
    exclusion_offset = input_params.get("exclusion_offset", 0.0)

    masks = soft_rasterize_star(centers, radii, grid_xy, temperature, exclusion_offset)

    HW = grid_xy.shape[0] * grid_xy.shape[1]
    masks_flat = masks.reshape(masks.shape[0], HW)  # (S, HW)
    overlap = pixel_area * (masks_flat @ masks_flat.T)  # (S, S)

    return jnp.sum(jnp.where(mask, overlap**2, 0.0))


def optimize_star_domains_raster(
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
    enclosure_offset=0.1,
    exclusion_offset=0.0,
    grid_resolution=128,
    grid_margin=0.5,
    temperature=0.05,
    optim_config=None,
    callback=None,
):
    """Optimise pure star domains using a raster-based collision loss for exclusion.

    Drop-in replacement for `star_vs_star.optimize_star_domains` that swaps the
    analytical `_multi_term_star_exclusion` for `raster_collision_loss`: pairwise
    overlap is measured as the pixel-wise product of soft domain masks, giving
    well-defined gradients even for complete overlaps.

    The enclosure constraint remains analytical (`_multi_term_star_enclosure`).
    The pixel grid is built once from the initial configuration and held fixed
    throughout optimisation.

    Args:
        S: number of star domains.
        initial_centers: (S, 2) starting centers.
        k_angles: angular resolution of each boundary polygon.
        target_areas: list of S values (float or None).
        initial_radius: fallback starting radius for sets without a target area.
        weight_target_area: weight for target-area penalty.
        weight_area: weight for area-minimisation regulariser.
        weight_perimeter: weight for perimeter-minimisation regulariser.
        weight_exclusion: weight for raster collision loss.
        weight_enclosure: weight for analytical star-vs-star enclosure.
        weight_smoothness: weight for adjacent-radii smoothness penalty.
        enclosures: list of (inner_idx, outer_idx) pairs.
        enclosure_offset: minimum inset for enclosure constraints.
        exclusion_offset: outward boundary shift for raster collision; positive
            values inflate each domain's soft mask, enforcing a minimum gap
            between non-overlapping domains.
        grid_resolution: pixels along the longer axis of the bounding box.
        grid_margin: extra margin (in scene units) around the bounding box.
        temperature: sigmoid sharpness for soft rasterization.
        optim_config: OptimConfig; uses defaults when None.
        callback: optional iteration callback.

    Returns:
        (results, history, problem) where results is a list of S dicts with
        "center" (2,), "radii" (K,), "angles" (K,).
    """
    angles = np.linspace(0, 2 * np.pi, k_angles, endpoint=False).astype(np.float32)
    initial_centers = np.asarray(initial_centers, dtype=np.float32)

    targets_raw = target_areas if target_areas is not None else [None] * S
    target_arr = np.array(
        [t if t is not None else np.nan for t in targets_raw], dtype=np.float32
    )

    initial_radii = np.zeros((S, k_angles), dtype=np.float32)
    for s in range(S):
        r0 = (
            _radius_from_target_area(target_arr[s], k_angles)
            if np.isfinite(target_arr[s])
            else initial_radius
        )
        initial_radii[s] = r0

    max_r = float(initial_radii.max())
    x_min = float(initial_centers[:, 0].min()) - max_r - grid_margin
    x_max = float(initial_centers[:, 0].max()) + max_r + grid_margin
    y_min = float(initial_centers[:, 1].min()) - max_r - grid_margin
    y_max = float(initial_centers[:, 1].max()) + max_r + grid_margin
    grid_xy, pixel_area = make_pixel_grid(x_min, x_max, y_min, y_max, grid_resolution)
    print(f"Pixel grid: {grid_xy.shape[0]}x{grid_xy.shape[1]}, pixel_area={pixel_area:.4f}")

    enclosure_mask = np.zeros((S, S), dtype=bool)
    for inner, outer in enclosures or []:
        enclosure_mask[inner, outer] = True

    exclusion_mask = _build_exclusion_mask(S, enclosures)

    input_parameters = {
        "angles": angles,
        "target_areas": target_arr,
        "enclosure_mask": enclosure_mask,
        "enclosure_offset": float(enclosure_offset),
        "grid_xy": grid_xy,
        "pixel_area": float(pixel_area),
        "exclusion_mask": exclusion_mask,
        "temperature": float(temperature),
        "exclusion_offset": float(exclusion_offset),
    }

    def initialize(_, seed):
        return {
            "centers": initial_centers.copy(),
            "radii": initial_radii.copy(),
        }

    terms = [
        ObjectiveTerm("target_area", _multi_term_target_area, weight_target_area),
        ObjectiveTerm("raster_excl", raster_collision_loss, weight_exclusion),
        ObjectiveTerm("star_enclose", _multi_term_star_enclosure, weight_enclosure),
        ObjectiveTerm("min_radius", _multi_term_min_radius, 10.0),
        ObjectiveTerm("smoothness", _multi_term_smoothness, weight_smoothness),
        ObjectiveTerm("area", _multi_term_area, weight_area),
        ObjectiveTerm("perimeter", _multi_term_perimeter, weight_perimeter),
    ]

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
