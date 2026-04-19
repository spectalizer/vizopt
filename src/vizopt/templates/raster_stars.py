import jax
import jax.numpy as jnp
import numpy as np

from vizopt.base import ObjectiveTerm, OptimizationProblemTemplate
from vizopt.components.stars import (
    BSpline,
    Discrete,
    Fourier,
    StarRepresentation,
    _multi_term_area,
    _multi_term_min_radius,
    _multi_term_perimeter,
    _multi_term_smoothness,
)
from vizopt.components.bspline_stars import (
    raster_collision_loss_bspline,
)
from vizopt.templates.star_vs_star import (
    _build_exclusion_mask,
    _multi_term_star_enclosure,
    _multi_term_target_area,
    _radius_from_target_area,
)


# ---------------------------------------------------------------------------
# Pixel grid
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Discrete (piecewise-linear) rasterization and loss
# ---------------------------------------------------------------------------


def soft_rasterize_star(
    centers,
    radii,
    grid_xy,
    temperature=0.05,
    offset=0.0,
):
    """Soft-rasterize n_sets star domains onto a pixel grid.

    For each pixel and each domain, computes a soft membership value in [0, 1]
    using a sigmoid of (r_interp + offset - dist) / temperature.

    Args:
        centers: (n_sets, 2) domain centers.
        radii: (n_sets, K) domain radii at K uniformly-spaced angles.
        grid_xy: (H, W, 2) pixel centre coordinates.
        temperature: sigmoid sharpness; smaller → harder boundary.
        offset: outward shift applied to the boundary before the sigmoid;
            positive values inflate the effective domain, enforcing a gap
            between non-overlapping domains.

    Returns:
        masks: (n_sets, H, W) soft membership values in (0, 1).
    """
    _, K = radii.shape

    # diff[s, h, w] = grid_xy[h, w] - centers[s]  →  (n_sets, H, W, 2)
    diff = grid_xy[None, :, :, :] - centers[:, None, None, :]

    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-12)  # (n_sets, H, W)

    # Double-where trick: safe for autodiff even at the origin
    rho2 = jnp.sum(diff**2, axis=-1)
    dx_safe = jnp.where(rho2 > 0, diff[..., 0], 1.0)
    dy_safe = jnp.where(rho2 > 0, diff[..., 1], 0.0)
    alpha = jnp.arctan2(dy_safe, dx_safe)  # (n_sets, H, W)

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
        "centers": (n_sets, 2)
        "radii":   (n_sets, K)
    input_params keys:
        "grid_xy":        (H, W, 2) pixel-centre coordinates
        "pixel_area":     scalar area of one pixel
        "exclusion_mask": (n_sets, n_sets) bool — True where (s, t) must not overlap
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


# ---------------------------------------------------------------------------
# Fourier rasterization and loss
# ---------------------------------------------------------------------------


def soft_rasterize_star_fourier(
    centers,
    fourier_coeffs,
    grid_xy,
    temperature=0.05,
    offset=0.0,
):
    """Soft-rasterize n_sets star domains using a Fourier boundary representation.

    Like `soft_rasterize_star` but r(θ) is evaluated directly from Fourier
    coefficients rather than linearly-interpolated discrete radii, giving exact
    gradients at any angle without interpolation artefacts.

    Args:
        centers: (n_sets, 2) domain centers.
        fourier_coeffs: (n_sets, 2M+1) Fourier coefficients [a₀, a₁, b₁, …].
        grid_xy: (H, W, 2) pixel centre coordinates.
        temperature: sigmoid sharpness; smaller → harder boundary.
        offset: outward boundary shift (same semantics as `soft_rasterize_star`).

    Returns:
        masks: (n_sets, H, W) soft membership values in (0, 1).
    """
    M = (fourier_coeffs.shape[-1] - 1) // 2

    diff = grid_xy[None, :, :, :] - centers[:, None, None, :]
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-12)

    rho2 = jnp.sum(diff**2, axis=-1)
    dx_safe = jnp.where(rho2 > 0, diff[..., 0], 1.0)
    dy_safe = jnp.where(rho2 > 0, diff[..., 1], 0.0)
    alpha = jnp.arctan2(dy_safe, dx_safe)  # (n_sets, H, W)

    def _r_fourier(coeffs_s, alpha_s):
        # coeffs_s: (2M+1,)  alpha_s: (H, W)
        ks = jnp.arange(1, M + 1)                        # (M,)
        theta = ks[:, None, None] * alpha_s[None, :, :]  # (M, H, W)
        r = coeffs_s[0]
        r = r + jnp.einsum("m,mhw->hw", coeffs_s[1::2], jnp.cos(theta))
        r = r + jnp.einsum("m,mhw->hw", coeffs_s[2::2], jnp.sin(theta))
        return r

    r_interp = jax.vmap(_r_fourier)(fourier_coeffs, alpha)  # (n_sets, H, W)
    return jax.nn.sigmoid((r_interp + offset - dist) / temperature)


def raster_collision_loss_fourier(optim_vars, input_params):
    """Raster-based pairwise collision loss using a Fourier boundary representation.

    Same semantics as `raster_collision_loss` but reads ``fourier_coeffs`` from
    *optim_vars* instead of ``radii``.

    optim_vars keys:
        "centers":        (n_sets, 2)
        "fourier_coeffs": (n_sets, 2M+1)
    input_params keys: same as `raster_collision_loss`.
    """
    centers = optim_vars["centers"]
    fourier_coeffs = optim_vars["fourier_coeffs"]
    grid_xy = input_params["grid_xy"]
    pixel_area = input_params["pixel_area"]
    mask = input_params["exclusion_mask"]
    temperature = input_params.get("temperature", 0.05)
    exclusion_offset = input_params.get("exclusion_offset", 0.0)

    masks = soft_rasterize_star_fourier(
        centers, fourier_coeffs, grid_xy, temperature, exclusion_offset
    )
    HW = grid_xy.shape[0] * grid_xy.shape[1]
    masks_flat = masks.reshape(masks.shape[0], HW)
    overlap = pixel_area * (masks_flat @ masks_flat.T)
    return jnp.sum(jnp.where(mask, overlap**2, 0.0))


_RASTER_COLLISION = {
    Discrete: raster_collision_loss,
    Fourier:  raster_collision_loss_fourier,
    BSpline:  raster_collision_loss_bspline,
}


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------


def optimize_star_domains_raster(
    n_sets,
    initial_centers,
    representation: StarRepresentation = None,
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
    exclusion_offset=0.1,
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
        n_sets: number of star domains.
        initial_centers: (n_sets, 2) starting centers.
        representation: a ``StarRepresentation`` instance (``Discrete``,
            ``Fourier``, or ``BSpline``) that controls the boundary
            parametrisation and its hyper-parameters. Defaults to
            ``Discrete(k_angles=64)``.
        target_areas: list of n_sets values (float or None).
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
        optim_config: optimization config dict; uses defaults when None.
        callback: optional iteration callback.

    Returns:
        (results, history, problem) where results is a list of n_sets dicts with
        keys ``"center"`` (2,), ``"radii"`` (K,), ``"angles"`` (K,), plus any
        representation-specific extras (e.g. ``"fourier_coeffs"`` or
        ``"bspline_ctrl"``).
    """
    if representation is None:
        representation = Discrete()

    k_angles = representation.k_angles
    angles = np.linspace(0, 2 * np.pi, k_angles, endpoint=False).astype(np.float32)
    angles_jnp = jnp.array(angles)
    initial_centers = np.asarray(initial_centers, dtype=np.float32)

    targets_raw = target_areas if target_areas is not None else [None] * n_sets
    target_arr = np.array(
        [t if t is not None else np.nan for t in targets_raw], dtype=np.float32
    )

    initial_radii = np.zeros((n_sets, k_angles), dtype=np.float32)
    for s in range(n_sets):
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
    print(
        f"Pixel grid: {grid_xy.shape[0]}x{grid_xy.shape[1]}, pixel_area={pixel_area:.4f}"
    )

    enclosure_mask = np.zeros((n_sets, n_sets), dtype=bool)
    for inner, outer in enclosures or []:
        enclosure_mask[inner, outer] = True
    exclusion_mask = _build_exclusion_mask(n_sets, enclosures)

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

    init_vars = representation.initialize_vars(n_sets, initial_radii, initial_centers)

    def initialize(*_):
        return {k: v.copy() for k, v in init_vars.items()}

    def wrap(fn):
        return representation.wrap(fn, angles_jnp)

    terms = [
        ObjectiveTerm("target_area",  wrap(_multi_term_target_area),    weight_target_area),
        ObjectiveTerm("raster_excl",  _RASTER_COLLISION[type(representation)], weight_exclusion),
        ObjectiveTerm("star_enclose", wrap(_multi_term_star_enclosure), weight_enclosure),
        ObjectiveTerm("min_radius",   wrap(_multi_term_min_radius),     10.0),
        ObjectiveTerm("smoothness",   wrap(_multi_term_smoothness),     weight_smoothness),
        ObjectiveTerm("area",         wrap(_multi_term_area),           weight_area),
        ObjectiveTerm("perimeter",    wrap(_multi_term_perimeter),      weight_perimeter),
    ]

    problem = OptimizationProblemTemplate(
        terms=terms,
        initialize=initialize,
        svg_configuration=representation.make_svg_configuration(),
    ).instantiate(input_parameters)

    optim_vars, history = problem.optimize(optim_config, callback=callback)

    radii_arr = np.array(representation.to_radii(optim_vars, angles_jnp))
    results = [
        {
            "center": np.array(optim_vars["centers"][s]),
            "radii": radii_arr[s],
            "angles": angles,
            **representation.extra_results(s, optim_vars),
        }
        for s in range(n_sets)
    ]
    return results, history, problem
