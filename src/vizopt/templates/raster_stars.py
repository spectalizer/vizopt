import jax
import jax.numpy as jnp
import numpy as np

from vizopt.base import (
    ObjectiveTerm,
    OptimizationProblem,
    OptimizationProblemTemplate,
    VizOptimizer,
)
from vizopt.components.bspline_stars import (
    raster_collision_loss_bspline,
)
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

    # Double-where trick: safe for autodiff even at the origin
    rho2 = jnp.sum(diff**2, axis=-1)
    dist = jnp.sqrt(rho2 + 1e-12)  # (n_sets, H, W)
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
    rho2 = jnp.sum(diff**2, axis=-1)
    dist = jnp.sqrt(rho2 + 1e-12)
    dx_safe = jnp.where(rho2 > 0, diff[..., 0], 1.0)
    dy_safe = jnp.where(rho2 > 0, diff[..., 1], 0.0)
    alpha = jnp.arctan2(dy_safe, dx_safe)  # (n_sets, H, W)

    def _r_fourier(coeffs_s, alpha_s):
        # coeffs_s: (2M+1,)  alpha_s: (H, W)
        ks = jnp.arange(1, M + 1)  # (M,)
        theta = ks[:, None, None] * alpha_s[None, :, :]  # (M, H, W)
        r = coeffs_s[0]
        r = r + jnp.einsum("m,mhw->hw", coeffs_s[1::2], jnp.cos(theta))
        r = r + jnp.einsum("m,mhw->hw", coeffs_s[2::2], jnp.sin(theta))
        return r

    r_interp = jax.vmap(_r_fourier)(fourier_coeffs, alpha)  # (n_sets, H, W)
    return jax.nn.sigmoid((r_interp + offset - dist) / temperature)


def raster_collision_loss_fourier(optim_vars, input_params):
    """Raster-based pairwise collision loss using a Fourier boundary representation.

    Same semantics as `raster_collision_loss` but reads `fourier_coeffs` from
    *optim_vars* instead of `radii`.

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
    Fourier: raster_collision_loss_fourier,
    BSpline: raster_collision_loss_bspline,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class RasterStarOptimizer(VizOptimizer):
    """Optimise pure star domains using a raster-based collision loss for exclusion.

    Drop-in replacement for :class:`~vizopt.templates.star_vs_star.StarDomainOptimizer`
    that swaps the analytical `star_excl` term for a pixel-wise raster overlap
    loss, giving well-defined gradients even for complete overlaps.

    The enclosure constraint remains analytical (:func:`_multi_term_star_enclosure`
    from :mod:`vizopt.templates.star_vs_star`). The pixel grid is built once from
    the initial configuration and held fixed throughout optimization.

    Args:
        n_sets: Number of star domains.
        initial_centers: `(n_sets, 2)` starting centers.
        representation: A :class:`~vizopt.components.stars.StarRepresentation`
            instance (`Discrete`, `Fourier`, or `BSpline`) that controls
            the boundary parametrisation. Defaults to `Discrete(k_angles=64)`.
        target_areas: List of n_sets values (float or None).
        initial_radius: Fallback starting radius for sets without a target area.
        weight_target_area: Weight for target-area penalty.
        weight_area: Weight for area-minimisation regulariser.
        weight_perimeter: Weight for perimeter-minimisation regulariser.
        weight_exclusion: Weight for raster collision loss.
        weight_enclosure: Weight for analytical star-vs-star enclosure.
        weight_smoothness: Weight for adjacent-radii smoothness penalty.
        enclosures: List of `(inner_idx, outer_idx)` pairs.
        enclosure_offset: Minimum inset for enclosure constraints.
        exclusion_offset: Outward boundary shift for raster collision; positive
            values inflate each domain's soft mask, enforcing a minimum gap.
        grid_resolution: Pixels along the longer axis of the bounding box.
        grid_margin: Extra margin (in scene units) around the bounding box.
        temperature: Sigmoid sharpness for soft rasterization.
    """

    def __init__(
        self,
        n_sets: int,
        initial_centers,
        *,
        representation: StarRepresentation = None,
        target_areas=None,
        initial_radius: float = 1.0,
        weight_target_area: float = 20.0,
        weight_area: float = 1.0,
        weight_perimeter: float = 0.5,
        weight_exclusion: float = 10.0,
        weight_enclosure: float = 20.0,
        weight_smoothness: float = 1.0,
        enclosures=None,
        enclosure_offset: float = 0.1,
        exclusion_offset: float = 0.1,
        grid_resolution: int = 128,
        grid_margin: float = 0.5,
        temperature: float = 0.05,
    ):
        self.n_sets = n_sets
        self.initial_centers = initial_centers
        self.representation = (
            representation if representation is not None else Discrete()
        )
        self.target_areas = target_areas
        self.initial_radius = initial_radius
        self.weight_target_area = weight_target_area
        self.weight_area = weight_area
        self.weight_perimeter = weight_perimeter
        self.weight_exclusion = weight_exclusion
        self.weight_enclosure = weight_enclosure
        self.weight_smoothness = weight_smoothness
        self.enclosures = enclosures
        self.enclosure_offset = enclosure_offset
        self.exclusion_offset = exclusion_offset
        self.grid_resolution = grid_resolution
        self.grid_margin = grid_margin
        self.temperature = temperature

    def _build_problem(self) -> OptimizationProblem:
        n_sets = self.n_sets
        representation = self.representation
        k_angles = representation.k_angles
        angles = np.linspace(0, 2 * np.pi, k_angles, endpoint=False).astype(np.float32)
        angles_jnp = jnp.array(angles)
        initial_centers = np.asarray(self.initial_centers, dtype=np.float32)

        targets_raw = (
            self.target_areas if self.target_areas is not None else [None] * n_sets
        )
        target_arr = np.array(
            [t if t is not None else np.nan for t in targets_raw], dtype=np.float32
        )

        initial_radii = np.zeros((n_sets, k_angles), dtype=np.float32)
        for s in range(n_sets):
            r0 = (
                _radius_from_target_area(target_arr[s], k_angles)
                if np.isfinite(target_arr[s])
                else self.initial_radius
            )
            initial_radii[s] = r0

        max_r = float(initial_radii.max())
        x_min = float(initial_centers[:, 0].min()) - max_r - self.grid_margin
        x_max = float(initial_centers[:, 0].max()) + max_r + self.grid_margin
        y_min = float(initial_centers[:, 1].min()) - max_r - self.grid_margin
        y_max = float(initial_centers[:, 1].max()) + max_r + self.grid_margin
        grid_xy, pixel_area = make_pixel_grid(
            x_min, x_max, y_min, y_max, self.grid_resolution
        )
        print(
            f"Pixel grid: {grid_xy.shape[0]}x{grid_xy.shape[1]}, pixel_area={pixel_area:.4f}"
        )

        enclosure_mask = np.zeros((n_sets, n_sets), dtype=bool)
        for inner, outer in self.enclosures or []:
            enclosure_mask[inner, outer] = True
        exclusion_mask = _build_exclusion_mask(n_sets, self.enclosures)

        input_parameters = {
            "angles": angles,
            "target_areas": target_arr,
            "enclosure_mask": enclosure_mask,
            "enclosure_offset": float(self.enclosure_offset),
            "grid_xy": grid_xy,
            "pixel_area": float(pixel_area),
            "exclusion_mask": exclusion_mask,
            "temperature": float(self.temperature),
            "exclusion_offset": float(self.exclusion_offset),
        }

        init_vars = representation.initialize_vars(
            n_sets, initial_radii, initial_centers
        )

        def initialize(*_):
            return {k: v.copy() for k, v in init_vars.items()}

        def wrap(fn):
            return representation.wrap(fn, angles_jnp)

        return OptimizationProblemTemplate(
            terms=[
                ObjectiveTerm(
                    "target_area",
                    wrap(_multi_term_target_area),
                    self.weight_target_area,
                ),
                ObjectiveTerm(
                    "raster_excl",
                    _RASTER_COLLISION[type(representation)],
                    self.weight_exclusion,
                ),
                ObjectiveTerm(
                    "star_enclose",
                    wrap(_multi_term_star_enclosure),
                    self.weight_enclosure,
                ),
                ObjectiveTerm("min_radius", wrap(_multi_term_min_radius), 10.0),
                ObjectiveTerm(
                    "smoothness", wrap(_multi_term_smoothness), self.weight_smoothness
                ),
                ObjectiveTerm("area", wrap(_multi_term_area), self.weight_area),
                ObjectiveTerm(
                    "perimeter", wrap(_multi_term_perimeter), self.weight_perimeter
                ),
            ],
            initialize=initialize,
            svg_configuration=representation.make_svg_configuration(),
        ).instantiate(input_parameters)

    @property
    def sets_(self) -> list[dict]:
        """Star boundary dicts from the last optimization result.

        Each dict has `"center"` (2,), `"radii"` (K,), `"angles"` (K,),
        plus any representation-specific extras.

        Raises:
            ValueError: If :meth:`optimize` has not been called yet.
        """
        if not hasattr(self, "result_"):
            raise ValueError("No result yet — call optimize() first.")
        optim_vars = self.result_.optim_vars
        angles = self.problem_.input_parameters["angles"]
        angles_jnp = jnp.array(angles)
        radii_arr = np.array(self.representation.to_radii(optim_vars, angles_jnp))
        return [
            {
                "center": np.array(optim_vars["centers"][s]),
                "radii": radii_arr[s],
                "angles": angles,
                **self.representation.extra_results(s, optim_vars),
            }
            for s in range(self.n_sets)
        ]
