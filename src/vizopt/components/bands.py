"""Reusable JAX loss terms and helpers for convex vertical-band domains.

A convex region is exactly the area between a concave upper boundary y2(x)
and a convex lower boundary y1(x) over x in [x_min, x_max]. Here each region
is represented by `x_bounds = [x_min, x_max]` plus `upper` and `lower`
arrays of K values sampled at K uniformly-spaced columns:

    x_k = x_min + k/(K-1) * (x_max - x_min),  k = 0, ..., K-1

Unlike the star-shaped representation in :mod:`vizopt.components.stars`,
`x_bounds` is itself an optimization variable (not a fixed input parameter),
since a band's horizontal extent is generally not known in advance.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from jax import numpy as jnp

from ..utils import _SVG_SET_COLORS

_MIN_WIDTH = 0.1
_MIN_THICKNESS = 0.1


# ---------------------------------------------------------------------------
# Column / interpolation primitives
# ---------------------------------------------------------------------------


def _column_positions(x_bounds, K):
    """Uniformly-spaced x positions for each set's K boundary columns.

    Args:
        x_bounds: (S, 2) `[x_min, x_max]` per set.
        K: number of columns.

    Returns:
        (S, K) array of x positions.
    """
    t = jnp.linspace(0.0, 1.0, K)  # (K,)
    x_min = x_bounds[:, 0:1]  # (S, 1)
    x_max = x_bounds[:, 1:2]  # (S, 1)
    return x_min + t[None, :] * (x_max - x_min)


def _interp_band_bounds(x_bounds, upper, lower, px):
    """Interpolate target bands' upper/lower boundaries at given x positions.

    Args:
        x_bounds: (T, 2) target domains' `[x_min, x_max]`.
        upper: (T, K) target domains' upper boundary values.
        lower: (T, K) target domains' lower boundary values.
        px: (S, 1, P) or (S, T, P) query x positions, broadcast against the
            T target domains (axis 1).

    Returns:
        Tuple `(upper_interp, lower_interp, x_min, x_max)`. The first two
        are shaped `(S, T, P)`; the last two are `(1, T, 1)` and
        broadcastable against `px`. Query points outside `[x_min, x_max]`
        get the boundary column's value (clamped extrapolation) — callers
        combine this with an explicit x-range violation.
    """
    K = upper.shape[1]
    x_min = x_bounds[None, :, 0:1]  # (1, T, 1)
    x_max = x_bounds[None, :, 1:2]  # (1, T, 1)
    width = jnp.maximum(x_max - x_min, 1e-6)

    raw_idx = (px - x_min) / width * (K - 1)
    clipped = jnp.clip(raw_idx, 0.0, K - 1)
    idx_lo = jnp.minimum(jnp.floor(clipped).astype(jnp.int32), K - 2)
    idx_hi = idx_lo + 1
    w_hi = clipped - idx_lo

    T = upper.shape[0]
    t_idx = jnp.broadcast_to(jnp.arange(T)[None, :, None], idx_lo.shape)
    upper_interp = (1.0 - w_hi) * upper[t_idx, idx_lo] + w_hi * upper[t_idx, idx_hi]
    lower_interp = (1.0 - w_hi) * lower[t_idx, idx_lo] + w_hi * lower[t_idx, idx_hi]
    return upper_interp, lower_interp, x_min, x_max


def _band_areas(x_bounds: jnp.ndarray, upper: jnp.ndarray, lower: jnp.ndarray):
    """Per-set band area via the trapezoid rule over K uniformly-spaced columns.

    Args:
        x_bounds: (S, 2) `[x_min, x_max]` per set.
        upper: (S, K) upper boundary values.
        lower: (S, K) lower boundary values.

    Returns:
        (S,) array of areas.
    """
    K = upper.shape[1]
    width = x_bounds[:, 1] - x_bounds[:, 0]  # (S,)
    dx = width / (K - 1)
    weights = jnp.ones((K,)).at[0].set(0.5).at[-1].set(0.5)
    return dx * jnp.sum((upper - lower) * weights[None, :], axis=1)


# ---------------------------------------------------------------------------
# ObjectiveTerm compute functions
#
# optim_vars keys: "x_bounds" (S, 2), "upper" (S, K), "lower" (S, K)
# ---------------------------------------------------------------------------


def _multi_term_band_area(optim_vars, input_params):
    """Sum of band areas over all sets."""
    return jnp.sum(
        _band_areas(optim_vars["x_bounds"], optim_vars["upper"], optim_vars["lower"])
    )


def _multi_term_target_area(optim_vars, input_params):
    """Penalises deviation of each set's area from its target.

    Sets with `target_areas[s] = nan` are ignored.
    """
    areas = _band_areas(
        optim_vars["x_bounds"], optim_vars["upper"], optim_vars["lower"]
    )
    target = jnp.asarray(input_params["target_areas"])  # (S,)
    has_target = jnp.isfinite(target)
    # Replace nan targets with current area so the true branch stays finite;
    # without this, (areas - nan)**2 = nan and 0*nan = nan in the backward pass.
    safe_target = jnp.where(has_target, target, areas)
    return jnp.sum(jnp.where(has_target, (areas - safe_target) ** 2, 0.0))


def _multi_term_band_perimeter(optim_vars, input_params):
    """Sum of band perimeters over all sets (upper + lower chains + end caps)."""
    x_bounds = optim_vars["x_bounds"]  # (S, 2)
    upper = optim_vars["upper"]  # (S, K)
    lower = optim_vars["lower"]  # (S, K)
    K = upper.shape[1]

    xs = _column_positions(x_bounds, K)  # (S, K)
    dx = xs[:, 1:] - xs[:, :-1]  # (S, K-1)
    d_upper = upper[:, 1:] - upper[:, :-1]
    d_lower = lower[:, 1:] - lower[:, :-1]
    upper_len = jnp.sqrt(dx**2 + d_upper**2 + 1e-12)
    lower_len = jnp.sqrt(dx**2 + d_lower**2 + 1e-12)
    end_caps = jnp.abs(upper[:, 0] - lower[:, 0]) + jnp.abs(upper[:, -1] - lower[:, -1])
    return jnp.sum(upper_len) + jnp.sum(lower_len) + jnp.sum(end_caps)


def _multi_term_band_convexity(optim_vars, input_params):
    """Penalty for non-convex band shapes.

    Convexity requires the upper boundary to be concave and the lower
    boundary to be convex. Uses the discrete second difference at each
    interior column, normalised by squared column spacing (analogous to the
    edge-length normalisation in `stars._multi_term_convexity`), and the
    same `max(0, violation)**2 + alpha * max(0, violation)` penalty shape.
    """
    x_bounds = optim_vars["x_bounds"]  # (S, 2)
    upper = optim_vars["upper"]  # (S, K)
    lower = optim_vars["lower"]  # (S, K)
    alpha = input_params["convexity_alpha"]
    K = upper.shape[1]

    dx = (x_bounds[:, 1] - x_bounds[:, 0]) / (K - 1)  # (S,)
    dx_sq = jnp.maximum(dx**2, 1e-6)[:, None]  # (S, 1)

    d2_upper = upper[:, 2:] - 2.0 * upper[:, 1:-1] + upper[:, :-2]  # (S, K-2)
    d2_lower = lower[:, 2:] - 2.0 * lower[:, 1:-1] + lower[:, :-2]

    upper_violation = jnp.maximum(0.0, d2_upper / dx_sq)
    lower_violation = jnp.maximum(0.0, -d2_lower / dx_sq)

    return jnp.sum(upper_violation**2 + alpha * upper_violation) + jnp.sum(
        lower_violation**2 + alpha * lower_violation
    )


def _multi_term_band_smoothness(optim_vars, input_params):
    """Penalty for sharp changes between adjacent columns of upper/lower."""
    upper = optim_vars["upper"]  # (S, K)
    lower = optim_vars["lower"]  # (S, K)
    return jnp.sum((upper[:, 1:] - upper[:, :-1]) ** 2) + jnp.sum(
        (lower[:, 1:] - lower[:, :-1]) ** 2
    )


def _multi_term_min_thickness(optim_vars, input_params):
    """Penalty for band thickness (upper - lower) falling below the minimum."""
    upper = optim_vars["upper"]  # (S, K)
    lower = optim_vars["lower"]  # (S, K)
    return jnp.sum(jnp.maximum(0.0, _MIN_THICKNESS - (upper - lower)) ** 2)


def _multi_term_min_width(optim_vars, input_params):
    """Penalty for band width (x_max - x_min) falling below the minimum."""
    x_bounds = optim_vars["x_bounds"]  # (S, 2)
    width = x_bounds[:, 1] - x_bounds[:, 0]
    return jnp.sum(jnp.maximum(0.0, _MIN_WIDTH - width) ** 2)


def _multi_term_band_enclosure(optim_vars, input_params):
    """Band-vs-band enclosure: boundary of inner sets must stay inside outer sets.

    For each `(inner, outer)` pair indicated by `input_params["enclosure_mask"]`,
    penalises boundary points (upper and lower chains) of the inner set that
    lie outside the interpolated boundary of the outer set, in x or in y.

    optim_vars keys: "x_bounds" (S, 2), "upper" (S, K), "lower" (S, K)
    input_params keys: "enclosure_mask" (S, S) bool — `[inner, outer]`
      Optional: "enclosure_offset" (float), minimum inset from the outer
      boundary.
    """
    x_bounds = optim_vars["x_bounds"]  # (S, 2)
    upper = optim_vars["upper"]  # (S, K)
    lower = optim_vars["lower"]  # (S, K)
    mask = input_params["enclosure_mask"]  # (S, S)
    offset = input_params.get("enclosure_offset", 0.0)
    K = upper.shape[1]

    xs = _column_positions(x_bounds, K)  # (S, K)
    points_x = jnp.concatenate([xs, xs], axis=1)  # (S, 2K)
    points_y = jnp.concatenate([lower, upper], axis=1)  # (S, 2K)

    px = points_x[:, None, :]  # (S, 1, 2K)
    py = points_y[:, None, :]  # (S, 1, 2K)

    upper_interp, lower_interp, x_min, x_max = _interp_band_bounds(
        x_bounds, upper, lower, px
    )

    x_violation = jnp.maximum(
        0.0, jnp.maximum((x_min + offset) - px, px - (x_max - offset))
    )
    y_violation = jnp.maximum(
        0.0,
        jnp.maximum(py - (upper_interp - offset), (lower_interp + offset) - py),
    )
    total = x_violation + y_violation  # (S, S, 2K) — [inner, outer, point]

    violations = jnp.where(mask[:, :, None], total, 0.0)
    return jnp.sum(violations**2)


def _multi_term_band_exclusion(optim_vars, input_params):
    """Band-vs-band exclusion: boundary and interior of each set must not enter another.

    For each pair `(s, t)` where `exclusion_mask[s, t]` is True, penalises
    boundary and interior points of `s` that lie inside the band domain of
    `t`, using an AABB-style penetration depth (minimum distance to the
    nearest of the four bounding sides), the same style as
    `stars._multi_term_label_label_collision`.

    optim_vars keys: "x_bounds" (S, 2), "upper" (S, K), "lower" (S, K)
    input_params keys: "exclusion_mask" (S, S) bool
      Optional: "exclusion_offset" (float), minimum required gap.
      Optional: "exclusion_interior_fracs" (list[float]), height fractions in
        (0, 1) at which to sample interior points at each column. Defaults
        to [0.5].
    """
    x_bounds = optim_vars["x_bounds"]  # (S, 2)
    upper = optim_vars["upper"]  # (S, K)
    lower = optim_vars["lower"]  # (S, K)
    mask = input_params["exclusion_mask"]  # (S, S)
    offset = input_params.get("exclusion_offset", 0.0)
    interior_fracs = input_params.get("exclusion_interior_fracs", [0.5])
    K = upper.shape[1]

    xs = _column_positions(x_bounds, K)  # (S, K)
    fracs = jnp.array(list(interior_fracs) + [0.0, 1.0])  # (F,)
    F = fracs.shape[0]

    points_x = jnp.broadcast_to(xs[:, None, :], (xs.shape[0], F, K)).reshape(
        xs.shape[0], F * K
    )
    points_y = (
        lower[:, None, :] + fracs[None, :, None] * (upper - lower)[:, None, :]
    ).reshape(xs.shape[0], F * K)

    px = points_x[:, None, :]  # (S, 1, F*K)
    py = points_y[:, None, :]  # (S, 1, F*K)

    upper_interp, lower_interp, x_min, x_max = _interp_band_bounds(
        x_bounds, upper, lower, px
    )

    pen_left = px - x_min + offset
    pen_right = (x_max + offset) - px
    pen_bottom = py - lower_interp + offset
    pen_top = (upper_interp + offset) - py
    penetration = jnp.minimum(
        jnp.minimum(pen_left, pen_right), jnp.minimum(pen_bottom, pen_top)
    )
    violation = jnp.maximum(0.0, penetration)  # (S, S, F*K) — [source, target, point]

    violations = jnp.where(mask[:, :, None], violation, 0.0)
    return jnp.sum(violations**2)


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------


def _init_rectangle(initial_centers, half_width, half_height, k_columns):
    """Initial axis-aligned rectangle per set.

    Mirrors the star convention of initializing as a regular polygon of
    constant radius: here the initial band is a rectangle of constant
    height across all columns.

    Args:
        initial_centers: (S, 2) `[cx, cy]` per set.
        half_width: (S,) initial half-width per set.
        half_height: (S,) initial half-height per set.
        k_columns: number of columns K.

    Returns:
        Tuple `(x_bounds, upper, lower)` with shapes `(S, 2)`, `(S, K)`,
        `(S, K)`.
    """
    S = initial_centers.shape[0]
    cx = initial_centers[:, 0]
    cy = initial_centers[:, 1]
    x_bounds = np.stack([cx - half_width, cx + half_width], axis=1).astype(np.float32)
    upper = np.broadcast_to((cy + half_height)[:, None], (S, k_columns)).astype(
        np.float32
    )
    lower = np.broadcast_to((cy - half_height)[:, None], (S, k_columns)).astype(
        np.float32
    )
    return x_bounds, upper.copy(), lower.copy()


# ---------------------------------------------------------------------------
# SVG animation helpers
# ---------------------------------------------------------------------------


def _svg_configuration_band_only(snapshots, input_params, size, margin: float = 0.05):
    """SVG configuration for pure band domains (no underlying circles).

    Args:
        margin: Empty border around the content, as a fraction of the
            content's span.
    """
    n_sets = snapshots[0][1]["x_bounds"].shape[0]
    K = snapshots[0][1]["upper"].shape[1]

    all_x, all_y = [], []
    for _, v in snapshots:
        x_bounds = np.array(v["x_bounds"])
        upper = np.array(v["upper"])
        lower = np.array(v["lower"])
        for s in range(n_sets):
            xs = np.linspace(x_bounds[s, 0], x_bounds[s, 1], K)
            all_x.extend(xs.tolist())
            all_x.extend(xs.tolist())
            all_y.extend(upper[s].tolist())
            all_y.extend(lower[s].tolist())

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    span = max(x_max - x_min, y_max - y_min)
    margin_abs = span * margin
    span += 2 * margin_abs
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    x_min = cx - span / 2
    y_max = cy + span / 2

    def to_svg(x, y):
        return (x - x_min) / span * size, (y_max - y) / span * size

    elements = []
    for s in range(n_sets):
        color = _SVG_SET_COLORS[s % len(_SVG_SET_COLORS)]
        points_frames = []
        for _, v in snapshots:
            x_bounds_s = np.array(v["x_bounds"][s])
            upper_s = np.array(v["upper"][s])
            lower_s = np.array(v["lower"][s])
            xs = np.linspace(x_bounds_s[0], x_bounds_s[1], K)
            pts = []
            for k in range(K):
                px, py = to_svg(xs[k], upper_s[k])
                pts.append(f"{px:.1f},{py:.1f}")
            for k in range(K - 1, -1, -1):
                px, py = to_svg(xs[k], lower_s[k])
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


# ---------------------------------------------------------------------------
# Band domain representation class hierarchy
# ---------------------------------------------------------------------------


@dataclass
class BandRepresentation(ABC):
    """Base class for convex vertical-band boundary representations.

    A representation determines how the boundary of each band domain is
    parametrised during optimisation. Unlike
    :class:`~vizopt.components.stars.StarRepresentation`, `x_bounds` (the
    domain's horizontal extent) is itself an optimisation variable, so
    `wrap`/`to_bounds` need no fixed external grid argument.

    Attributes:
        k_columns: number of uniformly-spaced columns used to sample the
            boundary. For `Discrete` this is also the number of optimised
            `(upper, lower)` pairs.
    """

    k_columns: int = 64

    @abstractmethod
    def initialize_vars(
        self,
        n_sets: int,
        initial_x_bounds: np.ndarray,
        initial_upper: np.ndarray,
        initial_lower: np.ndarray,
    ) -> dict:
        """Return the initial optimisation-variable dict.

        Args:
            n_sets: number of band domains.
            initial_x_bounds: (n_sets, 2) initial `[x_min, x_max]`.
            initial_upper: (n_sets, k_columns) initial upper boundary values.
            initial_lower: (n_sets, k_columns) initial lower boundary values.
        """

    def wrap(self, fn):
        """Adapt a loss term expecting `optim_vars["upper"/"lower"]` to this representation.

        The default identity is correct for `Discrete`, which already stores
        upper/lower directly.
        """
        return fn

    @abstractmethod
    def to_bounds(self, optim_vars: dict) -> tuple:
        """Convert optimisation variables to `(upper, lower)` arrays of shape `(n_sets, k_columns)`."""

    def extra_results(self, s: int, optim_vars: dict) -> dict:
        """Representation-specific extras added to the result dict of set *s*.

        Empty by default (`Discrete` has no extras).
        """
        return {}

    def make_svg_configuration(self, base_svg_fn=None):
        """Return an `svg_configuration` function compatible with the animation helper.

        Converts representation-specific vars to `upper`/`lower` on each
        snapshot so any base SVG renderer that reads those keys can be
        reused for all representations.

        Args:
            base_svg_fn: the underlying `(snapshots, input_params, size) →
                elements` function to delegate to after conversion. Defaults
                to :func:`_svg_configuration_band_only`.
        """
        if base_svg_fn is None:
            base_svg_fn = _svg_configuration_band_only

        def svg_configuration(snapshots, input_params, size, margin: float = 0.05):
            converted = []
            for i, v in snapshots:
                upper, lower = self.to_bounds(v)
                converted.append(
                    (i, {**v, "upper": np.array(upper), "lower": np.array(lower)})
                )
            return base_svg_fn(converted, input_params, size, margin=margin)

        return svg_configuration


@dataclass
class Discrete(BandRepresentation):
    """One `(upper, lower)` pair per column, linearly interpolated.

    The optimisation variables are `x_bounds` of shape `(n_sets, 2)` and
    `upper`/`lower` of shape `(n_sets, k_columns)`. This is the only
    representation for now — the simplest and the default.
    """

    def initialize_vars(self, n_sets, initial_x_bounds, initial_upper, initial_lower):
        return {
            "x_bounds": initial_x_bounds.copy(),
            "upper": initial_upper.copy(),
            "lower": initial_lower.copy(),
        }

    def to_bounds(self, optim_vars):
        return optim_vars["upper"], optim_vars["lower"]
