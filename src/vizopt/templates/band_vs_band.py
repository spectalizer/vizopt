"""Convex vertical-band domain optimization without underlying circles.

Finds convex regions — each the area between a concave upper boundary y2(x)
and a convex lower boundary y1(x) over x in [x_min, x_max] — satisfying
enclosure/exclusion constraints with optional per-set area targets. Each
boundary is parametrized as `x_bounds` plus K `(upper, lower)` pairs at
uniformly-spaced columns.

Complementary to :mod:`vizopt.templates.star_vs_star`: restricted to convex
shapes, in exchange for enclosure/exclusion checks that reduce to pointwise
comparisons instead of general polygon containment.

General convex-band loss terms and helpers live in
:mod:`vizopt.components.bands`.
"""

import numpy as np

from ..base import (
    ObjectiveTerm,
    OptimizationProblem,
    OptimizationProblemTemplate,
    VizOptimizer,
)
from ..components.bands import (
    BandRepresentation,
    Discrete,
    _init_rectangle,
    _multi_term_band_area,
    _multi_term_band_convexity,
    _multi_term_band_enclosure,
    _multi_term_band_exclusion,
    _multi_term_band_perimeter,
    _multi_term_band_smoothness,
    _multi_term_min_thickness,
    _multi_term_min_width,
    _multi_term_target_area,
)
from .star_vs_star import _build_exclusion_mask


def _half_height_from_target_area(target_area: float, half_width: float) -> float:
    """Initial half-height so a rectangle of the given half-width has the target area."""
    return float(target_area / (4.0 * half_width))


class BandDomainOptimizer(VizOptimizer):
    """Optimise convex vertical-band domains with no underlying circles.

    Complementary to :class:`~vizopt.templates.star_vs_star.StarDomainOptimizer`:
    each boundary is restricted to a convex region, represented as an upper
    boundary y2(x) and lower boundary y1(x) sampled at K uniformly-spaced
    columns over x in [x_min, x_max], instead of a star-shaped center + K
    radii. This makes containment/overlap checks reduce to pointwise
    comparisons, at the cost of only being able to represent convex shapes.

    For sets with a target area the area term pulls toward that value; for
    sets without one the plain area-minimisation term acts as a regulariser.

    Args:
        n_sets: Number of band domains.
        initial_centers: `(n_sets, 2)` starting centers for each domain.
        representation: Band domain parametrization. Only
            :class:`~vizopt.components.bands.Discrete` (default) exists so far.
        target_areas: List of n_sets values (float or None). None means no area
            target for that set; its area is then minimised by `weight_area`.
        initial_half_width: Starting half-width for all sets.
        initial_half_height: Fallback starting half-height for sets without a
            target area.
        weight_target_area: Weight for the target-area penalty.
        weight_area: Weight for the area-minimisation regulariser.
        weight_perimeter: Weight for the perimeter-minimisation regulariser.
        weight_exclusion: Weight for band-vs-band exclusion (all pairs s≠t).
            Set to 0 for purely nested layouts.
        weight_enclosure: Weight for band-vs-band enclosure constraints.
        weight_smoothness: Weight for adjacent-column smoothness penalty.
        weight_convexity: Weight for the convexity penalty. Default 0.0
            (disabled). Enable it if the other terms pull a boundary into a
            non-convex configuration.
        convexity_alpha: Linear-term coefficient in the convexity penalty.
        enclosures: List of `(inner_idx, outer_idx)` pairs.
        exclusion_offset: Minimum gap enforced between non-nested boundaries.
        enclosure_offset: Minimum inset enforced for enclosure constraints.
        exclusion_interior_fracs: Interior height fractions for the exclusion term.
    """

    def __init__(
        self,
        n_sets: int,
        initial_centers,
        *,
        representation: BandRepresentation | None = None,
        target_areas=None,
        initial_half_width: float = 1.0,
        initial_half_height: float = 1.0,
        weight_target_area: float = 20.0,
        weight_area: float = 1.0,
        weight_perimeter: float = 0.5,
        weight_exclusion: float = 10.0,
        weight_enclosure: float = 20.0,
        weight_smoothness: float = 1.0,
        weight_convexity: float = 0.0,
        convexity_alpha: float = 1.0,
        enclosures=None,
        exclusion_offset: float = 0.1,
        enclosure_offset: float = 0.1,
        exclusion_interior_fracs=None,
    ):
        self.n_sets = n_sets
        self.initial_centers = initial_centers
        self.representation = (
            representation if representation is not None else Discrete()
        )
        self.target_areas = target_areas
        self.initial_half_width = initial_half_width
        self.initial_half_height = initial_half_height
        self.weight_target_area = weight_target_area
        self.weight_area = weight_area
        self.weight_perimeter = weight_perimeter
        self.weight_exclusion = weight_exclusion
        self.weight_enclosure = weight_enclosure
        self.weight_smoothness = weight_smoothness
        self.weight_convexity = weight_convexity
        self.convexity_alpha = convexity_alpha
        self.enclosures = enclosures
        self.exclusion_offset = exclusion_offset
        self.enclosure_offset = enclosure_offset
        self.exclusion_interior_fracs = (
            exclusion_interior_fracs if exclusion_interior_fracs is not None else [0.5]
        )

    def _build_problem(self) -> OptimizationProblem:
        n_sets = self.n_sets
        representation = self.representation
        k_columns = representation.k_columns
        initial_centers = np.asarray(self.initial_centers, dtype=np.float32)

        targets_raw = (
            self.target_areas if self.target_areas is not None else [None] * n_sets
        )
        target_arr = np.array(
            [t if t is not None else np.nan for t in targets_raw], dtype=np.float32
        )

        half_width = np.full(n_sets, self.initial_half_width, dtype=np.float32)
        half_height = np.full(n_sets, self.initial_half_height, dtype=np.float32)
        for s in range(n_sets):
            if np.isfinite(target_arr[s]):
                half_height[s] = _half_height_from_target_area(
                    float(target_arr[s]), float(half_width[s])
                )

        initial_x_bounds, initial_upper, initial_lower = _init_rectangle(
            initial_centers, half_width, half_height, k_columns
        )

        enclosure_mask = np.zeros((n_sets, n_sets), dtype=bool)
        for inner, outer in self.enclosures or []:
            enclosure_mask[inner, outer] = True

        exclusion_mask = _build_exclusion_mask(n_sets, self.enclosures)

        input_parameters = {
            "target_areas": target_arr,
            "enclosure_mask": enclosure_mask,
            "exclusion_mask": exclusion_mask,
            "exclusion_offset": float(self.exclusion_offset),
            "enclosure_offset": float(self.enclosure_offset),
            "exclusion_interior_fracs": self.exclusion_interior_fracs,
            "convexity_alpha": np.float32(self.convexity_alpha),
        }

        init_vars = representation.initialize_vars(
            n_sets, initial_x_bounds, initial_upper, initial_lower
        )

        def initialize(_, seed):
            return {k: v.copy() for k, v in init_vars.items()}

        def wrap(fn):
            return representation.wrap(fn)

        return OptimizationProblemTemplate(
            terms=[
                ObjectiveTerm(
                    "target_area",
                    wrap(_multi_term_target_area),
                    self.weight_target_area,
                ),
                ObjectiveTerm(
                    "band_excl",
                    wrap(_multi_term_band_exclusion),
                    self.weight_exclusion,
                ),
                ObjectiveTerm(
                    "band_enclose",
                    wrap(_multi_term_band_enclosure),
                    self.weight_enclosure,
                ),
                ObjectiveTerm("min_width", wrap(_multi_term_min_width), 10.0),
                ObjectiveTerm("min_thickness", wrap(_multi_term_min_thickness), 10.0),
                ObjectiveTerm(
                    "smoothness",
                    wrap(_multi_term_band_smoothness),
                    self.weight_smoothness,
                ),
                ObjectiveTerm(
                    "convexity",
                    wrap(_multi_term_band_convexity),
                    self.weight_convexity,
                ),
                ObjectiveTerm("area", wrap(_multi_term_band_area), self.weight_area),
                ObjectiveTerm(
                    "perimeter",
                    wrap(_multi_term_band_perimeter),
                    self.weight_perimeter,
                ),
            ],
            initialize=initialize,
            svg_configuration=representation.make_svg_configuration(),
        ).instantiate(input_parameters)

    @property
    def sets_(self) -> list[dict]:
        """Band boundary dicts from the last optimization result.

        Each dict has `"x"` (K,), `"upper"` (K,), `"lower"` (K,), `"x_min"`,
        `"x_max"`, plus any representation-specific extras.

        Raises:
            ValueError: If :meth:`optimize` has not been called yet.
        """
        if not hasattr(self, "result_"):
            raise ValueError("No result yet — call optimize() first.")
        optim_vars = self.result_.optim_vars
        upper, lower = self.representation.to_bounds(optim_vars)
        upper = np.array(upper)
        lower = np.array(lower)
        x_bounds = np.array(optim_vars["x_bounds"])
        K = upper.shape[1]
        return [
            {
                "x": np.linspace(x_bounds[s, 0], x_bounds[s, 1], K),
                "upper": upper[s],
                "lower": lower[s],
                "x_min": float(x_bounds[s, 0]),
                "x_max": float(x_bounds[s, 1]),
                **self.representation.extra_results(s, optim_vars),
            }
            for s in range(self.n_sets)
        ]
