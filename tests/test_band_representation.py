"""Tests for the BandRepresentation class hierarchy and loss-term helpers."""

import numpy as np
import pytest

from vizopt.components.bands import (
    BandRepresentation,
    Discrete,
    _multi_term_band_area,
    _multi_term_band_convexity,
    _multi_term_band_enclosure,
    _multi_term_band_exclusion,
    _multi_term_band_perimeter,
    _multi_term_min_thickness,
    _multi_term_min_width,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SETS = 3
K = 8
INITIAL_X_BOUNDS = np.array([[-1.0, 1.0]] * N_SETS, dtype=np.float32)
INITIAL_UPPER = np.ones((N_SETS, K), dtype=np.float32) * 2.0
INITIAL_LOWER = np.ones((N_SETS, K), dtype=np.float32) * -2.0


# ---------------------------------------------------------------------------
# Discrete
# ---------------------------------------------------------------------------


def test_discrete_is_band_representation():
    assert isinstance(Discrete(), BandRepresentation)


def test_discrete_initialize_vars_shapes():
    rep = Discrete(k_columns=K)
    vars_ = rep.initialize_vars(N_SETS, INITIAL_X_BOUNDS, INITIAL_UPPER, INITIAL_LOWER)
    assert vars_["x_bounds"].shape == (N_SETS, 2)
    assert vars_["upper"].shape == (N_SETS, K)
    assert vars_["lower"].shape == (N_SETS, K)


def test_discrete_initialize_vars_copies():
    rep = Discrete(k_columns=K)
    vars_ = rep.initialize_vars(N_SETS, INITIAL_X_BOUNDS, INITIAL_UPPER, INITIAL_LOWER)
    vars_["upper"][0, 0] = 999.0
    assert INITIAL_UPPER[0, 0] == pytest.approx(2.0)


def test_discrete_to_bounds_identity():
    rep = Discrete(k_columns=K)
    vars_ = rep.initialize_vars(N_SETS, INITIAL_X_BOUNDS, INITIAL_UPPER, INITIAL_LOWER)
    upper, lower = rep.to_bounds(vars_)
    assert np.allclose(np.array(upper), INITIAL_UPPER)
    assert np.allclose(np.array(lower), INITIAL_LOWER)


def test_discrete_wrap_is_identity():
    rep = Discrete(k_columns=K)

    def fn(v, p):
        return v["upper"].sum()

    wrapped = rep.wrap(fn)
    assert wrapped is fn


def test_discrete_extra_results_empty():
    rep = Discrete(k_columns=K)
    vars_ = rep.initialize_vars(N_SETS, INITIAL_X_BOUNDS, INITIAL_UPPER, INITIAL_LOWER)
    assert rep.extra_results(0, vars_) == {}


def test_make_svg_configuration_returns_callable():
    cfg = Discrete(k_columns=K).make_svg_configuration()
    assert callable(cfg)


# ---------------------------------------------------------------------------
# Loss-term correctness — hand-computable rectangle (K=3)
# ---------------------------------------------------------------------------

_RECT_X_BOUNDS = np.array([[0.0, 2.0]], dtype=np.float32)  # width 2
_RECT_UPPER = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)  # height 2 (upper - lower)
_RECT_LOWER = np.array([[-1.0, -1.0, -1.0]], dtype=np.float32)


def test_band_area_of_rectangle():
    vars_ = {
        "x_bounds": _RECT_X_BOUNDS,
        "upper": _RECT_UPPER,
        "lower": _RECT_LOWER,
    }
    area = _multi_term_band_area(vars_, {})
    assert float(area) == pytest.approx(4.0)  # width 2 * height 2


def test_band_perimeter_of_rectangle():
    vars_ = {
        "x_bounds": _RECT_X_BOUNDS,
        "upper": _RECT_UPPER,
        "lower": _RECT_LOWER,
    }
    perimeter = _multi_term_band_perimeter(vars_, {})
    assert float(perimeter) == pytest.approx(8.0)  # 2 * (2 + 2)


def test_min_thickness_penalizes_thin_band():
    vars_ = {
        "upper": np.array([[0.01, 0.01]], dtype=np.float32),
        "lower": np.array([[0.0, 0.0]], dtype=np.float32),
    }
    assert float(_multi_term_min_thickness(vars_, {})) > 0.0


def test_min_thickness_zero_for_thick_band():
    vars_ = {
        "upper": np.array([[1.0, 1.0]], dtype=np.float32),
        "lower": np.array([[-1.0, -1.0]], dtype=np.float32),
    }
    assert float(_multi_term_min_thickness(vars_, {})) == pytest.approx(0.0)


def test_min_width_penalizes_narrow_band():
    vars_ = {"x_bounds": np.array([[0.0, 0.01]], dtype=np.float32)}
    assert float(_multi_term_min_width(vars_, {})) > 0.0


def test_min_width_zero_for_wide_band():
    vars_ = {"x_bounds": np.array([[0.0, 2.0]], dtype=np.float32)}
    assert float(_multi_term_min_width(vars_, {})) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Loss-term correctness — convexity sign conventions
# ---------------------------------------------------------------------------


def test_convexity_zero_for_flat_band():
    vars_ = {
        "x_bounds": _RECT_X_BOUNDS,
        "upper": _RECT_UPPER,
        "lower": _RECT_LOWER,
    }
    penalty = _multi_term_band_convexity(vars_, {"convexity_alpha": 0.0})
    assert float(penalty) == pytest.approx(0.0)


def test_convexity_penalizes_dipping_upper_and_bumping_lower():
    # upper dips down in the middle (convex-shaped, violates required concavity);
    # lower bumps up in the middle (concave-shaped, violates required convexity).
    vars_ = {
        "x_bounds": _RECT_X_BOUNDS,  # dx = 1 -> dx_sq = 1
        "upper": np.array([[1.0, 0.0, 1.0]], dtype=np.float32),  # d2 = +2
        "lower": np.array([[0.0, 1.0, 0.0]], dtype=np.float32),  # d2 = -2
    }
    penalty = _multi_term_band_convexity(vars_, {"convexity_alpha": 0.0})
    # each violation is max(0, 2) = 2 -> squared = 4, summed over upper + lower
    assert float(penalty) == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Loss-term correctness — enclosure / exclusion
# ---------------------------------------------------------------------------

# set 0 = inner (small), set 1 = outer (large, fully containing inner with margin)
_NESTED_X_BOUNDS = np.array([[-1.0, 1.0], [-5.0, 5.0]], dtype=np.float32)
_NESTED_UPPER = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]], dtype=np.float32)
_NESTED_LOWER = np.array([[-1.0, -1.0, -1.0], [-3.0, -3.0, -3.0]], dtype=np.float32)


def test_enclosure_zero_when_nested():
    mask = np.array([[False, True], [False, False]])  # inner=0 must be inside outer=1
    vars_ = {
        "x_bounds": _NESTED_X_BOUNDS,
        "upper": _NESTED_UPPER,
        "lower": _NESTED_LOWER,
    }
    penalty = _multi_term_band_enclosure(vars_, {"enclosure_mask": mask})
    assert float(penalty) == pytest.approx(0.0)


def test_enclosure_positive_when_inner_pokes_out():
    mask = np.array([[False, True], [False, False]])
    upper = _NESTED_UPPER.copy()
    upper[0, 1] = 10.0  # inner's top boundary now exceeds outer's (3.0)
    vars_ = {"x_bounds": _NESTED_X_BOUNDS, "upper": upper, "lower": _NESTED_LOWER}
    penalty = _multi_term_band_enclosure(vars_, {"enclosure_mask": mask})
    assert float(penalty) > 0.0


def test_exclusion_zero_when_separated():
    # set 0 sits far to the right of set 1; no overlap
    x_bounds = np.array([[10.0, 12.0], [-5.0, 5.0]], dtype=np.float32)
    mask = np.array([[False, True], [True, False]])
    vars_ = {"x_bounds": x_bounds, "upper": _NESTED_UPPER, "lower": _NESTED_LOWER}
    penalty = _multi_term_band_exclusion(vars_, {"exclusion_mask": mask})
    assert float(penalty) == pytest.approx(0.0)


def test_exclusion_positive_when_overlapping():
    mask = np.array([[False, True], [True, False]])
    vars_ = {
        "x_bounds": _NESTED_X_BOUNDS,
        "upper": _NESTED_UPPER,
        "lower": _NESTED_LOWER,
    }
    # set 0 is nested entirely inside set 1 -> deeply overlapping, not excluded
    penalty = _multi_term_band_exclusion(vars_, {"exclusion_mask": mask})
    assert float(penalty) > 0.0
