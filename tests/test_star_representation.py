"""Tests for the StarRepresentation class hierarchy and Fourier helpers."""

import numpy as np
import jax.numpy as jnp
import pytest

from vizopt.components.stars import (
    BSpline,
    Discrete,
    Fourier,
    StarRepresentation,
    _build_membership,
    _wrap_fourier_term,
    fourier_to_radii,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SETS = 3
K = 16
INITIAL_CENTERS = np.zeros((N_SETS, 2), dtype=np.float32)
INITIAL_RADII = np.ones((N_SETS, K), dtype=np.float32) * 2.0
ANGLES = np.linspace(0, 2 * np.pi, K, endpoint=False).astype(np.float32)
ANGLES_JNP = jnp.array(ANGLES)


# ---------------------------------------------------------------------------
# Discrete
# ---------------------------------------------------------------------------


def test_discrete_is_star_representation():
    assert isinstance(Discrete(), StarRepresentation)


def test_discrete_initialize_vars_shapes():
    rep = Discrete(k_angles=K)
    vars_ = rep.initialize_vars(N_SETS, INITIAL_RADII, INITIAL_CENTERS)
    assert vars_["centers"].shape == (N_SETS, 2)
    assert vars_["radii"].shape == (N_SETS, K)


def test_discrete_initialize_vars_copies():
    rep = Discrete(k_angles=K)
    vars_ = rep.initialize_vars(N_SETS, INITIAL_RADII, INITIAL_CENTERS)
    vars_["radii"][0, 0] = 999.0
    assert INITIAL_RADII[0, 0] == pytest.approx(2.0)


def test_discrete_to_radii_identity():
    rep = Discrete(k_angles=K)
    vars_ = rep.initialize_vars(N_SETS, INITIAL_RADII, INITIAL_CENTERS)
    radii = rep.to_radii(vars_, ANGLES_JNP)
    assert np.allclose(np.array(radii), INITIAL_RADII)


def test_discrete_wrap_is_identity():
    rep = Discrete(k_angles=K)
    fn = lambda v, p: v["radii"].sum()
    wrapped = rep.wrap(fn, ANGLES_JNP)
    assert wrapped is fn


def test_discrete_extra_results_empty():
    rep = Discrete(k_angles=K)
    vars_ = rep.initialize_vars(N_SETS, INITIAL_RADII, INITIAL_CENTERS)
    assert rep.extra_results(0, vars_) == {}


# ---------------------------------------------------------------------------
# Fourier
# ---------------------------------------------------------------------------


def test_fourier_initialize_vars_shapes():
    rep = Fourier(k_angles=K, n_harmonics=4)
    vars_ = rep.initialize_vars(N_SETS, INITIAL_RADII, INITIAL_CENTERS)
    assert vars_["centers"].shape == (N_SETS, 2)
    assert vars_["fourier_coeffs"].shape == (N_SETS, 2 * 4 + 1)


def test_fourier_initialize_dc_term():
    rep = Fourier(k_angles=K, n_harmonics=4)
    vars_ = rep.initialize_vars(N_SETS, INITIAL_RADII, INITIAL_CENTERS)
    # DC term should equal the first element of initial_radii per set
    np.testing.assert_allclose(vars_["fourier_coeffs"][:, 0], INITIAL_RADII[:, 0])


def test_fourier_initialize_ac_terms_zero():
    rep = Fourier(k_angles=K, n_harmonics=4)
    vars_ = rep.initialize_vars(N_SETS, INITIAL_RADII, INITIAL_CENTERS)
    np.testing.assert_allclose(vars_["fourier_coeffs"][:, 1:], 0.0)


def test_fourier_to_radii_dc_only_gives_constant():
    rep = Fourier(k_angles=K, n_harmonics=4)
    vars_ = rep.initialize_vars(N_SETS, INITIAL_RADII, INITIAL_CENTERS)
    radii = np.array(rep.to_radii(vars_, ANGLES_JNP))
    assert radii.shape == (N_SETS, K)
    # With only a DC term, all radii should equal the DC value
    for s in range(N_SETS):
        np.testing.assert_allclose(radii[s], vars_["fourier_coeffs"][s, 0], rtol=1e-5)


def test_fourier_wrap_injects_radii():
    rep = Fourier(k_angles=K, n_harmonics=4)
    vars_ = rep.initialize_vars(N_SETS, INITIAL_RADII, INITIAL_CENTERS)
    received = {}

    def fn(v, p):
        received["radii"] = v["radii"]
        return jnp.array(0.0)

    wrapped = rep.wrap(fn, ANGLES_JNP)
    wrapped(vars_, {})
    assert "radii" in received
    assert received["radii"].shape == (N_SETS, K)


def test_fourier_extra_results():
    rep = Fourier(k_angles=K, n_harmonics=4)
    vars_ = rep.initialize_vars(N_SETS, INITIAL_RADII, INITIAL_CENTERS)
    extras = rep.extra_results(0, vars_)
    assert "fourier_coeffs" in extras
    assert extras["fourier_coeffs"].shape == (2 * 4 + 1,)


# ---------------------------------------------------------------------------
# BSpline
# ---------------------------------------------------------------------------


def test_bspline_initialize_vars_shapes():
    rep = BSpline(k_angles=K, n_ctrl_pts=8)
    vars_ = rep.initialize_vars(N_SETS, INITIAL_RADII, INITIAL_CENTERS)
    assert vars_["centers"].shape == (N_SETS, 2)
    assert vars_["bspline_ctrl"].shape == (N_SETS, 8)


def test_bspline_initialize_ctrl_pts_constant():
    rep = BSpline(k_angles=K, n_ctrl_pts=8)
    vars_ = rep.initialize_vars(N_SETS, INITIAL_RADII, INITIAL_CENTERS)
    # All control points should be set to the mean initial radius
    for s in range(N_SETS):
        np.testing.assert_allclose(
            vars_["bspline_ctrl"][s], INITIAL_RADII[s, 0], rtol=1e-5
        )


def test_bspline_to_radii_uniform_ctrl_gives_constant():
    rep = BSpline(k_angles=K, n_ctrl_pts=8)
    vars_ = rep.initialize_vars(N_SETS, INITIAL_RADII, INITIAL_CENTERS)
    radii = np.array(rep.to_radii(vars_, ANGLES_JNP))
    assert radii.shape == (N_SETS, K)
    for s in range(N_SETS):
        np.testing.assert_allclose(radii[s], INITIAL_RADII[s, 0], rtol=1e-4)


def test_bspline_wrap_injects_radii():
    rep = BSpline(k_angles=K, n_ctrl_pts=8)
    vars_ = rep.initialize_vars(N_SETS, INITIAL_RADII, INITIAL_CENTERS)
    received = {}

    def fn(v, p):
        received["radii"] = v["radii"]
        return jnp.array(0.0)

    wrapped = rep.wrap(fn, ANGLES_JNP)
    wrapped(vars_, {})
    assert "radii" in received
    assert received["radii"].shape == (N_SETS, K)


def test_bspline_extra_results():
    rep = BSpline(k_angles=K, n_ctrl_pts=8)
    vars_ = rep.initialize_vars(N_SETS, INITIAL_RADII, INITIAL_CENTERS)
    extras = rep.extra_results(0, vars_)
    assert "bspline_ctrl" in extras
    assert extras["bspline_ctrl"].shape == (8,)


# ---------------------------------------------------------------------------
# fourier_to_radii
# ---------------------------------------------------------------------------


def test_fourier_to_radii_dc_only():
    angles = jnp.linspace(0, 2 * np.pi, 8, endpoint=False)
    coeffs = jnp.array([[3.0, 0.0, 0.0]])  # n_sets=1, M=1: [a0, a1, b1]
    radii = np.array(fourier_to_radii(coeffs, angles))
    np.testing.assert_allclose(radii, 3.0, rtol=1e-5)


def test_fourier_to_radii_cosine_harmonic():
    angles = jnp.linspace(0, 2 * np.pi, 64, endpoint=False)
    # r(θ) = 2 + cos(θ)  →  coeffs = [2, 1, 0]
    coeffs = jnp.array([[2.0, 1.0, 0.0]])
    radii = np.array(fourier_to_radii(coeffs, angles))
    expected = 2.0 + np.cos(np.array(angles))
    np.testing.assert_allclose(radii[0], expected, atol=1e-5)


def test_fourier_to_radii_output_shape():
    angles = jnp.linspace(0, 2 * np.pi, 32, endpoint=False)
    coeffs = jnp.zeros((4, 5))  # n_sets=4, M=2
    radii = fourier_to_radii(coeffs, angles)
    assert radii.shape == (4, 32)


# ---------------------------------------------------------------------------
# _wrap_fourier_term
# ---------------------------------------------------------------------------


def test_wrap_fourier_term_passes_radii():
    angles = jnp.linspace(0, 2 * np.pi, K, endpoint=False)
    coeffs = jnp.zeros((2, 3))  # DC only, M=1
    coeffs = coeffs.at[:, 0].set(1.5)
    vars_ = {"centers": jnp.zeros((2, 2)), "fourier_coeffs": coeffs}
    received = {}

    def fn(v, p):
        received["radii"] = v["radii"]
        return jnp.array(0.0)

    wrapped = _wrap_fourier_term(fn, angles)
    wrapped(vars_, {})
    assert received["radii"].shape == (2, K)
    np.testing.assert_allclose(np.array(received["radii"]), 1.5, atol=1e-5)


def test_wrap_fourier_term_preserves_other_vars():
    angles = jnp.linspace(0, 2 * np.pi, K, endpoint=False)
    coeffs = jnp.zeros((1, 3))
    extra = jnp.array([42.0])
    vars_ = {"centers": jnp.zeros((1, 2)), "fourier_coeffs": coeffs, "extra": extra}
    received = {}

    def fn(v, p):
        received["extra"] = v["extra"]
        return jnp.array(0.0)

    _wrap_fourier_term(fn, angles)(vars_, {})
    assert float(received["extra"][0]) == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# _build_membership
# ---------------------------------------------------------------------------


def test_build_membership_shape():
    sets = [[0, 1], [1, 2], [0, 2]]
    m = _build_membership(S=3, N=3, sets=sets)
    assert m.shape == (3, 3)


def test_build_membership_values():
    sets = [[0, 2], [1]]
    m = _build_membership(S=2, N=3, sets=sets)
    assert bool(m[0, 0]) is True
    assert bool(m[0, 1]) is False
    assert bool(m[0, 2]) is True
    assert bool(m[1, 0]) is False
    assert bool(m[1, 1]) is True
    assert bool(m[1, 2]) is False


def test_build_membership_dtype_bool():
    m = _build_membership(S=1, N=2, sets=[[0]])
    assert m.dtype == bool


# ---------------------------------------------------------------------------
# make_svg_configuration
# ---------------------------------------------------------------------------


def test_make_svg_configuration_returns_callable():
    for rep in [Discrete(k_angles=K), Fourier(k_angles=K), BSpline(k_angles=K)]:
        cfg = rep.make_svg_configuration()
        assert callable(cfg)
