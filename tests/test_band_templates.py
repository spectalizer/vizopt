"""Integration tests for the BandDomainOptimizer template."""

import numpy as np
import pytest

from vizopt.base import OptimConfig
from vizopt.components.bands import _multi_term_band_exclusion
from vizopt.templates.band_vs_band import BandDomainOptimizer

_FAST = OptimConfig(n_iters=5, learning_rate=1e-2)


def _NO_PRINT(*_):
    pass


def _two_set_problem(**kwargs) -> BandDomainOptimizer:
    """Minimal two-set problem for smoke-testing the optimizer."""
    centers = np.array([[0.0, 0.0], [6.0, 0.0]], dtype=np.float32)
    optimizer = BandDomainOptimizer(2, centers, target_areas=[4.0, 6.0], **kwargs)
    optimizer.optimize(optim_config=_FAST, callback=_NO_PRINT)
    return optimizer


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


def test_optimize_returns_sets_list():
    opt = _two_set_problem()
    assert isinstance(opt.sets_, list)
    assert len(opt.sets_) == 2


def test_optimize_result_keys():
    for s in _two_set_problem().sets_:
        assert "x" in s
        assert "upper" in s
        assert "lower" in s
        assert "x_min" in s
        assert "x_max" in s


def test_optimize_result_shapes():
    for s in _two_set_problem().sets_:
        assert s["x"].shape == (64,)
        assert s["upper"].shape == (64,)
        assert s["lower"].shape == (64,)


def test_optimize_upper_above_lower():
    for s in _two_set_problem().sets_:
        assert np.all(s["upper"] >= s["lower"])


def test_optimize_x_matches_bounds():
    for s in _two_set_problem().sets_:
        assert s["x"][0] == pytest.approx(s["x_min"])
        assert s["x"][-1] == pytest.approx(s["x_max"])


def test_optimize_history_has_term_keys():
    history = _two_set_problem().result_.history
    assert len(history) > 0
    for record in history:
        assert "iteration" in record
        assert "total" in record


def test_no_result_before_optimize_raises():
    opt = BandDomainOptimizer(1, np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        _ = opt.sets_


# ---------------------------------------------------------------------------
# Enclosure / exclusion behaviour
# ---------------------------------------------------------------------------


def test_enclosure_pulls_inner_inside_outer():
    """A small target-area inner set forced inside a large outer set should
    end up with a tighter x-range than the (much larger) outer set."""
    centers = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    opt = BandDomainOptimizer(
        2,
        centers,
        target_areas=[1.0, 20.0],
        enclosures=[(0, 1)],
        initial_half_width=0.5,
    )
    opt.optimize(
        optim_config=OptimConfig(n_iters=300, learning_rate=5e-2), callback=_NO_PRINT
    )
    inner, outer = opt.sets_
    assert inner["x_max"] - inner["x_min"] < outer["x_max"] - outer["x_min"]
    assert inner["x_min"] >= outer["x_min"] - 0.5
    assert inner["x_max"] <= outer["x_max"] + 0.5


def test_exclusion_reduces_overlap():
    """Two overlapping sets with exclusion enabled should see the exclusion
    penalty drop substantially from its initial (fully overlapping) value.
    Centers are slightly offset (not identical) so gradient descent has a
    direction to break the initial overlap along — two perfectly symmetric
    sets would move in lockstep forever and never separate."""
    centers = np.array([[-0.2, 0.1], [0.2, -0.1]], dtype=np.float32)
    opt = BandDomainOptimizer(
        2,
        centers,
        target_areas=[2.0, 2.0],
        weight_exclusion=20.0,
        initial_half_width=1.0,
    )
    problem = opt._build_problem()
    initial_vars = problem.initialize(problem.input_parameters, 0)
    initial_penalty = float(
        _multi_term_band_exclusion(initial_vars, problem.input_parameters)
    )

    opt.optimize(
        optim_config=OptimConfig(n_iters=300, learning_rate=5e-2), callback=_NO_PRINT
    )
    final_penalty = float(
        _multi_term_band_exclusion(
            opt.result_.optim_vars, opt.problem_.input_parameters
        )
    )
    assert final_penalty < 0.5 * initial_penalty
