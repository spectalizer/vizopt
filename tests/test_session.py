"""Tests for vizopt.session"""

import jax.numpy as jnp
import numpy as np
import pytest

from vizopt.base import ObjectiveTerm, OptimConfig, OptimizationProblemTemplate
from vizopt.session import OptimizationSession


def _NO_PRINT(*_):
    pass


def _points_problem(var_scales=None, attraction_multiplier=1.0):
    """Three 2D points pulled towards the origin and towards each other."""
    terms = [
        ObjectiveTerm(
            name="origin",
            compute=lambda v, p: jnp.sum(v["points"] ** 2),
        ),
        ObjectiveTerm(
            name="attraction",
            compute=lambda v, p: jnp.sum(
                (v["points"][:, None, :] - v["points"][None, :, :]) ** 2
            ),
            multiplier=attraction_multiplier,
        ),
    ]
    template = OptimizationProblemTemplate(
        terms=terms,
        initialize=lambda p, seed: {"points": jnp.array(p["initial_points"])},
    )
    initial_points = [[1.0, 2.0], [-3.0, 1.0], [2.0, -2.0]]
    return template.instantiate(
        {"initial_points": initial_points}, var_scales=var_scales
    )


def _config(**kwargs):
    defaults = dict(n_iters=200, learning_rate=0.05, decay_lr_to=1.0)
    return OptimConfig(**(defaults | kwargs))


# --- construction and stepping ---


def test_session_returns_session_at_iteration_zero():
    session = _points_problem().session(_config())
    assert isinstance(session, OptimizationSession)
    assert session.iteration == 0
    assert session.last_step is None
    np.testing.assert_allclose(session.vars["points"][0], [1.0, 2.0])


def test_step_advances_iteration():
    session = _points_problem().session(_config())
    result = session.step(5)
    assert session.iteration == 5
    assert result.iteration == 4
    assert result is session.last_step


def test_step_rejects_non_positive_n():
    with pytest.raises(ValueError):
        _points_problem().session(_config()).step(0)


def test_step_minimizes():
    session = _points_problem().session(_config())
    session.step(500)
    assert float(jnp.max(jnp.abs(session.vars["points"]))) < 0.1


def test_session_matches_optimize():
    """optimize() is a thin loop over a session: same config, same result."""
    problem = _points_problem()
    config = _config(n_iters=50, decay_lr_to=0.1)
    result = problem.optimize(config, callback=_NO_PRINT)
    session = problem.session(config)
    session.step(50)
    np.testing.assert_allclose(
        session.vars["points"], result.optim_vars["points"], rtol=1e-6
    )


def test_vars_are_physical_with_var_scales():
    session = _points_problem(var_scales={"points": jnp.array([10.0, 0.5])}).session(
        _config()
    )
    np.testing.assert_allclose(session.vars["points"][1], [-3.0, 1.0], rtol=1e-6)


def test_step_function_is_compiled_once_per_problem():
    problem = _points_problem()
    first = problem.session(_config())
    second = problem.session(_config(learning_rate=0.1, n_iters=10))
    assert first._step_function is second._step_function


# --- pinning ---


@pytest.mark.parametrize("var_scales", [None, {"points": jnp.array([10.0, 0.5])}])
def test_pin_holds_value_while_others_move(var_scales):
    session = _points_problem(var_scales=var_scales).session(_config())
    session.pin("points", 0, value=[4.0, 4.0])
    session.step(200)
    points = session.vars["points"]
    np.testing.assert_allclose(points[0], [4.0, 4.0], rtol=1e-6)
    # The free points are pulled towards the pinned one, away from the origin.
    assert float(jnp.mean(points[1:, 0])) > 0.5


def test_pin_without_value_pins_in_place():
    session = _points_problem().session(_config())
    session.step(10)
    before = session.vars["points"][2]
    session.pin("points", 2)
    session.step(50)
    np.testing.assert_allclose(session.vars["points"][2], before, rtol=1e-6)


def test_repinning_moves_pinned_entry():
    """A drag is a sequence of pins with new values."""
    session = _points_problem().session(_config())
    session.pin("points", 0, value=[1.0, 1.0])
    session.step(5)
    session.pin("points", 0, value=[2.0, 3.0])
    session.step(5)
    np.testing.assert_allclose(session.vars["points"][0], [2.0, 3.0], rtol=1e-6)


def test_is_pinned_mask():
    session = _points_problem().session(_config())
    session.pin("points", 1)
    mask = np.asarray(session.is_pinned("points"))
    assert mask.shape == (3, 2)
    assert mask[1].all() and not mask[0].any() and not mask[2].any()


def test_unpin_releases_entry():
    session = _points_problem().session(_config())
    session.pin("points", 0, value=[4.0, 4.0])
    session.step(10)
    session.unpin("points", 0)
    session.step(500)
    assert float(jnp.max(jnp.abs(session.vars["points"]))) < 0.1


def test_unpin_all():
    session = _points_problem().session(_config())
    session.pin("points", 0)
    session.pin("points", 2)
    session.unpin_all()
    assert not np.asarray(session.is_pinned("points")).any()


def test_set_value_without_pin_lets_optimizer_move_entry():
    session = _points_problem().session(_config())
    session.set_value("points", 0, [5.0, 5.0])
    np.testing.assert_allclose(session.vars["points"][0], [5.0, 5.0], rtol=1e-6)
    session.step(10)
    assert float(session.vars["points"][0, 0]) < 5.0


def test_set_value_whole_variable():
    session = _points_problem().session(_config())
    session.set_value("points", None, jnp.zeros((3, 2)))
    np.testing.assert_allclose(session.vars["points"], 0.0)


# --- weights ---


def test_weights_default_to_multipliers():
    session = _points_problem(attraction_multiplier=2.0).session(_config())
    assert session.weights == {"origin": 1.0, "attraction": 2.0}


def test_set_weight_matches_built_in_multiplier():
    """Changing a weight live is equivalent to building with that multiplier."""
    live = _points_problem().session(_config())
    live.set_weight("attraction", 5.0)
    live.step(30)
    built = _points_problem(attraction_multiplier=5.0).session(_config())
    built.step(30)
    np.testing.assert_allclose(live.vars["points"], built.vars["points"], rtol=1e-6)
    unchanged = _points_problem().session(_config())
    unchanged.step(30)
    assert not np.allclose(live.vars["points"], unchanged.vars["points"])


def test_set_weight_unknown_term_raises():
    with pytest.raises(KeyError):
        _points_problem().session(_config()).set_weight("nope", 1.0)


def test_set_weight_on_inactive_term_raises():
    session = _points_problem(attraction_multiplier=0.0).session(_config())
    with pytest.raises(ValueError, match="attraction"):
        session.set_weight("attraction", 1.0)


def test_record_uses_live_weights():
    session = _points_problem().session(_config())
    session.set_weight("attraction", 3.0)
    session.step()
    record = session.record()
    assert record["attraction"] == pytest.approx(3.0 * record["attraction_unweighted"])


def test_record_before_step_raises():
    with pytest.raises(ValueError):
        _points_problem().session(_config()).record()


# --- learning rate ---


def test_reheat_restarts_decay():
    """After a full cosine decay to 0, steps stop moving; reheating resumes."""
    session = _points_problem().session(_config(n_iters=20, decay_lr_to=0.0))
    session.step(20)
    before = session.vars["points"]
    session.step(5)
    np.testing.assert_allclose(session.vars["points"], before)
    session.reheat()
    session.step(5)
    assert not np.allclose(session.vars["points"], before)


def test_reheat_with_new_learning_rate():
    session = _points_problem().session(_config())
    session.reheat(n_iters=10, learning_rate=0.0)
    before = session.vars["points"]
    session.step(3)
    np.testing.assert_allclose(session.vars["points"], before)
