"""Tests for vizopt.base"""

import pytest
import jax.numpy as jnp
from pydantic import BaseModel, ValidationError

from vizopt.base import (
    ObjectiveTerm,
    OptimConfig,
    OptimizationProblem,
    OptimizationProblemTemplate,
    build_objective,
)

_NO_PRINT = lambda *_: None  # silent callback for tests


# --- ObjectiveTerm ---


def test_objective_term_defaults():
    term = ObjectiveTerm(name="foo", compute=lambda v, p: jnp.sum(v["x"]))
    assert term.multiplier == 1.0
    assert term.schedule is None


def test_objective_term_fields():
    compute = lambda v, p: v["x"]
    term = ObjectiveTerm(name="bar", compute=compute, multiplier=2.5)
    assert term.name == "bar"
    assert term.compute is compute
    assert term.multiplier == 2.5


# --- build_objective ---


def test_build_objective_empty_terms():
    """An objective with no terms must return exactly 0.0 for any input."""
    fun = build_objective([], input_parameters={})
    assert float(fun({"x": jnp.array(5.0)}, jnp.int32(0))) == 0.0


def test_build_objective_zero_multiplier_excluded():
    """Terms with multiplier=0.0 must not be called."""
    called = []

    def bad_compute(v, p):
        called.append(True)
        return v["x"]

    term = ObjectiveTerm(name="zero", compute=bad_compute, multiplier=0.0)
    build_objective([term], input_parameters={})({"x": jnp.array(1.0)}, jnp.int32(0))
    assert called == []


def test_build_objective_single_term():
    term = ObjectiveTerm(name="sq", compute=lambda v, p: v["x"] ** 2, multiplier=3.0)
    fun = build_objective([term], input_parameters={})
    assert float(fun({"x": jnp.array(2.0)}, jnp.int32(0))) == pytest.approx(12.0)


def test_build_objective_multiple_terms():
    t1 = ObjectiveTerm(name="a", compute=lambda v, p: v["x"], multiplier=2.0)
    t2 = ObjectiveTerm(name="b", compute=lambda v, p: v["x"] * 3.0, multiplier=1.0)
    fun = build_objective([t1, t2], input_parameters={})
    # 2*1 + 1*3 = 5
    assert float(fun({"x": jnp.array(1.0)}, jnp.int32(0))) == pytest.approx(5.0)


def test_build_objective_uses_input_parameters():
    term = ObjectiveTerm(
        name="scaled",
        compute=lambda v, p: v["x"] * p["scale"],
        multiplier=1.0,
    )
    fun = build_objective([term], input_parameters={"scale": jnp.array(4.0)})
    assert float(fun({"x": jnp.array(2.0)}, jnp.int32(0))) == pytest.approx(8.0)


def test_build_objective_schedule_applied():
    """Schedule modulates the effective weight at each step."""
    schedule = lambda step: jnp.where(step == 0, 0.0, 1.0)
    term = ObjectiveTerm(
        name="s", compute=lambda v, p: v["x"], multiplier=1.0, schedule=schedule
    )
    fun = build_objective([term], input_parameters={})
    assert float(fun({"x": jnp.array(5.0)}, jnp.int32(0))) == pytest.approx(0.0)
    assert float(fun({"x": jnp.array(5.0)}, jnp.int32(1))) == pytest.approx(5.0)


# --- OptimizationProblemTemplate.instantiate ---


def _make_simple_template() -> OptimizationProblemTemplate:
    """Template that minimizes x^2."""
    term = ObjectiveTerm(name="sq", compute=lambda v, p: v["x"] ** 2, multiplier=1.0)
    return OptimizationProblemTemplate(
        terms=[term],
        initialize=lambda p, seed: {"x": jnp.array(p.get("x0", 1.0))},
    )


def test_instantiate_returns_problem():
    problem = _make_simple_template().instantiate({"x0": 2.0})
    assert isinstance(problem, OptimizationProblem)


def test_instantiate_weight_overrides():
    problem = _make_simple_template().instantiate({}, weight_overrides={"sq": 5.0})
    assert problem.terms[0].multiplier == 5.0


def test_instantiate_unknown_weight_override_raises():
    with pytest.raises(KeyError, match="unknown_term"):
        _make_simple_template().instantiate({}, weight_overrides={"unknown_term": 1.0})


def test_instantiate_pydantic_validates_ok():
    class Params(BaseModel):
        value: float

    term = ObjectiveTerm(name="t", compute=lambda v, p: v["x"], multiplier=1.0)
    template = OptimizationProblemTemplate(
        terms=[term],
        initialize=lambda p, seed: {"x": jnp.array(p["value"])},
        input_params_class=Params,
    )
    problem = template.instantiate({"value": 3.0})
    assert isinstance(problem, OptimizationProblem)


def test_instantiate_pydantic_validation_error():
    class Params(BaseModel):
        value: float

    term = ObjectiveTerm(name="t", compute=lambda v, p: v["x"], multiplier=1.0)
    template = OptimizationProblemTemplate(
        terms=[term],
        initialize=lambda p, seed: {"x": jnp.array(p["value"])},
        input_params_class=Params,
    )
    with pytest.raises(ValidationError):
        template.instantiate({"value": {"nested": "dict"}})


# --- OptimizationProblem.optimize ---


def _simple_problem(x0: float = 3.0) -> OptimizationProblem:
    """Minimize x^2 starting from x0."""
    return _make_simple_template().instantiate({"x0": x0})


def test_optimize_returns_optim_vars_and_history():
    optim_vars, history = _simple_problem().optimize(
        OptimConfig(n_iters=10, learning_rate=0.1), callback=_NO_PRINT
    )
    assert isinstance(optim_vars, dict)
    assert isinstance(history, list)


def test_optimize_minimizes():
    optim_vars, _ = _simple_problem(x0=3.0).optimize(
        OptimConfig(n_iters=500, learning_rate=0.01), callback=_NO_PRINT
    )
    assert abs(float(optim_vars["x"])) < 0.5


def test_optimize_history_structure():
    _, history = _simple_problem().optimize(
        OptimConfig(n_iters=20, learning_rate=0.01), track_every=5, callback=_NO_PRINT
    )
    assert len(history) == 4  # steps 0, 5, 10, 15
    for record in history:
        assert "iteration" in record
        assert "total" in record
        assert "sq" in record


def test_optimize_track_every():
    _, history = _simple_problem().optimize(
        OptimConfig(n_iters=100, learning_rate=0.01), track_every=25, callback=_NO_PRINT
    )
    assert [r["iteration"] for r in history] == [0, 25, 50, 75]


def test_optimize_callback_called_every_iteration():
    calls = []
    _simple_problem().optimize(
        OptimConfig(n_iters=5, learning_rate=0.01),
        callback=lambda i, *_: calls.append(i),
    )
    assert calls == [0, 1, 2, 3, 4]
