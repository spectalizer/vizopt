"""Base classes"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from jax import Array, jit
from jax import numpy as jnp

OptimVars = TypeVar("OptimVars")
InputParams = TypeVar("InputParams")

Callback = Callable[[int, Array, Any, Any], None]


@dataclass
class OptimConfig:
    """Configuration for the gradient-descent optimizer.

    Attributes:
        n_iters: Number of optimization iterations.
        learning_rate: Step size for the Adam optimizer.
        b1: Adam exponential decay rate for the first moment. Default 0.9.
        b2: Adam exponential decay rate for the second moment. Default 0.999.
        n_restarts: Number of random restarts. The run with the lowest final
            loss is returned. Default 1 (single run).
        seed: Base random seed passed to ``initialize``. Restart ``i``
            receives ``seed + i``.
        track_every: Record per-term history every this many iterations.
    """

    n_iters: int = 1000
    learning_rate: float = 0.001
    b1: float = 0.9
    b2: float = 0.999
    decay_lr_to: float = 0.1
    n_restarts: int = 1
    seed: int = 0
    track_every: int = 10


@dataclass
class OptimizationResult(Generic[OptimVars]):
    """Result returned by [`OptimizationProblem.optimize`][vizopt.base.OptimizationProblem.optimize].

    Attributes:
        optim_vars: Optimized variables in physical (un-scaled) space.
        history: Per-iteration records. Each dict has keys ``"iteration"``,
            ``"total"``, one key per term name (schedule-weighted value), one
            per term name suffixed ``_unscheduled`` (end-weighted), and one per
            term name suffixed ``_unweighted`` (raw, un-multiplied value).
        final_loss: Scalar loss of the best run at the last iteration.
    """

    optim_vars: OptimVars
    history: list[dict]
    final_loss: float


@dataclass
class ObjectiveTerm:
    """A term in an objective function.

    Attributes:
        name: A name for the term, e.g. "total distance".
        compute: A function that computes the value of the term
            with arguments optim_vars, input_parameters
        multiplier: A multiplicative factor for the term.
        schedule: Optional JAX-compatible callable ``(step: Array) -> Array``
            that returns a scalar multiplier for the given iteration step.
            The effective weight is ``multiplier * schedule(step)``.
            Must use JAX ops (e.g. ``jnp.minimum``, ``jnp.where``) so that
            it can be traced through without recompilation.
            ``None`` means constant 1.0 (no scheduling).
    """

    name: str
    compute: Callable[[Any, Any], Array]
    multiplier: float = 1.0
    schedule: Callable[[Array], Array] | None = None


def build_objective(
    terms: list[ObjectiveTerm],
    input_parameters: Any,
    var_scales: dict | None = None,
) -> Callable[[OptimVars, Array], Array]:
    """Build a composite objective function from a list of terms.

    Args:
        terms: Objective terms to sum, each weighted by its multiplier.
        input_parameters: Fixed data passed to each term's compute function.
        var_scales: Optional per-variable scale factors. When provided, each
            ``optim_vars[k]`` is multiplied by ``var_scales[k]`` before being
            passed to any term's ``compute`` function, so the optimizer works
            in a normalised space while loss terms always receive physical
            values. Values may be scalars or arrays (broadcast over the
            variable's shape). Keys absent from ``var_scales`` are left
            unscaled.

    Returns:
        A callable ``fun(optim_vars, step) -> scalar`` suitable for gradient
        descent. ``step`` is the current iteration as a JAX int32 array and
        is passed to each term's ``schedule`` (if any).
    """

    active_terms = [t for t in terms if t.multiplier != 0.0]

    def fun_to_minimize(optim_vars: OptimVars, step: Array) -> Array:
        if var_scales is not None:
            optim_vars = {k: v * var_scales.get(k, 1.0) for k, v in optim_vars.items()}
        return sum(
            (
                term.compute(optim_vars, input_parameters)
                * term.multiplier
                * (term.schedule(step) if term.schedule is not None else 1.0)
                for term in active_terms
            ),
            jnp.zeros(()),
        )

    return fun_to_minimize


@dataclass
class OptimizationProblemTemplate(Generic[InputParams, OptimVars]):
    """A template for a class of optimization problems.

    An instance represents a specific *type* of optimization problem
    (e.g. bubble layout optimization), independently of any particular
    input data. Call :meth:`instantiate` with concrete input parameters
    to obtain a runnable :class:`OptimizationProblem`.

    If ``input_params_class`` is provided, it must be a Pydantic model class.
    ``instantiate`` will call ``model_validate`` on the supplied parameters,
    triggering Pydantic validation and coercion before the problem is created.

    Attributes:
        terms: Objective terms defining the loss function.
        initialize: Callable that produces initial optimization variables
            from input_parameters.
        input_params_class: Optional Pydantic model class for input parameters.
            When set, validation is performed at instantiation time.
        plot_configuration: Optional callable to visualize a configuration.
            Signature: ``plot_configuration(optim_vars, input_parameters)``.
        svg_configuration: Optional callable to produce SVG element specs for
            animation. Signature:
            ``svg_configuration(snapshots, input_parameters, size) -> list[dict]``
            where each dict has a ``"tag"`` key and SVG attribute keys; list
            values are animated per-frame, scalar values are static.
    """

    terms: list[ObjectiveTerm]
    initialize: Callable[[InputParams, int], OptimVars]
    input_params_class: type[InputParams] | None = None
    plot_configuration: Callable[[OptimVars, InputParams], None] | None = None
    svg_configuration: Callable[[list, InputParams, int], list[dict]] | None = None

    def instantiate(
        self,
        input_parameters: InputParams,
        weight_overrides: dict[str, float] | None = None,
        var_scales: dict | None = None,
    ) -> "OptimizationProblem[InputParams, OptimVars]":
        """Create a runnable problem instance from concrete input parameters.

        If ``input_params_class`` is set, validates ``input_parameters`` via
        ``model_validate`` before creating the problem. The plain dict is passed
        through to the problem unchanged (Pydantic is used for validation only,
        so that ``input_parameters`` remains a JAX-compatible pytree).

        Args:
            input_parameters: Fixed data for this problem instance.
            weight_overrides: Optional mapping of term name to multiplier.
                Overrides the default multiplier for the named terms.
                Unknown names raise ``KeyError``.
            var_scales: Optional per-variable scale factors used to normalise
                ``optim_vars`` during optimisation. See :func:`build_objective`
                for details. Values may be scalars or arrays.

        Returns:
            An :class:`OptimizationProblem` ready to optimize.

        Raises:
            KeyError: If a name in ``weight_overrides`` does not match any term.
            pydantic.ValidationError: If ``input_params_class`` is set and
                validation fails.
        """
        if self.input_params_class is not None:
            # validate only
            self.input_params_class.model_validate(input_parameters)  # type: ignore
        terms = self.terms
        if weight_overrides:
            term_names = {term.name for term in terms}
            unknown = set(weight_overrides) - term_names
            if unknown:
                raise KeyError(f"Unknown term name(s) in weight_overrides: {unknown}")
            terms = [
                ObjectiveTerm(
                    t.name, t.compute, weight_overrides.get(t.name, t.multiplier)
                )
                for t in terms
            ]
        return OptimizationProblem(
            input_parameters,
            terms,
            self.initialize,
            self.plot_configuration,
            self.svg_configuration,
            var_scales,
        )


@dataclass
class OptimizationProblem(Generic[InputParams, OptimVars]):
    """An optimization problem.

    Attributes:
        input_parameters: Fixed data for the problem (not optimized).
        terms: Objective terms defining the loss function.
        initialize: Callable that produces initial optimization variables
            from input_parameters.
        plot_configuration: Optional callable to visualize a configuration.
            Signature: ``plot_configuration(optim_vars, input_parameters)``.
        svg_configuration: Optional callable to produce SVG element specs for
            animation. Signature:
            ``svg_configuration(snapshots, input_parameters, size) -> list[dict]``.
        var_scales: Optional per-variable scale factors. See
            :func:`build_objective` for details.
    """

    input_parameters: InputParams
    terms: list[ObjectiveTerm]
    initialize: Callable[[InputParams, int], OptimVars]
    plot_configuration: Callable[[OptimVars, InputParams], None] | None = None
    svg_configuration: Callable[[list, InputParams, int], list[dict]] | None = None
    var_scales: dict | None = None
    result: "OptimizationResult[OptimVars] | None" = field(default=None, init=False, repr=False)

    def plot(self, **kwargs) -> None:
        """Plot the last optimization result using ``plot_configuration``.

        Keyword arguments are forwarded to ``plot_configuration``, allowing
        optional display flags (e.g. ``show_arrows=True``).

        Raises:
            ValueError: If ``plot_configuration`` is not set or ``optimize()``
                has not been called yet.
        """
        if self.plot_configuration is None:
            raise ValueError("plot_configuration is not set on this problem.")
        if self.result is None:
            raise ValueError("No result yet — call optimize() first.")
        self.plot_configuration(self.result.optim_vars, self.input_parameters, **kwargs)

    def optimize(
        self,
        optim_config: OptimConfig | None = None,
        callback: Callback | None = None,
    ) -> "OptimizationResult[OptimVars]":
        """Run gradient descent to minimize the objective.

        When ``optim_config.n_restarts > 1``, the optimization is run that
        many times with seeds ``seed``, ``seed + 1``, …. The result with the
        lowest final loss is returned.

        Args:
            optim_config: Optimizer settings (iterations, learning rate, seeds,
                restarts, track_every). Uses [OptimConfig][vizopt.base.OptimConfig]
                defaults when ``None``.
            callback: Optional callback called after each iteration with
                (iteration, loss, optim_vars, grads).

        Returns:
            An [OptimizationResult][vizopt.base.OptimizationResult] with the
            optimized variables, per-term history, and final loss of the best run.
        """
        from . import jaxopt  # lazy import to avoid circular dependency

        config = optim_config or OptimConfig()
        has_schedules = any(t.schedule is not None for t in self.terms)
        final_step = jnp.int32(config.n_iters - 1)

        # When schedules are active and no user callback is provided, tracking_callback
        # handles printing (showing both scheduled and unscheduled totals). Otherwise
        # fall back to the standard print callback.
        user_callback = callback
        if user_callback is None and not has_schedules:
            user_callback = jaxopt.default_print_callback

        fun = build_objective(self.terms, self.input_parameters, self.var_scales)

        _terms = self.terms
        _input_parameters = self.input_parameters

        @jit
        def _compute_all_terms(physical_vars):
            return {
                term.name: term.compute(physical_vars, _input_parameters)
                for term in _terms
            }

        best_vars: OptimVars | None = None
        best_history: list[dict] = []
        best_loss = float("inf")

        for restart in range(config.n_restarts):
            history: list[dict] = []
            _last_unscheduled = [0.0]  # updated every track_every steps

            def tracking_callback(
                i_iter: int,
                loss_value: Array,
                optim_vars: OptimVars,
                grads: Any,
                _history: list[dict] = history,
                _last_unscheduled: list[float] = _last_unscheduled,
            ) -> None:
                if self.var_scales is not None:
                    physical_vars = {
                        k: v * self.var_scales.get(k, 1.0)
                        for k, v in optim_vars.items()
                    }
                else:
                    physical_vars = optim_vars
                if (i_iter % config.track_every == 0) or i_iter == config.n_iters - 1:
                    record: dict = {"iteration": i_iter, "total": float(loss_value)}
                    step = jnp.int32(i_iter)
                    term_values = _compute_all_terms(physical_vars)
                    unscheduled_total = 0.0
                    for term in self.terms:
                        raw = float(term_values[term.name])
                        sched = (
                            float(term.schedule(step))
                            if term.schedule is not None
                            else 1.0
                        )
                        end_sched = (
                            float(term.schedule(final_step))
                            if term.schedule is not None
                            else 1.0
                        )
                        record[term.name] = raw * term.multiplier * sched
                        record[f"{term.name}_unscheduled"] = (
                            raw * term.multiplier * end_sched
                        )
                        record[f"{term.name}_unweighted"] = raw
                        unscheduled_total += raw * term.multiplier * end_sched
                    _last_unscheduled[0] = unscheduled_total
                    _history.append(record)
                if user_callback is not None:
                    user_callback(i_iter, loss_value, physical_vars, grads)
                elif (
                    has_schedules
                    and (i_iter % 100 == 0)
                    or i_iter == config.n_iters - 1
                ):
                    print(
                        f"Iteration {i_iter}: loss = {float(loss_value):.6f}"
                        f"  unscheduled loss = {_last_unscheduled[0]:.6f}"
                    )

            optim_vars = self.initialize(self.input_parameters, config.seed + restart)
            if self.var_scales is not None:
                optim_vars = {
                    k: v / self.var_scales.get(k, 1.0) for k, v in optim_vars.items()
                }
            optim_vars_result, final_loss = jaxopt.optimize_gradient_descent(
                optim_vars,
                fun,
                n_iters=config.n_iters,
                learning_rate=config.learning_rate,
                b1=config.b1,
                b2=config.b2,
                decay_lr_to=config.decay_lr_to,
                callback=tracking_callback,
            )
            if self.var_scales is not None:
                optim_vars_result = {
                    k: v * self.var_scales.get(k, 1.0)
                    for k, v in optim_vars_result.items()
                }
            if best_vars is None or final_loss < best_loss:
                best_loss = final_loss
                best_vars = optim_vars_result
                best_history = history

        assert best_vars is not None
        self.result = OptimizationResult(best_vars, best_history, best_loss)
        return self.result


def default_print_callback(i_iter: int, loss_value: Array, *_: Any) -> None:
    """Print the loss value after every nth optimization iteration"""
    if i_iter % 100 == 0:
        print(f"Iteration {i_iter}: loss = {loss_value}")


class VizOptimizer(ABC):
    """Base class for user-facing visualization optimizers.

    Subclasses implement :meth:`_build_problem` to turn stored hyperparameters
    into a configured :class:`OptimizationProblem`. The base class handles the
    optimize → plot lifecycle and stores fitted state in ``problem_`` and
    ``result_`` after :meth:`optimize` is called.
    """

    @abstractmethod
    def _build_problem(self) -> OptimizationProblem:
        """Build an :class:`OptimizationProblem` from stored hyperparameters."""
        ...

    def optimize(
        self,
        optim_config: OptimConfig | None = None,
        callback: Callback | None = None,
    ) -> OptimizationResult:
        """Build the problem and run gradient descent.

        Args:
            optim_config: Optimizer settings. Uses :class:`OptimConfig` defaults
                when ``None``.
            callback: Optional callback ``(iteration, loss, optim_vars, grads)``.

        Returns:
            An :class:`OptimizationResult` with optimized variables, per-term
            history, and final loss.
        """
        self.problem_: OptimizationProblem = self._build_problem()
        self.result_: OptimizationResult = self.problem_.optimize(optim_config, callback=callback)
        return self.result_

    def plot(self, **kwargs) -> None:
        """Plot the last optimization result.

        Raises:
            ValueError: If :meth:`optimize` has not been called yet.
        """
        if not hasattr(self, "problem_"):
            raise ValueError("No result yet — call optimize() first.")
        self.problem_.plot(**kwargs)
