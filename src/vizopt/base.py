"""Base classes"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from jax import Array
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
        n_restarts: Number of random restarts. The run with the lowest final
            loss is returned. Default 1 (single run).
        seed: Base random seed passed to ``initialize``. Restart ``i``
            receives ``seed + i``.
    """

    n_iters: int = 1000
    learning_rate: float = 0.001
    n_restarts: int = 1
    seed: int = 0


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
    compute: Callable[[OptimVars, Any], Array]
    multiplier: float = 1.0
    schedule: Callable[[Array], Array] | None = None


def build_objective(
    terms: list[ObjectiveTerm],
    input_parameters: Any,
) -> Callable[[OptimVars, Array], Array]:
    """Build a composite objective function from a list of terms.

    Args:
        terms: Objective terms to sum, each weighted by its multiplier.
        input_parameters: Fixed data passed to each term's compute function.

    Returns:
        A callable ``fun(optim_vars, step) -> scalar`` suitable for gradient
        descent. ``step`` is the current iteration as a JAX int32 array and
        is passed to each term's ``schedule`` (if any).
    """

    active_terms = [t for t in terms if t.multiplier != 0.0]

    def fun_to_minimize(optim_vars: OptimVars, step: Array) -> Array:
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
    ) -> "OptimizationProblem[InputParams, OptimVars]":
        """Create a runnable problem instance from concrete input parameters.

        If ``input_params_class`` is set, validates ``input_parameters`` via
        ``model_validate`` before creating the problem. The plain dict is passed
        through to the problem unchanged (Pydantic is used for validation only,
        so that ``input_parameters`` remains a JAX-compatible pytree).

        Args:
            input_parameters: Fixed data for this problem instance.
            x: Optional mapping of term name to multiplier.
                Overrides the default multiplier for the named terms.
                Unknown names raise ``KeyError``.

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
            input_parameters, terms, self.initialize, self.plot_configuration,
            self.svg_configuration,
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
    """

    input_parameters: InputParams
    terms: list[ObjectiveTerm]
    initialize: Callable[[InputParams, int], OptimVars]
    plot_configuration: Callable[[OptimVars, InputParams], None] | None = None
    svg_configuration: Callable[[list, InputParams, int], list[dict]] | None = None

    def optimize(
        self,
        optim_config: OptimConfig | None = None,
        callback: Callback | None = None,
        track_every: int = 10,
    ) -> tuple[OptimVars, list[dict]]:
        """Run gradient descent to minimize the objective.

        When ``optim_config.n_restarts > 1``, the optimization is run that
        many times with seeds ``seed``, ``seed + 1``, …. The result with the
        lowest final loss is returned.

        Args:
            optim_config: Optimizer settings (iterations, learning rate, seeds,
                restarts). Uses :class:`OptimConfig` defaults when ``None``.
            callback: Optional callback called after each iteration with
                (iteration, loss, optim_vars, grads).
            track_every: Record per-term history every this many iterations.

        Returns:
            Tuple of (optimized variables, history). History is a list of
            dicts with keys ``"iteration"``, ``"total"``, and one key per
            term name containing the weighted term value at that iteration.
            When using multiple restarts, history corresponds to the best run.
        """
        from . import jaxopt  # lazy import to avoid circular dependency

        config = optim_config or OptimConfig()

        if callback is None:
            callback = jaxopt.default_print_callback

        fun = build_objective(self.terms, self.input_parameters)

        best_vars: OptimVars | None = None
        best_history: list[dict] = []
        best_loss = float("inf")

        for restart in range(config.n_restarts):
            history: list[dict] = []

            def tracking_callback(
                i_iter: int, loss_value: Array, optim_vars: OptimVars, grads: Any,
                _history: list[dict] = history,
            ) -> None:
                if i_iter % track_every == 0:
                    record: dict = {"iteration": i_iter, "total": float(loss_value)}
                    step = jnp.int32(i_iter)
                    for term in self.terms:
                        sched = float(term.schedule(step)) if term.schedule is not None else 1.0
                        record[term.name] = float(
                            term.compute(optim_vars, self.input_parameters)
                            * term.multiplier
                            * sched
                        )
                    _history.append(record)
                if callback is not None:
                    callback(i_iter, loss_value, optim_vars, grads)

            optim_vars = self.initialize(self.input_parameters, config.seed + restart)
            optim_vars_result, final_loss = jaxopt.optimize_gradient_descent(
                optim_vars,
                fun,
                n_iters=config.n_iters,
                learning_rate=config.learning_rate,
                callback=tracking_callback,
            )
            if final_loss < best_loss:
                best_loss = final_loss
                best_vars = optim_vars_result
                best_history = history

        assert best_vars is not None
        return best_vars, best_history


def default_print_callback(i_iter: int, loss_value: Array, *_: Any) -> None:
    """Print the loss value after every nth optimization iteration"""
    if i_iter % 100 == 0:
        print(f"Iteration {i_iter}: loss = {loss_value}")
