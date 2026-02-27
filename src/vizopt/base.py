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
class ObjectiveTerm:
    """A term in an objective function.

    Attributes:
        name: A name for the term, e.g. "total distance".
        compute: A function that computes the value of the term
            with arguments optim_vars, input_parameters
        multiplier: A multiplicative factor for the term.
    """

    name: str
    compute: Callable[[OptimVars, Any], Array]
    multiplier: float = 1.0


def build_objective(
    terms: list[ObjectiveTerm],
    input_parameters: Any,
) -> Callable[[OptimVars], Array]:
    """Build a composite objective function from a list of terms.

    Args:
        terms: Objective terms to sum, each weighted by its multiplier.
        input_parameters: Fixed data passed to each term's compute function.

    Returns:
        A callable ``fun(optim_vars) -> scalar`` suitable for gradient descent.
    """

    def fun_to_minimize(optim_vars: OptimVars) -> Array:
        return sum(
            (
                term.compute(optim_vars, input_parameters) * term.multiplier
                for term in terms
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
    """

    terms: list[ObjectiveTerm]
    initialize: Callable[[InputParams], OptimVars]
    input_params_class: type[InputParams] | None = None
    plot_configuration: Callable[[OptimVars, InputParams], None] | None = None

    def instantiate(
        self, input_parameters: InputParams
    ) -> "OptimizationProblem[InputParams, OptimVars]":
        """Create a runnable problem instance from concrete input parameters.

        If ``input_params_class`` is set, validates ``input_parameters`` via
        ``model_validate`` before creating the problem. The plain dict is passed
        through to the problem unchanged (Pydantic is used for validation only,
        so that ``input_parameters`` remains a JAX-compatible pytree).

        Args:
            input_parameters: Fixed data for this problem instance.

        Returns:
            An :class:`OptimizationProblem` ready to optimize.

        Raises:
            pydantic.ValidationError: If ``input_params_class`` is set and
                validation fails.
        """
        if self.input_params_class is not None:
            # validate only
            self.input_params_class.model_validate(input_parameters)  # type: ignore
        return OptimizationProblem(
            input_parameters, self.terms, self.initialize, self.plot_configuration
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
    """

    input_parameters: InputParams
    terms: list[ObjectiveTerm]
    initialize: Callable[[InputParams], OptimVars]
    plot_configuration: Callable[[OptimVars, InputParams], None] | None = None

    def optimize(
        self,
        n_iters: int = 1000,
        learning_rate: float = 0.001,
        callback: Callback | None = None,
        track_every: int = 10,
    ) -> tuple[OptimVars, list[dict]]:
        """Run gradient descent to minimize the objective.

        Args:
            n_iters: Number of optimization iterations.
            learning_rate: Step size for Adam optimizer.
            callback: Optional callback called after each iteration with
                (iteration, loss, optim_vars, grads).
            track_every: Record per-term history every this many iterations.

        Returns:
            Tuple of (optimized variables, history). History is a list of
            dicts with keys ``"iteration"``, ``"total"``, and one key per
            term name containing the weighted term value at that iteration.
        """
        from . import jaxopt  # lazy import to avoid circular dependency

        if callback is None:
            callback = jaxopt.default_print_callback

        history: list[dict] = []

        def tracking_callback(
            i_iter: int, loss_value: Array, optim_vars: OptimVars, grads: Any
        ) -> None:
            if i_iter % track_every == 0:
                record: dict = {"iteration": i_iter, "total": float(loss_value)}
                for term in self.terms:
                    record[term.name] = float(
                        term.compute(optim_vars, self.input_parameters)
                        * term.multiplier
                    )
                history.append(record)
            if callback is not None:
                callback(i_iter, loss_value, optim_vars, grads)

        optim_vars = self.initialize(self.input_parameters)
        fun = build_objective(self.terms, self.input_parameters)
        optim_vars_opt, _ = jaxopt.optimize_gradient_descent(
            optim_vars,
            fun,
            n_iters=n_iters,
            learning_rate=learning_rate,
            callback=tracking_callback,
        )
        return optim_vars_opt, history


def default_print_callback(i_iter: int, loss_value: Array, *_: Any) -> None:
    """Print the loss value after every nth optimization iteration"""
    if i_iter % 100 == 0:
        print(f"Iteration {i_iter}: loss = {loss_value}")
