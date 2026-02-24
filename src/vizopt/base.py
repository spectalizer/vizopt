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
class OptimizationProblem(Generic[InputParams, OptimVars]):
    """An optimization problem.

    Attributes:
        input_parameters: Fixed data for the problem (not optimized).
        terms: Objective terms defining the loss function.
        initialize: Callable that produces initial optimization variables
            from input_parameters.
    """

    input_parameters: InputParams
    terms: list[ObjectiveTerm]
    initialize: Callable[[InputParams], OptimVars]

    def optimize(
        self,
        n_iters: int = 1000,
        learning_rate: float = 0.001,
        callback: Callback | None = None,
    ) -> tuple[OptimVars, float]:
        """Run gradient descent to minimize the objective.

        Args:
            n_iters: Number of optimization iterations.
            learning_rate: Step size for Adam optimizer.
            callback: Optional callback called after each iteration with
                (iteration, loss, optim_vars, grads).

        Returns:
            Tuple of (optimized variables, final loss value).
        """
        from . import jaxopt  # lazy import to avoid circular dependency

        optim_vars = self.initialize(self.input_parameters)
        fun = build_objective(self.terms, self.input_parameters)
        return jaxopt.optimize_gradient_descent(
            optim_vars,
            fun,
            n_iters=n_iters,
            learning_rate=learning_rate,
            callback=callback,
        )
