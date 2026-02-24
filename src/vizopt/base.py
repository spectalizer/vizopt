"""Base classes"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from jax import Array
from jax import numpy as jnp

OptimVars = TypeVar("OptimVars")


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
