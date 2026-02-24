"""Base classes"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from jax import Array

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
