from collections.abc import Callable
from typing import Any, TypeVar

import jax
import optax
from jax import Array

Params = TypeVar("Params")
Callback = Callable[[int, Array, Any, Any], None]


def default_print_callback(i_iter: int, loss_value: Array, *_: Any) -> None:
    """Print the loss value after every nth optimization iteration"""
    if i_iter % 10 == 0:
        print(f"Iteration {i_iter}: loss = {loss_value}")


def optimize_gradient_descent(
    params: Params,
    fun_to_minimize: Callable[[Params], Array],
    learning_rate: float = 0.001,
    n_iters: int = 1000,
    callback: Callback | None = None,
) -> tuple[Params, Array]:
    """Minimize a function by gradient descent"""
    if callback is None:
        callback = default_print_callback
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def perform_optim_step(params, opt_state):
        """Do one gradient-based optimization step"""
        loss_value, grads = jax.value_and_grad(fun_to_minimize)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value, grads

    for i_iter in range(n_iters):
        params, opt_state, loss_value, grads = perform_optim_step(params, opt_state)
        if callback is not None:
            callback(i_iter, loss_value, params, grads)

    return params, loss_value
