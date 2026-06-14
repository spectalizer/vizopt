"""Optimization using JAX and Optax"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from jax import Array

from .base import Callback, OptimVars, default_print_callback


def optimize_gradient_descent(
    params: OptimVars,
    fun_to_minimize: Callable[[OptimVars, Array], Array],
    learning_rate: float = 0.002,
    b1: float = 0.95,
    b2: float = 0.95,
    n_iters: int = 1000,
    callback: Callback | None = None,
    decay_lr_to: float = 0.0,
) -> tuple[OptimVars, float]:
    """Minimize a function by gradient descent.

    Args:
        params: Initial optimization variables.
        fun_to_minimize: Scalar-valued function to minimize.
        learning_rate: Peak learning rate.
        b1: Adam beta1.
        b2: Adam beta2.
        n_iters: Number of optimization steps.
        callback: Called each iteration with (i_iter, loss, params, grads).
        decay_lr_to: Final learning rate as a fraction of `learning_rate`.
            0.0 means full cosine decay to zero; 1.0 means constant rate.
    """
    if callback is None:
        callback = default_print_callback
    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=n_iters,
        alpha=decay_lr_to,
    )
    optimizer = optax.chain(optax.scale_by_adam(b1=b1, b2=b2), optax.scale_by_learning_rate(schedule))
    opt_state = optimizer.init(params)

    @jax.jit
    def perform_optim_step(params, opt_state, step):
        """Do one gradient-based optimization step"""
        loss_value, grads = jax.value_and_grad(lambda p: fun_to_minimize(p, step))(
            params
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value, grads

    for i_iter in range(n_iters):
        step = jnp.int32(i_iter)
        params, opt_state, loss_value, grads = perform_optim_step(
            params, opt_state, step
        )
        if callback is not None:
            callback(i_iter, loss_value, params, grads)

    return params, float(loss_value)
