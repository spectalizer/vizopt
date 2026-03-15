"""Schedule factory functions for loss term weight scheduling.

Schedules are JAX-compatible callables ``(step: Array) -> Array`` that scale
a term's effective weight over the course of optimization. They must use JAX
ops (e.g. ``jnp.clip``) so they can be traced through without recompilation.
"""

from jax import numpy as jnp


def warmup(delay_frac: float, ramp_frac: float, n_iters: int):
    """Linear warmup: ramps from 0.01 to 1.0 over a window of the run.

    Args:
        delay_frac: Fraction of n_iters before ramping starts.
        ramp_frac: Fraction of n_iters over which to ramp up.
        n_iters: Total number of optimization iterations.

    Returns:
        JAX-compatible callable ``(step: Array) -> Array``.
    """
    delay = delay_frac * n_iters
    duration = max(ramp_frac * n_iters, 1.0)

    def fn(step):
        return jnp.clip((step - delay) / duration, 0.01, 1.0)

    return fn


def cooldown(peak_frac: float, ramp_frac: float, n_iters: int):
    """Linear cooldown: ramps from 1.0 down to 0.01 over a window of the run.

    Args:
        peak_frac: Fraction of n_iters at which the weight is still 1.0.
        ramp_frac: Fraction of n_iters over which to ramp down.
        n_iters: Total number of optimization iterations.

    Returns:
        JAX-compatible callable ``(step: Array) -> Array``.
    """
    peak = peak_frac * n_iters
    duration = max(ramp_frac * n_iters, 1.0)

    def fn(step):
        return jnp.clip((peak - step) / duration + 1.0, 0.01, 1.0)

    return fn


def make_term_schedules(params: dict, n_iters: int) -> dict:
    """Build a ``term_schedules`` dict from a flat parameter dict.

    Args:
        params: Dict with fractional schedule parameters:
            ``collision_delay``, ``collision_ramp``,
            ``exclusion_delay``, ``exclusion_ramp``,
            ``area_delay``, ``area_ramp``,
            ``perimeter_delay``, ``perimeter_ramp``,
            ``attraction_peak``, ``attraction_ramp``.
            All values are fractions of ``n_iters``.
        n_iters: Total number of optimization iterations. Schedules scale
            automatically — the same params work for any run length.

    Returns:
        Dict mapping term name to a JAX-compatible schedule callable.
    """
    return {
        "circle_collision": warmup(params["collision_delay"], params["collision_ramp"], n_iters),
        "exclusion":        warmup(params["exclusion_delay"], params["exclusion_ramp"], n_iters),
        "area":             warmup(params["area_delay"],      params["area_ramp"],      n_iters),
        "perimeter":        warmup(params["perimeter_delay"], params["perimeter_ramp"], n_iters),
        "set_attraction":   cooldown(params["attraction_peak"], params["attraction_ramp"], n_iters),
    }
