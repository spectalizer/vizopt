"""Schedule factory functions for loss term weight scheduling.

Schedules are JAX-compatible callables `(step: Array) -> Array` that scale
a term's effective weight over the course of optimization. They must use JAX
ops (e.g. `jnp.clip`) so they can be traced through without recompilation.
"""

from dataclasses import dataclass, field

from jax import numpy as jnp


def warmup(delay_frac: float, ramp_frac: float, n_iters: int):
    """Linear warmup: ramps from 0.01 to 1.0 over a window of the run.

    Args:
        delay_frac: Fraction of n_iters before ramping starts.
        ramp_frac: Fraction of n_iters over which to ramp up.
        n_iters: Total number of optimization iterations.

    Returns:
        JAX-compatible callable `(step: Array) -> Array`.
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
        JAX-compatible callable `(step: Array) -> Array`.
    """
    peak = peak_frac * n_iters
    duration = max(ramp_frac * n_iters, 1.0)

    def fn(step):
        return jnp.clip((peak - step) / duration + 1.0, 0.01, 1.0)

    return fn


@dataclass
class TermSchedules:
    """Per-term weight schedules with warmup/cooldown categorization.

    Attributes:
        schedules: Dict mapping term name to a JAX-compatible schedule callable,
            ready to pass as `term_schedules` to any optimizer.
        quality_terms: Term names whose schedules end at 1 (warmup). Use these
            when computing a quality score from the final loss history.
        relaxation_terms: Term names whose schedules end at 0 (cooldown). These
            are expected to fade out and should be excluded from quality scoring.
    """

    schedules: dict = field(default_factory=dict)
    quality_terms: set = field(default_factory=set)
    relaxation_terms: set = field(default_factory=set)


def make_term_schedules(params: dict, n_iters: int) -> TermSchedules:
    """Build a :class:`TermSchedules` from a flat parameter dict.

    Term names and schedule types are inferred from param key suffixes:

    - `{term}_delay` + `{term}_ramp` → :func:`warmup` for `term`
    - `{term}_peak`  + `{term}_ramp` → :func:`cooldown` for `term`

    Args:
        params: Flat dict whose keys follow the naming convention above.
            All values are fractions of `n_iters` in `[0, 1]`.
        n_iters: Total number of optimization iterations. Schedules scale
            automatically — the same params work for any run length.

    Returns:
        :class:`TermSchedules` with `schedules`, `quality_terms`, and
        `relaxation_terms` populated from `params`.
    """
    schedules: dict = {}
    quality_terms: set = set()
    relaxation_terms: set = set()

    for key, val in params.items():
        if key.endswith("_delay"):
            term = key[:-6]
            ramp = params.get(f"{term}_ramp")
            if ramp is not None:
                schedules[term] = warmup(val, ramp, n_iters)
                quality_terms.add(term)
        elif key.endswith("_peak"):
            term = key[:-5]
            ramp = params.get(f"{term}_ramp")
            if ramp is not None:
                schedules[term] = cooldown(val, ramp, n_iters)
                relaxation_terms.add(term)
        elif not key.endswith("_ramp"):
            raise ValueError(
                f"Unrecognised schedule param {key!r}. "
                "Keys must end with '_delay', '_peak', or '_ramp'."
            )

    return TermSchedules(
        schedules=schedules,
        quality_terms=quality_terms,
        relaxation_terms=relaxation_terms,
    )
