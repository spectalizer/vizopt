"""Steppable optimization sessions.

An `OptimizationSession` owns the optimizer state of one run and advances it
on demand, so that something outside the loop (a batch `optimize()` call, an
interactive server, a notebook cell) decides when to step and may change the
state in between: move or pin variables, change term weights, or restart the
learning-rate decay.

Everything that may change between steps lives in a `controls` pytree passed
as an argument to the jitted step function, so such changes never trigger a
recompilation.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import optax
from jax import Array

if TYPE_CHECKING:
    from .base import OptimConfig, OptimizationProblem, OptimVars

StepFunction = Callable[[Any, Any, Array, dict], tuple[Any, Any, Array, Any]]
"""A jitted `(params, opt_state, step, controls) -> (params, opt_state, loss, grads)`."""


@dataclass
class StepResult:
    """Outcome of the last step performed by `OptimizationSession.step`.

    Attributes:
        iteration: Index of the step that was just performed.
        loss: Loss value evaluated before that step's update.
        grads: Gradients of the loss, in the optimizer's (scaled) space.
    """

    iteration: int
    loss: Array
    grads: Any


def _cosine_learning_rate(step: Array, controls: dict) -> Array:
    """Cosine-decayed learning rate, restarted at `controls["lr_start"]`.

    Matches `optax.cosine_decay_schedule`, but with every parameter traced so
    that it can be changed between steps without recompiling.
    """
    elapsed = (step - controls["lr_start"]).astype(jnp.float32)
    frac = jnp.clip(elapsed / controls["lr_decay_steps"], 0.0, 1.0)
    cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * frac))
    alpha = controls["lr_final_fraction"]
    return controls["learning_rate"] * ((1.0 - alpha) * cosine + alpha)


def make_step_function(
    fun_to_minimize: Callable[[Any, Array, dict[str, Array]], Array],
    b1: float,
    b2: float,
) -> tuple[StepFunction, optax.GradientTransformation]:
    """Build the jitted Adam step used by `OptimizationSession`.

    After each Adam update, pinned entries (`controls["pin_mask"]`) are
    projected back onto `controls["pin_values"]`. Gradients are not masked,
    so Adam's moment estimates keep tracking the force acting on a pinned
    entry and it does not jump when released.

    Args:
        fun_to_minimize: Loss `fun(params, step, weights) -> scalar`.
        b1: Adam beta1.
        b2: Adam beta2.

    Returns:
        The jitted step function and the underlying Adam transformation
        (needed to initialize the optimizer state).
    """
    adam = optax.scale_by_adam(b1=b1, b2=b2)

    @jax.jit
    def step_function(params, opt_state, step, controls):
        loss_value, grads = jax.value_and_grad(
            lambda p: fun_to_minimize(p, step, controls["weights"])
        )(params)
        updates, opt_state = adam.update(grads, opt_state)
        learning_rate = _cosine_learning_rate(step, controls)
        params = {
            k: jnp.where(
                controls["pin_mask"][k],
                controls["pin_values"][k],
                params[k] - learning_rate * updates[k],
            )
            for k in params
        }
        return params, opt_state, loss_value, grads

    return step_function, adam


class OptimizationSession:
    """A steppable, interactively steerable optimization run.

    Created via `OptimizationProblem.session` (or `VizOptimizer.session`).
    The optimizer works in scaled space (see `var_scales` on
    `OptimizationProblem`), but every public method of the session takes
    and returns physical-space values.

    Example:
        session = problem.session(OptimConfig(learning_rate=0.01))
        session.step(100)
        session.pin("positions", 3, value=[0.0, 1.0])
        session.reheat()
        session.step(100)
        positions = session.vars["positions"]

    Args:
        problem: The problem to optimize.
        config: Optimizer settings. `n_iters` sets the length of the
            learning-rate decay (and the step at which term schedules are
            evaluated for the `_unscheduled` records); the session itself can
            be stepped indefinitely.
        seed: Random seed passed to the problem's `initialize`. Defaults to
            `config.seed`.
    """

    def __init__(
        self,
        problem: "OptimizationProblem",
        config: "OptimConfig",
        seed: int | None = None,
    ) -> None:
        self.problem = problem
        self.config = config
        self.iteration = 0
        self.last_step: StepResult | None = None

        initial_vars = problem.initialize(
            problem.input_parameters, config.seed if seed is None else seed
        )
        var_scales = problem.var_scales or {}
        self._scales = {k: jnp.asarray(var_scales.get(k, 1.0)) for k in initial_vars}
        self._params = {
            k: jnp.asarray(v) / self._scales[k] for k, v in initial_vars.items()
        }

        self._step_function, adam = problem._step_function(config.b1, config.b2)
        self._opt_state = adam.init(self._params)
        self._controls: dict = {
            "weights": {
                t.name: jnp.float32(t.multiplier)
                for t in problem.terms
                if t.multiplier != 0.0
            },
            "learning_rate": jnp.float32(config.learning_rate),
            "lr_start": jnp.int32(0),
            "lr_decay_steps": jnp.float32(max(config.n_iters, 1)),
            "lr_final_fraction": jnp.float32(config.decay_lr_to),
            "pin_mask": {k: jnp.zeros(v.shape, bool) for k, v in self._params.items()},
            "pin_values": dict(self._params),
        }

    # --- stepping ---

    def step(self, n: int = 1) -> StepResult:
        """Perform `n` optimization steps.

        Args:
            n: Number of steps to perform.

        Returns:
            The result of the last step performed.
        """
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}.")
        for _ in range(n):
            self._params, self._opt_state, loss_value, grads = self._step_function(
                self._params, self._opt_state, jnp.int32(self.iteration), self._controls
            )
            self.last_step = StepResult(self.iteration, loss_value, grads)
            self.iteration += 1
        assert self.last_step is not None
        return self.last_step

    # --- state access ---

    @property
    def vars(self) -> "OptimVars":
        """Current optimization variables, in physical space."""
        return {k: v * self._scales[k] for k, v in self._params.items()}

    @property
    def weights(self) -> dict[str, float]:
        """Current multiplier of every term (0.0 for inactive terms)."""
        return {
            t.name: float(self._controls["weights"].get(t.name, 0.0))
            for t in self.problem.terms
        }

    def is_pinned(self, name: str) -> Array:
        """Boolean mask of the pinned entries of variable `name`."""
        return self._controls["pin_mask"][name]

    def term_values(self) -> dict[str, Array]:
        """Raw (unweighted) value of every term at the current variables."""
        return self.problem._compute_all_terms(self.vars)

    def record(self) -> dict:
        """Per-term history record for the last step performed.

        Returns:
            A dict with keys `"iteration"`, `"total"`, `"total_unscheduled"`,
            and, per term, its schedule-weighted value, plus the same with
            suffixes `_unscheduled` (weighted at the final scheduled step
            `config.n_iters - 1`) and `_unweighted` (raw).

        Raises:
            ValueError: If no step has been performed yet.
        """
        if self.last_step is None:
            raise ValueError("No step performed yet — call step() first.")
        step = jnp.int32(self.last_step.iteration)
        final_step = jnp.int32(self.config.n_iters - 1)
        weights = self.weights
        term_values = self.term_values()
        record: dict = {
            "iteration": self.last_step.iteration,
            "total": float(self.last_step.loss),
        }
        unscheduled_total = 0.0
        for term in self.problem.terms:
            raw = float(term_values[term.name])
            sched = 1.0 if term.schedule is None else float(term.schedule(step))
            end_sched = (
                1.0 if term.schedule is None else float(term.schedule(final_step))
            )
            record[term.name] = raw * weights[term.name] * sched
            record[f"{term.name}_unscheduled"] = raw * weights[term.name] * end_sched
            record[f"{term.name}_unweighted"] = raw
            unscheduled_total += raw * weights[term.name] * end_sched
        record["total_unscheduled"] = unscheduled_total
        return record

    # --- steering ---

    def set_value(self, name: str, index: Any, value: Any) -> None:
        """Overwrite (part of) a variable, in physical space.

        Unlike `pin`, the optimizer is free to move the entry again on the
        next step.

        Args:
            name: Variable name, a key of `vars`.
            index: Numpy-style index into the variable (e.g. `3` for the
                fourth row of an `(N, 2)` array), or `None` for the whole
                variable.
            value: New physical value, broadcastable to the indexed part.
        """
        physical = self._params[name] * self._scales[name]
        if index is None:
            physical = jnp.broadcast_to(jnp.asarray(value), physical.shape)
        else:
            physical = physical.at[index].set(value)
        self._params[name] = (physical / self._scales[name]).astype(
            self._params[name].dtype
        )

    def pin(self, name: str, index: Any = None, value: Any = None) -> None:
        """Hold (part of) a variable fixed, optionally moving it first.

        Pinned entries are reset to their pinned value after every step,
        while the remaining variables keep optimizing around them. Pinning
        an already-pinned entry with a new `value` moves it (this is how a
        drag is implemented).

        Args:
            name: Variable name, a key of `vars`.
            index: Numpy-style index into the variable, or `None` for the
                whole variable.
            value: Optional physical value to move the entries to; when
                omitted, they are pinned where they currently are.
        """
        if value is not None:
            self.set_value(name, index, value)
        mask = self._controls["pin_mask"][name]
        if index is None:
            mask = jnp.ones_like(mask)
        else:
            mask = mask.at[index].set(True)
        self._controls["pin_mask"] = {**self._controls["pin_mask"], name: mask}
        self._controls["pin_values"] = {
            **self._controls["pin_values"],
            name: self._params[name],
        }

    def unpin(self, name: str, index: Any = None) -> None:
        """Release pinned entries of a variable.

        Args:
            name: Variable name, a key of `vars`.
            index: Numpy-style index into the variable, or `None` to release
                every pinned entry of the variable.
        """
        mask = self._controls["pin_mask"][name]
        if index is None:
            mask = jnp.zeros_like(mask)
        else:
            mask = mask.at[index].set(False)
        self._controls["pin_mask"] = {**self._controls["pin_mask"], name: mask}

    def unpin_all(self) -> None:
        """Release every pinned entry of every variable."""
        for name in self._params:
            self.unpin(name)

    def set_weight(self, name: str, multiplier: float) -> None:
        """Change the multiplier of a term for subsequent steps.

        Args:
            name: Term name.
            multiplier: New multiplier. Setting it to 0.0 disables the term's
                contribution (the term is still evaluated).

        Raises:
            KeyError: If no term has this name.
            ValueError: If the term was inactive (multiplier 0.0) when the
                problem was built; such terms are excluded from the compiled
                objective and cannot be enabled live.
        """
        term_names = {t.name for t in self.problem.terms}
        if name not in term_names:
            raise KeyError(f"Unknown term name: {name!r}")
        if name not in self._controls["weights"]:
            raise ValueError(
                f"Term {name!r} had multiplier 0.0 when the problem was built "
                "and is excluded from the objective; give it a non-zero "
                "multiplier at instantiation to adjust it live."
            )
        self._controls["weights"] = {
            **self._controls["weights"],
            name: jnp.float32(multiplier),
        }

    def reheat(
        self, n_iters: int | None = None, learning_rate: float | None = None
    ) -> None:
        """Restart the learning-rate decay from the current iteration.

        Use after an interaction (e.g. a drag) to let the layout re-settle,
        in the spirit of d3-force's alpha.

        Args:
            n_iters: Length of the new decay; defaults to `config.n_iters`.
            learning_rate: New peak learning rate; defaults to the current one.
        """
        self._controls["lr_start"] = jnp.int32(self.iteration)
        if n_iters is not None:
            self._controls["lr_decay_steps"] = jnp.float32(max(n_iters, 1))
        if learning_rate is not None:
            self._controls["learning_rate"] = jnp.float32(learning_rate)
