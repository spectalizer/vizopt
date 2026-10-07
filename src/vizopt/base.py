"""Base classes"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from jax import Array, jit
from jax import numpy as jnp
from pydantic import BaseModel

from .scene import Scene
from .session import OptimizationSession, StepFunction, make_step_function

# `optim_vars` and `input_parameters` are always plain dicts (JAX-compatible
# pytrees); these aliases document that contract. Values are left as `Any`
# because `initialize` may hand back numpy arrays that JAX only converts once
# tracing starts. `input_params_class` on a template validates the dict against
# a Pydantic model but never replaces it.
OptimVars = dict[str, Any]
InputParams = dict[str, Any]

Callback = Callable[[int, Array, Any, Any], bool | None]
"""A per-iteration optimization callback.

Called as `callback(i_iter, loss_value, optim_vars, grads)`. A callback may
return a truthy value to request the optimizer stop after the current
iteration; returning `None` (the common case — most callbacks just print or
record a snapshot) means "keep going".
"""


@dataclass
class OptimConfig:
    """Configuration for the gradient-descent optimizer.

    Attributes:
        n_iters: Number of optimization iterations.
        learning_rate: Step size for the Adam optimizer.
        b1: Adam exponential decay rate for the first moment. Default 0.9.
        b2: Adam exponential decay rate for the second moment. Default 0.999.
        n_restarts: Number of random restarts. The run with the lowest final
            loss is returned. Default 1 (single run).
        seed: Base random seed passed to `initialize`. Restart `i`
            receives `seed + i`.
        track_every: Record per-term history every this many iterations.
        early_stop_patience: If set, stop once the (unscheduled) total loss
            hasn't improved by at least a relative `early_stop_tol` fraction
            over the last `early_stop_patience` iterations. Checked at
            `track_every` granularity (piggybacking on the history recording
            that already happens then, so this adds no extra device syncs);
            values smaller than `track_every` are rounded up to it. `None`
            (default) disables early stopping.
        early_stop_tol: Relative-improvement threshold for early stopping;
            see `early_stop_patience`. Ignored when `early_stop_patience` is
            `None`.
    """

    n_iters: int = 1000
    learning_rate: float = 0.001
    b1: float = 0.9
    b2: float = 0.999
    decay_lr_to: float = 0.1
    n_restarts: int = 1
    seed: int = 0
    track_every: int = 10
    early_stop_patience: int | None = None
    early_stop_tol: float = 1e-4


@dataclass
class OptimizationResult:
    """Result returned by [`OptimizationProblem.optimize`][vizopt.base.OptimizationProblem.optimize].

    Attributes:
        optim_vars: Optimized variables in physical (un-scaled) space.
        history: Per-iteration records. Each dict has keys `"iteration"`,
            `"total"`, `"total_unscheduled"` (sum of every term's
            end-weighted value; used for `early_stop_patience` plateau
            detection, since it isn't confounded by schedule ramping), one
            key per term name (schedule-weighted value), one per term name
            suffixed `_unscheduled` (end-weighted), and one per term name
            suffixed `_unweighted` (raw, un-multiplied value).
        final_loss: Scalar loss of the best run at the last iteration.
    """

    optim_vars: OptimVars
    history: list[dict]
    final_loss: float


@dataclass
class ObjectiveTerm:
    """A term in an objective function.

    Attributes:
        name: A name for the term, e.g. "total distance".
        compute: A function that computes the value of the term
            with arguments optim_vars, input_parameters
        multiplier: A multiplicative factor for the term.
        schedule: Optional JAX-compatible callable `(step: Array) -> Array`
            that returns a scalar multiplier for the given iteration step.
            The effective weight is `multiplier * schedule(step)`.
            Must use JAX ops (e.g. `jnp.minimum`, `jnp.where`) so that
            it can be traced through without recompilation.
            `None` means constant 1.0 (no scheduling).
    """

    name: str
    compute: Callable[[Any, Any], Array]
    multiplier: float = 1.0
    schedule: Callable[[Array], Array] | None = None


def build_objective(
    terms: list[ObjectiveTerm],
    input_parameters: Any,
    var_scales: dict | None = None,
) -> Callable[..., Array]:
    """Build a composite objective function from a list of terms.

    Args:
        terms: Objective terms to sum, each weighted by its multiplier.
        input_parameters: Fixed data passed to each term's compute function.
        var_scales: Optional per-variable scale factors. When provided, each
            `optim_vars[k]` is multiplied by `var_scales[k]` before being
            passed to any term's `compute` function, so the optimizer works
            in a normalised space while loss terms always receive physical
            values. Values may be scalars or arrays (broadcast over the
            variable's shape). Keys absent from `var_scales` are left
            unscaled.

    Returns:
        A callable `fun(optim_vars, step, weights=None) -> scalar` suitable
        for gradient descent. `step` is the current iteration as a JAX int32
        array and is passed to each term's `schedule` (if any). `weights`
        optionally maps term names to multipliers that replace the terms'
        own `multiplier`; passing them as (traced) arguments lets weights
        change between steps without recompiling. Terms with
        `multiplier=0.0` are excluded entirely, whatever `weights` says.
    """

    active_terms = [t for t in terms if t.multiplier != 0.0]

    def fun_to_minimize(
        optim_vars: OptimVars, step: Array, weights: dict | None = None
    ) -> Array:
        if var_scales is not None:
            optim_vars = {k: v * var_scales.get(k, 1.0) for k, v in optim_vars.items()}
        return sum(
            (
                term.compute(optim_vars, input_parameters)
                * (term.multiplier if weights is None else weights[term.name])
                * (term.schedule(step) if term.schedule is not None else 1.0)
                for term in active_terms
            ),
            jnp.zeros(()),
        )

    return fun_to_minimize


@dataclass
class OptimizationProblemTemplate:
    """A template for a class of optimization problems.

    An instance represents a specific *type* of optimization problem
    (e.g. bubble layout optimization), independently of any particular
    input data. Call :meth:`instantiate` with concrete input parameters
    to obtain a runnable :class:`OptimizationProblem`.

    If `input_params_class` is provided, it must be a Pydantic model class.
    `instantiate` will call `model_validate` on the supplied parameters,
    triggering Pydantic validation and coercion before the problem is created.

    Attributes:
        terms: Objective terms defining the loss function.
        initialize: Callable that produces initial optimization variables
            from input_parameters.
        input_params_class: Optional Pydantic model class for input parameters.
            When set, validation is performed at instantiation time.
        plot_configuration: Optional callable to visualize a configuration.
            Signature: `plot_configuration(optim_vars, input_parameters)`.
        svg_configuration: Optional callable to produce SVG element specs for
            animation. Signature:
            `svg_configuration(snapshots, input_parameters, size) -> list[dict]`
            where each dict has a `"tag"` key and SVG attribute keys; list
            values are animated per-frame, scalar values are static.
        scene_configuration: Optional callable describing a configuration as
            a [Scene][vizopt.scene.Scene], for rendering outside of Python
            (e.g. a live browser frontend). Signature:
            `scene_configuration(optim_vars, input_parameters) -> Scene`.
    """

    terms: list[ObjectiveTerm]
    initialize: Callable[[InputParams, int], OptimVars]
    input_params_class: type[BaseModel] | None = None
    plot_configuration: Callable[[OptimVars, InputParams], None] | None = None
    svg_configuration: Callable[[list, InputParams, int], list[dict]] | None = None
    scene_configuration: Callable[[OptimVars, InputParams], Scene] | None = None

    def instantiate(
        self,
        input_parameters: InputParams,
        weight_overrides: dict[str, float] | None = None,
        var_scales: dict | None = None,
    ) -> "OptimizationProblem":
        """Create a runnable problem instance from concrete input parameters.

        If `input_params_class` is set, validates `input_parameters` via
        `model_validate` before creating the problem. The plain dict is passed
        through to the problem unchanged (Pydantic is used for validation only,
        so that `input_parameters` remains a JAX-compatible pytree).

        Args:
            input_parameters: Fixed data for this problem instance.
            weight_overrides: Optional mapping of term name to multiplier.
                Overrides the default multiplier for the named terms.
                Unknown names raise `KeyError`.
            var_scales: Optional per-variable scale factors used to normalise
                `optim_vars` during optimisation. See :func:`build_objective`
                for details. Values may be scalars or arrays.

        Returns:
            An :class:`OptimizationProblem` ready to optimize.

        Raises:
            KeyError: If a name in `weight_overrides` does not match any term.
            pydantic.ValidationError: If `input_params_class` is set and
                validation fails.
        """
        if self.input_params_class is not None:
            # validate only
            self.input_params_class.model_validate(input_parameters)
        terms = self.terms
        if weight_overrides:
            term_names = {term.name for term in terms}
            unknown = set(weight_overrides) - term_names
            if unknown:
                raise KeyError(f"Unknown term name(s) in weight_overrides: {unknown}")
            terms = [
                replace(t, multiplier=weight_overrides.get(t.name, t.multiplier))
                for t in terms
            ]
        return OptimizationProblem(
            input_parameters=input_parameters,
            terms=terms,
            initialize=self.initialize,
            plot_configuration=self.plot_configuration,
            svg_configuration=self.svg_configuration,
            scene_configuration=self.scene_configuration,
            var_scales=var_scales,
        )


@dataclass
class OptimizationProblem:
    """An optimization problem.

    Attributes:
        input_parameters: Fixed data for the problem (not optimized).
        terms: Objective terms defining the loss function.
        initialize: Callable that produces initial optimization variables
            from input_parameters.
        plot_configuration: Optional callable to visualize a configuration.
            Signature: `plot_configuration(optim_vars, input_parameters)`.
        svg_configuration: Optional callable to produce SVG element specs for
            animation. Signature:
            `svg_configuration(snapshots, input_parameters, size) -> list[dict]`.
        scene_configuration: Optional callable describing a configuration as
            a [Scene][vizopt.scene.Scene]. Signature:
            `scene_configuration(optim_vars, input_parameters) -> Scene`.
        var_scales: Optional per-variable scale factors. See
            :func:`build_objective` for details.
    """

    input_parameters: InputParams
    terms: list[ObjectiveTerm]
    initialize: Callable[[InputParams, int], OptimVars]
    plot_configuration: Callable[[OptimVars, InputParams], None] | None = None
    svg_configuration: Callable[[list, InputParams, int], list[dict]] | None = None
    scene_configuration: Callable[[OptimVars, InputParams], Scene] | None = None
    var_scales: dict | None = None
    result: "OptimizationResult | None" = field(default=None, init=False, repr=False)
    _step_functions: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        terms = self.terms
        input_parameters = self.input_parameters

        @jit
        def compute_all_terms(physical_vars: OptimVars) -> dict[str, Array]:
            return {
                term.name: term.compute(physical_vars, input_parameters)
                for term in terms
            }

        self._compute_all_terms = compute_all_terms

    def _step_function(self, b1: float, b2: float) -> tuple[StepFunction, Any]:
        """Jitted Adam step for this problem, compiled once per (b1, b2)."""
        if (b1, b2) not in self._step_functions:
            fun = build_objective(self.terms, self.input_parameters, self.var_scales)
            self._step_functions[(b1, b2)] = make_step_function(fun, b1, b2)
        return self._step_functions[(b1, b2)]

    def plot(self, **kwargs) -> None:
        """Plot the last optimization result using `plot_configuration`.

        Keyword arguments are forwarded to `plot_configuration`, allowing
        optional display flags (e.g. `show_arrows=True`).

        Raises:
            ValueError: If `plot_configuration` is not set or `optimize()`
                has not been called yet.
        """
        if self.plot_configuration is None:
            raise ValueError("plot_configuration is not set on this problem.")
        if self.result is None:
            raise ValueError("No result yet — call optimize() first.")
        self.plot_configuration(self.result.optim_vars, self.input_parameters, **kwargs)

    def scene(self, optim_vars: OptimVars | None = None) -> Scene:
        """Describe a configuration as a [Scene][vizopt.scene.Scene].

        Args:
            optim_vars: Physical-space variables to describe; defaults to the
                last optimization result.

        Returns:
            The scene produced by `scene_configuration`.

        Raises:
            ValueError: If `scene_configuration` is not set, or if
                `optim_vars` is omitted and `optimize()` has not been called.
        """
        if self.scene_configuration is None:
            raise ValueError("scene_configuration is not set on this problem.")
        if optim_vars is None:
            if self.result is None:
                raise ValueError("No result yet — call optimize() first.")
            optim_vars = self.result.optim_vars
        return self.scene_configuration(optim_vars, self.input_parameters)

    def session(
        self, optim_config: OptimConfig | None = None, seed: int | None = None
    ) -> OptimizationSession:
        """Start a steppable optimization run.

        Use this instead of `optimize` to drive the optimizer step by step
        and steer it in between (pin or move variables, change weights,
        reheat the learning rate), e.g. from an interactive frontend.
        `n_restarts`, `track_every` and early stopping are not used by a
        session; they belong to `optimize`.

        Args:
            optim_config: Optimizer settings. Uses
                [OptimConfig][vizopt.base.OptimConfig] defaults when `None`.
            seed: Seed for `initialize`; defaults to `optim_config.seed`.

        Returns:
            A fresh [OptimizationSession][vizopt.session.OptimizationSession]
            at iteration 0.
        """
        return OptimizationSession(self, optim_config or OptimConfig(), seed)

    def optimize(
        self,
        optim_config: OptimConfig | None = None,
        callback: Callback | None = None,
    ) -> "OptimizationResult":
        """Run gradient descent to minimize the objective.

        A batch run on top of [session][vizopt.base.OptimizationProblem.session]:
        steps a fresh session `n_iters` times, recording history and checking
        for early stopping along the way.

        When `optim_config.n_restarts > 1`, the optimization is run that
        many times with seeds `seed`, `seed + 1`, …. The result with the
        lowest final loss is returned.

        Args:
            optim_config: Optimizer settings (iterations, learning rate, seeds,
                restarts, track_every). Uses [OptimConfig][vizopt.base.OptimConfig]
                defaults when `None`.
            callback: Optional callback called after each iteration with
                (iteration, loss, optim_vars, grads).

        Returns:
            An [OptimizationResult][vizopt.base.OptimizationResult] with the
            optimized variables, per-term history, and final loss of the best run.
        """
        config = optim_config or OptimConfig()
        has_schedules = any(t.schedule is not None for t in self.terms)

        # When schedules are active and no user callback is provided, the loop
        # prints both scheduled and unscheduled totals. Otherwise fall back to
        # the standard print callback.
        user_callback = callback
        if user_callback is None and not has_schedules:
            user_callback = default_print_callback

        best_vars: OptimVars | None = None
        best_history: list[dict] = []
        best_loss = float("inf")

        for restart in range(config.n_restarts):
            session = self.session(config, seed=config.seed + restart)
            history: list[dict] = []
            last_unscheduled = 0.0
            for i_iter in range(config.n_iters):
                step_result = session.step()
                physical_vars = session.vars
                is_last = i_iter == config.n_iters - 1

                plateau_stop = False
                if i_iter % config.track_every == 0 or is_last:
                    history.append(session.record())
                    last_unscheduled = history[-1]["total_unscheduled"]
                    if config.early_stop_patience is not None:
                        window = max(
                            1, config.early_stop_patience // config.track_every
                        )
                        if len(history) > window:
                            prev_total = history[-window - 1]["total_unscheduled"]
                            plateau_stop = (
                                prev_total - last_unscheduled
                                < config.early_stop_tol * abs(prev_total)
                            )

                user_stop = None
                if user_callback is not None:
                    user_stop = user_callback(
                        i_iter, step_result.loss, physical_vars, step_result.grads
                    )
                elif i_iter % 100 == 0 or is_last:
                    print(
                        f"Iteration {i_iter}: loss = {float(step_result.loss):.6f}"
                        f"  unscheduled loss = {last_unscheduled:.6f}"
                    )
                if plateau_stop or user_stop:
                    break

            assert session.last_step is not None
            final_loss = float(session.last_step.loss)
            if best_vars is None or final_loss < best_loss:
                best_loss = final_loss
                best_vars = session.vars
                best_history = history

        assert best_vars is not None
        self.result = OptimizationResult(best_vars, best_history, best_loss)
        return self.result


def default_print_callback(i_iter: int, loss_value: Array, *_: Any) -> None:
    """Print the loss value after every nth optimization iteration"""
    if i_iter % 100 == 0:
        print(f"Iteration {i_iter}: loss = {loss_value}")


class VizOptimizer(ABC):
    """Base class for user-facing visualization optimizers.

    Subclasses implement :meth:`_build_problem` to turn stored hyperparameters
    into a configured :class:`OptimizationProblem`. The base class handles the
    optimize → plot lifecycle and stores fitted state in `problem_` and
    `result_` after :meth:`optimize` is called.
    """

    @abstractmethod
    def _build_problem(self) -> OptimizationProblem:
        """Build an :class:`OptimizationProblem` from stored hyperparameters."""
        ...

    def optimize(
        self,
        optim_config: OptimConfig | None = None,
        callback: Callback | None = None,
    ) -> OptimizationResult:
        """Build the problem and run gradient descent.

        Args:
            optim_config: Optimizer settings. Uses :class:`OptimConfig` defaults
                when `None`.
            callback: Optional callback `(iteration, loss, optim_vars, grads)`.

        Returns:
            An :class:`OptimizationResult` with optimized variables, per-term
            history, and final loss.
        """
        self.problem_: OptimizationProblem = self._build_problem()
        self.result_: OptimizationResult = self.problem_.optimize(
            optim_config, callback=callback
        )
        return self.result_

    def session(
        self, optim_config: OptimConfig | None = None, seed: int | None = None
    ) -> OptimizationSession:
        """Build the problem and start a steppable optimization run.

        See [OptimizationProblem.session][vizopt.base.OptimizationProblem.session].
        The built problem is stored in `problem_`; `result_` is not set, since
        a session has no natural end.

        Args:
            optim_config: Optimizer settings. Uses :class:`OptimConfig` defaults
                when `None`.
            seed: Seed for `initialize`; defaults to `optim_config.seed`.

        Returns:
            A fresh [OptimizationSession][vizopt.session.OptimizationSession].
        """
        self.problem_ = self._build_problem()
        return self.problem_.session(optim_config, seed)

    def plot(self, **kwargs) -> None:
        """Plot the last optimization result.

        Raises:
            ValueError: If :meth:`optimize` has not been called yet.
        """
        if not hasattr(self, "problem_"):
            raise ValueError("No result yet — call optimize() first.")
        self.problem_.plot(**kwargs)

    def animate(self, callback, **kwargs) -> Any:
        """Create a matplotlib animation of the optimization progress.

        Convenience wrapper around :func:`~vizopt.animation.animate` that
        renders each snapshot via this optimizer's `plot_configuration`. To
        save as a GIF, call `.save(path, writer="pillow", fps=...)` on the
        returned animation.

        Args:
            callback: A :class:`~vizopt.animation.SnapshotCallback` passed to
                :meth:`optimize`, or a raw list of `(iteration, optim_vars)`
                tuples.
            **kwargs: Forwarded to :func:`~vizopt.animation.animate`.

        Returns:
            A `matplotlib.animation.FuncAnimation`.

        Raises:
            ValueError: If :meth:`optimize` has not been called yet.
        """
        if not hasattr(self, "problem_"):
            raise ValueError("No result yet — call optimize() first.")
        from .animation import animate

        snapshots = callback.snapshots if hasattr(callback, "snapshots") else callback
        return animate(self.problem_, snapshots, **kwargs)

    def animate_svg(self, callback, **kwargs) -> str:
        """Create an animated SVG from a SnapshotCallback.

        Convenience wrapper around
        :func:`~vizopt.animation.snapshots_to_animated_svg` that uses this
        optimizer's built problem and defaults `history` to the result from
        the last :meth:`optimize` call.

        Args:
            callback: A :class:`~vizopt.animation.SnapshotCallback` passed to
                :meth:`optimize`, or a raw list of `(iteration, optim_vars)`
                tuples.
            **kwargs: Forwarded to
                :func:`~vizopt.animation.snapshots_to_animated_svg`.
                `history` defaults to `self.result_.history`.

        Returns:
            An SVG string suitable for saving or displaying with
            `IPython.display.SVG`.

        Raises:
            ValueError: If :meth:`optimize` has not been called yet.
        """
        if not hasattr(self, "problem_"):
            raise ValueError("No result yet — call optimize() first.")
        from .animation import snapshots_to_animated_svg

        snapshots = callback.snapshots if hasattr(callback, "snapshots") else callback
        kwargs.setdefault("history", self.result_.history)
        return snapshots_to_animated_svg(self.problem_, snapshots, **kwargs)
