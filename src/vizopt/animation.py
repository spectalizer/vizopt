"""Animation utilities for visualizing optimization progress."""

from typing import Any

import jax
import numpy as np
from jax import Array

from .base import OptimizationProblem, default_print_callback


class SnapshotCallback:
    """Callback that saves a numpy copy of ``optim_vars`` at regular intervals.

    Pass an instance as the ``callback`` argument to
    :meth:`OptimizationProblem.optimize`. Snapshots accumulate in
    :attr:`snapshots` and can be passed to :func:`animate`.

    Args:
        every: Save a snapshot every this many iterations.

    Attributes:
        snapshots: List of ``(iteration, optim_vars)`` tuples, one per
            recorded step.

    Example::

        cb = SnapshotCallback(every=100)
        optim_vars_opt, history = problem.optimize(n_iters=2000, callback=cb)
        anim = animate(problem, cb.snapshots)
    """

    def __init__(self, every: int = 10) -> None:
        self.every = every
        self.snapshots: list[tuple[int, Any]] = []

    def __call__(
        self, i_iter: int, loss_value: Array, optim_vars: Any, grads: Any
    ) -> None:
        default_print_callback(i_iter, loss_value)
        if i_iter % self.every == 0:
            self.snapshots.append((i_iter, jax.tree.map(np.array, optim_vars)))


def animate(
    problem: OptimizationProblem,
    snapshots: list[tuple[int, Any]],
    interval: int = 200,
) -> Any:
    """Create an animation of the optimization process.

    Renders each ``optim_vars`` snapshot via ``problem.plot_configuration``
    and assembles the frames into a ``FuncAnimation``.

    Args:
        problem: The optimization problem; must have ``plot_configuration`` set.
        snapshots: List of ``(iteration, optim_vars)`` tuples as produced by
            :class:`SnapshotCallback`.
        interval: Delay between frames in milliseconds.

    Returns:
        A ``matplotlib.animation.FuncAnimation``. In a Jupyter notebook,
        display with ``IPython.display.HTML(anim.to_jshtml())``.

    Raises:
        ValueError: If ``problem.plot_configuration`` is not set or
            ``snapshots`` is empty.
    """
    from matplotlib import animation
    from matplotlib import pyplot as plt

    if problem.plot_configuration is None:
        raise ValueError("problem.plot_configuration must be set to animate.")
    if not snapshots:
        raise ValueError("snapshots is empty.")

    images = []
    for _, optim_vars in snapshots:
        problem.plot_configuration(optim_vars, problem.input_parameters)
        fig = plt.gcf()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).copy()  # type: ignore
        images.append(buf.reshape(h, w, 4))
        plt.close(fig)

    fig_anim, ax_anim = plt.subplots()
    ax_anim.axis("off")
    fig_anim.tight_layout(pad=0)
    im = ax_anim.imshow(images[0])

    def update(frame: int) -> list:
        im.set_data(images[frame])
        return [im]

    return animation.FuncAnimation(
        fig_anim, update, frames=len(images), interval=interval, blit=True
    )
