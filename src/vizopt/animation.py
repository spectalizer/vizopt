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


def snapshots_to_animated_svg(
    problem: "OptimizationProblem",
    snapshots: list[tuple[int, Any]],
    fps: int = 10,
    size: int = 500,
    calc_mode: str = "linear",
) -> str:
    """Create an animated SVG from optimization snapshots.

    Uses ``problem.svg_configuration`` to obtain per-element SVG specs, then
    builds SMIL ``<animate>`` elements for attributes that vary across frames.

    Args:
        problem: The optimization problem; must have ``svg_configuration`` set.
        snapshots: List of ``(iteration, optim_vars)`` tuples as produced by
            :class:`SnapshotCallback`.
        fps: Frames per second.
        size: Width and height of the output SVG in pixels.
        calc_mode: ``"linear"`` for smooth interpolation or ``"discrete"``
            for instant jumps between frames.

    Returns:
        An SVG string. Save with ``Path("out.svg").write_text(svg)`` or
        display in Jupyter with ``IPython.display.SVG(data=svg)``.

    Raises:
        ValueError: If ``problem.svg_configuration`` is not set or
            ``snapshots`` is empty.
    """
    if problem.svg_configuration is None:
        raise ValueError("problem.svg_configuration must be set to use snapshots_to_animated_svg.")
    if not snapshots:
        raise ValueError("snapshots is empty.")

    elements = problem.svg_configuration(snapshots, problem.input_parameters, size)
    n_frames = len(snapshots)
    total_dur = n_frames / fps

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">',
        f'  <rect width="{size}" height="{size}" fill="white"/>',
    ]
    for el in elements:
        tag = el["tag"]
        static = {k: v for k, v in el.items() if k not in ("tag", "_text") and not isinstance(v, list)}
        animated = {k: v for k, v in el.items() if k not in ("tag", "_text") and isinstance(v, list)}
        inner_text = el.get("_text", "")
        attr_str = " ".join(f'{k}="{v}"' for k, v in static.items())
        lines.append(f"  <{tag} {attr_str}>")
        if inner_text:
            lines.append(f"    {inner_text}")
        for attr, values in animated.items():
            lines.append(f"    {smil_animate(attr, values, n_frames, total_dur, calc_mode)}")
        lines.append(f"  </{tag}>")
    lines.append("</svg>")
    return "\n".join(lines)


def smil_animate(
    attr_name: str,
    per_frame_values: list[str],
    n_frames: int,
    total_dur: float,
    calc_mode: str = "linear",
) -> str:
    """Build a SMIL ``<animate>`` element string for a numeric SVG attribute.

    Args:
        attr_name: SVG attribute to animate (e.g. ``"cx"``, ``"opacity"``).
        per_frame_values: Stringified values, one per frame.
        n_frames: Total number of frames, used to space ``keyTimes`` evenly.
        total_dur: Animation duration in seconds.
        calc_mode: ``"linear"`` or ``"discrete"``. Linear appends the first
            value at ``keyTime`` ``1.0`` to close the loop smoothly.

    Returns:
        An SVG ``<animate>`` element string.
    """
    if calc_mode == "linear":
        key_times = ";".join(f"{fi / n_frames:.6f}" for fi in range(n_frames)) + ";1.000000"
        values = ";".join(per_frame_values + per_frame_values[:1])
    else:
        key_times = ";".join(f"{fi / n_frames:.6f}" for fi in range(n_frames))
        values = ";".join(per_frame_values)
    return (
        f'<animate attributeName="{attr_name}" calcMode="{calc_mode}"'
        f' values="{values}" keyTimes="{key_times}"'
        f' dur="{total_dur:.3f}s" repeatCount="indefinite"/>'
    )
