"""Animation utilities for visualizing optimization progress."""

import math
from typing import Any

import jax
import numpy as np
from jax import Array
from matplotlib.ticker import AutoLocator

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
    history: list[dict] | None = None,
    loss_curve_height: int = 120,
    log_scale: bool = False,
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
        history: Optional list of history dicts as returned by
            :meth:`OptimizationProblem.optimize` (each dict has an
            ``"iteration"`` key and a ``"total"`` key with the aggregate loss).
            When provided, a loss curve is rendered below the animation with an
            animated marker tracking the current frame.
        loss_curve_height: Height in pixels of the loss curve panel, used only
            when ``history`` is provided.
        log_scale: If ``True``, the loss axis uses a log10 scale.

    Returns:
        An SVG string. Save with ``Path("out.svg").write_text(svg)`` or
        display in Jupyter with ``IPython.display.SVG(data=svg)``.

    Raises:
        ValueError: If ``problem.svg_configuration`` is not set or
            ``snapshots`` is empty.
    """
    if problem.svg_configuration is None:
        raise ValueError(
            "problem.svg_configuration must be set to use snapshots_to_animated_svg."
        )
    if not snapshots:
        raise ValueError("snapshots is empty.")

    elements = problem.svg_configuration(snapshots, problem.input_parameters, size)
    n_frames = len(snapshots)
    total_dur = n_frames / fps

    total_height = size + (loss_curve_height if history else 0)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{total_height}">',
        f'  <rect width="{size}" height="{total_height}" fill="white"/>',
    ]
    for el in elements:
        tag = el["tag"]
        static = {
            k: v
            for k, v in el.items()
            if k not in ("tag", "_text") and not isinstance(v, list)
        }
        animated = {
            k: v
            for k, v in el.items()
            if k not in ("tag", "_text") and isinstance(v, list)
        }
        inner_text = el.get("_text", "")
        attr_str = " ".join(f'{k}="{v}"' for k, v in static.items())
        lines.append(f"  <{tag} {attr_str}>")
        if inner_text:
            lines.append(f"    {inner_text}")
        for attr, values in animated.items():
            lines.append(
                f"    {smil_animate(attr, values, n_frames, total_dur, calc_mode)}"
            )
        lines.append(f"  </{tag}>")

    if history:
        lines += _loss_curve_svg_lines(
            history,
            snapshots,
            size,
            loss_curve_height,
            n_frames,
            total_dur,
            calc_mode,
            log_scale,
        )

    lines.append("</svg>")
    return "\n".join(lines)


def _loss_curve_svg_lines(
    history: list[dict],
    snapshots: list[tuple[int, Any]],
    size: int,
    panel_height: int,
    n_frames: int,
    total_dur: float,
    calc_mode: str,
    log_scale: bool = False,
) -> list[str]:
    """Build SVG lines for a loss curve panel placed below the main animation.

    Args:
        history: List of history dicts with ``"iteration"`` and ``"total"`` keys.
        snapshots: List of ``(iteration, optim_vars)`` tuples.
        size: Width of the SVG (same as the animation width).
        panel_height: Height of the loss curve panel in pixels.
        n_frames: Number of animation frames.
        total_dur: Total animation duration in seconds.
        calc_mode: SMIL ``calcMode`` (``"linear"`` or ``"discrete"``).
        log_scale: If ``True``, the loss axis uses a log10 scale.

    Returns:
        A list of SVG element strings to append before the closing ``</svg>``.
    """
    pad_left = 40
    pad_right = 12
    pad_top = 12
    pad_bottom = 24
    panel_y = size  # y-offset of the panel inside the SVG

    iters = [d["iteration"] for d in history]
    losses = [float(d["total"]) for d in history]

    min_iter, max_iter = min(iters), max(iters)
    if log_scale:
        _log_min = math.floor(math.log10(min(losses)))
        _log_max = math.ceil(math.log10(max(losses)))
        min_loss, max_loss = 10.0**_log_min, 10.0**_log_max
        _log_range = _log_max - _log_min or 1.0
    else:
        _ticks = AutoLocator().tick_values(min(losses), max(losses))
        min_loss, max_loss = float(_ticks[0]), float(_ticks[-1])
    loss_range = max_loss - min_loss or 1.0

    plot_w = size - pad_left - pad_right
    plot_h = panel_height - pad_top - pad_bottom

    def to_x(it: float) -> float:
        return pad_left + (it - min_iter) / (max_iter - min_iter or 1) * plot_w

    def to_y(loss: float) -> float:
        if log_scale:
            t = (math.log10(loss) - _log_min) / _log_range
        else:
            t = (loss - min_loss) / loss_range
        return panel_y + pad_top + (1.0 - t) * plot_h

    # Polyline points for full loss curve
    curve_pts = [(to_x(it), to_y(l)) for it, l in zip(iters, losses)]
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in curve_pts)

    # Cumulative arc-length along the polyline (for stroke-dashoffset animation)
    cumulative = [0.0]
    for i in range(1, len(curve_pts)):
        dx = curve_pts[i][0] - curve_pts[i - 1][0]
        dy = curve_pts[i][1] - curve_pts[i - 1][1]
        cumulative.append(cumulative[-1] + math.sqrt(dx * dx + dy * dy))
    total_len = cumulative[-1] or 1.0

    # For each snapshot frame, find cumulative length at that snapshot's iteration
    def _cum_len_at(snap_iter: int) -> float:
        idx = 0
        for j, it in enumerate(iters):
            if it <= snap_iter:
                idx = j
        return cumulative[idx]

    # stroke-dashoffset per frame: total_len - d means only the first d px are drawn
    dashoffsets = [
        f"{total_len - _cum_len_at(sit):.1f}" for sit, _ in snapshots
    ]

    # Animated marker x positions — one per snapshot frame
    snapshot_iters = [it for it, _ in snapshots]
    marker_xs = [f"{to_x(it):.1f}" for it in snapshot_iters]

    # Axis tick labels (y-axis: min and max)
    y_label_max = f"{max_loss:.3g}"
    y_label_min = f"{min_loss:.3g}"

    lines = [
        "  <!-- loss curve panel -->",
        f'  <rect x="0" y="{panel_y}" width="{size}" height="{panel_height}" fill="#f8f8f8"/>',
        # Axes
        f'  <line x1="{pad_left}" y1="{panel_y + pad_top}" x2="{pad_left}" y2="{panel_y + pad_top + plot_h}" stroke="#888" stroke-width="1"/>',
        f'  <line x1="{pad_left}" y1="{panel_y + pad_top + plot_h}" x2="{pad_left + plot_w}" y2="{panel_y + pad_top + plot_h}" stroke="#888" stroke-width="1"/>',
        # Y-axis tick labels
        f'  <text x="{pad_left - 4}" y="{panel_y + pad_top + 4}" text-anchor="end" font-size="9" fill="#555">{y_label_max}</text>',
        f'  <text x="{pad_left - 4}" y="{panel_y + pad_top + plot_h}" text-anchor="end" font-size="9" fill="#555">{y_label_min}</text>',
        # X-axis label
        f'  <text x="{pad_left + plot_w // 2}" y="{panel_y + panel_height - 4}" text-anchor="middle" font-size="9" fill="#555">iteration</text>',
        # Full curve — light/thin, represents the "future" portion
        f'  <polyline points="{points}" fill="none" stroke="#4477cc" stroke-width="1" stroke-linejoin="round" opacity="0.3"/>',
        # Animated "past" overlay — thick, reveals up to the current frame via stroke-dashoffset
        f'  <polyline points="{points}" fill="none" stroke="#4477cc" stroke-width="2.5" stroke-linejoin="round"',
        f'    stroke-dasharray="{total_len:.1f}" stroke-dashoffset="{total_len:.1f}">',
        f'    {smil_animate("stroke-dashoffset", dashoffsets, n_frames, total_dur, calc_mode)}',
        "  </polyline>",
        # Animated vertical marker line
        f'  <line y1="{panel_y + pad_top}" y2="{panel_y + pad_top + plot_h}" stroke="gray" stroke-width="1.5" stroke-dasharray="3,2">',
        f'    {smil_animate("x1", marker_xs, n_frames, total_dur, calc_mode)}',
        f'    {smil_animate("x2", marker_xs, n_frames, total_dur, calc_mode)}',
        "  </line>",
    ]

    # Per-frame iteration + loss label (SMIL cannot animate text content directly,
    # so one <text> per frame uses a discrete display animation: "inline" only
    # during its own frame slot, "none" for all others).
    label_x = pad_left + plot_w
    label_y = panel_y + pad_top + 10
    for fi, (snap_iter, _) in enumerate(snapshots):
        closest = min(history, key=lambda d: abs(d["iteration"] - snap_iter))
        snap_loss = float(closest["total"])
        label = f"iter {snap_iter}  loss {snap_loss:.3g}"
        display_values = ["none"] * n_frames
        display_values[fi] = "inline"
        lines += [
            f'  <text x="{label_x}" y="{label_y}" text-anchor="end" font-size="9" fill="#333" display="none">',
            f"    {label}",
            f'    {smil_animate("display", display_values, n_frames, total_dur, "discrete")}',
            "  </text>",
        ]

    return lines


def chronophotograph(
    problem: OptimizationProblem,
    snapshots: list[tuple[int, Any]],
    n_frames: int = 8,
    alpha_start: float = 0.15,
    alpha_end: float = 1.0,
) -> Any:
    """Create a static chronophotography composite of the optimization process.

    Renders a selection of evenly-spaced snapshots and overlays them into a
    single static figure.  Earlier frames are faint ghosts; the latest frame
    is fully opaque, so the eye reads the trajectory at a glance.

    Args:
        problem: The optimization problem; must have ``plot_configuration`` set.
        snapshots: List of ``(iteration, optim_vars)`` tuples as produced by
            :class:`SnapshotCallback`.
        n_frames: Number of evenly-spaced frames to overlay.  Clamped to
            ``len(snapshots)``.
        alpha_start: Blend weight of the earliest selected frame (0–1).
        alpha_end: Blend weight of the latest selected frame (0–1).

    Returns:
        A matplotlib ``Figure`` containing the composite image.

    Raises:
        ValueError: If ``problem.plot_configuration`` is not set or
            ``snapshots`` is empty.
    """
    from matplotlib import pyplot as plt

    if problem.plot_configuration is None:
        raise ValueError("problem.plot_configuration must be set to chronophotograph.")
    if not snapshots:
        raise ValueError("snapshots is empty.")

    # Pick evenly-spaced indices across the snapshot list
    n = min(n_frames, len(snapshots))
    indices = np.linspace(0, len(snapshots) - 1, n, dtype=int)
    selected = [snapshots[i] for i in indices]
    alphas = np.linspace(alpha_start, alpha_end, n)

    # Render each frame to an RGBA float array
    images: list[np.ndarray] = []
    for _, optim_vars in selected:
        problem.plot_configuration(optim_vars, problem.input_parameters)
        fig = plt.gcf()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).copy()  # type: ignore
        images.append(buf.reshape(h, w, 4).astype(np.float32) / 255.0)
        plt.close(fig)

    # Alpha-composite onto a white canvas: earlier frames leave faint traces,
    # the final frame dominates.  Because all rendered backgrounds are white,
    # white-on-white stays white and only the coloured marks accumulate.
    canvas = np.ones_like(images[0])
    for img, alpha in zip(images, alphas):
        canvas = (1.0 - alpha) * canvas + alpha * img

    fig_out, ax_out = plt.subplots()
    ax_out.imshow(canvas)
    ax_out.axis("off")
    fig_out.tight_layout(pad=0)
    return fig_out


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
        key_times = (
            ";".join(f"{fi / n_frames:.6f}" for fi in range(n_frames)) + ";1.000000"
        )
        values = ";".join(per_frame_values + per_frame_values[:1])
    else:
        key_times = ";".join(f"{fi / n_frames:.6f}" for fi in range(n_frames))
        values = ";".join(per_frame_values)
    return (
        f'<animate attributeName="{attr_name}" calcMode="{calc_mode}"'
        f' values="{values}" keyTimes="{key_times}"'
        f' dur="{total_dur:.3f}s" repeatCount="indefinite"/>'
    )
