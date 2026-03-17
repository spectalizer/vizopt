"""Optimizing color palettes."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from vizopt.base import ObjectiveTerm, OptimizationProblemTemplate


def lab_to_rgb(Lab):
    """Convert CIE L*a*b* values to sRGB.

    Args:
        Lab: Array of shape (n, 3) with CIE L*a*b* values.

    Returns:
        Array of shape (n, 3) with sRGB values clipped to [0, 1].
    """
    L, a, b = Lab[:, 0], Lab[:, 1], Lab[:, 2]
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    def f_inv(t):
        return np.where(t > 6 / 29, t**3, 3 * (6 / 29) ** 2 * (t - 4 / 29))

    XYZ = np.stack([0.95047 * f_inv(fx), f_inv(fy), 1.08883 * f_inv(fz)], axis=1)

    M = np.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ]
    )
    rgb_lin = XYZ @ M.T

    rgb = np.where(
        rgb_lin <= 0.0031308, 12.92 * rgb_lin, 1.055 * rgb_lin ** (1 / 2.4) - 0.055
    )
    return np.clip(rgb, 0, 1)


def rgb_to_lab(rgb):
    """Convert sRGB values to CIE L*a*b* (D65 illuminant), differentiable via JAX.

    Args:
        rgb: Array of shape (n, 3) with sRGB values in [0, 1].

    Returns:
        Array of shape (n, 3) with CIE L*a*b* values.
    """
    rgb_lin = jnp.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = jnp.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = (rgb_lin @ M.T) / jnp.array([0.95047, 1.00000, 1.08883])
    delta = 6 / 29
    f = jnp.where(xyz > delta**3, xyz ** (1 / 3), xyz / (3 * delta**2) + 4 / 29)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return jnp.stack([L, a, b], axis=1)


def _build_rgb(optim_vars, input_parameters):
    """Reconstruct the full (n, 3) sRGB array from free logit params.

    Args:
        optim_vars: Dict with key ``"logit_rgb"`` of shape (n_free, 3).
        input_parameters: Problem input dict with keys ``"n"``, ``"free_idx"``,
            ``"fixed_idx"``, and ``"fixed_rgb"``.

    Returns:
        sRGB array of shape (n, 3) with values in [0, 1].
    """
    rgb = jax.nn.sigmoid(optim_vars["logit_rgb"])
    full = jnp.zeros((input_parameters["n"], 3))
    full = full.at[input_parameters["free_idx"]].set(rgb)
    if input_parameters["fixed_idx"].shape[0] > 0:
        full = full.at[input_parameters["fixed_idx"]].set(input_parameters["fixed_rgb"])
    return full


def _stress(optim_vars, input_parameters):
    lab = rgb_to_lab(_build_rgb(optim_vars, input_parameters))
    idx_i = input_parameters["idx_i"]
    idx_j = input_parameters["idx_j"]
    targets = input_parameters["targets"]
    color_dists = jnp.sqrt(((lab[idx_i] - lab[idx_j]) ** 2).sum(axis=-1) + 1e-8)
    return jnp.mean((color_dists - targets) ** 2)


def _coverage(optim_vars, input_parameters):
    lab = rgb_to_lab(_build_rgb(optim_vars, input_parameters))
    return -(lab[:, 1].var() + lab[:, 2].var())


def plot_colored_words(optim_vars, input_parameters):
    colors = np.array(_build_rgb(optim_vars, input_parameters))
    labels = input_parameters["labels"]
    _, ax = plt.subplots(figsize=(7, 2))
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.scatter(
            i, 0.15, color=color, s=500, zorder=3, edgecolors="0.3", linewidths=0.5
        )
        ax.text(i, -0.05, label, ha="center", va="top", fontsize=12)
    ax.set_xlim(-0.7, len(labels) - 0.3)
    ax.set_ylim(-0.3, 0.45)
    ax.axis("off")
    plt.tight_layout()


def _color_svg_configuration(snapshots, input_parameters, size):
    """Build SVG element specs for the animated color palette.

    Args:
        snapshots: List of ``(iteration, optim_vars)`` tuples.
        input_parameters: Problem input dict with ``"labels"`` and color data.
        size: SVG canvas size in pixels.

    Returns:
        List of element dicts for
        :func:`~vizopt.animation.snapshots_to_animated_svg`.
    """
    labels = input_parameters["labels"]
    n = len(labels)

    padding = size * 0.1
    slot = (size - 2 * padding) / n
    cx_list = [padding + slot * (i + 0.5) for i in range(n)]
    r = min(slot * 0.35, size * 0.1)
    cy = size * 0.4
    text_y = cy + r + size * 0.07
    font_size = max(10, int(size * 0.04))

    per_color_fills = [[] for _ in range(n)]
    for _, optim_vars in snapshots:
        rgb = np.array(_build_rgb(optim_vars, input_parameters))
        for i in range(n):
            rv, gv, bv = rgb[i]
            per_color_fills[i].append(f"#{int(rv*255):02x}{int(gv*255):02x}{int(bv*255):02x}")

    elements = []
    for i in range(n):
        elements.append({
            "tag": "circle",
            "cx": f"{cx_list[i]:.1f}",
            "cy": f"{cy:.1f}",
            "r": f"{r:.1f}",
            "stroke": "#555555",
            "stroke-width": "1",
            "fill": per_color_fills[i],
        })
        elements.append({
            "tag": "text",
            "x": f"{cx_list[i]:.1f}",
            "y": f"{text_y:.1f}",
            "text-anchor": "middle",
            "font-size": f"{font_size}",
            "font-family": "sans-serif",
            "fill": "#333333",
            "_text": str(labels[i]),
        })
    return elements


color_palette_template = OptimizationProblemTemplate(
    terms=[
        ObjectiveTerm(name="stress", compute=_stress),
        ObjectiveTerm(name="coverage", compute=_coverage, multiplier=0.5),
    ],
    initialize=lambda params: {"logit_rgb": params["logit_init"]},
    plot_configuration=plot_colored_words,
    svg_configuration=_color_svg_configuration,
)


def build_color_input_parameters(
    distances,
    fixed_colors=None,
    *,
    target_max_delta_e=50.0,
    seed=None,
):
    """Build the ``input_parameters`` dict for :obj:`color_palette_template`.

    Args:
        distances: Symmetric pairwise distance matrix of shape (n, n). If a
            DataFrame, its index is used as labels.
        fixed_colors: Map from label (DataFrame index value) or integer position
            to an sRGB tuple/array in [0, 1]. Those colors are held fixed.
        target_max_delta_e: The largest pairwise distance maps to this ΔE value.
        seed: Integer random seed. When ``None`` (default), uses an MDS warm-start.

    Returns:
        Dict of input parameters suitable for
        ``color_palette_template.instantiate()``.
    """
    if isinstance(distances, pd.DataFrame):
        labels = list(distances.index)
        D = np.array(distances, dtype=float)
    else:
        D = np.array(distances, dtype=float)
        labels = list(range(len(D)))

    n = len(D)
    fixed_colors = fixed_colors or {}

    # Resolve fixed_colors keys to integer indices
    label_to_idx = {label: i for i, label in enumerate(labels)}
    fixed_idx_map = {}
    for key, rgb in fixed_colors.items():
        idx = label_to_idx[key] if key in label_to_idx else int(key)
        fixed_idx_map[idx] = np.asarray(rgb, dtype=float)

    free_idx = jnp.array([i for i in range(n) if i not in fixed_idx_map])
    fixed_idx = (
        jnp.array(sorted(fixed_idx_map)) if fixed_idx_map else jnp.array([], dtype=int)
    )
    fixed_rgb = (
        jnp.array([fixed_idx_map[i] for i in sorted(fixed_idx_map)])
        if fixed_idx_map
        else jnp.zeros((0, 3))
    )

    # Pre-compute pair indices and scaled target distances
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    idx_i = jnp.array([i for i, _ in pairs])
    idx_j = jnp.array([j for _, j in pairs])
    pair_dists = jnp.array([D[i, j] for i, j in pairs], dtype=float)
    targets = pair_dists / pair_dists.max() * target_max_delta_e

    rng = jax.random.PRNGKey(0 if seed is None else seed)
    logit_init = jax.random.normal(rng, shape=(n, 3))

    return {
        "n": n,
        "labels": labels,
        "free_idx": free_idx,
        "fixed_idx": fixed_idx,
        "fixed_rgb": fixed_rgb,
        "idx_i": idx_i,
        "idx_j": idx_j,
        "targets": targets,
        "logit_init": jnp.array(logit_init[np.array(free_idx)]),
    }


def optimize_colors(
    distances,
    fixed_colors=None,
    *,
    target_max_delta_e=50.0,
    learning_rate=0.05,
    n_iters=1000,
    callback=None,
    seed=None,
):
    """Optimize a palette so pairwise CIELAB ΔE distances match ``distances``.

    Args:
        distances: Symmetric pairwise distance matrix of shape (n, n). If a
            DataFrame, its index is used as labels for ``fixed_colors`` keys.
        fixed_colors: Map from label (DataFrame index value) or integer position
            to an sRGB tuple/array in [0, 1]. Those colors are held fixed during
            optimization.
        target_max_delta_e: The largest pairwise distance maps to this ΔE value.
        learning_rate: Adam learning rate.
        n_iters: Number of gradient-descent iterations.
        callback: Called as ``callback(i, loss, params, grads)`` after each step.
        seed: Integer random seed. When ``None`` (default), uses an MDS warm-start.
            Pass an integer to use random initialization instead, enabling multiple
            restarts with different starting points.

    Returns:
        Tuple of (colors, history) where colors is an sRGB array of shape (n, 3)
        in [0, 1] and history is a list of dicts recorded every 10 iterations
        with keys ``"iteration"``, ``"total"``, ``"stress"``, and ``"coverage"``.
    """
    input_parameters = build_color_input_parameters(
        distances, fixed_colors, target_max_delta_e=target_max_delta_e, seed=seed
    )
    problem = color_palette_template.instantiate(input_parameters)
    optim_vars, history = problem.optimize(
        n_iters=n_iters,
        learning_rate=learning_rate,
        callback=callback,
    )
    return np.array(_build_rgb(optim_vars, input_parameters)), history
