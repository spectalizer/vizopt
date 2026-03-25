"""Optimizing color palettes."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from vizopt.base import ObjectiveTerm, OptimizationProblemTemplate, OptimConfig


def oklab_to_rgb(Lab):
    """Convert OKLAB values to sRGB.

    Args:
        Lab: Array of shape (n, 3) with OKLAB values (L in [0, 1]).

    Returns:
        Array of shape (n, 3) with sRGB values clipped to [0, 1].
    """
    M2_inv = np.array(
        [
            [1.0000000000, 0.3963377774, 0.2158037573],
            [1.0000000000, -0.1055613458, -0.0638541728],
            [1.0000000000, -0.0894841775, -1.2914855480],
        ]
    )
    lms_ = Lab @ M2_inv.T
    lms = lms_**3
    M1_inv = np.array(
        [
            [4.0767416621, -3.3077115913, 0.2309699292],
            [-1.2684380046, 2.6097574011, -0.3413193965],
            [-0.0041960863, -0.7034186147, 1.7076147010],
        ]
    )
    rgb_lin = lms @ M1_inv.T
    rgb = np.where(
        rgb_lin <= 0.0031308, 12.92 * rgb_lin, 1.055 * rgb_lin ** (1 / 2.4) - 0.055
    )
    return np.clip(rgb, 0, 1)


def rgb_to_oklab(rgb):
    """Convert sRGB values to OKLAB, differentiable via JAX.

    Args:
        rgb: Array of shape (n, 3) with sRGB values in [0, 1].

    Returns:
        Array of shape (n, 3) with OKLAB values (L in [0, 1], a and b in ~[-0.4, 0.4]).
    """
    rgb_lin = jnp.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M1 = jnp.array(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ]
    )
    lms = rgb_lin @ M1.T
    lms_ = jnp.cbrt(lms)
    M2 = jnp.array(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ]
    )
    return lms_ @ M2.T


def rgb_to_oklch(rgb):
    """Convert sRGB values to OKLCH, differentiable via JAX.

    OKLCH is the polar form of OKLAB: L (lightness), C (chroma), H (hue in radians).

    Args:
        rgb: Array of shape (n, 3) with sRGB values in [0, 1].

    Returns:
        Array of shape (n, 3) with OKLCH values: L in [0, 1], C >= 0, H in [-pi, pi].
    """
    lab = rgb_to_oklab(rgb)
    L = lab[:, 0]
    C = jnp.sqrt(lab[:, 1] ** 2 + lab[:, 2] ** 2)
    H = jnp.arctan2(lab[:, 2], lab[:, 1])
    return jnp.stack([L, C, H], axis=1)


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
    lab = rgb_to_oklab(_build_rgb(optim_vars, input_parameters))
    idx_i = input_parameters["idx_i"]
    idx_j = input_parameters["idx_j"]
    targets = input_parameters["targets"]
    color_dists = jnp.sqrt(((lab[idx_i] - lab[idx_j]) ** 2).sum(axis=-1) + 1e-8)
    return jnp.mean((color_dists - targets) ** 2)


def _coverage(optim_vars, input_parameters):
    lab = rgb_to_oklab(_build_rgb(optim_vars, input_parameters))
    idx_i = input_parameters["idx_i"]
    idx_j = input_parameters["idx_j"]
    color_dists = jnp.sqrt(((lab[idx_i] - lab[idx_j]) ** 2).sum(axis=-1) + 1e-8)
    temperature = input_parameters.get("coverage_temperature", 0.05)
    return temperature * jax.scipy.special.logsumexp(-color_dists / temperature)


def _luminosity(optim_vars, input_parameters):
    target_L = input_parameters.get("target_L")
    if target_L is None:
        return jnp.zeros(())
    lab = rgb_to_oklab(_build_rgb(optim_vars, input_parameters))
    return jnp.mean((lab[:, 0] - target_L) ** 2)


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
            per_color_fills[i].append(
                f"#{int(rv*255):02x}{int(gv*255):02x}{int(bv*255):02x}"
            )

    elements = []
    for i in range(n):
        elements.append(
            {
                "tag": "circle",
                "cx": f"{cx_list[i]:.1f}",
                "cy": f"{cy:.1f}",
                "r": f"{r:.1f}",
                "stroke": "#555555",
                "stroke-width": "1",
                "fill": per_color_fills[i],
            }
        )
        elements.append(
            {
                "tag": "text",
                "x": f"{cx_list[i]:.1f}",
                "y": f"{text_y:.1f}",
                "text-anchor": "middle",
                "font-size": f"{font_size}",
                "font-family": "sans-serif",
                "fill": "#333333",
                "_text": str(labels[i]),
            }
        )
    return elements



def build_color_input_parameters(
    distances,
    fixed_colors=None,
    *,
    target_max_delta_e=0.3,
    target_L=None,
    coverage_temperature=0.05,
    seed=None,
):
    """Build the ``input_parameters`` dict for color palette optimization.

    Args:
        distances: Symmetric pairwise distance matrix of shape (n, n). If a
            DataFrame, its index is used as labels.
        fixed_colors: Map from label (DataFrame index value) or integer position
            to an sRGB tuple/array in [0, 1]. Those colors are held fixed.
        target_max_delta_e: The largest pairwise distance maps to this OKLAB ΔE
            value. OKLAB distances are in [0, ~0.4] for typical sRGB colors;
            the default of 0.3 spans most of the gamut.
        target_L: Target OKLAB/OKLCH lightness in [0, 1], or ``None`` to
            disable the luminosity term. ~0.75 for light mode, ~0.85 for dark mode.
        coverage_temperature: Temperature for the soft-min coverage term. Smaller
            values focus more sharply on the closest pair (harder min); larger
            values average more broadly across all pairs. Default 0.05.
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
        "target_L": target_L,
        "coverage_temperature": coverage_temperature,
    }


def build_color_problem(
    input_parameters,
    *,
    stress_weight=1.0,
    coverage_weight=0.5,
    luminosity_weight=1.0,
):
    """Instantiate an :class:`~vizopt.base.OptimizationProblem` for color palette optimization.

    Useful when you need the problem object directly, e.g. to pass to
    :func:`~vizopt.animation.snapshots_to_animated_svg`.

    Args:
        input_parameters: Dict returned by :func:`build_color_input_parameters`.
        stress_weight: Multiplier for the stress term (default 1.0).
        coverage_weight: Multiplier for the coverage term (default 0.5).
        luminosity_weight: Multiplier for the luminosity term (default 1.0).

    Returns:
        :class:`~vizopt.base.OptimizationProblem` ready to call ``.optimize()`` on.
    """
    template = OptimizationProblemTemplate(
        terms=[
            ObjectiveTerm(name="stress", compute=_stress, multiplier=stress_weight),
            ObjectiveTerm(name="coverage", compute=_coverage, multiplier=coverage_weight),
            ObjectiveTerm(name="luminosity", compute=_luminosity, multiplier=luminosity_weight),
        ],
        initialize=lambda params, _seed: {"logit_rgb": params["logit_init"]},
        plot_configuration=plot_colored_words,
        svg_configuration=_color_svg_configuration,
    )
    return template.instantiate(input_parameters)


def optimize_colors(
    distances,
    fixed_colors=None,
    *,
    target_max_delta_e=0.3,
    target_L=None,
    init_seed=None,
    optim_config: OptimConfig | None = None,
    callback=None,
    stress_weight=1.0,
    coverage_weight=0.5,
    luminosity_weight=1.0,
    coverage_temperature=0.05,
):
    """Optimize a palette so pairwise OKLAB ΔE distances match ``distances``.

    Args:
        distances: Symmetric pairwise distance matrix of shape (n, n). If a
            DataFrame, its index is used as labels for ``fixed_colors`` keys.
        fixed_colors: Map from label (DataFrame index value) or integer position
            to an sRGB tuple/array in [0, 1]. Those colors are held fixed during
            optimization.
        target_max_delta_e: The largest pairwise distance maps to this OKLAB ΔE
            value. OKLAB distances are in [0, ~0.4] for typical sRGB colors.
        target_L: Target OKLAB/OKLCH lightness for all colors, in [0, 1].
            ``None`` disables the luminosity term. Typical values: ~0.75 for
            light-mode palettes, ~0.85 for dark-mode palettes.
        init_seed: Integer random seed for initialization. When ``None``
            (default), uses an MDS warm-start. Pass an integer to use random
            initialization instead.
        optim_config: Optimizer settings (iterations, learning rate, seeds,
            restarts). Uses :class:`OptimConfig` defaults when ``None``.
        callback: Called as ``callback(i, loss, params, grads)`` after each step.
        stress_weight: Multiplier for the stress term (default 1.0).
        coverage_weight: Multiplier for the coverage term (default 0.5).
        luminosity_weight: Multiplier for the luminosity term (default 1.0).
        coverage_temperature: Temperature for the soft-min coverage term (default 0.05).

    Returns:
        Tuple of (colors, history) where colors is an sRGB array of shape (n, 3)
        in [0, 1] and history is a list of dicts recorded every 10 iterations
        with keys ``"iteration"``, ``"total"``, ``"stress"``, ``"coverage"``,
        and ``"luminosity"``.
    """
    input_parameters = build_color_input_parameters(
        distances,
        fixed_colors,
        target_max_delta_e=target_max_delta_e,
        target_L=target_L,
        coverage_temperature=coverage_temperature,
        seed=init_seed,
    )
    problem = build_color_problem(
        input_parameters,
        stress_weight=stress_weight,
        coverage_weight=coverage_weight,
        luminosity_weight=luminosity_weight,
    )
    optim_vars, history = problem.optimize(optim_config, callback=callback)
    return np.array(_build_rgb(optim_vars, input_parameters)), history
