import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from vizopt.jaxopt import optimize_gradient_descent


def classical_mds(D, k=3):
    """Embed a distance matrix into Euclidean coordinates via classical MDS.

    Args:
        D: Square symmetric distance matrix of shape (n, n).
        k: Number of embedding dimensions.

    Returns:
        Coordinate array of shape (n, k).
    """
    n = len(D)
    D2 = np.array(D) ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D2 @ H

    eigenvalues, eigenvectors = np.linalg.eigh(B)
    eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]

    # Drop negative eigenvalues (edit distance is not Euclidean)
    k_pos = (eigenvalues > 1e-10).sum()
    k_use = min(k, k_pos)
    coords = eigenvectors[:, :k_use] * np.sqrt(eigenvalues[:k_use])
    if k_use < k:
        coords = np.hstack([coords, np.zeros((n, k - k_use))])
    return coords


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
        Optimized sRGB colors in [0, 1] of shape (n, 3), one row per item.
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

    def build_rgb(params):
        rgb = jax.nn.sigmoid(params["logit_rgb"])
        full = jnp.zeros((n, 3))
        full = full.at[free_idx].set(rgb)
        if fixed_idx.shape[0] > 0:
            full = full.at[fixed_idx].set(fixed_rgb)
        return full

    def loss_fn(params):
        lab = rgb_to_lab(build_rgb(params))
        color_dists = jnp.sqrt(((lab[idx_i] - lab[idx_j]) ** 2).sum(axis=-1) + 1e-8)
        stress = jnp.mean((color_dists - targets) ** 2)
        coverage = -(lab[:, 1].var() + lab[:, 2].var())
        return stress + 0.5 * coverage

    if seed is None:
        # MDS warm-start
        coords = classical_mds(D)
        target_half_ranges = np.array([25.0, 50.0, 50.0])
        uniform_scale = (target_half_ranges / np.abs(coords).max(axis=0)).min()
        Lab_init = coords * uniform_scale + np.array([55.0, 0.0, 0.0])
        rgb_init = np.clip(lab_to_rgb(Lab_init), 1e-4, 1 - 1e-4)
        logit_init = np.log(rgb_init / (1 - rgb_init))
    else:
        rng = jax.random.PRNGKey(seed)
        logit_init = jax.random.normal(rng, shape=(n, 3))

    params0 = {"logit_rgb": jnp.array(logit_init[np.array(free_idx)])}

    params_opt = optimize_gradient_descent(
        params0,
        loss_fn,
        learning_rate=learning_rate,
        n_iters=n_iters,
        callback=callback,
    )

    return np.array(build_rgb(params_opt))
