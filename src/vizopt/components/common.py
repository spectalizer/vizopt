import jax
import numpy as np
from jax import numpy as jnp


def multiple_bbox_intersections(bbox_matrix: np.ndarray, other_bbox_matrix: np.ndarray):
    """Calculate the pairwise intersections of two sets of bounding boxes

    This vectorized implementation is more efficient than the avoided double for loop

    Args:
        bbox_matrix: numpy array of shape (n, 2, 2)
            dimensions: points, min and max, xy coordinates
        other_bbox_matrix: numpy array of shape (m, 2, 2)
            dimensions: points, min and max, xy coordinates

    Returns:
        numpy array of shape (n, m)
    """
    rep_bbox_matrix = jnp.repeat(bbox_matrix, other_bbox_matrix.shape[0], axis=0)
    rep_other_bbox_matrix = jnp.tile(other_bbox_matrix, (bbox_matrix.shape[0], 1, 1))

    coord_max = rep_bbox_matrix[:, 1, :]
    coord_min = rep_bbox_matrix[:, 0, :]
    coord_max_other = rep_other_bbox_matrix[:, 1, :]
    coord_min_other = rep_other_bbox_matrix[:, 0, :]
    intersects = jnp.clip(
        coord_max - coord_min_other, a_max=coord_max_other - coord_min
    )
    x_intersect = jnp.clip(intersects[:, 0], 0, np.inf)
    y_intersect = jnp.clip(intersects[:, 1], 0, np.inf)
    intersect_prods = x_intersect * y_intersect
    return intersect_prods.reshape(bbox_matrix.shape[0], -1)


def should_be_positive_activation(x_value: float, factor=100.0):
    """A penalty for negative values."""
    return factor * jax.nn.relu(-x_value)


def calculate_total_width_penalty_ignoring_radii(node_xys: np.ndarray):
    """A penalty for the overall width and height of the drawing.

    Args:
        node_xys: Array of node positions with shape (n, 2).
    """
    return jnp.sum(jnp.max(node_xys, axis=0) - jnp.min(node_xys, axis=0))


def calculate_total_width_penalty_for_circular_layout(
    node_xys: np.ndarray, node_radii: np.ndarray
):
    """A penalty for the overall width and height of the drawing with circular nodes.

    Args:
        node_xys: Array of node positions with shape (n, 2).
        node_radii: Array of node radii with shape (n,).
    """
    return jnp.sum(
        jnp.max(node_xys + node_radii.reshape(-1, 1), axis=0)
        - jnp.min(node_xys - node_radii.reshape(-1, 1), axis=0)
    )


def calculate_collision_penalty(node_xys, node_radii, collision_pairs, offset=1.0):
    """Vectorized collision penalty for node pairs that should not overlap.

    Args:
        node_xys: Array of node positions with shape (n, 2).
        node_radii: Array of node radii with shape (n,).
        collision_pairs: Integer array of shape (k, 2) with index pairs to check.
        offset: Minimum required gap between circle boundaries.

    Returns:
        Scalar penalty value.
    """
    if len(collision_pairs) == 0:
        return jnp.array(0.0)

    pos_a = node_xys[collision_pairs[:, 0]]
    pos_b = node_xys[collision_pairs[:, 1]]
    dists = jnp.sqrt(jnp.sum((pos_a - pos_b) ** 2, axis=1))
    radii_a = node_radii[collision_pairs[:, 0]]
    radii_b = node_radii[collision_pairs[:, 1]]
    d_minus_radiuses = dists - offset - radii_a - radii_b
    return jnp.sum(should_be_positive_activation(d_minus_radiuses))
