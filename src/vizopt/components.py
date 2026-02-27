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


def calculate_total_width_penalty(node_xys: np.ndarray):
    """A penalty for the overall width and height of the drawing.

    Args:
        node_xys: Array of node positions with shape (n, 2).
    """
    return jnp.sum(jnp.max(node_xys, axis=0) - jnp.min(node_xys, axis=0))
