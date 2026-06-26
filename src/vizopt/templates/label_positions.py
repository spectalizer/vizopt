"""Optimizing label positions"""

import numpy as np
from jax import numpy as jnp
from matplotlib import pyplot as plt
from pydantic import BaseModel, ConfigDict, model_validator

from ..base import ObjectiveTerm, OptimizationProblem, OptimizationProblemTemplate, VizOptimizer
from ..components import common


class LabelPositionParams(BaseModel):
    """Input parameters for the label position optimization problem"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    point_positions: np.ndarray  # shape (n, 2)
    rectangle_sizes: np.ndarray  # shape (n, 2)

    @model_validator(mode="after")
    def check_shapes(self) -> "LabelPositionParams":
        for name, arr in [
            ("point_positions", self.point_positions),
            ("rectangle_sizes", self.rectangle_sizes),
        ]:
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(f"{name} must have shape (n, 2), got {arr.shape}")
        if self.point_positions.shape[0] != self.rectangle_sizes.shape[0]:
            raise ValueError(
                f"point_positions and rectangle_sizes must have the same n, "
                f"got {self.point_positions.shape[0]} and {self.rectangle_sizes.shape[0]}"
            )
        return self


def _initialize(input_parameters, _seed):
    return {"rectangle_positions": input_parameters["point_positions"].copy()}


def _calculate_intersection_loss(optim_vars, input_parameters):
    """Pairwise intersections between label bounding boxes"""
    bbox_matrix = jnp.stack(
        [
            optim_vars["rectangle_positions"],
            optim_vars["rectangle_positions"] + input_parameters["rectangle_sizes"],
        ],
        axis=1,
    )

    interlabel_inters_matrix = common.multiple_bbox_intersections(
        bbox_matrix, bbox_matrix
    )
    interlabel_inters_array = interlabel_inters_matrix[
        np.triu_indices(interlabel_inters_matrix.shape[0], 1)
    ]
    return jnp.sum(interlabel_inters_array)


def _calculate_point_label_distance_loss(optim_vars, input_parameters):
    """Distances between points and the respective labels"""
    return jnp.sum(
        (optim_vars["rectangle_positions"] - input_parameters["point_positions"]) ** 2
    )


def _plot_rectangles(optim_vars, input_parameters):
    """Plot label rectangles and points"""
    _, ax = plt.subplots(figsize=(4, 3))
    n_rect = optim_vars["rectangle_positions"].shape[0]
    for i_rect in range(n_rect):
        ax.add_patch(
            plt.Rectangle(
                optim_vars["rectangle_positions"][i_rect, :],
                input_parameters["rectangle_sizes"][i_rect, 0],
                input_parameters["rectangle_sizes"][i_rect, 1],
                alpha=0.5,
            )
        )
        link_x = [
            optim_vars["rectangle_positions"][i_rect, 0],
            input_parameters["point_positions"][i_rect, 0],
        ]
        link_y = [
            optim_vars["rectangle_positions"][i_rect, 1],
            input_parameters["point_positions"][i_rect, 1],
        ]
        ax.plot(link_x, link_y)

    ax.scatter(
        input_parameters["point_positions"][:, 0],
        input_parameters["point_positions"][:, 1],
    )


class LabelPositionOptimizer(VizOptimizer):
    """Optimize label rectangle positions to avoid overlap while staying near points.

    Args:
        point_positions: Array of shape ``(n, 2)`` with the anchor point coordinates.
        rectangle_sizes: Array of shape ``(n, 2)`` with ``(width, height)`` of each label.
        weight_intersection: Weight for the pairwise bounding-box intersection loss.
        weight_distance: Weight for the point-to-label distance loss.
    """

    def __init__(
        self,
        point_positions,
        rectangle_sizes,
        *,
        weight_intersection: float = 5.0,
        weight_distance: float = 1.0,
    ):
        self.point_positions = np.asarray(point_positions, dtype=np.float32)
        self.rectangle_sizes = np.asarray(rectangle_sizes, dtype=np.float32)
        self.weight_intersection = weight_intersection
        self.weight_distance = weight_distance

    def _build_problem(self) -> OptimizationProblem:
        input_parameters = {
            "point_positions": self.point_positions,
            "rectangle_sizes": self.rectangle_sizes,
        }
        return OptimizationProblemTemplate(
            terms=[
                ObjectiveTerm("intersection_loss", _calculate_intersection_loss, self.weight_intersection),
                ObjectiveTerm("point_label_distance", _calculate_point_label_distance_loss, self.weight_distance),
            ],
            initialize=_initialize,
            input_params_class=LabelPositionParams,
            plot_configuration=_plot_rectangles,
        ).instantiate(input_parameters)

    @property
    def label_positions_(self) -> np.ndarray:
        """Optimized label rectangle positions, shape ``(n, 2)``.

        Raises:
            ValueError: If :meth:`optimize` has not been called yet.
        """
        if not hasattr(self, "result_"):
            raise ValueError("No result yet — call optimize() first.")
        return np.array(self.result_.optim_vars["rectangle_positions"])
