"""Optimizing label positions"""

import numpy as np
from jax import numpy as jnp
from matplotlib import pyplot as plt
from pydantic import BaseModel, ConfigDict, model_validator

from .. import components
from ..base import ObjectiveTerm, OptimizationProblemTemplate


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


def initialize(input_parameters, _seed):
    return {"rectangle_positions": input_parameters["point_positions"].copy()}


def calculate_intersection_loss(optim_vars, input_parameters):
    """Pairwise intersections between label bounding boxes"""
    bbox_matrix = jnp.stack(
        [
            optim_vars["rectangle_positions"],
            optim_vars["rectangle_positions"] + input_parameters["rectangle_sizes"],
        ],
        axis=1,
    )

    interlabel_inters_matrix = components.multiple_bbox_intersections(
        bbox_matrix, bbox_matrix
    )
    interlabel_inters_array = interlabel_inters_matrix[
        np.triu_indices(interlabel_inters_matrix.shape[0], 1)
    ]
    return jnp.sum(interlabel_inters_array)


def calculate_point_label_distance_loss(optim_vars, input_parameters):
    """Distances between points and the respective labels"""
    return jnp.sum(
        (optim_vars["rectangle_positions"] - input_parameters["point_positions"]) ** 2
    )


def plot_rectangles(optim_vars, input_parameters):
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


label_position_template = OptimizationProblemTemplate(
    terms=[
        ObjectiveTerm(
            name="intersection_loss",
            compute=calculate_intersection_loss,
            multiplier=5.0,
        ),
        ObjectiveTerm(
            name="point_label_distance", compute=calculate_point_label_distance_loss
        ),
    ],
    initialize=initialize,
    input_params_class=LabelPositionParams,
    plot_configuration=plot_rectangles,
)
