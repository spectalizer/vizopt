"""Star-shaped boundary optimization for sets of circles.

Finds star-shaped (radially convex) regions enclosing each set of circles
while minimizing area/perimeter and avoiding overlap with other sets.

Each boundary is parametrized as a center + K radii at uniformly-spaced angles.
General star-domain loss terms and helpers live in
:mod:`vizopt.components.stars`.
"""

import jax.numpy as jnp
import networkx as nx
import numpy as np

from ...base import (
    Callback,
    ObjectiveTerm,
    OptimConfig,
    OptimizationProblem,
    OptimizationProblemTemplate,
    VizOptimizer,
)
from ...components.stars import (
    Discrete,
    StarRepresentation,
    _build_membership,
    _init_centers_and_radii,
    _multi_term_area,
    _multi_term_circle_collision,
    _multi_term_convexity,
    _multi_term_enclosure_movable,
    _multi_term_exclusion_movable,
    _multi_term_label_element_exclusion,
    _multi_term_label_enclosure,
    _multi_term_label_label_collision,
    _multi_term_label_top_attraction,
    _multi_term_min_radius,
    _multi_term_perimeter,
    _multi_term_position_anchor,
    _multi_term_set_attraction,
    _multi_term_smoothness,
    _multi_term_total_bounding_box,
    _svg_configuration_movable,
)
from ...schedules import TermSchedules
from .graph_utils import offsets_from_graph


def _leaf_circles_from_graph(inclusion_graph: nx.DiGraph):
    names = [n for n in inclusion_graph.nodes if inclusion_graph.out_degree(n) == 0]
    circles = np.array(
        [
            [*inclusion_graph.nodes[n]["center"], inclusion_graph.nodes[n]["r"]]
            for n in names
        ],
        dtype=np.float32,
    )
    name_to_idx = {name: i for i, name in enumerate(names)}
    return names, circles, name_to_idx


def _sets_from_graph(inclusion_graph: nx.DiGraph, leaf_names, name_to_idx):
    leaf_set = set(leaf_names)
    set_names = [
        n
        for n in nx.topological_sort(inclusion_graph)
        if inclusion_graph.out_degree(n) > 0
    ]
    sets_idx = [
        sorted(
            name_to_idx[n]
            for n in nx.descendants(inclusion_graph, sname)
            if n in leaf_set
        )
        for sname in set_names
    ]
    return set_names, sets_idx


class EulerDiagram(VizOptimizer):
    """Star-shaped boundary optimizer for sets of circles.

    Circle positions are optimization variables alongside the star boundaries.
    Construct directly from arrays or via :meth:`from_graph`.

    Args:
        circles: array of shape ``(N, 3)`` with columns ``[cx, cy, r]``.
        sets: list of S subsets, each a collection of integer indices into ``circles``.
        weight_area: weight for the area objective.
        weight_perimeter: weight for the perimeter objective.
        weight_enclosure: weight for the enclosure penalty.
        weight_exclusion: weight for the exclusion penalty.
        weight_smoothness: weight for the smoothness penalty.
        weight_convexity: weight for the convexity penalty. Default 0.0 (disabled).
        weight_position_anchor: weight for penalizing circle positions deviating
            from their initial positions.
        weight_circle_collision: weight for penalizing overlapping circles.
        weight_bounding_box: weight for minimizing total bounding box extent.
            Default 0.0 (disabled).
        weight_set_attraction: weight for pulling circles toward their set center.
            Default 0.0 (disabled).
        circle_collision_alpha: linear term coefficient in the circle collision
            penalty. Default 0.0 (pure quadratic).
        convexity_alpha: linear term coefficient in the convexity penalty.
            Default 1.0.
        offsets: padding per ``(set, circle)`` pair. Scalar, shape ``(N,)``,
            or shape ``(S, N)``.
        label_rect_size: ``(hw, hh)`` half-extents of the label rectangle.
            When set, each star boundary encloses a floating label rect whose
            position is an optimization variable.
        weight_label_enclosure: weight for the label enclosure term.
        weight_label_element_exclusion: weight for label-circle exclusion.
        weight_label_collision: weight for label-label collision.
        weight_label_top: weight for the upward-attraction term on labels.
        representation: star domain parametrization. One of :class:`Discrete`
            (default), :class:`Fourier`, or :class:`BSpline`.
        term_schedules: optional dict or :class:`~vizopt.schedules.TermSchedules`
            mapping term name to a JAX-compatible schedule callable.
        set_names: display names for the S sets. Defaults to ``["Set 0", ...]``.
        leaf_names: display names for the N circles. Defaults to ``[0, 1, ...]``.
        set_colors: colors for the S sets used in plot/SVG output.
    """

    def __init__(
        self,
        circles,
        sets,
        *,
        weight_area: float = 1.0,
        weight_perimeter: float = 1.0,
        weight_enclosure: float = 20.0,
        weight_exclusion: float = 10.0,
        weight_smoothness: float = 1.0,
        weight_convexity: float = 0.0,
        weight_position_anchor: float = 1.0,
        weight_circle_collision: float = 10.0,
        weight_bounding_box: float = 0.0,
        weight_set_attraction: float = 0.0,
        circle_collision_alpha: float = 0.0,
        convexity_alpha: float = 1.0,
        offsets: float | np.ndarray = 0.1,
        label_rect_size: tuple[float, float] | None = None,
        weight_label_enclosure: float = 10.0,
        weight_label_element_exclusion: float = 10.0,
        weight_label_collision: float = 10.0,
        weight_label_top: float = 1.0,
        representation: StarRepresentation | None = None,
        term_schedules=None,
        set_names: list[str] | None = None,
        leaf_names: list | None = None,
        set_colors=None,
    ):
        self.circles = np.asarray(circles, dtype=np.float32)
        if self.circles.ndim == 1:
            self.circles = self.circles[None, :]
        self.sets = sets
        self.weight_area = weight_area
        self.weight_perimeter = weight_perimeter
        self.weight_enclosure = weight_enclosure
        self.weight_exclusion = weight_exclusion
        self.weight_smoothness = weight_smoothness
        self.weight_convexity = weight_convexity
        self.weight_position_anchor = weight_position_anchor
        self.weight_circle_collision = weight_circle_collision
        self.weight_bounding_box = weight_bounding_box
        self.weight_set_attraction = weight_set_attraction
        self.circle_collision_alpha = circle_collision_alpha
        self.convexity_alpha = convexity_alpha
        self.offsets = offsets
        self.label_rect_size = label_rect_size
        self.weight_label_enclosure = weight_label_enclosure
        self.weight_label_element_exclusion = weight_label_element_exclusion
        self.weight_label_collision = weight_label_collision
        self.weight_label_top = weight_label_top
        self.representation = representation if representation is not None else Discrete()
        self.term_schedules = term_schedules
        S, N = len(sets), len(self.circles)
        self.set_names = set_names if set_names is not None else [f"Set {s}" for s in range(S)]
        self.leaf_names = leaf_names if leaf_names is not None else list(range(N))
        self.set_colors = set_colors

    @classmethod
    def from_graph(
        cls,
        inclusion_graph: nx.DiGraph,
        *,
        offsets=None,
        **kwargs,
    ) -> "EulerDiagram":
        """Construct from a DiGraph where leaves are circles and internal nodes are sets.

        Leaf nodes (out-degree 0) become circles; internal nodes (out-degree > 0)
        become sets. A leaf belongs to a set if it is a descendant of that set.

        Args:
            inclusion_graph: DiGraph with parent→child edges (edge ``(u, v)`` means
                ``v ⊂ u``). Leaf nodes must carry ``center`` (``[x, y]``) and
                ``r`` (float) attributes. Internal nodes may carry a ``color``
                attribute used in plot output.
            offsets: padding per ``(set, circle)`` pair. When ``None`` (default),
                computed automatically from the graph hierarchy via
                :func:`~vizopt.templates.euler.graph_utils.offsets_from_graph`.
            **kwargs: forwarded to :meth:`__init__`.

        Returns:
            A configured :class:`EulerDiagram` ready to call :meth:`optimize`.
        """
        leaf_names, circles, name_to_idx = _leaf_circles_from_graph(inclusion_graph)
        set_names, sets = _sets_from_graph(inclusion_graph, leaf_names, name_to_idx)
        if offsets is None:
            offsets = offsets_from_graph(inclusion_graph, set_names, leaf_names)
        set_colors = [inclusion_graph.nodes[n].get("color") for n in set_names]
        if any(c is None for c in set_colors):
            set_colors = None
        return cls(
            circles,
            sets,
            offsets=offsets,
            set_names=set_names,
            leaf_names=leaf_names,
            set_colors=set_colors,
            **kwargs,
        )

    def _build_problem(self) -> OptimizationProblem:
        circles_array = self.circles
        sets = self.sets
        representation = self.representation
        N, S = len(circles_array), len(sets)

        angles = np.linspace(0, 2 * np.pi, representation.k_angles, endpoint=False).astype(
            np.float32
        )
        angles_jnp = jnp.array(angles)

        initial_circle_positions = circles_array[:, :2].copy()
        circle_radii = circles_array[:, 2].copy()

        membership = _build_membership(S, N, sets)
        initial_centers, initial_radii = _init_centers_and_radii(circles_array, sets, angles)
        offsets_array = np.broadcast_to(
            np.asarray(self.offsets, dtype=np.float32), (S, N)
        ).copy()

        has_label = self.label_rect_size is not None
        if has_label:
            label_hw = np.full(S, self.label_rect_size[0], dtype=np.float32)
            label_hh = np.full(S, self.label_rect_size[1], dtype=np.float32)
            k_top = int(np.argmin(np.abs(angles - np.pi / 2)))
        else:
            label_hw = label_hh = np.empty(S, dtype=np.float32)
            k_top = 0

        input_parameters = {
            "circle_radii": circle_radii,
            "initial_circle_positions": initial_circle_positions,
            "angles": angles,
            "membership": membership,
            "offsets": offsets_array,
            "circle_collision_alpha": np.float32(self.circle_collision_alpha),
            "convexity_alpha": np.float32(self.convexity_alpha),
        }
        if has_label:
            input_parameters["label_rect_hw"] = label_hw
            input_parameters["label_rect_hh"] = label_hh

        init_vars = representation.initialize_vars(S, initial_radii, initial_centers)

        pos_scale_x = max(float(np.std(circles_array[:, 0])), float(circle_radii.mean()))
        pos_scale_y = max(float(np.std(circles_array[:, 1])), float(circle_radii.mean()))
        rad_scale = float(initial_radii.mean())
        pos_scale_arr = np.array([pos_scale_x, pos_scale_y], dtype=np.float32)
        var_scales = {"centers": pos_scale_arr, "circle_positions": pos_scale_arr}
        for key in init_vars:
            if key != "centers":
                var_scales[key] = np.float32(rad_scale)
        if has_label:
            var_scales["label_positions"] = pos_scale_arr

        if has_label:
            initial_label_positions = initial_centers.copy()
            initial_label_positions[:, 1] += initial_radii[:, k_top] - label_hh

        def initialize(_, seed):
            d = {
                **{k: v.copy() for k, v in init_vars.items()},
                "circle_positions": initial_circle_positions.copy(),
            }
            if has_label:
                d["label_positions"] = initial_label_positions.copy()
            return d

        def wrap(fn):
            return representation.wrap(fn, angles_jnp)

        schedules = (
            self.term_schedules.schedules
            if isinstance(self.term_schedules, TermSchedules)
            else self.term_schedules
        ) or {}

        terms = [
            ObjectiveTerm("enclosure", wrap(_multi_term_enclosure_movable), self.weight_enclosure, schedules.get("enclosure")),
            ObjectiveTerm("exclusion", wrap(_multi_term_exclusion_movable), self.weight_exclusion, schedules.get("exclusion")),
            ObjectiveTerm("min_radius", wrap(_multi_term_min_radius), 10.0, schedules.get("min_radius")),
            ObjectiveTerm("smoothness", wrap(_multi_term_smoothness), self.weight_smoothness, schedules.get("smoothness")),
            ObjectiveTerm("convexity", wrap(_multi_term_convexity), self.weight_convexity, schedules.get("convexity")),
            ObjectiveTerm("area", wrap(_multi_term_area), self.weight_area, schedules.get("area")),
            ObjectiveTerm("perimeter", wrap(_multi_term_perimeter), self.weight_perimeter, schedules.get("perimeter")),
            ObjectiveTerm("position_anchor", _multi_term_position_anchor, self.weight_position_anchor, schedules.get("position_anchor")),
            ObjectiveTerm("circle_collision", _multi_term_circle_collision, self.weight_circle_collision, schedules.get("circle_collision")),
            ObjectiveTerm("bounding_box", wrap(_multi_term_total_bounding_box), self.weight_bounding_box, schedules.get("bounding_box")),
            ObjectiveTerm("set_attraction", _multi_term_set_attraction, self.weight_set_attraction, schedules.get("set_attraction")),
        ]
        if has_label:
            terms += [
                ObjectiveTerm("label_enclosure", wrap(_multi_term_label_enclosure), self.weight_label_enclosure, schedules.get("label_enclosure")),
                ObjectiveTerm("label_element_exclusion", _multi_term_label_element_exclusion, self.weight_label_element_exclusion, schedules.get("label_element_exclusion")),
                ObjectiveTerm("label_collision", _multi_term_label_label_collision, self.weight_label_collision, schedules.get("label_collision")),
                ObjectiveTerm("label_top", _multi_term_label_top_attraction, self.weight_label_top, schedules.get("label_top")),
            ]

        return OptimizationProblemTemplate(
            terms=terms,
            initialize=initialize,
            plot_configuration=_make_plot_configuration(
                self.set_names, self.leaf_names, representation, has_label, self.set_colors
            ),
            svg_configuration=representation.make_svg_configuration(_svg_configuration_movable),
        ).instantiate(input_parameters, var_scales=var_scales)

    @property
    def sets_(self) -> list[dict]:
        """Star boundary dicts from the last optimization result.

        Each dict has ``"center"``, ``"radii"``, ``"angles"``, and (when a label
        rect was used) ``"label_center"``.

        Raises:
            ValueError: If :meth:`optimize` has not been called yet.
        """
        if not hasattr(self, "result_"):
            raise ValueError("No result yet — call optimize() first.")
        optim_vars = self.result_.optim_vars
        angles = self.problem_.input_parameters["angles"]
        radii_arr = np.array(self.representation.to_radii(optim_vars, jnp.array(angles)))
        has_label = "label_positions" in optim_vars
        return [
            {
                "center": np.array(optim_vars["centers"][s]),
                "radii": radii_arr[s],
                "angles": angles,
                **({"label_center": np.array(optim_vars["label_positions"][s])} if has_label else {}),
                **self.representation.extra_results(s, optim_vars),
            }
            for s in range(len(self.set_names))
        ]

    @property
    def circles_(self) -> np.ndarray:
        """Optimized circle positions as an ``(N, 3)`` array of ``[cx, cy, r]``.

        Raises:
            ValueError: If :meth:`optimize` has not been called yet.
        """
        if not hasattr(self, "result_"):
            raise ValueError("No result yet — call optimize() first.")
        circle_radii = self.problem_.input_parameters["circle_radii"]
        return np.concatenate(
            [np.array(self.result_.optim_vars["circle_positions"]), circle_radii[:, None]], axis=1
        )


def _make_plot_configuration(set_names, leaf_names, representation, has_label, set_colors=None):
    def plot_configuration(optim_vars, input_params, show_arrows=False, ax=None):
        from matplotlib import pyplot as plt

        angles = input_params["angles"]
        circle_radii = input_params["circle_radii"]
        initial_positions = input_params["initial_circle_positions"]
        radii_arr = np.array(representation.to_radii(optim_vars, jnp.array(angles)))
        centers = np.array(optim_vars["centers"])
        circle_positions = np.array(optim_vars["circle_positions"])
        S = len(set_names)
        colors = set_colors if set_colors is not None else plt.cm.tab10(np.linspace(0, 0.9, S))

        _own_figure = ax is None
        if _own_figure:
            _, ax = plt.subplots(figsize=(7, 7))
        for s, (name, color) in enumerate(zip(set_names, colors)):
            cx, cy = centers[s]
            radii = radii_arr[s]
            bx = np.append(cx + radii * np.cos(angles), cx + radii[0] * np.cos(angles[0]))
            by = np.append(cy + radii * np.sin(angles), cy + radii[0] * np.sin(angles[0]))
            ax.fill(bx, by, alpha=0.15, color=color)
            ax.plot(bx, by, color=color, linewidth=2, label=name)

        for i, (name, pos) in enumerate(zip(leaf_names, circle_positions)):
            r = float(circle_radii[i])
            ox, oy = float(initial_positions[i, 0]), float(initial_positions[i, 1])
            nx_, ny_ = float(pos[0]), float(pos[1])
            if show_arrows and np.hypot(nx_ - ox, ny_ - oy) > 0.01:
                ax.annotate("", xy=(nx_, ny_), xytext=(ox, oy),
                            arrowprops=dict(arrowstyle="->", color="k", lw=1.0))
                ax.add_patch(plt.Circle((ox, oy), r, facecolor="none", edgecolor="dimgray",
                                        linewidth=1, linestyle="--", alpha=0.4))
            ax.add_patch(plt.Circle((nx_, ny_), r, facecolor="lightyellow", alpha=0.9,
                                    edgecolor="dimgray", linewidth=1.5))
            ax.text(nx_, ny_, str(name), ha="center", va="center", fontsize=9, fontweight="bold")

        if has_label:
            label_positions = np.array(optim_vars["label_positions"])
            for name, color, lp in zip(set_names, colors, label_positions):
                ax.text(float(lp[0]), float(lp[1]), name, ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)
        else:
            ax.legend(fontsize=9)

        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.margins(0.15)
        ax.axis("off")
        if _own_figure:
            plt.tight_layout()

    return plot_configuration
