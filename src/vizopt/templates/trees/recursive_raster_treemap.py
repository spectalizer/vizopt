"""Recursive raster-based treemap.

Lays out a rooted tree (or forest) as nested, space-filling star-shaped
domains: each node's children are jointly optimized with
`~vizopt.templates.raster_stars.RasterStarOptimizer` (raster collision for
mutual exclusion between siblings, an analytic containment term against the
node's own already-fitted boundary, and a compactness term to close the
whitespace a plain exclusion+area+perimeter objective would otherwise leave
behind), then recursed into depth-first, using each child's *achieved* area
— not its nominal target — as the next level's container budget.

This does not subclass `~vizopt.base.VizOptimizer`: that ABC is built around
a single `~vizopt.base.OptimizationProblem`, while a recursive treemap runs
one independent optimization per non-leaf node and stitches the results
together, so none of VizOptimizer's `_build_problem`/`plot`/`animate`
contract applies.
"""

import networkx as nx
import numpy as np
from jax import numpy as jnp
from matplotlib import pyplot as plt

from ...base import ObjectiveTerm, OptimConfig
from ...components.common import calculate_total_width_penalty_for_circular_layout
from ...components.stars import (
    Discrete,
    StarRepresentation,
    radius_at_angle,
    star_polygon_area,
)
from ...treemap import squarify_layout
from ..raster_stars import RasterStarOptimizer
from ..star_vs_star import _dist_and_angle

# ---------------------------------------------------------------------------
# Extra loss terms
#
# optim_vars keys: "centers" (n_sets, 2), "radii" (n_sets, K)
# ---------------------------------------------------------------------------


def _term_compactness(optim_vars, _input_params):
    """Bounding-box penalty pulling domains together to close whitespace.

    Raster exclusion is purely repulsive and only engages once domains are
    already touching; squarify-style seeding leaves every inscribed circle
    with slack relative to its assigned rectangle (worst at the corners),
    so nothing otherwise closes that gap.
    """
    centers = optim_vars["centers"]
    max_radii = jnp.max(optim_vars["radii"], axis=1)
    return calculate_total_width_penalty_for_circular_layout(centers, max_radii)


def _term_contained_in_frozen_parent(optim_vars, input_params):
    """Boundary of every domain must stay inside a fixed (non-optimized) parent star.

    Mirrors `~vizopt.templates.star_vs_star._multi_term_star_enclosure`, but
    the outer side is a constant read from `input_params` rather than a
    variable, since the parent's boundary was already fitted at the previous
    recursion depth and is not re-optimized here.

    input_params keys: "angles" (K,), "parent_center" (2,), "parent_radii" (K,)
        — parent_radii must be interpolated on the same K/angle grid as "angles".
    Optional input_params keys: "containment_offset" (float)
    """
    centers = optim_vars["centers"]
    radii = optim_vars["radii"]
    angles = input_params["angles"]
    parent_center = input_params["parent_center"]
    parent_radii = input_params["parent_radii"]
    n_sets, K = radii.shape

    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)  # (K, 2)
    points = (
        centers[:, None, :] + radii[:, :, None] * directions[None, :, :]
    )  # (n_sets, K, 2)

    diff = points - parent_center[None, None, :]  # (n_sets, K, 2)
    dist, alpha = _dist_and_angle(diff)  # (n_sets, K) each

    delta_theta = 2 * jnp.pi / K
    frac_idx = (alpha % (2 * jnp.pi)) / delta_theta
    idx_lo = jnp.floor(frac_idx).astype(jnp.int32) % K
    idx_hi = (idx_lo + 1) % K
    w_hi = frac_idx - jnp.floor(frac_idx)

    r_lo = parent_radii[idx_lo]  # (n_sets, K)
    r_hi = parent_radii[idx_hi]
    r_interp = (1.0 - w_hi) * r_lo + w_hi * r_hi

    offset = input_params.get("containment_offset", 0.0)
    violation = jnp.maximum(0.0, dist - (r_interp - offset))
    return jnp.sum(violation**2)


# ---------------------------------------------------------------------------
# Sibling-group fitting
# ---------------------------------------------------------------------------


def _branching_buckets(graph: nx.DiGraph, sizes: dict) -> list[int]:
    """Sorted distinct branching factors across every non-empty node in graph.

    A node's branching factor is its number of positive-weight children.
    Used to pad each sibling group to the smallest *observed* branching
    factor that covers it, rather than a single tree-wide maximum: the
    raster exclusion term is O(n_sets^2 x H x W), so padding a 2-child
    directory up to the tree's largest (e.g. 12-child) node wastes real
    compute every iteration, not just compile time. Padding to the next
    actual branching factor present in the tree is never worse than the
    global-max scheme for any node (the node that IS the max still gets
    padded to exactly its own size) and is a strict generalization: passing
    a single-element override reproduces the old uniform-padding behaviour.
    """
    counts = {
        len([c for c in graph.successors(n) if sizes.get(c, 0) > 0])
        for n in graph.nodes
        if sizes.get(n, 0) > 0 and graph.out_degree(n) > 0
    }
    return sorted(counts)


def _bucket_size(n: int, buckets: list[int]) -> int:
    """Smallest value in buckets that is >= n."""
    for b in buckets:
        if b >= n:
            return b
    return n


def _pad_siblings(nodes, target_areas, initial_centers, n_pad_to):
    """Pad a sibling group to a fixed size with near-zero-area dummy domains.

    Dummy slots are co-located with the first real sibling: their near-zero
    target area shrinks them away, so where they start barely matters, but
    placing them off in empty space would needlessly inflate the compactness
    bounding-box term.
    """
    n = len(nodes)
    n_pad = n_pad_to - n
    if n_pad <= 0:
        return list(nodes), list(target_areas), initial_centers
    pad_names = [f"__pad_{i}" for i in range(n_pad)]
    pad_targets = [1e-6] * n_pad
    pad_centers = np.tile(initial_centers[:1], (n_pad, 1))
    return (
        list(nodes) + pad_names,
        list(target_areas) + pad_targets,
        np.concatenate([initial_centers, pad_centers], axis=0),
    )


def _fit_siblings(
    nodes,
    target_areas,
    initial_centers,
    *,
    representation=None,
    parent_center=None,
    parent_radii=None,
    grid_resolution=48,
    n_iters=1200,
    learning_rate=0.01,
    exclusion_offset=0.1,
    containment_offset=0.1,
    weight_compactness=5.0,
    weight_containment=30.0,
    weight_target_area=20.0,
    weight_area=0.3,
    weight_perimeter=0.3,
    weight_exclusion=10.0,
    weight_smoothness=1.0,
    temperature=0.08,
    early_stop_patience=300,
    early_stop_tol=1e-3,
):
    """Fit RasterStarOptimizer to one sibling group, optionally contained in a frozen parent."""
    avg_radius = float(np.sqrt(np.mean(target_areas) / np.pi))
    optimizer = RasterStarOptimizer(
        n_sets=len(nodes),
        initial_centers=initial_centers,
        representation=representation,
        target_areas=target_areas,
        initial_radius=avg_radius,
        grid_resolution=grid_resolution,
        weight_target_area=weight_target_area,
        weight_area=weight_area,
        weight_perimeter=weight_perimeter,
        weight_exclusion=weight_exclusion,
        weight_smoothness=weight_smoothness,
        exclusion_offset=exclusion_offset,
        temperature=temperature,
    )
    optimizer.problem_ = optimizer._build_problem()
    angles_jnp = jnp.array(optimizer.problem_.input_parameters["angles"])
    wrap = optimizer.representation.wrap
    optimizer.problem_.terms.append(
        ObjectiveTerm(
            "compactness",
            wrap(_term_compactness, angles_jnp),
            multiplier=weight_compactness,
        )
    )
    if parent_center is not None:
        optimizer.problem_.input_parameters["parent_center"] = jnp.array(parent_center)
        optimizer.problem_.input_parameters["parent_radii"] = jnp.array(parent_radii)
        optimizer.problem_.input_parameters["containment_offset"] = containment_offset
        optimizer.problem_.terms.append(
            ObjectiveTerm(
                "contained_in_parent",
                wrap(_term_contained_in_frozen_parent, angles_jnp),
                multiplier=weight_containment,
            )
        )
    optimizer.result_ = optimizer.problem_.optimize(
        OptimConfig(
            n_iters=n_iters,
            learning_rate=learning_rate,
            early_stop_patience=early_stop_patience,
            early_stop_tol=early_stop_tol,
        ),
        callback=lambda *_: None,
    )
    return dict(zip(nodes, optimizer.sets_))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class RasterTreemapOptimizer:
    """Recursive raster-based star-domain treemap for a rooted tree.

    Args:
        graph: A tree (arborescence), e.g. from
            `~vizopt.introspection.build_file_tree`. Leaves have out-degree 0;
            internal nodes (out-degree > 0) are recursed into.
        sizes: Node -> subtree weight, e.g. from
            `~vizopt.introspection.compute_subtree_sizes`. Nodes with weight 0
            are dropped, matching `~vizopt.treemap.squarify_layout`'s
            convention (it requires positive weights).
        root: Node to start from; its direct children become the top,
            unconstrained level. Defaults to the graph's unique in-degree-0 node.
        mean_leaf_area: Target area for an average-weight top-level child;
            sets the overall scale of the layout (root's budget is
            `mean_leaf_area * n_root_children`).
        fill_fraction: Fraction of a parent's *achieved* area its children are
            allowed to target. Below 1.0, leaving slack for packing
            inefficiency (gaps at corners, exclusion margins) rather than
            forcing children to claim area they can't actually occupy.
        representation: A `~vizopt.components.stars.StarRepresentation` instance
            (`Discrete`, `Fourier`, or `BSpline`) controlling the boundary
            parametrisation, shared across every level of the recursion (so a
            parent's frozen `parent_radii` — interpolated on its own
            `k_angles` grid — always lines up with the `k_angles` its
            children optimize against). Defaults to `Discrete(k_angles=64)`.
        branching_buckets: Sorted list of `n_sets` values every sibling-group
            optimization may be padded to (see `_pad_siblings`); each group is
            padded to the smallest bucket that covers it. Defaults to the
            sorted distinct branching factors actually present in `graph`
            (see `_branching_buckets`), so a small sibling group is never
            padded up to the tree's largest node. Pass a single-element list
            to reproduce old-style uniform padding to one fixed `n_sets`
            (e.g. for a future JIT-compile-reuse experiment, which needs a
            fixed shape across calls; under the current
            `~vizopt.jaxopt.optimize_gradient_descent`, a fresh closure
            rebuilt on every `optimize()` call, no padding scheme saves
            compilation today).
        early_stop_patience, early_stop_tol: Forwarded to every sibling-group
            fit's `~vizopt.base.OptimConfig`. Most sibling groups (especially
            small ones, now that `branching_buckets` no longer pads them up
            to the tree's largest node) converge well before `n_iters`;
            defaults on here since a recursive treemap runs many independent
            fits where this reliably saves iterations, unlike the general
            `OptimConfig` default (`None`, off) meant for a single
            hand-tuned run. Pass `early_stop_patience=None` to disable.
        grid_resolution, n_iters, learning_rate, exclusion_offset,
            containment_offset, weight_compactness, weight_containment,
            weight_target_area, weight_area, weight_perimeter,
            weight_exclusion, weight_smoothness, temperature: Forwarded to
            every sibling-group fit; see `_fit_siblings` and
            `~vizopt.templates.raster_stars.RasterStarOptimizer`.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        sizes: dict,
        *,
        root=None,
        mean_leaf_area: float = 3.0,
        fill_fraction: float = 0.85,
        representation: StarRepresentation | None = None,
        branching_buckets: list[int] | None = None,
        grid_resolution: int = 48,
        n_iters: int = 1200,
        learning_rate: float = 0.01,
        exclusion_offset: float = 0.1,
        containment_offset: float = 0.1,
        weight_compactness: float = 5.0,
        weight_containment: float = 30.0,
        weight_target_area: float = 20.0,
        weight_area: float = 0.3,
        weight_perimeter: float = 0.3,
        weight_exclusion: float = 10.0,
        weight_smoothness: float = 1.0,
        temperature: float = 0.08,
        early_stop_patience: int | None = 300,
        early_stop_tol: float = 1e-3,
    ):
        self.graph = graph
        self.sizes = sizes
        self.root = (
            root
            if root is not None
            else next(n for n in graph.nodes if graph.in_degree(n) == 0)
        )
        self.mean_leaf_area = mean_leaf_area
        self.fill_fraction = fill_fraction
        self.representation = (
            representation if representation is not None else Discrete()
        )
        self.branching_buckets = branching_buckets
        self.grid_resolution = grid_resolution
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.exclusion_offset = exclusion_offset
        self.containment_offset = containment_offset
        self.weight_compactness = weight_compactness
        self.weight_containment = weight_containment
        self.weight_target_area = weight_target_area
        self.weight_area = weight_area
        self.weight_perimeter = weight_perimeter
        self.weight_exclusion = weight_exclusion
        self.weight_smoothness = weight_smoothness
        self.temperature = temperature
        self.early_stop_patience = early_stop_patience
        self.early_stop_tol = early_stop_tol

    def optimize(self) -> dict:
        """Run the recursive layout.

        Returns:
            Dict mapping every non-root, positive-weight node to its star
            result dict (`"center"`, `"radii"`, `"angles"`). Also stored as
            `self.result_`.
        """
        graph, sizes, root = self.graph, self.sizes, self.root

        buckets = self.branching_buckets
        if buckets is None:
            buckets = _branching_buckets(graph, sizes)

        root_children = [c for c in graph.successors(root) if sizes.get(c, 0) > 0]
        root_area = self.mean_leaf_area * len(root_children)

        fit_kwargs = dict(
            representation=self.representation,
            grid_resolution=self.grid_resolution,
            n_iters=self.n_iters,
            learning_rate=self.learning_rate,
            exclusion_offset=self.exclusion_offset,
            containment_offset=self.containment_offset,
            weight_compactness=self.weight_compactness,
            weight_containment=self.weight_containment,
            weight_target_area=self.weight_target_area,
            weight_area=self.weight_area,
            weight_perimeter=self.weight_perimeter,
            weight_exclusion=self.weight_exclusion,
            weight_smoothness=self.weight_smoothness,
            temperature=self.temperature,
            early_stop_patience=self.early_stop_patience,
            early_stop_tol=self.early_stop_tol,
        )

        results: dict = {}

        def _recurse(node, node_area, parent_center, parent_radii):
            children = [c for c in graph.successors(node) if sizes.get(c, 0) > 0]
            if not children:
                return
            budget = (
                node_area if parent_center is None else node_area * self.fill_fraction
            )
            raw = np.array([sizes[c] for c in children], dtype=np.float64)
            target_areas = (raw / raw.sum() * budget).tolist()

            side = float(np.sqrt(budget))
            cx, cy = (0.0, 0.0) if parent_center is None else parent_center
            rect = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
            child_rects = squarify_layout([(c, sizes[c]) for c in children], rect)
            initial_centers = np.array(
                [
                    [
                        (child_rects[c][0] + child_rects[c][2]) / 2,
                        (child_rects[c][1] + child_rects[c][3]) / 2,
                    ]
                    for c in children
                ],
                dtype=np.float32,
            )

            pad_to = _bucket_size(len(children), buckets)
            padded_names, padded_targets, padded_centers = _pad_siblings(
                children, target_areas, initial_centers, pad_to
            )
            fitted = _fit_siblings(
                padded_names,
                padded_targets,
                padded_centers,
                parent_center=parent_center,
                parent_radii=parent_radii,
                **fit_kwargs,
            )

            for c in children:
                results[c] = fitted[c]
                if graph.out_degree(c) > 0:
                    area = star_polygon_area(np.asarray(fitted[c]["radii"]))
                    _recurse(c, area, fitted[c]["center"], fitted[c]["radii"])

        _recurse(root, root_area, None, None)
        self.result_ = results
        return results

    @property
    def node_shapes_(self) -> dict:
        """Star result dicts for every non-root, positive-weight node.

        Raises:
            ValueError: If `optimize` has not been called yet.
        """
        if not hasattr(self, "result_"):
            raise ValueError("No result yet — call optimize() first.")
        return self.result_

    def plot(self, *, ax=None, label_fn=None):
        """Plot every domain: directories as dashed outlines, leaves filled.

        Args:
            ax: Axes to draw on. A new figure and axes are created if None.
            label_fn: Callable mapping a node to its display label. Defaults
                to `node.name` when present (e.g. `pathlib.Path`), else `str(node)`.

        Returns:
            The axes the treemap was drawn on.

        Raises:
            ValueError: If `optimize` has not been called yet.
        """
        results = self.node_shapes_
        if label_fn is None:
            label_fn = lambda n: getattr(n, "name", None) or str(n)  # noqa: E731

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))

        depths = nx.shortest_path_length(self.graph, self.root)
        dir_nodes = sorted(
            (n for n in results if self.graph.out_degree(n) > 0),
            key=lambda n: depths[n],
        )
        leaf_nodes = [n for n in results if self.graph.out_degree(n) == 0]

        for n in dir_nodes:
            res = results[n]
            cx, cy = res["center"]
            r, angs = res["radii"], res["angles"]
            bx = np.append(cx + r * np.cos(angs), cx + r[0] * np.cos(angs[0]))
            by = np.append(cy + r * np.sin(angs), cy + r[0] * np.sin(angs[0]))
            ax.plot(bx, by, lw=1.5, ls="--", color="black", alpha=0.5)
            ax.text(
                cx,
                cy + radius_at_angle(r, np.pi / 2),
                label_fn(n),
                ha="center",
                va="bottom",
                fontsize=7,
                style="italic",
                color="dimgray",
            )

        for n in leaf_nodes:
            res = results[n]
            cx, cy = res["center"]
            r, angs = res["radii"], res["angles"]
            bx = np.append(cx + r * np.cos(angs), cx + r[0] * np.cos(angs[0]))
            by = np.append(cy + r * np.sin(angs), cy + r[0] * np.sin(angs[0]))
            ax.fill(bx, by, alpha=0.4)
            ax.plot(bx, by, lw=1.0)
            ax.text(
                cx,
                cy,
                label_fn(n),
                ha="center",
                va="center",
                fontsize=6,
                fontweight="bold",
            )

        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.margins(0.05)
        ax.axis("off")
        return ax
