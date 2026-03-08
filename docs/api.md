# API Reference

## vizopt.base

Core abstractions for the optimization framework.

---

### `ObjectiveTerm`

```python
@dataclass
class ObjectiveTerm:
    name: str
    compute: Callable[[OptimVars, Any], Array]
    multiplier: float = 1.0
```

A named, weighted term in a composite loss function.

| Attribute | Type | Description |
|---|---|---|
| `name` | `str` | Identifier used in history dicts and weight overrides |
| `compute` | `Callable` | `compute(optim_vars, input_parameters) -> scalar` |
| `multiplier` | `float` | Weight; set to `0.0` to disable this term |

---

### `build_objective`

```python
def build_objective(
    terms: list[ObjectiveTerm],
    input_parameters: Any,
) -> Callable[[OptimVars], Array]
```

Combines a list of `ObjectiveTerm`s into a single `fun(optim_vars) -> scalar` suitable for gradient descent. Terms with `multiplier=0.0` are skipped.

---

### `OptimizationProblemTemplate`

```python
@dataclass
class OptimizationProblemTemplate:
    terms: list[ObjectiveTerm]
    initialize: Callable[[InputParams], OptimVars]
    input_params_class: type | None = None
    plot_configuration: Callable | None = None
```

A reusable template for a class of optimization problems.

#### `.instantiate(input_parameters, weight_overrides=None)`

Creates a runnable `OptimizationProblem`.

| Parameter | Type | Description |
|---|---|---|
| `input_parameters` | `Any` | Fixed data for this problem instance |
| `weight_overrides` | `dict[str, float] \| None` | Override term multipliers by name |

**Raises:** `KeyError` if a name in `weight_overrides` does not match any term. `pydantic.ValidationError` if `input_params_class` is set and validation fails.

---

### `OptimizationProblem`

```python
@dataclass
class OptimizationProblem:
    input_parameters: InputParams
    terms: list[ObjectiveTerm]
    initialize: Callable[[InputParams], OptimVars]
    plot_configuration: Callable | None = None
```

A concrete runnable optimization problem.

#### `.optimize(n_iters=1000, learning_rate=0.001, callback=None, track_every=10)`

Runs Adam gradient descent to minimize the objective.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_iters` | `int` | `1000` | Number of iterations |
| `learning_rate` | `float` | `0.001` | Adam step size |
| `callback` | `Callable \| None` | `None` | Called each iteration: `(i, loss, optim_vars, grads)` |
| `track_every` | `int` | `10` | Record history every N iterations |

**Returns:** `(optim_vars, history)` where `history` is a list of dicts with keys `"iteration"`, `"total"`, and one key per term name.

---

## vizopt.bubblejax

Specialized bubble layout optimizer built on top of the general framework.

---

### `optimize_circular_layout_with_enclosed_nodes`

```python
def optimize_circular_layout_with_enclosed_nodes(
    inclusion_tree: nx.DiGraph,
    weight_total_size: float = 2.0,
    optim_kwargs: dict | None = None,
) -> tuple[dict, dict]
```

Optimize positions and radii for a circular node layout with inclusion constraints. No graph edges — inclusion relationships only.

**Inclusion tree convention:** edge `(u, v)` means node `v` is contained inside node `u`. Leaf nodes must have a `size` node attribute (fixed radius). Non-leaf nodes will have their radii optimized.

**Loss terms:**

| Term | Description |
|---|---|
| `total_size` | Penalizes large overall layout extent |
| `collision` | Penalizes overlapping non-included nodes |
| `non_inclusion` | Penalizes nodes not fitting inside their parent |

**Returns:** `(pos, radii)` where `pos` maps node name → `(x, y)` and `radii` maps non-leaf node name → optimized radius.

---

### `optimize_circular_layout_with_enclosed_and_linked_nodes`

```python
def optimize_circular_layout_with_enclosed_and_linked_nodes(
    graph: nx.Graph,
    inclusion_tree: nx.DiGraph,
    weight_edge_length: float = 1.0,
    weight_total_size: float = 2.0,
    optim_kwargs: dict | None = None,
) -> tuple[dict, dict]
```

Optimize positions and radii for a circular node layout with both graph edges and inclusion constraints.

Graph nodes must have a `size` attribute (fixed radius). Nodes that appear in `inclusion_tree` but not in `graph` are treated as enclosing nodes with optimizable radii.

**Loss terms:**

| Term | Description |
|---|---|
| `edge_length` | Penalizes long graph edges |
| `total_size` | Penalizes large overall layout extent |
| `collision` | Penalizes overlapping non-included nodes |
| `non_inclusion` | Penalizes nodes not fitting inside their parent |

**Returns:** `(pos, radii)` where `pos` maps node name → `(x, y)` and `radii` maps enclosing node name → optimized radius.

---

## vizopt.animation

Optimization progress visualization.

---

### `SnapshotCallback`

```python
class SnapshotCallback:
    snapshots: list
```

Callback that saves copies of `optim_vars` at regular intervals.

```python
callback = SnapshotCallback(every=50)
problem.optimize(n_iters=1000, callback=callback)
# callback.snapshots contains numpy copies of optim_vars
```

---

### `animate`

```python
def animate(snapshots, problem) -> FuncAnimation
```

Renders each snapshot via `problem.plot_configuration` and returns a `matplotlib.animation.FuncAnimation`.

---

## vizopt.components

Reusable JAX loss components.

---

### `multiple_bbox_intersections`

```python
def multiple_bbox_intersections(bboxes_a, bboxes_b) -> Array
```

Vectorized pairwise bounding-box intersection areas. Input shape `(n, 2, 2)`, returns `(n, m)` matrix of intersection areas.
