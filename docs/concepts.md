# Concepts

## Overview

vizopt separates *problem definition* from *problem instantiation*, following a template pattern:

```
ObjectiveTerm(s)
       ↓
OptimizationProblemTemplate   ←  initialize function
       ↓ .instantiate(input_parameters)
OptimizationProblem
       ↓ .optimize()
(optim_vars, history)
```

## ObjectiveTerm

An `ObjectiveTerm` is a named, weighted component of the loss function:

```python
from vizopt.base import ObjectiveTerm

term = ObjectiveTerm(
    name="edge_length",
    compute=lambda optim_vars, input_params: ...,  # returns a JAX scalar
    multiplier=1.0,
)
```

- `compute(optim_vars, input_parameters)` — called during optimization; must be JAX-traceable
- `multiplier` — weight for this term; set to `0.0` to disable it entirely

## OptimizationProblemTemplate

A template defines a *class* of problems — the loss terms and how to initialize variables — independently of any specific data:

```python
from vizopt.base import OptimizationProblemTemplate

template = OptimizationProblemTemplate(
    terms=[term_a, term_b],
    initialize=lambda input_params: {"x": jnp.zeros(10)},
    input_params_class=MyPydanticModel,   # optional, for validation
    plot_configuration=my_plot_fn,        # optional
)
```

### Weight overrides

You can override term weights at instantiation time without redefining the template:

```python
problem = template.instantiate(
    input_parameters,
    weight_overrides={"edge_length": 2.0},
)
```

## OptimizationProblem

A concrete runnable instance created via `template.instantiate(input_parameters)`:

```python
optim_vars, history = problem.optimize(
    n_iters=1000,
    learning_rate=0.001,
    track_every=10,
)
```

- `optim_vars` — the optimized variables (a plain dict / JAX pytree)
- `history` — list of dicts with keys `"iteration"`, `"total"`, and one entry per term name

## JAX Design Patterns

**Pre-processing outside JAX**: Convert Python/NetworkX data to numpy arrays *before* building the loss function. JAX traces through array operations, not Python loops.

**optim_vars are plain dicts**: This makes them JAX-compatible pytrees that Optax can differentiate through. Example: `{"node_xys": array, "variable_node_radii": array}`.

**JIT compilation**: `build_objective()` produces a function that gets JIT-compiled by the optimizer — avoid Python-level branching inside `compute` functions.

## Loss Function Composition

`build_objective(terms, input_parameters)` combines terms into a single scalar loss:

```
loss(optim_vars) = Σ term.multiplier × term.compute(optim_vars, input_parameters)
```

Terms with `multiplier=0.0` are skipped entirely.

## Optimization History

`history` is a list of dicts recorded every `track_every` iterations:

```python
[
    {"iteration": 0,   "total": 42.3, "edge_length": 10.1, "collision": 32.2},
    {"iteration": 10,  "total": 38.7, "edge_length": 9.4,  "collision": 29.3},
    ...
]
```

Convert to a DataFrame for easy plotting:

```python
import pandas as pd
df = pd.DataFrame(history)
df.plot(x="iteration", y=["total", "edge_length", "collision"])
```

## Animation

Use `SnapshotCallback` and `animate()` from `vizopt.animation` to visualize the optimization process:

```python
from vizopt.animation import SnapshotCallback, animate

callback = SnapshotCallback(every=50)
optim_vars, history = problem.optimize(n_iters=1000, callback=callback)

anim = animate(callback.snapshots, problem)
anim.save("layout.gif")
```
