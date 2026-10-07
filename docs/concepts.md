# Concepts

## Overview

vizopt has two layers:

**User-facing** — `VizOptimizer` subclasses like `EulerDiagram`. They expose a sklearn-style API: store hyperparameters in `__init__`, run with `.optimize()`, access fitted state via trailing-underscore attributes.

**Framework** — `OptimizationProblemTemplate` / `OptimizationProblem`. The lower-level building blocks that `VizOptimizer` assembles internally. Use these directly when building a new optimizer.

```
VizOptimizer subclass (e.g. EulerDiagram)
       ↓ ._build_problem()
OptimizationProblemTemplate   ←  ObjectiveTerm(s) + initialize function
       ↓ .instantiate(input_parameters)
OptimizationProblem
       ↓ .optimize()                     ↓ .session()
OptimizationResult                 OptimizationSession
(optim_vars, history, final_loss)  (step, pin, set_weight, reheat, …)
```

## VizOptimizer

`VizOptimizer` is the base class for all user-facing visualization optimizers. Subclasses implement `_build_problem()` to turn stored hyperparameters into a configured `OptimizationProblem`. The base class handles the rest.

```python
diagram = EulerDiagram(circles, sets, weight_enclosure=20.0)
# or
diagram = EulerDiagram.from_graph(inclusion_graph)

result = diagram.optimize(OptimConfig(n_iters=1000))

diagram.sets_     # list of per-set dicts with center, radii, angles
diagram.circles_  # (N, 3) array of optimized [cx, cy, r]
diagram.plot()    # inherited from VizOptimizer
```

Fitted state lives in `diagram.problem_` and `diagram.result_` after `optimize()` is called. Domain-specific outputs (like `sets_` and `circles_`) are properties that read from `result_.optim_vars`.

### Implementing a new VizOptimizer

```python
from vizopt.base import OptimizationProblem, OptimizationProblemTemplate, VizOptimizer

class MyLayout(VizOptimizer):
    def __init__(self, data, *, weight_x=1.0):
        self.data = data
        self.weight_x = weight_x

    def _build_problem(self) -> OptimizationProblem:
        # build terms, initialize, input_parameters ...
        return OptimizationProblemTemplate(
            terms=[...],
            initialize=...,
        ).instantiate(input_parameters)

    @property
    def result_positions(self):
        return self.result_.optim_vars["positions"]
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
    initialize=lambda input_params, seed: {"x": jnp.zeros(10)},
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
result = problem.optimize(
    OptimConfig(n_iters=1000, learning_rate=0.001),
)
```

- `optim_vars` — the optimized variables (a plain dict / JAX pytree)
- `history` — list of dicts with keys `"iteration"`, `"total"`, and one entry per term name (see [Optimization History](#optimization-history))
- `final_loss` — loss of the best run at its last iteration

With `OptimConfig(n_restarts=k)`, the problem is solved `k` times with seeds `seed, seed + 1, …` and the best run is returned.

## OptimizationSession

`optimize()` is a batch loop on top of a lower-level, steppable *session*. Use a session directly when something outside the loop should decide when to step and change the state in between, e.g. an interactive frontend where the user drags an element while the optimization keeps running:

```python
session = problem.session(OptimConfig(learning_rate=0.01))
session.step(200)

session.pin("node_xys", 3, value=[1.0, 2.0])   # hold node 3 at (1, 2); others re-flow
session.set_weight("collision", 5.0)           # change a term weight live
session.reheat()                                # restart the learning-rate decay
session.step(200)

session.unpin("node_xys", 3)
session.vars["node_xys"]                        # current values, in physical space
session.record()                                # per-term values, like a history entry
```

Weights, pins and the learning rate are passed to the jitted step as arguments, so none of these calls trigger a recompilation. `VizOptimizer.session()` builds the problem and returns a session in one call.

### Scenes and the live server

For rendering outside of Python, a template can describe a configuration as a `vizopt.scene.Scene`: a JSON-serializable list of circles, lines, polygons and text in data coordinates. Elements that can be dragged carry a `DragBinding(var, index)`, telling a frontend which variable entry to pin when the user drags them:

```python
scene = session.scene()
payload = scene.to_json_dict()   # send to the browser
# on drag of an element with element.drag = DragBinding(var="node_xys", index=3):
session.pin("node_xys", 3, value=[x, y])
```

`vizopt.server.serve(optimizer, optim_config)` (requires `pip install "vizopt[server]"`) does exactly this for you: it runs the session in a background thread, streams scenes and loss values over a WebSocket at about 30 frames per second, and serves a browser app that renders them and sends drags, pause, reheat, weight and reset commands back. Each interaction reheats the learning rate; after `optim_config.n_iters` quiet iterations the run settles and stops using CPU until the next interaction.

## JAX Design Patterns

**Pre-processing outside JAX**: Convert Python/NetworkX data to numpy arrays *before* building the loss function. JAX traces through array operations, not Python loops.

**optim_vars are plain dicts**: This makes them JAX-compatible pytrees that Optax can differentiate through. Example: `{"node_xys": array, "variable_node_radii": array}`.

**JIT compilation**: `build_objective()` produces a function that gets JIT-compiled by the optimizer, once per problem — avoid Python-level branching inside `compute` functions.

## Loss Function Composition

`build_objective(terms, input_parameters)` combines terms into a single scalar loss:

```
loss(optim_vars, step) = Σ term.multiplier × term.schedule(step) × term.compute(optim_vars, input_parameters)
```

`schedule` is optional (constant 1 when unset); see `vizopt.schedules` for warmup/cooldown factories. Terms with `multiplier=0.0` are skipped entirely — they cannot be turned on later via `session.set_weight()`.

## Optimization History

`history` is a list of dicts recorded every `OptimConfig.track_every` iterations:

```python
[
    {"iteration": 0,   "total": 42.3, "edge_length": 10.1, "collision": 32.2},
    {"iteration": 10,  "total": 38.7, "edge_length": 9.4,  "collision": 29.3},
    ...
]
```

Each record also holds, per term, `<term>_unweighted` (raw value) and `<term>_unscheduled` (weighted as at the end of the schedule), plus `"total_unscheduled"`, which early stopping (`OptimConfig.early_stop_patience`) monitors.

Convert to a DataFrame for easy plotting:

```python
import pandas as pd
df = pd.DataFrame(result.history)
df.plot(x="iteration", y=["total", "edge_length", "collision"])
```

## Animation

Use `SnapshotCallback` and `animate()` from `vizopt.animation` to visualize the optimization process:

```python
from vizopt.animation import SnapshotCallback, animate

callback = SnapshotCallback(every=50)
result = problem.optimize(OptimConfig(n_iters=1000), callback=callback)

anim = animate(problem, callback.snapshots)
anim.save("layout.gif", writer="pillow")
```

On a `VizOptimizer`, `optimizer.animate(callback)` and `optimizer.animate_svg(callback)` (animated SVG, no matplotlib rendering per frame) do the same after `optimizer.optimize(..., callback=callback)`.
