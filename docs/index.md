# vizopt

**vizopt** is a mathematical optimization library for data visualization. It provides a general framework for defining and solving layout optimization problems — such as bubble layouts and label placement — using [JAX](https://jax.readthedocs.io/) for automatic differentiation and JIT compilation.

## Features

- **General optimization framework** — define multi-objective loss functions from composable terms
- **Gradient descent via Adam** — efficient JAX-based optimization with JIT compilation
- **Bubble layout** — circular node layouts with hierarchical inclusion constraints
- **NetworkX integration** — works directly with NetworkX graphs
- **Steppable sessions** — pin or move variables, change weights and reheat between steps, for interactive use
- **Pydantic validation** — optional input validation for problem templates

## Examples

See the [examples gallery](examples/index.md) for worked examples.

## Quick Example

```python
import numpy as np
from vizopt.base import OptimConfig
from vizopt.templates.circle_packing import CirclePackingOptimizer

radii = np.random.default_rng(0).uniform(0.1, 1.0, size=20).tolist()

optimizer = CirclePackingOptimizer(radii, weight_total_size=10.0, collision_offset=0.05)
optimizer.optimize(OptimConfig(n_iters=3000, learning_rate=0.01))
optimizer.plot()
positions = optimizer.positions_  # list of (x, y) tuples
```

## Installation

```bash
pip install vizopt
```

Requires Python 3.13+.
