# vizopt

**vizopt** is a mathematical optimization library for data visualization. It provides a general framework for defining and solving layout optimization problems — such as bubble layouts and label placement — using [JAX](https://jax.readthedocs.io/) for automatic differentiation and JIT compilation.

## Features

- **General optimization framework** — define multi-objective loss functions from composable terms
- **Gradient descent via Adam** — efficient JAX-based optimization with JIT compilation
- **Bubble layout** — circular node layouts with hierarchical inclusion constraints
- **NetworkX integration** — works directly with NetworkX graphs
- **Pydantic validation** — optional input validation for problem templates

## Examples

See the [examples gallery](examples/index.md) for worked examples.

## Quick Example



## Installation

```bash
pip install vizopt
```

Requires Python 3.13+.
