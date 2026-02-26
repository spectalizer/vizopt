# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**vizopt** is a mathematical optimization library for data visualization. It provides a general framework for defining and solving layout optimization problems (e.g., bubble layouts, label placement) using JAX for automatic differentiation and JIT compilation.

## Development Commands

This project uses `uv` as the package manager and build system.

```bash
# Install dependencies
uv sync

# Format code with black
uv run black .

# Run tests (when tests are added)
uv run pytest

# Run Jupyter notebooks for examples
uv run jupyter notebook examples/optimize_label_positions.ipynb
uv run jupyter notebook examples/examples_with_bubbles.ipynb
```

## Architecture

### Core Components

1. **[base.py](src/vizopt/base.py)** - Core abstractions for the optimization framework
   - `ObjectiveTerm`: A named, weighted term in a composite loss function (name, compute, multiplier)
   - `build_objective()`: Combines a list of `ObjectiveTerm`s into a single `fun(optim_vars) -> scalar`
   - `OptimizationProblemTemplate`: A reusable template for a class of problems — holds terms, an `initialize` function, optional Pydantic `input_params_class` for validation, and optional `plot_configuration`
   - `OptimizationProblem`: A concrete runnable instance created via `template.instantiate(input_parameters)`; exposes `.optimize()` which returns `(optim_vars, history)`

2. **[jaxopt.py](src/vizopt/jaxopt.py)** - Generic gradient descent optimizer
   - `optimize_gradient_descent()`: Wraps Optax's Adam optimizer with JAX JIT compilation
   - Low-level entry point; normally called indirectly via `OptimizationProblem.optimize()`

3. **[components.py](src/vizopt/components.py)** - Reusable JAX loss components
   - `multiple_bbox_intersections()`: Vectorized pairwise bounding-box intersection areas; shape `(n, 2, 2)` inputs, returns `(n, m)` matrix

4. **[animation.py](src/vizopt/animation.py)** - Optimization progress visualization
   - `SnapshotCallback`: Callback that saves numpy copies of `optim_vars` at regular intervals into `.snapshots`
   - `animate()`: Renders each snapshot via `problem.plot_configuration` and returns a `FuncAnimation`

5. **[bubblejax.py](src/vizopt/bubblejax.py)** - Specialized bubble layout optimizer
   - `optimize_bubble_layout()`: Main entry point for optimizing graph layouts with inclusion constraints
   - Pre-processes NetworkX graphs into JAX-compatible numpy arrays for vectorized operations

### Key Architectural Concepts

#### General Optimization Workflow

The framework separates *problem definition* from *problem instantiation*:

1. Define `ObjectiveTerm`s (loss components with names, compute functions, and multipliers)
2. Create an `OptimizationProblemTemplate` with those terms, an `initialize` function, optional Pydantic class for input validation, and optional `plot_configuration`
3. Call `template.instantiate(input_parameters)` → `OptimizationProblem`
4. Call `problem.optimize(n_iters, learning_rate, callback, track_every)` → `(optim_vars, history)`

`history` is a list of dicts with keys `"iteration"`, `"total"`, and one key per term name (weighted values), recorded every `track_every` iterations.

#### Input Parameters and Validation

Input parameters are plain dicts (JAX-compatible pytrees) passed unchanged to loss functions. If `input_params_class` is set on a template, `model_validate` is called at instantiation time for type/shape checking (Pydantic), but the dict itself flows through unmodified.

#### JAX-Specific Design Patterns

- **Pre-processing**: All non-JAX data (e.g., NetworkX graphs) is converted to numpy arrays before optimization to avoid Python loops in JAX-traced functions
- **Vectorization**: Loss components use fully vectorized array operations rather than loops
- **JIT compilation**: The composite loss function built by `build_objective()` is JIT-compiled via `jaxopt.optimize_gradient_descent()`
- **Parameter dictionaries**: `optim_vars` are plain dicts (e.g., `{"rectangle_positions": ...}`)

#### Bubble Layout (bubblejax.py)

`bubblejax.py` implements the bubble layout use-case on top of the general framework:

- Handles two NetworkX graphs: **graph** (nodes/edges to lay out) and **inclusion_tree** (DiGraph where `(u, v)` means `u` is inside `v`)
- Multi-objective loss: edge lengths, total width, collision penalty (pre-computed `collision_pairs`), non-inclusion penalty
- Two node types: fixed-radius nodes (from graph) and enclosing nodes (radii are optimization variables)

### Data Flow

1. Define terms and template (problem class definition)
2. Call `template.instantiate(input_parameters)` — validates inputs, creates `OptimizationProblem`
3. Call `problem.optimize()` — initializes vars, JIT-compiles loss, runs Adam, records history
4. Optionally use `SnapshotCallback` + `animate()` for animated visualization

## Python Environment

- Requires Python 3.13+
- Primary dependencies: JAX, Optax, NetworkX, matplotlib, pandas, pydantic
- Dev dependencies: black (formatting), pytest (testing), ipykernel (notebooks)

## Style guide

- Google-style docstrings