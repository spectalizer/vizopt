# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**vizopt** is a mathematical optimization library for data visualization, specifically designed for optimizing graph layouts with hierarchical inclusion constraints ("bubble layouts"). It uses JAX for automatic differentiation and JIT compilation to efficiently optimize layouts via gradient descent.

## Development Commands

This project uses `uv` as the package manager and build system.

```bash
# Install dependencies
uv sync

# Format code with black
uv run black .

# Run tests (when tests are added)
uv run pytest

# Run Jupyter notebook for examples
uv run jupyter notebook examples/examples_with_bubbles.ipynb
```

## Architecture

### Core Components

The codebase consists of two primary modules that work together:

1. **[jaxopt.py](src/vizopt/jaxopt.py)** - Generic gradient descent optimizer
   - `optimize_gradient_descent()`: Wraps Optax's Adam optimizer with JAX JIT compilation
   - Accepts arbitrary loss functions and parameters as dictionaries
   - Provides callback mechanism for monitoring optimization progress

2. **[bubblejax.py](src/vizopt/bubblejax.py)** - Specialized bubble layout optimizer
   - `optimize_bubble_layout()`: Main entry point for optimizing graph layouts with inclusion constraints
   - Uses the generic optimizer from jaxopt.py
   - Pre-processes NetworkX graphs into JAX-compatible numpy arrays for vectorized operations

### Key Architectural Concepts

#### Inclusion Trees
The central concept is the **inclusion tree**: a directed graph where an edge `(u, v)` means node `u` is contained within node `v`. For example, in the examples notebook, "Munich" is in "Germany", "Vienna" is in "Austria", etc.

The optimizer handles two separate graphs:
- **graph**: The main NetworkX Graph with nodes to lay out and edges to draw
- **inclusion_tree**: A NetworkX DiGraph defining hierarchical containment relationships

#### Multi-Objective Optimization

The layout optimization minimizes a weighted sum of four competing objectives:

1. **Edge lengths** - Shorter edges improve readability
2. **Total width** - Compact layouts are preferred
3. **Collision penalty** - Nodes that shouldn't overlap must stay separated
4. **Non-inclusion penalty** - Nodes must stay inside their container nodes

The collision detection is smart: nodes are allowed to overlap if one contains the other (based on the inclusion tree). This is pre-computed into a `collision_pairs` array.

#### JAX-Specific Design Patterns

The code uses several JAX-specific patterns:

- **Pre-processing**: All graph data structures (NetworkX objects) are converted to numpy arrays before optimization to avoid Python loops in JAX-traced functions
- **Vectorization**: Operations like collision detection and inclusion constraints are fully vectorized using array indexing rather than loops
- **JIT compilation**: The main loss function `function_to_minimize()` is JIT-compiled for performance
- **Parameter dictionaries**: Optimization parameters are passed as nested dictionaries (e.g., `{"node_xys": ..., "enclosing_node_radii": ...}`)

#### Node Radii Management

Nodes come in two types:
- **Fixed-radius nodes**: Nodes from the main graph with pre-defined sizes
- **Enclosing nodes**: Nodes that only appear in the inclusion tree (not the main graph) - their radii are optimization parameters

The `get_node_radii()` function concatenates both types into a single array for use in loss calculations.

### Data Flow

1. Input: NetworkX Graph + NetworkX DiGraph (inclusion tree)
2. Pre-process: Convert to numpy arrays (edge indices, collision pairs, etc.)
3. Initialize: Random positions, initial enclosing radii
4. Optimize: Gradient descent on multi-objective loss function
5. Output: Optimized positions dictionary + enclosing radii dictionary + loss history plot

## Python Environment

- Requires Python 3.13+
- Primary dependencies: JAX, Optax, NetworkX, matplotlib, pandas
- Dev dependencies: black (formatting), pytest (testing), ipykernel (notebooks)

## Style guide

- Google-style docstrings