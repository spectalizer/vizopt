# vizopt

Mathematical optimization for data visualization, specifically designed for graph layouts with hierarchical inclusion constraints ("bubble layouts").

Uses JAX for automatic differentiation and JIT compilation to efficiently optimize layouts via gradient descent.

## Installation

```bash
pip install vizopt
```

## Quick Start

```python
import networkx as nx
from vizopt.bubblejax import optimize_bubble_layout

# Create a graph
graph = nx.Graph()
graph.add_edges_from([("Munich", "Vienna"), ("Vienna", "Prague")])

# Define inclusion tree (cities in countries)
inclusion_tree = nx.DiGraph()
inclusion_tree.add_edges_from([
    ("Munich", "Germany"),
    ("Vienna", "Austria"),
    ("Prague", "Czechia"),
])

# Optimize layout
result = optimize_bubble_layout(
    graph=graph,
    inclusion_tree=inclusion_tree,
    node_radii={"Munich": 0.3, "Vienna": 0.3, "Prague": 0.3},
)
```

## Features

- Multi-objective optimization (edge lengths, compactness, collision avoidance, inclusion constraints)
- Efficient JAX-based gradient descent with JIT compilation
- Handles arbitrary hierarchical inclusion relationships
- NetworkX integration

## Examples

See [examples/examples_with_bubbles.ipynb](examples/examples_with_bubbles.ipynb) for detailed usage.

## License

MIT

## For developers

### Documentation

Using Zensical.

`uv run zensical serve`

`uv run python scripts/nb_to_md.py --execute examples/circle_packing.ipynb docs/examples/from-notebook-circle-packing.md`