# vizopt

Mathematical optimization for data visualization, specifically designed for graph layouts with hierarchical inclusion constraints ("bubble layouts").

Uses JAX for automatic differentiation and JIT compilation to efficiently optimize layouts via gradient descent.

Read the documentation [https://spectalizer.github.io/vizopt/](https://spectalizer.github.io/vizopt/).

## Installation

```bash
pip install vizopt
```

To use Optuna-based schedule search (e.g. the `star_curriculum` notebook):

```bash
uv sync --group hyperoptim
```

## Quick Start

```python
import numpy as np
from vizopt.templates import circle_packing

# Define circle radii
rng = np.random.default_rng(0)
radii = rng.uniform(0.1, 1.0, size=20).tolist()

# Pack circles to minimize overlap and bounding box size
positions = circle_packing.optimize_circle_packing(
    radii=radii,
    weight_total_size=10.0,
    collision_offset=0.05,
    optim_kwargs={"n_iters": 3000, "learning_rate": 0.01},
)
# positions is a list of (x, y) tuples, one per circle
```

## Features

- Multi-objective optimization (edge lengths, compactness, collision avoidance, inclusion constraints)
- Efficient JAX-based gradient descent with JIT compilation
- Handles arbitrary hierarchical inclusion relationships
- Automatic per-variable normalization so optimizer performance is independent of input coordinate scale
- NetworkX integration with a consistent DiGraph convention: **parent → child edges** (`(u, v)` means `v ⊂ u`)

## Examples

See [examples/examples_with_bubbles.ipynb](examples/examples_with_bubbles.ipynb) for detailed usage.

## License

MIT

## For developers

### Quality assurance

Tests run automatically on every push and pull request via GitHub Actions:

```bash
uv run pytest
```

Type-check all notebooks locally (not in CI):

```bash
uv run python scripts/convert_all_notebooks.py
```

This converts each notebook to a temporary `.py` file, runs `pyright` across all of them, then deletes the generated files. Pass `--no-cleanup` to keep them for inspection.

### Documentation

Using Zensical.

`uv run zensical serve`

`uv run python scripts/nb_to_md.py --execute examples/circle_packing.ipynb docs/examples/from-notebook-circle-packing.md`