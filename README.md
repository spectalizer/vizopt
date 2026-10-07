# vizopt

Mathematical optimization for data visualization, specifically designed for graph layouts with hierarchical inclusion constraints ("bubble layouts").

Uses JAX for automatic differentiation and JIT compilation to efficiently optimize layouts via gradient descent.

Read the documentation [https://spectalizer.github.io/vizopt/](https://spectalizer.github.io/vizopt/).

## Installation

```bash
pip install vizopt
```

To run optimizations live in the browser (drag elements while the optimization keeps going):

```bash
pip install "vizopt[server]"
```

To use Optuna-based schedule search (e.g. the `star_curriculum` notebook):

```bash
uv sync --group hyperoptim
```

## Quick Start

```python
import numpy as np
from vizopt.base import OptimConfig
from vizopt.templates.circle_packing import CirclePackingOptimizer

# Define circle radii
rng = np.random.default_rng(0)
radii = rng.uniform(0.1, 1.0, size=20).tolist()

# Pack circles to minimize overlap and bounding box size
optimizer = CirclePackingOptimizer(radii, weight_total_size=10.0, collision_offset=0.05)
optimizer.optimize(OptimConfig(n_iters=3000, learning_rate=0.01))
positions = optimizer.positions_  # list of (x, y) tuples, one per circle
optimizer.plot()
```

For step-by-step control (e.g. dragging an element while the optimization keeps running), start a session instead of calling `optimize()`:

```python
session = optimizer.session(OptimConfig(learning_rate=0.01))
session.step(500)
session.pin("node_xys", 0, value=[0.0, 0.0])  # hold circle 0 at the origin
session.reheat()
session.step(500)
```

### Live in the browser

```python
import networkx as nx
from vizopt.base import OptimConfig
from vizopt.server import serve
from vizopt.templates.layered_graph import LayeredGraphOptimizer

dag = nx.DiGraph([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
serve(LayeredGraphOptimizer(dag, min_distance=1.5), OptimConfig(n_iters=2000, learning_rate=3e-3))
```

This opens `http://127.0.0.1:8765`, where the layout optimizes live: drag a node and the rest re-flows around it (shift-drop keeps it pinned, double-click unpins), pause, reheat or reset. Templates support this by providing a `scene_configuration`; so far the layered graph layout does.

## Features

- Multi-objective optimization (edge lengths, compactness, collision avoidance, inclusion constraints)
- Efficient JAX-based gradient descent with JIT compilation
- Steppable optimization sessions: pin or move variables, change term weights and reheat the learning rate between steps, without recompiling
- Live browser frontend (FastAPI + WebSocket server, TypeScript + D3 app) to watch and steer optimizations interactively
- Handles arbitrary hierarchical inclusion relationships
- Automatic per-variable normalization so optimizer performance is independent of input coordinate scale
- NetworkX integration with a consistent DiGraph convention: **parent → child edges** (`(u, v)` means `v ⊂ u`)

## Examples

See the [examples gallery](https://spectalizer.github.io/vizopt/examples/) in the documentation, rendered from the notebooks in [notebooks/examples/](notebooks/examples/). More exploratory work lives in [notebooks/experiments/](notebooks/experiments/).

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
uv run python scripts/convert_all_notebooks_to_py.py
```

This converts each notebook to a temporary `.py` file, runs `pyright` across all of them, then deletes the generated files. Pass `--no-cleanup` to keep them for inspection.

### Frontend

The live frontend lives in `frontend/` (Vite + TypeScript + D3). With Node 24+:

```bash
cd frontend
npm install
npm run build    # bundles into src/vizopt/server/static/, served by vizopt.server
npm run dev      # hot-reloading dev server; proxies /ws to a running serve(...) on port 8765
npm run codegen  # after changing src/vizopt/scene.py or src/vizopt/server/protocol.py
```

The built bundle is gitignored but packaged into wheels: run `npm run build` before building a release.

### Documentation

Using Zensical.

`uv run zensical serve`

Render all example notebooks into `docs/examples/`:

`uv run python scripts/convert_all_notebooks_to_md.py --execute`

Or a single one:

`uv run python scripts/nb_to_md.py --execute notebooks/examples/circle_packing.ipynb docs/examples/circle-packing.md`