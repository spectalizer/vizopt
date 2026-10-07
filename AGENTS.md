# AGENTS.md

Guidance for AI agents working in this repository.

## Project Overview

**vizopt** is a mathematical optimization library for data visualization. It provides a general framework for defining and solving layout optimization problems (e.g., star-shaped set boundaries, label placement) using JAX for automatic differentiation and JIT compilation.

## Development Commands

This project uses `uv` as the package manager and build system.

```bash
# Install dependencies
uv sync

# Format code with black
uv run black .

# Run tests (also run in CI on every push / PR)
uv run pytest

# Run Jupyter notebooks: curated examples (rendered into the docs) and experiments
uv run jupyter notebook notebooks/examples/layered_graph_layouts.ipynb
uv run jupyter notebook notebooks/experiments/examples_with_bubbles.ipynb
```

Optional dependency groups: `uv sync --group milp` (PuLP/HiGHS, for `milp_euler_rectangles.py`) and `uv sync --group hyperoptim` (Optuna, for schedule search notebooks). The `server` extra (FastAPI, uvicorn) powers `vizopt.server` and is included in the `dev` group.

Frontend (Node 24+, in `frontend/`):

```bash
cd frontend
npm install
npm run build      # type-check + bundle into src/vizopt/server/static/ (served by vizopt.server)
npm run dev        # Vite dev server with hot reload; proxies /ws to a running serve(...) on port 8765
npm run codegen    # after changing scene.py or server/protocol.py: re-export the schema, regenerate TS types
```

## Architecture

### Core Components

1. **[base.py](src/vizopt/base.py)** - Core abstractions for the optimization framework
   - `ObjectiveTerm`: A named, weighted term in a composite loss function (name, compute, multiplier)
   - `build_objective()`: Combines a list of `ObjectiveTerm`s into a single `fun(optim_vars, step, weights=None) -> scalar`
   - `OptimizationProblemTemplate`: A reusable template for a class of problems — holds terms, an `initialize` function, optional Pydantic `input_params_class` for validation, and optional `plot_configuration`
   - `OptimizationProblem`: A concrete runnable instance created via `template.instantiate(input_parameters)`; exposes `.session()` (a steppable run) and `.optimize()` (a batch run on top of a session) which returns an `OptimizationResult` (fields: `optim_vars`, `history`, `final_loss`)
   - `OptimConfig`: Optimizer settings (iterations, learning rate and decay, Adam betas, restarts, seed, history tracking, early stopping)
   - `VizOptimizer`: ABC for user-facing template classes; subclasses implement `_build_problem()`, the base provides `optimize()`, `session()`, `plot()`, `animate()`, `animate_svg()` (see Template Module Structure)

2. **[session.py](src/vizopt/session.py)** - Steppable, steerable gradient descent
   - `OptimizationSession`: Owns one run's Adam state; `step(n)`, `vars` (physical space), `pin`/`unpin`/`set_value` (hold or move variable entries), `set_weight`, `reheat` (restart the cosine learning-rate decay), `record()` (per-term history record)
   - Everything that may change between steps (weights, learning rate, pin masks/values) is a `controls` pytree passed as an argument to the jitted step, so steering never recompiles. The jitted step is cached on the `OptimizationProblem` per `(b1, b2)`, so restarts and new sessions reuse it
   - Pinning is a projection after the Adam update (gradients are not masked), so Adam's moments keep tracking the force on a pinned entry and it does not jump when released
   - Foundation for interactive frontends (drag an element while the optimization keeps running)

2b. **[scene.py](src/vizopt/scene.py)** - Declarative, JSON-serializable scene descriptions for non-Python frontends
   - Pydantic primitives `Circle`, `Line`, `Polygon`, `Text` (discriminated by `kind`, each with a frame-stable `id`, a `Style`, and an optional `DragBinding(var, index)`) inside a `Scene`
   - Positions in data coordinates; sizes in data units unless the field ends in `_px` / says `"px"`
   - A `DragBinding` means the element's anchor point is `optim_vars[var][index]`, so a frontend turns a drag into `session.pin(var, index, value=[x, y])` with no template-specific code
   - Templates opt in via `scene_configuration(optim_vars, input_parameters) -> Scene`; exposed as `problem.scene()` and `session.scene()`. Implemented by `LayeredGraphOptimizer`, `TreeLayoutOptimizer`, `CirclePackingOptimizer` and `EulerDiagram` (star regions as translucent `Polygon`s, circles and floating set labels draggable)
   - `node_link_scene()` builds a draggable node-link diagram (edges, pixel-sized nodes, labels) from positions and edge indices; the layered graph and tree layout scenes are thin wrappers around it
   - `tests/test_scene.py` checks every template's scene generically (unique ids, JSON round trip, every `DragBinding` sitting exactly at its variable entry) — add new templates to its `_TEMPLATES` table
   - Its JSON Schema is exported as part of the live-server protocol (see `server/`); re-run `npm run codegen` in `frontend/` after changing the models

3. **[components/](src/vizopt/components/)** - Reusable JAX loss components and shape representations
   - [common.py](src/vizopt/components/common.py): generic penalties — `multiple_bbox_intersections()` (vectorized pairwise bbox intersection areas, `(n, 2, 2)` inputs → `(n, m)` matrix), `calculate_collision_penalty()`, width penalties, `should_be_positive_activation()`
   - [stars.py](src/vizopt/components/stars.py): star-shaped (radially convex) domains — the `StarRepresentation` ABC with `Discrete`, `Fourier` and `BSpline` parametrizations, plus the shared private loss terms (enclosure, exclusion, area, perimeter, smoothness, convexity, labels, …) and SVG helpers used by the star templates
   - [bspline_stars.py](src/vizopt/components/bspline_stars.py): B-spline boundary math behind the `BSpline` representation, and B-spline raster soft-membership for `RasterStarOptimizer`
   - [bands.py](src/vizopt/components/bands.py): convex vertical-band domains (see Convex Band Sets)

4. **[templates/](src/vizopt/templates/)** - User-facing `VizOptimizer` subclasses, one problem family per module
   - [circle_packing.py](src/vizopt/templates/circle_packing.py) `CirclePackingOptimizer`, [label_positions.py](src/vizopt/templates/label_positions.py) `LabelPositionOptimizer`, [layered_graph.py](src/vizopt/templates/layered_graph.py) `LayeredGraphOptimizer`, [color.py](src/vizopt/templates/color.py) `ColorPaletteOptimizer` (OKLab palettes)
   - [nested_circles.py](src/vizopt/templates/nested_circles.py) `NestedCirclesOptimizer` / `LinkedNestedCirclesOptimizer`: circle-based Euler diagrams with inclusion constraints
   - [euler/](src/vizopt/templates/euler/): star-shaped Euler diagrams around circles (`stars_vs_circles.py`, `EulerDiagram`) or rectangles (`stars_vs_rectangles.py`, `EulerDiagramRect`)
   - [star_vs_star.py](src/vizopt/templates/star_vs_star.py) `StarDomainOptimizer` / `StarVsStarOptimizer`, [band_vs_band.py](src/vizopt/templates/band_vs_band.py) `BandDomainOptimizer`, [raster_stars.py](src/vizopt/templates/raster_stars.py) `RasterStarOptimizer`: region layout without underlying elements
   - [trees/](src/vizopt/templates/trees/): node-link `TreeLayoutOptimizer` and the recursive star-domain treemap `RasterTreemapOptimizer` (not a `VizOptimizer`: it runs one fit per sibling group)

5. **[animation.py](src/vizopt/animation.py)** - Optimization progress visualization
   - `SnapshotCallback`: Callback that saves numpy copies of `optim_vars` at regular intervals into `.snapshots`
   - `animate()`: Renders each snapshot via `problem.plot_configuration` and returns a `FuncAnimation`
   - `snapshots_to_animated_svg()` / `smil_animate()`: animated SVGs from the template's `svg_configuration`; `chronophotograph()`: overlay of snapshots in one figure

6. **[schedules.py](src/vizopt/schedules.py)** - Loss term weight scheduling
   - `warmup()` / `cooldown()`: JAX-compatible schedule factories that ramp a term's weight up or down over a fraction of the run
   - `make_term_schedules()`: Builds a `TermSchedules` from a flat parameter dict, for the `term_schedules` argument of `EulerDiagram` / `EulerDiagramRect`

7. **[server/](src/vizopt/server/)** - Live, interactive optimization in the browser (`server` extra)
   - `serve(optimizer, optim_config)`: starts a session and serves it with uvicorn at `http://127.0.0.1:8765`, including the built frontend
   - [live.py](src/vizopt/server/live.py) `LiveSession`: synchronous state machine (`validate`, `apply`, `tick`, `frame`) turning protocol messages into session steering, plus a worker thread that steps at a target fps and publishes latest-only frames. Each interaction reheats; after `settle_iters` iterations the run settles (idles) like d3-force
   - [protocol.py](src/vizopt/server/protocol.py): Pydantic WebSocket messages, discriminated by `type` — server sends `hello` (terms, weights, and the run's loss history so far, so late joiners see the whole curve), then `frame`s (scene, metrics, pinned entries, weights) and `error`s; clients send `drag_start`/`drag`/`drag_end`, `unpin`, `pause`/`resume`, `reheat`, `set_weight`, `reset`
   - [app.py](src/vizopt/server/app.py) `create_app()`: FastAPI app with the `/ws` endpoint (one sender task per client, so frames and errors never interleave) and the static frontend at `/`
   - The protocol JSON Schema (`scripts/export_protocol_schema.py` → `frontend/src/protocol/protocol.schema.json`) is committed; `tests/test_server.py` fails when it is out of date with the models

8. **[frontend/](frontend/)** - The browser app: Vite + TypeScript + D3 (d3-selection/zoom/drag/scale), no framework
   - `src/protocol/protocol.ts` is generated from the schema (json-schema-to-typescript) — never edit it by hand
   - `src/view.ts` `SceneView`: renders any `Scene` generically (keyed data join on element ids, data → screen scales with fit, zoom and pan; `_px` sizes stay fixed on zoom), and turns drags on elements with a `DragBinding` into data-space positions
   - `src/main.ts`: toolbar, terms panel (per-term value, log-scale weight slider sending `set_weight`, toggle to plot the term), WebSocket wiring (`src/connection.ts` reconnects with backoff); drag moves and slider changes are coalesced to one message per animation frame
   - `src/chart.ts` `LossChart`: loss over iterations on a log y-axis with crosshair tooltip — the total plus up to 3 selected terms; series colors are slots 1-3 of the dataviz reference palette (CSS `--series-*`, validated for CVD/normal vision against the sidebar surface in light and dark mode)
   - The built bundle (`src/vizopt/server/static/`) is gitignored but included in wheels, so run `npm run build` before a local `uv build`; the `publish.yml` workflow does this before building releases
   - `[tool.uv.build-backend] source-exclude` keeps `.mypy_cache` / `__pycache__` out of sdists and wheels (uv_build ignores `.gitignore`)

9. **Other modules**
   - [treemap.py](src/vizopt/treemap.py): classic squarified treemap layout (non-optimization baseline)
   - [milp_euler_rectangles.py](src/vizopt/milp_euler_rectangles.py): MILP-based Euler diagram with rectangular sets (needs the `milp` group)
   - [introspection.py](src/vizopt/introspection.py): visualizes this project's own structure (file treemaps, import/class graphs)
   - [examples/sets.py](src/vizopt/examples/sets.py): example set-hierarchy graphs used by notebooks and tests

### Key Architectural Concepts

#### General Optimization Workflow

The framework separates *problem definition* from *problem instantiation*:

1. Define `ObjectiveTerm`s (loss components with names, compute functions, and multipliers)
2. Create an `OptimizationProblemTemplate` with those terms, an `initialize` function, optional Pydantic class for input validation, and optional rendering hooks (`plot_configuration` for matplotlib, `svg_configuration` for animated SVG, `scene_configuration` for interactive frontends)
3. Call `template.instantiate(input_parameters)` → `OptimizationProblem`
4. Call `problem.optimize(optim_config, callback)` → `OptimizationResult`, or `problem.session(optim_config)` → `OptimizationSession` to step and steer the run yourself

`OptimizationResult` has fields `optim_vars`, `history`, and `final_loss`. `history` is a list of dicts with keys `"iteration"`, `"total"`, and one key per term name (weighted values), recorded every `track_every` iterations.

#### Input Parameters and Validation

Input parameters are plain dicts (JAX-compatible pytrees) passed unchanged to loss functions. If `input_params_class` is set on a template, `model_validate` is called at instantiation time for type/shape checking (Pydantic), but the dict itself flows through unmodified.

#### JAX-Specific Design Patterns

- **Pre-processing**: All non-JAX data (e.g., NetworkX graphs) is converted to numpy arrays before optimization to avoid Python loops in JAX-traced functions
- **Vectorization**: Loss components use fully vectorized array operations rather than loops
- **JIT compilation**: The composite loss function built by `build_objective()` is JIT-compiled into the Adam step of `session.make_step_function()`; values that change between steps (weights, pins, learning rate) are passed as traced arguments, never closed over
- **Parameter dictionaries**: `optim_vars` are plain dicts (e.g., `{"rectangle_positions": ...}`)

#### Variable Normalization

High-level functions can pass a `var_scales` dict to `OptimizationProblemTemplate.instantiate()` to normalize optimization variables. The optimizer then works in a scaled space while all loss terms and callbacks always receive physical-space values.

**Mechanism** (`base.py` and `session.py`):
- `build_objective()` wraps the loss: `physical_vars[k] = optim_vars[k] * var_scales[k]` before calling any term
- `OptimizationSession` divides the initial variables by their scales and keeps its optimizer state in scaled space; its `vars` property and all its steering methods (`pin`, `set_value`) work in physical space
- `optimize()` passes `session.vars` to history recording and to the user callback — so `SnapshotCallback` and the `optim_vars_panel` of animated SVGs always see physical values. Only the `grads` passed to callbacks are in scaled space

**Convention**: scale values may be scalars or arrays. Arrays allow per-axis scaling (e.g. `[scale_x, scale_y]` for 2D position variables, which broadcast over `(N, 2)` arrays). Keys absent from `var_scales` are left unscaled.

**In `templates/euler/stars_vs_circles.py`**: scales are computed from the input circles and applied per variable group:
- `"centers"`, `"circle_positions"` and (with labels) `"label_positions"`: `[max(std_x, mean_r), max(std_y, mean_r)]` — the `max` guards against the degenerate single-circle case
- radii-like variables (`"radii"`, `"fourier_coeffs"`, `"bspline_ctrl"`): `mean(initial_radii)` — detected by iterating `init_vars.keys()` and treating every non-`"centers"` key as radii-scale, so all three representations are handled without naming them explicitly

#### Star-Shaped Sets (components/stars.py, templates/euler/stars_vs_circles.py)

`EulerDiagram` implements circle-set boundary optimization on top of the general framework:

- Input: N circles (cx, cy, r) and S subsets (or a graph via `from_graph()`); each subset gets its own star-shaped boundary
- Circle positions are optimization variables alongside the boundaries (kept near their inputs by a position-anchor term)
- Multi-objective loss: enclosure, exclusion (no overlap with non-members), area, perimeter, smoothness, and optional terms (convexity, circle collision, set attraction, bounding box, floating set labels)
- Boundaries are a center plus a `StarRepresentation`: `Discrete` (K radii at uniformly-spaced angles), `Fourier` coefficients or `BSpline` control points — all evaluated to radii on the same angle grid, so loss terms are representation-agnostic
- `EulerDiagramRect` (`stars_vs_rectangles.py`) is the same for axis-aligned rectangles; `StarDomainOptimizer` (`star_vs_star.py`) drops the underlying elements entirely

#### Convex Band Sets (components/bands.py, templates/band_vs_band.py)

Complementary to the star-shaped representation: a convex planar region is exactly the area between a concave upper boundary y2(x) and a convex lower boundary y1(x) over x in [x_min, x_max], so containment/overlap checks reduce to pointwise comparisons instead of general polygon clipping — at the cost of only being able to represent convex shapes.

- `BandDomainOptimizer` (`templates/band_vs_band.py`) is the convex-only analog of `StarDomainOptimizer` (`templates/star_vs_star.py`): pure region layout, no underlying circles, with enclosure/exclusion masks and optional per-set target areas
- Each boundary is parametrized as `x_bounds` (`[x_min, x_max]`, itself an optimization variable) plus K `(upper, lower)` pairs at uniformly-spaced columns — the `Discrete` representation in `components/bands.py` (the `BandRepresentation` ABC mirrors `StarRepresentation` for future smooth variants)
- Multi-objective loss: target/plain area, perimeter, smoothness, convexity (optional), min-width/min-thickness guards, and band-vs-band enclosure/exclusion
- Enclosure/exclusion evaluate a point against a target band by interpolating its `upper`/`lower` arrays at the point's x (linear interpolation over the K columns, clamped outside `[x_min, x_max]`), then comparing y — the band analog of the star representation's angle-interpolated radius comparison

#### Template Module Structure

A template module lives under `src/vizopt/templates/` and exposes one or more `VizOptimizer` subclasses. The constructor accepts domain inputs (data arrays, graphs) and hyperparameters (weights, representation choices) as named arguments and stores them as instance attributes — no computation happens yet. `_build_problem()` is the single method subclasses must implement: it converts all non-JAX inputs to numpy arrays, builds the list of `ObjectiveTerm`s, constructs an `OptimizationProblemTemplate`, and returns the result of `.instantiate(input_parameters, var_scales=...)`. Result properties (named with a trailing underscore, e.g. `sets_`, `circles_`, `positions_`) extract meaningful domain outputs from `self.result_.optim_vars` and raise `ValueError` if called before `optimize()`. When the problem is naturally specified by a graph, a `from_graph()` classmethod provides an ergonomic entry point that derives circles/rectangles and set membership from the graph topology and then delegates to `__init__`. Private helper functions for loss terms and plot configuration live in the same file or a companion file in `components/`; nothing from these helpers is re-exported at the package level.

### Data Flow

1. Define terms and template (problem class definition)
2. Call `template.instantiate(input_parameters)` — validates inputs, creates `OptimizationProblem`
3. Call `problem.optimize()` — creates a session per restart (initializes vars, JIT-compiles the step once per problem), steps it with Adam, records history
4. Optionally use `SnapshotCallback` + `animate()` for animated visualization

## Python Environment

- Requires Python 3.13+
- Primary dependencies: JAX, Optax, NetworkX, matplotlib, pandas, pydantic; `server` extra: FastAPI, uvicorn
- Dev dependencies: black (formatting), ruff (linting), pyright (type checking), pytest + pytest-cov + httpx2 (testing, incl. FastAPI's TestClient), ipykernel / nbconvert / nbformat (notebooks), zensical + mkdocstrings-python (docs), the `server` extra

## Documentation

The docs site is built with [Zensical](https://zensical.org) (configured in `zensical.toml`) and the API reference is auto-generated from docstrings via [mkdocstrings](https://mkdocstrings.github.io).

```bash
# Install docs dependencies
uv add mkdocstrings-python

# Serve docs locally
uv run zensical serve

# Build static site
uv run zensical build
```

- Docs source lives in `docs/`; `docs/api.md` uses `::: vizopt.module.Symbol` directives — do not write API docs by hand there
- All public functions and classes must have Google-style docstrings so mkdocstrings can render them

## Graph Conventions

When a NetworkX `DiGraph` encodes a set hierarchy or containment relationship, the project-wide convention is:

- **Edge direction: parent → child** — an edge `(u, v)` means `v` is a member of (contained in) `u`.
- **Leaves** (`out_degree == 0`): terminal elements with no children.
- **Internal nodes** (`out_degree > 0`): composite sets or containers.
- **Membership**: a leaf belongs to a set if it is reachable from the set via `nx.descendants`.

## Style guide

- Google-style docstrings, no double backticks, no repetition of type hints in docstrings, no `:meth:`, `:class:` etc.
- No imports inside functions unless genuinely needed (e.g. lazy imports for optional/heavy deps like `matplotlib`)

## General guidelines

### Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.


### Add unit tests when you add functionality

Unit tests in the `tests` folder, using pytest.


### Prioritize long-term API quality over short-term convenience.

This project is in early development. Breaking changes are acceptable — refactor callers when needed rather than compromising the API to preserve backwards compatibility. Named return objects (e.g. `OptimizationResult`) are preferred over positional tuples even when they require migrating many call sites.

### Keep core docs up-to-date

Always make sure that project documentation, specifically AGENTS.md and README.md, remains up-to-date with code changes.
Proactively surface outdated documentation.