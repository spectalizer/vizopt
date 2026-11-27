"""Launch an example optimization live in the browser.

Usage:
    uv run python scripts/example_interactive_launch.py                  # layered graph
    uv run python scripts/example_interactive_launch.py british-isles    # Euler diagram
    uv run python scripts/example_interactive_launch.py --list
"""

import argparse

import networkx as nx

from vizopt.base import OptimConfig, VizOptimizer
from vizopt.examples.sets import make_british_islands_graph
from vizopt.server import serve
from vizopt.templates.euler.stars_vs_circles import EulerDiagram
from vizopt.templates.layered_graph import LayeredGraphOptimizer


def layered_graph() -> tuple[VizOptimizer, OptimConfig]:
    """A small DAG laid out left to right; drag nodes to rearrange it."""
    dag = nx.DiGraph(
        [
            ("A", "B"),
            ("A", "C"),
            ("B", "D"),
            ("B", "E"),
            ("C", "E"),
            ("C", "F"),
            ("D", "G"),
            ("E", "G"),
            ("F", "G"),
        ]
    )
    return (
        LayeredGraphOptimizer(dag, min_distance=1.5, weight_edge_direction=0.0),
        OptimConfig(n_iters=2000, learning_rate=3e-3),
    )


def british_isles() -> tuple[VizOptimizer, OptimConfig]:
    """British Isles territories as circles inside star-shaped nested sets."""
    # Weights from the British Islands section of notebooks/examples/star_and_circle.ipynb.
    diagram = EulerDiagram.from_graph(
        make_british_islands_graph(include_british_isles=True),
        weight_area=1.0,
        weight_perimeter=2.0,
        weight_exclusion=20.0,
        weight_smoothness=3.0,
        weight_position_anchor=3.0,
        weight_circle_collision=100.0,
        weight_set_attraction=1.0,
        circle_collision_alpha=1.0,
    )
    return diagram, OptimConfig(n_iters=3000, learning_rate=2e-3)


EXAMPLES = {
    "layered-graph": layered_graph,
    "british-isles": british_isles,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch an example optimization live in the browser."
    )
    parser.add_argument(
        "example",
        nargs="?",
        default="layered-graph",
        choices=list(EXAMPLES),
        help="which example to run (default: layered-graph)",
    )
    parser.add_argument("--list", action="store_true", help="list examples and exit")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open a browser tab"
    )
    args = parser.parse_args()

    if args.list:
        for name, build in EXAMPLES.items():
            print(f"{name:15} {build.__doc__}")
        return

    optimizer, optim_config = EXAMPLES[args.example]()
    serve(optimizer, optim_config, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
