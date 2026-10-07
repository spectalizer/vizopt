import networkx as nx

from vizopt.base import OptimConfig
from vizopt.server import serve
from vizopt.templates.layered_graph import LayeredGraphOptimizer

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
serve(
    LayeredGraphOptimizer(dag, min_distance=1.5, weight_edge_direction=0.0),
    OptimConfig(n_iters=2000, learning_rate=3e-3),
)
