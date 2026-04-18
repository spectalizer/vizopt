import networkx as nx
import numpy as np


def graph_to_optimizer_inputs(G):
    """Derive vizopt optimizer inputs from a set-hierarchy DiGraph.

    Args:
        G: A ``networkx.DiGraph`` whose nodes carry ``target_area`` (float or
            None), ``center`` ([x, y]), and ``color`` (hex string) attributes,
            and whose edges point from child to parent (child ⊂ parent).

    Returns:
        Tuple of ``(set_names, idx, enclosures, target_areas, initial_centers)``
        in the formats expected by ``optimize_star_domains_raster`` and similar.
    """
    set_names = list(G.nodes)
    idx = {name: i for i, name in enumerate(set_names)}
    enclosures = [(idx[u], idx[v]) for u, v in G.edges]
    target_areas = [G.nodes[n].get("target_area") for n in set_names]
    initial_centers = np.array([G.nodes[n]["center"] for n in set_names], dtype=np.float32)
    return set_names, idx, enclosures, target_areas, initial_centers


def get_leaf_circles(G):
    """Extract non-enclosing nodes (in-degree 0) as circles.

    Circle radius is derived as ``sqrt(target_area / π)``, the radius of a disc
    with the given area.

    Args:
        G: A DiGraph from ``make_british_islands_graph``.

    Returns:
        Tuple of ``(names, circles, name_to_idx)`` where ``circles`` is a list
        of ``(cx, cy, r)`` tuples and ``name_to_idx`` maps name to integer index.
    """
    names = [n for n in G.nodes if G.in_degree(n) == 0]
    circles = [(*G.nodes[n]["center"], np.sqrt(G.nodes[n]["target_area"] / np.pi)) for n in names]
    name_to_idx = {name: i for i, name in enumerate(names)}
    return names, circles, name_to_idx


def get_enclosing_sets(G, leaf_names):
    """Return enclosing (in-degree > 0) nodes and their leaf memberships.

    Set names are returned in topological order (innermost first).

    Args:
        G: The DiGraph.
        leaf_names: Ordered list of leaf territory names (from ``get_leaf_circles``).

    Returns:
        Tuple of ``(set_names, sets_idx)`` where ``set_names`` is the ordered list
        of enclosing node names and ``sets_idx`` is a list of lists of indices into
        ``leaf_names``.
    """
    set_names = [n for n in nx.topological_sort(G) if G.in_degree(n) > 0]
    leaf_set = set(leaf_names)
    name_to_idx = {name: i for i, name in enumerate(leaf_names)}
    sets_idx = [
        sorted(name_to_idx[n] for n in nx.ancestors(G, sname) if n in leaf_set)
        for sname in set_names
    ]
    return set_names, sets_idx


def make_british_islands_graph(include_ireland_island: bool = True) -> nx.DiGraph:
    """Build the British Isles set-hierarchy as a DiGraph.

    Nodes carry ``target_area``, ``center``, and ``color`` attributes for use
    with ``graph_to_optimizer_inputs`` and ``get_leaf_circles``. Aggregate set
    nodes have ``target_area=None``; leaf territory nodes have a numeric value.

    Args:
        include_ireland_island: When True, adds "Ireland island" as the union
            of Northern Ireland and the Republic of Ireland. When False, the
            Republic of Ireland sits directly inside "British Islands".

    Returns:
        A ``networkx.DiGraph`` ready for ``graph_to_optimizer_inputs``.
    """
    G = nx.DiGraph()

    G.add_node("England",             target_area=3.40, center=[ 2.0,  0.0], color="#4472c4")
    G.add_node("Scotland",            target_area=2.00, center=[ 1.5,  4.5], color="#70ad47")
    G.add_node("Wales",               target_area=0.55, center=[ 0.0,  1.0], color="#ff0000")
    G.add_node("Northern Ireland",    target_area=0.40, center=[-2.5,  3.5], color="#ffc000")
    G.add_node("Republic of Ireland", target_area=1.80, center=[-2.5,  0.5], color="#169b62")
    G.add_node("Crown Dependencies",  target_area=None, center=[ 1.5, -1.5], color="#7030a0")
    G.add_node("Great Britain",       target_area=None, center=[ 1.5,  2.0], color="#264478")
    G.add_node("United Kingdom",      target_area=None, center=[ 0.5,  2.5], color="#1f3864")
    G.add_node("Ireland island",      target_area=None, center=[-2.5,  2.0], color="#375623")
    G.add_node("British Islands",     target_area=None, center=[ 0.5,  1.5], color="#808080")
    G.add_node("Isle of Man",         target_area=0.07, center=[-0.8,  2.5], color="#c05780")
    G.add_node("Jersey",              target_area=0.04, center=[ 3.0, -3.5], color="#e8a838")
    G.add_node("Guernsey",            target_area=0.03, center=[ 1.2, -3.8], color="#2e9bba")

    # Edges: child → parent  (child ⊂ parent)
    G.add_edge("England",             "Great Britain")
    G.add_edge("Scotland",            "Great Britain")
    G.add_edge("Wales",               "Great Britain")
    G.add_edge("Great Britain",       "United Kingdom")
    G.add_edge("Northern Ireland",    "United Kingdom")
    G.add_edge("United Kingdom",      "British Islands")
    G.add_edge("Crown Dependencies",  "British Islands")
    G.add_edge("Isle of Man",         "Crown Dependencies")
    G.add_edge("Jersey",              "Crown Dependencies")
    G.add_edge("Guernsey",            "Crown Dependencies")

    if include_ireland_island:
        G.add_edge("Northern Ireland",    "Ireland island")
        G.add_edge("Republic of Ireland", "Ireland island")
        G.add_edge("Ireland island",      "British Islands")
    else:
        G.remove_node("Ireland island")
        G.add_edge("Republic of Ireland", "British Islands")

    return G
