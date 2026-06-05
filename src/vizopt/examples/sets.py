import networkx as nx
import numpy as np


def graph_to_optimizer_inputs(G):
    """Derive vizopt optimizer inputs from a set-hierarchy DiGraph.

    Args:
        G: A ``networkx.DiGraph`` whose nodes carry ``target_area`` (float or
            None), ``center`` ([x, y]), and ``color`` (hex string) attributes,
            and whose edges point from parent to child (parent ⊃ child).

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
    """Extract leaf nodes (out-degree 0) as circles.

    Args:
        G: A DiGraph with parent→child edges. Leaf nodes must carry ``center``
            ([x, y]) and ``r`` (float) attributes.

    Returns:
        Tuple of ``(names, circles, name_to_idx)`` where ``circles`` is a list
        of ``(cx, cy, r)`` tuples and ``name_to_idx`` maps name to integer index.
    """
    names = [n for n in G.nodes if G.out_degree(n) == 0]
    circles = [(*G.nodes[n]["center"], G.nodes[n]["r"]) for n in names]
    name_to_idx = {name: i for i, name in enumerate(names)}
    return names, circles, name_to_idx


def get_enclosing_sets(G, leaf_names):
    """Return enclosing (out-degree > 0) nodes and their leaf memberships.

    Set names are returned in topological order (outermost first).

    Args:
        G: A DiGraph with parent→child edges.
        leaf_names: Ordered list of leaf territory names (from ``get_leaf_circles``).

    Returns:
        Tuple of ``(set_names, sets_idx)`` where ``set_names`` is the ordered list
        of enclosing node names and ``sets_idx`` is a list of lists of indices into
        ``leaf_names``.
    """
    set_names = [n for n in nx.topological_sort(G) if G.out_degree(n) > 0]
    leaf_set = set(leaf_names)
    name_to_idx = {name: i for i, name in enumerate(leaf_names)}
    sets_idx = [
        sorted(name_to_idx[n] for n in nx.descendants(G, sname) if n in leaf_set)
        for sname in set_names
    ]
    return set_names, sets_idx


def make_multiples_of_primes_graph(
    primes=(2, 3, 5), max_n=11, r=0.4, layout_r=4.0
) -> nx.DiGraph:
    """Build a set-hierarchy graph for multiples of primes among 1..max_n.

    Each unique element that belongs to at least one prime's multiples becomes
    a leaf node. Each prime becomes an internal set node whose children are its
    multiples. Leaf circles are placed on a ring for a symmetric starting layout.

    Args:
        primes: Primes to use as sets.
        max_n: Upper bound (inclusive) of the integer range to consider.
        r: Circle radius assigned to every leaf node.
        layout_r: Radius of the ring used to spread initial positions.

    Returns:
        A ``networkx.DiGraph`` with parent→child edges, ready for
        ``optimize_multiple_radially_convex_sets_with_movable_circles_from_graph``.
        Leaf nodes carry ``center`` and ``r`` attributes; set nodes carry only
        their name.
    """
    multiple_dict = {p: [k for k in range(1, max_n + 1) if k % p == 0] for p in primes}
    elements = sorted(set(n for ms in multiple_dict.values() for n in ms))

    n_elements = len(elements)
    pos_angles = np.linspace(0, 2 * np.pi, n_elements, endpoint=False)

    G: nx.DiGraph = nx.DiGraph()
    for i, elem in enumerate(elements):
        G.add_node(
            elem,
            center=[float(layout_r * np.cos(pos_angles[i])),
                    float(layout_r * np.sin(pos_angles[i]))],
            r=r,
        )
    for p in primes:
        G.add_node(f"multiples_of_{p}")
        for m in multiple_dict[p]:
            G.add_edge(f"multiples_of_{p}", m)

    return G


def make_animals_graph(r=0.5) -> nx.DiGraph:
    """Build a simple animal-taxonomy set hierarchy as a DiGraph.

    Leaves and their memberships:

    - Bears: Mammals, Terrestrial Animals
    - Whales: Mammals, Marine Animals
    - Snakes: Terrestrial Animals
    - Sharks: Marine Animals, Marine Fish

    All sets are subsets of Animals. Marine Fish is a subset of Marine Animals.

    Args:
        r: Circle radius assigned to every leaf node.

    Returns:
        A ``networkx.DiGraph`` with parent→child edges. Leaf nodes carry
        ``center`` and ``r`` attributes.
    """
    G: nx.DiGraph = nx.DiGraph()

    G.add_node("Bears",  center=[ 2.0,  1.0], r=r)
    G.add_node("Whales", center=[-2.0,  1.0], r=r)
    G.add_node("Snakes", center=[ 2.0, -1.0], r=r)
    G.add_node("Sharks", center=[-2.0, -1.0], r=r)

    G.add_node("Mammals")
    G.add_node("Terrestrial Animals")
    G.add_node("Marine Animals")
    G.add_node("Marine Fish")
    G.add_node("Animals")

    G.add_edge("Animals", "Mammals")
    G.add_edge("Animals", "Terrestrial Animals")
    G.add_edge("Animals", "Marine Animals")
    G.add_edge("Mammals", "Bears")
    G.add_edge("Mammals", "Whales")
    G.add_edge("Terrestrial Animals", "Bears")
    G.add_edge("Terrestrial Animals", "Snakes")
    G.add_edge("Marine Animals", "Whales")
    G.add_edge("Marine Animals", "Marine Fish")
    G.add_edge("Marine Fish", "Sharks")

    return G


def make_british_islands_graph(include_ireland_island: bool = True) -> nx.DiGraph:
    """Build the British Isles set-hierarchy as a DiGraph.

    Nodes carry ``target_area``, ``center``, and ``color`` attributes for use
    with ``graph_to_optimizer_inputs`` and ``get_leaf_circles``. Aggregate set
    nodes have ``target_area=None``; leaf territory nodes have a numeric value.
    Leaf nodes also carry an ``r`` attribute (``sqrt(target_area / π)``).

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

    # Edges: parent → child  (parent ⊃ child)
    G.add_edge("Great Britain",       "England")
    G.add_edge("Great Britain",       "Scotland")
    G.add_edge("Great Britain",       "Wales")
    G.add_edge("United Kingdom",      "Great Britain")
    G.add_edge("United Kingdom",      "Northern Ireland")
    G.add_edge("British Islands",     "United Kingdom")
    G.add_edge("British Islands",     "Crown Dependencies")
    G.add_edge("Crown Dependencies",  "Isle of Man")
    G.add_edge("Crown Dependencies",  "Jersey")
    G.add_edge("Crown Dependencies",  "Guernsey")

    if include_ireland_island:
        G.add_edge("Ireland island",      "Northern Ireland")
        G.add_edge("Ireland island",      "Republic of Ireland")
        G.add_edge("British Islands",     "Ireland island")
    else:
        G.remove_node("Ireland island")
        G.add_edge("British Islands",     "Republic of Ireland")

    # Add r to leaf nodes for use with stars_vs_circles.from_graph API
    for n in G.nodes:
        area = G.nodes[n].get("target_area")
        if area is not None:
            G.nodes[n]["r"] = float(np.sqrt(area / np.pi))

    return G
