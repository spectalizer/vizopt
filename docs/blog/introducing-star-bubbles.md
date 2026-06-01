

## What are Euler diagrams?

The mind naturally groups things.
Bears are mammals. Mammals are animals. Bears are also terrestrial animals, which are also animals. Whales are mammals but not terrestral animals.
Some programming languages are object-oriented, some are statically types, and many are both.
Be it in linguistics or biology, you will find sets, sets containing other sets, and sets intersecting in different ways.

Euler diagrams are the visual representation of these *containment* and *intersection* relations.

*Are we talking about Venn diagrams?*
No. Venn diagrams represent every possible set intersection, including empty intersections.

The actual definition: *closed shapes...*

(the search for the right primitives/parameterization)

## Euler diagrams with circles

The canonical representation uses circles. But circles are rigid.


There is another problem: circles waste space. Consider the simplest case — two equally-sized circles of radius r, enclosed in the smallest circle that contains both. Pack them side by side and the enclosing circle has radius 2r. Its area is 4πr², while the two inner circles together cover only 2πr²: half the enclosing region is empty (more with some space between enclosed and enclosing circles), belonging to neither subset. With multiple levels of nesting this compounds — each layer inflates the container, but the added space is dead area that carries no information. A tight boundary would hug the contents and waste nothing. Rectangles pack more efficiently and power treemaps well — but they cannot represent every set configuration, and some are mathematically impossible to express with axis-aligned regions (see [Appendix: Rectangles](#appendix-rectangles)).

![Three levels of binary nesting: leaf circles cover only 12.5% of the enclosing area.](img/circle_nesting.svg)



## Introducing radially convex sets

A region is *radially convex* (or *star-shaped*) if there is a center point from which every point on the boundary is directly visible — no part of the boundary hides behind another. Imagine standing at the center of a room: if you can see every wall from that single spot, the room is star-shaped. Every convex shape qualifies, but so do many non-convex ones: a crescent does not, a star polygon does.

This is precisely the condition studied in the art gallery problem, which asks how many guards are needed to watch every point in a room. For a star-shaped room, one guard standing at the center always suffices.

![Three examples: a convex blob, a non-convex star-shaped region, and a C-shape that is not star-shaped.](img/radially_convex_sets.svg)

Star-shaped regions admit a compact parameterization: fix a center, then specify one radius per direction. Sample K angles uniformly in [0, 2π), and the boundary becomes a vector of K radii — a fixed-size, continuous representation that gradient descent can move through freely. Circles are the special case where all K radii are equal; every other shape in the family is reachable by letting them differ.

Setting up a simple version

Playing with objective terms
Collision, inclusion, total drawing area, bubble area, perimeter, convexity, smoothness, distance to anchor, attraction and repulsion

Playing with parameterization
(Splines, Fourier, simple)

Playing with optimization
(Curriculum etc.)

### Examples
EU orgs
Academic disciplines
Natural languages
Programming languages
Consonants


### Conclusions


I have often advocated for the use of mathematical optimization in data visualization, but this is probably the best use case for it I have ever encountered.
This is expensive but it is worth it

### Related work

Max Fürbringer's tree of birds

---

## Appendix: Rectangles {#appendix-rectangles}

*To be written.*

Rectangles are a natural candidate for set boundaries — axis-aligned, easy to render, and the basis of treemaps. Two arguments against them:

**Expressiveness.** There exist set configurations (intersections and containments) that cannot be realized by any arrangement of axis-aligned rectangles. This section demonstrates such a configuration and proves it is impossible to represent exactly.

**Optimization landscape.** Rectangle-based objectives tend to be less smooth than their star-polygon counterparts — intersection area between two rectangles is a piecewise-linear function of position, with kinks wherever an edge crosses another edge. This roughness makes gradient-based optimization harder and convergence less reliable. *To be elaborated.* 