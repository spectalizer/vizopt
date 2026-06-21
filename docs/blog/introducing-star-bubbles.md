# Introducing Stars-and-Bubbles Euler diagrams

Smoother better visualization of overlapping sets using mathematical optimization.

TODO Add a nice animation somewhere near the beginning of the article.



## What are Euler diagrams?

### Example gallery
EU orgs
Academic disciplines
Natural languages
Programming languages
Consonants

The mind naturally groups things.
Bears are mammals. Mammals are animals. Bears are also terrestrial animals, which are also animals. Whales are mammals but not terrestral animals.
Some programming languages are object-oriented, some are statically types, and many are both.
Be it in linguistics or biology, you will find sets, sets containing other sets, and sets intersecting in different ways.

Euler diagrams are the visual representation of these *containment* and *intersection* relations.

*Interesting fact 1: Euler diagrams are not Venn diagrams*
No, we are not talking about Venn diagrams here. Venn diagrams represent every possible set intersection, including empty intersections.

The actual definition: *closed shapes...*
Venn diagrams represent sets as closed shapes, whereby the shape representing a subset of a given set is contained within the shape representing the superset.

Two kinds of Euler diagrams: showing individual elements (which is possible for small discrete sets) or not.

Requirements for Euler diagrams: closed **connex** shapes... 

*Interesting fact 3: Euler diagrams are a superset of treemaps*

With some imagination, you can see that treemaps are just tightly packed Euler diagrams for specific sets of sets.
If you took the animal examples and considered only phylogenetic relations, you should end up with a strictly hierarchical set of sets, which would be equivalent to a tree (with the elements as leaves). Representing parent-child relationships in trees geometrically is the basic of treemaps. 

## Searching for Better Primitives
(the search for the right primitives/parameterization)

Now *closed shape* is a wonderfully general description, applying to circles and rectangles as well as more complex polygonal and curved regions of the plane, but this generality is not really helpful when it comes to defining an algorithm that should draw actual Euler diagrams.

We want to look for a nice *family* of shapes.

### Euler diagrams with circles

A typical instantiation of Euler diagrams uses circles, but circles are rather rigid, and they waste space.

Consider the simplest case: two equally-sized circles of radius *r*, enclosed in the smallest circle that contains both. Pack them side by side and the enclosing circle has radius *2r*. Its area is *4πr²*, while the two inner circles together cover only *2πr²*: half the enclosing region is empty (more with some space between enclosed and enclosing circles), belonging to neither subset. With multiple levels of nesting this compounds: each layer inflates the container, but the added space is mostly dead area that carries no information. A tight boundary would hug the contents and waste nothing. 

### Rectangles

Rectangles pack more efficiently and power treemaps well, but they are as rigid as circles... but they cannot represent every set configuration, and some are mathematically impossible to express with axis-aligned regions (see [Appendix: Rectangles](#appendix-rectangles)).

![Three levels of binary nesting: leaf circles cover only 12.5% of the enclosing area.](img/circle_nesting.svg)

*Interesting fact 3: Some set configurations cannot be represented with a nice Euler diagram*


## Introducing radially convex sets

A region is *radially convex* (or *star-shaped*) if there is a center point from which every point on the boundary is directly visible — no part of the boundary hides behind another. Imagine standing at the center of a room: if you can see every wall from that single spot, the room is star-shaped. Every convex shape qualifies, but so do many non-convex ones: a crescent does not, a star polygon does.

This is precisely the condition studied in the art gallery problem, which asks how many guards are needed to watch every point in a room. For a star-shaped room, one guard standing at the center always suffices.

![Three examples: a convex blob, a non-convex star-shaped region, and a C-shape that is not star-shaped.](img/radially_convex_sets.svg)

Star-shaped regions admit a compact parameterization: fix a center, then specify one radius per direction. Sample K angles uniformly in [0, 2π), and the boundary becomes a vector of K radii — a fixed-size, continuous representation that gradient descent can move through freely. Circles are the special case where all K radii are equal; every other shape in the family is reachable by letting them differ.

## Smooth optimization of smooth shapes

... Because the boundary is a smooth function of its K radii, every desideratum we care about — enclosure, compactness, non-overlap — can be written as a differentiable loss term and the whole thing handed to an optimizer. (This is also why rectangles lose: their intersection area is a piecewise-linear function with gradient discontinuities at every edge crossing, making the landscape rough. Star polygons stay smooth throughout.)

So here is the idea: let us represent set elements with circles and sets as *star-shaped* radially convex sets and find the best possible arrangement of these circles and radially convex sets accordingl.

Setting up the optimization...

The optimizer jointly adjusts two things: the shape of each boundary (its center and K radii) and the positions of the circles themselves. The total loss is a weighted sum of terms in three groups.

**Hard constraints.** 

* *Enclosure* checks that every member circle is inside its boundary: for each ray angle, it computes the minimum radius that would just graze the circle at that angle, and penalizes any shortfall.

* *Exclusion* is the mirror image: for each non-member circle, it penalizes any radius that reaches into it. Both carry high weights.

* *Circle collision* prevents circles from overlapping each other as they move.

*How does the check work?* For each angle θ, project the vector from the set center to the circle center onto the ray: the along-ray component is *tang* and the perpendicular offset is *perp*. The circle's shadow along that ray spans from *tang* − √(r² − perp²) (near edge) to *tang* + √(r² − perp²) (far edge). Enclosure requires the boundary radius to reach the far edge; exclusion requires it to stop before the near edge. Violations are penalized quadratically.

For exclusion this is equivalent to checking that the boundary point lies outside the circle (distance d ≥ r) — a boundary point outside the circle is exactly one that falls short of the near edge. For enclosure the threshold is the far edge, not the circle center, so the analogous check d ≤ r would be wrong: a boundary point can be inside the circle while only reaching the near side.

![Enclosure and exclusion: the circle's shadow along the ray and the two critical radii.](img/enclosure_geometry.svg)

This works cheaply because the moving elements are circles. A star-vs-star intersection — needed if boundaries could also be arbitrary shapes — has no closed form: it requires comparing two full polygons, an operation roughly K times heavier per pair. Keeping the circles as circles and only the boundaries as stars is thus justified by runtime efficiency. Besides, it visually differentiates elements from sets.

**Aesthetic objectives.** pushing toward compact shapes.

* *Area* penalizes fat blobs

* *Perimeter* penalizes elongated or wiggly outlines. 

(Their interplay is easiest to see with circles held fixed: try turning the area weight up and watch the boundaries shrink to hug their members, or turn the perimeter weight up and watch them round off. In the movable variant these interact with the anchor term, and the balance between them is one of the more satisfying knobs to tune.)

**Regularization.**

* *Min-radius* keeps boundaries from collapsing to a point.

* *Smoothness* penalizes squared differences between adjacent radii, discouraging jagged outlines.

* *Position anchor* penalizes circles for drifting far from their initial positions, which may (e.g. in the case geographic entities) or may not carry meaning.


Implementation:
All terms are implemented in JAX, allowing for automatic differentiation and JIT-compilation. We run a gradient-based optimizer (e.g. Adam) for a few thousand iterations.

Advanced topic 1: Representation
(Splines, Fourier, simple)

Advanced topic 2: Curriculum learning
(Curriculum etc.)




TODO How can you do it at home? Using the `vizopt` package

*Interesting fact 4: Even where Euler diagrams are possible, there may be a better way to represent overlapping sets*
(matrix etc.)

### Conclusions


I have often advocated for the use of mathematical optimization in data visualization, but this is probably the best use case for it I have ever encountered.
This is expensive but it is worth it.

TODO Wax lyrical about the foam.




### References and related work

[Max Fürbringer](https://en.wikipedia.org/wiki/Max_F%C3%BCrbringer)'s beautiful and inspiring tree of birds in cross section.

*Interesting fact 5: Overlapping sets are the same as hypergraphs*


## Appendix: Rectangles {#appendix-rectangles}

*To be written.*

Rectangles are a natural candidate for set boundaries — axis-aligned, easy to render, and the basis of treemaps. Two arguments against them:

**Expressiveness.** There exist set configurations (intersections and containments) that cannot be realized by any arrangement of axis-aligned rectangles. This section demonstrates such a configuration and proves it is impossible to represent exactly.

**Optimization landscape.** Rectangle-based objectives tend to be less smooth than their star-polygon counterparts — intersection area between two rectangles is a piecewise-linear function of position, with kinks wherever an edge crosses another edge. This roughness makes gradient-based optimization harder and convergence less reliable. *To be elaborated.* 

---

## TODOs

- **Animation near the top** — add a compelling animation early in the article to hook the reader before the theory sections.
- **"How to use it at home" section** — write the practical walkthrough showing how to use the `vizopt` package to reproduce results.
- **Conclusion prose** — expand the conclusion; wax lyrical about the foam metaphor.
- **Appendix: Rectangles** — write the impossibility proof and the optimization landscape argument (currently placeholder stubs).