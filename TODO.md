## Next

* refactor all templates to use VizOptimizer

* clean clean clean

* tests including the notebooks

* visuals in example gallery

* a bit of noise to separate overlapping points at first (or more generally some "annealing")

* adjust readme

* LogSumExp etc. for smoother collision losses: how generally?

* tests, test coverage etc.

* random initialization more systematic

* scaling, relative coordinates etc.: we now have some scaling mechanism but this is not widely used

* An introspection module: visualizing vizopt
    * layered graph diagram of class inheritance
    * treemap etc.

* Remove all double backticks (my new pet peeve) from all docstrings

* Raster-based Euler is the nicest but slowest
How can we make it faster? Initialization from circle-based? Hyperparameter optimization? Other ideas?

* New templates
    * tree layout optimization

* treemap module should be in a folder with other non-optimization heuristics useful for initialization and comparison

* VizOptimizers non-jaxopt (MILP etc.)


## In the long run

* Template to D3js

* Template to D3js, with user allowed to drag objects and interact with the optimization procedure...
