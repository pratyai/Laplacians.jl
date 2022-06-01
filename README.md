# Laplacians.jl

[![codecov](https://codecov.io/gh/danspielman/Laplacians.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/danspielman/Laplacians.jl)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://danspielman.github.io/Laplacians.jl/dev)

[![Laplacians](http://pkg.julialang.org/badges/Laplacians_0.6.svg)](http://pkg.julialang.org/detail/Laplacians)

Laplacians is a package containing graph algorithms, with an emphasis on tasks related to spectral and algebraic graph theory. It contains (and will contain more) code for solving systems of linear equations in graph Laplacians, low stretch spanning trees, sparsifiation, clustering, local clustering, and optimization on graphs.

All graphs are represented by sparse adjacency matrices. This is both for speed, and because our main concerns are algebraic tasks. It does not handle dynamic graphs. It would be very slow to implement dynamic graphs this way.

The documentation may be found by clicking on one of the "docs" links above.


## Current Development Version

To get the current version of the master branch, run `pkg> add Laplacians#master`

# Version 1.2.0

This version is compatible with Julia 1.4 and 1.5, but not earlier versions.

Changes:

* Added two graph generators: `complete_bipartite_graph`, `star_graph`.
* Added a function `line_graph` that computes the line graph of an input graph.


# Version 1.1.1

Change: minor bug fix for spectral graph drawing.

Verified compatibility with Julia 1.2.

# Version 1.1.0

Changes:

* Updating to use Julia's new Registrator.
* Added `harmonic_interp` to interpolate harmonic functions on graphs.  This is the fundamental routine used in Label Propagation / Semi-Supervised Learning on Graphs.
* Added a function `read_graph` to replace `readIJ` and `readIJV`.  It is a little more robust.
* Cleaned up `maxflow` so that it now returns a flow and cut as a matrix and set.
* Made `pcg` a little more general.
* Added `fiedler` for computing Fiedler vectors and values.  That is, the smallest nonzero eigenvalue of the Laplacian.
* Fixed a bug in `thicken` that could cause it to loop forever, and cause `chimera` to do the same.
* Changed the graph drawing code to use Plots instead of PyPlot.

# Version 1.0.1

Changes:

* Added `latin_square_graph` and `latin_square`.
* Allow `plot_graph` to plot in 3D.
* Fixed performance bug due to lazy matrix transpose.
* Changed more function names to agree with Julia naming conventions.
* Update documentation and examples.

# Version 1.0.0

This version works with Julia version 1.0.0.

# Verson 0.3.1

Changes:

* The major change in this version is to the chimera and wted_chimera graph generators.  They are now faster, and incorporate two-lifts and thickening.  The old versions, using the pseudorandom generator from Julia V0.6 and Versions 0.2 of Laplacians, may be accessed by using the flag `ver=Laplacians.V06`, as in

  ~~~julia
  a = chimera(2000, 1, ver=Laplacians.V06)
  ~~~

  There do seem to be differences in the very low order bits of graphs generated by wted_chimera with the V06 option and those generated in Julia V0.6.  Not sure why.

  The old generator is obtained by using the `RandomV06` package for Julia.

* Changed the names of many functions to bring closer to the Julia standard naming scheme.  New names are empty_graph, path_graph, ring_graph, complete_graph, generalized_ring, rand_gen_ring, product_graph, join_graphs, two_lift ...  Set deprecation warnings for the old names.

* Moved `lex.jl` to the directory `buggy`, as on further testing we found bugs in it.

* dropped wGrid3, as it produced a 4d grid so probably wasn't being used anyway.  Dropped wGrid2 also.

## Version 0.3.0, July 18 (or so), 2017

This is the first version that is compatible with Julia 0.7.  Other changes:

* Dropped support for samplingSDDM and samplingLap solvers.
* The behavior of rand in Julia 0.7 is different, and this has changed the behavior of chimera.  So, the chimera graphs generated in Version 0.3.0 and beyond will be different from those before.

## Version 0.2.2, December 28, 2017

Fixed two bugs: one in shortestPaths, and one that prevented passing some parameters to approxchol_sddm.  Improved the documentation for solving linear equations.

## Version 0.2.1, September 18, 2017

Fixed a bug in `approxchol_sddm` that caused it to be slow.

## Version 0.2.0, July 17, 2017

This version is compatible with Julia 0.6.  It will not work with
Julia 0.5.X.

Changes:

* Added `approxchol_sddm`, a wrapper of `approxchol_lap` that solves
  SDDM systems.

## Version 0.1.4, June 6, 2017

This is the current version.  It is what you retrieve when you run `Pkg.add("Laplacians")`.

Changes:

* Added `sparsify`, an implementation of sparsification by effective resistance sampling, following Spielman and Srivastava.
* Added `approxQual` and `conditionNumber` for checking how well one graph approximates another.
* Fixed a bug in the solution of Laplacian systems in disconnected graphs.

## Version 0.1.3, June 2, 2017

Major Changes:

* Changed the name of the approximate Cholesky solver from `edgeElimLap` to `approxchol_lap`.  Made improvements in this solver.
* Improved PCG so that it can now detect stagnation.  Made options to do this even better when using it with a good preconditioner, like `approxchol_lap`.
* Added in code for comparing the running times of solvers.  The difficulty here is that we need to stop them if they run too long.  Added code to do this with threads inside Julia, and with `gtimeout` when calling Matlab to use icc, CMG, or LAMG.

## Version 0.1.2, April 2, 2017


Major Changes:

* added `edgeElimLap` - a fast Laplacian solver.
* fixed a bug in the unweighted version of `akpw`.

## Version 0.1.1, December 26, 2016

Changelist:

* All of the linear equation solvers now have the same interface, and the Laplacian solvers work for disconnected graphs.
* Some support for calling solvers from Matlab has been added.
* Documentation is now through Documenter.jl.

## Version 0.0.3 / 0.1.0, November 20, 2016

Versions 0.0.3 and 0.1.0 are the same.
These versions works with Julia 0.5.

Warning: the behavior of chimera and wtedChimera differs between Julia 0.4 and Julia 0.5 because randperm acts differently in these.

## Version 0.0.2, November 19, 2016

This is the version that works with Julia 0.4.
It was captured right before the upgrade to Julia 0.5



# Acknowledgements

The development of this package has been supported in part by the
National Science Foundation Award CCF-1562041  and by the Office of Naval
Research Award N00014-16-1-2374.

