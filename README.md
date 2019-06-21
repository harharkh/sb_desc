# gromov_wasser

`gromov_wasser` is a C library to estimate the 1-Gromov--Wasserstein distance
defined by F. Memoli in [Memoli 2007][Memoli2007] and [Memoli 2011][Memoli2011]. 
The value is not exact because the calculation involves a nonconvex quadratic
optimization problem (NP-hard), and instead of solving this globally a local
optimum is found by the alternate convex search as described in, e.g.,
[Gorski 2007][Gorski2007]. This is the approach used by F. Memoli in the 
references above, and by R. Hendrikson in [Hendrikson 2016][Hendrikson2016].

The quality of the result depends on the initialization of the coupling between
the metric measure spaces. The First Lower Bound (FLB) used by F. Memoli and
R. Hendrikson is slightly more expensive to calculate than the outer product
of the measures of the finite metric measure spaces, but reduces the number of
iterations of the alternate convex search. The FLB is suggested as the default
for consistency with the literature.

The steps of the alternate convex search involve finding the 1-Wasserstein
distance for a particular cost matrix. The main contribution of this library is
to use the entropy-regularized 1-Wasserstein distance as described in
[Cuturi 2013][Cuturi2013] to accelerate the computation. This version also uses
the log-domain stabilization and \f$\eta\f$-scaling heuristic described in
[Schmitzer 2016][Schmitzer2016] and the overrelaxation scheme described in
[Thibault et. al. 2017][Thibault2017]. The greedy algorithm described in
[Altschuler et. al. 2017][Altschuler2017] slows down the calculation (perhaps
by interfering with vectorization) and the kernel truncation and multi-scale
scheme described in [Schmitzer 2016][Schmitzer2016] seem to be relevant only
for larger problems (spaces are expected to contain fewer than 32 points).

### Documentation

Inline documentation is provided. You can generate a more readable html version
by installing `doxygen` and typing the following on the command line:
```
$ make docs
```
This generates a `docs` folder and a `docs.html` linking to the main page of
the documentation in the working directory. Open `docs.html` and you should
find this README file as the main page.

### Dependencies

Apart from a c11-conformant compiler (e.g., `gcc` or `clang`), you must have
the following dependencies installed:

 * A CBLAS library (e.g. [OpenBLAS][OpenBLAS] or [ATLAS][ATLAS])
 * [LAPACK][LAPACK] (included in default builds of [OpenBLAS][OpenBLAS])

These should be available through any reasonable package manager, but could 
be installed manually if you are working in an environment without one.
Depending on the installation, you might need to modify the `LIBS` and 
`LIB_LOCATION_OPT` variables in the included `makefile`.

### Usage

This project is intended to be used as a static library by other programs. By
default, the library is installed to `/usr/local/lib` and the headers for the
API are installed to `/usr/local/include/gromov_wasser`. These directories are 
not necessarily on the default search path, so you should probably pass the 
`-I/usr/local/include/gromov_wasser` and `-L/usr/local/lib` options along with
the `-lgromov_wasser` option to your compiler. If you are not on a POSIX
compliant system or want the library and headers installed elsewhere, modify
the `LIBRARY_DIR` and `HEADERS_DIR` variables in the included `makefile`.

Once the installation directories are specified, you can compile and install
the library with the following commands:
```
$ make
$ sudo make install
```
You should get a message indicating that everything went well. You should then
probably try to compile and run the example program in `src/example.c` with the
following commands:
```
$ make example
$ ./gromov_wasser
```
This should print the processor time elapsed in `gw_gromov()` (where the actual
work is done), the final coupling of the spaces as a matrix, and the estimated
1-Gromov--Wasserstein distance (0.257648). Note that the arguments to
`gw_gromov()` are vectors and matrices as defined in `gw_structs.h` and
developed in `gw_vector.h` and `gw_matrix.h`. You will need to store the
distance matrices and measures for your spaces in these structures.

The example shows the simplest way to do so, writing everything as arrays and
calling `gw_vec_of_arr()` and `gw_mat_of_arr()`. If you want to work with files
instead, `gw_vec_fprintf()`, `gw_vec_fscanf()`, `gw_mat_fprintf()`, and
`gw_mat_fscanf()` handle reading and writing vectors and matrices in a human
readable format. If for some reason you want to work with the vectors directly,
you will probably need a call to `gw_vec_malloc()` to allocate a vector, and
`gw_vec_set()` or `gw_vec_subcpy()` to set the elements. `GW_VEC_FREE_ALL()`
and `GW_MAT_FREE_ALL()` are provided to free a set of vectors and matrices when
you are done.

While `gw_gromov()` requires the distance matrices (metrics) for the spaces,
you could instead have a weighted adjacency matrix where the weights indicate
edge lengths. The function `gw_adj_to_dist()` transforms such an adjacency
matrix into a distance matrix using Djikstra's algorithm.

All structs and functions exposed in the API have the prefix `gw` to reduce the
chance of naming conflicts. Any functions with the prefix `_` are for internal
use only, and their behavior could change with any release.

Finally, if you want to uninstall the library and headers, use the command:
```
$ make uninstall
```
This should delete the library and header files, but will not delete any
directories created during the installation.

### Performance

There are several guidelines to be followed in regards to the arguments of
`gw_gromov()`. The measures should be probability measures, and your spaces
should not contain any vertices of measure zero. If they do, remove the
vertices or add a small positive measure to them. Your distance matrices should
satisfy the triangle inequality, and the algorithm is more stable when the
median value in the distance matrices is of order one. If they are not, you can
divide by a constant and multiply the distance returned by the same constant.

Some considerable effort has been expended to make the calculation as fast as
possible. At the time of this writing there do not seem to be any other 
algorithmic enhancements available in the literature, the programming language
(C) does not use a runtime or garbage collector, and calls to efficient linear
algebra libraries (CBLAS) are made wherever possible.

For a small speedup, you could comment out the four defines in `src/safety.h`
before compiling the source. These control various runtime checks in the linear
algebra routines, including for invalid indices, mathematical overflow, memory
allocation errors, etc. This only decreases runtime by a couple of percent
though, and could make understanding when you have given invalid or malformed
input much more difficult.

Finally, you could add the `-ffast-math` option to the `FLAGS` variable in the
`makefile`. This could decrease runtime by a factor of two, but is really not
recommended. You have no guarantee that the program will give correct results
with this flag enabled.

### MATLAB

If you prefer to use MATLAB instead of C, the file `src/gw_matlab.c` can be
compiled as a mex function and called within the MATLAB environment. Detailed
instructions are provided in `gw_matlab.c`. Informal benchmarks show this to be
around 10 to 100 times faster than the native MATLAB implementation of the same
algorithms.

### Modules

This project contains several modules that could be useful for other purposes,
if you were so inclined. These include:

 * Basic linear algebra in `gw_vector.h`, `gw_matrix.h`, and a few other header
 files. This is mostly intended as a less painful wrapper around BLAS, with a
 few other routines for convenience. Memory allocation is always explicit. The
 main motivation for this over the GSL is just the license.
 * Calculation of the entropy-regularized 1-Wasserstein distance in
 `gw_wasserstein.h`. This is likely of interest only to researchers in computer
 vision at the moment.

## License

`gromov_wasser` is licensed under either of

 * [Apache License, Version 2.0][LICENSE-APACHE] (https://www.apache.org/licenses/LICENSE-2.0)
 * [MIT License][LICENSE-MIT] (https://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.

### Contributors

Jeremy Mason (jkmason@ucdavis.edu)

[Altschuler2017]: http://arxiv.org/abs/1705.09634
[Cuturi2013]: https://arxiv.org/abs/1306.0895
[Gorski2007]: https://doi.org/10.1007/s00186-007-0161-1
[Hendrikson2016]: http://hdl.handle.net/10062/50406 
[Memoli2007]: http://dx.doi.org/10.2312/SPBG/SPBG07/081-090
[Memoli2011]: https://doi.org/10.1007/s10208-011-9093-5
[Schmitzer2016]: https://arxiv.org/abs/1610.06519
[Thibault2017]: https://arxiv.org/abs/1711.01851
[ATLAS]: http://math-atlas.sourceforge.net/
[ARPACK]: https://github.com/opencollab/arpack-ng
[LAPACK]: http://netlib.org/lapack/
[OpenBLAS]: http://www.openblas.net/
[LICENSE-APACHE]: https://www.apache.org/licenses/LICENSE-2.0
[LICENSE-MIT]: https://opensource.org/licenses/MIT
