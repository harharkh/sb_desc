# sb_descriptors

`sb_descriptors` is a C library to calculate the spherical Bessel descriptors 
defined by E. Kocer, J. K. Mason, and H. Erturk in [Kocer 2019 a][Kocer2019a]
and [Kocer 2019 b]. The intention is for these to be used to describe local
atomic environments as is necessary for, e.g., machine learning potentials.

The main contribution of this library is the efficient calculation of radial
basis functions based on the eponymous spherical Bessel functions of the first
kind. This uses a stabilized version of the continued fraction algorithm of
[Lentz 1990][Lentz1990] since this is faster than the functions available in
standard libraries and the loss of accuracy is acceptable.

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

These should be available through any reasonable package manager, but could 
be installed manually if you are working in an environment without one.
Depending on the installation, you might need to modify the `LIBS` and 
`LIB_LOCATION_OPT` variables in the included `makefile`.

### Usage

This project is intended to be used as a dynamic library by other programs. By
default, the library is installed to `/usr/local/lib` and the headers for the
API are installed to `/usr/local/include/sbdesc`. These directories are not
necessarily on the default search path, so you should probably pass the 
`-I/usr/local/include/sbdesc` and `-L/usr/local/lib` options along with the
`-lsbdesc` option to your compiler. If you are not on a POSIX compliant system
or want the library and headers installed elsewhere, modify the `LIBRARY_DIR`
and `HEADERS_DIR` variables in the included `makefile`.

Once the installation directories are specified, you can compile and install
the library with the following commands:
```
$ make
$ make install
```
You should get a message indicating that everything went well. You should then
probably try to compile and run the example program in `src/example.c` with the
following commands:
```
$ make example
$ ./sb_example
```
This should print 15 descriptors for the selected environment (the first and
last descriptors should be `0.031871` and `0.483945`).

When executing the example program, you could instead get the following error:
```
./sb_example: error while loading shared libraries: libsbdesc.so: cannot open
shared object file: No such file or directory
```
This indicates that your operating system is not able to find the library.
Fortunately, there is more than one way to resolve this issue. The preferred
option is to execute the following command (requires root privileges):
```
$ ldconfig /usr/local/lib/
```
(or the appropriate variant for a different installation directory) to update
the shared library cache. If you don't have root privileges, you could instead
uncomment the line:
```
# example : FLAGS_OPT += -Wl,-rpath=$(LIBRARY_DIR)
```
in the makefile and recompile the example. This includes a path to the library
in the executable, but is less flexible in general.

Note that several of the arguments to `sb_descriptors()` are arrays of doubles.
You will need to store the relative atomic coordinates, atomic weights, and the
resulting descriptors in this way, but the array is a fundamental enough data
structure that most languages should provide some means of doing so.

All structs and functions exposed in the API have the prefix `sb` to reduce the
chance of naming conflicts. Any functions with the prefix `_` are for internal
use only, and their behavior could change with any release.

Finally, if you want to uninstall the library and headers, use the command:
```
$ make uninstall
```
This should delete the library and header files, but will not delete any
directories created during the installation.

### Python

A Python interface to the library is provided in `sb_desc.py`, and an example
script in `example.py`. If you have Python and NumPy installed, you should be
able to run this script with the following command:
```
$ python src/example.py
```
This should print 15 descriptors for the selected environment (the first and
last descriptors should be `0.031871` and `0.483945`).

When executing the example script, you could instead get the following error:
```
OSError: /usr/local/lib/libsbdesc.so: undefined symbol: cblas_dscal
```
This indicates that your Python installation is not able to find a `cblas`
library. You can resolve this with the following command:
```
LD_PRELOAD=/usr/lib/libcblas.so python src/example.py
```
(or the appropriate variant for a different installation directory) to force
the appropriate library to be preloaded.

### Performance

Some considerable effort has been expended to make the calculation as fast as
possible. At the time of this writing there do not seem to be any other obvious
algorithmic enhancements available in the literature, the programming language
(C) does not use a runtime or garbage collector, and calls to efficient linear
algebra libraries (CBLAS) are made wherever possible. The result is roughly an
order of magnitude faster than the equivalent MATLAB code.

For a small speedup, you could comment out the four defines in `src/safety.h`
before compiling the source. These control various runtime checks in the linear
algebra routines, including for invalid indices, mathematical overflow, memory
allocation errors, etc. This only decreases runtime by a couple of percent
though, and could make understanding when you have given invalid or malformed
input much more difficult.

Finally, you could add the `-ffast-math` option to the `FLAGS` variable in the
`makefile`. This could decrease runtime further, but is really not recommended.
You have no guarantee that the program will give correct results with this flag
enabled.

### Modules

This project contains several modules that could be useful for other purposes,
if you were so inclined. These include:

 * Basic linear algebra in `sb_vector.h`, `sb_matrix.h`, and a few other header
 files. This is mostly intended as a less painful wrapper around BLAS, with a
 few other routines for convenience. Memory allocation is always explicit. The
 main motivation for this over the GSL is just the license.

## License

`sb_descriptors` is licensed under either of

 * [Apache License, Version 2.0][LICENSE-APACHE] (https://www.apache.org/licenses/LICENSE-2.0)
 * [MIT License][LICENSE-MIT] (https://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.

### Contributors

Jeremy Mason (jkmason@ucdavis.edu)

[Kocer2019a]: https://aip.scitation.org/doi/10.1063/1.5086167
[Lentz1990]: https://doi.org/10.1063/1.168382
[ATLAS]: http://math-atlas.sourceforge.net/
[OpenBLAS]: http://www.openblas.net/
[LICENSE-APACHE]: https://www.apache.org/licenses/LICENSE-2.0
[LICENSE-MIT]: https://opensource.org/licenses/MIT
