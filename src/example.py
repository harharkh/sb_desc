# Copyright 2018 Jeremy Mason
#
# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.
# 
# \file example.py
# Contains a example of invoking the shared library `libsbdesc.so` with Python.
# Requires a NumPy installation. Identical in operation to `example.c`.
# 
# The NumPy installation available through my package manager (Debian Stretch)
# does not appear to link to libcblas.so. The command:
# 
# LD_PRELOAD=/usr/lib/libcblas.so python src/example.py
# 
# forces the loading of the appropriate library though, and the script below
# works. Perhaps this is due to the packages available through the repositories
# being compiled in a particular environment, and compiling from source would
# resolve the issue.

import ctypes
import numpy as np
import numpy.ctypeslib as ctl

class Py_sb_desc:
    def __init__(self, libname, libdir):
        self.c_sb_desc = ctl.load_library(libname, libdir).sb_descriptors

    def __call__(self, desc, n_max, disp, weights, n_atom, rc):
        self.c_sb_desc(
            ctl.as_ctypes(desc),    \
            ctypes.c_uint32(n_max), \
            ctl.as_ctypes(disp),    \
            ctl.as_ctypes(weights), \
            ctypes.c_uint32(n_atom),\
            ctypes.c_double(rc))
        
if __name__ == '__main__':
    # Sets the number of descriptors returned
    n_max = 4

    # Allocate memory for the result.
    desc = np.empty((n_max + 1) * (n_max + 2) // 2)

    # Number of atoms in the environment
    n_atom = 4

    # Displacements to the surrounding atoms in Angstroms
    # [x_1, y_1, z_1, ...]
    disp = np.array([1.3681827, -1.3103517, -1.3131874, -1.5151760,   \
            1.3360077, -1.3477119, -1.3989598, -1.2973683, 1.3679189, \
            1.2279369, 1.3400378,  1.4797429])

    # Weights for the surrounding atoms
    weights = np.array([1., 1., 1., 1.])

    # Cutoff radius in Angstroms
    rc = 3.7711

    # Create the python function object
    py_sb_desc = Py_sb_desc('libsbdesc.so', '/usr/local/lib/');

    # Actual calculation
    py_sb_desc(desc, n_max, disp, weights, n_atom, rc)

    np.set_printoptions(formatter={'float': '{:0.6f}'.format})
    print(desc)

    print("Completed successfully!")
