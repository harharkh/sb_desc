# Copyright 2018 Jeremy Mason
#
# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.

'''
Defines the Sb_desc class to interface Python with `libsbdesc.so`.
'''

import ctypes as ct
import numpy as np
import numpy.ctypeslib as ctl

class Sb_desc:
    '''
    Calculates spherical Bessel descriptors for the given atomic environment.
    While the description below specifies several of the argument as numpy
    arrays, anything that exposes the __array_interface__ should be accepted.

    Parameters
    ----------
    `disp`: numpy array
        The relative Cartesian coordinates of the surrounding atoms in the
        format `[x_1, y_1, z_1, ...]`
    `weights`: numpy array
        Atomic weights used to contruct the neighbor density function, e.g.
        `[1., ...]`
    `rc`: double
        Cutoff radius for the environment
    `n_atom`: int
        Number of atoms in the environment
    `n_max`: int
        The function evaluates (n_max + 1) * (n_max + 2) / 2 descriptors

    Returns
    ----------
    `desc`: numpy array
        Holds the descriptors, labelled by (n, l) and ordered lexicographically

    Warning
    ----------
    You are reponsible for ensuring that enough memory is allocated for the
    relevant arrays, and should expect undefined behavior otherwise. The
    lengths should be:
    `disp`: at least `3 * n_atom`
    `weights`: at least `n_atom`
    '''

    def __init__(self, libname, libdir):
        self.c_sb_desc = ctl.load_library(libname, libdir).sb_descriptors

    def __call__(self, disp, weights, rc, n_atom, n_max):
        desc = np.empty((n_max + 1) * (n_max + 2) // 2)
        self.c_sb_desc(
            ctl.as_ctypes(desc),    \
            ctl.as_ctypes(disp),    \
            ctl.as_ctypes(weights), \
            ct.c_double(rc),        \
            ct.c_uint32(n_atom),    \
            ct.c_uint32(n_max))
        return desc

