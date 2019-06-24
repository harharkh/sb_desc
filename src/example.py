# Copyright 2018 Jeremy Mason
#
# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.
 
'''
An example that uses the sb_desc module to interface with the `libsbdesc.so`
library. Identical in operation to `example.c`. You could need to change the
definition of `sb_desc_fun` if you installed the library in a non-standard 
location.
'''

import numpy as np
from sb_desc import Sb_desc

if __name__ == '__main__':
    # Create the python function object
    sb_desc_fun = Sb_desc('libsbdesc.so', '/usr/local/lib/');

    # Displacements to the surrounding atoms in Angstroms
    # [x_1, y_1, z_1, ...]
    disp = np.array([1.3681827, -1.3103517, -1.3131874, -1.5151760,   \
            1.3360077, -1.3477119, -1.3989598, -1.2973683, 1.3679189, \
            1.2279369, 1.3400378,  1.4797429])

    # Weights for the surrounding atoms
    weights = np.array([1., 1., 1., 1.])

    # Cutoff radius in Angstroms
    rc = 3.7711

    # Number of atoms in the environment
    n_atom = 4

    # Sets the number of descriptors returned
    n_max = 4

    # Actual calculation
    desc = sb_desc_fun(disp, weights, rc, n_atom, n_max)

    np.set_printoptions(formatter={'float': '{:0.6f}'.format})
    print(desc)

    print("Completed successfully!")
