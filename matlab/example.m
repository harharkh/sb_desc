% Copyright 2018 Jeremy Mason
%
% Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
% http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT
% or http://opensource.org/licenses/MIT>, at your option. This file may not
% be copied, modified, or distributed except according to those terms.
% 
% An example that uses the sb_desc function to interface with the
% libsbdesc.so library. Identical in operation to example.c. You could
% need to change the path information in sb_desc if you installed the
% library in a non-standard location.

% Displacements to the surrounding atoms in Angstroms
pos = [ 1.3681827, -1.3103517, -1.3131874, ...
    -1.5151760,  1.3360077, -1.3477119, ...
    -1.3989598, -1.2973683,  1.3679189, ...
     1.2279369,  1.3400378,  1.4797429];

% Weights for the surrounding atoms
weights = [1., 1., 1., 1.];

% Cutoff radius in Angstroms
rc = 3.7711;

% Number of atoms in the environment
n_atom = 4;

% Sets the number of descriptors returned
n_max = 4;

% Actual calculation
desc = sb_desc(pos, weights, rc, n_atom, n_max);
disp(desc);

disp('Completed successfully!')
