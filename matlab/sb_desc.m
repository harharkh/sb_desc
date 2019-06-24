% Copyright 2018 Jeremy Mason
%
% Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
% http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT
% or http://opensource.org/licenses/MIT>, at your option. This file may not
% be copied, modified, or distributed except according to those terms.
% 
% Calculates spherical Bessel descriptors for the given atomic environment.
%
% Parameters
% ----------
% - pos: The relative Cartesian coordinates of the surrounding atoms in
%     the format [x_1, y_1, z_1, ...]
% - weights:  Atomic weights used to contruct the neighbor density
%     function, e.g. [1., ...]
% - rc: Cutoff radius for the environment
% - n_atom: Number of atoms in the environment
% - n_max: The function evaluates (n_max + 1) * (n_max + 2) / 2 descriptors
% 
% Returns
% ----------
% - desc: The descriptors, labelled by (n, l) and ordered lexicographically
% 
% Warning
% ----------
% You are reponsible for ensuring that enough memory is allocated for the
% relevant arrays, and should expect undefined behavior otherwise. The
% lengths should be:
% - pos: at least 3 * n_atom
% - weights: at least n_atom

function desc = sb_desc(pos, weights, rc, n_atom, n_max)
    if ~libisloaded('libsbdesc')
        % Should be consistent with the makefile
        path_to_lib = '/usr/local/lib/libsbdesc.so';
        path_to_header = '/usr/local/include/sbdesc/sb_desc.h';

        % Ignore warning about #pragma once
        [~, msgId] = lastwarn;
        warnStruct = warning('off', msgId);

        % Load the library
        loadlibrary(path_to_lib, path_to_header);

        % Restore warnings
        warning(warnStruct);
    end

    desc = zeros((n_max + 1) * (n_max + 2) / 2, 1);
    [~, desc, ~, ~] = calllib('libsbdesc','sb_descriptors', desc, pos, weights, rc, n_atom, n_max);   
end
