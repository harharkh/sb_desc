// Copyright 2018 Jeremy Mason
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#include <stdio.h>      // printf
#include <stdint.h>     // uint32_t
#include "sb_descrip.h" // sb_descriptors
#include "sb_matrix.h"  // sb_mat_of_arr
#include "sb_structs.h" // sb_mat
#include "sb_utility.h" // sb_srandn
#include "sb_vector.h"  // sb_vec_of_arr

/// An example where `sb_descriptors()` is used to calculate the spherical
/// Bessel descriptors for an atomic environment containing four atoms; the
/// first and second descriptors should be `0.031870` and `0.138078`.
int main(void) {
  // Cutoff radius in Angstroms.
  double rc = 3.7711; 

  // The number of descriptors returned is `n_max (n_max + 1) / 2`.
  uint32_t n_max = 6;
  
  // Relative Cartesian coordinates of the surrounding atoms in Angstroms. The
  // format follows the pattern `[x_1, y_1, z_1, x_2, y_2, z_2, ...]`.
  double a[] = {1.3681827, -1.3103517, -1.3131874, -1.515176, 1.3360077,
    -1.3477119, -1.3989598, -1.2973683, 1.3679189, 1.2279369, 1.3400378,
    1.4797429};
  sb_vec * disp = sb_vec_of_arr(a, 12, 'c');
  
  // Weights for the atoms when calculating the neighbor density function.
  sb_vec * weight = sb_vec_malloc(4, 'c');
  sb_vec_set_all(weight, 1.);

  // Number of atoms in the environment. Generally `disp` and `weight` should
  // be allocated to accomodate the maximum number of atoms that could occur
  // within the cutoff radius.
  uint32_t n_atom = 4;

  // Need to allocate memory for the result.
  sb_vec * desc = sb_vec_malloc(n_max * (n_max + 1) / 2, 'c');

  SB_TIC;
  sb_descriptors(desc, rc, n_max, disp, weight, n_atom);
  SB_TOC;

  sb_vec_print(desc, "Descriptors: ", "%.4f");
  
  SB_VEC_FREE_ALL(disp, weight, desc);

  printf("Completed successfully!\n");
}
