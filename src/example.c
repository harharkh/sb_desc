// Copyright 2018 Jeremy Mason
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#include <stddef.h>     // size_t
#include <stdio.h>      // printf
#include <stdint.h>     // uint32_t
#include "sb_desc.h"    // sb_descriptors
#include "sb_utility.h" // SB_TIC

/// An example where `sb_descriptors()` is used to calculate the spherical
/// Bessel descriptors for an atomic environment containing four atoms; the
/// first and last descriptors should be `0.031871` and `0.483945`.
int main(void) {
  // Sets the number of descriptors returned.
  uint32_t n_max = 4;

  // Allocate memory for the result.
  double desc[15] = { 0. };

  // Number of atoms in the environment.
  uint32_t n_atom = 4;

  // Displacements to the surrounding atoms in Angstroms.
  // [x_1, y_1, z_1, ...]
  double disp[12] = {
     1.3681827, -1.3103517, -1.3131874,
    -1.5151760,  1.3360077, -1.3477119,
    -1.3989598, -1.2973683,  1.3679189,
     1.2279369,  1.3400378,  1.4797429
  };

  // Weights for the surrounding atoms.
  double weights[4] = { 1., 1., 1., 1. };

  // Cutoff radius in Angstroms.
  double rc = 3.7711; 

  SB_TIC;
  sb_descriptors(desc, n_max, disp, weights, n_atom, rc);
  SB_TOC;

  // Output the result
  printf("SB descriptors:\n");
  for (size_t a = 0; a < (n_max + 1) * (n_max + 2) / 2; ++a)
    printf("%.6f\n", desc[a]);
  
  printf("Completed successfully!\n");
}
