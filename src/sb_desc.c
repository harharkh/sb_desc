// Copyright 2018 Jeremy Mason
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! \file sb_desc.c
//! Contains functions that actually calculate the spherical Bessel descriptors.

#include <stdint.h>     // uint32_t
#include <stdlib.h>     // abort
#include "sb_desc.h"
#include "sb_utility.h" // SB_CHK_ERR
#include "safety.h"

/// Calculates spherical Bessel descriptors for the given atomic environment.
/// `desc` should contain space for the descriptors, labelled by (n, l) and
/// ordered lexicographically. `disp` should contain the relative Cartesian 
/// coordinates of the surrouding atoms in the format `[x_1, y_1, z_1, ...]`.
/// `weights` should contain the weights used in the construction of the
/// neighbor density function (e.g., `[1., ...]`).
///
/// # Parameters
/// - `desc`: pointer to an array to hold the result
/// - `n_max`: defines the number of descriptors calculated
/// - `disp`: pointer to an array of the relative displacements
/// - `weights`: pointer to an array of the atomic weights
/// - `n_atom`: number of atoms in the environment
/// - `rc`: cutoff radius for the environment
///
/// # Returns
/// A copy of `desc`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `desc`, `disp`, and `weights` are not `NULL`
/// - `SAFE_FINITE`: `rc` is nonnegative
///
/// # Warning
/// You are reponsible for ensuring that enough memory is allocated for the
/// relevant arrays, and should expect undefined behavior otherwise. The
/// lengths should be:
/// - `desc`: exactly `(n_max + 1) * (n_max + 2) / 2`
/// - `disp`: at least `3 * n_atom`
/// - `weights`: at least `n_atom`
/// 
/// # Examples
/// ```
/// #include <stddef.h>     // size_t
/// #include <stdio.h>      // printf
/// #include <stdint.h>     // uint32_t
/// #include "sb_desc.h"    // sb_descriptors
/// #include "sb_utility.h" // SB_TIC
/// 
/// /// An example where `sb_descriptors()` is used to calculate the spherical
/// /// Bessel descriptors for an atomic environment containing four atoms; the
/// /// first and second descriptors should be `0.031870` and `0.138078`.
/// int main(void) {
///   // Sets the number of descriptors returned.
///   uint32_t n_max = 4;
/// 
///   // Allocate memory for the result.
///   double desc[15] = { 0. };
/// 
///   // Number of atoms in the environment.
///   uint32_t n_atom = 4;
/// 
///   // Displacements to the surrounding atoms in Angstroms.
///   // [x_1, y_1, z_1, ...]
///   double disp[12] = {
///      1.3681827, -1.3103517, -1.3131874,
///     -1.5151760,  1.3360077, -1.3477119,
///     -1.3989598, -1.2973683,  1.3679189,
///      1.2279369,  1.3400378,  1.4797429
///   };
/// 
///   // Weights for the surrounding atoms.
///   double weights[4] = { 1., 1., 1., 1. };
/// 
///   // Cutoff radius in Angstroms.
///   double rc = 3.7711; 
/// 
///   SB_TIC;
///   sb_descriptors(desc, n_max, disp, weights, n_atom, rc);
///   SB_TOC;
/// 
///   // Output the result
///   // Labelled by (n, l), ordered lexicographically
///   printf("SB descriptors:\n");
///   for (size_t a = 0; a < (n_max + 1) * (n_max + 2) / 2; ++a)
///     printf("%.6f\n", desc[a]);
///   
///   printf("Completed successfully!\n");
/// }
/// ```
double * sb_descriptors(
    double * restrict desc,
    const uint32_t n_max, 
    const double * restrict disp,
    const double * restrict weights,
    const uint32_t n_atom,
    const double rc) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!desc, abort(), "sb_descriptors: desc cannot be NULL");
  SB_CHK_ERR(!disp, abort(), "sb_descriptors: disp cannot be NULL");
  SB_CHK_ERR(!weights, abort(), "sb_descriptors: weights cannot be NULL");
#endif
#ifdef SAFE_FINITE
  SB_CHK_ERR(rc < 0., abort(), "sb_descriptors: rc cannot be negative");
#endif

  return desc;
}
