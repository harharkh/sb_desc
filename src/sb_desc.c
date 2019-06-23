// Copyright 2018 Jeremy Mason
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! \file sb_desc.c
//! Contains functions that actually calculate the spherical Bessel descriptors.

#include <cblas.h>      // dscal
#include <math.h>       // pow
#include <stdint.h>     // uint32_t
#include <stdlib.h>     // abort
#include "sbessel.h"    // _sbessel
#include "sb_desc.h"
#include "sb_matrix.h"  // sb_mat_malloc
#include "sb_structs.h" // sb_vec
#include "sb_utility.h" // SB_CHK_ERR
#include "sb_vector.h"  // sb_vec_calloc
#include "safety.h"

// Lookup tables used in get_radial_basis. Built using function in `tables.c`.
static const double _u_data[10152] = {
  #include "unl.tbl"
};
static const size_t _u_n_max = 140;
static const size_t _u_rows = 143;
static const size_t _u_cols = 142;

static const double _c1_data[10011] = {
  #include "c1.tbl"
};
static const size_t _c1_n_max = 140;

static const double _c2_data[10011] = {
  #include "c2.tbl"
};
static const size_t _c2_n_max = 140;

/// Calculates the radial basis functions for the spherical Bessel descriptors.
/// Numerical efficiency depends heavily on the lookup tables defined above, 
/// allowing the function to be reduced to evaluating the recursion relations.
///
/// # Parameters
/// - `gnl`: pointer to a matrix to hold the result
/// - `r_data`: radial coordinates of the atoms
/// - `n_max`: defines the number of descriptors calculated
/// - `l`: order of the spherical Bessel functions
/// - `n_atom`: number of atoms in the environment
/// - `rc`: cutoff radius for the environment
///
/// # Returns
/// A copy of `gnl`
static sb_mat * get_radial_basis(
    sb_mat * gnl,
    double * r_data,
    uint32_t n_max, 
    uint32_t l,
    uint32_t n_atom,
    double rc) {
  // access lookup tables directly
  const double * u_data  = _u_data  + l * _u_rows;
  const double * c1_data = _c1_data + l * (2 * _c1_n_max - l + 3) / 2;
  const double * c2_data = _c2_data + l * (2 * _c2_n_max - l + 3) / 2;

  // gnl->n_cols
  const size_t n_cols = gnl->n_cols;

  // forward declaration of variables without initializations
  size_t n, a;
  double u0, u1, u2, d0, d1, e;
  double * g_data;

  // fnl built in gnl
  g_data = gnl->data;
  for (n = 0; n < n_cols; ++n) {
    for (a = 0; a < n_atom; ++a) {
      g_data[a] = c1_data[n] * _sbessel(l, r_data[a] * u_data[n])
        - c2_data[n] * _sbessel(l, r_data[a] * u_data[n + 1]);
    }
    g_data += n_atom;
  }
  sb_mat_smul(gnl, pow(rc, -1.5));

  // initialize quantities used for recursion
  u1 = SB_SQR(u_data[0]);
  u2 = SB_SQR(u_data[1]);
  d1 = 1.;

  // convert to gnl
  g_data = gnl->data;
  for (n = 1; n < n_cols; ++n) {
    u0 = u1;
    u1 = u2;
    u2 = SB_SQR(u_data[n + 1]);

    e = (u0 * u2) / ((u0 + u1) * (u1 + u2));
    d0 = d1;
    d1 = 1. - e / d0;

    g_data += n_atom;
    cblas_dscal(n_atom, 1. / sqrt(d1), g_data, 1);
    cblas_daxpy(n_atom, sqrt(e / (d1 * d0)), g_data - n_atom, 1, g_data, 1);
  }

  sb_mat_print(gnl, "gnl: ", "%.4g");
  
  return gnl;
}

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
/// - `desc`: `(n_max + 1) * (n_max + 2) / 2`
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
    double * restrict desc_arr,
    const uint32_t n_max, 
    double * restrict disp_arr,
    const double * restrict weights_arr,
    const uint32_t n_atom,
    const double rc) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!desc_arr, abort(), "sb_descriptors: desc cannot be NULL");
  SB_CHK_ERR(!disp_arr, abort(), "sb_descriptors: disp cannot be NULL");
  SB_CHK_ERR(!weights_arr, abort(), "sb_descriptors: weights cannot be NULL");
#endif
#ifdef SAFE_FINITE
  SB_CHK_ERR(rc < 0., abort(), "sb_descriptors: rc cannot be negative");
#endif
  // Convert raw pointers to sb_vec and sb_mat
  sb_vec * desc = malloc(sizeof(sb_vec));
  SB_CHK_ERR(!desc, abort(), "sb_descriptors: failed to allocate desc");

  desc->n_elem = (n_max + 1) * (n_max + 2) / 2;
  desc->data   = desc_arr;
  desc->layout = 'c';

  sb_mat * disp = malloc(sizeof(sb_mat));
  SB_CHK_ERR(!disp, abort(), "sb_descriptors: failed to allocate disp");

  disp->n_rows = 3;
  disp->n_cols = n_atom;
  disp->n_elem = 3 * n_atom;
  disp->data   = disp_arr;

  double * data1, * data2;
  size_t a, b, c;

  // Calculate radial coordinates
  sb_vec * radius = sb_vec_calloc(n_atom, 'r');
  data1 = disp->data;
  data2 = radius->data;
  for (a = 0; a < n_atom; ++a) {
    for (b = 0; b < 3; ++b) {
      *data2 += SB_SQR(data1[b]);
    }
    data1 += 3;
    data2 += 1;
  }
  sb_vec_sqrt(radius);

  // Normalize displacement vectors
  sb_mat_vdiv(disp, radius, 'c');
  sb_vec_smul(radius, 1. / rc);

  // Calculate angle cosines
  sb_mat * gamma = sb_mat_malloc(n_atom, n_atom);
  sb_mat_mm_mul(gamma, disp, disp, "tn");

  // Legendre polynomials
  sb_mat * * lp = malloc((n_max + 1) * sizeof(sb_mat *));
  SB_CHK_ERR(!lp, abort(), "sb_descriptors: failed to allocate lp");
  for (a = 0; a <= n_max; ++a) {
    lp[a] = sb_mat_malloc(n_atom, n_atom);
  }

  sb_mat_set_all(lp[0], 1.);
  sb_mat_memcpy(lp[1], gamma);
  for (a = 2; a <= n_max; ++a) { // l = a
    sb_mat_memcpy(lp[a], gamma);
    sb_mat_pmul(lp[a], lp[a - 1]);
    sb_mat_smul(lp[a], (2. * a - 1.) / (a - 1.));
    sb_mat_psub(lp[a], lp[a - 2]);
    sb_mat_smul(lp[a], (a - 1.) / a);
  }

  // Include multiplier here to simplify calculation below
  for (a = 0; a <= n_max; ++a) {
    sb_mat_smul(lp[a], (2. * a + 1.) / 12.566370614359172);
  }
    
  // Radial basis functions
  sb_mat * * gnl = malloc((n_max + 1) * sizeof(sb_mat *));
  SB_CHK_ERR(!gnl, abort(), "sb_descriptors: failed to allocate gnl");
  for (a = 0; a <= n_max; ++a) { // l = a
    gnl[a] = sb_mat_malloc(n_atom, n_max - a + 1);
    get_radial_basis(gnl[a], radius->data, n_max, a, n_atom, rc);
  }
  abort();

  /*
  c = 0;
  for (a = 0; a <= n_max; ++a) {
    for (b = 0; b <= a; ++b) {
      desc[c++] = ;
    }
  }

    
    % power-spectrum calculation
    % (np + 1, l + 1)
    % np = n + l is the index for the power spectrum
    % n = np - l is the index for the calculation of the basis functions
    pnl = zeros(n_max + 1, n_max + 1);
    for np = 1:(n_max + 1) % offset by one for indexing
        for l = 1:np % offset by one for indexing
            pnl(np, l) = gnl(:, np - l + 1, l)' * lp(:, :, l) * gnl(:, np - l + 1, l);
        end
    end
end
  */

  // Free memory
  for (a = 0; a <= n_max; ++a) {
    SB_MAT_FREE_ALL(lp[a], gnl[a]);
  }
  SB_FREE_ALL(desc, disp, lp, gnl);
  SB_VEC_FREE_ALL(radius);
  SB_MAT_FREE_ALL(gamma);

  return desc_arr;
}
