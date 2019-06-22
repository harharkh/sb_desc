// Copyright 2018 Jeremy Mason
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! \file sbesselz.c
//! Contains function to calculate the roots of the spherical Bessel functions
//! of the first kind in the restricted setting of this application.

#include <math.h>       // sin
#include <stdbool.h>    // true
#include <stddef.h>     // size_t
#include <stdint.h>     // uint32_t
#include <stdlib.h>     // abort
#include <string.h>     // memset
#include "sb_structs.h" // sb_mat
#include "sb_utility.h" // SB_CHK_ERR
#include "sbessel.h"    // _sbessel
#include "sbesselz.h"

#define PI 3.141592653589793

#define TOL1 2.220446049250313e-15
#define TOL2 2.220446049250313e-16
#define MAXIT 10000

/// Halley's method to find the roots of the spherical Bessel functions of the
/// first kind. The derivatives are calculated using recursion relations and 
/// incorporated directly into the equation for the update.
///
/// # Parameters
/// - `l`: order of the spherical Bessel function
/// - `lwr`: lower bound on the search interval
/// - `upr`: upper bound on the search interval
///
/// # Returns
/// Position of a root within the interval.
static double halley(uint32_t l, double lwr, double upr) {
  const uint32_t l1 = l + 1;
  const double   d1 = l * l1;
  const double   d2 = 4. * l + 2.;

  double x = (lwr + upr) / 2.;

  double a, b, x2, a2, ab, b2, dx;
  size_t iter;
  for (iter = 0; iter < MAXIT; ++iter) {
    a = _sbessel( l, x);
    if (fabs(a) < TOL2) { return x; }
    b = _sbessel(l1, x);

    x2 = SB_SQR(x);
    a2 = SB_SQR(a);
    ab = a * b;
    b2 = SB_SQR(b);

    dx = -2. * x * (l * a2 - x * ab);
    dx /= ((d1 + x2) * a2 - d2 * x * ab + 2. * x2 * b2);
    if (fabs(dx) < TOL1) { return x; }

    // update bounds
    if (dx > 0.) {
      lwr = x;
    } else {
      upr = x;
    }

    // verify that change is within bounds
    if ((upr - x) < dx || dx < (lwr - x)) {
      dx = (upr - lwr) / 2. - x;
    }

    x += dx;
  }

  SB_CHK_ERR(iter == MAXIT, , "halley: failed to converge");
  return x;
}

/// Calculates the roots of the spherical Bessel functions of the first kind. 
/// The roots are calculated iteratively using Halley's algorithm, with those
/// of the spherical Bessel function of the preceeding order used to bracket
/// the search interval. The function returns the first `n_max - l + 2`
/// positive roots of `j_l(r)` in the `l`th column of `u_nl` for increasing
/// values of `n`. Numerical error appears to be within a factor of ten of the
/// machine precision.
///
/// # Parameters
/// - `u_nl`: matrix to hold the result
/// - `n_max`: defines the number of roots calculated
///
/// # Returns
/// A copy of `u_nl`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `u_nl` is not `NULL`
/// - `SAFE_LENGTH`: `u_nl` contains n_max + 2 rows and n_max + 1 columns
///
/// # Examples
/// ```
/// #include <stddef.h>     // size_t
/// #include <stdint.h>     // uint32_t
/// #include <stdio.h>      // FILE
/// #include "sb_matrix.h"  // sb_mat_malloc
/// #include "sb_structs.h" // sb_mat
/// #include "sb_utility.h" // SB_MAX
/// #include "sbessel.h"    // _sbessel
/// #include "sbesselz.h"   // _sbesselz
/// 
/// // Use this example to calculate the lookup table for the roots of the
/// // spherical Bessel functions of the first kind. Adjust the value of n_max,
/// // execute, and modify the initial size information in `sb_roots.src`.
/// int main(void) {
///   uint32_t n_max = 141;
/// 
///   // Calculate the roots for the lookup table
///   sb_mat * u_nl = sb_mat_malloc(n_max + 2, n_max + 1);
///   _sbesselz(u_nl, n_max);
/// 
///   // Verify that j_l(r) vanishes at the roots
///   double err;
///   double max_err = 0.;
///   for (size_t l = 0; l < u_nl->n_cols; ++l) {
///     for (size_t n = 0; n < u_nl->n_rows - l; ++n) {
///       err = _sbessel(l, sb_mat_get(u_nl, n, l));
///       max_err = SB_MAX(err, max_err);
///     }
///   }
///   printf("Max error: %g\n", max_err);
/// 
///   // Write the roots to file
///   FILE * f = fopen("src/sb_roots.tbl", "w");
///   sb_mat_fprintf(f, u_nl, "%.16g,");
///   fclose(f);
/// 
///   SB_MAT_FREE_ALL(u_nl);
/// }
/// ```
sb_mat * _sbesselz(sb_mat * u_nl, uint32_t n_max) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!u_nl, abort(), "_sbesselz: u_nl cannot be NULL");
#endif
  size_t n_rows = n_max + 2;
  size_t n_cols = n_max + 1;
#ifdef SAFE_LENGTH
  SB_CHK_ERR(u_nl->n_rows == n_rows, abort(),
      "_sbesselz: u_nl must have n_max + 2 rows");
  SB_CHK_ERR(u_nl->n_cols == n_cols, abort(),
      "_sbesselz: u_nl must have n_max + 1 cols");
#endif
  // forward declaration of variables without initializations
  uint32_t n, l;

  // work directly with backing memory
  double * u_data = u_nl->data;

  // j_0(x) = sin(r) / r
  for (n = 0; n < n_rows; ++n) {
    u_data[n] = (n + 1) * PI;
  }
  u_data = u_data + n_rows;

  for (l = 1; l < n_cols; ++l) {
    for (n = 0; n < n_rows - l; ++n) {
      // initialize halley with roots of j_{l-1}(r)
      u_data[n] = halley(l, u_data[n - n_rows], u_data[n + 1 - n_rows]);
    }
    // zero out rest of column, value of n persists
    memset(u_data + n, 0, l * sizeof(double));
    u_data = u_data + n_rows;
  }

  return u_nl;
}
