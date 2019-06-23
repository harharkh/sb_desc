// Copyright 2018 Jeremy Mason
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! \file tables.c
//! Contains functions to generate the lookup tables used in `sb_descriptors()`.

#include <math.h>       // sqrt
#include <stddef.h>     // size_t
#include <stdint.h>     // uint32_t
#include <stdlib.h>     // abort
#include "sb_structs.h" // sb_vec
#include "sb_utility.h" // SB_CHK_ERR
#include "sb_vector.h"  // sb_vec_malloc
#include "sbessel.h"    // _sbessel
#include "tables.h"

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
/// the search interval.
///
/// # Parameters
/// - `unl`: pointer to vector that holds the result
/// - `n_max`: defines the number of roots calculated
///
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `unl` is not `NULL`
/// - `SAFE_LENGTH`: `unl` contains (n_max + 1) * (n_max + 4) / 2 elems
///
/// # Examples
/// ```
/// #include <stdint.h>     // uint32_t
/// #include <stdio.h>      // FILE
/// #include "sb_structs.h" // sb_vec
/// #include "sb_utility.h" // SB_MAX
/// #include "sb_vector.h"  // sb_vec_malloc
/// #include "sbessel.h"    // _sbessel
/// #include "tables.h"     // _build_unl_tbl
/// 
/// // Generates the lookup table for the roots of the spherical Bessel
/// // functions of the first kind. WARNING: if you regenerate the table, you
/// // need to manually remove the two leading entries of `unl.tbl` and update
/// // the definitions of `_u_data`, `_u_rows` and `_u_cols` in `sb_desc.c`.
/// int main(void) {
///   uint32_t n_max = 140;
/// 
///   // Calculate the roots for the lookup table
///   sb_vec * unl = sb_vec_malloc((n_max + 1) * (n_max + 4) / 2, 'c');
///   _build_unl_tbl(unl, n_max);
/// 
///   uint32_t l, n, a;
///
///   // Verify that j_l(r) vanishes at the roots
///   double err;
///   double max_err = 0.;
///   for (l = 0; l < n_max + 1; ++l) {
///     for (n = 0; n < n_max + 2 - l; ++n) {
///       a = l * (2 * n_max - l + 5) / 2 + n;
///       err = _sbessel(l, sb_vec_get(unl, a));
///       max_err = SB_MAX(err, max_err);
///     }
///   }
///   printf("Max error: %g\n", max_err);
/// 
///   // Write the roots to file
///   FILE * f = fopen("src/unl.tbl", "w");
///   sb_vec_fprintf(f, unl, "%.16g,");
///   fclose(f);
/// 
///   SB_VEC_FREE_ALL(unl);
/// }
/// ```
void _build_unl_tbl(sb_vec * unl, const uint32_t n_max) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!unl, abort(), "_build_unl_tbl: unl cannot be NULL");
#endif
  size_t n_elem = (n_max + 1) * (n_max + 4) / 2;
#ifdef SAFE_LENGTH
  SB_CHK_ERR(unl->n_elem != n_elem, abort(),
      "_build_unl_tbl: unl must have (n_max + 1) * (n_max + 4) / 2 elems");
#endif
  // forward declaration of variables without initializations
  uint32_t n, l, a, b;

  // work directly with backing memory
  double * u_data = unl->data;

  // j_0(x) = sin(r) / r
  for (n = 0; n < n_max + 2; ++n) {
    u_data[n] = (n + 1) * PI;
  }

  for (l = 1; l < n_max + 1; ++l) {
    for (n = 0; n < n_max + 2 - l; ++n) {
      // initialize halley with roots of j_{l-1}(r)
      a = l * (2 * n_max - l + 5) / 2 + n;
      b = (l - 1) * (2 * n_max - l + 6) / 2 + n;
      u_data[a] = halley(l, u_data[b], u_data[b + 1]);
    }
  }

  return;
}

/// Calculates the coefficients necessary to construct the `f_nl(x)` appearing
/// in the definition of the radial basis functions.
///
/// # Parameters
/// - `c1`: pointer to vector to hold the first set of coefficients
/// - `c2`: pointer to vector to hold the second set of coefficients
/// - `n_max`: defines the number of coefficients calculated
///
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `c1` and `c2` are not `NULL`
/// - `SAFE_LENGTH`: `c1` and `c2` contain (n_max + 1) * (n_max + 2) / 2 elems
///
/// # Examples
/// ```
/// #include <stdint.h>     // uint32_t
/// #include <stdio.h>      // FILE
/// #include "sb_structs.h" // sb_vec
/// #include "sb_vector.h"  // sb_vec_malloc
/// #include "tables.h"     // _build_fnl_tbl
/// 
/// // Generates the lookup tables for the coefficients necessary to construct
/// // the `f_nl(x)` appearing in the definition of the radial basis functions.
/// // WARNING: if you regenerate the tables, you need to manually remove the
/// // two leading entries of `c1.tbl` and `c2.tbl` and update the
/// // definitions of `c1` and `c2` in `sb_desc.c`.
/// int main(void) {
///   uint32_t n_max = 140;
///   uint32_t n_elem = (n_max + 1) * (n_max + 2) / 2;
/// 
///   // Calculate the coefficients
///   sb_vec * c1 = sb_vec_malloc(n_elem, 'c');
///   sb_vec * c2 = sb_vec_malloc(n_elem, 'c');
///   _build_fnl_tbl(c1, c2, n_max);
/// 
///   // Write the coefficients to file
///   FILE * f;
///
///   f = fopen("src/c1.tbl", "w");
///   sb_vec_fprintf(f, c1, "%.16g,");
///   fclose(f);
///
///   f = fopen("src/c2.tbl", "w");
///   sb_vec_fprintf(f, c2, "%.16g,");
///   fclose(f);
/// 
///   SB_VEC_FREE_ALL(c1, c2);
/// }
/// ```
void _build_fnl_tbl(sb_vec * c1, sb_vec * c2, const uint32_t n_max) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!c1, abort(), "_build_fnl_tbl: c1 cannot be NULL");
  SB_CHK_ERR(!c2, abort(), "_build_fnl_tbl: c2 cannot be NULL");
#endif
  const size_t n_elem = (n_max + 1) * (n_max + 2) / 2;
#ifdef SAFE_LENGTH
  SB_CHK_ERR(c1->n_elem != n_elem, abort(),
      "_build_fnl_tbl: c1 must have (n_max + 1) * (n_max + 2) / 2 elems");
  SB_CHK_ERR(c2->n_elem != n_elem, abort(),
      "_build_fnl_tbl: c2 must have (n_max + 1) * (n_max + 2) / 2 elems");
#endif
  // Calculate the necessary roots
  sb_vec * unl = sb_vec_malloc((n_max + 1) * (n_max + 4) / 2, 'c');
  _build_unl_tbl(unl, n_max);

  // forward declaration of variables without initializations
  uint32_t n, l, a, b;
  double coeff, u0, u1;

  // work directly with backing memory
  double * u_data  = unl->data;
  double * c1_data = c1->data;
  double * c2_data = c2->data;

  for (l = 0; l < n_max + 1; ++l) {
    for (n = 0; n < n_max - l + 1; ++n) {
      a = l * (2 * n_max - l + 5) / 2 + n;
      u0 = u_data[a];
      u1 = u_data[a + 1];
      coeff = sqrt(2. / (SB_SQR(u0) + SB_SQR(u1)));

      b = l * (2 * n_max - l + 3) / 2 + n;
      c1_data[b] = u1 / _sbessel(l + 1, u0) * coeff;
      c2_data[b] = u0 / _sbessel(l + 1, u1) * coeff;
    }
  }

  SB_VEC_FREE_ALL(unl);

  return;
}
