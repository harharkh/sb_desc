// Copyright 2018 Jeremy Mason
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! \file sbesselz.c
//! Contains function to calculate the roots of the spherical Bessel functions
//! of the first kind in the restricted setting of this application.

#pragma once

#include <math.h>       // sin
#include <stddef.h>     // size_t
#include <stdint.h>     // uint32_t
#include <stdlib.h>     // abort
#include <string.h>     // memset
#include "sb_structs.h" // sb_mat
#include "sb_utility.h" // SB_CHK_ERR
#include "sbesselz.h"

#define PI 3.141592653589793

/// Calculates the roots of the spherical Bessel functions of the first kind. 
/// The roots are calculated iteratively using Halley's algorithm, with those
/// of the spherical Bessel function of the preceeding order used to bracket
/// the search interval. The function returns the first `n_max - l + 2`
/// positive roots of `j_l(r)` in the `l`th column of `u_nl` for increasing
/// values of `n`.
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
/// #include <stdio.h>
/// #include <stdint.h>
/// #include "sbessel.h"
/// 
/// int main(void) {
///   uint32_t l = 2;
///   double   r = 8.;
/// 
///   printf("Expected: -0.11105245\n");
///   printf("Result:   %.8f\n", _sbessel(l, r));
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
  size_t n, l;

  // work directly with backing memory
  double * u_data = u_nl->data;

  // j_0(x) = sin(r) / r
  for (n = 0; n < n_rows; ++n) {
    u_data[n] = n * PI;
  }
  u_data = u_data + n_rows;

  for (l = 1; l < n_cols; ++l) {
    for (n = 0; n < n_rows - l; ++n) {
      // initialize halley with roots of j_{l-1}(r)
      u_data[n] = halley(l, u_data[n - n_rows], u_data[n + 1 - n_rows]);
    }
    // zero out rest of column, value of n persists
    memset(u_data[n], 0, l * sizeof(double));
    u_data = u_data + n_rows;
  }

  return u_nl;
}



function [x] = halley(l, lwr_bnd, upr_bnd)
    % ADJUSTABLE
    TOL1 = 2.2204e-13;
    TOL2 = 2.2204e-14;
    MAXIT = 10000;
    
    x = (lwr_bnd + upr_bnd) / 2.;
    l1 = l + 1;

    for iter = 1:MAXIT
        a = spherical_bessel(l, x);
        b = spherical_bessel(l1, x);
        
        if abs(a) < TOL2
            break;
        end
        
        x2 = x * x;
        dx = -2. * x * a * (l * a - x * b) / ...
            ((l * l1 + x2) * a * a - 2. * x * (2. * l + 1.) * a * b + 2. * x2 * b * b);
        
        if abs(dx) < TOL1
            break;
        end

        if dx > 0.
            lwr_bnd = x;
        else
            upr_bnd = x;
        end
        if (upr_bnd - x) < dx || dx < (lwr_bnd - x)
            dx = (upr_bnd - lwr_bnd) / 2. - x;
        end
        
        x = x + dx;
    end

    if iter > MAXIT - 1
        disp('Failed to converge.');
    end
end
