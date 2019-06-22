// Copyright 2018 Jeremy Mason
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! \file sbessel.c
//! Contains function to calculate the spherical Bessel functions of the first
//! kind in the restricted setting of this application.

#include <math.h>       // sin, cos
#include <stdbool.h>    // bool
#include <stdint.h>     // uint32_t
#include <stdlib.h>     // abort
#include "sb_utility.h" // SB_CHK_ERR
#include "sbessel.h"
#include "safety.h"

#define TOL1 2e-14
#define TOL2 2e-15
#define THRESH 100.
#define MAXIT 10000
    
#define PI 3.141592653589793
#define PI2 1.570796326794897
#define PI32 0.098174770424681
    
/// Calculate the spherical Bessel functions of the first kind using the
/// continued fraction algorithm of W. J. Lentz in Computers in Physics 4, 403
/// (1990). The algorithm is translated from FORTRAN77 to C11 and lightly
/// modified to improve numerical stability.
///
/// # Parameters
/// - `l`: spherical Bessel function order
/// - `r`: spherical Bessel function argument
///
/// # Returns
/// Value of the spherical Bessel function
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_FINITE`: `r` is nonnegative
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
double _sbessel(uint32_t l, double r) {
#ifdef SAFE_FINITE
  SB_CHK_ERR(r < 0., abort(), "_sbessel: r cannot be negative");
#endif
  // Boundary condition known exactly
  if (r < TOL2) {
    if (l == 0) {
      return 1.;
    } else {
      return 0.;
    }
  }

  // Small values of l known exactly
  if (l == 0) {
    return sin(r) / r;
  }
  if (l == 1) {
    return (sin(r) / r - cos(r)) / r;
  }

  bool nflag = false;
  bool dflag = false;

  double num = 0.;
  double den = 0.;
  double pdt = 1.;
  double pwr = 0.;
    
  double b = 0.;
  uint32_t iter = 0;
  double base = 0.;

  // j_0 as the base function is strictly preferrable with regard to the
  // error, except close to integer multiples of pi where j_0 approaches
  // zero. j_1 is used instead in this situation.
  if (fabs(fmod(r + PI2, PI) - PI2) > PI32) {
    iter = 0;
    base = sin(r) / r;
  } else {
    iter = 1;
    base = (sin(r) / r - cos(r)) / r;
  }

  while (nflag || dflag || (fabs((num - den) / den) > TOL1) || (iter <= l)) {
    // Calculating b directly significantly reduces numerical error
    b = (2. * iter + 3.) / r;

    num = b - num;
    if (!nflag) {
      if (fabs(num) > TOL2) {
        pdt = pdt * num;
        num = 1. / num;
      } else {
        nflag = true;
        num = 0.;
        pdt = -pdt;
      }
    } else {
      nflag = false;
      num = 0.;
    }

    if (fabs(pdt) > THRESH || fabs(pdt) < 1. / THRESH) {
      pwr = pwr - log(fabs(pdt));
      pdt = copysign(1., pdt);
    }

    iter += 1;
    SB_CHK_ERR(iter > MAXIT, break, "_sbessel: failed to converge");

    if (iter > l) {
      den = b - den;
      if (!dflag) {
        if (fabs(den) > TOL2) {
          pdt = pdt / den;
          den = 1. / den;
        } else {
          dflag = true;
          den = 0.;
          pdt = -pdt;
        }
      } else {
        dflag = false;
        den = 0.;
      }
    }
  }

  return base / pdt * exp(pwr);
}
