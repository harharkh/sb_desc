// Copyright 2018 Jeremy Mason
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! \file utility.c
//! Contains type definitions, macros, and utility functions used by the
//! sb_dist project.

#include <math.h>    // cos
#include <stdbool.h> // bool
#include <stdint.h>  // uint32_t
#include <stdio.h>   // FILE, NULL
#include <stdlib.h>  // rand
#include <time.h>    // clock_t
#include "sb_utility.h"

FILE * sb_error_log = NULL;
_Thread_local clock_t sb_tic_time;
static _Thread_local uint32_t randn_state;

/// Sets the error log used by `SB_CHK_ERR`. Internally uses an `extern`
/// variable, only needs to be called once (usually in `main`). NOTE: This is 
/// not thread safe by design. If you call this, be certain to do so before any
/// threads are spawned.
///
/// # Parameters
/// - `f`: `FILE` pointer to the error log
///
/// # Returns
/// No return value
/// 
/// # Examples
/// ```
/// #include <stdio.h>
/// #include <stdlib.h>
/// #include "sb_utility.h"
/// 
/// int main(void) {
///   double numer = 1.;
///   double denom = 0.;
/// 
///   FILE * file = fopen("error_log.txt", "w");
///   sb_set_error_log(file);
/// 
///   SB_CHK_ERR(denom == 0., exit(1), "denominator must be nonzero");
///   printf("%g / %g = %g\n", numer, denom, numer / denom);
/// 
///   fclose(file);
/// }
/// ```
void sb_set_error_log(FILE * f) {
  sb_error_log = f;
}

/// Sets the internal state required by `sb_randn()` to generate normally
/// distributed random numbers. NOTE: Uses thread-local storage, and should be
/// called once per thread before any code that uses `sb_randn()`.
///
/// # Parameters
/// - `seed`: `uint32_t` containing the seed value
///
/// # Returns
/// No return value
/// 
/// # Examples
/// ```
/// #include <stdio.h>
/// #include <stdlib.h>
/// #include "sb_utility.h"
/// 
/// int main(void) {
///   sb_srandn(142857);
///
///   printf("Five normally distributed random numbers:\n");
///   for (size_t a = 0; a < 5; ++a) {
///     printf("% .4f\n", sb_randn());
///   }
/// }
/// ```
void sb_srandn(uint32_t seed) {
  randn_state = seed;
}

/// Basic normally distributed random number generator that applies the
/// Box--Muller transform to xorshift values. The internal state should be set
/// by `sb_srandn()` before this function is called.
///
/// # Parameters
/// No parameters
///
/// # Returns
/// A normally distributed random number (mean zero, standard deviation one)
/// 
/// # Examples
/// ```
/// #include <stdio.h>
/// #include <stdlib.h>
/// #include "sb_utility.h"
/// 
/// int main(void) {
///   sb_srandn(142857);
///
///   printf("Five normally distributed random numbers:\n");
///   for (size_t a = 0; a < 5; ++a) {
///     printf("% .4f\n", sb_randn());
///   }
/// }
/// ```
double sb_randn(void) {
  // cast is higher precedence than addition
  static const double UINT32_MAX_P1 = (double) UINT32_MAX + 1.;

  // only need to generate half the time
  static _Thread_local double z;
  static _Thread_local bool stored = false;
  if (stored) {
    stored = false;
    return z;
  }

  double u;

  randn_state ^= (randn_state << 13);
  randn_state ^= (randn_state >> 17);
  randn_state ^= (randn_state <<  5);

  // u \in (0., 1.)
  u = ((double) randn_state + 0.5) / UINT32_MAX_P1;
  double r = sqrt(-2. * log(u));

  randn_state ^= (randn_state << 13);
  randn_state ^= (randn_state >> 17);
  randn_state ^= (randn_state <<  5);

  // u \in (0., 1.)
  u = ((double) randn_state + 0.5) / UINT32_MAX_P1;
  double arg = 6.283185307179586 * u;

  // store value for subsequent call
  z = r * sin(arg);
  stored = true;

  return r * cos(arg);
}
