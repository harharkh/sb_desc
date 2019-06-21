// Copyright 2018 Jeremy Mason
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! \file gw_utility.h
//! Contains type definitions, macros, and utility functions used by the gw_dist
//! project.

#pragma once

#include <stdint.h> // uint32_t
#include <stdio.h>  // FILE
#include <time.h>   // clock_t

/// Function macro for error checking. When the assertion is true, prints a
/// message to the error log set by `set_error_log` in `utility.c` (defaults to
/// `stderr`) and executes an error action.
///
/// # Parameters
/// - `assertion`: expression to be evaluated, often a return value
/// - `error_action`: statement to execute when assertion is true
/// - `...`: format string and any arguments to be sent to `fprintf`
///
/// # Extern Parameters
/// - `error_log`: `FILE` pointer to the error log, set by `set_error_log`,
///                defaults to `stderr`
///
/// # Returns
/// No return value
/// 
/// # Examples
/// ```
/// #include <stdio.h>
/// #include <stdlib.h>
/// #include "gw_utility.h"
///
/// int main(void) {
///   double numer = 1.;
///   double denom = 0.;
///
///   GW_CHK_ERR(denom == 0., exit(1), "denominator must be nonzero");
///   printf("%g / %g = %g\n", numer, denom, numer / denom);
/// }
/// ```
#define GW_CHK_ERR(assertion, error_action, ...) do {           \
  if (assertion) {                                              \
    fprintf(gw_error_log ? gw_error_log : stderr, __VA_ARGS__); \
    fprintf(gw_error_log ? gw_error_log : stderr, "\n");        \
    error_action;                                               \
  }                                                             \
} while (0)

/// Calls `free` for every vector in the argument list.
///
/// # Parameters
/// - `...`: pointers to be freed
///
/// # Returns
/// No return value.
/// 
/// # Examples
/// ```
/// #include <stdio.h>
/// #include <stdlib.h>
/// #include <string.h>
///
/// int main(void) {
///   int    a[] = {1,  4,  2,  8,  5,  7}; 
///   double b[] = {1., 4., 2., 8., 5., 7.};
/// 
///   int    * c = malloc(6 * sizeof(int));
///   double * d = malloc(6 * sizeof(double));
/// 
///   memcpy(c, a, 6 * sizeof(int));
///   memcpy(d, b, 6 * sizeof(double));
/// 
///   printf("Second elements are: %d, %g\n", c[2], d[2]);
/// 
///   GW_FREE_ALL(c, d);
/// }
/// ```
#define GW_FREE_ALL(...) do {                           \
  void * sentinel = (int []){0};                     \
  void ** list = (void * []){__VA_ARGS__, sentinel}; \
  for (size_t a = 0; list[a] != sentinel; ++a) {     \
    free(list[a]);                                   \
  }                                                  \
} while (0)

/// Returns the maximum of the two arguments. Arguments should be comparable
/// by the `>` operator and not have any side effects.
///
/// # Parameters
/// - `a`: first quantity
/// - `b`: second quantity
///
/// # Returns
/// The maximum of `a` or `b`
/// 
/// # Examples
/// ```
/// #include <stdio.h>
///
/// int main(void) {
///   int a = -2;
///   int b =  4;
///   printf("The maximum of %d and %d is: %d\n", a, b, GW_MAX(a, b));
/// }
/// ```
#define GW_MAX(a, b) (((a) > (b)) ? (a) : (b))

/// Returns the minimum of the two arguments. Arguments should be comparable
/// by the `<` operator and not have any side effects.
///
/// # Parameters
/// - `a`: first quantity
/// - `b`: second quantity
///
/// # Returns
/// The minimum of `a` or `b`
/// 
/// # Examples
/// ```
/// #include <stdio.h>
///
/// int main(void) {
///   int a = -2;
///   int b =  4;
///   printf("The minimum of %d and %d is: %d\n", a, b, GW_MIN(a, b));
/// }
/// ```
#define GW_MIN(a, b) (((a) < (b)) ? (a) : (b))

/// Squares the value of `a`. Expression used for the argument should not have
/// any side effects.
///
/// # Parameters
/// - `a`: variable
///
/// # Returns
/// The square of `a`
/// 
/// # Examples
/// ```
/// #include <assert.h>
/// #include "utility.h"
///
/// int main(void) {
///   int x = 1;
///   int y = 4;
///   
///   assert(GW_SQR(x) = 1);
///   assert(GW_SQR(y) = 16);
/// }
/// ```
#define GW_SQR(a) ((a) * (a))

/// Swaps the values of `a` and `b`. Expressions used for the arguments should
/// not have any side effects.
///
/// # Parameters
/// - `a`: first variable
/// - `b`: second variable
/// - `c`: scratch variable
///
/// # Returns
/// No return value
/// 
/// # Examples
/// ```
/// #include <assert.h>
/// #include "utility.h"
///
/// int main(void) {
///   int tmp;
/// 
///   int x = 1;
///   int y = 4;
///   
///   GW_SWAP(x, y, tmp);
///   assert(x = 4);
///   
///   int * px = &x;
///   int * py = &y;
///   
///   GW_SWAP(*px, *py, tmp);
///   assert(*px = 1);
/// }
/// ```
#define GW_SWAP(a, b, c) do { (c) = (a); (a) = (b); (b) = (c); } while (0)

/// Macro that starts a basic timer. Should be paired with `GW_TOC`. NOTE: Uses
/// thread-local storage. Do not call `GW_TIC` and `GW_TOC` in different
/// threads.
///
/// # Parameters
/// No parameters
///
/// # Extern Parameters
/// - `tic_time`: `clock_t` holding the time at `GW_TIC`
/// 
/// # Returns
/// No return value
/// 
/// # Examples
/// ```
/// #include <stddef.h>
/// #include "utility.h"
///
/// int main(void) {
///   GW_TIC;
///   
///   int tmp;
///   int x = 1;
///   int y = 4;
///
///   for (size_t a = 0; a < 1000000; ++a) {
///     int * px = &x;
///     int * py = &y;
///     GW_SWAP(*px, *py, tmp);
///   }
///
///   GW_TOC;
/// }
/// ```
#define GW_TIC do { gw_tic_time = clock(); } while (0)

/// Macro that stops a basic timer. Should be paired with `GW_TIC`. NOTE: Uses
/// thread-local storage. Do not call `GW_TIC` and `GW_TOC` in different
/// threads.
///
/// # Parameters
/// No parameters
///
/// # Returns
/// No return value
/// 
/// # Examples
/// ```
/// #include <stddef.h>
/// #include "utility.h"
///
/// int main(void) {
///   GW_TIC;
///   
///   int tmp;
///   int x = 1;
///   int y = 4;
///
///   for (size_t a = 0; a < 1000000; ++a) {
///     int * px = &x;
///     int * py = &y;
///     GW_SWAP(*px, *py, tmp);
///   }
///
///   GW_TOC;
/// }
/// ```
#define GW_TOC do {                                      \
  printf(                                                \
      "Time elapsed: %.2f ms\n",                         \
      1000. * (clock() - gw_tic_time) / CLOCKS_PER_SEC); \
} while (0)

extern FILE * gw_error_log;
extern _Thread_local clock_t gw_tic_time;

void gw_set_error_log(FILE * f);

void   gw_srandn(uint32_t seed);
double gw_randn (void);
