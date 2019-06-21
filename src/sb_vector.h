// Copyright 2018 Jeremy Mason
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! \file gw_vector.h
//! Contains functions to manipulate vectors. Intended as a relatively less
//! painful wrapper around Level 1 CBLAS.

#pragma once

#include <stddef.h>     // size_t
#include <stdio.h>      // FILE
#include <stdlib.h>     // abort
#include "gw_structs.h" // gw_mat
#include "gw_utility.h" // GW_CHK_ERR
#include "safety.h"

/// Calls `gw_vec_free()` for every vector in the argument list.
///
/// # Parameters
/// - `...`: pointer to vectors to be freed
///
/// # Returns
/// No return value.
/// 
/// # Examples
/// ```
/// #include "gw_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7.};
///   gw_vec * v = gw_vec_of_arr(a,     3, 'r');
///   gw_vec * w = gw_vec_of_arr(a + 3, 3, 'r');
/// 
///   gw_vec_print(v, "v: ", "%g");
///   gw_vec_print(w, "w: ", "%g");
///
///   GW_VEC_FREE_ALL(v, w);
///   // Pointers to `v` and `w` now invalid
/// }
/// ```
#define GW_VEC_FREE_ALL(...) do {                        \
  void * sentinel = (int []){0};                         \
  gw_vec ** list = (gw_vec * []){__VA_ARGS__, sentinel}; \
  for (size_t a = 0; list[a] != sentinel; ++a) {         \
    gw_vec_free(list[a]);                                \
  }                                                      \
} while (0)

// Allocate

gw_vec * gw_vec_malloc(size_t n_elem, char layout);
gw_vec * gw_vec_calloc(size_t n_elem, char layout);
gw_vec * gw_vec_of_arr(const double * a, size_t n_elem, char layout);
gw_vec * gw_vec_clone (const gw_vec * v);
gw_vec * gw_vec_linear(double begin, double end, size_t n_elem);
void     gw_vec_free  (gw_vec * v);

// Initialize

gw_vec * gw_vec_set_zero (gw_vec * v);
gw_vec * gw_vec_set_all  (gw_vec * v, double x);
gw_vec * gw_vec_set_basis(gw_vec * v, size_t i);

// Copy and exchange

gw_vec * gw_vec_memcpy(gw_vec * restrict dest, const gw_vec * restrict src);
gw_vec * gw_vec_subcpy(
    gw_vec * restrict v,
    size_t i,
    const double * restrict a,
    size_t n);

void gw_vec_swap(gw_vec * restrict v, gw_vec * restrict w);
void gw_vec_swap_elems(gw_vec * v, size_t i, size_t j);

// Read and write

int gw_vec_fwrite (FILE * stream, const gw_vec * v);
int gw_vec_fprintf(FILE * stream, const gw_vec * v, const char * format);
int gw_vec_print  (const gw_vec * v, const char * str, const char * format);

gw_vec * gw_vec_fread (FILE * stream);
gw_vec * gw_vec_fscanf(FILE * stream);

// Operations

gw_vec * gw_vec_abs(gw_vec * v);
gw_vec * gw_vec_exp(gw_vec * v);
gw_vec * gw_vec_log(gw_vec * v);
gw_vec * gw_vec_pow(gw_vec * v, double x);
gw_vec * gw_vec_sqrt(gw_vec * v);

gw_vec * gw_vec_sadd(gw_vec * v, double x);
gw_vec * gw_vec_smul(gw_vec * v, double x);

gw_vec * gw_vec_padd(gw_vec * restrict v, const gw_vec * restrict w);
gw_vec * gw_vec_psub(gw_vec * restrict v, const gw_vec * restrict w);
gw_vec * gw_vec_pmul(gw_vec * restrict v, const gw_vec * restrict w);
gw_vec * gw_vec_pdiv(gw_vec * restrict v, const gw_vec * restrict w);

gw_vec * gw_vec_rxay(gw_vec * r, double x, double y);
gw_vec * gw_vec_raxv(gw_vec * restrict r, double x, const gw_vec * restrict v);
gw_vec * gw_vec_rvsw(
    gw_vec * restrict r,
    const gw_vec * restrict v,
    const gw_vec * restrict w);

gw_vec * gw_vec_rdpvdwp(
    gw_vec * restrict r,
    const gw_vec * restrict v,
    const gw_vec * restrict w);

double gw_vec_sum (const gw_vec * v);
double gw_vec_norm(const gw_vec * v);
double gw_vec_dot (const gw_vec * restrict v, const gw_vec * restrict w);

gw_mat * gw_vec_outer(
    gw_mat * restrict A,
    const gw_vec * restrict v,
    const gw_vec * restrict w);

// Transformations

gw_vec * gw_vec_reverse (gw_vec * v);
gw_vec * gw_vec_sort_inc(gw_vec * v);
gw_vec * gw_vec_sort_dec(gw_vec * v);
gw_vec * gw_vec_trans   (gw_vec * v);

// Properties

int gw_vec_is_equal (const gw_vec * restrict v, const gw_vec * restrict w);
int gw_vec_is_zero  (const gw_vec * v);
int gw_vec_is_pos   (const gw_vec * v);
int gw_vec_is_neg   (const gw_vec * v);
int gw_vec_is_nonneg(const gw_vec * v);
int gw_vec_is_finite(const gw_vec * v);

double gw_vec_max(const gw_vec * v);
double gw_vec_min(const gw_vec * v);
double gw_vec_abs_max(const gw_vec * v);

size_t gw_vec_max_index(const gw_vec * v);
size_t gw_vec_min_index(const gw_vec * v);
size_t gw_vec_abs_max_index(const gw_vec * v);

// Indexing

/// Gets the `i`th element of the vector.
///
/// # Parameters
/// - `v`: const pointer to the vector
/// - `i`: index of the element
///
/// # Returns
/// The value of the `i`th element
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
/// - `SAFE_LENGTH`: `i` is a valid index
/// 
/// # Examples
/// ```
/// #include <stdio.h>
/// #include "gw_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   gw_vec * v = gw_vec_of_arr(a, 3, 'r');
///   
///   // Print the element
///   printf("v(1): %g\n", gw_vec_get(v, 1));
///   
///   GW_VEC_FREE_ALL(v);
/// }
/// ```
inline double gw_vec_get(const gw_vec * v, size_t i) {
#ifdef SAFE_MEMORY
  GW_CHK_ERR(!v, abort(), "gw_vec_get: v cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  GW_CHK_ERR(i >= v->n_elem, abort(), "gw_vec_get: index out of bounds");
#endif
  return v->data[i];
}

/// Sets the `i`th element of the vector.
///
/// # Parameters
/// - `v`: pointer to the vector
/// - `i`: index of the element
/// - `x`: new value of the element
///
/// # Returns
/// No return value
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
/// - `SAFE_LENGTH`: `i` is a valid index
/// 
/// # Examples
/// ```
/// #include <stdio.h>
/// #include "gw_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   gw_vec * v = gw_vec_of_arr(a, 3, 'r');
///   
///   // Update the element and print
///   printf("v(1) before: %g\n", gw_vec_get(v, 1));
///   gw_vec_set(v, 1, 8.);
///   printf("v(1) after:  %g\n", gw_vec_get(v, 1));
///   
///   GW_VEC_FREE_ALL(v);
/// }
/// ```
inline void gw_vec_set(gw_vec * v, size_t i, double x) {
#ifdef SAFE_MEMORY
  GW_CHK_ERR(!v, abort(), "gw_vec_set: v cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  GW_CHK_ERR(i >= v->n_elem, abort(), "gw_vec_set: index out of bounds");
#endif
  v->data[i] = x;
}

/// Returns a pointer to the `i`th element of the vector.
///
/// # Parameters
/// - `v`: pointer to the vector
/// - `i`: index of the element
///
/// # Returns
/// A `double` pointer to the `i`th element
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
/// - `SAFE_LENGTH`: `i` is a valid index
/// 
/// # Examples
/// ```
/// #include <stdio.h>
/// #include "gw_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   gw_vec * v = gw_vec_of_arr(a, 3, 'r');
///   
///   // Update the element and print
///   printf("v(1) before: %g\n", gw_vec_get(v, 1));
///   *gw_vec_ptr(v, 1) = 8.;
///   printf("v(1) after:  %g\n", gw_vec_get(v, 1));
///   
///   GW_VEC_FREE_ALL(v);
/// }
/// ```
inline double * gw_vec_ptr(gw_vec * v, size_t i) {
#ifdef SAFE_MEMORY
  GW_CHK_ERR(!v, abort(), "gw_vec_ptr: v cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  GW_CHK_ERR(i >= v->n_elem, abort(), "gw_vec_ptr: index out of bounds");
#endif
  return v->data + i;
}
