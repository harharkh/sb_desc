// Copyright 2018 Jeremy Mason
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! \file sb_vector.h
//! Contains functions to manipulate vectors. Intended as a relatively less
//! painful wrapper around Level 1 CBLAS.

#pragma once

#include <stddef.h>     // size_t
#include <stdio.h>      // FILE
#include <stdlib.h>     // abort
#include "sb_structs.h" // sb_mat
#include "sb_utility.h" // SB_CHK_ERR
#include "safety.h"

/// Calls `sb_vec_free()` for every vector in the argument list.
///
/// # Parameters
/// - `...`: pointer to vectors to be freed
///
/// # Returns
/// No return value.
/// 
/// # Examples
/// ```
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7.};
///   sb_vec * v = sb_vec_of_arr(a,     3, 'r');
///   sb_vec * w = sb_vec_of_arr(a + 3, 3, 'r');
/// 
///   sb_vec_print(v, "v: ", "%g");
///   sb_vec_print(w, "w: ", "%g");
///
///   SB_VEC_FREE_ALL(v, w);
///   // Pointers to `v` and `w` now invalid
/// }
/// ```
#define SB_VEC_FREE_ALL(...) do {                        \
  void * sentinel = (int []){0};                         \
  sb_vec ** list = (sb_vec * []){__VA_ARGS__, sentinel}; \
  for (size_t a = 0; list[a] != sentinel; ++a) {         \
    sb_vec_free(list[a]);                                \
  }                                                      \
} while (0)

// Allocate

sb_vec * sb_vec_malloc(size_t n_elem, char layout);
sb_vec * sb_vec_calloc(size_t n_elem, char layout);
sb_vec * sb_vec_of_arr(const double * a, size_t n_elem, char layout);
sb_vec * sb_vec_clone (const sb_vec * v);
sb_vec * sb_vec_linear(double begin, double end, size_t n_elem);
void     sb_vec_free  (sb_vec * v);

// Initialize

sb_vec * sb_vec_set_zero (sb_vec * v);
sb_vec * sb_vec_set_all  (sb_vec * v, double x);
sb_vec * sb_vec_set_basis(sb_vec * v, size_t i);

// Copy and exchange

sb_vec * sb_vec_memcpy(sb_vec * restrict dest, const sb_vec * restrict src);
sb_vec * sb_vec_subcpy(
    sb_vec * restrict v,
    size_t i,
    const double * restrict a,
    size_t n);

void sb_vec_swap(sb_vec * restrict v, sb_vec * restrict w);
void sb_vec_swap_elems(sb_vec * v, size_t i, size_t j);

// Read and write

int sb_vec_fwrite (FILE * stream, const sb_vec * v);
int sb_vec_fprintf(FILE * stream, const sb_vec * v, const char * format);
int sb_vec_print  (const sb_vec * v, const char * str, const char * format);

sb_vec * sb_vec_fread (FILE * stream);
sb_vec * sb_vec_fscanf(FILE * stream);

// Operations

sb_vec * sb_vec_abs(sb_vec * v);
sb_vec * sb_vec_exp(sb_vec * v);
sb_vec * sb_vec_log(sb_vec * v);
sb_vec * sb_vec_pow(sb_vec * v, double x);
sb_vec * sb_vec_sqrt(sb_vec * v);

sb_vec * sb_vec_sadd(sb_vec * v, double x);
sb_vec * sb_vec_smul(sb_vec * v, double x);

sb_vec * sb_vec_padd(sb_vec * restrict v, const sb_vec * restrict w);
sb_vec * sb_vec_psub(sb_vec * restrict v, const sb_vec * restrict w);
sb_vec * sb_vec_pmul(sb_vec * restrict v, const sb_vec * restrict w);
sb_vec * sb_vec_pdiv(sb_vec * restrict v, const sb_vec * restrict w);

sb_vec * sb_vec_rxay(sb_vec * r, double x, double y);
sb_vec * sb_vec_raxv(sb_vec * restrict r, double x, const sb_vec * restrict v);
sb_vec * sb_vec_rvsw(
    sb_vec * restrict r,
    const sb_vec * restrict v,
    const sb_vec * restrict w);

double sb_vec_sum (const sb_vec * v);
double sb_vec_norm(const sb_vec * v);
double sb_vec_dot (const sb_vec * restrict v, const sb_vec * restrict w);

sb_mat * sb_vec_outer(
    sb_mat * restrict A,
    const sb_vec * restrict v,
    const sb_vec * restrict w);

// Transformations

sb_vec * sb_vec_reverse (sb_vec * v);
sb_vec * sb_vec_sort_inc(sb_vec * v);
sb_vec * sb_vec_sort_dec(sb_vec * v);
sb_vec * sb_vec_trans   (sb_vec * v);

// Properties

int sb_vec_is_equal (const sb_vec * restrict v, const sb_vec * restrict w);
int sb_vec_is_zero  (const sb_vec * v);
int sb_vec_is_pos   (const sb_vec * v);
int sb_vec_is_neg   (const sb_vec * v);
int sb_vec_is_nonneg(const sb_vec * v);
int sb_vec_is_finite(const sb_vec * v);

double sb_vec_max(const sb_vec * v);
double sb_vec_min(const sb_vec * v);
double sb_vec_abs_max(const sb_vec * v);

size_t sb_vec_max_index(const sb_vec * v);
size_t sb_vec_min_index(const sb_vec * v);
size_t sb_vec_abs_max_index(const sb_vec * v);

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
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
///   
///   // Print the element
///   printf("v(1): %g\n", sb_vec_get(v, 1));
///   
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
inline double sb_vec_get(const sb_vec * v, size_t i) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_get: v cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(i >= v->n_elem, abort(), "sb_vec_get: index out of bounds");
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
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
///   
///   // Update the element and print
///   printf("v(1) before: %g\n", sb_vec_get(v, 1));
///   sb_vec_set(v, 1, 8.);
///   printf("v(1) after:  %g\n", sb_vec_get(v, 1));
///   
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
inline void sb_vec_set(sb_vec * v, size_t i, double x) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_set: v cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(i >= v->n_elem, abort(), "sb_vec_set: index out of bounds");
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
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
///   
///   // Update the element and print
///   printf("v(1) before: %g\n", sb_vec_get(v, 1));
///   *sb_vec_ptr(v, 1) = 8.;
///   printf("v(1) after:  %g\n", sb_vec_get(v, 1));
///   
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
inline double * sb_vec_ptr(sb_vec * v, size_t i) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_ptr: v cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(i >= v->n_elem, abort(), "sb_vec_ptr: index out of bounds");
#endif
  return v->data + i;
}
