// Copyright 2018 Jeremy Mason
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! \file gw_matrix.h
//! Contains functions to manipulate matrices. Intended as a relatively less
//! painful wrapper around Level 2 and 3 CBLAS. Column-major order is likely
//! slightly faster than row-major order because of the underlying FORTRAN 
//! routines.

#pragma once

#include <stddef.h>     // size_t
#include <stdio.h>      // FILE
#include <stdlib.h>     // abort
#include "gw_structs.h" // gw_mat
#include "gw_utility.h" // GW_CHK_ERR
#include "safety.h"

/// Calls `gw_mat_free()` for every matrix in the argument list.
///
/// # Parameters
/// - `...`: pointer to matrices to be freed
///
/// # Returns
/// No return value.
/// 
/// # Examples
/// ```
/// #include "gw_matrix.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2., 8.};
///   gw_mat * A = gw_mat_of_arr(a,     3, 3);
///   gw_mat * B = gw_mat_of_arr(a + 1, 3, 3);
/// 
///   gw_mat_print(A, "A: ", "%g");
///   gw_mat_print(B, "B: ", "%g");
///
///   GW_MAT_FREE_ALL(A, B);
///   // Pointers to `A` and `B` now invalid
/// }
/// ```
#define GW_MAT_FREE_ALL(...) do {                        \
  void * sentinel = (int []){0};                         \
  gw_mat ** list = (gw_mat * []){__VA_ARGS__, sentinel}; \
  for (size_t a = 0; list[a] != sentinel; ++a) {         \
    gw_mat_free(list[a]);                                \
  }                                                      \
} while (0)

// Allocate

gw_mat * gw_mat_malloc(size_t n_rows, size_t n_cols);
gw_mat * gw_mat_calloc(size_t n_rows, size_t n_cols);
gw_mat * gw_mat_of_arr(const double * a, size_t n_rows, size_t n_cols);
gw_mat * gw_mat_clone (const gw_mat * A);
void     gw_mat_free  (gw_mat * A);

// Initialize

gw_mat * gw_mat_set_zero(gw_mat * A);
gw_mat * gw_mat_set_all (gw_mat * A, double x);
gw_mat * gw_mat_set_ident(gw_mat * A);

gw_vec * gw_mat_get_row (gw_vec * restrict v, const gw_mat * restrict A, size_t i);
gw_vec * gw_mat_get_col (gw_vec * restrict v, const gw_mat * restrict A, size_t j);
gw_vec * gw_mat_get_diag(gw_vec * restrict v, const gw_mat * restrict A);

gw_mat * gw_mat_set_row (gw_mat * restrict A, size_t i, const gw_vec * restrict v);
gw_mat * gw_mat_set_col (gw_mat * restrict A, size_t j, const gw_vec * restrict v);
gw_mat * gw_mat_set_diag(gw_mat * restrict A, const gw_vec * restrict v);

// Copy and exchange

gw_mat * gw_mat_memcpy(gw_mat * restrict dest, const gw_mat * restrict src);
gw_mat * gw_mat_subcpy(
    gw_mat * restrict A,
    size_t i,
    const double * restrict a,
    size_t n);

void     gw_mat_swap(gw_mat * restrict A, gw_mat * restrict B);
gw_mat * gw_mat_swap_row(gw_mat * A, size_t i, size_t j);
gw_mat * gw_mat_swap_col(gw_mat * A, size_t i, size_t j);

// Read and write

int gw_mat_fwrite (FILE * stream, const gw_mat * A);
int gw_mat_fprintf(FILE * stream, const gw_mat * A, const char * format);
int gw_mat_print  (const gw_mat * A, const char * str, const char * format);

gw_mat * gw_mat_fread  (FILE * stream);
gw_mat * gw_mat_fscanf (FILE * stream);

// Operations

gw_mat * gw_mat_abs(gw_mat * A);
gw_mat * gw_mat_exp(gw_mat * A);
gw_mat * gw_mat_log(gw_mat * A);
gw_mat * gw_mat_pow(gw_mat * A, double x);
gw_mat * gw_mat_sqrt(gw_mat * A);

gw_mat * gw_mat_sadd(gw_mat * A, double x);
gw_mat * gw_mat_smul(gw_mat * A, double x);

gw_mat * gw_mat_vadd(gw_mat * restrict A, const gw_vec * restrict v, char dir);
gw_mat * gw_mat_vsub(gw_mat * restrict A, const gw_vec * restrict v, char dir);
gw_mat * gw_mat_vmul(gw_mat * restrict A, const gw_vec * restrict v, char dir);
gw_mat * gw_mat_vdiv(gw_mat * restrict A, const gw_vec * restrict v, char dir);

gw_mat * gw_mat_padd(gw_mat * restrict A, const gw_mat * restrict B);
gw_mat * gw_mat_psub(gw_mat * restrict A, const gw_mat * restrict B);
gw_mat * gw_mat_pmul(gw_mat * restrict A, const gw_mat * restrict B);
gw_mat * gw_mat_pdiv(gw_mat * restrict A, const gw_mat * restrict B);

gw_vec * gw_mat_mv_mul(
    gw_vec * restrict v,
    const gw_mat * restrict A,
    const gw_vec * restrict w,
    char op);

gw_vec * gw_mat_vm_mul(
    gw_vec * restrict v,
    const gw_vec * restrict w,
    const gw_mat * restrict A,
    char op);

gw_mat * gw_mat_mm_mul(
    gw_mat * restrict A,
    const gw_mat * restrict B,
    const gw_mat * restrict C,
    const char * restrict ops);

gw_vec * gw_mat_sum(gw_vec * restrict v, const gw_mat * restrict A, const char dir);

// Transformations

gw_mat * gw_vec_to_mat (gw_vec ** v, size_t n_rows, size_t n_cols);
gw_vec * gw_mat_to_vec (gw_mat ** A);
gw_mat * gw_mat_reshape(gw_mat * A, size_t n_rows, size_t n_cols);
gw_mat * gw_mat_trans(gw_mat * A);

// Properties

int gw_mat_is_equal (const gw_mat * restrict A, const gw_mat * restrict B);
int gw_mat_is_zero  (const gw_mat * A);
int gw_mat_is_pos   (const gw_mat * A);
int gw_mat_is_neg   (const gw_mat * A);
int gw_mat_is_nonneg(const gw_mat * A);
int gw_mat_is_finite(const gw_mat * A);

double gw_mat_max(const gw_mat * A);
double gw_mat_min(const gw_mat * A);
double gw_mat_abs_max(const gw_mat * A);

size_t gw_mat_max_index(const gw_mat * A);
size_t gw_mat_min_index(const gw_mat * A);
size_t gw_mat_abs_max_index(const gw_mat * v);

// Indexing

/// Gets the `(i, j)`th element of the matrix.
///
/// # Parameters
/// - `A`: const pointer to the matrix
/// - `i`: row index of the element
/// - `j`: column index of the element
///
/// # Returns
/// The value of the `(i, j)`th element
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
/// - `SAFE_LENGTH`: `i` and `j` are valid indices
/// 
/// # Examples
/// ```
/// #include <stdio.h>
/// #include "gw_matrix.h"
/// #include "gw_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   gw_mat * A = gw_mat_of_arr(a, 3, 3);
///   gw_mat_print(A, "A: ", "%g");
///   
///   // Print the element
///   printf("A(1, 2): %g\n", gw_mat_get(A, 1, 2));
///   
///   GW_MAT_FREE_ALL(A);
/// }
/// ```
inline double gw_mat_get(const gw_mat * A, size_t i, size_t j) {
#ifdef SAFE_MEMORY
  GW_CHK_ERR(!A, abort(), "gw_mat_get: A cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  GW_CHK_ERR(i >= A->n_rows, abort(), "gw_mat_get: row index out of bounds");
  GW_CHK_ERR(j >= A->n_cols, abort(), "gw_mat_get: column index out of bounds");
#endif
  return A->data[j * A->n_rows + i];
}

/// Sets the `(i, j)`th element of the matrix.
///
/// # Parameters
/// - `A`: pointer to the matrix
/// - `i`: row index of the element
/// - `j`: column index of the element
/// - `x`: new value of the element
///
/// # Returns
/// No return value
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
/// - `SAFE_LENGTH`: `i` and `j` are valid indices
/// 
/// # Examples
/// ```
/// #include <stdio.h>
/// #include "gw_matrix.h"
/// #include "gw_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   gw_mat * A = gw_mat_of_arr(a, 3, 3);
///   gw_mat_print(A, "A: ", "%g");
///   
///   // Update the element and print
///   printf("A(1, 2) before: %g\n", gw_mat_get(A, 1, 2));
///   gw_mat_set(A, 1, 2, 8.);
///   printf("A(1, 2) after:  %g\n", gw_mat_get(A, 1, 2));
///   
///   GW_MAT_FREE_ALL(A);
/// }
/// ```
inline void gw_mat_set(gw_mat * A, size_t i, size_t j, double x) {
#ifdef SAFE_MEMORY
  GW_CHK_ERR(!A, abort(), "gw_mat_set: A cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  GW_CHK_ERR(i >= A->n_rows, abort(), "gw_mat_get: row index out of bounds");
  GW_CHK_ERR(j >= A->n_cols, abort(), "gw_mat_get: column index out of bounds");
#endif
  A->data[j * A->n_rows + i] = x;
}

/// Returns a pointer to the `(i, j)`th element of the matrix.
///
/// # Parameters
/// - `A`: pointer to the matrix
/// - `i`: row index of the element
/// - `i`: column index of the element
///
/// # Returns
/// A `double` pointer to the `(i, j)`th element
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
/// - `SAFE_LENGTH`: `i` and `j` are valid indices
/// 
/// # Examples
/// ```
/// #include <stdio.h>
/// #include "gw_matrix.h"
/// #include "gw_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   gw_mat * A = gw_mat_of_arr(a, 3, 3);
///   gw_mat_print(A, "A: ", "%g");
///   
///   // Update the element and print
///   printf("A(1, 2) before: %g\n", gw_mat_get(A, 1, 2));
///   *gw_mat_ptr(A, 1, 2) = 8.;
///   printf("A(1, 2) before: %g\n", gw_mat_get(A, 1, 2));
///   
///   GW_MAT_FREE_ALL(A);
/// }
/// ```
inline double * gw_mat_ptr(gw_mat * A, size_t i, size_t j) {
#ifdef SAFE_MEMORY
  GW_CHK_ERR(!A, abort(), "gw_mat_set: A cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  GW_CHK_ERR(i >= A->n_rows, abort(), "gw_mat_get: row index out of bounds");
  GW_CHK_ERR(j >= A->n_cols, abort(), "gw_mat_get: column index out of bounds");
#endif
  return A->data + j * A->n_rows + i;
}
