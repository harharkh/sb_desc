// Copyright 2018 Jeremy Mason
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! \file sb_matrix.h
//! Contains functions to manipulate matrices. Intended as a relatively less
//! painful wrapper around Level 2 and 3 CBLAS. Column-major order is likely
//! slightly faster than row-major order because of the underlying FORTRAN 
//! routines.

#pragma once

#include <stddef.h>     // size_t
#include <stdio.h>      // FILE
#include <stdlib.h>     // abort
#include "sb_structs.h" // sb_mat
#include "sb_utility.h" // SB_CHK_ERR
#include "safety.h"

/// Calls `sb_mat_free()` for every matrix in the argument list.
///
/// # Parameters
/// - `...`: pointer to matrices to be freed
///
/// # Returns
/// No return value.
/// 
/// # Examples
/// ```
/// #include "sb_matrix.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2., 8.};
///   sb_mat * A = sb_mat_of_arr(a,     3, 3);
///   sb_mat * B = sb_mat_of_arr(a + 1, 3, 3);
/// 
///   sb_mat_print(A, "A: ", "%g");
///   sb_mat_print(B, "B: ", "%g");
///
///   SB_MAT_FREE_ALL(A, B);
///   // Pointers to `A` and `B` now invalid
/// }
/// ```
#define SB_MAT_FREE_ALL(...) do {                        \
  void * sentinel = (int []){0};                         \
  sb_mat ** list = (sb_mat * []){__VA_ARGS__, sentinel}; \
  for (size_t a = 0; list[a] != sentinel; ++a) {         \
    sb_mat_free(list[a]);                                \
  }                                                      \
} while (0)

// Allocate

sb_mat * sb_mat_malloc(size_t n_rows, size_t n_cols);
sb_mat * sb_mat_calloc(size_t n_rows, size_t n_cols);
sb_mat * sb_mat_of_arr(const double * a, size_t n_rows, size_t n_cols);
sb_mat * sb_mat_clone (const sb_mat * A);
void     sb_mat_free  (sb_mat * A);

// Initialize

sb_mat * sb_mat_set_zero(sb_mat * A);
sb_mat * sb_mat_set_all (sb_mat * A, double x);
sb_mat * sb_mat_set_ident(sb_mat * A);

sb_vec * sb_mat_get_row (sb_vec * restrict v, const sb_mat * restrict A, size_t i);
sb_vec * sb_mat_get_col (sb_vec * restrict v, const sb_mat * restrict A, size_t j);
sb_vec * sb_mat_get_diag(sb_vec * restrict v, const sb_mat * restrict A);

sb_mat * sb_mat_set_row (sb_mat * restrict A, size_t i, const sb_vec * restrict v);
sb_mat * sb_mat_set_col (sb_mat * restrict A, size_t j, const sb_vec * restrict v);
sb_mat * sb_mat_set_diag(sb_mat * restrict A, const sb_vec * restrict v);

// Copy and exchange

sb_mat * sb_mat_memcpy(sb_mat * restrict dest, const sb_mat * restrict src);
sb_mat * sb_mat_subcpy(
    sb_mat * restrict A,
    size_t i,
    const double * restrict a,
    size_t n);

void     sb_mat_swap(sb_mat * restrict A, sb_mat * restrict B);
sb_mat * sb_mat_swap_row(sb_mat * A, size_t i, size_t j);
sb_mat * sb_mat_swap_col(sb_mat * A, size_t i, size_t j);

// Read and write

int sb_mat_fwrite (FILE * stream, const sb_mat * A);
int sb_mat_fprintf(FILE * stream, const sb_mat * A, const char * format);
int sb_mat_print  (const sb_mat * A, const char * str, const char * format);

sb_mat * sb_mat_fread  (FILE * stream);
sb_mat * sb_mat_fscanf (FILE * stream);

// Operations

sb_mat * sb_mat_abs(sb_mat * A);
sb_mat * sb_mat_exp(sb_mat * A);
sb_mat * sb_mat_log(sb_mat * A);
sb_mat * sb_mat_pow(sb_mat * A, double x);
sb_mat * sb_mat_sqrt(sb_mat * A);

sb_mat * sb_mat_sadd(sb_mat * A, double x);
sb_mat * sb_mat_smul(sb_mat * A, double x);

sb_mat * sb_mat_vadd(sb_mat * restrict A, const sb_vec * restrict v, char dir);
sb_mat * sb_mat_vsub(sb_mat * restrict A, const sb_vec * restrict v, char dir);
sb_mat * sb_mat_vmul(sb_mat * restrict A, const sb_vec * restrict v, char dir);
sb_mat * sb_mat_vdiv(sb_mat * restrict A, const sb_vec * restrict v, char dir);

sb_mat * sb_mat_padd(sb_mat * restrict A, const sb_mat * restrict B);
sb_mat * sb_mat_psub(sb_mat * restrict A, const sb_mat * restrict B);
sb_mat * sb_mat_pmul(sb_mat * restrict A, const sb_mat * restrict B);
sb_mat * sb_mat_pdiv(sb_mat * restrict A, const sb_mat * restrict B);

sb_vec * sb_mat_mv_mul(
    sb_vec * restrict v,
    const sb_mat * restrict A,
    const sb_vec * restrict w,
    char op);

sb_vec * sb_mat_vm_mul(
    sb_vec * restrict v,
    const sb_vec * restrict w,
    const sb_mat * restrict A,
    char op);

sb_mat * sb_mat_mm_mul(
    sb_mat * restrict A,
    const sb_mat * restrict B,
    const sb_mat * restrict C,
    const char * restrict ops);

sb_vec * sb_mat_sum(sb_vec * restrict v, const sb_mat * restrict A, const char dir);

// Transformations

sb_mat * sb_vec_to_mat (sb_vec ** v, size_t n_rows, size_t n_cols);
sb_vec * sb_mat_to_vec (sb_mat ** A);
sb_mat * sb_mat_reshape(sb_mat * A, size_t n_rows, size_t n_cols);
sb_mat * sb_mat_trans(sb_mat * A);

// Properties

int sb_mat_is_equal (const sb_mat * restrict A, const sb_mat * restrict B);
int sb_mat_is_zero  (const sb_mat * A);
int sb_mat_is_pos   (const sb_mat * A);
int sb_mat_is_neg   (const sb_mat * A);
int sb_mat_is_nonneg(const sb_mat * A);
int sb_mat_is_finite(const sb_mat * A);

double sb_mat_max(const sb_mat * A);
double sb_mat_min(const sb_mat * A);
double sb_mat_abs_max(const sb_mat * A);

size_t sb_mat_max_index(const sb_mat * A);
size_t sb_mat_min_index(const sb_mat * A);
size_t sb_mat_abs_max_index(const sb_mat * v);

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
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
///   sb_mat_print(A, "A: ", "%g");
///   
///   // Print the element
///   printf("A(1, 2): %g\n", sb_mat_get(A, 1, 2));
///   
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
inline double sb_mat_get(const sb_mat * A, size_t i, size_t j) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_get: A cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(i >= A->n_rows, abort(), "sb_mat_get: row index out of bounds");
  SB_CHK_ERR(j >= A->n_cols, abort(), "sb_mat_get: column index out of bounds");
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
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
///   sb_mat_print(A, "A: ", "%g");
///   
///   // Update the element and print
///   printf("A(1, 2) before: %g\n", sb_mat_get(A, 1, 2));
///   sb_mat_set(A, 1, 2, 8.);
///   printf("A(1, 2) after:  %g\n", sb_mat_get(A, 1, 2));
///   
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
inline void sb_mat_set(sb_mat * A, size_t i, size_t j, double x) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_set: A cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(i >= A->n_rows, abort(), "sb_mat_get: row index out of bounds");
  SB_CHK_ERR(j >= A->n_cols, abort(), "sb_mat_get: column index out of bounds");
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
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
///   sb_mat_print(A, "A: ", "%g");
///   
///   // Update the element and print
///   printf("A(1, 2) before: %g\n", sb_mat_get(A, 1, 2));
///   *sb_mat_ptr(A, 1, 2) = 8.;
///   printf("A(1, 2) before: %g\n", sb_mat_get(A, 1, 2));
///   
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
inline double * sb_mat_ptr(sb_mat * A, size_t i, size_t j) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_set: A cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(i >= A->n_rows, abort(), "sb_mat_get: row index out of bounds");
  SB_CHK_ERR(j >= A->n_cols, abort(), "sb_mat_get: column index out of bounds");
#endif
  return A->data + j * A->n_rows + i;
}
