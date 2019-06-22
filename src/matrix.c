// Copyright 2018 Jeremy Mason
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! \file matrix.c
//! Contains functions to manipulate matrices. Intended as a relatively less
//! painful wrapper around Level 2 and 3 CBLAS. Column-major order is likely
//! slightly faster than row-major order because of the underlying FORTRAN 
//! routines.

#include <cblas.h>      // daxpy
#include <math.h>       // exp
#include <stdbool.h>    // bool
#include <stddef.h>     // size_t
#include <stdio.h>      // EOF
#include <stdlib.h>     // abort
#include <string.h>     // memcpy
#include "sb_structs.h" // sb_mat
#include "sb_matrix.h"
#include "sb_utility.h" // SB_CHK_ERR
#include "sb_vector.h"  // sb_vec_is_finite
#include "safety.h"

/// Constructs a matrix with the required capacity.
///
/// # Parameters
/// - `n_rows`: number of rows of the matrix
/// - `n_cols`: number of columns of the matrix
///
/// # Returns
/// A `sb_mat` pointer to the allocated matrix, or `NULL` if the allocation
/// fails
/// 
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   sb_mat * A = sb_mat_malloc(3, 3);
///
///   // Fill the matrix with some values and print
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat_subcpy(A, 0, a, 9);
///   sb_mat_print(A, "A: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
sb_mat * sb_mat_malloc(size_t n_rows, size_t n_cols) {
  sb_mat * out = malloc(sizeof(sb_mat));
  SB_CHK_ERR(!out, return NULL, "sb_mat_malloc: failed to allocate matrix");

  size_t n_elem = n_rows * n_cols;

  double * data = malloc(n_elem * sizeof(double));
  SB_CHK_ERR(!data, free(out); return NULL,
      "sb_mat_malloc: failed to allocate data");

  out->n_rows = n_rows;
  out->n_cols = n_cols;
  out->n_elem = n_elem;
  out->data   = data;

  return out;
}

/// Constructs a matrix with the required capacity and initializes all elements
/// to zero. Requires support for the IEC 60559 standard.
///
/// # Parameters
/// - `n_rows`: number of rows of the matrix
/// - `n_cols`: number of columns of the matrix
///
/// # Returns
/// A `sb_mat` pointer to the allocated matrix, or `NULL` if the allocation
/// fails
/// 
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   sb_mat * A = sb_mat_calloc(3, 3);
///
///   // Initialized to zeros
///   sb_mat_print(A, "A before: ", "%g");
///
///   // Fill the matrix with some values and print
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat_subcpy(A, 0, a, 9);
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
sb_mat * sb_mat_calloc(size_t n_rows, size_t n_cols) {
  sb_mat * out = malloc(sizeof(sb_mat));
  SB_CHK_ERR(!out, return NULL, "sb_mat_calloc: failed to allocate matrix");

  size_t n_elem = n_rows * n_cols;

  double * data = calloc(n_elem, sizeof(double));
  SB_CHK_ERR(!data, free(out); return NULL,
      "sb_mat_calloc: failed to allocate data");

  out->n_rows = n_rows;
  out->n_cols = n_cols;
  out->n_elem = n_elem;
  out->data   = data;

  return out;
}

/// Constructs a matrix with the required capacity and initializes elements to
/// the first `n_rows * n_cols` elements of array `a`. The array must contain
/// at least `n_rows * n_cols` elements.
///
/// # Parameters
/// - `a`: array to be copied into the matrix
/// - `n_rows`: number of rows of the matrix
/// - `n_cols`: number of columns of the matrix
///
/// # Returns
/// A `sb_mat` pointer to the allocated matrix, or `NULL` if the allocation
/// fails
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `a` is not `NULL`
/// 
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
/// 
///   // Construct a matrix from the array
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
///   sb_mat_print(A, "A: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
sb_mat * sb_mat_of_arr(const double * a, size_t n_rows, size_t n_cols) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!a, abort(), "sb_mat_of_arr: a cannot be NULL");
#endif
  sb_mat * out = malloc(sizeof(sb_mat));
  SB_CHK_ERR(!out, return NULL, "sb_mat_of_arr: failed to allocate matrix");

  size_t n_elem = n_rows * n_cols;

  double * data = malloc(n_elem * sizeof(double));
  SB_CHK_ERR(!data, free(out); return NULL,
      "sb_mat_of_arr: failed to allocate data");

  out->n_rows = n_rows;
  out->n_cols = n_cols;
  out->n_elem = n_elem;
  out->data   = memcpy(data, a, n_elem * sizeof(double));

  return out;
}

/// Constructs a matrix as a deep copy of an existing matrix. The state of the
/// existing matrix must be valid.
///
/// # Parameters
/// - `A`: pointer to the matrix to be copied
///
/// # Returns
/// A `sb_mat` pointer to the allocated matrix, or `NULL` if the allocation
/// fails
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
/// 
///   // `A` and `B` contain the same elements
///   sb_mat_c * B = sb_mat_clone(A);
///   sb_mat_print(B, "B before: ", "%g");
///
///   // Fill `B` with some values and print
///   sb_mat_subcpy(B, 0, a + 1, 3);
///   sb_mat_print(B, "B after: ", "%g");
///
///   // `A` is unchanged
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A, B);
/// }
/// ```
sb_mat * sb_mat_clone(const sb_mat * A) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_clone: A cannot be NULL");
#endif
  sb_mat * out = malloc(sizeof(sb_mat));
  SB_CHK_ERR(!out, return NULL, "sb_mat_clone: failed to allocate matrix");

  size_t n_elem = A->n_elem;

  double * data = malloc(n_elem * sizeof(double));
  SB_CHK_ERR(!data, free(out); return NULL,
      "sb_mat_clone: failed to allocate data");

  *out = *A;
  out->data = memcpy(data, A->data, n_elem * sizeof(double));

  return out;
}

/// Deconstructs a matrix.
///
/// # Parameters
/// - `A`: pointer to the matrix
///
/// # Returns
/// No return value
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
/// 
///   // Allocates a pointer to mat
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
///   sb_mat_print(A, "A: ", "%g");
///
///   sb_mat_free(A);
///   // Pointer to `A` is now invalid
/// }
/// ```
void sb_mat_free(sb_mat * A) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_free: A cannot be NULL");
#endif
  free(A->data);
  free(A);
}

/// Sets all elements of `A` to zero. Requires support for the IEC 60559
/// standard.
///
/// # Parameters
/// - `A`: pointer to the matrix
///
/// # Returns
/// A copy of `A`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
///   
///   // Set all elements to zero
///   sb_mat_print(A, "A before: ", "%g");
///   sb_mat_set_zero(A);
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
sb_mat * sb_mat_set_zero(sb_mat * A) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_set_zero: A cannot be NULL");
#endif
  return memset(A->data, 0, A->n_elem * sizeof(double));
}

/// Sets all elements of `A` to `x`.
///
/// # Parameters
/// - `A`: pointer to the matrix
/// - `x`: value for the elements
///
/// # Returns
/// A copy of `A`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
///   
///   // Set all elements to 8.
///   sb_mat_print(A, "A before: ", "%g");
///   sb_mat_set_all(A, 8.);
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
sb_mat * sb_mat_set_all(sb_mat * A, double x) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_set_all: A cannot be NULL");
#endif
  double * data = A->data;
  for (size_t a = 0; a < A->n_elem; ++a) {
    data[a] = x;
  }
  return A;
}

/// Sets all elements of `A` to zero, except for elements on the main diagonal
/// which are set to one. `A` must be a square matrix. Requires support for the
/// IEC 60559 standard.
///
/// # Parameters
/// - `A`: pointer to the matrix
///
/// # Returns
/// A copy of `A`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
/// - `SAFE_LENGTH`: `A` is a square matrix
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
///   
///   // Set `A` to the identity
///   sb_mat_print(A, "A before: ", "%g");
///   sb_mat_set_ident(A);
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
sb_mat * sb_mat_set_ident(sb_mat * A) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_set_ident: A cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(A->n_rows != A->n_cols, abort(),
      "sb_mat_set_ident: A must be square");
#endif
  double * data = A->data;
  size_t n_elem = A->n_elem;

  memset(data, 0, n_elem * sizeof(double));
  for (size_t a = 0; a < n_elem; a += A->n_rows + 1) {
    data[a] = 1.;
  }

  return A;
}

/// Copies the `i`th row of the matrix `A` into the row vector `v`. `v` and `A`
/// must not overlap in memory.
///
/// # Parameters
/// - `v`: pointer to the vector
/// - `A`: pointer to the matrix
/// - `i`: index of the row to be copied
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` and `A` are not `NULL`
/// - `SAFE_LAYOUT`: `v` is a row vector
/// - `SAFE_LENGTH`: `v` and `A` have the same number of columns and `i` is a
///                  valid row index
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
///   sb_vec * v = sb_vec_malloc(3, 'r');
/// 
///   // Set v to the first row
///   sb_mat_print(A, "A: ", "%g");
///   sb_mat_get_row(v, A, 1);
///   sb_vec_print(v, "v: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_vec * sb_mat_get_row(sb_vec * restrict v, const sb_mat * restrict A, size_t i) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_mat_get_row: v cannot be NULL");
  SB_CHK_ERR(!A, abort(), "sb_mat_get_row: A cannot be NULL");
#endif
#ifdef SAFE_LAYOUT
  SB_CHK_ERR(v->layout != 'r', abort(), "sb_mat_get_row: v must be a row vector");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(v->n_elem != A->n_cols, abort(),
      "sb_mat_get_row: v and A must have same number of columns");
  SB_CHK_ERR(i >= A->n_rows, abort(),
      "sb_mat_get_row: invalid row index");
#endif
  cblas_dcopy(A->n_cols, A->data + i, A->n_rows, v->data, 1);
  return v;
}

/// Copies the `i`th column of the matrix `A` into the column vector `v`. `v`
/// and `A` must not overlap in memory.
///
/// # Parameters
/// - `v`: pointer to the vector
/// - `A`: pointer to the matrix
/// - `j`: index of the column to be copied
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` and `A` are not `NULL`
/// - `SAFE_LAYOUT`: `v` is a column vector
/// - `SAFE_LENGTH`: `v` and `A` have the same number of rows and `j` is a
///                  valid row index
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
///   sb_vec * v = sb_vec_malloc(3, 'c');
/// 
///   // Set v to the first column
///   sb_mat_print(A, "A: ", "%g");
///   sb_mat_get_col(v, A, 1);
///   sb_vec_print(v, "v: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_vec * sb_mat_get_col(sb_vec * restrict v, const sb_mat * restrict A, size_t j) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_mat_get_col: v cannot be NULL");
  SB_CHK_ERR(!A, abort(), "sb_mat_get_col: A cannot be NULL");
#endif
#ifdef SAFE_LAYOUT
  SB_CHK_ERR(v->layout != 'c', abort(), "sb_mat_get_col: v must be a column vector");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(v->n_elem != A->n_rows, abort(),
      "sb_mat_get_col: v and A must have same number of rows");
  SB_CHK_ERR(j >= A->n_cols, abort(),
      "sb_mat_get_col: invalid column index");
#endif
  size_t n_rows = A->n_rows;
  memcpy(v->data, A->data + j * n_rows, n_rows * sizeof(double));
  return v;
}

/// Copies the diagonal of the matrix `A` into the vector `v`. `A` must be a
/// square matrix, and `v` and `A` must not overlap in memory.
///
/// # Parameters
/// - `v`: pointer to the vector
/// - `A`: pointer to the matrix
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` and `A` are not `NULL`
/// - `SAFE_LENGTH`: `A` is square, and the number of elements in `v` is the
///                  same as the number of rows in `A`
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
///   sb_vec * v = sb_vec_malloc(3, 'c');
/// 
///   // Set v to the main diagonal
///   sb_mat_print(A, "A: ", "%g");
///   sb_mat_get_diag(v, A);
///   sb_vec_print(v, "v: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_vec * sb_mat_get_diag(sb_vec * restrict v, const sb_mat * restrict A) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_mat_get_diag: v cannot be NULL");
  SB_CHK_ERR(!A, abort(), "sb_mat_get_diag: A cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(A->n_rows != A->n_cols, abort(),
      "sb_mat_get_diag: A must be square");
  SB_CHK_ERR(v->n_elem != A->n_rows, abort(),
      "sb_mat_get_diag: number of elements in v should equal number of rows of A");
#endif
  size_t n_rows = A->n_rows;
  cblas_dcopy(n_rows, A->data, n_rows + 1, v->data, 1);
  return v;
}

/// Copies the row vector `v` into the `i`th row of the matrix `A`. `v` and `A`
/// must not overlap in memory.
///
/// # Parameters
/// - `A`: pointer to the matrix
/// - `i`: index of the row to be written
/// - `v`: pointer to the vector
///
/// # Returns
/// A copy of `A`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` and `A` are not `NULL`
/// - `SAFE_LAYOUT`: `v` is a row vector
/// - `SAFE_LENGTH`: `v` and `A` have the same number of columns and `i` is a
///                  valid row index
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
///   sb_vec * v = sb_vec_of_arr(a + 2, 3, 'r');
/// 
///   // Set the first row to v
///   sb_mat_print(A, "A before: ", "%g");
///   sb_vec_print(v, "v: ", "%g");
///   sb_mat_set_row(A, 1, v);
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_mat * sb_mat_set_row(sb_mat * restrict A, size_t i, const sb_vec * restrict v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_set_row: A cannot be NULL");
  SB_CHK_ERR(!v, abort(), "sb_mat_set_row: v cannot be NULL");
#endif
#ifdef SAFE_LAYOUT
  SB_CHK_ERR(v->layout != 'r', abort(), "sb_mat_set_row: v must be a row vector");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(v->n_elem != A->n_cols, abort(),
      "sb_mat_set_row: v and A must have same number of columns");
  SB_CHK_ERR(i >= A->n_rows, abort(),
      "sb_mat_set_row: invalid row index");
#endif
  cblas_dcopy(A->n_cols, v->data, 1, A->data + i, A->n_rows);
  return A;
}

/// Copies the column vector `v` into the `i`th column of the matrix `A`. `v`
/// and `A` must not overlap in memory.
///
/// # Parameters
/// - `A`: pointer to the matrix
/// - `j`: index of the column to be written
/// - `v`: pointer to the vector
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` and `A` are not `NULL`
/// - `SAFE_LAYOUT`: `v` is a column vector
/// - `SAFE_LENGTH`: `v` and `A` have the same number of rows and `j` is a
///                  valid row index
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
///   sb_vec * v = sb_vec_of_arr(a + 2, 3, 'c');
/// 
///   // Set the first column to v
///   sb_mat_print(A, "A before: ", "%g");
///   sb_vec_print(v, "v: ", "%g");
///   sb_mat_set_col(A, 1, v);
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_mat * sb_mat_set_col(sb_mat * restrict A, size_t j, const sb_vec * restrict v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_set_col: A cannot be NULL");
  SB_CHK_ERR(!v, abort(), "sb_mat_set_col: v cannot be NULL");
#endif
#ifdef SAFE_LAYOUT
  SB_CHK_ERR(v->layout != 'c', abort(), "sb_mat_set_col: v must be a column vector");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(v->n_elem != A->n_rows, abort(),
      "sb_mat_set_col: v and A must have same number of rows");
  SB_CHK_ERR(j >= A->n_cols, abort(),
      "sb_mat_set_col: invalid column index");
#endif
  size_t n_rows = A->n_rows;
  memcpy(A->data + j * n_rows, v->data, n_rows * sizeof(double));
  return A;
}

/// Copies the vector `v` into the diagonal of the matrix `A`. `A` must be a
/// square matrix, and `v` and `A` must not overlap in memory.
///
/// # Parameters
/// - `v`: pointer to the vector
/// - `A`: pointer to the matrix
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` and `A` are not `NULL`
/// - `SAFE_LENGTH`: `A` is square, and the number of elements in `v` is the
///                  same as the number of rows in `A`
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
///   sb_vec * v = sb_vec_of_arr(a + 2, 3, 'c');
/// 
///   // Set the main diagonal to v
///   sb_mat_print(A, "A before: ", "%g");
///   sb_vec_print(v, "v: ", "%g");
///   sb_mat_set_diag(A, v);
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_mat * sb_mat_set_diag(sb_mat * restrict A, const sb_vec * restrict v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_get_diag: A cannot be NULL");
  SB_CHK_ERR(!v, abort(), "sb_mat_get_diag: v cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(A->n_rows != A->n_cols, abort(),
      "sb_mat_get_diag: A must be square");
  SB_CHK_ERR(v->n_elem != A->n_rows, abort(),
      "sb_mat_get_diag: number of elements in v should equal number of rows of A");
#endif
  size_t n_rows = A->n_rows;
  cblas_dcopy(n_rows, v->data, 1, A->data, n_rows + 1);
  return A;
}

/// Copies contents of the `src` matrix into the `dest` matrix. `src` and `dest`
/// must have the same number of rows and columns and not overlap in memory.
///
/// # Parameters
/// - `dest`: pointer to destination matrix
/// - `src`: const pointer to source matrix
///
/// # Returns
/// A copy of `dest`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `src` and `dest` are not `NULL`
/// - `SAFE_LENGTH`: `src` and `dest` have same number of rows and columns
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2., 8.};
///   sb_mat * A = sb_mat_of_arr(a,     3, 3);
///   sb_mat * B = sb_mat_of_arr(a + 1, 3, 3);
///
///   // Overwrite elements of `A` and print
///   sb_mat_print(A, "A before: ", "%g");
///   sb_mat_print(B, "B before: ", "%g");
///   sb_mat_memcpy(A, B);
///   sb_mat_print(A, "A after: ", "%g");
///   sb_mat_print(B, "B after: ", "%g");
///
///   SB_MAT_FREE_ALL(A, B);
/// }
/// ```
sb_mat * sb_mat_memcpy(sb_mat * restrict dest, const sb_mat * restrict src) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!dest, abort(), "sb_mat_memcpy: dest cannot be NULL");
  SB_CHK_ERR(!src, abort(), "sb_mat_memcpy: src cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(dest->n_rows != src->n_rows, abort(),
      "sb_mat_memcpy: dest and src must have same number of rows");
  SB_CHK_ERR(dest->n_cols != src->n_cols, abort(),
      "sb_mat_memcpy: dest and src must have same number of columns");
#endif
  memcpy(dest->data, src->data, src->n_elem * sizeof(double));
  return dest;
}

/// Copies `n` elements of array `a` into the memory of matrix `A` starting at
/// index `i`. `A` must have enough capacity, `a` must contain at least `n`
/// elements, and `A` and `a` must not overlap in memory.
///
/// # Parameters
/// - `A`: pointer to destination matrix
/// - `i`: index of `A` where the copy will start
/// - `a`: pointer to elements that will be copied
/// - `n`: number of elements to copy
///
/// # Returns
/// A copy of `A`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` and `a` are not `NULL`
/// - `SAFE_LENGTH`: `A` has enough capacity
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
///
///   // Overwrite elements of `A` and print
///   sb_mat_print(A, "A before: ", "%g");
///   sb_mat_subcpy(A, 0, a + 1, 3);
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
sb_mat * sb_mat_subcpy(
    sb_mat * restrict A,
    size_t i,
    const double * restrict a,
    size_t n) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_subcpy: A cannot be NULL");
  SB_CHK_ERR(!a, abort(), "sb_mat_subcpy: a cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(A->n_elem - i < n, abort(),
      "sb_mat_subcpy: A does not have enough capacity");
#endif
  memcpy(A->data + i, a, n * sizeof(double));
  return A;
}

/// Swaps the contents of `A` and `B` by exchanging data pointers. Matrices must
/// have the same number of rows and columns and not overlap in memory.
///
/// # Parameters
/// - `A`: pointer to the first matrix
/// - `B`: pointer to the second matrix
///
/// # Returns
/// No return value
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` and `B` are not `NULL`
/// - `SAFE_LENGTH`: `A` and `B` have the same numbers of rows and columns
///
/// # Examples
/// ```
/// #include "matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2., 8.};
///   sb_mat * A = sb_mat_of_arr(a,     3, 3);
///   sb_mat * B = sb_mat_of_arr(a + 1, 3, 3);
/// 
///   // Print matrices before and after
///   sb_mat_print(A, "A before: ", "%g");
///   sb_mat_print(B, "B before: ", "%g");
///
///   sb_mat_swap(A, B);
///
///   sb_mat_print(A, "A after: ", "%g");
///   sb_mat_print(B, "B after: ", "%g");
///
///   SB_MAT_FREE_ALL(A, B);
/// }
/// ```
void sb_mat_swap(sb_mat * restrict A, sb_mat * restrict B) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_swap: A cannot be NULL");
  SB_CHK_ERR(!B, abort(), "sb_mat_swap: B cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(A->n_rows != B->n_rows, abort(),
      "sb_mat_swap: A and B must have same number of rows");
  SB_CHK_ERR(A->n_cols != B->n_cols, abort(),
      "sb_mat_swap: A and B must have same number of columns");
#endif
  double * scratch;
  SB_SWAP(A->data, B->data, scratch);
}

/// Swaps the `i`th and `j`th rows of the matrix `A`.
///
/// # Parameters
/// - `A`: pointer to the matrix
///
/// # Returns
/// A copy of `A`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
/// - `SAFE_LENGTH`: `i` and `j` are valid row indices
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
/// 
///   // Print matrix before and after
///   sb_mat_print(A, "A before: ", "%g");
///   sb_mat_swap_row(A, 0, 1);
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
sb_mat * sb_mat_swap_row(sb_mat * A, size_t i, size_t j) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_swap_row: A cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(i >= A->n_rows, abort(),
      "sb_mat_swap_row: i must be a valid row index");
  SB_CHK_ERR(j >= A->n_rows, abort(),
      "sb_mat_swap_row: j must be a valid row index");
#endif
  double * data = A->data;
  size_t n_rows = A->n_rows;
  cblas_dswap(A->n_cols, data + i, n_rows, data + j, n_rows);
  return A;
}

/// Swaps the `i`th and `j`th columns of the matrix `A`.
///
/// # Parameters
/// - `A`: pointer to the matrix
///
/// # Returns
/// A copy of `A`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
/// - `SAFE_LENGTH`: `i` and `j` are valid column indices
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
/// 
///   // Print matrix before and after
///   sb_mat_print(A, "A before: ", "%g");
///   sb_mat_swap_col(A, 0, 1);
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
sb_mat * sb_mat_swap_col(sb_mat * A, size_t i, size_t j) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_swap_row: A cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(i >= A->n_cols, abort(),
      "sb_mat_swap_row: i must be a valid column index");
  SB_CHK_ERR(j >= A->n_cols, abort(),
      "sb_mat_swap_row: j must be a valid column index");
#endif
  double * data = A->data;
  size_t n_rows = A->n_rows;
  cblas_dswap(n_rows, data + i * n_rows, 1, data + j * n_rows, 1);
  return A;
}

/// Writes the matrix `A` to `stream` in a binary format. The data is written
/// in the native binary format of the architecture, and may not be portable.
///
/// # Parameters
/// - `stream`: an open I/O stream
/// - `A`: pointer to the matrix
///
/// # Returns
/// `0` on success, or `1` if the write fails
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
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
///   
///   // Write the matrix to file
///   FILE * f = fopen("matrix.bin", "wb");
///   sb_mat_fwrite(f, A);
///   fclose(f);
///   
///   // Read the matrix from file
///   FILE * g = fopen("matrix.bin", "rb");
///   sb_mat * B = sb_mat_fread(g);
///   fclose(g);
///   
///   // Matrices have the same contents
///   sb_mat_print(A, "A: ", "%g");
///   sb_mat_print(B, "B: ", "%g");
///   
///   SB_MAT_FREE_ALL(A, B);
/// }
/// ```
int sb_mat_fwrite(FILE * stream, const sb_mat * A) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_fwrite: A cannot be NULL");
#endif
  size_t n_write;

  n_write = fwrite(&(A->n_rows), sizeof(size_t), 1, stream);
  SB_CHK_ERR(n_write != 1, return 1, "sb_mat_fwrite: fwrite failed");

  n_write = fwrite(&(A->n_cols), sizeof(size_t), 1, stream);
  SB_CHK_ERR(n_write != 1, return 1, "sb_mat_fwrite: fwrite failed");

  size_t n_elem = A->n_elem; 
  n_write = fwrite(A->data, sizeof(double), n_elem, stream);
  SB_CHK_ERR(n_write != n_elem, return 1, "sb_mat_fwrite: fwrite failed");

  return 0;
}

/// Writes the matrix `A` to `stream`. The number of elements and elements are
/// written in a human readable format.
///
/// # Parameters
/// - `stream`: an open I/O stream
/// - `A`: pointer to the matrix
/// - `format`: a format specifier for the elements
///
/// # Returns
/// `0` on success, or `1` if the write fails
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
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
///   
///   // Write the matrix to file
///   FILE * f = fopen("matrix.txt", "w");
///   sb_mat_fprintf(f, A, "%lg");
///   fclose(f);
///   
///   // Read the matrix from file
///   FILE * g = fopen("matrix.txt", "r");
///   sb_mat * B = sb_mat_fscanf(g);
///   fclose(g);
///   
///   // Matrices have the same contents
///   sb_mat_print(A, "A: ", "%g");
///   sb_mat_print(B, "B: ", "%g");
///   
///   SB_MAT_FREE_ALL(A, B);
/// }
/// ```
int sb_mat_fprintf(FILE * stream, const sb_mat * A, const char * format) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_fprintf: A cannot be NULL");
#endif
  int status;

  status = fprintf(stream, "%zu %zu", A->n_rows, A->n_cols);
  SB_CHK_ERR(status < 0, return 1, "sb_mat_fprintf: fprintf failed");

  double * data = A->data;
  for (size_t a = 0; a < A->n_elem; ++a) {
    status = putc(' ', stream);
    SB_CHK_ERR(status == EOF, return 1, "sb_mat_fprintf: putc failed");
    status = fprintf(stream, format, data[a]);
    SB_CHK_ERR(status < 0, return 1, "sb_mat_fprintf: fprintf failed");
  }
  status = putc('\n', stream);
  SB_CHK_ERR(status == EOF, return 1, "sb_mat_fprintf: putc failed");

  return 0;
}

/// Prints the matrix `A` to stdout. Output is slightly easier to read than for
/// `sb_mat_fprintf()`. Mainly indended for debugging.
///
/// # Parameters
/// - `A`: pointer to the matrix
/// - `str`: a string to describe the matrix
/// - `format`: a format specifier for the elements
///
/// # Returns
/// `0` on success, or `1` if the print fails
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4.};
///   sb_mat * A = sb_mat_of_arr(a, 2, 4);
///   sb_mat * B = sb_mat_of_arr(a, 4, 2);
///
///   // Prints the contents of `A` and `B` to stdout
///   sb_mat_print(A, "A: ", "%g");
///   sb_mat_print(B, "B: ", "%g");
///
///   SB_MAT_FREE_ALL(A, B);
/// }
/// ```
int sb_mat_print(const sb_mat * A, const char * str, const char * format) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_print: A cannot be NULL");
#endif
  int status;

  status = printf("%s\n", str);
  SB_CHK_ERR(status < 0, return 1, "sb_mat_print: printf failed");

  size_t n_elem = A->n_elem;
  double * data = A->data;

  char   buffer[128];
  char * dec;
  bool   dec_mark[n_elem];
  bool   any_mark = false;

  unsigned char length;
  unsigned char len_head[n_elem];
  unsigned char max_head = 0;
  unsigned char len_tail[n_elem];
  unsigned char max_tail = 0;
  for (size_t a = 0; a < n_elem; ++a) {
    status = snprintf(buffer, 128, format, data[a]);
    SB_CHK_ERR(status < 0, return 1, "sb_mat_print: snprintf failed");
    dec = strchr(buffer, '.');
    if (dec) {
      dec_mark[a] = true;
      any_mark    = true;
      length = (unsigned char)(dec - buffer);
      len_head[a] = length;
      if (length > max_head) { max_head = length; }
      length = strlen(buffer) - length - 1;
      if (length > max_tail) { max_tail = length; }
      len_tail[a] = length;
    } else {
      dec_mark[a] = false;
      length = strlen(buffer);
      len_head[a] = length;
      if (length > max_head) { max_head = length; }
      len_tail[a] = 0;
    }
  }

  size_t n_rows = A->n_rows;
  size_t n_cols = A->n_cols;
  size_t index;
  for (size_t r = 0; r < n_rows; ++r) {
    for (size_t c = 0; c < n_cols; ++c) {
      index = c * n_rows + r;
      for (unsigned char s = 0; s < max_head - len_head[index]; ++s) {
        status = putchar(' ');
        SB_CHK_ERR(status == EOF, return 1, "sb_mat_print: putchar failed");
      }
      status = printf(format, data[index]);
      SB_CHK_ERR(status < 0, return 1, "sb_mat_print: printf failed");
      if (any_mark && !dec_mark[index]) {
        status = putchar(' ');
        SB_CHK_ERR(status == EOF, return 1, "sb_mat_print: putchar failed");
      }
      for (unsigned char s = 0; s < max_tail - len_tail[index] + 1; ++s) {
        status = putchar(' ');
        SB_CHK_ERR(status == EOF, return 1, "sb_mat_print: putchar failed");
      }
    }
    status = putchar('\n');
    SB_CHK_ERR(status == EOF, return 1, "sb_mat_print: putchar failed");
  }

  return 0;
}

/// Reads binary data from `stream` into the matrix returned by the function.
/// The data must be written in the native binary format of the architecture, 
/// preferably by `sb_mat_fwrite()`.
///
/// # Parameters
/// - `stream`: an open I/O stream
///
/// # Returns
/// A `sb_mat` pointer to the matrix read from `stream`, or `NULL` if the read
/// or memory allocation fails
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
///   
///   // Write the matrix to file
///   FILE * f = fopen("matrix.bin", "wb");
///   sb_mat_fwrite(f, A);
///   fclose(f);
///   
///   // Read the matrix from file
///   FILE * g = fopen("matrix.bin", "rb");
///   sb_mat * B = sb_mat_fread(g);
///   fclose(g);
///   
///   // Matrices have the same contents
///   sb_mat_print(A, "A: ", "%g");
///   sb_mat_print(B, "B: ", "%g");
///   
///   SB_MAT_FREE_ALL(A, B);
/// }
/// ```
sb_mat * sb_mat_fread(FILE * stream) {
  size_t n_read;

  size_t n_rows;
  n_read = fread(&n_rows, sizeof(size_t), 1, stream);
  SB_CHK_ERR(n_read != 1, return NULL, "sb_mat_fread: fread failed");

  size_t n_cols;
  n_read = fread(&n_cols, sizeof(size_t), 1, stream);
  SB_CHK_ERR(n_read != 1, return NULL, "sb_mat_fread: fread failed");

  sb_mat * out = sb_mat_malloc(n_rows, n_cols);
  SB_CHK_ERR(!out, return NULL, "sb_mat_fread: sb_mat_malloc failed");

  size_t n_elem = out->n_elem;
  n_read = fread(out->data, sizeof(double), n_elem, stream);
  SB_CHK_ERR(n_read != n_elem, sb_mat_free(out); return NULL,
      "sb_mat_fread: fread failed");

  return out;
}

/// Reads formatted data from `stream` into the matrix returned by the function.
///
/// # Parameters
/// - `stream`: an open I/O stream
///
/// # Returns
/// A `sb_mat` pointer to the matrix read from `stream`, or `NULL` if the scan
/// or memory allocation fails
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
///   
///   // Write the matrix to file
///   FILE * f = fopen("matrix.txt", "w");
///   sb_mat_fprintf(f, A, "%lg");
///   fclose(f);
///   
///   // Read the matrix from file
///   FILE * g = fopen("matrix.txt", "r");
///   sb_mat * B = sb_mat_fscanf(g);
///   fclose(g);
///   
///   // Matrices have the same contents
///   sb_mat_print(A, "A: ", "%g");
///   sb_mat_print(B, "B: ", "%g");
///   
///   SB_MAT_FREE_ALL(A, B);
/// }
/// ```
sb_mat * sb_mat_fscanf(FILE * stream) {
  int n_scan;

  size_t n_rows;
  size_t n_cols;
  n_scan = fscanf(stream, "%zu %zu", &n_rows, &n_cols);
  SB_CHK_ERR(n_scan != 2, return NULL, "sb_mat_fscanf: fscanf failed");

  sb_mat * out = sb_mat_malloc(n_rows, n_cols);
  SB_CHK_ERR(!out, return NULL, "sb_mat_fscanf: sb_mat_malloc failed");

  double * data = out->data;
  for (size_t a = 0; a < out->n_elem; ++a) {
    n_scan = fscanf(stream, "%lg", data + a);
    SB_CHK_ERR(n_scan != 1, sb_mat_free(out); return NULL,
        "sb_mat_fscanf: fscanf failed");
  }

  return out;
}

/// Takes the absolute value of every element of the matrix.
///
/// # Parameters
/// - `A`: pointer to the matrix
///
/// # Returns
/// A copy of `A`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
/// - `SAFE_FINITE`: elements of `A` are finite
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., -4., 2., 8., 5., -7., -1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
/// 
///   // Take absolute value
///   sb_mat_print(A, "A before: ", "%g");
///   sb_mat_abs(A);
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
sb_mat * sb_mat_abs(sb_mat * A) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_abs: A cannot be NULL");
#endif
  double * data = A->data;
  for (size_t a = 0; a < A->n_elem; ++a) {
    data[a] = fabs(data[a]);
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_mat_is_finite(A), abort(), "sb_mat_abs: element not finite");
#endif
  return A;
}

/// Takes the exponent base `e` of every element of the matrix.
///
/// # Parameters
/// - `A`: pointer to the matrix
///
/// # Returns
/// A copy of `A`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
/// - `SAFE_FINITE`: elements of `A` are finite
///
/// # Examples
/// ```
/// #include <math.h>
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///  double a[] = {log(1.), log(4.), log(2.),
///                log(8.), log(5.), log(7.),
///                log(1.), log(4.), log(2.)};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
/// 
///   // Take exponent base e
///   sb_mat_print(A, "A before: ", "%g");
///   sb_mat_exp(A);
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
sb_mat * sb_mat_exp(sb_mat * A) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_exp: A cannot be NULL");
#endif
  double * data = A->data;
  for (size_t a = 0; a < A->n_elem; ++a) {
    data[a] = exp(data[a]);
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_mat_is_finite(A), abort(), "sb_mat_exp: element not finite");
#endif
  return A;
}

/// Takes the logarithm base `e` of every element of the matrix.
///
/// # Parameters
/// - `A`: pointer to the matrix
///
/// # Returns
/// A copy of `A`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
/// - `SAFE_FINITE`: elements of `A` are finite
///
/// # Examples
/// ```
/// #include <math.h>
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {exp(1.), exp(4.), exp(2.),
///                 exp(8.), exp(5.), exp(7.),
///                 exp(1.), exp(4.), exp(2.)};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
/// 
///   // Take exponent base e
///   sb_mat_print(A, "A before: ", "%g");
///   sb_mat_log(A);
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
sb_mat * sb_mat_log(sb_mat * A) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_log: A cannot be NULL");
#endif
  double * data = A->data;
  for (size_t a = 0; a < A->n_elem; ++a) {
    data[a] = log(data[a]);
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_mat_is_finite(A), abort(), "sb_mat_log: element not finite");
#endif
  return A;
}

/// Exponentiates every element of the matrix `A` by `x`.
///
/// # Parameters
/// - `A`: pointer to the matrix
/// - `x`: scalar exponent of the elements
///
/// # Returns
/// A copy of `A`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
/// - `SAFE_FINITE`: elements of `A` are finite
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
/// 
///   // Exponentiate every element by -1.
///   sb_mat_print(A, "A before: ", "%g");
///   sb_mat_pow(A, -1.);
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
sb_mat * sb_mat_pow(sb_mat * A, double x) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_pow: A cannot be NULL");
#endif
  double * data = A->data;
  for (size_t a = 0; a < A->n_elem; ++a) {
    data[a] = pow(data[a], x);
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_mat_is_finite(A), abort(), "sb_mat_pow: element not finite");
#endif
  return A;
}

/// Takes the square root of every element of the matrix `A`.
///
/// # Parameters
/// - `A`: pointer to the matrix
/// - `x`: scalar exponent of the elements
///
/// # Returns
/// A copy of `A`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
/// - `SAFE_FINITE`: elements of `A` are finite
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 16., 4., 64., 25., 49., 1., 16., 4.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
/// 
///   // Take the square root of every element
///   sb_mat_print(A, "A before: ", "%g");
///   sb_mat_sqrt(A);
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
sb_mat * sb_mat_sqrt(sb_mat * A) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_sqrt: A cannot be NULL");
#endif
  double * data = A->data;
  for (size_t a = 0; a < A->n_elem; ++a) {
    data[a] = sqrt(data[a]);
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_mat_is_finite(A), abort(), "sb_mat_sqrt: element not finite");
#endif
  return A;
}

/// Scalar addition of `x` to every element of the matrix `A`.
///
/// # Parameters
/// - `A`: pointer to the matrix
/// - `x`: scalar added to the elements
///
/// # Returns
/// A copy of `A`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
/// - `SAFE_FINITE`: elements of `A` are finite
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
/// 
///   // Add 2. to every element
///   sb_mat_print(A, "A before: ", "%g");
///   sb_mat_sadd(A, 2.);
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
sb_mat * sb_mat_sadd(sb_mat * A, double x) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_sadd: A cannot be NULL");
#endif
  double * data = A->data;
  for (size_t a = 0; a < A->n_elem; ++a) {
    data[a] += x;
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_mat_is_finite(A), abort(), "sb_mat_sadd: element not finite");
#endif
  return A;
}

/// Scalar multiplication of `x` with every element of the matrix `A`.
///
/// # Parameters
/// - `A`: pointer to the matrix
/// - `x`: scalar mutiplier for the elements
///
/// # Returns
/// A copy of `A`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
/// - `SAFE_FINITE`: elements of `A` are finite
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
/// 
///   // Multiply every element by -1.
///   sb_mat_print(A, "A before: ", "%g");
///   sb_mat_smul(A, -1.);
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
sb_mat * sb_mat_smul(sb_mat * A, double x) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_smul: A cannot be NULL");
#endif
  cblas_dscal(A->n_elem, x, A->data, 1);
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_mat_is_finite(A), abort(), "sb_mat_smul: elements not finite");
#endif
  return A;
}

/// Adds to every row or column of the matrix `A` the corresponding element of
/// the vector `v`. `A` and `v` must not overlap in memory.
///
/// # Parameters
/// - `A`: pointer to the matrix
/// - `v`: pointer to the vector
/// - `dir`: one of 'r' or 'c'
///
/// # Returns
/// A copy of `A`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` and `v` are not `NULL`
/// - `SAFE_LAYOUT`: the layout of `v` is compatible with `dir`
/// - `SAFE_LENGTH`: `A` and `v` have compatible dimensions
/// - `SAFE_FINITE`: elements of `A` are finite
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // Add v to every column of A
///   sb_vec_print(v, "v: ", "%g");
///   sb_mat_print(A, "A before: ", "%g");
///   sb_mat_vadd(A, v, 'c');
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_mat * sb_mat_vadd(sb_mat * restrict A, const sb_vec * restrict v, char dir) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_vadd: A cannot be NULL");
  SB_CHK_ERR(!v, abort(), "sb_mat_vadd: v cannot be NULL");
#endif
  size_t n_rows = A->n_rows;
  size_t n_cols = A->n_cols;
  double * A_data = A->data;
  double * v_data = v->data;
  switch (dir) {
    case 'c':
#ifdef SAFE_LAYOUT
      SB_CHK_ERR(v->layout != 'r', abort(), "sb_mat_vadd: v must be a row vector");
#endif
#ifdef SAFE_LENGTH
      SB_CHK_ERR(A->n_cols != v->n_elem, abort(),
          "sb_mat_vadd: A and v must have same number of columns");
#endif
      for (size_t a = 0; a < n_rows; ++a) {
        cblas_daxpy(n_cols, 1., v_data, 1, A_data + a, n_rows);
      }
      break;
    case 'r':
#ifdef SAFE_LAYOUT
      SB_CHK_ERR(v->layout != 'c', abort(), "sb_mat_radd: v must be a column vector");
#endif
#ifdef SAFE_LENGTH
      SB_CHK_ERR(A->n_rows != v->n_elem, abort(),
          "sb_mat_radd: A and v must have same number of rows");
#endif
      for (size_t a = 0; a < n_cols; ++a) {
        cblas_daxpy(n_rows, 1., v_data, 1, A_data + a * n_rows, 1);
      }
      break;
    default:
      SB_CHK_ERR(true, abort(), "sb_mat_vadd: dir must 'c' or 'r'");
      break;
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_mat_is_finite(A), abort(), "sb_mat_vadd: elements not finite");
#endif
  return A;
}

/// Subtracts from every row or column of the matrix `A` the corresponding
/// element of the vector `v`. `A` and `v` must not overlap in memory.
///
/// # Parameters
/// - `A`: pointer to the matrix
/// - `v`: pointer to the vector
/// - `dir`: one of 'r' or 'c'
///
/// # Returns
/// A copy of `A`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` and `v` are not `NULL`
/// - `SAFE_LAYOUT`: the layout of `v` is compatible with `dir`
/// - `SAFE_LENGTH`: `A` and `v` have compatible dimensions
/// - `SAFE_FINITE`: elements of `A` are finite
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // Add v to every column of A
///   sb_vec_print(v, "v: ", "%g");
///   sb_mat_print(A, "A before: ", "%g");
///   sb_mat_vsub(A, v, 'c');
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_mat * sb_mat_vsub(sb_mat * restrict A, const sb_vec * restrict v, char dir) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_vsub: A cannot be NULL");
  SB_CHK_ERR(!v, abort(), "sb_mat_vsub: v cannot be NULL");
#endif
  size_t n_rows = A->n_rows;
  size_t n_cols = A->n_cols;
  double * A_data = A->data;
  double * v_data = v->data;
  switch (dir) {
    case 'c':
#ifdef SAFE_LAYOUT
      SB_CHK_ERR(v->layout != 'r', abort(), "sb_mat_vsub: v must be a row vector");
#endif
#ifdef SAFE_LENGTH
      SB_CHK_ERR(A->n_cols != v->n_elem, abort(),
          "sb_mat_vsub: A and v must have same number of columns");
#endif
      for (size_t a = 0; a < n_rows; ++a) {
        cblas_daxpy(n_cols, -1., v_data, 1, A_data + a, n_rows);
      }
      break;
    case 'r':
#ifdef SAFE_LAYOUT
      SB_CHK_ERR(v->layout != 'c', abort(), "sb_mat_rsub: v must be a column vector");
#endif
#ifdef SAFE_LENGTH
      SB_CHK_ERR(A->n_rows != v->n_elem, abort(),
          "sb_mat_rsub: A and v must have same number of rows");
#endif
      for (size_t a = 0; a < n_cols; ++a) {
        cblas_daxpy(n_rows, -1., v_data, 1, A_data + a * n_rows, 1);
      }
      break;
    default:
      SB_CHK_ERR(true, abort(), "sb_mat_vsub: dir must 'c' or 'r'");
      break;
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_mat_is_finite(A), abort(), "sb_mat_vsub: elements not finite");
#endif
  return A;
}

/// Multiplies every row or column of the matrix `A` by the corresonding
/// element of the vector `v`. `A` and `v` must not overlap in memory.
///
/// # Parameters
/// - `A`: pointer to the matrix
/// - `v`: pointer to the vector
/// - `dir`: one of 'r' or 'c'
///
/// # Returns
/// A copy of `A`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` and `v` are not `NULL`
/// - `SAFE_LAYOUT`: the layout of `v` is compatible with `dir`
/// - `SAFE_LENGTH`: `A` and `v` have compatible dimensions
/// - `SAFE_FINITE`: elements of `A` are finite
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // Multiplies every column of A by corresponding element of v
///   sb_vec_print(v, "v: ", "%g");
///   sb_mat_print(A, "A before: ", "%g");
///   sb_mat_vmul(A, v, 'c');
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_mat * sb_mat_vmul(sb_mat * restrict A, const sb_vec * restrict v, char dir) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_vmul: A cannot be NULL");
  SB_CHK_ERR(!v, abort(), "sb_mat_vmul: v cannot be NULL");
#endif
  size_t n_rows = A->n_rows;
  size_t n_cols = A->n_cols;
  double * A_data = A->data;
  double * v_data = v->data;
  switch (dir) {
    case 'c':
#ifdef SAFE_LAYOUT
      SB_CHK_ERR(v->layout != 'r', abort(), "sb_mat_vmul: v must be a row vector");
#endif
#ifdef SAFE_LENGTH
      SB_CHK_ERR(A->n_cols != v->n_elem, abort(),
          "sb_mat_vmul: A and v must have same number of columns");
#endif
      for (size_t a = 0; a < n_cols; ++a) {
        cblas_dscal(n_rows, v_data[a], A_data + a * n_rows, 1);
      }
      break;
    case 'r':
#ifdef SAFE_LAYOUT
      SB_CHK_ERR(v->layout != 'c', abort(), "sb_mat_vmul: v must be a column vector");
#endif
#ifdef SAFE_LENGTH
      SB_CHK_ERR(A->n_rows != v->n_elem, abort(),
          "sb_mat_vmul: A and v must have same number of rows");
#endif
      for (size_t a = 0; a < n_rows; ++a) {
        cblas_dscal(n_cols, v_data[a], A_data + a, n_rows);
      }
      break;
    default:
      SB_CHK_ERR(true, abort(), "sb_mat_vmul: dir must 'c' or 'r'");
      break;
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_mat_is_finite(A), abort(), "sb_mat_vmul: elements not finite");
#endif
  return A;
}

/// Divides every column or row of the matrix `A` by the corresonding element
/// of the vector `v`. `A` and `v` must not overlap in memory.
///
/// # Parameters
/// - `A`: pointer to the matrix
/// - `v`: pointer to the vector
/// - `dir`: one of 'r' or 'c'
///
/// # Returns
/// A copy of `A`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` and `v` are not `NULL`
/// - `SAFE_LAYOUT`: the layout of `v` is compatible with `dir`
/// - `SAFE_LENGTH`: `A` and `v` have compatible dimensions
/// - `SAFE_FINITE`: elements of `A` are finite
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // Multiplies every column of A by corresponding element of v
///   sb_vec_print(v, "v: ", "%g");
///   sb_mat_print(A, "A before: ", "%g");
///   sb_mat_vdiv(A, v, 'c');
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_mat * sb_mat_vdiv(sb_mat * restrict A, const sb_vec * restrict v, char dir) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_vdiv: A cannot be NULL");
  SB_CHK_ERR(!v, abort(), "sb_mat_vdiv: v cannot be NULL");
#endif
  size_t n_rows = A->n_rows;
  size_t n_cols = A->n_cols;
  double * A_data = A->data;
  double * v_data = v->data;
  switch (dir) {
    case 'c':
#ifdef SAFE_LAYOUT
      SB_CHK_ERR(v->layout != 'r', abort(), "sb_mat_vdiv: v must be a row vector");
#endif
#ifdef SAFE_LENGTH
      SB_CHK_ERR(A->n_cols != v->n_elem, abort(),
          "sb_mat_vdiv: A and v must have same number of columns");
#endif
      for (size_t a = 0; a < n_cols; ++a) {
        cblas_dscal(n_rows, 1. / v_data[a], A_data + a * n_rows, 1);
      }
      break;
    case 'r':
#ifdef SAFE_LAYOUT
      SB_CHK_ERR(v->layout != 'c', abort(), "sb_mat_rdiv: v must be a column vector");
#endif
#ifdef SAFE_LENGTH
      SB_CHK_ERR(A->n_rows != v->n_elem, abort(),
          "sb_mat_rdiv: A and v must have same number of rows");
#endif
      for (size_t a = 0; a < n_rows; ++a) {
        cblas_dscal(n_cols, 1. / v_data[a], A_data + a, n_rows);
      }
      break;
    default:
      SB_CHK_ERR(true, abort(), "sb_mat_vdiv: dir must 'c' or 'r'");
      break;
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_mat_is_finite(A), abort(), "sb_mat_vdiv: elements not finite");
#endif
  return A;
}

/// Pointwise addition of elements of the matrix `B` to elements of the matrix
/// `A`. `A` and `B` must not overlap in memory.
///
/// # Parameters
/// - `A`: pointer to the first matrix
/// - `B`: pointer to the second matrix
///
/// # Returns
/// A copy of `A`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` and `B` are not `NULL`
/// - `SAFE_LENGTH`: `A` and `B` have the same number of rows and columns
/// - `SAFE_FINITE`: elements of `A` are finite
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2., 8.};
///   sb_mat * A = sb_mat_of_arr(a,     3, 3);
///   sb_mat * B = sb_mat_of_arr(a + 1, 3, 3);
/// 
///   // Add B to A
///   sb_mat_print(A, "A: ", "%g");
///   sb_mat_print(B, "B: ", "%g");
///   sb_mat_padd(A, B);
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A, B);
/// }
/// ```
sb_mat * sb_mat_padd(sb_mat * restrict A, const sb_mat * restrict B) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_padd: A cannot be NULL");
  SB_CHK_ERR(!B, abort(), "sb_mat_padd: B cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(A->n_rows != B->n_rows, abort(),
      "sb_mat_padd: A and B must have same number of rows");
  SB_CHK_ERR(A->n_cols != B->n_cols, abort(),
      "sb_mat_padd: A and B must have same number of columns");
#endif
  cblas_daxpy(A->n_elem, 1., B->data, 1, A->data, 1);
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_mat_is_finite(A), abort(), "sb_mat_padd: elements not finite");
#endif
  return A;
}

/// Pointwise subtraction of elements of the matrix `B` from elements of the
/// matrix `A`. `A` and `B` must not overlap in memory.
///
/// # Parameters
/// - `A`: pointer to the first matrix
/// - `B`: pointer to the second matrix
///
/// # Returns
/// A copy of `A`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` and `B` are not `NULL`
/// - `SAFE_LENGTH`: `A` and `B` have the same number of rows and columns
/// - `SAFE_FINITE`: elements of `A` are finite
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2., 8.};
///   sb_mat * A = sb_mat_of_arr(a,     3, 3);
///   sb_mat * B = sb_mat_of_arr(a + 1, 3, 3);
/// 
///   // Subtract B from A
///   sb_mat_print(A, "A: ", "%g");
///   sb_mat_print(B, "B: ", "%g");
///   sb_mat_psub(A, B);
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A, B);
/// }
/// ```
sb_mat * sb_mat_psub(sb_mat * restrict A, const sb_mat * restrict B) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_psub: A cannot be NULL");
  SB_CHK_ERR(!B, abort(), "sb_mat_psub: B cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(A->n_rows != B->n_rows, abort(),
      "sb_mat_psub: A and B must have same number of rows");
  SB_CHK_ERR(A->n_cols != B->n_cols, abort(),
      "sb_mat_psub: A and B must have same number of columns");
#endif
  cblas_daxpy(A->n_elem, -1., B->data, 1, A->data, 1);
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_mat_is_finite(A), abort(), "sb_mat_psub: elements not finite");
#endif
  return A;
}

/// Pointwise multiplication of elements of the matrix `A` with elements of the
/// matrix `B`. `A` and `B` must not overlap in memory.
///
/// # Parameters
/// - `A`: pointer to the first matrix
/// - `B`: pointer to the second matrix
///
/// # Returns
/// A copy of `A`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` and `B` are not `NULL`
/// - `SAFE_LENGTH`: `A` and `B` have the same number of rows and columns
/// - `SAFE_FINITE`: elements of `A` are finite
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2., 8.};
///   sb_mat * A = sb_mat_of_arr(a,     3, 3);
///   sb_mat * B = sb_mat_of_arr(a + 1, 3, 3);
/// 
///   // Multiply elements of A by elements of B
///   sb_mat_print(A, "A: ", "%g");
///   sb_mat_print(B, "B: ", "%g");
///   sb_mat_pmul(A, B);
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A, B);
/// }
/// ```
sb_mat * sb_mat_pmul(sb_mat * restrict A, const sb_mat * restrict B) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_pmul: A cannot be NULL");
  SB_CHK_ERR(!B, abort(), "sb_mat_pmul: B cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(A->n_rows != B->n_rows, abort(),
      "sb_mat_pmul: A and B must have same number of rows");
  SB_CHK_ERR(A->n_cols != B->n_cols, abort(),
      "sb_mat_pmul: A and B must have same number of columns");
#endif
  double * A_data = A->data;
  double * B_data = B->data;
  for (size_t a = 0; a < A->n_elem; ++a) {
    A_data[a] *= B_data[a];
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_mat_is_finite(A), abort(), "sb_mat_pmul: element not finite");
#endif
  return A;
}

/// Pointwise division of elements of the matrix `A` by elements of the matrix
/// `B`. `A` and `B` must not overlap in memory.
///
/// # Parameters
/// - `A`: pointer to the first matrix
/// - `B`: pointer to the second matrix
///
/// # Returns
/// A copy of `A`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` and `B` are not `NULL`
/// - `SAFE_LENGTH`: `A` and `B` have the same number of rows and columns
/// - `SAFE_FINITE`: elements of `A` are finite
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2., 8.};
///   sb_mat * A = sb_mat_of_arr(a,     3, 3);
///   sb_mat * B = sb_mat_of_arr(a + 1, 3, 3);
/// 
///   // Divide elements of A by elements of B
///   sb_mat_print(A, "A: ", "%g");
///   sb_mat_print(B, "B: ", "%g");
///   sb_mat_pdiv(A, B);
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A, B);
/// }
/// ```
sb_mat * sb_mat_pdiv(sb_mat * restrict A, const sb_mat * restrict B) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_pdiv: A cannot be NULL");
  SB_CHK_ERR(!B, abort(), "sb_mat_pdiv: B cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(A->n_rows != B->n_rows, abort(),
      "sb_mat_pdiv: A and B must have same number of rows");
  SB_CHK_ERR(A->n_cols != B->n_cols, abort(),
      "sb_mat_pdiv: A and B must have same number of columns");
#endif
  double * A_data = A->data;
  double * B_data = B->data;
  for (size_t a = 0; a < A->n_elem; ++a) {
    A_data[a] /= B_data[a];
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_mat_is_finite(A), abort(), "sb_mat_pdiv: element not finite");
#endif
  return A;
}

/// Matrix-vector multiplication of `op(A)` and `w`, where `op` can be nothing
/// or a transpose. The result is stored in the vector `v`, and any previous
/// values in `v` are overwritten. `v`, `A` and `w` must not overlap in memory.
///
/// # Parameters
/// - `v`: pointer to the resulting vector
/// - `A`: pointer to the left matrix
/// - `w`: pointer to the right vector
/// - `op`: 'n' or 't'
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v`, `A` and `w` are not `NULL`
/// - `SAFE_LAYOUT`: `v` and `w` must be column vectors
/// - `SAFE_LENGTH`: `v`, `A` and `w` have compatible dimensions
/// - `SAFE_FINITE`: elements of `v` are finite
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
/// #Include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_vec * v = sb_vec_malloc(3, 'c');
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
///   sb_vec * w = sb_vec_of_arr(a + 1, 3, 'c');
/// 
///   // Multiply A by w and store in v
///   sb_mat_print(A, "A: ", "%g");
///   sb_vec_print(w, "w: ", "%g");
///   sb_mat_mv_mul(v, A, w, 'n');
///   sb_vec_print(v, "v: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
///   SB_VEC_FREE_ALL(v, w);
/// }
/// ```
sb_vec * sb_mat_mv_mul(
    sb_vec * restrict v,
    const sb_mat * restrict A,
    const sb_vec * restrict w,
    char op) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_mat_mv_mul: v cannot be NULL");
  SB_CHK_ERR(!A, abort(), "sb_mat_mv_mul: A cannot be NULL");
  SB_CHK_ERR(!w, abort(), "sb_mat_mv_mul: w cannot be NULL");
#endif
#ifdef SAFE_LAYOUT
  SB_CHK_ERR(v->layout != 'c', abort(), "sb_mat_mv_mul: v must be a column vector");
  SB_CHK_ERR(w->layout != 'c', abort(), "sb_mat_mv_mul: w must be a column vector");
#endif
  switch (op) {
    case 'n':
#ifdef SAFE_LENGTH
      SB_CHK_ERR(v->n_elem != A->n_rows, abort(),
          "sb_mat_mv_mul: v and A must have same number of rows");
      SB_CHK_ERR(A->n_cols != w->n_elem, abort(),
          "sb_mat_mv_mul: A and w must have compatible inner dimensions");
#endif
      cblas_dgemv(CblasColMajor, CblasNoTrans, A->n_rows, A->n_cols,
          1., A->data, A->n_rows, w->data, 1, 0., v->data, 1);
      break;
    case 't':
#ifdef SAFE_LENGTH
      SB_CHK_ERR(v->n_elem != A->n_cols, abort(),
          "sb_mat_mv_mul: v and A^T must have same number of rows");
      SB_CHK_ERR(A->n_rows != w->n_elem, abort(),
          "sb_mat_mv_mul: A^T and w must have compatible inner dimensions");
#endif
      cblas_dgemv(CblasColMajor, CblasTrans, A->n_rows, A->n_cols,
          1., A->data, A->n_rows, w->data, 1, 0., v->data, 1);
      break;
    default:
      SB_CHK_ERR(true, abort(), "sb_mat_mv_mul: op must 'n' or 't'");
      break;
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_vec_is_finite(v), abort(), "sb_mat_mv_mul: elements not finite");
#endif
  return v;
}

/// Vector-matrix multiplication of `w` and `op(A)`, where `op` can be nothing
/// or a transpose. The result is stored in the vector `v`, and any previous
/// values in `v` are overwritten. `v`, `w` and `A` must not overlap in memory.
///
/// # Parameters
/// - `v`: pointer to the resulting vector
/// - `w`: pointer to the left vector
/// - `A`: pointer to the right matrix
/// - `op`: 'n' or 't'
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v`, `w` and `A` are not `NULL`
/// - `SAFE_LAYOUT`: `v` and `w` must be row vectors
/// - `SAFE_LENGTH`: `v`, `w` and `A` have compatible dimensions
/// - `SAFE_FINITE`: elements of `v` are finite
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_vec * v = sb_vec_malloc(3, 'r');
///   sb_vec * w = sb_vec_of_arr(a + 1, 3, 'r');
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
/// 
///   // Multiply w by A and store in v
///   sb_vec_print(w, "w: ", "%g");
///   sb_mat_print(A, "A: ", "%g");
///   sb_mat_vm_mul(v, w, A, 'n');
///   sb_vec_print(v, "v: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
///   SB_VEC_FREE_ALL(v, w);
/// }
/// ```
sb_vec * sb_mat_vm_mul(
    sb_vec * restrict v,
    const sb_vec * restrict w,
    const sb_mat * restrict A,
    char op) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_mat_vm_mul: v cannot be NULL");
  SB_CHK_ERR(!w, abort(), "sb_mat_vm_mul: w cannot be NULL");
  SB_CHK_ERR(!A, abort(), "sb_mat_vm_mul: A cannot be NULL");
#endif
#ifdef SAFE_LAYOUT
  SB_CHK_ERR(v->layout != 'r', abort(), "sb_mat_vm_mul: v must be a row vector");
  SB_CHK_ERR(w->layout != 'r', abort(), "sb_mat_vm_mul: w must be a row vector");
#endif
  switch (op) {
    case 'n':
#ifdef SAFE_LENGTH
      SB_CHK_ERR(v->n_elem != A->n_cols, abort(),
          "sb_mat_vm_mul: v and A must have same number of columns");
      SB_CHK_ERR(A->n_rows != w->n_elem, abort(),
          "sb_mat_vm_mul: A and w must have compatible inner dimensions");
#endif
      cblas_dgemv(CblasColMajor, CblasTrans, A->n_rows, A->n_cols,
          1., A->data, A->n_rows, w->data, 1, 0., v->data, 1);
      break;
    case 't':
#ifdef SAFE_LENGTH
      SB_CHK_ERR(v->n_elem != A->n_rows, abort(),
          "sb_mat_vm_mul: v and A^T must have same number of columns");
      SB_CHK_ERR(A->n_cols != w->n_elem, abort(),
          "sb_mat_vm_mul: A^T and w must have compatible inner dimensions");
#endif
      cblas_dgemv(CblasColMajor, CblasNoTrans, A->n_rows, A->n_cols,
          1., A->data, A->n_rows, w->data, 1, 0., v->data, 1);
      break;
    default:
      SB_CHK_ERR(true, abort(), "sb_mat_vm_mul: op must 'n' or 't'");
      break;
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_vec_is_finite(v), abort(), "sb_mat_vm_mul: elements not finite");
#endif
  return v;
}

/// Matrix multiplication of `op(B)` and `op(C)`, where `op` can be nothing or
/// a transpose. The result is stored in `A`, and any previous values in `A`
/// are overwritten. `A`, `B` and `C` must not overlap in memory.
///
/// # Parameters
/// - `A`: pointer to the resulting matrix
/// - `B`: pointer to the left matrix
/// - `C`: pointer to the right matrix
/// - `ops`: one of "nn", "nt", "tn", or "tt"
///
/// # Returns
/// A copy of `A`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A`, `B` and `C` are not `NULL`
/// - `SAFE_LENGTH`: `A`, `B` and `C` have compatible dimensions
/// - `SAFE_FINITE`: elements of `A` are finite
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2., 8.};
///   sb_mat * A = sb_mat_malloc(3, 3);
///   sb_mat * B = sb_mat_of_arr(a,     3, 3);
///   sb_mat * C = sb_mat_of_arr(a + 1, 3, 3);
/// 
///   // Multiply B by C and store in A
///   sb_mat_print(B, "B: ", "%g");
///   sb_mat_print(C, "C: ", "%g");
///   sb_mat_mm_mul(A, B, C, "nn");
///   sb_mat_print(A, "A: ", "%g");
///
///   SB_MAT_FREE_ALL(A, B, C);
/// }
/// ```
sb_mat * sb_mat_mm_mul(
    sb_mat * restrict A,
    const sb_mat * restrict B,
    const sb_mat * restrict C,
    const char * restrict ops) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_mm_mul: A cannot be NULL");
  SB_CHK_ERR(!B, abort(), "sb_mat_mm_mul: B cannot be NULL");
  SB_CHK_ERR(!C, abort(), "sb_mat_mm_mul: C cannot be NULL");
#endif
  switch ((*ops << 8) + *(ops + 1)) {
    case 0x6E6E: // "nn"
#ifdef SAFE_LENGTH
      SB_CHK_ERR(A->n_rows != B->n_rows, abort(),
          "sb_mat_mm_mul: A and B must have same number of rows");
      SB_CHK_ERR(A->n_cols != C->n_cols, abort(),
          "sb_mat_mm_mul: A and C must have same number of columns");
      SB_CHK_ERR(B->n_cols != C->n_rows, abort(),
          "sb_mat_mm_mul: B and C must have compatible inner dimensions");
#endif
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A->n_rows, A->n_cols, B->n_cols,
          1., B->data, B->n_rows, C->data, C->n_rows, 0., A->data, A->n_rows);
      break;
    case 0x6E74: // "nt"
#ifdef SAFE_LENGTH
      SB_CHK_ERR(A->n_rows != B->n_rows, abort(),
          "sb_mat_mm_mul: A and B must have same number of rows");
      SB_CHK_ERR(A->n_cols != C->n_rows, abort(),
          "sb_mat_mm_mul: A and C^T must have same number of columns");
      SB_CHK_ERR(B->n_cols != C->n_cols, abort(),
          "sb_mat_mm_mul: B and C^T must have compatible inner dimensions");
#endif
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, A->n_rows, A->n_cols, B->n_cols,
          1., B->data, B->n_rows, C->data, C->n_rows, 0., A->data, A->n_rows);
      break;
    case 0x746E: // "tn"
#ifdef SAFE_LENGTH
      SB_CHK_ERR(A->n_rows != B->n_cols, abort(),
          "sb_mat_mm_mul: A and B^T must have same number of rows");
      SB_CHK_ERR(A->n_cols != C->n_cols, abort(),
          "sb_mat_mm_mul: A and C must have same number of columns");
      SB_CHK_ERR(B->n_rows != C->n_rows, abort(),
          "sb_mat_mm_mul: B^T and C must have compatible inner dimensions");
#endif
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, A->n_rows, A->n_cols, B->n_rows,
          1., B->data, B->n_rows, C->data, C->n_rows, 0., A->data, A->n_rows);
      break;
    case 0x7474: // "tt"
#ifdef SAFE_LENGTH
      SB_CHK_ERR(A->n_rows != B->n_cols, abort(),
          "sb_mat_mm_mul: A and B^T must have same number of rows");
      SB_CHK_ERR(A->n_cols != C->n_rows, abort(),
          "sb_mat_mm_mul: A and C^T must have same number of columns");
      SB_CHK_ERR(B->n_rows != C->n_cols, abort(),
          "sb_mat_mm_mul: B^T and C^T must have compatible inner dimensions");
#endif
      cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, A->n_rows, A->n_cols, B->n_rows,
          1., B->data, B->n_rows, C->data, C->n_rows, 0., A->data, A->n_rows);
      break;
    default:
      SB_CHK_ERR(true, abort(),
          "sb_mat_mm_mul: ops must be one of \"nn\", \"nt\", \"tn\", or \"tt\"");
      break;
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_mat_is_finite(A), abort(), "sb_mat_mm_mul: elements not finite");
#endif
  return A;
}

/// Sums over the rows or columns of the matrix `A`. The result is stored in
/// `v`, and any previous values in `v` are overwritten. `v` and `A` must not
/// overlap in memory.
///
/// # Parameters
/// - `v`: pointer to the resulting vector
/// - `A`: pointer to the matrix to be summed
/// - `dir`: one of 'r' or 'c'
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` and `A` are not `NULL`
/// - `SAFE_LAYOUT`: the layout of `v` is consistent with `dir`
/// - `SAFE_LENGTH`: `v` and `A` have compatible dimensions
/// - `SAFE_FINITE`: elements of `v` are finite
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_vec * v = sb_vec_malloc(3, 'c');
///   sb_vec * w = sb_vec_malloc(3, 'r');
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
/// 
///   // Sum over the rows of A
///   sb_mat_print(A, "A: ", "%g");
///   sb_mat_sum(v, A, 'r');
///   sb_vec_print(v, "v: ", "%g");
///   sb_mat_sum(w, A, 'c');
///   sb_vec_print(w, "w: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
///   SB_VEC_FREE_ALL(v, w);
/// }
/// ```
sb_vec * sb_mat_sum(sb_vec * restrict v, const sb_mat * restrict A, const char dir) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_mat_sum: v cannot be NULL");
  SB_CHK_ERR(!A, abort(), "sb_mat_sum: A cannot be NULL");
#endif
  size_t n_rows = A->n_rows;
  size_t n_cols = A->n_cols;
  double * A_data = A->data;
  double * v_data = v->data;
  switch (dir) {
    case 'c':
#ifdef SAFE_LAYOUT
      SB_CHK_ERR(v->layout != 'r', abort(),
          "sb_mat_sum: v must be a row vector");
#endif
#ifdef SAFE_LENGTH
      SB_CHK_ERR(v->n_elem != A->n_cols, abort(),
          "sb_mat_sum: v and A must have same number of columns");
#endif
      sb_vec_set_zero(v);
      for (size_t a = 0; a < n_rows; ++a) {
        cblas_daxpy(n_cols, 1., A_data + a, n_rows, v_data, 1);
      }
      break;
    case 'r':
#ifdef SAFE_LAYOUT
      SB_CHK_ERR(v->layout != 'c', abort(),
          "sb_mat_sum: v must be a column vector");
#endif
#ifdef SAFE_LENGTH
      SB_CHK_ERR(v->n_elem != A->n_rows, abort(),
          "sb_mat_sum: v and A must have same number of rows");
#endif
      sb_vec_set_zero(v);
      for (size_t a = 0; a < n_cols; ++a) {
        cblas_daxpy(n_rows, 1., A_data + a * n_rows, 1, v_data, 1);
      }
      break;
    default:
      SB_CHK_ERR(true, abort(), "sb_mat_sum: dir must 'c' or 'r'");
      break;
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_vec_is_finite(v), abort(), "sb_mat_sum: elements not finite");
#endif
  return v;
}

/// Converts the column vector `*v` into a matrix with the indicated number of
/// rows and columns. The number of elements is preserved. NOTE: `*v` is freed
/// before the function returns and is set to `NULL`. The returned matrix
/// should be freed as usual.
///
/// # Parameters
/// - `v`: pointer to pointer to the vector
/// - `n_rows`: number of rows of the returned matrix
/// - `n_cols`: number of columns of the returned matrix
///
/// # Returns
/// A pointer to a matrix containing the elements of `*v` in column-major order
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `*v` is not `NULL`
/// - `SAFE_LAYOUT`: `*v` is a column vector
/// - `SAFE_LENGTH`: the number of elements in `*v` is `n_rows * n_cols`
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 9, 'c');
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
/// 
///   // A is equal to B
///   sb_mat * B = sb_vec_to_mat(&v, 3, 3);
///   assert(sb_mat_is_equal(A, B));
///
///   SB_MAT_FREE_ALL(A, B);
///   //SB_VEC_FREE_ALL(v); // <-- double free
/// }
/// ```
sb_mat * sb_vec_to_mat(sb_vec ** v, size_t n_rows, size_t n_cols) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!(*v), abort(), "sb_vec_to_mat: *v cannot be NULL");
#endif
#ifdef SAFE_LAYOUT
  SB_CHK_ERR((*v)->layout != 'c', abort(),
      "sb_vec_to_mat: *v must be a column vector");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR((*v)->n_elem != n_rows * n_cols, abort(),
      "sb_vec_to_mat: number of elements must be preserved");
#endif
  sb_mat * out = malloc(sizeof(sb_mat));
  SB_CHK_ERR(!out, return NULL, "sb_vec_to_mat: failed to allocate matrix");

  out->n_rows = n_rows;
  out->n_cols = n_cols;
  out->n_elem = (*v)->n_elem;
  out->data   = (*v)->data;

  free(*v);
  *v = NULL;
  return out;
}

/// Converts the matrix `A` into a column vector. The number of elements is
/// preserved. NOTE: `*A` is freed before the function returns and is set to
/// `NULL`, but the returned vector should be freed as usual.
///
/// # Parameters
/// - `A`: pointer to pointer to the matrix
///
/// # Returns
/// A pointer to a vector containing the elements of `*A` in column-major order
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `*A` is not `NULL`
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
///   sb_vec * v = sb_vec_of_arr(a, 9, 'c');
/// 
///   // v is equal to w
///   sb_vec * w = sb_mat_to_sb_vec(&A);
///   assert(sb_vec_is_equal(v, w));
///
///   SB_VEC_FREE_ALL(v, w);
///   //SB_MAT_FREE_ALL(A); // <-- double free
/// }
/// ```
sb_vec * sb_mat_to_vec(sb_mat ** A) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!(*A), abort(), "sb_mat_to_vec: *A cannot be NULL");
#endif
  sb_vec * out = malloc(sizeof(sb_vec));
  SB_CHK_ERR(!out, return NULL, "sb_mat_to_vec: failed to allocate vector");

  out->n_elem = (*A)->n_elem;
  out->data   = (*A)->data;
  out->layout = 'c';

  free(*A);
  *A = NULL;
  return out;
}

/// Reshapes the matrix `A` to have the indicated number of rows and columns
/// with the elements in column-major order. The number of elements must be
/// preserved.
///
/// # Parameters
/// - `A`: pointer to the matrix
/// - `n_rows`: number of rows in the returned matrix
/// - `n_cols`: number of columns in the returned matrix
///
/// # Returns
/// A copy of `A`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
/// - `SAFE_LENGTH`: the number of elements in `A` is `n_rows * n_cols`
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4.};
///   sb_mat * A = sb_mat_of_arr(a, 4, 2);
///   sb_mat * B = sb_mat_of_arr(a, 2, 4);
/// 
///   // A is reshaped to equal B
///   sb_mat_reshape(A, 2, 4);
///   assert(sb_mat_is_equal(A, B));
///
///   SB_MAT_FREE_ALL(A, B);
/// }
/// ```
sb_mat * sb_mat_reshape(sb_mat * A, size_t n_rows, size_t n_cols) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_reshape: A cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(A->n_elem != n_rows * n_cols, abort(),
      "sb_mat_reshape: number of elements must be preserved");
#endif
  A->n_rows = n_rows;
  A->n_cols = n_cols;
  return A;
}

/// Transposes the matrix `A` using a memory allocation for scratch space. NOTE:
/// Since the matrix multiplication functions allow the matrix to be transposed
/// at lower cost, this function is usually unnecesary.
///
/// # Parameters
/// - `A`: pointer to the matrix
///
/// # Returns
/// A copy of `A`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
///
/// # Examples
/// ```
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4.};
///   sb_mat * A = sb_mat_of_arr(a, 4, 2);
/// 
///   // A is transposed
///   sb_mat_print(A, "A before: ", "%g");
///   sb_mat_trans(A);
///   sb_mat_print(A, "A after: ", "%g");
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
sb_mat * sb_mat_trans(sb_mat * A) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_trans: A cannot be NULL");
#endif
  size_t n_elem = A->n_elem;
  double * T_data = malloc(n_elem * sizeof(double));
  SB_CHK_ERR(!T_data, return A, "sb_mat_trans: failed to allocate memory");

  size_t n_rows = A->n_rows;
  size_t n_cols = A->n_cols;
  double * A_data = A->data;
  for (size_t a = 0; a < n_rows; ++a) {
    cblas_dcopy(n_cols, A_data + a, n_rows, T_data + a * n_cols, 1);
  }

  A->n_rows = n_cols;
  A->n_cols = n_rows;
  A->data = T_data;
  free(A_data);

  return A;
}

/// Checks the matrices `A` and `B` for equality of elements. Matrices must
/// have the same number of rows and columns.
///
/// # Parameters
/// - `A`: pointer to the first matrix
/// - `B`: pointer to the second matrix
///
/// # Returns
/// `1` if `A` and `B` are equal, and `0` otherwise
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` and `B` are not `NULL`
/// - `SAFE_LENGTH`: `A` and `B` have the same number of rows and columns
/// - `SAFE_FINITE`: elements of `A` and `B` are finite
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2., 8.};
///   sb_mat * A = sb_mat_of_arr(a,     3, 3);
///   sb_mat * B = sb_mat_of_arr(a + 1, 3, 3);
/// 
///   // A and B are equal after sb_mat_memcpy
///   assert(!sb_mat_is_equal(A, B));
///   sb_mat_memcpy(B, A);
///   assert(sb_mat_is_equal(A, B));
///
///   SB_MAT_FREE_ALL(A, B);
/// }
/// ```
int sb_mat_is_equal(const sb_mat * restrict A, const sb_mat * restrict B) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_is_equal: A cannot be NULL");
  SB_CHK_ERR(!B, abort(), "sb_mat_is_equal: B cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(A->n_rows != B->n_rows, abort(),
      "sb_mat_is_equal: A and B must have same number of rows");
  SB_CHK_ERR(A->n_cols != B->n_cols, abort(),
      "sb_mat_is_equal: A and B must have same number of columns");
#endif
  double * A_data = A->data;
  double * B_data = B->data;
  for (size_t a = 0; a < A->n_elem; ++a) {
#ifdef SAFE_FINITE
    SB_CHK_ERR(!isfinite(A_data[a]), abort(), "sb_mat_is_equal: A is not finite");
    SB_CHK_ERR(!isfinite(B_data[a]), abort(), "sb_mat_is_equal: B is not finite");
#endif
    if (A_data[a] != B_data[a]) {
      return 0;
    }
  }
  return 1;
}

/// Checks if all elements of the matrix `A` are zero.
///
/// # Parameters
/// - `A`: pointer to the matrix
///
/// # Returns
/// `1` if all elements of `A` are zero, and `0` otherwise
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
/// 
///   // A is zero after sb_mat_set_zero
///   assert(!sb_mat_is_zero(A));
///   sb_mat_set_zero(A);
///   assert(sb_mat_is_zero(A));
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
int sb_mat_is_zero(const sb_mat * A) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_is_zero: A cannot be NULL");
#endif
  double * data = A->data;
  for (size_t a = 0; a < A->n_elem; ++a) {
    if (data[a] != 0.) {
      return 0;
    }
  }
  return 1;
}

/// Checks if all elements of the matrix `A` are positive.
///
/// # Parameters
/// - `A`: pointer to the matrix
///
/// # Returns
/// `1` if all elements of `A` are positive, and `0` otherwise
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
/// - `SAFE_FINITE`: elements of `A` are finite
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., -4., 2., -8., 5., 7., 1., 4., -2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
/// 
///   // A is positive after sb_mat_abs
///   assert(!sb_mat_is_pos(A));
///   sb_mat_abs(A);
///   assert(sb_mat_is_pos(A));
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
int sb_mat_is_pos(const sb_mat * A) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_is_pos: A cannot be NULL");
#endif
  double * data = A->data;
  for (size_t a = 0; a < A->n_elem; ++a) {
#ifdef SAFE_FINITE
    SB_CHK_ERR(!isfinite(data[a]), abort(), "sb_mat_is_pos: A is not finite");
#endif
    if (data[a] <= 0.) {
      return 0;
    }
  }
  return 1;
}

/// Checks if all elements of the matrix `A` are negative.
///
/// # Parameters
/// - `A`: pointer to the matrix
///
/// # Returns
/// `1` if all elements of `A` are negative, and `0` otherwise
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
/// - `SAFE_FINITE`: elements of `A` are finite
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., -4., 2., -8., 5., 7., 1., 4., -2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
/// 
///   // A is negative after
///   assert(!sb_mat_is_neg(A));
///   sb_mat_smul(sb_mat_abs(A), -1.);
///   assert(sb_mat_is_neg(A));
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
int sb_mat_is_neg(const sb_mat * A) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_is_neg: A cannot be NULL");
#endif
  double * data = A->data;
  for (size_t a = 0; a < A->n_elem; ++a) {
#ifdef SAFE_FINITE
    SB_CHK_ERR(!isfinite(data[a]), abort(), "sb_mat_is_neg: A is not finite");
#endif
    if (data[a] >= 0.) {
      return 0;
    }
  }
  return 1;
}

/// Checks if all elements of the matrix `A` are nonnegative.
///
/// # Parameters
/// - `A`: pointer to the matrix
///
/// # Returns
/// `1` if all elements of `A` are nonnegative, and `0` otherwise
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
/// - `SAFE_FINITE`: elements of `A` are finite
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {0., -4., 2., -8., 5., 7., 1., 4., -2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
/// 
///   // A is nonnegative after
///   assert(!sb_mat_is_nonneg(A));
///   sb_mat_abs(A);
///   assert( sb_mat_is_nonneg(A));
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
int sb_mat_is_nonneg(const sb_mat * A) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_is_nonneg: A cannot be NULL");
#endif
  double * data = A->data;
  for (size_t a = 0; a < A->n_elem; ++a) {
#ifdef SAFE_FINITE
    SB_CHK_ERR(!isfinite(data[a]), abort(), "sb_mat_is_nonneg: A is not finite");
#endif
    if (data[a] < 0.) {
      return 0;
    }
  }
  return 1;
}

/// Checks if all elements of the matrix `A` are not infinite or `NaN`.
///
/// # Parameters
/// - `A`: pointer to the matrix
///
/// # Returns
/// `1` if all elements of `A` are not infinite or `NaN`, and `0` otherwise
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include <math.h>
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., INFINITY, 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
/// 
///   assert(!sb_mat_is_finite(A));
///   sb_mat_set(A, 1, 0, NAN);
///   assert(!sb_mat_is_finite(A));
///   sb_mat_set(A, 1, 0, 4.);
///   assert( sb_mat_is_finite(A));
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
int sb_mat_is_finite(const sb_mat * A) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_is_finite: A cannot be NULL");
#endif
  double * data = A->data;
  for (size_t a = 0; a < A->n_elem; ++a) {
    if (!isfinite(data[a])) {
      return 0;
    }
  }
  return 1;
}

/// Finds the value of the maximum element of `A`.
///
/// # Parameters
/// - `A`: pointer to the matrix
///
/// # Returns
/// Value of the maximum element
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
/// - `SAFE_LENGTH`: number of elements is nonzero
/// - `SAFE_FINITE`: elements of `A` are finite
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
/// 
///   // maximum value of A is 8.
///   assert(sb_mat_max(A) == 8.);
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
double sb_mat_max(const sb_mat * A) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_max: A cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(A->n_elem == 0, abort(), "sb_mat_max: n_elem must be nonzero");
#endif
  double * data = A->data;
  double max_val = -INFINITY;
  for (size_t a = 0; a < A->n_elem; ++a) {
#ifdef SAFE_FINITE
    SB_CHK_ERR(!isfinite(data[a]), abort(), "sb_mat_max: A is not finite");
#endif
    if (data[a] > max_val) {
      max_val = data[a];
    }
  }
  return max_val;
}

/// Finds the value of the minimum element of `A`.
///
/// # Parameters
/// - `A`: pointer to the matrix
///
/// # Returns
/// Value of the minimum element
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
/// - `SAFE_LENGTH`: number of elements is nonzero
/// - `SAFE_FINITE`: elements of `A` are finite
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
/// 
///   // value of min is 1
///   assert(sb_mat_min(A) == 1.);
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
double sb_mat_min(const sb_mat * A) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_min: A cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(A->n_elem == 0, abort(), "sb_mat_min: n_elem must be nonzero");
#endif
  double * data = A->data;
  double min_val = INFINITY;
  for (size_t a = 0; a < A->n_elem; ++a) {
#ifdef SAFE_FINITE
    SB_CHK_ERR(!isfinite(data[a]), abort(), "sb_mat_min: A is not finite");
#endif
    if (data[a] < min_val) {
      min_val = data[a];
    }
  }
  return min_val;
}

/// Finds the maximum absolute value of the elements of `A`.
///
/// # Parameters
/// - `A`: pointer to the matrix
///
/// # Returns
/// Maximum absolute value of the elements
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
/// - `SAFE_LENGTH`: number of elements is nonzero
/// - `SAFE_FINITE`: elements of `A` are finite
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., -4., 2., -8., 5., -7., 1., -4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
/// 
///   // maximum absolute value of A is 8.
///   assert(sb_mat_abs_max(A) == 8.);
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
double sb_mat_abs_max(const sb_mat * A) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_abs_max: A cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(A->n_elem == 0, abort(), "sb_mat_abs_max: n_elem must be nonzero");
#endif
  double * data = A->data;
  size_t index = cblas_idamax(A->n_elem, data, 1);
#ifdef SAFE_FINITE
  SB_CHK_ERR(!isfinite(data[index]), abort(), "sb_mat_abs_max: A is not finite");
#endif
  return fabs(data[index]);
}

/// Finds the index of the maximum element of `A`.
///
/// # Parameters
/// - `A`: pointer to the matrix
///
/// # Returns
/// Index of the maximum element
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
/// - `SAFE_LENGTH`: number of elements is nonzero
/// - `SAFE_FINITE`: elements of `A` are finite
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
/// 
///   // index of max is 3
///   assert(sb_mat_max_index(A) == 3);
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
size_t sb_mat_max_index(const sb_mat * A) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_max_index: A cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(A->n_elem == 0, abort(), "sb_mat_max_index: n_elem must be nonzero");
#endif
  double * data = A->data;
  double max_val = -INFINITY;
  size_t max_ind = 0;
  for (size_t a = 0; a < A->n_elem; ++a) {
#ifdef SAFE_FINITE
    SB_CHK_ERR(!isfinite(data[a]), abort(), "sb_mat_max_index: A is not finite");
#endif
    if (data[a] > max_val) {
      max_val = data[a];
      max_ind = a;
    }
  }
  return max_ind;
}

/// Finds the index of the minimum element of `A`.
///
/// # Parameters
/// - `A`: pointer to the matrix
///
/// # Returns
/// Index of the minimum element
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
/// - `SAFE_LENGTH`: number of elements is nonzero
/// - `SAFE_FINITE`: elements of `A` are finite
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7., 1., 4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
/// 
///   // index of min is 0
///   assert(sb_mat_min_index(A) == 0);
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
size_t sb_mat_min_index(const sb_mat * A) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_min_index: A cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(A->n_elem == 0, abort(), "sb_mat_min_index: n_elem must be nonzero");
#endif
  double * data = A->data;
  double min_val = INFINITY;
  size_t min_ind = 0;
  for (size_t a = 0; a < A->n_elem; ++a) {
#ifdef SAFE_FINITE
    SB_CHK_ERR(!isfinite(data[a]), abort(), "sb_mat_min_index: A is not finite");
#endif
    if (data[a] < min_val) {
      min_val = data[a];
      min_ind = a;
    }
  }
  return min_ind;
}

/// Finds the index of the maximum absolute value of the elements of `A`.
///
/// # Parameters
/// - `A`: pointer to the matrix
///
/// # Returns
/// Index of the maximum absolute value of the elements
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `A` is not `NULL`
/// - `SAFE_LENGTH`: number of elements is nonzero
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
///
/// int main(void) {
///   double a[] = {1., -4., 2., -8., 5., -7., 1., -4., 2.};
///   sb_mat * A = sb_mat_of_arr(a, 3, 3);
/// 
///   // index of the maximum absolute value of A is 3
///   assert(sb_mat_abs_max_index(A) == 3);
///
///   SB_MAT_FREE_ALL(A);
/// }
/// ```
size_t sb_mat_abs_max_index(const sb_mat * A) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!A, abort(), "sb_mat_abs_max_index: A cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(A->n_elem == 0, abort(),
      "sb_mat_abs_max_index: n_elem must be nonzero");
#endif
  return cblas_idamax(A->n_elem, A->data, 1);
}

extern inline double   sb_mat_get(const sb_mat * A, size_t i, size_t j);
extern inline void     sb_mat_set(sb_mat * A, size_t i, size_t j, double x);
extern inline double * sb_mat_ptr(sb_mat * A, size_t i, size_t j);
