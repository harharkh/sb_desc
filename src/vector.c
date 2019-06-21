// Copyright 2018 Jeremy Mason
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! \file vector.c
//! Contains functions to manipulate vectors. Intended as a relatively less
//! painful wrapper around Level 1 CBLAS.

#include <cblas.h>      // daxpy
#include <float.h>      // DBL_EPSILON
#include <math.h>       // exp
#include <stdbool.h>    // bool
#include <stddef.h>     // size_t
#include <stdio.h>      // EOF
#include <stdlib.h>     // abort
#include <string.h>     // memcpy
#include "sb_matrix.h"  // sb_mat_is_finite
#include "sb_structs.h" // sb_mat
#include "sb_utility.h" // SB_CHK_ERR
#include "sb_vector.h"
#include "safety.h"

/// Constructs a vector with the required capacity.
///
/// # Parameters
/// - `n_elem`: capacity of the vector
/// - `layout`: `c` for a column vector, `r` for a row vector
///
/// # Returns
/// A `sb_vec` pointer to the allocated vector, or `NULL` if the allocation fails
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_LAYOUT`: `layout` is `c` or `r`
/// 
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   sb_vec * v = sb_vec_malloc(3, 'r');
///
///   // Fill the vector with some values and print
///   double a[] = {1., 4., 2.};
///   sb_vec_subcpy(v, 0, a, 3);
///   sb_vec_print(v, "v: ", "%g");
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_vec * sb_vec_malloc(size_t n_elem, char layout) {
#ifdef SAFE_LAYOUT
  SB_CHK_ERR(layout != 'c' && layout != 'r', abort(),
      "sb_vec_malloc: layout must be 'c' or 'r'");
#endif
  sb_vec * out = malloc(sizeof(sb_vec));
  SB_CHK_ERR(!out, return NULL, "sb_vec_malloc: failed to allocate vector");

  double * data = malloc(n_elem * sizeof(double));
  SB_CHK_ERR(!data, free(out); return NULL, "sb_vec_malloc: failed to allocate data");

  out->n_elem = n_elem;
  out->data   = data;
  out->layout = layout;

  return out;
}

/// Constructs a vector with the required capacity and initializes all elements
/// to zero. Requires support for the IEC 60559 standard.
///
/// # Parameters
/// - `n_elem`: capacity of the vector
/// - `layout`: `c` for a column vector, `r` for a row vector
///
/// # Returns
/// A `sb_vec` pointer to the allocated vector, or `NULL` if the allocation fails
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_LAYOUT`: `layout` is `c` or `r`
/// 
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   sb_vec * v = sb_vec_calloc(3, 'r');
///
///   // Initialized to zeros
///   sb_vec_print(v, "v before: ", "%g");
///
///   // Fill the vector with some values and print
///   double a[] = {1., 4., 2.};
///   sb_vec_subcpy(v, 0, a, 3);
///   sb_vec_print(v, "v after: ", "%g");
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_vec * sb_vec_calloc(size_t n_elem, char layout) {
#ifdef SAFE_LAYOUT
  SB_CHK_ERR(layout != 'c' && layout != 'r', abort(),
      "sb_vec_calloc: layout must be 'c' or 'r'");
#endif
  sb_vec * out = malloc(sizeof(sb_vec));
  SB_CHK_ERR(!out, return NULL, "sb_vec_calloc: failed to allocate vector");

  double * data = calloc(n_elem, sizeof(double));
  SB_CHK_ERR(!data, free(out); return NULL, "sb_vec_calloc: failed to allocate data");

  out->n_elem = n_elem;
  out->data   = data;
  out->layout = layout;

  return out;
}

/// Constructs a vector with the required capacity and initializes elements to
/// the first `n_elem` elements of array `a`. The array must contain at least
/// `n_elem` elements.
///
/// # Parameters
/// - `a`: array to be copied into the vector
/// - `n_elem`: capacity of the vector
/// - `layout`: `c` for a column vector, `r` for a row vector
///
/// # Returns
/// A `sb_vec` pointer to the allocated vector, or `NULL` if the allocation fails
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `a` is not `NULL`
/// - `SAFE_LAYOUT`: `layout` is `c` or `r`
/// 
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
/// 
///   // Construct a vector from the array
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
///   sb_vec_print(v, "v: ", "%g");
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_vec * sb_vec_of_arr(const double * a, size_t n_elem, char layout) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!a, abort(), "sb_vec_of_arr: a cannot be NULL");
#endif
#ifdef SAFE_LAYOUT
  SB_CHK_ERR(layout != 'c' && layout != 'r', abort(),
      "sb_vec_of_arr: layout must be 'c' or 'r'");
#endif
  sb_vec * out = malloc(sizeof(sb_vec));
  SB_CHK_ERR(!out, return NULL, "sb_vec_of_arr: failed to allocate vector");

  double * data = malloc(n_elem * sizeof(double));
  SB_CHK_ERR(!data, free(out); return NULL, "sb_vec_of_arr: failed to allocate data");

  out->n_elem = n_elem;
  out->data   = memcpy(data, a, n_elem * sizeof(double));
  out->layout = layout;

  return out;
}

/// Constructs a vector as a deep copy of an existing vector. The state of the
/// existing vector must be valid.
///
/// # Parameters
/// - `v`: pointer to the vector to be copied
///
/// # Returns
/// A `sb_vec` pointer to the allocated vector, or `NULL` if the allocation fails
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // `v` and `w` contain the same elements
///   sb_vec * w = sb_vec_clone(v);
///   sb_vec_print(w, "w before: ", "%g");
///
///   // Fill `w` with some values and print
///   sb_vec_subcpy(w, 0, a + 3, 3);
///   sb_vec_print(w, "w after: ", "%g");
///
///   // `v` is unchanged
///   sb_vec_print(v, "v after: ", "%g");
///
///   SB_VEC_FREE_ALL(v, w);
/// }
/// ```
sb_vec * sb_vec_clone(const sb_vec * v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_clone: v cannot be NULL");
#endif
  sb_vec * out = malloc(sizeof(sb_vec));
  SB_CHK_ERR(!out, return NULL, "sb_vec_clone: failed to allocate vector");

  size_t n_elem = v->n_elem;

  double * data = malloc(n_elem * sizeof(double));
  SB_CHK_ERR(!data, free(out); return NULL, "sb_vec_clone: failed to allocate data");

  *out = *v;
  out->data = memcpy(data, v->data, n_elem * sizeof(double));

  return out;
}

/// Constructs a column vector containing `n_elem` elements in equal intervals
/// from `begin` to `end`. Must contain at least one element.
///
/// # Parameters
/// - `begin`: beginning of the interval
/// - `end`: end of the interval
/// - `step`: step within the interval
///
/// # Returns
/// A `sb_vec` pointer to the allocated vector, or `NULL` if the allocation fails
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_LENGTH`: `v` contains at least two elements
/// - `SAFE_MEMORY`: `v` is not `NULL`
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   sb_vec * v = sb_vec_linear(4., 0., 5);
///   sb_vec_print(v, "v: ", "%g");
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_vec * sb_vec_linear(double begin, double end, size_t n_elem) {
#ifdef SAFE_LENGTH
  SB_CHK_ERR(n_elem < 2, abort(), "sb_vec_range: must contain at least two elements");
#endif
  sb_vec * out = malloc(sizeof(sb_vec));
  SB_CHK_ERR(!out, return NULL, "sb_vec_range: failed to allocate sb_vector");

  double * data = malloc(n_elem * sizeof(double));
  SB_CHK_ERR(!data, free(out); return NULL, "sb_vec_range: failed to allocate data");

  size_t n_elem_1 = n_elem - 1;
  double step = (end - begin) / n_elem_1;
  for (size_t a = 0; a < n_elem_1; ++a) {
    data[a] = fma(a, step, begin);
  }
  data[n_elem_1] = end;

  out->n_elem = n_elem;
  out->data   = data;
  out->layout = 'c';

  return out;
}

/// Deconstructs a vector.
///
/// # Parameters
/// - `v`: pointer to the vector
///
/// # Returns
/// No return value
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
/// 
///   // Allocates a pointer to vec
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
///   sb_vec_print(v, "v: ", "%g");
///
///   sb_vec_free(v);
///   // Pointer to `v` is now invalid
/// }
/// ```
void sb_vec_free(sb_vec * v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_free: v cannot be NULL");
#endif
  free(v->data);
  free(v);
}

/// Sets all elements of `v` to zero. Requires support for the IEC 60559
/// standard.
///
/// # Parameters
/// - `v`: pointer to the vector
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
///   
///   // Set all elements to zero
///   sb_vec_print(v, "v before: ", "%g");
///   sb_vec_set_zero(v);
///   sb_vec_print(v, "v after: ", "%g");
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_vec * sb_vec_set_zero(sb_vec * v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_set_zero: v cannot be NULL");
#endif
  return memset(v->data, 0, v->n_elem * sizeof(double));
}

/// Sets all elements of `v` to `x`.
///
/// # Parameters
/// - `v`: pointer to the vector
/// - `x`: value for the elements
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
///   
///   // Set all elements to 8.
///   sb_vec_print(v, "v before: ", "%g");
///   sb_vec_set_all(v, 8.);
///   sb_vec_print(v, "v after: ", "%g");
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_vec * sb_vec_set_all(sb_vec * v, double x) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_set_all: v cannot be NULL");
#endif
  double * data = v->data;
  for (size_t a = 0; a < v->n_elem; ++a) {
    data[a] = x;
  }
  return v;
}

/// Sets all elements of `v` to zero, except for the `i`th element which is set
/// to one. Requires support for the IEC 60559 standard.
///
/// # Parameters
/// - `v`: pointer to the vector
/// - `i`: index of the element with value one
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
/// - `SAFE_LENGTH`: `i` is a valid index
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
///   
///   // Set `v` to second basis vector
///   sb_vec_print(v, "v before: ", "%g");
///   sb_vec_set_basis(v, 1);
///   sb_vec_print(v, "v after: ", "%g");
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_vec * sb_vec_set_basis(sb_vec * v, size_t i) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_set_basis: v cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(i >= v->n_elem, abort(), "sb_vec_set_basis: index out of bounds");
#endif
  ((double *) memset(v->data, 0, v->n_elem * sizeof(double)))[i] = 1.;
  return v;
}

/// Copies contents of the `src` vector into the `dest` vector. `src` and `dest`
/// must have the same length, the same layout, and not overlap in memory.
///
/// # Parameters
/// - `dest`: pointer to destination vector
/// - `src`: const pointer to source vector
///
/// # Returns
/// A copy of `dest`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `src` and `dest` are not `NULL`
/// - `SAFE_LAYOUT`: `src` and `dest` have same layout
/// - `SAFE_LENGTH`: `src` and `dest` have same length
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7.};
///   sb_vec * v = sb_vec_of_arr(a,     3, 'r');
///   sb_vec * w = sb_vec_of_arr(a + 3, 3, 'r');
///
///   // Overwrite elements of `v` and print
///   sb_vec_print(v, "v before: ", "%g");
///   sb_vec_print(w, "w before: ", "%g");
///   sb_vec_memcpy(v, w);
///   sb_vec_print(v, "v after: ", "%g");
///   sb_vec_print(w, "w after: ", "%g");
///
///   SB_VEC_FREE_ALL(v, w);
/// }
/// ```
sb_vec * sb_vec_memcpy(sb_vec * restrict dest, const sb_vec * restrict src) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!dest, abort(), "sb_vec_memcpy: dest cannot be NULL");
  SB_CHK_ERR(!src, abort(), "sb_vec_memcpy: src cannot be NULL");
#endif
#ifdef SAFE_LAYOUT
  SB_CHK_ERR(dest->layout != src->layout, abort(),
      "sb_vec_memcpy: dest and src must have same layout");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(dest->n_elem != src->n_elem, abort(),
      "sb_vec_memcpy: dest and src must have same length");
#endif
  memcpy(dest->data, src->data, src->n_elem * sizeof(double));
  return dest;
}

/// Copies `n` elements of array `a` into vector `v` starting at index `i`. `v`
/// must have enough capacity, `a` must contain at least `n` elements, and `v`
/// and `a` must not overlap in memory.
///
/// # Parameters
/// - `v`: pointer to destination vector
/// - `i`: index of `v` where the copy will start
/// - `a`: pointer to elements that will be copied
/// - `n`: number of elements to copy
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` and `a` are not `NULL`
/// - `SAFE_LENGTH`: `v` has enough capacity
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
///
///   // Overwrite elements of `v` and print
///   sb_vec_print(v, "v before: ", "%g");
///   sb_vec_subcpy(v, 1, a + 3, 2);
///   sb_vec_print(v, "v after: ", "%g");
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_vec * sb_vec_subcpy(
    sb_vec * restrict v,
    size_t i,
    const double * restrict a,
    size_t n) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_subcpy: v cannot be NULL");
  SB_CHK_ERR(!a, abort(), "sb_vec_subcpy: a cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(v->n_elem - i < n, abort(),
      "sb_vec_subcpy: v does not have enough capacity");
#endif
  memcpy(v->data + i, a, n * sizeof(double));
  return v;
}

/// Swaps the contents of `v` and `w` by exchanging data pointers. Vectors must
/// have the same length and layout and not overlap in memory.
///
/// # Parameters
/// - `v`: pointer to the first vector
/// - `w`: pointer to the second vector
///
/// # Returns
/// No return value
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` and `w` are not `NULL`
/// - `SAFE_LAYOUT`: `v` and `w` have the same layout
/// - `SAFE_LENGTH`: `v` and `w` have the same length
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7.};
///   sb_vec * v = sb_vec_of_arr(a,     3, 'r');
///   sb_vec * w = sb_vec_of_arr(a + 3, 3, 'r');
/// 
///   // Print vectors before and after
///   sb_vec_print(v, "v before: ", "%g");
///   sb_vec_print(w, "w before: ", "%g");
///
///   sb_vec_swap(v, w);
///
///   sb_vec_print(v, "v after: ", "%g");
///   sb_vec_print(w, "w after: ", "%g");
///
///   SB_VEC_FREE_ALL(v, w);
/// }
/// ```
void sb_vec_swap(sb_vec * restrict v, sb_vec * restrict w) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_swap: v cannot be NULL");
  SB_CHK_ERR(!w, abort(), "sb_vec_swap: w cannot be NULL");
#endif
#ifdef SAFE_LAYOUT
  SB_CHK_ERR(v->layout != w->layout, abort(),
      "sb_vec_swap: v and w must have same layout");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(v->n_elem != w->n_elem, abort(),
      "sb_vec_swap: v and w must have same length");
#endif
  double * scratch;
  SB_SWAP(v->data, w->data, scratch);
}

/// Swaps the `i`th and `j`th elements of a vector.
///
/// # Parameters
/// - `v`: pointer to the vector
/// - `i`: index of first element
/// - `j`: index of second element
///
/// # Returns
/// No return value
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
/// - `SAFE_LENGTH`: `i` and `j` are valid indices
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // Print vector before and after
///   sb_vec_print(v, "v before: ", "%g");
///   sb_vec_swap_elems(v, 0, 1);
///   sb_vec_print(v, "v after: ", "%g");
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
void sb_vec_swap_elems(sb_vec * v, size_t i, size_t j) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_swap_elems: v cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(i >= v->n_elem, abort(), "sb_vec_swap_elems: index out of bounds");
  SB_CHK_ERR(j >= v->n_elem, abort(), "sb_vec_swap_elems: index out of bounds");
#endif
  double * data = v->data;
  double scratch;
  SB_SWAP(data[i], data[j], scratch);
}

/// Writes the vector `v` to `stream` in a binary format. The data is written
/// in the native binary format of the architecture, and may not be portable.
///
/// # Parameters
/// - `stream`: an open I/O stream
/// - `v`: pointer to the vector
///
/// # Returns
/// `0` on success, or `1` if the write fails
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
///
/// # Examples
/// ```
/// #include <stdio.h>
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
///   
///   // Write the vector to file
///   FILE * f = fopen("vector.bin", "wb");
///   sb_vec_fwrite(f, v);
///   fclose(f);
///   
///   // Read the vector from file
///   FILE * g = fopen("vector.bin", "rb");
///   sb_vec * w = sb_vec_fread(g);
///   fclose(g);
///   
///   // Vectors have the same contents
///   sb_vec_print(v, "v: ", "%g");
///   sb_vec_print(w, "w: ", "%g");
///   
///   SB_VEC_FREE_ALL(v, w);
/// }
/// ```
int sb_vec_fwrite(FILE * stream, const sb_vec * v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_fwrite: v cannot be NULL");
#endif
  size_t n_write;

  size_t n_elem = v->n_elem; 
  n_write = fwrite(&n_elem, sizeof(size_t), 1, stream);
  SB_CHK_ERR(n_write != 1, return 1, "sb_vec_fwrite: fwrite failed");

  n_write = fwrite(&(v->layout), sizeof(char), 1, stream);
  SB_CHK_ERR(n_write != 1, return 1, "sb_vec_fwrite: fwrite failed");

  n_write = fwrite(v->data, sizeof(double), n_elem, stream);
  SB_CHK_ERR(n_write != n_elem, return 1, "sb_vec_fwrite: fwrite failed");

  return 0;
}

/// Reads binary data from `stream` into the vector returned by the function.
/// Writes the vector `v` to `stream`. The number of elements, layout, and
/// elements are written in a human readable format.
///
/// # Parameters
/// - `stream`: an open I/O stream
/// - `v`: pointer to the vector
/// - `format`: a format specifier for the elements
///
/// # Returns
/// `0` on success, or `1` if the write fails
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
///
/// # Examples
/// ```
/// #include <stdio.h>
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double f[] = {1., 4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
///   
///   // Write the vector to file
///   FILE * f = fopen("vector.txt", "w");
///   sb_vec_fprintf(f, v, "%lg");
///   fclose(f);
///   
///   // Read the sb_vector from file
///   FILE * g = fopen("vector.txt", "r");
///   sb_vec * w = sb_vec_fscanf(g);
///   fclose(g);
///   
///   // Vectors have the same contents
///   sb_vec_print(v, "v: ", "%g");
///   sb_vec_print(w, "w: ", "%g");
///   
///   SB_VEC_FREE_ALL(v, w);
/// }
/// ```
int sb_vec_fprintf(FILE * stream, const sb_vec * v, const char * format) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_fprintf: v cannot be NULL");
#endif
  int status;

  size_t n_elem = v->n_elem; 
  status = fprintf(stream, "%zu %c", n_elem, v->layout);
  SB_CHK_ERR(status < 0, return 1, "sb_vec_fprintf: fprintf failed");

  double * data = v->data;
  for (size_t a = 0; a < n_elem; ++a) {
    status = putc(' ', stream);
    SB_CHK_ERR(status == EOF, return 1, "sb_vec_fprintf: putc failed");
    status = fprintf(stream, format, data[a]);
    SB_CHK_ERR(status < 0, return 1, "sb_vec_fprintf: fprintf failed");
  }
  status = putc('\n', stream);
  SB_CHK_ERR(status == EOF, return 1, "sb_vec_fprintf: putc failed");

  return 0;
}

/// Prints the vector `v` to stdout. Output is slightly easier to read than for
/// `sb_vec_fprintf()`. Mainly indended for debugging.
///
/// # Parameters
/// - `v`: pointer to the vector
/// - `str`: a string to describe the vector
/// - `format`: a format specifier for the elements
///
/// # Returns
/// `0` on success, or `1` if the print fails
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double array[] = {1., 4., 2., 8., 5., 7.};
///   sb_vec * v = sb_vec_of_arr(array,     3, 'r');
///   sb_vec * w = sb_vec_of_arr(array + 3, 3, 'c');
///
///   // Prints the contents of `v` and `w` to stdout
///   sb_vec_print(v, "v: ", "%g");
///   sb_vec_print(w, "w: ", "%g");
///
///   SB_VEC_FREE_ALL(v, w);
/// }
/// ```
int sb_vec_print(const sb_vec * v, const char * str, const char * format) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_print: v cannot be NULL");
#endif
  int status;

  status = printf("%s\n", str);
  SB_CHK_ERR(status < 0, return 1, "sb_vec_print: printf failed");

  size_t n_elem = v->n_elem;
  double * data = v->data;

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
    SB_CHK_ERR(status < 0, return 1, "sb_vec_print: snprintf failed");
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

  for (size_t a = 0; a < n_elem; ++a) {
    for (unsigned char s = 0; s < max_head - len_head[a]; ++s) {
      status = putchar(' ');
      SB_CHK_ERR(status == EOF, return 1, "sb_vec_print: putchar failed");
    }
    status = printf(format, data[a]);
    SB_CHK_ERR(status < 0, return 1, "sb_vec_print: printf failed");
    if (any_mark && !dec_mark[a]) {
      status = putchar(' ');
      SB_CHK_ERR(status == EOF, return 1, "sb_vec_print: putchar failed");
    }
    for (unsigned char s = 0; s < max_tail - len_tail[a]; ++s) {
      status = putchar(' ');
      SB_CHK_ERR(status == EOF, return 1, "sb_vec_print: putchar failed");
    }
    status = putchar(v->layout == 'r' ? ' ' : '\n');
    SB_CHK_ERR(status == EOF, return 1, "sb_vec_print: putchar failed");
  }
  if (v->layout == 'r') {
    status = putchar('\n');
    SB_CHK_ERR(status == EOF, return 1, "sb_vec_print: putchar failed");
  }

  return 0;
}

/// The data must be written in the native binary format of the architecture, 
/// preferably by `sb_vec_fwrite()`.
///
/// # Parameters
/// - `stream`: an open I/O stream
///
/// # Returns
/// A `sb_vec` pointer to the sb_vector read from `stream`, or `NULL` if the read or 
/// memory allocation fails
/// 
/// # Examples
/// ```
/// #include <stdio.h>
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
///   
///   // Write the vector to file
///   FILE * f = fopen("vector.bin", "wb");
///   sb_vec_fwrite(f, v);
///   fclose(f);
///   
///   // Read the vector from file
///   FILE * g = fopen("vector.bin", "rb");
///   sb_vec * w = sb_vec_fread(g);
///   fclose(g);
///   
///   // Vectors have the same contents
///   sb_vec_print(v, "v: ", "%g");
///   sb_vec_print(w, "w: ", "%g");
///   
///   SB_VEC_FREE_ALL(v, w);
/// }
/// ```
sb_vec * sb_vec_fread(FILE * stream) {
  size_t n_read;

  size_t n_elem;
  n_read = fread(&n_elem, sizeof(size_t), 1, stream);
  SB_CHK_ERR(n_read != 1, return NULL, "sb_vec_fread: fread failed");

  char layout;
  n_read = fread(&layout, sizeof(char), 1, stream);
  SB_CHK_ERR(n_read != 1, return NULL, "sb_vec_fread: fread failed");

  sb_vec * out = sb_vec_malloc(n_elem, layout);
  SB_CHK_ERR(!out, return NULL, "sb_vec_fread: sb_vec_malloc failed");

  n_read = fread(out->data, sizeof(double), n_elem, stream);
  SB_CHK_ERR(n_read != n_elem, sb_vec_free(out); return NULL, "sb_vec_fread: fread failed");

  return out;
}

/// Reads formatted data from `stream` into the vector returned by the function.
///
/// # Parameters
/// - `stream`: an open I/O stream
///
/// # Returns
/// A `sb_vec` pointer to the vector read from `stream`, or `NULL` if the scan or 
/// memory allocation fails
/// 
/// # Examples
/// ```
/// #include <stdio.h>
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
///   
///   // Write the vector to file
///   FILE * f = fopen("vector.txt", "w");
///   sb_vec_fprintf(f, v, "%lg");
///   fclose(f);
///   
///   // Read the vector from file
///   FILE * g = fopen("vector.txt", "r");
///   sb_vec * w = sb_vec_fscanf(g);
///   fclose(g);
///   
///   // Vectors have the same contents
///   sb_vec_print(v, "v: ", "%g");
///   sb_vec_print(w, "w: ", "%g");
///   
///   SB_VEC_FREE_ALL(v, w);
/// }
/// ```
sb_vec * sb_vec_fscanf(FILE * stream) {
  int n_scan;

  size_t n_elem;
  char layout;
  n_scan = fscanf(stream, "%zu %c", &n_elem, &layout);
  SB_CHK_ERR(n_scan != 2, return NULL, "sb_vec_fscanf: fscanf failed");

  sb_vec * out = sb_vec_malloc(n_elem, layout);
  SB_CHK_ERR(!out, return NULL, "sb_vec_fscanf: sb_vec_malloc failed");

  double * data = out->data;
  for (size_t a = 0; a < n_elem; ++a) {
    n_scan = fscanf(stream, "%lg", data + a);
    SB_CHK_ERR(n_scan != 1, sb_vec_free(out); return NULL, "sb_vec_fscanf: fscanf failed");
  }

  return out;
}

/// Takes the absolute value of every element of the vector.
///
/// # Parameters
/// - `v`: pointer to the vector
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
/// - `SAFE_FINITE`: elements of `v` are finite
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {-1., 4., -2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // Take absolute value
///   sb_vec_print(v, "v before: ", "%g");
///   sb_vec_abs(v);
///   sb_vec_print(v, "v after: ", "%g");
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_vec * sb_vec_abs(sb_vec * v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_abs: v cannot be NULL");
#endif
  double * data = v->data;
  for (size_t a = 0; a < v->n_elem; ++a) {
    data[a] = fabs(data[a]);
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_vec_is_finite(v), abort(), "sb_vec_abs: element not finite");
#endif
  return v;
}

/// Takes the exponent base `e` of every element of the vector.
///
/// # Parameters
/// - `v`: pointer to the vector
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
/// - `SAFE_FINITE`: elements of `v` are finite
///
/// # Examples
/// ```
/// #include <math.h>
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {log(1.), log(4.), log(2.)};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // Take exponent base e
///   sb_vec_print(v, "v before: ", "%g");
///   sb_vec_exp(v);
///   sb_vec_print(v, "v after: ", "%g");
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_vec * sb_vec_exp(sb_vec * v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_exp: v cannot be NULL");
#endif
  double * data = v->data;
  for (size_t a = 0; a < v->n_elem; ++a) {
    data[a] = exp(data[a]);
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_vec_is_finite(v), abort(), "sb_vec_exp: element not finite");
#endif
  return v;
}

/// Takes the logarithm base `e` of every element of the vector.
///
/// # Parameters
/// - `v`: pointer to the vector
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
/// - `SAFE_FINITE`: elements of `v` are finite
///
/// # Examples
/// ```
/// #include <math.h>
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {exp(1.), exp(4.), exp(2.)};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // Take logarithm base e
///   sb_vec_print(v, "v before: ", "%g");
///   sb_vec_log(v);
///   sb_vec_print(v, "v after: ", "%g");
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_vec * sb_vec_log(sb_vec * v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_log: v cannot be NULL");
#endif
  double * data = v->data;
  for (size_t a = 0; a < v->n_elem; ++a) {
    data[a] = log(data[a]);
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_vec_is_finite(v), abort(), "sb_vec_log: element not finite");
#endif
  return v;
}

/// Exponentiates every element of the vector `v` by `x`.
///
/// # Parameters
/// - `v`: pointer to the vector
/// - `x`: scalar exponent of the elements
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
/// - `SAFE_FINITE`: elements of `v` are finite
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // Exponentiate every element by -1.
///   sb_vec_print(v, "v before: ", "%g");
///   sb_vec_smul(v, -1.);
///   sb_vec_print(v, "v after: ", "%g");
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_vec * sb_vec_pow(sb_vec * v, double x) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_pow: v cannot be NULL");
#endif
  double * data = v->data;
  for (size_t a = 0; a < v->n_elem; ++a) {
    data[a] = pow(data[a], x);
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_vec_is_finite(v), abort(), "sb_vec_pow: element not finite");
#endif
  return v;
}

/// Takes the square root of every element of the vector `v`.
///
/// # Parameters
/// - `v`: pointer to the vector
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
/// - `SAFE_FINITE`: elements of `v` are finite
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 16., 4.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // Take the square root of every element
///   sb_vec_print(v, "v before: ", "%g");
///   sb_vec_sqrt(v);
///   sb_vec_print(v, "v after: ", "%g");
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_vec * sb_vec_sqrt(sb_vec * v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_sqrt: v cannot be NULL");
#endif
  double * data = v->data;
  for (size_t a = 0; a < v->n_elem; ++a) {
    data[a] = sqrt(data[a]);
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_vec_is_finite(v), abort(), "sb_vec_sqrt: element not finite");
#endif
  return v;
}

/// Scalar addition of `x` to every element of the vector `v`.
///
/// # Parameters
/// - `v`: pointer to the vector
/// - `x`: scalar added to the elements
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
/// - `SAFE_FINITE`: elements of `v` are finite
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // Add 2. to every element
///   sb_vec_print(v, "v before: ", "%g");
///   sb_vec_sadd(v, 2.);
///   sb_vec_print(v, "v after: ", "%g");
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_vec * sb_vec_sadd(sb_vec * v, double x) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_sadd: v cannot be NULL");
#endif
  double * data = v->data;
  for (size_t a = 0; a < v->n_elem; ++a) {
    data[a] += x;
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_vec_is_finite(v), abort(), "sb_vec_sadd: element not finite");
#endif
  return v;
}

/// Scalar multiplication of `x` with every element of the vector `v`.
///
/// # Parameters
/// - `v`: pointer to the vector
/// - `x`: scalar mutiplier for the elements
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
/// - `SAFE_FINITE`: elements of `v` are finite
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // Multiply every element by -1.
///   sb_vec_print(v, "v before: ", "%g");
///   sb_vec_smul(v, -1.);
///   sb_vec_print(v, "v after: ", "%g");
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_vec * sb_vec_smul(sb_vec * v, double x) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_smul: v cannot be NULL");
#endif
  cblas_dscal(v->n_elem, x, v->data, 1);
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_vec_is_finite(v), abort(), "sb_vec_smul: elements not finite");
#endif
  return v;
}

/// Pointwise addition of elements of the vector `w` to elements of the vector
/// `v`. `v` and `w` must not overlap in memory.
///
/// # Parameters
/// - `v`: pointer to the first vector
/// - `w`: pointer to the second vector
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` and `w` are not `NULL`
/// - `SAFE_LAYOUT`: `v` and `w` have the same layout
/// - `SAFE_LENGTH`: `v` and `w` have the same length
/// - `SAFE_FINITE`: elements of `v` are finite
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7.};
///   sb_vec * v = sb_vec_of_arr(a,     3, 'r');
///   sb_vec * w = sb_vec_of_arr(a + 3, 3, 'r');
/// 
///   // Add w to v
///   sb_vec_print(v, "v before: ", "%g");
///   sb_vec_padd(v, w);
///   sb_vec_print(v, "v after: ", "%g");
///
///   SB_VEC_FREE_ALL(v, w);
/// }
/// ```
sb_vec * sb_vec_padd(sb_vec * restrict v, const sb_vec * restrict w) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_padd: v cannot be NULL");
  SB_CHK_ERR(!w, abort(), "sb_vec_padd: w cannot be NULL");
#endif
#ifdef SAFE_LAYOUT
  SB_CHK_ERR(v->layout != w->layout, abort(),
      "sb_vec_padd: v and w must have same layout");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(v->n_elem != w->n_elem, abort(),
      "sb_vec_padd: v and w must have same length");
#endif
  cblas_daxpy(v->n_elem, 1., w->data, 1, v->data, 1);
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_vec_is_finite(v), abort(), "sb_vec_padd: elements not finite");
#endif
  return v;
}

/// Pointwise subtraction of elements of the vector `w` from elements of the
/// vector `v`. `v` and `w` must not overlap in memory.
///
/// # Parameters
/// - `v`: pointer to the first vector
/// - `w`: pointer to the second vector
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` and `w` are not `NULL`
/// - `SAFE_LAYOUT`: `v` and `w` have the same layout
/// - `SAFE_LENGTH`: `v` and `w` have the same length
/// - `SAFE_FINITE`: elements of `v` are finite
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7.};
///   sb_vec * v = sb_vec_of_arr(a,     3, 'r');
///   sb_vec * w = sb_vec_of_arr(a + 3, 3, 'r');
/// 
///   // Subtract w from v
///   sb_vec_print(v, "v before: ", "%g");
///   sb_vec_psub(v, w);
///   sb_vec_print(v, "v after: ", "%g");
///
///   SB_VEC_FREE_ALL(v, w);
/// }
/// ```
sb_vec * sb_vec_psub(sb_vec * restrict v, const sb_vec * restrict w) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_psub: v cannot be NULL");
  SB_CHK_ERR(!w, abort(), "sb_vec_psub: w cannot be NULL");
#endif
#ifdef SAFE_LAYOUT
  SB_CHK_ERR(v->layout != w->layout, abort(),
      "sb_vec_psub: v and w must have same layout");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(v->n_elem != w->n_elem, abort(),
      "sb_vec_psub: v and w must have same length");
#endif
  cblas_daxpy(v->n_elem, -1., w->data, 1, v->data, 1);
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_vec_is_finite(v), abort(), "sb_vec_psub: elements not finite");
#endif
  return v;
}

/// Pointwise multiplication of elements of the vector `v` with elements of the
/// vector `w`. `v` and `w` must not overlap in memory.
///
/// # Parameters
/// - `v`: pointer to the first vector
/// - `w`: pointer to the second vector
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` and `w` are not `NULL`
/// - `SAFE_LAYOUT`: `v` and `w` have the same layout
/// - `SAFE_LENGTH`: `v` and `w` have the same length
/// - `SAFE_FINITE`: elements of `v` are finite
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7.};
///   sb_vec * v = sb_vec_of_arr(a,     3, 'r');
///   sb_vec * w = sb_vec_of_arr(a + 3, 3, 'r');
/// 
///   // Multiply elements of v by elements of w
///   sb_vec_print(v, "v before: ", "%g");
///   sb_vec_pmul(v, w);
///   sb_vec_print(v, "v after: ", "%g");
///
///   SB_VEC_FREE_ALL(v, w);
/// }
/// ```
sb_vec * sb_vec_pmul(sb_vec * restrict v, const sb_vec * restrict w) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_pmul: v cannot be NULL");
  SB_CHK_ERR(!w, abort(), "sb_vec_pmul: w cannot be NULL");
#endif
#ifdef SAFE_LAYOUT
  SB_CHK_ERR(v->layout != w->layout, abort(),
      "sb_vec_pmul: v and w must have same layout");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(v->n_elem != w->n_elem, abort(),
      "sb_vec_pmul: v and w must have same length");
#endif
  double * v_data = v->data;
  double * w_data = w->data;
  for (size_t a = 0; a < v->n_elem; ++a) {
    v_data[a] *= w_data[a];
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_vec_is_finite(v), abort(), "sb_vec_pmul: element not finite");
#endif
  return v;
}

/// Pointwise division of elements of the vector `v` by elements of the vector
/// `w`. `v` and `w` must not overlap in memory.
///
/// # Parameters
/// - `v`: pointer to the first vector
/// - `w`: pointer to the second vector
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` and `w` are not `NULL`
/// - `SAFE_LAYOUT`: `v` and `w` have the same layout
/// - `SAFE_LENGTH`: `v` and `w` have the same length
/// - `SAFE_FINITE`: elements of `v` are finite
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7.};
///   sb_vec * v = sb_vec_of_arr(a,     3, 'r');
///   sb_vec * w = sb_vec_of_arr(a + 3, 3, 'r');
/// 
///   // Divide elements of v by elements of w
///   sb_vec_print(v, "v before: ", "%g");
///   sb_vec_pdiv(v, w);
///   sb_vec_print(v, "v after: ", "%g");
///
///   SB_VEC_FREE_ALL(v, w);
/// }
/// ```
sb_vec * sb_vec_pdiv(sb_vec * restrict v, const sb_vec * restrict w) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_pdiv: v cannot be NULL");
  SB_CHK_ERR(!w, abort(), "sb_vec_pdiv: w cannot be NULL");
#endif
#ifdef SAFE_LAYOUT
  SB_CHK_ERR(v->layout != w->layout, abort(),
      "sb_vec_pdiv: v and w must have same layout");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(v->n_elem != w->n_elem, abort(),
      "sb_vec_pdiv: v and w must have same length");
#endif
  double * v_data = v->data;
  double * w_data = w->data;
  for (size_t a = 0; a < v->n_elem; ++a) {
    v_data[a] /= w_data[a];
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_vec_is_finite(v), abort(), "sb_vec_pdiv: element not finite");
#endif
  return v;
}

/// Performs the operation \f$\mathbf{r} x + y\f$ where `x` and `y` are scalars
/// and `r` is modified (x result add y). 
///
/// # Parameters
/// - `r`: pointer to the vector
/// - `x`: scalar mutiplier for the elements
/// - `y`: scalar summand for the elements
///
/// # Returns
/// A copy of `r`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `r` is not `NULL`
/// - `SAFE_FINITE`: elements of `r` are finite
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   sb_vec * r = sb_vec_of_arr(a, 3, 'r');
/// 
///   // Multiply by -1. and add 2.
///   sb_vec_print(r, "r before: ", "%g");
///   sb_vec_rxay(r, -1., 2.);
///   sb_vec_print(r, "r after: ", "%g");
///
///   SB_VEC_FREE_ALL(r);
/// }
/// ```
sb_vec * sb_vec_rxay(sb_vec * r, double x, double y) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!r, abort(), "sb_vec_rxay: r cannot be NULL");
#endif
  double * data = r->data;
  for (size_t a = 0; a < r->n_elem; ++a) {
    data[a] = fma(data[a], x, y);
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_vec_is_finite(r), abort(), "sb_vec_rxay: elements not finite");
#endif
  return r;
}

/// Performs the operation \f$\mathbf{r} + x \mathbf{v}\f$ where `x` is a
/// scalar, `v` is a vector and `r` is modified (result add x vector). `r` and
/// `v` must not overlap in memory.
///
/// # Parameters
/// - `r`: pointer to the first vector
/// - `x`: scalar multiplier for `v`
/// - `v`: pointer to the second vector
///
/// # Returns
/// A copy of `r`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `r` and `v` are not `NULL`
/// - `SAFE_LAYOUT`: `r` and `v` have the same layout
/// - `SAFE_LENGTH`: `r` and `v` have the same length
/// - `SAFE_FINITE`: elements of `r` are finite
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7.};
///   sb_vec * r = sb_vec_of_arr(a,     3, 'r');
///   sb_vec * v = sb_vec_of_arr(a + 3, 3, 'r');
/// 
///   // Multiply v by 2. and add to r
///   sb_vec_print(r, "r before: ", "%g");
///   sb_vec_raxv(r, 2., v);
///   sb_vec_print(r, "r after: ", "%g");
///
///   SB_VEC_FREE_ALL(r, v);
/// }
/// ```
sb_vec * sb_vec_raxv(sb_vec * restrict r, double x, const sb_vec * restrict v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!r, abort(), "sb_vec_raxv: r cannot be NULL");
  SB_CHK_ERR(!v, abort(), "sb_vec_raxv: v cannot be NULL");
#endif
#ifdef SAFE_LAYOUT
  SB_CHK_ERR(r->layout != v->layout, abort(),
      "sb_vec_raxv: r and v must have same layout");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(r->n_elem != v->n_elem, abort(),
      "sb_vec_raxv: r and v must have same length");
#endif
  cblas_daxpy(r->n_elem, x, v->data, 1, r->data, 1);
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_vec_is_finite(r), abort(), "sb_vec_raxv: elements not finite");
#endif
  return r;
}

/// Performs the operation \f$\mathbf{r} \otimes \mathbf{v} - \mathbf{w}\f$
/// where `v` and `w` are vectors, \f$\otimes\f$ is pointwise multiplication,
/// and `r` is modified (result v subtract w). `r`, `v` and `w` must not
/// overlap in memory.
///
/// # Parameters
/// - `r`: pointer to the first vector
/// - `v`: pointer to the second vector
/// - `w`: pointer to the third vector
///
/// # Returns
/// A copy of `r`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `r`, `v` and `w` are not `NULL`
/// - `SAFE_LAYOUT`: `r`, `v` and `w` have the same layout
/// - `SAFE_LENGTH`: `r`, `v` and `w` have the same length
/// - `SAFE_FINITE`: elements of `r` are finite
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5.};
///   sb_vec * r = sb_vec_of_arr(a,     3, 'r');
///   sb_vec * v = sb_vec_of_arr(a + 1, 3, 'r');
///   sb_vec * w = sb_vec_of_arr(a + 2, 3, 'r');
/// 
///   // Pointwise multiply r by v and pointwise subtract w
///   sb_vec_print(r, "r before: ", "%g");
///   sb_vec_rvsw(r, v, w);
///   sb_vec_print(r, "r after: ", "%g");
///
///   SB_VEC_FREE_ALL(r, v, w);
/// }
/// ```
sb_vec * sb_vec_rvsw(
    sb_vec * restrict r,
    const sb_vec * restrict v,
    const sb_vec * restrict w) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!r, abort(), "sb_vec_rvsw: r cannot be NULL");
  SB_CHK_ERR(!v, abort(), "sb_vec_rvsw: v cannot be NULL");
  SB_CHK_ERR(!w, abort(), "sb_vec_rvsw: w cannot be NULL");
#endif
#ifdef SAFE_LAYOUT
  SB_CHK_ERR(r->layout != v->layout, abort(),
      "sb_vec_rvsw: r and v must have same layout");
  SB_CHK_ERR(r->layout != w->layout, abort(),
      "sb_vec_rvsw: r and w must have same layout");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(r->n_elem != v->n_elem, abort(),
      "sb_vec_rvsw: r and v must have same length");
  SB_CHK_ERR(r->n_elem != w->n_elem, abort(),
      "sb_vec_rvsw: r and w must have same length");
#endif
  double * r_data = r->data;
  double * v_data = v->data;
  double * w_data = w->data;
  for (size_t a = 0; a < r->n_elem; ++a) {
    r_data[a] = fma(r_data[a], v_data[a], -w_data[a]);
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_vec_is_finite(r), abort(), "sb_vec_rvsw: elements not finite");
#endif
  return r;
}

/// Performs \f$\mathbf{r} \oslash (\mathbf{v} \oslash \mathbf{w})\f$ where `v`
/// and `w` are vectors, \f$\oslash\f$ is pointwise division, and `r` is
/// modified (result divide paren v divide w paren). `r`, `v` and `w` must not
/// overlap in memory.
///
/// # Parameters
/// - `r`: pointer to the first vector
/// - `v`: pointer to the second vector
/// - `w`: pointer to the third vector
///
/// # Returns
/// A copy of `r`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `r`, `v` and `w` are not `NULL`
/// - `SAFE_LAYOUT`: `r`, `v` and `w` have the same layout
/// - `SAFE_LENGTH`: `r`, `v` and `w` have the same length
/// - `SAFE_FINITE`: elements of `r` are finite
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5.};
///   sb_vec * r = sb_vec_of_arr(a,     3, 'r');
///   sb_vec * v = sb_vec_of_arr(a + 1, 3, 'r');
///   sb_vec * w = sb_vec_of_arr(a + 2, 3, 'r');
/// 
///   // Pointwise division of r by ratio of v and w
///   sb_vec_print(r, "r before: ", "%g");
///   sb_vec_rdpvdwp(r, v, w);
///   sb_vec_print(r, "r after: ", "%g");
///
///   SB_VEC_FREE_ALL(r, v, w);
/// }
/// ```
sb_vec * sb_vec_rdpvdwp(
    sb_vec * restrict r,
    const sb_vec * restrict v,
    const sb_vec * restrict w) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!r, abort(), "sb_vec_rdpvdwp: r cannot be NULL");
  SB_CHK_ERR(!v, abort(), "sb_vec_rdpvdwp: v cannot be NULL");
  SB_CHK_ERR(!w, abort(), "sb_vec_rdpvdwp: w cannot be NULL");
#endif
#ifdef SAFE_LAYOUT
  SB_CHK_ERR(r->layout != v->layout, abort(),
      "sb_vec_rdpvdwp: r and v must have same layout");
  SB_CHK_ERR(r->layout != w->layout, abort(),
      "sb_vec_rdpvdwp: r and w must have same layout");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(r->n_elem != v->n_elem, abort(),
      "sb_vec_rdpvdwp: r and v must have same length");
  SB_CHK_ERR(r->n_elem != w->n_elem, abort(),
      "sb_vec_rdpvdwp: r and w must have same length");
#endif
  double * r_data = r->data;
  double * v_data = v->data;
  double * w_data = w->data;
  for (size_t a = 0; a < r->n_elem; ++a) {
    r_data[a] /= v_data[a] / w_data[a];
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_vec_is_finite(r), abort(), "sb_vec_rdpvdwp: elements not finite");
#endif
  return r;
}

/// Finds the sum of the elements of the vector `v`.
///
/// # Parameters
/// - `v`: pointer to the vector
///
/// # Returns
/// The sum of the elements of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
/// - `SAFE_FINITE`: sum is finite
///
/// # Examples
/// ```
/// #include <stdio.h>
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // Sum the elements of v
///   sb_vec_print(v, "v: ", "%g");
///   printf("sum of v: %g\n", sb_vec_sum(v));
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
double sb_vec_sum(const sb_vec * v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_sum: v cannot be NULL");
#endif
  double * data = v->data;
  double out = 0.;
  for (size_t a = 0; a < v->n_elem; ++a) {
    out += data[a];
  }
#ifdef SAFE_FINITE
  SB_CHK_ERR(!isfinite(out), abort(), "sb_vec_sum: sum not finite");
#endif
  return out;
}

/// Finds the Euclidean norm of the vector `v`.
///
/// # Parameters
/// - `v`: pointer to the vector
///
/// # Returns
/// The Euclidean norm of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
/// - `SAFE_FINITE`: norm is finite
///
/// # Examples
/// ```
/// #include <stdio.h>
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // Norm of v
///   sb_vec_print(v, "v: ", "%g");
///   printf("norm of v: %g\n", sb_vec_norm(v));
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
double sb_vec_norm(const sb_vec * v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_norm: v cannot be NULL");
#endif
  double out = cblas_dnrm2(v->n_elem, v->data, 1);
#ifdef SAFE_FINITE
  SB_CHK_ERR(!isfinite(out), abort(), "sb_vec_norm: norm not finite");
#endif
  return out;
}

/// Finds the dot product of the vectors `v` and `w`. Layout of `v` and `w` is
/// ignored.
///
/// # Parameters
/// - `v`: pointer to the first vector
/// - `w`: pointer to the second vector
///
/// # Returns
/// The dot product of `v` and `w`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` and `w` are not `NULL`
/// - `SAFE_LENGTH`: `v` and `w` have the same length
/// - `SAFE_FINITE`: dot product is finite
///
/// # Examples
/// ```
/// #include <stdio.h>
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7.};
///   sb_vec * v = sb_vec_of_arr(a,     3, 'r');
///   sb_vec * w = sb_vec_of_arr(a + 3, 3, 'r');
/// 
///   // Find the dot product of v and w
///   sb_vec_print(v, "v: ", "%g");
///   sb_vec_print(w, "w: ", "%g");
///   printf("dot product: %g\n", sb_vec_dot(v, w));
///
///   SB_VEC_FREE_ALL(v, w);
/// }
/// ```
double sb_vec_dot(const sb_vec * restrict v, const sb_vec * restrict w) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_dot: v cannot be NULL");
  SB_CHK_ERR(!w, abort(), "sb_vec_dot: w cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(v->n_elem != w->n_elem, abort(),
      "sb_vec_dot: v and w must have same length");
#endif
  double out = cblas_ddot(v->n_elem, v->data, 1, w->data, 1);
#ifdef SAFE_FINITE
  SB_CHK_ERR(!isfinite(out), abort(), "sb_vec_dot: dot product not finite");
#endif
  return out;
}

/// Finds the outer product of the vectors `v` and `w` and adds the result to
/// the matrix `A`. `v` and `w` must be column and row vectors, respectively,
/// and the matrix `A` must have the same number of rows as `v` and the same
/// number of columns as `w`.
///
/// # Parameters
/// - `v`: pointer to the first vector
/// - `w`: pointer to the second vector
/// - `A`: pointer to the matrix
///
/// # Returns
/// A copy of `A`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v`, `w` and `A` are not `NULL`
/// - `SAFE_LAYOUT`: `v` is a column vector and `w` is a row vector
/// - `SAFE_LENGTH`: `v` and `A` have the same number of rows, and `w` and
///                  `A` have the same number of columns
/// - `SAFE_FINITE`: elements of `A` are finite
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include "sb_matrix.h"
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7.};
///   sb_vec * v = sb_vec_of_arr(a,     3, 'c');
///   sb_vec * w = sb_vec_of_arr(a + 3, 3, 'r');
/// 
///   // Store the outer product in A
///   sb_mat * A = sb_mat_calloc(3, 3);
///   sb_mat_print(sb_vec_outer(A, v, w), "A: ", "%g");
///
///   SB_VEC_FREE_ALL(v, w);
/// }
/// ```
sb_mat * sb_vec_outer(
    sb_mat * restrict A,
    const sb_vec * restrict v,
    const sb_vec * restrict w) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_outer: v cannot be NULL");
  SB_CHK_ERR(!w, abort(), "sb_vec_outer: w cannot be NULL");
  SB_CHK_ERR(!A, abort(), "sb_vec_outer: A cannot be NULL");
#endif
#ifdef SAFE_LAYOUT
  SB_CHK_ERR(v->layout != 'c', abort(), "sb_vec_outer: v must be a column vector");
  SB_CHK_ERR(w->layout != 'r', abort(), "sb_vec_outer: w must be a row vector");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(v->n_elem != A->n_rows, abort(),
      "sb_vec_outer: v and A must have same number of rows");
  SB_CHK_ERR(w->n_elem != A->n_cols, abort(),
      "sb_vec_outer: w and A must have same number of cols");
#endif
  cblas_dger(CblasColMajor, v->n_elem, w->n_elem, 1., v->data, 1, w->data, 1,
      A->data, A->n_rows);
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_mat_is_finite(A), abort(), "sb_vec_outer: A is not finite");
#endif
  return A;
}

/// Reverses the order of elements in the vector `v`.
///
/// # Parameters
/// - `v`: pointer to the vector
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // Print vector and reverse
///   sb_vec_print(v, "v before: ", "%g");
///   sb_vec_reverse(v);
///   sb_vec_print(v, "v after: ", "%g");
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_vec * sb_vec_reverse(sb_vec * v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_reverse: v cannot be NULL");
#endif
  size_t n_elem = v->n_elem;
  double * data = v->data;

  size_t b;
  double scratch;
  for (size_t a = 0; a < n_elem / 2; ++a) {
    b = n_elem - a - 1;
    SB_SWAP(data[a], data[b], scratch);
  }
  return v;
}

// FOR USE ONLY WITH SB_VEC_SORT_INC
static int dcmp_inc(const void * pa, const void * pb) {
  double a = *(const double *)pa;
  double b = *(const double *)pb;

  if (a < b) { return -1; }
  if (b < a) { return  1; }
  return 0;
}

/// Sorts the elements of the vector `v` in increasing order.
///
/// # Parameters
/// - `v`: pointer to the vector
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
/// - `SAFE_FINITE`: elements of `v` are finite
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // Sort vector and print
///   sb_vec_print(v, "v before: ", "%g");
///   sb_vec_sort_inc(v);
///   sb_vec_print(v, "v after: ", "%g");
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_vec * sb_vec_sort_inc(sb_vec * v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_sort_inc: v cannot be NULL");
#endif
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_vec_is_finite(v), abort(), "sb_vec_sort_inc: v is not finite");
#endif
  qsort(v->data, v->n_elem, sizeof(double), dcmp_inc);
  return v;
}

// FOR USE ONLY WITH SB_VEC_SORT_DEC
static int dcmp_dec(const void * pa, const void * pb) {
  double a = *(const double *)pa;
  double b = *(const double *)pb;

  if (b < a) { return -1; }
  if (a < b) { return  1; }
  return 0;
}

/// Sorts the elements of the vector `v` in decreasing order.
///
/// # Parameters
/// - `v`: pointer to the vector
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
/// - `SAFE_FINITE`: elements of `v` are finite
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // Sort vector and print
///   sb_vec_print(v, "v before: ", "%g");
///   sb_vec_sort_dec(v);
///   sb_vec_print(v, "v after: ", "%g");
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_vec * sb_vec_sort_dec(sb_vec * v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_sort_dec: v cannot be NULL");
#endif
#ifdef SAFE_FINITE
  SB_CHK_ERR(!sb_vec_is_finite(v), abort(), "sb_vec_sort_dec: v is not finite");
#endif
  qsort(v->data, v->n_elem, sizeof(double), dcmp_dec);
  return v;
}

/// Transposes the vector `v`. NOTE: This is a non-op when `SAFE_LAYOUT` is not
/// defined, possibly leading to unexpected behavior with e.g. sb_vec_print().
///
/// # Parameters
/// - `v`: pointer to the vector
///
/// # Returns
/// A copy of `v`
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
/// - `SAFE_LAYOUT`: `layout` is `c` or `r`
///
/// # Examples
/// ```
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // Print vector and transpose
///   sb_vec_print(v, "v before: ", "%g");
///   sb_vec_trans(v);
///   sb_vec_print(v, "v after: ", "%g");
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
sb_vec * sb_vec_trans(sb_vec * v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_trans: v cannot be NULL");
#endif
#ifdef SAFE_LAYOUT
  SB_CHK_ERR(v->layout != 'c' && v->layout != 'r', abort(),
      "sb_vec_trans: v has invalid layout");
  v->layout = (v->layout == 'c' ? 'r' : 'c');
#endif
  return v;
}

/// Checks the vectors `v` and `w` for equality of elements. Vectors must have
/// the same length and layout.
///
/// # Parameters
/// - `v`: pointer to the first vector
/// - `w`: pointer to the second vector
///
/// # Returns
/// `1` if `v` and `w` are equal, and `0` otherwise
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` and `w` are not `NULL`
/// - `SAFE_LAYOUT`: `v` and `w` have the same layout
/// - `SAFE_LENGTH`: `v` and `w` have the same length
/// - `SAFE_FINITE`: elements of `v` and `w` are finite
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2., 8., 5., 7.};
///   sb_vec * v = sb_vec_of_arr(a,     3, 'r');
///   sb_vec * w = sb_vec_of_arr(a + 3, 3, 'r');
/// 
///   // v and w are equal after sb_vec_memcpy
///   assert(!sb_vec_is_equal(v, w));
///   sb_vec_memcpy(w, v);
///   assert( sb_vec_is_equal(v, w));
///
///   SB_VEC_FREE_ALL(v, w);
/// }
/// ```
int sb_vec_is_equal(const sb_vec * restrict v, const sb_vec * restrict w) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_is_equal: v cannot be NULL");
  SB_CHK_ERR(!w, abort(), "sb_vec_is_equal: w cannot be NULL");
#endif
#ifdef SAFE_LAYOUT
  SB_CHK_ERR(v->layout != w->layout, abort(),
      "sb_vec_is_equal: v and w must have same layout");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(v->n_elem != w->n_elem, abort(),
      "sb_vec_is_equal: v and w must have same length");
#endif
  double * v_data = v->data;
  double * w_data = w->data;
  for (size_t a = 0; a < v->n_elem; ++a) {
#ifdef SAFE_FINITE
    SB_CHK_ERR(!isfinite(v_data[a]), abort(), "sb_vec_is_equal: v is not finite");
    SB_CHK_ERR(!isfinite(w_data[a]), abort(), "sb_vec_is_equal: w is not finite");
#endif
    if (v_data[a] != w_data[a]) {
      return 0;
    }
  }
  return 1;
}

/// Checks if all elements of the vector `v` are zero.
///
/// # Parameters
/// - `v`: pointer to the vector
///
/// # Returns
/// `1` if all elements of `v` are zero, and `0` otherwise
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // v is zero after sb_vec_set_zero
///   assert(!sb_vec_is_zero(v));
///   sb_vec_set_zero(v);
///   assert( sb_vec_is_zero(v));
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
int sb_vec_is_zero(const sb_vec * v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_is_zero: v cannot be NULL");
#endif
  double * data = v->data;
  for (size_t a = 0; a < v->n_elem; ++a) {
    if (data[a] != 0.) {
      return 0;
    }
  }
  return 1;
}

/// Checks if all elements of the vector `v` are positive.
///
/// # Parameters
/// - `v`: pointer to the vector
///
/// # Returns
/// `1` if all elements of `v` are positive, and `0` otherwise
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
/// - `SAFE_FINITE`: elements of `v` are finite
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {-1., -4., -2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // v is positive after sb_vec_abs
///   assert(!sb_vec_is_pos(v));
///   sb_vec_abs(v);
///   assert( sb_vec_is_pos(v));
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
int sb_vec_is_pos(const sb_vec * v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_is_pos: v cannot be NULL");
#endif
  double * data = v->data;
  for (size_t a = 0; a < v->n_elem; ++a) {
#ifdef SAFE_FINITE
    SB_CHK_ERR(!isfinite(data[a]), abort(), "sb_vec_is_pos: v is not finite");
#endif
    if (data[a] <= 0.) {
      return 0;
    }
  }
  return 1;
}

/// Checks if all elements of the vector `v` are negative.
///
/// # Parameters
/// - `v`: pointer to the vector
///
/// # Returns
/// `1` if all elements of `v` are negative, and `0` otherwise
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
/// - `SAFE_FINITE`: elements of `v` are finite
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., -4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // v is negative after
///   assert(!sb_vec_is_neg(v));
///   sb_vec_smul(sb_vec_abs(v), -1.);
///   assert( sb_vec_is_neg(v));
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
int sb_vec_is_neg(const sb_vec * v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_is_neg: v cannot be NULL");
#endif
  double * data = v->data;
  for (size_t a = 0; a < v->n_elem; ++a) {
#ifdef SAFE_FINITE
    SB_CHK_ERR(!isfinite(data[a]), abort(), "sb_vec_is_neg: v is not finite");
#endif
    if (data[a] >= 0.) {
      return 0;
    }
  }
  return 1;
}

/// Checks if all elements of the vector `v` are nonnegative.
///
/// # Parameters
/// - `v`: pointer to the vector
///
/// # Returns
/// `1` if all elements of `v` are nonnegative, and `0` otherwise
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
/// - `SAFE_FINITE`: elements of `v` are finite
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {0., -1., 4.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // v is nonnegative after
///   assert(!sb_vec_is_nonneg(v));
///   sb_vec_abs(v);
///   assert( sb_vec_is_nonneg(v));
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
int sb_vec_is_nonneg(const sb_vec * v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_is_nonneg: v cannot be NULL");
#endif
  double * data = v->data;
  for (size_t a = 0; a < v->n_elem; ++a) {
#ifdef SAFE_FINITE
    SB_CHK_ERR(!isfinite(data[a]), abort(), "sb_vec_is_nonneg: v is not finite");
#endif
    if (data[a] < 0.) {
      return 0;
    }
  }
  return 1;
}

/// Checks if all elements of the vector `v` are not infinite or `NaN`.
///
/// # Parameters
/// - `v`: pointer to the sb_vector
///
/// # Returns
/// `1` if all elements of `v` are not infinite or `NaN`, and `0` otherwise
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include <math.h>
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., INFINITY, 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   assert(!sb_vec_is_finite(v));
///   sb_vec_set(v, 1, NAN);
///   assert(!sb_vec_is_finite(v));
///   sb_vec_set(v, 1, 4.);
///   assert( sb_vec_is_finite(v));
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
int sb_vec_is_finite(const sb_vec * v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_is_finite: v cannot be NULL");
#endif
  double * data = v->data;
  for (size_t a = 0; a < v->n_elem; ++a) {
    if (!isfinite(data[a])) {
      return 0;
    }
  }
  return 1;
}

/// Finds the value of the maximum element of `v`.
///
/// # Parameters
/// - `v`: pointer to the vector
///
/// # Returns
/// Value of the maximum element
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
/// - `SAFE_LENGTH`: number of elements is nonzero
/// - `SAFE_FINITE`: elements of `v` are finite
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // maximum value of v is 4.
///   assert(sb_vec_max(v) == 4.);
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
double sb_vec_max(const sb_vec * v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_max: v cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(v->n_elem == 0, abort(), "sb_vec_max: n_elem must be nonzero");
#endif
  double * data = v->data;
  double max_val = -INFINITY;
  for (size_t a = 0; a < v->n_elem; ++a) {
#ifdef SAFE_FINITE
    SB_CHK_ERR(!isfinite(data[a]), abort(), "sb_vec_max: v is not finite");
#endif
    if (data[a] > max_val) {
      max_val = data[a];
    }
  }
  return max_val;
}

/// Finds the value of the minimum element of `v`.
///
/// # Parameters
/// - `v`: pointer to the vector
///
/// # Returns
/// Value of the minimum element
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
/// - `SAFE_LENGTH`: number of elements is nonzero
/// - `SAFE_FINITE`: elements of `v` are finite
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // value of min is 1
///   assert(sb_vec_min(v) == 1.);
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
double sb_vec_min(const sb_vec * v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_min: v cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(v->n_elem == 0, abort(), "sb_vec_min: n_elem must be nonzero");
#endif
  double * data = v->data;
  double min_val = INFINITY;
  for (size_t a = 0; a < v->n_elem; ++a) {
#ifdef SAFE_FINITE
    SB_CHK_ERR(!isfinite(data[a]), abort(), "sb_vec_min: v is not finite");
#endif
    if (data[a] < min_val) {
      min_val = data[a];
    }
  }
  return min_val;
}

/// Finds the maximum absolute value of the elements of `v`.
///
/// # Parameters
/// - `v`: pointer to the vector
///
/// # Returns
/// Maximum absolute value of the elements
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
/// - `SAFE_LENGTH`: number of elements is nonzero
/// - `SAFE_FINITE`: elements of `v` are finite
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., -4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // maximum absolute value of v is 4.
///   assert(sb_vec_abs_max(v) == 4.);
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
double sb_vec_abs_max(const sb_vec * v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_abs_max: v cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(v->n_elem == 0, abort(), "sb_vec_abs_max: n_elem must be nonzero");
#endif
  double * data = v->data;
  size_t index = cblas_idamax(v->n_elem, data, 1);
#ifdef SAFE_FINITE
  SB_CHK_ERR(!isfinite(data[index]), abort(), "sb_vec_abs_max: v is not finite");
#endif
  return fabs(data[index]);
}

/// Finds the index of the maximum element of `v`.
///
/// # Parameters
/// - `v`: pointer to the sb_vector
///
/// # Returns
/// Index of the maximum element
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
/// - `SAFE_LENGTH`: number of elements is nonzero
/// - `SAFE_FINITE`: elements of `v` are finite
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // index of max is 1
///   assert(sb_vec_max_index(v) == 1);
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
size_t sb_vec_max_index(const sb_vec * v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_max_index: v cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(v->n_elem == 0, abort(), "sb_vec_max_index: n_elem must be nonzero");
#endif
  double * data = v->data;
  double max_val = -INFINITY;
  size_t max_ind = 0;
  for (size_t a = 0; a < v->n_elem; ++a) {
#ifdef SAFE_FINITE
    SB_CHK_ERR(!isfinite(data[a]), abort(), "sb_vec_max_index: v is not finite");
#endif
    if (data[a] > max_val) {
      max_val = data[a];
      max_ind = a;
    }
  }
  return max_ind;
}

/// Finds the index of the minimum element of `v`.
///
/// # Parameters
/// - `v`: pointer to the vector
///
/// # Returns
/// Index of the minimum element
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
/// - `SAFE_LENGTH`: number of elements is nonzero
/// - `SAFE_FINITE`: elements of `v` are finite
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., 4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // index of min is 0
///   assert(sb_vec_min_index(v) == 0);
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
size_t sb_vec_min_index(const sb_vec * v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_min_index: v cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(v->n_elem == 0, abort(), "sb_vec_min_index: n_elem must be nonzero");
#endif
  double * data = v->data;
  double min_val = INFINITY;
  size_t min_ind = 0;
  for (size_t a = 0; a < v->n_elem; ++a) {
#ifdef SAFE_FINITE
    SB_CHK_ERR(!isfinite(data[a]), abort(), "sb_vec_min_index: v is not finite");
#endif
    if (data[a] < min_val) {
      min_val = data[a];
      min_ind = a;
    }
  }
  return min_ind;
}

/// Finds the index of the maximum absolute value of the elements of `v`.
///
/// # Parameters
/// - `v`: pointer to the vector
///
/// # Returns
/// Index of the maximum absolute value of the elements
/// 
/// # Performance
/// The following preprocessor definitions (usually in `safety.h`) enable 
/// various safety checks:
/// - `SAFE_MEMORY`: `v` is not `NULL`
/// - `SAFE_LENGTH`: number of elements is nonzero
///
/// # Examples
/// ```
/// #include <assert.h>
/// #include "sb_structs.h"
/// #include "sb_vector.h"
///
/// int main(void) {
///   double a[] = {1., -4., 2.};
///   sb_vec * v = sb_vec_of_arr(a, 3, 'r');
/// 
///   // index of the maximum absolute value of v is 1
///   assert(sb_vec_abs_max_index(v) == 1);
///
///   SB_VEC_FREE_ALL(v);
/// }
/// ```
size_t sb_vec_abs_max_index(const sb_vec * v) {
#ifdef SAFE_MEMORY
  SB_CHK_ERR(!v, abort(), "sb_vec_abs_min_index: v cannot be NULL");
#endif
#ifdef SAFE_LENGTH
  SB_CHK_ERR(v->n_elem == 0, abort(), "sb_vec_abs_min_index: n_elem must be nonzero");
#endif
  return cblas_idamax(v->n_elem, v->data, 1);
}

extern inline double sb_vec_get(const sb_vec * v, size_t i);
extern inline void sb_vec_set(sb_vec * v, size_t i, double x);
extern inline double * sb_vec_ptr(sb_vec * v, size_t i);
