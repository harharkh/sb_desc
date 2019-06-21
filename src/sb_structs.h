// Copyright 2018 Jeremy Mason
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#pragma once

#include <stddef.h> // size_t

/// Type of a vector of doubles, with a row or column vector indicated by the
/// value of `layout`. Designed to be used with linear algebra routines.
typedef struct {
  size_t   n_elem; ///< number of elements
  double * data;   ///< pointer to the memory backing the vector
  char     layout; ///< `c` for a column vector, `r` for a row vector
} gw_vec;

/// Type of a matrix of doubles with column-major order. Designed to be used
/// with linear algebra routines.
typedef struct {
  size_t   n_rows; ///< number of rows
  size_t   n_cols; ///< number of columns
  size_t   n_elem; ///< number of elements
  double * data;   ///< pointer to the memory backing the matrix
} gw_mat;
