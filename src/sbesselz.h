// Copyright 2018 Jeremy Mason
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! \file sbesselz.h
//! Contains function to calculate the zeros of the spherical Bessel functions
//! of the first kind in the restricted setting of this application.

#pragma once

#include <stdint.h>     // uint32_t
#include "sb_structs.h" // sb_mat

sb_mat * _sbesselz(sb_mat * u_nl, const uint32_t n_max);
