// Copyright 2018 Jeremy Mason
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! \file tables.h
//! Contains functions to generate the lookup tables used in `sb_descriptors()`.

#pragma once

#include <stdint.h>     // uint32_t
#include "sb_structs.h" // sb_mat

void _build_unl_tbl(sb_mat * unl, const uint32_t n_max);
void _build_fnl_tbl(sb_mat * fnl_c1, sb_mat * fnl_c2, const uint32_t n_max);
