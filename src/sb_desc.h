// Copyright 2018 Jeremy Mason
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! \file sb_desc.h
//! Contains functions that actually calculate the SB descriptors.

#pragma once

#include <stdint.h>     // uint32_t

double * sb_descriptors(
    double * desc,
    double * disp,
    const double * weights,
    const double rc,
    const uint32_t n_atom, 
    const uint32_t n_max);
