// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Dynamic strided-layout projection for generic boundary rewriting.

#ifndef LOOM_TRANSFORMS_ENCODING_LAYOUT_PROJECTION_H_
#define LOOM_TRANSFORMS_ENCODING_LAYOUT_PROJECTION_H_

#include "loom/transforms/boundary/projection_rule.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the compiler-owned CFG block argument layout transport rule.
const loom_boundary_projection_rule_t* loom_cfg_layout_boundary_projection_rule(
    void);

// Returns the compiler-owned LoopLike state layout transport rule.
const loom_boundary_projection_rule_t*
loom_loop_layout_boundary_projection_rule(void);

// Returns the compiler-owned callable physical-encoding transport rule.
const loom_boundary_projection_rule_t*
loom_callable_encoding_boundary_projection_rule(void);

// Maps one logical layout axis to its transported scalar component. Returns
// false for schemas owned by another rule or axes fixed in the schema.
bool loom_layout_boundary_projection_component_for_axis(
    const loom_boundary_projection_schema_t* schema, uint8_t axis,
    uint8_t* out_component);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_ENCODING_LAYOUT_PROJECTION_H_
