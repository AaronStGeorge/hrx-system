// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Matrix contraction reference lowering.

#ifndef LOOM_PASSES_VECTOR_TO_SCALAR_MMA_H_
#define LOOM_PASSES_VECTOR_TO_SCALAR_MMA_H_

#include "loom/passes/vector/to_scalar_lanes.h"

#ifdef __cplusplus
extern "C" {
#endif

// Lowers a vector.mma whose result can be represented as a dense full-logical
// matrix fragment. Returns out_handled=false when the result remains
// target-shaped or numeric/encoded forms need reference semantics that this
// generic decomposition does not cover.
iree_status_t loom_vector_to_scalar_lower_mma(
    loom_vector_to_scalar_state_t* state, bool* out_handled,
    loom_value_id_t* out_replacement);

// Materializes one result lane of a vector.mma. Dense full-logical result
// vectors accept their normal vector indices; target-shaped matrix-fragment
// consumers may pass logical rank-2 matrix indices.
// Returns out_materialized=false when the op is not an MMA or uses a shape that
// the target-independent reference decomposition does not cover.
iree_status_t loom_vector_to_scalar_try_materialize_mma_lane(
    loom_vector_to_scalar_state_t* state, loom_op_t* op,
    loom_vector_to_scalar_index_list_t indices, bool* out_materialized,
    loom_value_id_t* out_lane);

// Returns true when every operand needed to materialize logical result lanes is
// present in value facts and covered by the target-independent reference
// semantics. This is a no-IR-emission predicate used before building fragment
// store loops.
bool loom_vector_to_scalar_mma_supports_logical_result_lanes(
    loom_vector_to_scalar_state_t* state, loom_op_t* op);

// Returns contract rejection bits for the vector.mma forms this reference
// lowering cannot scalarize.
uint32_t loom_vector_to_scalar_mma_reference_rejection_bits(
    loom_vector_to_scalar_state_t* state);

// Returns the first role-local contract rejection detail for the vector.mma
// forms this reference lowering cannot scalarize.
uint32_t loom_vector_to_scalar_mma_reference_rejection_detail(
    loom_vector_to_scalar_state_t* state);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_PASSES_VECTOR_TO_SCALAR_MMA_H_
