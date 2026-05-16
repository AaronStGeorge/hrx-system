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

// Lowers a vector.mma over dense full-logical matrix fragments. Returns
// out_handled=false when the op uses target-shaped physical fragments, encoded
// operands, or numeric forms that need a richer reference decomposition.
iree_status_t loom_vector_to_scalar_lower_mma(
    loom_vector_to_scalar_state_t* state, bool* out_handled,
    loom_value_id_t* out_replacement);

// Materializes one logical result lane of a dense full-logical vector.mma.
// Returns out_materialized=false when the op is not an MMA or uses a shape that
// the target-independent reference decomposition does not cover.
iree_status_t loom_vector_to_scalar_try_materialize_mma_lane(
    loom_vector_to_scalar_state_t* state, loom_op_t* op,
    loom_vector_to_scalar_index_list_t indices, bool* out_materialized,
    loom_value_id_t* out_lane);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_PASSES_VECTOR_TO_SCALAR_MMA_H_
