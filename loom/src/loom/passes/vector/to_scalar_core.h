// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core state shared by vector-to-scalar lowering clusters.

#ifndef LOOM_PASSES_VECTOR_TO_SCALAR_CORE_H_
#define LOOM_PASSES_VECTOR_TO_SCALAR_CORE_H_

#include "iree/base/api.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/pass/types.h"
#include "loom/rewrite/rewriter.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_vector_to_scalar_stat_e {
  // Number of vector ops lowered by this pass.
  LOOM_VECTOR_TO_SCALAR_STAT_OPS_LOWERED = 0,
  // Number of scf.for loops created by this pass.
  LOOM_VECTOR_TO_SCALAR_STAT_LOOPS_CREATED = 1,
  // Number of scalar lane programs materialized by this pass.
  LOOM_VECTOR_TO_SCALAR_STAT_LANES_MATERIALIZED = 2,
} loom_vector_to_scalar_stat_t;

typedef struct loom_vector_to_scalar_descriptor_t
    loom_vector_to_scalar_descriptor_t;

typedef struct loom_vector_to_scalar_state_t {
  // Current pass instance owning statistics and transient arena state.
  loom_pass_t* pass;
  // Rewriter used to insert replacement lane IR and maintain use-def state.
  loom_rewriter_t* rewriter;
  // Vector op currently being scalarized.
  loom_op_t* op;
  // Descriptor that selects the lane-program family for |op|.
  const loom_vector_to_scalar_descriptor_t* descriptor;
  // Rewriter value checkpoint used to preserve result names on new values.
  loom_value_id_t value_checkpoint;
  // Result ordinal being scalarized for multi-result vector ops.
  uint16_t result_ordinal;
  // Vector result type currently being expanded into scalar lanes.
  loom_type_t vector_type;
  // Scalar type produced by each lane program.
  loom_type_t result_scalar_type;
  // Source location assigned to replacement ops.
  loom_location_id_t location;
} loom_vector_to_scalar_state_t;

// Returns the op defining |value_id|, or NULL when the value is not an op
// result that can be rematerialized from its defining operation.
static inline loom_op_t* loom_vector_to_scalar_value_def_op(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return NULL;
  }
  loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return NULL;
  }
  return loom_value_def_op(value);
}

#ifdef __cplusplus
}
#endif

#endif  // LOOM_PASSES_VECTOR_TO_SCALAR_CORE_H_
