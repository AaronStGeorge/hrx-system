// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_VECTOR_PACKET_LOAD_H_
#define LOOM_TRANSFORMS_VECTOR_PACKET_LOAD_H_

#include "loom/transforms/vector/packet_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// A selected load expansion owned by the legalization arena. Selection borrows
// the source load's address operands and attributes. They remain unchanged
// until emission; rewrites of other vector results cannot change these
// scalar-index and view operands. Each plan is emitted once, at its source
// load's position.
typedef struct loom_vector_packet_load_plan_t {
  // Next load in the caller's emission order, initially NULL.
  struct loom_vector_packet_load_plan_t* next;
  // Source load retained until this plan replaces it.
  loom_op_t* source_op;
  // Optional authored contract result retained by a reporting caller.
  const loom_target_contract_query_result_t* authored_query_result;
  // View read by the source load.
  loom_value_id_t view;
  // Original vector type reconstructed from the selected packets.
  loom_type_t vector_type;
  // Borrowed scalar SSA address operands, unchanged through emission.
  loom_value_slice_t dynamic_indices;
  // Borrowed immutable logical origin indices.
  loom_attribute_t static_indices;
  // Authored cache policy forwarded to the generated reads.
  loom_vector_memory_cache_policy_t cache_policy;
  // Number of vector packets, including a possible scalar-built suffix.
  uint32_t component_count;
  // Number of scalar loads in the terminal suffix, or zero.
  uint32_t scalar_lane_count;
  // Selected widths; a zero last entry denotes the scalar-built suffix.
  const uint32_t* lane_counts;
  // Placed scratch for emitted packet values, filled only during emission.
  loom_value_id_t* components;
  // Placed scratch for scalar_lane_count values in the scalar-built suffix.
  loom_value_id_t* scalar_values;
} loom_vector_packet_load_plan_t;

// Selects a load expansion no wider than |maximum_lane_count|; zero retains the
// ordinary target-selected width. A NULL plan means the load cannot be split.
// All target and fact queries finish before this returns. No IR is changed.
iree_status_t loom_vector_packet_load_plan_select(
    loom_target_legalization_context_t* context, loom_op_t* source_op,
    const loom_target_vector_packet_policy_t* policy,
    uint32_t maximum_lane_count, loom_vector_packet_load_plan_t** out_plan);

// Retains the exact widths admitted within an already-selected shape. A NULL
// policy keeps its regular geometric widths. An exact expansion that exceeds
// the static packet bound returns a NULL plan without changing IR.
iree_status_t loom_vector_packet_load_plan_allocate(
    loom_target_legalization_context_t* context, loom_op_t* source_op,
    const loom_target_vector_packet_policy_t* policy,
    const loom_vector_memory_footprint_t* footprint,
    const loom_vector_packet_shape_t* shape,
    loom_vector_memory_cache_policy_t cache_policy,
    loom_vector_packet_load_plan_t** out_plan);

// Consumes the retained widths and address information without source-analysis
// queries. Allocation and rewrite bookkeeping remain fallible. The caller
// finishes planning every load in a batch before emitting its first load.
iree_status_t loom_vector_packet_load_plan_emit(
    loom_target_legalization_context_t* context,
    loom_vector_packet_load_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_PACKET_LOAD_H_
