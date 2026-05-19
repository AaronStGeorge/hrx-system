// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/copy_decision.h"

#include <string.h>

#include "loom/codegen/low/allocation/move_topology.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

static bool loom_low_allocation_copy_decision_value_ordinal_for_value(
    const loom_low_allocation_copy_decision_context_t* context,
    loom_value_id_t value_id, loom_value_ordinal_t* out_value_ordinal) {
  const loom_value_ordinal_t value_ordinal =
      loom_module_value_ordinal_scratch_lookup(context->module, value_id);
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      value_ordinal >= context->liveness->value_count) {
    return false;
  }
  *out_value_ordinal = value_ordinal;
  return true;
}

static iree_status_t
loom_low_allocation_copy_decision_assignment_index_for_value(
    const loom_low_allocation_copy_decision_context_t* context,
    loom_value_id_t value_id, uint32_t* out_assignment_index) {
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (loom_low_allocation_copy_decision_value_ordinal_for_value(
          context, value_id, &value_ordinal)) {
    const uint32_t assignment_index =
        context->assignment_indices_by_value_ordinal[value_ordinal];
    if (assignment_index != UINT32_MAX) {
      *out_assignment_index = assignment_index;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "low allocation references value %u without an "
                          "assignment",
                          (unsigned)value_id);
}

static iree_status_t loom_low_allocation_copy_decision_plan_record(
    const loom_low_allocation_copy_decision_context_t* context,
    loom_low_allocation_copy_decision_plan_t* plan, const loom_op_t* op) {
  uint32_t source_assignment_index = 0;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_copy_decision_assignment_index_for_value(
          context, loom_low_copy_source(op), &source_assignment_index));
  uint32_t result_assignment_index = 0;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_copy_decision_assignment_index_for_value(
          context, loom_low_copy_result(op), &result_assignment_index));

  const loom_low_allocation_assignment_t* source_assignment =
      &context->assignments[source_assignment_index];
  const loom_low_allocation_assignment_t* result_assignment =
      &context->assignments[result_assignment_index];
  const bool coalesced =
      loom_low_allocation_assignment_is_register_like(source_assignment) &&
      loom_low_allocation_assignment_is_register_like(result_assignment) &&
      loom_low_allocation_storage_assignment_locations_share(
          context->descriptor_set, source_assignment, result_assignment);
  plan->decisions[plan->decision_count++] =
      (loom_low_allocation_copy_decision_t){
          .source_value_id = loom_low_copy_source(op),
          .result_value_id = loom_low_copy_result(op),
          .source_assignment_index = source_assignment_index,
          .result_assignment_index = result_assignment_index,
          .kind = coalesced ? LOOM_LOW_ALLOCATION_COPY_COALESCED
                            : LOOM_LOW_ALLOCATION_COPY_MATERIALIZED,
      };
  if (coalesced) {
    ++plan->coalesced_count;
  } else {
    ++plan->materialized_count;
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_copy_decision_plan_build(
    const loom_low_allocation_copy_decision_context_t* context,
    iree_arena_allocator_t* arena,
    loom_low_allocation_copy_decision_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (loom_low_allocation_copy_decision_plan_t){0};

  const iree_host_size_t copy_count =
      loom_low_allocation_move_topology_count_copy_ops(context->body);
  if (copy_count == 0) {
    return iree_ok_status();
  }

  loom_low_allocation_copy_decision_plan_t plan = {0};
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, copy_count, sizeof(*plan.decisions), (void**)&plan.decisions));
  memset(plan.decisions, 0, copy_count * sizeof(*plan.decisions));

  loom_block_t* block = NULL;
  loom_region_for_each_block(context->body, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (!loom_low_copy_isa(op)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_copy_decision_plan_record(context, &plan, op));
    }
  }
  *out_plan = plan;
  return iree_ok_status();
}
