// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "loom/ops/cfg/ops.h"
#include "loom/target/arch/amdgpu/lower/internal.h"
#include "loom/target/arch/amdgpu/target_refs.h"

enum {
  LOOM_AMDGPU_BRANCH_PLAN_THEN_MASKED_REGION = 0,
  LOOM_AMDGPU_BRANCH_PLAN_IF_ELSE_DIAMOND = 1,
};

typedef struct loom_amdgpu_region_exit_edge_t {
  // Source terminator whose successor edge exits the divergent region.
  const loom_op_t* terminator;
  // Successor ordinal on terminator that exits the divergent region.
  uint8_t successor_index;
} loom_amdgpu_region_exit_edge_t;

typedef struct loom_amdgpu_masked_region_t {
  // Source CFG edges that leave the divergent region for its continuation.
  loom_amdgpu_region_exit_edge_t* exit_edges;
  // Number of entries in exit_edges.
  iree_host_size_t exit_edge_count;
} loom_amdgpu_masked_region_t;

typedef struct loom_amdgpu_branch_plan_t {
  // Low-only block that restores EXEC and branches to restore_dest.
  loom_block_t* restore_block;
  // Effective low destination after EXEC has been restored.
  loom_block_t* restore_dest;
  // Low-only block that computes the inactive else mask for if/else diamonds.
  loom_block_t* else_dispatch_block;
  // Original low destination for the else body.
  loom_block_t* else_body_block;
} loom_amdgpu_branch_plan_t;

static iree_status_t loom_amdgpu_emit_plain_cond_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_condition, loom_block_t* low_true_dest,
    loom_block_t* low_false_dest) {
  loom_op_t* low_cond_br_op = NULL;
  return loom_low_cond_br_build(loom_low_lower_context_builder(context),
                                low_condition, low_true_dest, low_false_dest,
                                source_op->location, &low_cond_br_op);
}

static iree_status_t loom_amdgpu_condition_is_reg_class(
    loom_low_lower_context_t* context, loom_type_t low_type,
    uint16_t reg_class_id, uint32_t unit_count, bool* out_match) {
  *out_match = false;
  bool is_class = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, reg_class_id, &is_class));
  *out_match =
      is_class && loom_type_register_unit_count(low_type) == unit_count;
  return iree_ok_status();
}

static uint16_t loom_amdgpu_source_region_block_index(
    loom_region_t* source_body, const loom_block_t* block) {
  uint16_t block_index = 0;
  const bool found =
      loom_region_try_block_index(source_body, block, &block_index);
  IREE_ASSERT(found);
  (void)found;
  return block_index;
}

static loom_region_t* loom_amdgpu_source_body(
    loom_low_lower_context_t* context) {
  loom_func_like_t source_function =
      loom_low_lower_context_source_function(context);
  loom_region_t* source_body = loom_func_like_body(source_function);
  IREE_ASSERT(source_body != NULL);
  return source_body;
}

static iree_host_size_t loom_amdgpu_successor_edge_count(
    loom_region_t* source_body) {
  iree_host_size_t edge_count = 0;
  for (uint16_t block_index = 0; block_index < source_body->block_count;
       ++block_index) {
    const loom_op_t* terminator =
        loom_block_const_last_op(loom_region_block(source_body, block_index));
    if (terminator != NULL) {
      edge_count += terminator->successor_count;
    }
  }
  return edge_count;
}

static iree_status_t loom_amdgpu_record_masked_region_exit_edge(
    loom_low_lower_context_t* context, const loom_op_t* diagnostic_op,
    const loom_op_t* terminator, uint8_t successor_index,
    loom_amdgpu_masked_region_t* region) {
  if (loom_cfg_br_isa(terminator) && loom_cfg_br_args(terminator).count != 0) {
    return loom_low_lower_emit_branch_constraint(
        context, diagnostic_op, IREE_SV("masked_region_exit_arguments_absent"));
  }
  region->exit_edges[region->exit_edge_count++] =
      (loom_amdgpu_region_exit_edge_t){
          .terminator = terminator,
          .successor_index = successor_index,
      };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_verify_masked_region_single_entry(
    loom_low_lower_context_t* context, const loom_op_t* diagnostic_op,
    loom_region_t* source_body, const uint8_t* in_region,
    const loom_block_t* region_entry) {
  const uint16_t region_entry_index =
      loom_amdgpu_source_region_block_index(source_body, region_entry);

  const loom_block_t* guard_block = diagnostic_op->parent_block;
  for (uint16_t block_index = 0; block_index < source_body->block_count;
       ++block_index) {
    const loom_block_t* source_block =
        loom_region_block(source_body, block_index);
    const loom_op_t* terminator = loom_block_const_last_op(source_block);
    if (terminator == NULL) {
      continue;
    }
    loom_block_t* const* successors = loom_op_const_successors(terminator);
    for (uint8_t successor_index = 0;
         successor_index < terminator->successor_count; ++successor_index) {
      const uint16_t successor_block_index =
          loom_amdgpu_source_region_block_index(source_body,
                                                successors[successor_index]);
      if (!in_region[successor_block_index]) {
        continue;
      }
      if (in_region[block_index]) {
        continue;
      }
      if (source_block == guard_block &&
          successor_block_index == region_entry_index) {
        continue;
      }
      return loom_low_lower_emit_branch_constraint(
          context, diagnostic_op, IREE_SV("masked_region_single_entry"));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_analyze_then_masked_region(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_masked_region_t* out_region) {
  *out_region = (loom_amdgpu_masked_region_t){0};
  loom_block_t* source_entry = loom_cfg_cond_br_true_dest(source_op);
  loom_block_t* source_continuation = loom_cfg_cond_br_false_dest(source_op);
  if (source_entry->arg_count != 0 || source_continuation->arg_count != 0) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("destination_block_arguments_absent"));
  }
  if (source_entry == source_continuation) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_single_continuation"));
  }

  loom_region_t* source_body = loom_amdgpu_source_body(context);
  const iree_host_size_t edge_capacity =
      loom_amdgpu_successor_edge_count(source_body);
  loom_amdgpu_region_exit_edge_t* exit_edges = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, edge_capacity, sizeof(*exit_edges), (void**)&exit_edges));

  uint8_t* in_region = NULL;
  uint16_t* stack = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, source_body->block_count, sizeof(*in_region),
      (void**)&in_region));
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, source_body->block_count, sizeof(*stack), (void**)&stack));
  memset(in_region, 0,
         (iree_host_size_t)source_body->block_count * sizeof(*in_region));

  const uint16_t entry_index =
      loom_amdgpu_source_region_block_index(source_body, source_entry);
  uint16_t stack_count = 0;
  stack[stack_count++] = entry_index;
  in_region[entry_index] = 1;

  loom_amdgpu_masked_region_t region = {
      .exit_edges = exit_edges,
      .exit_edge_count = 0,
  };
  while (stack_count != 0) {
    const uint16_t block_index = stack[--stack_count];
    loom_block_t* source_block = loom_region_block(source_body, block_index);
    const loom_op_t* terminator = loom_block_const_last_op(source_block);
    if (terminator == NULL || terminator->successor_count == 0) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_exits_by_cfg"));
    }

    loom_block_t* const* successors = loom_op_const_successors(terminator);
    for (uint8_t successor_index = 0;
         successor_index < terminator->successor_count; ++successor_index) {
      loom_block_t* successor = successors[successor_index];
      if (successor == source_continuation) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_record_masked_region_exit_edge(
            context, source_op, terminator, successor_index, &region));
        continue;
      }

      const uint16_t successor_block_index =
          loom_amdgpu_source_region_block_index(source_body, successor);
      if (in_region[successor_block_index]) {
        continue;
      }
      in_region[successor_block_index] = 1;
      stack[stack_count++] = successor_block_index;
    }
  }

  if (region.exit_edge_count == 0) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_single_continuation"));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_verify_masked_region_single_entry(
      context, source_op, source_body, in_region, source_entry));
  *out_region = region;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_prepare_then_masked_region(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_amdgpu_masked_region_t region = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_analyze_then_masked_region(context, source_op, &region));

  loom_amdgpu_branch_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*plan), (void**)&plan));
  *plan = (loom_amdgpu_branch_plan_t){0};
  IREE_RETURN_IF_ERROR(
      loom_low_lower_append_low_block(context, &plan->restore_block));
  IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
      context, source_op, 1, plan->restore_block, &plan->restore_dest));

  for (iree_host_size_t i = 0; i < region.exit_edge_count; ++i) {
    loom_block_t* exit_restore_dest = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
        context, region.exit_edges[i].terminator,
        region.exit_edges[i].successor_index, plan->restore_block,
        &exit_restore_dest));
    if (exit_restore_dest != plan->restore_dest) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_exit_edge_interposition"));
    }
  }

  return loom_low_lower_set_branch_plan(
      context, source_op,
      loom_low_lower_plan_make(LOOM_AMDGPU_BRANCH_PLAN_THEN_MASKED_REGION,
                               plan));
}

static bool loom_amdgpu_try_if_else_diamond(
    const loom_op_t* source_op, const loom_op_t** out_true_terminator,
    const loom_op_t** out_false_terminator) {
  *out_true_terminator = NULL;
  *out_false_terminator = NULL;
  loom_block_t* true_dest = loom_cfg_cond_br_true_dest(source_op);
  loom_block_t* false_dest = loom_cfg_cond_br_false_dest(source_op);
  if (true_dest->arg_count != 0 || false_dest->arg_count != 0 ||
      true_dest == false_dest || true_dest->op_count == 0 ||
      false_dest->op_count == 0) {
    return false;
  }
  const loom_op_t* true_terminator = loom_block_const_last_op(true_dest);
  const loom_op_t* false_terminator = loom_block_const_last_op(false_dest);
  if (!loom_cfg_br_isa(true_terminator) || !loom_cfg_br_isa(false_terminator) ||
      loom_cfg_br_args(true_terminator).count != 0 ||
      loom_cfg_br_args(false_terminator).count != 0) {
    return false;
  }
  loom_block_t* merge_block = loom_cfg_br_dest(true_terminator);
  if (merge_block != loom_cfg_br_dest(false_terminator) ||
      merge_block == true_dest || merge_block == false_dest ||
      merge_block->arg_count != 0) {
    return false;
  }
  *out_true_terminator = true_terminator;
  *out_false_terminator = false_terminator;
  return true;
}

static iree_status_t loom_amdgpu_try_emit_if_else_diamond_argument_error(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_emitted) {
  *out_emitted = false;
  loom_block_t* true_dest = loom_cfg_cond_br_true_dest(source_op);
  loom_block_t* false_dest = loom_cfg_cond_br_false_dest(source_op);
  if (true_dest->arg_count != 0 || false_dest->arg_count != 0 ||
      true_dest == false_dest || true_dest->op_count == 0 ||
      false_dest->op_count == 0) {
    return iree_ok_status();
  }
  const loom_op_t* true_terminator = loom_block_const_last_op(true_dest);
  const loom_op_t* false_terminator = loom_block_const_last_op(false_dest);
  if (!loom_cfg_br_isa(true_terminator) || !loom_cfg_br_isa(false_terminator)) {
    return iree_ok_status();
  }
  loom_block_t* merge_block = loom_cfg_br_dest(true_terminator);
  if (merge_block != loom_cfg_br_dest(false_terminator) ||
      merge_block == true_dest || merge_block == false_dest ||
      merge_block->arg_count == 0) {
    return iree_ok_status();
  }
  *out_emitted = true;
  return loom_low_lower_emit_branch_constraint(
      context, source_op, IREE_SV("merge_block_arguments_absent"));
}

static iree_status_t loom_amdgpu_prepare_if_else_diamond(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_op_t* true_terminator, const loom_op_t* false_terminator) {
  loom_amdgpu_branch_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*plan), (void**)&plan));
  *plan = (loom_amdgpu_branch_plan_t){0};
  IREE_RETURN_IF_ERROR(
      loom_low_lower_append_low_block(context, &plan->else_dispatch_block));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_append_low_block(context, &plan->restore_block));

  IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
      context, source_op, 1, plan->else_dispatch_block,
      &plan->else_body_block));

  loom_block_t* previous_true_merge = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
      context, true_terminator, 0, plan->else_dispatch_block,
      &previous_true_merge));
  IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
      context, false_terminator, 0, plan->restore_block, &plan->restore_dest));
  if (previous_true_merge != plan->restore_dest) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("branch_arms_merge_without_arguments"));
  }

  return loom_low_lower_set_branch_plan(
      context, source_op,
      loom_low_lower_plan_make(LOOM_AMDGPU_BRANCH_PLAN_IF_ELSE_DIAMOND, plan));
}

static iree_status_t loom_amdgpu_prepare_exec_mask_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_op_t* true_terminator = NULL;
  const loom_op_t* false_terminator = NULL;
  if (loom_amdgpu_try_if_else_diamond(source_op, &true_terminator,
                                      &false_terminator)) {
    return loom_amdgpu_prepare_if_else_diamond(
        context, source_op, true_terminator, false_terminator);
  }
  bool emitted_diagnostic = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_if_else_diamond_argument_error(
      context, source_op, &emitted_diagnostic));
  if (emitted_diagnostic) {
    return iree_ok_status();
  }
  return loom_amdgpu_prepare_then_masked_region(context, source_op);
}

iree_status_t loom_amdgpu_prepare_branch(void* user_data,
                                         loom_low_lower_context_t* context,
                                         const loom_op_t* source_terminator) {
  (void)user_data;
  if (!loom_cfg_cond_br_isa(source_terminator)) {
    return iree_ok_status();
  }

  loom_type_t condition_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
      context, source_terminator, loom_cfg_cond_br_condition(source_terminator),
      &condition_type));

  bool is_sgpr_mask = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_condition_is_reg_class(
      context, condition_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2,
      &is_sgpr_mask));
  if (!is_sgpr_mask) {
    return iree_ok_status();
  }
  return loom_amdgpu_prepare_exec_mask_branch(context, source_terminator);
}

static iree_status_t loom_amdgpu_emit_exec_restore_block(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_block_t* restore_block,
    loom_block_t* restore_dest) {
  if (restore_block->op_count != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU EXEC restore block was emitted twice");
  }

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, restore_block);
  loom_op_t* restore_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &saved_exec, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
      &restore_op);
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status =
        loom_low_br_build(builder, restore_dest, /*args=*/NULL,
                          /*args_count=*/0, source_op->location, &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_else_dispatch_block(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_type_t active_type,
    const loom_amdgpu_branch_plan_t* plan) {
  if (plan->else_dispatch_block->op_count != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU else dispatch block was emitted twice");
  }

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, plan->else_dispatch_block);

  loom_op_t* else_active_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64_EXEC,
      &saved_exec, 1, loom_named_attr_slice_empty(), &active_type, 1,
      &else_active_op);
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_emit_plain_cond_branch(
        context, source_op, loom_op_const_results(else_active_op)[0],
        plan->else_body_block, plan->restore_block);
  }

  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_exec_mask_cond_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_condition, loom_block_t* low_true_dest,
    loom_block_t* low_false_dest, loom_type_t condition_type) {
  loom_low_lower_plan_t branch_plan = loom_low_lower_plan_empty();
  if (!loom_low_lower_lookup_branch_plan(context, source_op, &branch_plan)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU divergent branch has no prepared plan");
  }
  const loom_amdgpu_branch_plan_t* plan =
      (const loom_amdgpu_branch_plan_t*)branch_plan.target_data;
  IREE_ASSERT(plan != NULL);
  if (branch_plan.id != LOOM_AMDGPU_BRANCH_PLAN_THEN_MASKED_REGION &&
      branch_plan.id != LOOM_AMDGPU_BRANCH_PLAN_IF_ELSE_DIAMOND) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU divergent branch plan id is invalid");
  }

  loom_type_t active_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &active_type));
  const loom_type_t result_types[] = {condition_type, active_type};

  loom_op_t* saveexec_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64,
      &low_condition, 1, loom_named_attr_slice_empty(), result_types,
      IREE_ARRAYSIZE(result_types), &saveexec_op));
  const loom_value_id_t saved_exec = loom_op_const_results(saveexec_op)[0];
  const loom_value_id_t active = loom_op_const_results(saveexec_op)[1];
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_exec_restore_block(
      context, source_op, saved_exec, plan->restore_block, plan->restore_dest));

  if (branch_plan.id == LOOM_AMDGPU_BRANCH_PLAN_IF_ELSE_DIAMOND) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_else_dispatch_block(
        context, source_op, saved_exec, active_type, plan));
  }

  return loom_amdgpu_emit_plain_cond_branch(context, source_op, active,
                                            low_true_dest, low_false_dest);
}

iree_status_t loom_amdgpu_emit_cond_branch(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_value_id_t low_condition,
                                           loom_block_t* low_true_dest,
                                           loom_block_t* low_false_dest) {
  (void)user_data;
  loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t condition_type = loom_module_value_type(module, low_condition);
  bool is_scc = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_condition_is_reg_class(
      context, condition_type, LOOM_AMDGPU_REG_CLASS_ID_SCC, 1, &is_scc));
  if (is_scc) {
    return loom_amdgpu_emit_plain_cond_branch(context, source_op, low_condition,
                                              low_true_dest, low_false_dest);
  }

  bool is_sgpr_mask = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_condition_is_reg_class(
      context, condition_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2,
      &is_sgpr_mask));
  if (is_sgpr_mask) {
    return loom_amdgpu_emit_exec_mask_cond_branch(
        context, source_op, low_condition, low_true_dest, low_false_dest,
        condition_type);
  }

  return loom_low_lower_emit_branch_condition_type_unsupported(
      context, source_op, condition_type, IREE_SV("amdgpu.branch_condition"));
}
