// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/cfg/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/lower/internal.h"

static iree_status_t loom_amdgpu_emit_plain_cond_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_condition, loom_block_t* low_true_dest,
    loom_block_t* low_false_dest) {
  loom_op_t* low_cond_br_op = NULL;
  return loom_low_cond_br_build(loom_low_lower_context_builder(context),
                                low_condition, low_true_dest, low_false_dest,
                                source_op->location, &low_cond_br_op);
}

static iree_status_t loom_amdgpu_emit_cond_branch_reject(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t reason) {
  return loom_low_lower_emit_reject(context, source_op, IREE_SV("branch"),
                                    IREE_SV("cfg.cond_br"), reason);
}

static iree_status_t loom_amdgpu_condition_is_reg_class(
    loom_low_lower_context_t* context, loom_type_t low_type,
    uint16_t reg_class_id, uint32_t unit_count, bool* out_match) {
  IREE_ASSERT_ARGUMENT(out_match);
  *out_match = false;
  bool is_class = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, reg_class_id, &is_class));
  *out_match =
      is_class && loom_type_register_unit_count(low_type) == unit_count;
  return iree_ok_status();
}

typedef enum loom_amdgpu_divergent_branch_shape_e {
  LOOM_AMDGPU_DIVERGENT_BRANCH_SHAPE_THEN_ONLY = 0,
  LOOM_AMDGPU_DIVERGENT_BRANCH_SHAPE_IF_ELSE = 1,
} loom_amdgpu_divergent_branch_shape_t;

typedef struct loom_amdgpu_divergent_branch_t {
  loom_amdgpu_divergent_branch_shape_t shape;
  const loom_op_t* true_branch_op;
  loom_block_t* low_merge_block;
} loom_amdgpu_divergent_branch_t;

static iree_status_t loom_amdgpu_require_empty_branch_to(
    loom_low_lower_context_t* context, const loom_op_t* source_cond_branch_op,
    const loom_op_t* source_branch_op, loom_block_t* expected_dest,
    iree_string_view_t reason) {
  if (!loom_cfg_br_isa(source_branch_op) ||
      loom_cfg_br_dest(source_branch_op) != expected_dest ||
      loom_cfg_br_args(source_branch_op).count != 0) {
    return loom_amdgpu_emit_cond_branch_reject(context, source_cond_branch_op,
                                               reason);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_analyze_divergent_branch_shape(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_divergent_branch_t* out_branch) {
  *out_branch = (loom_amdgpu_divergent_branch_t){0};
  loom_block_t* source_true_dest = loom_cfg_cond_br_true_dest(source_op);
  loom_block_t* source_false_dest = loom_cfg_cond_br_false_dest(source_op);
  if (source_true_dest->arg_count != 0 || source_false_dest->arg_count != 0) {
    return loom_amdgpu_emit_cond_branch_reject(
        context, source_op,
        IREE_SV("AMDGPU divergent branch lowering currently requires "
                "destination blocks without arguments"));
  }
  if (source_true_dest->op_count == 0) {
    return loom_amdgpu_emit_cond_branch_reject(
        context, source_op,
        IREE_SV("AMDGPU divergent branch lowering requires a then block "
                "terminator"));
  }

  const loom_op_t* true_terminator = loom_block_const_last_op(source_true_dest);
  if (loom_cfg_br_isa(true_terminator) &&
      loom_cfg_br_dest(true_terminator) == source_false_dest &&
      loom_cfg_br_args(true_terminator).count == 0 &&
      source_true_dest != source_false_dest) {
    loom_block_t* low_merge_block = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_block(context, source_false_dest,
                                                     &low_merge_block));
    *out_branch = (loom_amdgpu_divergent_branch_t){
        .shape = LOOM_AMDGPU_DIVERGENT_BRANCH_SHAPE_THEN_ONLY,
        .true_branch_op = true_terminator,
        .low_merge_block = low_merge_block,
    };
    return iree_ok_status();
  }

  if (source_false_dest->op_count == 0) {
    return loom_amdgpu_emit_cond_branch_reject(
        context, source_op,
        IREE_SV("AMDGPU divergent branch lowering currently requires a "
                "then-only or if/else diamond"));
  }
  if (!loom_cfg_br_isa(true_terminator)) {
    return loom_amdgpu_emit_cond_branch_reject(
        context, source_op,
        IREE_SV("AMDGPU divergent branch lowering currently requires a "
                "then-only or if/else diamond"));
  }
  loom_block_t* source_merge_block = loom_cfg_br_dest(true_terminator);
  if (source_merge_block == source_true_dest ||
      source_merge_block == source_false_dest) {
    return loom_amdgpu_emit_cond_branch_reject(
        context, source_op,
        IREE_SV("AMDGPU divergent if/else lowering requires a distinct merge "
                "block"));
  }
  if (source_merge_block->arg_count != 0) {
    return loom_amdgpu_emit_cond_branch_reject(
        context, source_op,
        IREE_SV("AMDGPU divergent branch lowering currently requires a merge "
                "block without arguments"));
  }
  const loom_op_t* false_terminator =
      loom_block_const_last_op(source_false_dest);
  IREE_RETURN_IF_ERROR(loom_amdgpu_require_empty_branch_to(
      context, source_op, true_terminator, source_merge_block,
      IREE_SV("AMDGPU divergent branch lowering currently requires a "
              "then-only or if/else diamond")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_require_empty_branch_to(
      context, source_op, false_terminator, source_merge_block,
      IREE_SV("AMDGPU divergent if/else lowering requires both arms to merge "
              "without arguments")));

  loom_block_t* low_merge_block = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_block(context, source_merge_block,
                                                   &low_merge_block));
  *out_branch = (loom_amdgpu_divergent_branch_t){
      .shape = LOOM_AMDGPU_DIVERGENT_BRANCH_SHAPE_IF_ELSE,
      .true_branch_op = true_terminator,
      .low_merge_block = low_merge_block,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_else_dispatch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_type_t active_type,
    loom_block_t* low_else_body, loom_block_t* low_merge_block,
    loom_block_t** out_dispatch_block) {
  *out_dispatch_block = NULL;
  loom_block_t* dispatch_block = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_append_low_block(context, &dispatch_block));

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, dispatch_block);

  loom_op_t* else_active_op = NULL;
  iree_status_t status = loom_low_lower_emit_descriptor_op(
      context, LOOM_AMDGPU_DESCRIPTOR_ID_S_XOR_B64_EXEC, &saved_exec, 1,
      loom_named_attr_slice_empty(), &active_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &else_active_op);
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_emit_plain_cond_branch(
        context, source_op, loom_op_const_results(else_active_op)[0],
        low_else_body, low_merge_block);
  }

  loom_builder_restore(builder, saved_ip);
  if (iree_status_is_ok(status)) {
    *out_dispatch_block = dispatch_block;
  }
  return status;
}

static iree_status_t loom_amdgpu_emit_exec_restore(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_block_t* low_merge_block) {
  if (low_merge_block->op_count != 0) {
    return loom_amdgpu_emit_cond_branch_reject(
        context, source_op,
        IREE_SV("AMDGPU divergent branch lowering requires the merge block to "
                "be emitted after the branch"));
  }

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, low_merge_block);
  loom_op_t* restore_op = NULL;
  iree_status_t status = loom_low_lower_emit_descriptor_op(
      context, LOOM_AMDGPU_DESCRIPTOR_ID_S_MOV_B64_EXEC, &saved_exec, 1,
      loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &restore_op);
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_exec_mask_cond_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_condition, loom_block_t* low_true_dest,
    loom_block_t* low_false_dest, loom_type_t condition_type) {
  loom_amdgpu_divergent_branch_t branch = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_analyze_divergent_branch_shape(context, source_op, &branch));

  loom_type_t active_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &active_type));
  const loom_type_t result_types[] = {condition_type, active_type};

  loom_op_t* saveexec_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_descriptor_op(
      context, LOOM_AMDGPU_DESCRIPTOR_ID_S_AND_SAVEEXEC_B64, &low_condition, 1,
      loom_named_attr_slice_empty(), result_types, IREE_ARRAYSIZE(result_types),
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &saveexec_op));
  const loom_value_id_t saved_exec = loom_op_const_results(saveexec_op)[0];
  const loom_value_id_t active = loom_op_const_results(saveexec_op)[1];
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_exec_restore(
      context, source_op, saved_exec, branch.low_merge_block));
  if (branch.shape == LOOM_AMDGPU_DIVERGENT_BRANCH_SHAPE_THEN_ONLY) {
    return loom_amdgpu_emit_plain_cond_branch(context, source_op, active,
                                              low_true_dest, low_false_dest);
  }

  loom_block_t* low_dispatch_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_else_dispatch(
      context, source_op, saved_exec, active_type, low_false_dest,
      branch.low_merge_block, &low_dispatch_block));
  IREE_RETURN_IF_ERROR(loom_low_lower_redirect_empty_branch_dest(
      context, branch.true_branch_op, low_dispatch_block));
  return loom_amdgpu_emit_plain_cond_branch(context, source_op, active,
                                            low_true_dest, low_dispatch_block);
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

  return loom_amdgpu_emit_cond_branch_reject(
      context, source_op,
      IREE_SV("AMDGPU conditional branch lowering requires an SCC condition or "
              "an SGPR-pair lane mask"));
}
