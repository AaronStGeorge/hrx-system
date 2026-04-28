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

static iree_status_t loom_amdgpu_verify_then_only_divergent_shape(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
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
  if (!loom_cfg_br_isa(true_terminator) ||
      loom_cfg_br_dest(true_terminator) != source_false_dest ||
      loom_cfg_br_args(true_terminator).count != 0) {
    return loom_amdgpu_emit_cond_branch_reject(
        context, source_op,
        IREE_SV("AMDGPU divergent branch lowering currently requires a "
                "then-only diamond"));
  }
  return iree_ok_status();
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
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_verify_then_only_divergent_shape(context, source_op));

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
      context, source_op, saved_exec, low_false_dest));
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

  return loom_amdgpu_emit_cond_branch_reject(
      context, source_op,
      IREE_SV("AMDGPU conditional branch lowering requires an SCC condition or "
              "an SGPR-pair lane mask"));
}
