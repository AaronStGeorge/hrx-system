// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/legalization.h"

#include "loom/ir/module.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"

static bool loom_amdgpu_legalizer_descriptor_set_is_amdgpu(
    const loom_low_descriptor_set_t* descriptor_set) {
  return descriptor_set != NULL &&
         descriptor_set->target_stable_id == LOOM_AMDGPU_TARGET_STABLE_ID;
}

static bool loom_amdgpu_match_all_value_type_is_supported(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
}

static iree_status_t loom_amdgpu_legalize_kernel_subgroup_match_all(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_amdgpu_legalizer_descriptor_set_is_amdgpu(
          context->descriptor_set)) {
    return iree_ok_status();
  }

  const loom_value_id_t value = loom_kernel_subgroup_match_all_value(op);
  const loom_type_t value_type = loom_module_value_type(context->module, value);
  if (!loom_amdgpu_match_all_value_type_is_supported(value_type)) {
    *out_result = (loom_target_legalizer_result_t){
        .action = context->mode == LOOM_TARGET_LEGALIZATION_MODE_FINAL
                      ? LOOM_TARGET_LEGALIZER_ACTION_REJECT_UNSUPPORTED_FINAL
                      : LOOM_TARGET_LEGALIZER_ACTION_DEFER,
    };
    return iree_ok_status();
  }

  const loom_value_id_t mask = loom_kernel_subgroup_match_all_mask(op);
  const loom_type_t mask_type = loom_module_value_type(context->module, mask);
  const loom_type_t i1_type = loom_type_scalar(LOOM_SCALAR_TYPE_I1);

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);

  loom_op_t* first_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_broadcast_first_build(
      &rewriter->builder, value, value_type, op->location, &first_op));
  const loom_value_id_t first_value =
      loom_kernel_subgroup_broadcast_first_result(first_op);

  loom_op_t* equal_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
      &rewriter->builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, value, first_value,
      value_type, i1_type, op->location, &equal_op));
  const loom_value_id_t equal = loom_scalar_cmpi_result(equal_op);

  loom_op_t* all_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_vote_all_build(
      &rewriter->builder, equal, i1_type, op->location, &all_op));
  const loom_value_id_t all_equal =
      loom_kernel_subgroup_vote_all_result(all_op);

  loom_op_t* active_mask_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_active_mask_build(
      &rewriter->builder, mask_type, op->location, &active_mask_op));
  const loom_value_id_t active_mask =
      loom_kernel_subgroup_active_mask_mask(active_mask_op);

  loom_op_t* zero_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      &rewriter->builder, loom_attr_i64(0), mask_type, op->location, &zero_op));
  const loom_value_id_t zero_mask = loom_scalar_constant_result(zero_op);

  loom_op_t* selected_mask_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_select_build(&rewriter->builder, all_equal,
                                             active_mask, zero_mask, mask_type,
                                             op->location, &selected_mask_op));
  const loom_value_id_t replacements[] = {
      loom_scf_select_result(selected_mask_op),
      all_equal,
  };
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements, IREE_ARRAYSIZE(replacements),
      value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, replacements, IREE_ARRAYSIZE(replacements)));

  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

static const loom_target_legalizer_entry_t kAmdgpuLegalizerEntries[] = {
    {
        .root_kind = LOOM_OP_KERNEL_SUBGROUP_MATCH_ALL,
        .legalize = loom_amdgpu_legalize_kernel_subgroup_match_all,
    },
};

const loom_target_legalizer_provider_t
    loom_amdgpu_target_legalizer_provider_storage = {
        .name = IREE_SVL("amdgpu"),
        .entries = kAmdgpuLegalizerEntries,
        .entry_count = IREE_ARRAYSIZE(kAmdgpuLegalizerEntries),
};

const loom_target_legalizer_provider_t* loom_amdgpu_target_legalizer_provider(
    void) {
  return &loom_amdgpu_target_legalizer_provider_storage;
}
