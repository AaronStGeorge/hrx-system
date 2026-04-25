// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower_rules.h"
#include "loom/ir/context.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/lower_internal.h"

static iree_status_t loom_amdgpu_dot_match_register(
    const loom_low_lower_rule_match_context_t* context,
    uint16_t descriptor_register_class_id, uint32_t register_unit_count,
    loom_low_lower_rule_mapped_value_t* out_mapped_value) {
  IREE_ASSERT_ARGUMENT(out_mapped_value);
  IREE_ASSERT_LT(descriptor_register_class_id,
                 context->descriptor_set->reg_class_count);
  *out_mapped_value = loom_low_lower_rule_mapped_value_register(
      descriptor_register_class_id, register_unit_count);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_dot_match_map_value(
    void* user_data, const loom_low_lower_rule_match_context_t* context,
    const loom_op_t* source_op, loom_value_id_t source_value_id,
    loom_low_lower_rule_mapped_value_t* out_mapped_value) {
  (void)user_data;
  (void)source_op;
  IREE_ASSERT_ARGUMENT(out_mapped_value);
  *out_mapped_value = loom_low_lower_rule_mapped_value_none();
  IREE_ASSERT_LT(source_value_id, context->module->values.count);

  const loom_type_t source_type =
      loom_module_value_type(context->module, source_value_id);
  if (loom_amdgpu_type_is_f32(source_type)) {
    return loom_amdgpu_dot_match_register(
        context, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, out_mapped_value);
  }
  if ((loom_amdgpu_type_is_i32(source_type) ||
       loom_amdgpu_type_is_address_scalar(source_type)) &&
      loom_amdgpu_module_value_prefers_vgpr(context->module, source_value_id)) {
    return loom_amdgpu_dot_match_register(
        context, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, out_mapped_value);
  }
  if (loom_amdgpu_type_is_i32(source_type) ||
      loom_amdgpu_type_is_address_scalar(source_type)) {
    return loom_amdgpu_dot_match_register(
        context, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, out_mapped_value);
  }

  const uint32_t vector_lane_count =
      loom_amdgpu_vector_32bit_lane_count(source_type);
  if (vector_lane_count != 0) {
    return loom_amdgpu_dot_match_register(context,
                                          LOOM_AMDGPU_REG_CLASS_ID_VGPR,
                                          vector_lane_count, out_mapped_value);
  }
  const uint32_t mask_lane_count =
      loom_amdgpu_vector_i1_lane_count(source_type);
  if (mask_lane_count != 0) {
    return loom_amdgpu_dot_match_register(
        context, LOOM_AMDGPU_REG_CLASS_ID_SGPR, mask_lane_count * 2u,
        out_mapped_value);
  }

  uint32_t unused_payload_bit_count = 0;
  uint32_t packed_register_count = 0;
  if (loom_amdgpu_type_packed_integer_storage(
          source_type, &unused_payload_bit_count, &packed_register_count)) {
    return loom_amdgpu_dot_match_register(
        context, LOOM_AMDGPU_REG_CLASS_ID_VGPR, packed_register_count,
        out_mapped_value);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_dot_descriptor_key(
    const loom_low_descriptor_set_t* descriptor_set, uint64_t descriptor_id,
    iree_string_view_t* out_key) {
  IREE_ASSERT_ARGUMENT(out_key);
  *out_key = iree_string_view_empty();
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_lookup_descriptor_by_id(descriptor_set,
                                                      descriptor_id);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "selected AMDGPU dot rule references a missing descriptor");
  }
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, descriptor_ordinal);
  if (descriptor == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "selected AMDGPU dot descriptor ordinal is out of range");
  }
  return loom_low_descriptor_set_string(descriptor_set,
                                        descriptor->key_string_offset, out_key);
}

iree_status_t loom_amdgpu_low_legality_verify_vector_dot(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_low_lower_rule_match_context_t match_context = {
      .module = loom_target_low_legality_module(context),
      .descriptor_set = loom_target_low_legality_descriptor_set(context),
      .map_value =
          {
              .fn = loom_amdgpu_dot_match_map_value,
              .user_data = NULL,
          },
  };
  loom_low_lower_rule_selection_t selection = {0};
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_set_select_with_match_context(
      &match_context, &loom_amdgpu_dot_rule_set, op, &selection));
  if (selection.rule == NULL) {
    const loom_low_lower_diagnostic_t* diagnostic =
        loom_low_lower_rule_set_selection_diagnostic(&loom_amdgpu_dot_rule_set,
                                                     selection);
    if (diagnostic != NULL) {
      return loom_target_low_legality_reject(
          context, provider, op, diagnostic->subject_kind,
          diagnostic->subject_name, diagnostic->reason);
    }
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("op"),
        loom_op_name(loom_target_low_legality_module(context), op),
        IREE_SV("AMDGPU source-to-low supports vector.dotf, vector.dot4i, and "
                "vector.dot8i4 forms; other vector dot families require "
                "additional target contracts"));
  }

  const uint64_t descriptor_id = loom_low_lower_rule_first_descriptor_id(
      &loom_amdgpu_dot_rule_set, selection.rule);
  if (descriptor_id == LOOM_LOW_DESCRIPTOR_ID_NONE) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "selected AMDGPU vector dot rule has no descriptor-backed emit");
  }
  iree_string_view_t descriptor_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_dot_descriptor_key(
      loom_target_low_legality_descriptor_set(context), descriptor_id,
      &descriptor_key));
  const iree_string_view_t reason =
      descriptor_id == LOOM_AMDGPU_DESCRIPTOR_ID_V_FMA_F32
          ? IREE_SV("selected AMDGPU f32 dot FMA contract")
          : IREE_SV("selected AMDGPU packed dot descriptor");
  return loom_target_low_legality_record_contract(
      context, provider, op, descriptor_key, IREE_SV("selected"), reason);
}
