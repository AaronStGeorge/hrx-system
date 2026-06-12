// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/feedback.h"

#include <stdint.h>

#include "loom/codegen/low/builder.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/feedback_abi.h"
#include "loom/target/arch/amdgpu/lower/data_symbol.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static loom_amdgpu_feedback_config_values_t
loom_amdgpu_feedback_config_values_empty(void) {
  return (loom_amdgpu_feedback_config_values_t){
      .address = LOOM_VALUE_ID_INVALID,
      .flags = LOOM_VALUE_ID_INVALID,
      .channel_base = LOOM_VALUE_ID_INVALID,
      .notify_signal = LOOM_VALUE_ID_INVALID,
  };
}

static loom_amdgpu_feedback_channel_header_values_t
loom_amdgpu_feedback_channel_header_values_empty(void) {
  return (loom_amdgpu_feedback_channel_header_values_t){
      .address = LOOM_VALUE_ID_INVALID,
      .record_length = LOOM_VALUE_ID_INVALID,
      .abi_version = LOOM_VALUE_ID_INVALID,
      .flags = LOOM_VALUE_ID_INVALID,
      .ring_base = LOOM_VALUE_ID_INVALID,
      .ring_capacity = LOOM_VALUE_ID_INVALID,
  };
}

static iree_status_t loom_amdgpu_feedback_build_offset_attr(
    loom_builder_t* builder, uint32_t byte_offset,
    loom_named_attr_t* out_attr) {
  loom_string_id_t offset_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_builder_intern_string(builder, IREE_SV("offset"), &offset_name_id));
  *out_attr = (loom_named_attr_t){
      .name_id = offset_name_id,
      .value = loom_attr_i64(byte_offset),
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_feedback_build_scalar_load(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t base_address,
    uint32_t byte_offset, loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_descriptor_ref(
      builder, descriptor_set, descriptor_ref, &descriptor, &opcode_id));

  loom_named_attr_t offset_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_offset_attr(
      builder, byte_offset, &offset_attr));
  loom_op_t* load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, opcode_id, &base_address,
      /*operand_count=*/1, loom_make_named_attr_slice(&offset_attr, 1),
      &result_type, /*result_count=*/1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &load_op));
  *out_value = loom_value_slice_get(loom_low_op_results(load_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_feedback_config_values(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t config_symbol, loom_location_id_t location,
    loom_amdgpu_feedback_config_values_t* out_values) {
  IREE_ASSERT_ARGUMENT(out_values);
  *out_values = loom_amdgpu_feedback_config_values_empty();

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &sgpr_x2_type));

  loom_amdgpu_feedback_config_values_t values =
      loom_amdgpu_feedback_config_values_empty();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_build_data_symbol_address(builder, descriptor_set,
                                            (loom_amdgpu_data_symbol_address_t){
                                                .symbol = config_symbol,
                                                .byte_offset = 0,
                                            },
                                            location, &values.address));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_scalar_load(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORD_OFFSET_ONLY, values.address,
      LOOM_AMDGPU_FEEDBACK_CONFIG_FLAGS_OFFSET, sgpr_type, location,
      &values.flags));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_scalar_load(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORDX2_OFFSET_ONLY, values.address,
      LOOM_AMDGPU_FEEDBACK_CONFIG_CHANNEL_BASE_OFFSET, sgpr_x2_type, location,
      &values.channel_base));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_scalar_load(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORDX2_OFFSET_ONLY, values.address,
      LOOM_AMDGPU_FEEDBACK_CONFIG_NOTIFY_SIGNAL_OFFSET, sgpr_x2_type, location,
      &values.notify_signal));

  *out_values = values;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_feedback_channel_header_values(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t channel_base, loom_location_id_t location,
    loom_amdgpu_feedback_channel_header_values_t* out_values) {
  IREE_ASSERT_ARGUMENT(out_values);
  *out_values = loom_amdgpu_feedback_channel_header_values_empty();

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &sgpr_x2_type));

  loom_amdgpu_feedback_channel_header_values_t values =
      loom_amdgpu_feedback_channel_header_values_empty();
  values.address = channel_base;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_scalar_load(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORD_OFFSET_ONLY, values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_RECORD_LENGTH_OFFSET, sgpr_type, location,
      &values.record_length));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_scalar_load(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORD_OFFSET_ONLY, values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_ABI_VERSION_OFFSET, sgpr_type, location,
      &values.abi_version));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_scalar_load(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORD_OFFSET_ONLY, values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_FLAGS_OFFSET, sgpr_type, location,
      &values.flags));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_scalar_load(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORDX2_OFFSET_ONLY, values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_RING_BASE_OFFSET, sgpr_x2_type, location,
      &values.ring_base));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_scalar_load(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORDX2_OFFSET_ONLY, values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_RING_CAPACITY_OFFSET, sgpr_x2_type, location,
      &values.ring_capacity));

  *out_values = values;
  return iree_ok_status();
}
