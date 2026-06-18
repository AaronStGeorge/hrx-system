// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/sanitizer_race_report.h"

#include "loom/codegen/low/builder.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/abi/feedback.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/feedback.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"

static iree_status_t loom_amdgpu_sanitizer_race_insert_block_after(
    loom_builder_t* builder, loom_block_t* after_block,
    loom_block_t** out_block) {
  *out_block = NULL;
  if (after_block->parent_region == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU sanitizer race report requires a low region block");
  }
  return loom_region_insert_block(builder->module, after_block->parent_region,
                                  (uint16_t)(after_block->region_index + 1),
                                  out_block);
}

static iree_status_t loom_amdgpu_sanitizer_race_require_data_register(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t value, uint32_t unit_count, iree_string_view_t value_name) {
  if (value >= builder->module->values.count) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU sanitizer race report value `%.*s` is an invalid low value",
        (int)value_name.size, value_name.data);
  }
  const loom_type_t type = loom_module_value_type(builder->module, value);
  if (!loom_low_type_is_register(type) ||
      loom_low_register_type_descriptor_set_stable_id(type) !=
          descriptor_set->stable_id ||
      loom_low_register_type_unit_count(type) != unit_count) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU sanitizer race report value `%.*s` has an unsupported register "
        "shape",
        (int)value_name.size, value_name.data);
  }
  const uint16_t register_class = loom_low_register_type_class_id(type);
  if (register_class != LOOM_AMDGPU_REG_CLASS_ID_SGPR &&
      register_class != LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU sanitizer race report value `%.*s` must be an SGPR or VGPR",
        (int)value_name.size, value_name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_require_register_class(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t value, uint32_t unit_count, uint16_t register_class,
    iree_string_view_t value_name) {
  if (value >= builder->module->values.count) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU sanitizer race report value `%.*s` is an invalid low value",
        (int)value_name.size, value_name.data);
  }
  const loom_type_t type = loom_module_value_type(builder->module, value);
  if (!loom_low_type_is_register(type) ||
      loom_low_register_type_descriptor_set_stable_id(type) !=
          descriptor_set->stable_id ||
      loom_low_register_type_unit_count(type) != unit_count ||
      loom_low_register_type_class_id(type) != register_class) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU sanitizer race report value `%.*s` has an unsupported register "
        "shape",
        (int)value_name.size, value_name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_descriptor_op(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    const loom_type_t* result_types, iree_host_size_t result_count,
    loom_location_id_t location, loom_op_t** out_op) {
  *out_op = NULL;
  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_descriptor_ref(
      builder, descriptor_set, descriptor_ref, &descriptor, &opcode_id));
  return loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, opcode_id, operands, operand_count,
      loom_make_named_attr_slice(NULL, 0), result_types, result_count,
      /*tied_results=*/NULL, /*tied_result_count=*/0, location, out_op);
}

static iree_status_t loom_amdgpu_sanitizer_race_build_u32_attr(
    loom_builder_t* builder, iree_string_view_t name, uint32_t value,
    loom_named_attr_t* out_attr) {
  *out_attr = (loom_named_attr_t){0};
  IREE_RETURN_IF_ERROR(
      loom_builder_intern_string(builder, name, &out_attr->name_id));
  out_attr->value = loom_attr_i64(value);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_const_u32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t value,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_descriptor_ref(
      builder, descriptor_set, descriptor_ref, &descriptor, &opcode_id));

  loom_named_attr_t imm32_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_u32_attr(
      builder, IREE_SV("imm32"), value, &imm32_attr));
  loom_op_t* const_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
      builder, descriptor_set, descriptor, opcode_id,
      loom_make_named_attr_slice(&imm32_attr, 1), result_type, location,
      &const_op));
  *out_value = loom_low_const_result(const_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_sgpr_u32_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t value, loom_location_id_t location, loom_value_id_t* out_value) {
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  return loom_amdgpu_sanitizer_race_build_const_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, value,
      sgpr_type, location, out_value);
}

static iree_status_t loom_amdgpu_sanitizer_race_build_sgpr_u64_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint64_t value, loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_sgpr_u32_const(
      builder, descriptor_set, (uint32_t)value, location, &low_value_lo));
  loom_value_id_t low_value_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_sgpr_u32_const(
      builder, descriptor_set, (uint32_t)(value >> 32), location,
      &low_value_hi));

  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &sgpr_x2_type));
  const loom_value_id_t parts[] = {low_value_lo, low_value_hi};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, parts, IREE_ARRAYSIZE(parts), sgpr_x2_type,
                            location, &concat_op));
  *out_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_sgpr64_nonzero_scc(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t value, loom_location_id_t location,
    loom_value_id_t* out_scc) {
  *out_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_register_class(
      builder, descriptor_set, value, 2, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
      IREE_SV("value")));
  loom_value_id_t zero64 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_sgpr_u64_const(
      builder, descriptor_set, 0, location, &zero64));

  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SCC, 1, &scc_type));
  const loom_value_id_t operands[] = {value, zero64};
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LG_U64,
      operands, IREE_ARRAYSIZE(operands), &scc_type, /*result_count=*/1,
      location, &compare_op));
  *out_scc = loom_value_slice_get(loom_low_op_results(compare_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_exec_narrow(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t lane_mask, loom_location_id_t location,
    loom_value_id_t* out_saved_exec) {
  if (out_saved_exec != NULL) {
    *out_saved_exec = LOOM_VALUE_ID_INVALID;
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_register_class(
      builder, descriptor_set, lane_mask, 2, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
      IREE_SV("lane_mask")));
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &sgpr_x2_type));
  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SCC, 1, &scc_type));
  const loom_type_t result_types[] = {sgpr_x2_type, scc_type};
  loom_op_t* saveexec_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64,
      &lane_mask, /*operand_count=*/1, result_types,
      IREE_ARRAYSIZE(result_types), location, &saveexec_op));
  if (out_saved_exec != NULL) {
    *out_saved_exec = loom_value_slice_get(loom_low_op_results(saveexec_op), 0);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_build_exec_restore(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t saved_exec, loom_location_id_t location) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_register_class(
      builder, descriptor_set, saved_exec, 2, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
      IREE_SV("saved_exec")));
  loom_op_t* restore_op = NULL;
  return loom_amdgpu_sanitizer_race_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &saved_exec, /*operand_count=*/1, /*result_types=*/NULL,
      /*result_count=*/0, location, &restore_op);
}

static iree_status_t loom_amdgpu_sanitizer_validate_race_report_source_values(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_report_source_t* source) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, source->dispatch_ptr, 2,
      IREE_SV("dispatch_ptr")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, source->workgroup_id_x, 1,
      IREE_SV("workgroup_id_x")));
  return loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, source->workitem_id_x, 1,
      IREE_SV("workitem_id_x"));
}

static iree_status_t loom_amdgpu_sanitizer_validate_race_report_values(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_race_report_t* report) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->check_kind, 1, IREE_SV("check_kind")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->flags, 1, IREE_SV("flags")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->memory_space, 1,
      IREE_SV("memory_space")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->current_access_kind, 1,
      IREE_SV("current_access_kind")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->prior_access_kind, 1,
      IREE_SV("prior_access_kind")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->access_size, 1, IREE_SV("access_size")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->current_site_id, 2,
      IREE_SV("current_site_id")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->prior_site_id, 2,
      IREE_SV("prior_site_id")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->memory_address, 2,
      IREE_SV("memory_address")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->shadow_address, 2,
      IREE_SV("shadow_address")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->shadow_value, 2,
      IREE_SV("shadow_value")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->current_workgroup_id_x, 1,
      IREE_SV("current_workgroup_id_x")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->current_workgroup_id_y, 1,
      IREE_SV("current_workgroup_id_y")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->current_workgroup_id_z, 1,
      IREE_SV("current_workgroup_id_z")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->current_workitem_id_x, 1,
      IREE_SV("current_workitem_id_x")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->current_workitem_id_y, 1,
      IREE_SV("current_workitem_id_y")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->current_workitem_id_z, 1,
      IREE_SV("current_workitem_id_z")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->prior_workgroup_id_x, 1,
      IREE_SV("prior_workgroup_id_x")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->prior_workgroup_id_y, 1,
      IREE_SV("prior_workgroup_id_y")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->prior_workgroup_id_z, 1,
      IREE_SV("prior_workgroup_id_z")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->prior_workitem_id_x, 1,
      IREE_SV("prior_workitem_id_x")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->prior_workitem_id_y, 1,
      IREE_SV("prior_workitem_id_y")));
  return loom_amdgpu_sanitizer_race_require_data_register(
      builder, descriptor_set, report->prior_workitem_id_z, 1,
      IREE_SV("prior_workitem_id_z"));
}

iree_status_t loom_amdgpu_build_sanitizer_race_report_payload(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    const loom_amdgpu_sanitizer_race_report_t* report,
    loom_location_id_t location) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_validate_race_report_values(
      builder, descriptor_set, report));

  const uint32_t payload_base = LOOM_AMDGPU_FEEDBACK_PACKET_BYTE_LENGTH;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_u32_constant(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_RECORD_LENGTH_OFFSET,
      LOOM_AMDGPU_TSAN_REPORT_BYTE_LENGTH, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_u32_constant(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_ABI_VERSION_OFFSET,
      LOOM_AMDGPU_TSAN_REPORT_ABI_VERSION, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CHECK_KIND_OFFSET,
      report->check_kind, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_FLAGS_OFFSET, report->flags,
      location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_MEMORY_SPACE_OFFSET,
      report->memory_space, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CURRENT_ACCESS_KIND_OFFSET,
      report->current_access_kind, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_ACCESS_KIND_OFFSET,
      report->prior_access_kind, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_ACCESS_SIZE_OFFSET,
      report->access_size, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b64(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CURRENT_SITE_ID_OFFSET,
      report->current_site_id, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b64(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_SITE_ID_OFFSET,
      report->prior_site_id, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b64(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_MEMORY_ADDRESS_OFFSET,
      report->memory_address, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b64(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_SHADOW_ADDRESS_OFFSET,
      report->shadow_address, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b64(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_SHADOW_VALUE_OFFSET,
      report->shadow_value, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKGROUP_ID_X_OFFSET,
      report->current_workgroup_id_x, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKGROUP_ID_Y_OFFSET,
      report->current_workgroup_id_y, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKGROUP_ID_Z_OFFSET,
      report->current_workgroup_id_z, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKITEM_ID_X_OFFSET,
      report->current_workitem_id_x, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKITEM_ID_Y_OFFSET,
      report->current_workitem_id_y, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKITEM_ID_Z_OFFSET,
      report->current_workitem_id_z, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKGROUP_ID_X_OFFSET,
      report->prior_workgroup_id_x, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKGROUP_ID_Y_OFFSET,
      report->prior_workgroup_id_y, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKGROUP_ID_Z_OFFSET,
      report->prior_workgroup_id_z, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKITEM_ID_X_OFFSET,
      report->prior_workitem_id_x, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKITEM_ID_Y_OFFSET,
      report->prior_workitem_id_y, location));
  return loom_amdgpu_build_feedback_packet_store_b32(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKITEM_ID_Z_OFFSET,
      report->prior_workitem_id_z, location);
}

static iree_status_t
loom_amdgpu_sanitizer_build_race_report_terminate_from_current_block(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t feedback_config_symbol,
    const loom_amdgpu_sanitizer_report_source_t* source,
    const loom_amdgpu_sanitizer_race_report_t* report,
    loom_location_id_t location, loom_block_t** out_terminal_block) {
  if (out_terminal_block != NULL) {
    *out_terminal_block = NULL;
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_validate_race_report_source_values(
      builder, descriptor_set, source));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_validate_race_report_values(
      builder, descriptor_set, report));
  if (builder->ip.before_op != NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU sanitizer race report producer must be "
                            "built at the end of a low block");
  }

  loom_block_t* config_block = builder->ip.block;
  loom_block_t* feedback_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_insert_block_after(
      builder, config_block, &feedback_block));
  loom_block_t* terminal_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_insert_block_after(
      builder, feedback_block, &terminal_block));
  if (out_terminal_block != NULL) {
    *out_terminal_block = terminal_block;
  }

  loom_amdgpu_feedback_config_values_t config_values = {};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_config_values(
      builder, descriptor_set, feedback_config_symbol, location,
      &config_values));
  loom_value_id_t feedback_enabled_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_config_enabled_scc(
      builder, descriptor_set, config_values.flags, location,
      &feedback_enabled_scc));
  loom_op_t* enabled_branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(builder, feedback_enabled_scc,
                                              feedback_block, terminal_block,
                                              location, &enabled_branch_op));

  loom_builder_set_block(builder, feedback_block);
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_channel_header_values(
      builder, descriptor_set, config_values.channel_base, location,
      &channel_values));
  const uint32_t packet_length = (uint32_t)loom_amdgpu_feedback_packet_length(
      LOOM_AMDGPU_TSAN_REPORT_BYTE_LENGTH);
  loom_amdgpu_feedback_reservation_t reservation = {};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_reservation(
      builder, descriptor_set, channel_values.address, channel_values.ring_base,
      channel_values.ring_capacity, packet_length, location, &reservation));

  loom_block_t* continuation_block = builder->ip.block;
  loom_block_t* report_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_insert_block_after(
      builder, continuation_block, &report_block));
  loom_value_id_t reservation_succeeded_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_reservation_succeeded_scc(
      builder, descriptor_set, reservation.reserved_mask, location,
      &reservation_succeeded_scc));
  loom_op_t* reserved_branch_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_cond_br_build(builder, reservation_succeeded_scc, report_block,
                             terminal_block, location, &reserved_branch_op));

  loom_builder_set_block(builder, report_block);
  loom_value_id_t saved_report_exec = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_exec_narrow(
      builder, descriptor_set, reservation.reserved_mask, location,
      &saved_report_exec));
  const loom_amdgpu_feedback_packet_header_t header = {
      .record_length = packet_length,
      .kind = LOOM_AMDGPU_FEEDBACK_PACKET_KIND_TSAN,
      .flags = LOOM_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC,
      .sequence = reservation.sequence,
      .source_dispatch_ptr = source->dispatch_ptr,
      .source_workgroup_id_x = source->workgroup_id_x,
      .source_workitem_id_x = source->workitem_id_x,
      .source_context = config_values.source_context,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_header(
      builder, descriptor_set, &reservation.packet_address, &header, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_race_report_payload(
      builder, descriptor_set, &reservation.packet_address, report, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_publish_packet(
      builder, descriptor_set, &reservation.packet_address,
      config_values.notify_signal, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_exec_restore(
      builder, descriptor_set, saved_report_exec, location));
  loom_op_t* report_branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_br_build(builder, terminal_block, /*args=*/NULL,
                                         /*args_count=*/0, location,
                                         &report_branch_op));

  loom_builder_set_block(builder, terminal_block);
  loom_op_t* return_op = NULL;
  return loom_low_return_build(builder, /*values=*/NULL, /*value_count=*/0,
                               location, &return_op);
}

iree_status_t loom_amdgpu_build_sanitizer_race_report_terminate(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t feedback_config_symbol,
    const loom_amdgpu_sanitizer_report_source_t* source,
    const loom_amdgpu_sanitizer_race_report_t* report,
    loom_location_id_t location) {
  return loom_amdgpu_sanitizer_build_race_report_terminate_from_current_block(
      builder, descriptor_set, feedback_config_symbol, source, report, location,
      /*out_terminal_block=*/NULL);
}

static iree_status_t loom_amdgpu_sanitizer_race_define_register_block_arg(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_block_t* block, uint16_t register_class, uint32_t unit_count,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, register_class, unit_count, &type));
  return loom_builder_define_block_arg(builder, block, type, out_value);
}

static iree_status_t loom_amdgpu_sanitizer_define_race_report_island_args(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_block_t* entry_block,
    loom_amdgpu_sanitizer_race_report_island_t* island) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2,
      &island->source_args.dispatch_ptr));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1,
      &island->source_args.workgroup_id_x));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->source_args.workitem_id_x));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.check_kind));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.flags));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.memory_space));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.current_access_kind));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.prior_access_kind));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.access_size));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2,
      &island->report_args.current_site_id));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2,
      &island->report_args.prior_site_id));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2,
      &island->report_args.memory_address));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2,
      &island->report_args.shadow_address));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2,
      &island->report_args.shadow_value));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.current_workgroup_id_x));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.current_workgroup_id_y));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.current_workgroup_id_z));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.current_workitem_id_x));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.current_workitem_id_y));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.current_workitem_id_z));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.prior_workgroup_id_x));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.prior_workgroup_id_y));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.prior_workgroup_id_z));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.prior_workitem_id_x));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.prior_workitem_id_y));
  return loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->report_args.prior_workitem_id_z);
}

iree_status_t loom_amdgpu_build_sanitizer_race_report_island(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_block_t* after_block, loom_symbol_ref_t feedback_config_symbol,
    loom_location_id_t location,
    loom_amdgpu_sanitizer_race_report_island_t* out_island) {
  IREE_ASSERT_ARGUMENT(out_island);
  *out_island = (loom_amdgpu_sanitizer_race_report_island_t){0};
  if (after_block->parent_region == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU sanitizer race report island requires a low region block");
  }

  loom_amdgpu_sanitizer_race_report_island_t island = {0};
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, after_block->parent_region,
      (uint16_t)(after_block->region_index + 1), &island.entry_block));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_define_race_report_island_args(
      builder, descriptor_set, island.entry_block, &island));
  loom_builder_set_block(builder, island.entry_block);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_build_race_report_terminate_from_current_block(
          builder, descriptor_set, feedback_config_symbol, &island.source_args,
          &island.report_args, location, &island.terminal_block));

  *out_island = island;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_sanitizer_race_report_branch(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_race_report_island_t* island,
    const loom_amdgpu_sanitizer_report_source_t* source,
    const loom_amdgpu_sanitizer_race_report_t* report,
    loom_location_id_t location) {
  if (builder->ip.before_op != NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU sanitizer race report branch must be built "
                            "at the end of a low block");
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_register_class(
      builder, descriptor_set, source->dispatch_ptr, 2,
      LOOM_AMDGPU_REG_CLASS_ID_SGPR, IREE_SV("dispatch_ptr")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_register_class(
      builder, descriptor_set, source->workgroup_id_x, 1,
      LOOM_AMDGPU_REG_CLASS_ID_SGPR, IREE_SV("workgroup_id_x")));

  loom_value_id_t args[26] = {0};
  args[0] = source->dispatch_ptr;
  args[1] = source->workgroup_id_x;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, source->workitem_id_x, 1, location, &args[2]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->check_kind, 1, location, &args[3]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->flags, 1, location, &args[4]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->memory_space, 1, location, &args[5]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->current_access_kind, 1, location,
      &args[6]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->prior_access_kind, 1, location,
      &args[7]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->access_size, 1, location, &args[8]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->current_site_id, 2, location, &args[9]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->prior_site_id, 2, location, &args[10]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->memory_address, 2, location, &args[11]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->shadow_address, 2, location, &args[12]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->shadow_value, 2, location, &args[13]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->current_workgroup_id_x, 1, location,
      &args[14]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->current_workgroup_id_y, 1, location,
      &args[15]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->current_workgroup_id_z, 1, location,
      &args[16]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->current_workitem_id_x, 1, location,
      &args[17]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->current_workitem_id_y, 1, location,
      &args[18]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->current_workitem_id_z, 1, location,
      &args[19]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->prior_workgroup_id_x, 1, location,
      &args[20]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->prior_workgroup_id_y, 1, location,
      &args[21]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->prior_workgroup_id_z, 1, location,
      &args[22]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->prior_workitem_id_x, 1, location,
      &args[23]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->prior_workitem_id_y, 1, location,
      &args[24]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->prior_workitem_id_z, 1, location,
      &args[25]));

  loom_op_t* branch_op = NULL;
  return loom_low_br_build(builder, island->entry_block, args,
                           IREE_ARRAYSIZE(args), location, &branch_op);
}

static iree_status_t
loom_amdgpu_sanitizer_race_split_current_block_on_failure_scc(
    loom_builder_t* builder, loom_value_id_t failure_scc,
    loom_location_id_t location,
    loom_amdgpu_sanitizer_race_report_failure_branch_t* out_branch) {
  *out_branch = (loom_amdgpu_sanitizer_race_report_failure_branch_t){0};
  if (builder->ip.before_op != NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU sanitizer race failure branch must be "
                            "built at the end of a low block");
  }

  loom_block_t* hot_block = builder->ip.block;
  loom_amdgpu_sanitizer_race_report_failure_branch_t branch = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_insert_block_after(
      builder, hot_block, &branch.continuation_block));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_insert_block_after(
      builder, branch.continuation_block, &branch.failure_block));
  loom_op_t* cond_branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(
      builder, failure_scc, branch.failure_block, branch.continuation_block,
      location, &cond_branch_op));

  *out_branch = branch;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_sanitizer_race_report_failure_mask_branch(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_race_report_island_t* island,
    loom_value_id_t failure_mask,
    const loom_amdgpu_sanitizer_report_source_t* source,
    const loom_amdgpu_sanitizer_race_report_t* report,
    loom_location_id_t location,
    loom_amdgpu_sanitizer_race_report_failure_branch_t* out_branch) {
  IREE_ASSERT_ARGUMENT(out_branch);
  *out_branch = (loom_amdgpu_sanitizer_race_report_failure_branch_t){0};
  loom_amdgpu_sanitizer_race_report_failure_branch_t branch = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_build_sanitizer_race_report_failure_mask_split(
          builder, descriptor_set, failure_mask, location, &branch));

  IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_race_report_branch(
      builder, descriptor_set, island, source, report, location));

  loom_builder_set_block(builder, branch.continuation_block);
  *out_branch = branch;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_sanitizer_race_report_failure_mask_split(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t failure_mask, loom_location_id_t location,
    loom_amdgpu_sanitizer_race_report_failure_branch_t* out_branch) {
  IREE_ASSERT_ARGUMENT(out_branch);
  *out_branch = (loom_amdgpu_sanitizer_race_report_failure_branch_t){0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_require_register_class(
      builder, descriptor_set, failure_mask, 2, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
      IREE_SV("failure_mask")));
  loom_value_id_t failure_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_sgpr64_nonzero_scc(
      builder, descriptor_set, failure_mask, location, &failure_scc));

  loom_amdgpu_sanitizer_race_report_failure_branch_t branch = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_split_current_block_on_failure_scc(
          builder, failure_scc, location, &branch));

  loom_builder_set_block(builder, branch.failure_block);
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_exec_narrow(
      builder, descriptor_set, failure_mask, location,
      /*out_saved_exec=*/NULL));

  *out_branch = branch;
  return iree_ok_status();
}
