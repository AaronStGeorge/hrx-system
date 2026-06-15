// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/sanitizer_report.h"

#include "loom/codegen/low/builder.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/feedback_abi.h"
#include "loom/target/arch/amdgpu/lower/control_packet.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/feedback.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"

static iree_status_t loom_amdgpu_sanitizer_insert_block_after(
    loom_builder_t* builder, loom_block_t* after_block,
    loom_block_t** out_block) {
  *out_block = NULL;
  if (after_block->parent_region == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU sanitizer report requires a low region block");
  }
  return loom_region_insert_block(builder->module, after_block->parent_region,
                                  (uint16_t)(after_block->region_index + 1),
                                  out_block);
}

static bool loom_amdgpu_sanitizer_access_kind_is_valid(
    loom_amdgpu_sanitizer_access_kind_t access_kind) {
  switch (access_kind) {
    case LOOM_AMDGPU_SANITIZER_ACCESS_KIND_UNKNOWN:
    case LOOM_AMDGPU_SANITIZER_ACCESS_KIND_READ:
    case LOOM_AMDGPU_SANITIZER_ACCESS_KIND_WRITE:
    case LOOM_AMDGPU_SANITIZER_ACCESS_KIND_ATOMIC:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_amdgpu_sanitizer_require_data_register(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t value, uint32_t unit_count, iree_string_view_t value_name) {
  if (value >= builder->module->values.count) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU sanitizer report value `%.*s` is an invalid low value",
        (int)value_name.size, value_name.data);
  }
  const loom_type_t type = loom_module_value_type(builder->module, value);
  if (!loom_low_type_is_register(type) ||
      loom_low_register_type_descriptor_set_stable_id(type) !=
          descriptor_set->stable_id ||
      loom_low_register_type_unit_count(type) != unit_count) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU sanitizer report value `%.*s` has an unsupported register "
        "shape",
        (int)value_name.size, value_name.data);
  }
  const uint16_t register_class = loom_low_register_type_class_id(type);
  if (register_class != LOOM_AMDGPU_REG_CLASS_ID_SGPR &&
      register_class != LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU sanitizer report value `%.*s` must be an SGPR or VGPR",
        (int)value_name.size, value_name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_validate_access_report(
    const loom_amdgpu_sanitizer_access_report_t* report,
    iree_string_view_t operation_name) {
  if (!loom_amdgpu_sanitizer_access_kind_is_valid(report->access_kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU sanitizer %.*s access kind is invalid",
                            (int)operation_name.size, operation_name.data);
  }
  if (report->flags != LOOM_AMDGPU_SANITIZER_REPORT_FLAG_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU sanitizer %.*s flags are invalid",
                            (int)operation_name.size, operation_name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_validate_access_report_values(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_access_report_t* report) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_require_data_register(
      builder, descriptor_set, report->fault_address, 2,
      IREE_SV("fault_address")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_require_data_register(
      builder, descriptor_set, report->access_size, 2, IREE_SV("access_size")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_require_data_register(
      builder, descriptor_set, report->site_id, 2, IREE_SV("site_id")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_require_data_register(
      builder, descriptor_set, report->shadow_address, 2,
      IREE_SV("shadow_address")));
  return loom_amdgpu_sanitizer_require_data_register(builder, descriptor_set,
                                                     report->shadow_value, 2,
                                                     IREE_SV("shadow_value"));
}

static iree_status_t loom_amdgpu_sanitizer_validate_source_values(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_report_source_t* source) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_require_data_register(
      builder, descriptor_set, source->dispatch_ptr, 2,
      IREE_SV("dispatch_ptr")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_require_data_register(
      builder, descriptor_set, source->workgroup_id_x, 1,
      IREE_SV("workgroup_id_x")));
  return loom_amdgpu_sanitizer_require_data_register(builder, descriptor_set,
                                                     source->workitem_id_x, 1,
                                                     IREE_SV("workitem_id_x"));
}

iree_status_t loom_amdgpu_build_sanitizer_access_report_payload(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    const loom_amdgpu_sanitizer_access_report_t* report,
    loom_location_id_t location) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_validate_access_report(
      report, IREE_SV("access report payload")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_validate_access_report_values(
      builder, descriptor_set, report));
  const uint32_t payload_base = LOOM_AMDGPU_FEEDBACK_PACKET_BYTE_LENGTH;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_u32_constant(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_RECORD_LENGTH_OFFSET,
      LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_BYTE_LENGTH, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_u32_constant(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_ABI_VERSION_OFFSET,
      LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_ABI_VERSION, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_u32_constant(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_ACCESS_KIND_OFFSET,
      report->access_kind, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_u32_constant(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_FLAGS_OFFSET,
      report->flags, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b64(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_FAULT_ADDRESS_OFFSET,
      report->fault_address, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b64(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_ACCESS_SIZE_OFFSET,
      report->access_size, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b64(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_SITE_ID_OFFSET,
      report->site_id, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b64(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_SHADOW_ADDRESS_OFFSET,
      report->shadow_address, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b64(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_SHADOW_VALUE_OFFSET,
      report->shadow_value, location));
  return loom_amdgpu_build_feedback_packet_store_u64_constant(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_RESERVED0_OFFSET, 0,
      location);
}

static iree_status_t
loom_amdgpu_sanitizer_build_access_report_trap_from_current_block(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t feedback_config_symbol,
    const loom_amdgpu_sanitizer_report_source_t* source,
    const loom_amdgpu_sanitizer_access_report_t* report,
    loom_location_id_t location, loom_block_t** out_trap_block) {
  if (out_trap_block != NULL) {
    *out_trap_block = NULL;
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_validate_access_report(
      report, IREE_SV("access report trap")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_validate_source_values(
      builder, descriptor_set, source));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_validate_access_report_values(
      builder, descriptor_set, report));
  if (builder->ip.before_op != NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU sanitizer report trap must be built at the "
                            "end of a low block");
  }

  loom_block_t* config_block = builder->ip.block;
  loom_block_t* feedback_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_insert_block_after(
      builder, config_block, &feedback_block));
  loom_block_t* trap_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_insert_block_after(
      builder, feedback_block, &trap_block));
  if (out_trap_block != NULL) {
    *out_trap_block = trap_block;
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
                                              feedback_block, trap_block,
                                              location, &enabled_branch_op));

  loom_builder_set_block(builder, feedback_block);
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_channel_header_values(
      builder, descriptor_set, config_values.channel_base, location,
      &channel_values));
  const uint32_t packet_length = (uint32_t)loom_amdgpu_feedback_packet_length(
      LOOM_AMDGPU_SANITIZER_ACCESS_REPORT_BYTE_LENGTH);
  loom_amdgpu_feedback_reservation_t reservation = {};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_reservation(
      builder, descriptor_set, channel_values.address, channel_values.ring_base,
      channel_values.ring_capacity, packet_length, location, &reservation));

  loom_block_t* continuation_block = builder->ip.block;
  loom_block_t* report_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_insert_block_after(
      builder, continuation_block, &report_block));
  loom_value_id_t reservation_succeeded_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_reservation_succeeded_scc(
      builder, descriptor_set, reservation.reserved_mask, location,
      &reservation_succeeded_scc));
  loom_op_t* reserved_branch_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_cond_br_build(builder, reservation_succeeded_scc, report_block,
                             trap_block, location, &reserved_branch_op));

  loom_builder_set_block(builder, report_block);
  const loom_amdgpu_feedback_packet_header_t header = {
      .record_length = packet_length,
      .kind = LOOM_AMDGPU_FEEDBACK_PACKET_KIND_ASAN,
      .flags = LOOM_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC,
      .sequence = reservation.sequence,
      .source_dispatch_ptr = source->dispatch_ptr,
      .source_workgroup_id_x = source->workgroup_id_x,
      .source_workitem_id_x = source->workitem_id_x,
      .source_context = config_values.source_context,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_header(
      builder, descriptor_set, &reservation.packet_address, &header, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_access_report_payload(
      builder, descriptor_set, &reservation.packet_address, report, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_publish_packet(
      builder, descriptor_set, &reservation.packet_address,
      config_values.notify_signal, location));
  loom_op_t* report_branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_br_build(builder, trap_block, /*args=*/NULL,
                                         /*args_count=*/0, location,
                                         &report_branch_op));

  loom_builder_set_block(builder, trap_block);
  return loom_amdgpu_build_control_packet_trap(
      builder, descriptor_set, LOOM_AMDGPU_SANITIZER_TRAP_ID, location);
}

iree_status_t loom_amdgpu_build_sanitizer_access_report_trap(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t feedback_config_symbol,
    const loom_amdgpu_sanitizer_report_source_t* source,
    const loom_amdgpu_sanitizer_access_report_t* report,
    loom_location_id_t location) {
  return loom_amdgpu_sanitizer_build_access_report_trap_from_current_block(
      builder, descriptor_set, feedback_config_symbol, source, report, location,
      /*out_trap_block=*/NULL);
}

static iree_status_t loom_amdgpu_sanitizer_require_register_class(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t value, uint32_t unit_count, uint16_t register_class,
    iree_string_view_t value_name) {
  if (value >= builder->module->values.count) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU sanitizer report value `%.*s` is an invalid low value",
        (int)value_name.size, value_name.data);
  }
  const loom_type_t type = loom_module_value_type(builder->module, value);
  if (!loom_low_type_is_register(type) ||
      loom_low_register_type_descriptor_set_stable_id(type) !=
          descriptor_set->stable_id ||
      loom_low_register_type_unit_count(type) != unit_count) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU sanitizer report value `%.*s` has an unsupported register "
        "shape",
        (int)value_name.size, value_name.data);
  }
  if (loom_low_register_type_class_id(type) != register_class) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU sanitizer report value `%.*s` has an unsupported register "
        "class",
        (int)value_name.size, value_name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_build_u32_attr(
    loom_builder_t* builder, iree_string_view_t name, uint32_t value,
    loom_named_attr_t* out_attr) {
  *out_attr = (loom_named_attr_t){0};
  IREE_RETURN_IF_ERROR(
      loom_builder_intern_string(builder, name, &out_attr->name_id));
  out_attr->value = loom_attr_i64(value);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_build_descriptor_op(
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

static iree_status_t loom_amdgpu_sanitizer_build_const_u32(
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_build_u32_attr(
      builder, IREE_SV("imm32"), value, &imm32_attr));
  loom_op_t* const_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
      builder, descriptor_set, descriptor, opcode_id,
      loom_make_named_attr_slice(&imm32_attr, 1), result_type, location,
      &const_op));
  *out_value = loom_low_const_result(const_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_build_sgpr_u32_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t value, loom_location_id_t location, loom_value_id_t* out_value) {
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  return loom_amdgpu_sanitizer_build_const_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, value,
      sgpr_type, location, out_value);
}

static iree_status_t loom_amdgpu_sanitizer_build_sgpr_u64_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint64_t value, loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_build_sgpr_u32_const(
      builder, descriptor_set, (uint32_t)value, location, &low_value_lo));
  loom_value_id_t low_value_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_build_sgpr_u32_const(
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

static iree_status_t loom_amdgpu_sanitizer_build_sgpr64_nonzero_scc(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t value, loom_location_id_t location,
    loom_value_id_t* out_scc) {
  *out_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_require_register_class(
      builder, descriptor_set, value, 2, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
      IREE_SV("value")));
  loom_value_id_t zero64 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_build_sgpr_u64_const(
      builder, descriptor_set, 0, location, &zero64));

  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SCC, 1, &scc_type));
  const loom_value_id_t operands[] = {value, zero64};
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LG_U64,
      operands, IREE_ARRAYSIZE(operands), &scc_type, /*result_count=*/1,
      location, &compare_op));
  *out_scc = loom_value_slice_get(loom_low_op_results(compare_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_build_exec_narrow(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t lane_mask, loom_location_id_t location) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_require_register_class(
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
  return loom_amdgpu_sanitizer_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64,
      &lane_mask, /*operand_count=*/1, result_types,
      IREE_ARRAYSIZE(result_types), location, &saveexec_op);
}

static iree_status_t loom_amdgpu_sanitizer_define_register_block_arg(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_block_t* block, uint16_t register_class, uint32_t unit_count,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, register_class, unit_count, &type));
  return loom_builder_define_block_arg(builder, block, type, out_value);
}

static iree_status_t loom_amdgpu_sanitizer_define_access_report_island_args(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_block_t* entry_block, loom_amdgpu_sanitizer_access_kind_t access_kind,
    loom_amdgpu_sanitizer_report_flags_t flags,
    loom_amdgpu_sanitizer_access_report_trap_island_t* island) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2,
      &island->source_args.dispatch_ptr));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1,
      &island->source_args.workgroup_id_x));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
      &island->source_args.workitem_id_x));
  island->report_args.access_kind = access_kind;
  island->report_args.flags = flags;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2,
      &island->report_args.fault_address));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2,
      &island->report_args.access_size));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2,
      &island->report_args.site_id));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2,
      &island->report_args.shadow_address));
  return loom_amdgpu_sanitizer_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2,
      &island->report_args.shadow_value);
}

iree_status_t loom_amdgpu_build_sanitizer_access_report_trap_island(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_block_t* after_block, loom_symbol_ref_t feedback_config_symbol,
    loom_amdgpu_sanitizer_access_kind_t access_kind,
    loom_amdgpu_sanitizer_report_flags_t flags, loom_location_id_t location,
    loom_amdgpu_sanitizer_access_report_trap_island_t* out_island) {
  IREE_ASSERT_ARGUMENT(out_island);
  *out_island = (loom_amdgpu_sanitizer_access_report_trap_island_t){0};
  const loom_amdgpu_sanitizer_access_report_t access_report = {
      .access_kind = access_kind,
      .flags = flags,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_validate_access_report(
      &access_report, IREE_SV("access report trap island")));
  if (after_block->parent_region == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU sanitizer report island requires a low region block");
  }

  loom_amdgpu_sanitizer_access_report_trap_island_t island = {
      .access_kind = access_kind,
      .flags = flags,
  };
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, after_block->parent_region,
      (uint16_t)(after_block->region_index + 1), &island.entry_block));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_define_access_report_island_args(
      builder, descriptor_set, island.entry_block, access_kind, flags,
      &island));
  loom_builder_set_block(builder, island.entry_block);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_build_access_report_trap_from_current_block(
          builder, descriptor_set, feedback_config_symbol, &island.source_args,
          &island.report_args, location, &island.trap_block));

  *out_island = island;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_validate_access_report_for_island(
    const loom_amdgpu_sanitizer_access_report_trap_island_t* island,
    const loom_amdgpu_sanitizer_access_report_t* report,
    iree_string_view_t operation_name) {
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_validate_access_report(report, operation_name));
  if (report->access_kind != island->access_kind ||
      report->flags != island->flags) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU sanitizer access report does not match its island");
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_sanitizer_access_report_trap_branch(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_access_report_trap_island_t* island,
    const loom_amdgpu_sanitizer_report_source_t* source,
    const loom_amdgpu_sanitizer_access_report_t* report,
    loom_location_id_t location) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_validate_access_report_for_island(
      island, report, IREE_SV("access report trap branch")));
  if (builder->ip.before_op != NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU sanitizer report branch must be built at "
                            "the end of a low block");
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_require_register_class(
      builder, descriptor_set, source->dispatch_ptr, 2,
      LOOM_AMDGPU_REG_CLASS_ID_SGPR, IREE_SV("dispatch_ptr")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_require_register_class(
      builder, descriptor_set, source->workgroup_id_x, 1,
      LOOM_AMDGPU_REG_CLASS_ID_SGPR, IREE_SV("workgroup_id_x")));

  loom_value_id_t args[8] = {0};
  args[0] = source->dispatch_ptr;
  args[1] = source->workgroup_id_x;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, source->workitem_id_x, 1, location, &args[2]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->fault_address, 2, location, &args[3]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->access_size, 2, location, &args[4]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->site_id, 2, location, &args[5]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->shadow_address, 2, location, &args[6]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, report->shadow_value, 2, location, &args[7]));
  loom_op_t* branch_op = NULL;
  return loom_low_br_build(builder, island->entry_block, args,
                           IREE_ARRAYSIZE(args), location, &branch_op);
}

static iree_status_t loom_amdgpu_sanitizer_split_current_block_on_failure_scc(
    loom_builder_t* builder, loom_value_id_t failure_scc,
    loom_location_id_t location,
    loom_amdgpu_sanitizer_access_report_failure_branch_t* out_branch) {
  *out_branch = (loom_amdgpu_sanitizer_access_report_failure_branch_t){0};
  if (builder->ip.before_op != NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU sanitizer failure branch must be built at "
                            "the end of a low block");
  }

  loom_block_t* hot_block = builder->ip.block;
  loom_amdgpu_sanitizer_access_report_failure_branch_t branch = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_insert_block_after(
      builder, hot_block, &branch.continuation_block));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_insert_block_after(
      builder, branch.continuation_block, &branch.failure_block));
  loom_op_t* cond_branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(
      builder, failure_scc, branch.failure_block, branch.continuation_block,
      location, &cond_branch_op));

  *out_branch = branch;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_sanitizer_access_report_failure_branch(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_access_report_trap_island_t* island,
    loom_value_id_t failure_scc,
    const loom_amdgpu_sanitizer_report_source_t* source,
    const loom_amdgpu_sanitizer_access_report_t* report,
    loom_location_id_t location,
    loom_amdgpu_sanitizer_access_report_failure_branch_t* out_branch) {
  IREE_ASSERT_ARGUMENT(out_branch);
  *out_branch = (loom_amdgpu_sanitizer_access_report_failure_branch_t){0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_validate_access_report_for_island(
      island, report, IREE_SV("access report failure branch")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_require_register_class(
      builder, descriptor_set, failure_scc, 1, LOOM_AMDGPU_REG_CLASS_ID_SCC,
      IREE_SV("failure_scc")));
  loom_amdgpu_sanitizer_access_report_failure_branch_t branch = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_split_current_block_on_failure_scc(
      builder, failure_scc, location, &branch));

  loom_builder_set_block(builder, branch.failure_block);
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_access_report_trap_branch(
      builder, descriptor_set, island, source, report, location));

  loom_builder_set_block(builder, branch.continuation_block);
  *out_branch = branch;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_sanitizer_access_report_failure_mask_branch(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_access_report_trap_island_t* island,
    loom_value_id_t failure_mask,
    const loom_amdgpu_sanitizer_report_source_t* source,
    const loom_amdgpu_sanitizer_access_report_t* report,
    loom_location_id_t location,
    loom_amdgpu_sanitizer_access_report_failure_branch_t* out_branch) {
  IREE_ASSERT_ARGUMENT(out_branch);
  *out_branch = (loom_amdgpu_sanitizer_access_report_failure_branch_t){0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_validate_access_report_for_island(
      island, report, IREE_SV("access report failure mask branch")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_require_register_class(
      builder, descriptor_set, failure_mask, 2, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
      IREE_SV("failure_mask")));
  loom_value_id_t failure_scc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_build_sgpr64_nonzero_scc(
      builder, descriptor_set, failure_mask, location, &failure_scc));

  loom_amdgpu_sanitizer_access_report_failure_branch_t branch = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_split_current_block_on_failure_scc(
      builder, failure_scc, location, &branch));

  loom_builder_set_block(builder, branch.failure_block);
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_build_exec_narrow(
      builder, descriptor_set, failure_mask, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_access_report_trap_branch(
      builder, descriptor_set, island, source, report, location));

  loom_builder_set_block(builder, branch.continuation_block);
  *out_branch = branch;
  return iree_ok_status();
}
