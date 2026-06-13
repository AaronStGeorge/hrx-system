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

iree_status_t loom_amdgpu_build_sanitizer_access_report_trap(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t feedback_config_symbol,
    const loom_amdgpu_sanitizer_report_source_t* source,
    const loom_amdgpu_sanitizer_access_report_t* report,
    loom_location_id_t location) {
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
