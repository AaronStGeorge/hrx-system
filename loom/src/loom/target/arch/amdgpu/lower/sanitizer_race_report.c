// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/sanitizer_race_report.h"

#include "loom/codegen/low/builder.h"
#include "loom/ir/module.h"
#include "loom/target/arch/amdgpu/abi/feedback.h"
#include "loom/target/arch/amdgpu/lower/feedback.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"

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
