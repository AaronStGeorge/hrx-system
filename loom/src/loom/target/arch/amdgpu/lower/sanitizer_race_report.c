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
#include "loom/target/arch/amdgpu/lower/feedback.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"

iree_status_t loom_amdgpu_build_sanitizer_race_shadow_capacity_report_payload(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    const loom_amdgpu_sanitizer_race_shadow_capacity_report_t* report,
    loom_location_id_t location) {
  const uint32_t payload_base = LOOM_AMDGPU_FEEDBACK_PACKET_BYTE_LENGTH;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_u32_constant(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_RECORD_LENGTH_OFFSET,
      LOOM_AMDGPU_TSAN_REPORT_BYTE_LENGTH, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_u32_constant(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_ABI_VERSION_OFFSET,
      LOOM_AMDGPU_TSAN_REPORT_ABI_VERSION, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_u32_constant(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CHECK_KIND_OFFSET,
      LOOM_AMDGPU_TSAN_CHECK_KIND_SHADOW_CAPACITY, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_u32_constant(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_FLAGS_OFFSET,
      LOOM_AMDGPU_TSAN_REPORT_FLAG_NONE, location));

  const struct {
    uint32_t offset;
    loom_value_id_t value;
  } u64_fields[] = {
      {LOOM_AMDGPU_TSAN_REPORT_CAPACITY_SHADOW_SIZE_OFFSET,
       report->shadow_size},
      {LOOM_AMDGPU_TSAN_REPORT_CAPACITY_REQUIRED_SHADOW_END_OFFSET,
       report->required_shadow_end},
      {LOOM_AMDGPU_TSAN_REPORT_CAPACITY_WORKGROUP_SHADOW_STRIDE_OFFSET,
       report->workgroup_shadow_stride},
      {LOOM_AMDGPU_TSAN_REPORT_CAPACITY_WORKGROUP_ORDINAL_OFFSET,
       report->workgroup_ordinal},
  };
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(u64_fields); ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b64(
        builder, descriptor_set, packet_address,
        payload_base + u64_fields[i].offset, u64_fields[i].value, location));
  }
  const struct {
    uint32_t offset;
    loom_value_id_t value;
  } u32_fields[] = {
      {LOOM_AMDGPU_TSAN_REPORT_CAPACITY_LOCAL_MEMORY_SIZE_OFFSET,
       report->workgroup_local_memory_size},
      {LOOM_AMDGPU_TSAN_REPORT_CAPACITY_SHADOW_CELL_SIZE_OFFSET,
       report->shadow_cell_size},
      {LOOM_AMDGPU_TSAN_REPORT_CAPACITY_SHADOW_HISTORY_COUNT_OFFSET,
       report->shadow_history_count},
      {LOOM_AMDGPU_TSAN_REPORT_CAPACITY_WORKGROUP_ID_X_OFFSET,
       report->current_workgroup_ids[0]},
      {LOOM_AMDGPU_TSAN_REPORT_CAPACITY_WORKGROUP_ID_Y_OFFSET,
       report->current_workgroup_ids[1]},
      {LOOM_AMDGPU_TSAN_REPORT_CAPACITY_WORKGROUP_ID_Z_OFFSET,
       report->current_workgroup_ids[2]},
      {LOOM_AMDGPU_TSAN_REPORT_CAPACITY_WORKITEM_ID_X_OFFSET,
       report->current_workitem_ids[0]},
      {LOOM_AMDGPU_TSAN_REPORT_CAPACITY_WORKITEM_ID_Y_OFFSET,
       report->current_workitem_ids[1]},
      {LOOM_AMDGPU_TSAN_REPORT_CAPACITY_WORKITEM_ID_Z_OFFSET,
       report->current_workitem_ids[2]},
  };
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(u32_fields); ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b32(
        builder, descriptor_set, packet_address,
        payload_base + u32_fields[i].offset, u32_fields[i].value, location));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_u32_constant(
      builder, descriptor_set, packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CAPACITY_RESERVED0_OFFSET, 0,
      location));
  for (uint32_t i = 0; i < 4; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_u64_constant(
        builder, descriptor_set, packet_address,
        payload_base + LOOM_AMDGPU_TSAN_REPORT_CAPACITY_RESERVED1_OFFSET +
            i * sizeof(uint64_t),
        0, location));
  }
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_sanitizer_build_shadow_capacity_report_payload_callback(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    const void* payload_context, loom_location_id_t location) {
  return loom_amdgpu_build_sanitizer_race_shadow_capacity_report_payload(
      builder, descriptor_set, packet_address,
      (const loom_amdgpu_sanitizer_race_shadow_capacity_report_t*)
          payload_context,
      location);
}

iree_status_t loom_amdgpu_build_sanitizer_race_shadow_capacity_report_terminate(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t feedback_config_symbol,
    const loom_amdgpu_feedback_packet_source_t* source,
    const loom_amdgpu_sanitizer_race_shadow_capacity_report_t* report,
    loom_location_id_t location) {
  const loom_amdgpu_feedback_packet_producer_t producer = {
      .payload_byte_length = LOOM_AMDGPU_TSAN_REPORT_BYTE_LENGTH,
      .packet_kind = LOOM_AMDGPU_FEEDBACK_PACKET_KIND_TSAN,
      .packet_flags = LOOM_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC,
      .source = source,
      .build_payload =
          loom_amdgpu_sanitizer_build_shadow_capacity_report_payload_callback,
      .payload_context = report,
  };
  return loom_amdgpu_build_feedback_packet_producer_terminate(
      builder, descriptor_set, feedback_config_symbol, &producer, location,
      /*out_terminal_block=*/NULL);
}

IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_amdgpu_sanitizer_race_concat_vgpr_words(loom_builder_t* builder,
                                             loom_type_t vgpr_x2_type,
                                             loom_value_id_t low_word,
                                             loom_value_id_t high_word,
                                             loom_location_id_t location,
                                             loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t words[] = {low_word, high_word};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, words, IREE_ARRAYSIZE(words), vgpr_x2_type,
                            location, &concat_op));
  *out_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_pack_data_race_report(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_race_report_t* report,
    loom_location_id_t location,
    loom_amdgpu_sanitizer_race_packed_report_t* out_report) {
  *out_report = (loom_amdgpu_sanitizer_race_packed_report_t){
      .check_kind = report->check_kind,
      .flags = report->flags,
  };
  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2, &vgpr_x2_type));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_concat_vgpr_words(
      builder, vgpr_x2_type, report->memory_space, report->current_access_kind,
      location, &out_report->detail_qwords[0]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_concat_vgpr_words(
      builder, vgpr_x2_type, report->prior_access_kind, report->access_size,
      location, &out_report->detail_qwords[1]));
  out_report->detail_qwords[2] = report->current_site_id;
  out_report->detail_qwords[3] = report->prior_site_id;
  out_report->detail_qwords[4] = report->memory_address;
  out_report->detail_qwords[5] = report->shadow_address;
  out_report->detail_qwords[6] = report->shadow_value;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_concat_vgpr_words(
      builder, vgpr_x2_type, report->current_workgroup_id_x,
      report->current_workgroup_id_y, location, &out_report->detail_qwords[7]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_concat_vgpr_words(
      builder, vgpr_x2_type, report->current_workgroup_id_z,
      report->current_workitem_id_x, location, &out_report->detail_qwords[8]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_concat_vgpr_words(
      builder, vgpr_x2_type, report->current_workitem_id_y,
      report->current_workitem_id_z, location, &out_report->detail_qwords[9]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_concat_vgpr_words(
      builder, vgpr_x2_type, report->prior_workgroup_id_x,
      report->prior_workgroup_id_y, location, &out_report->detail_qwords[10]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_concat_vgpr_words(
      builder, vgpr_x2_type, report->prior_workgroup_id_z,
      report->prior_workitem_id_x, location, &out_report->detail_qwords[11]));
  return loom_amdgpu_sanitizer_race_concat_vgpr_words(
      builder, vgpr_x2_type, report->prior_workitem_id_y,
      report->prior_workitem_id_z, location, &out_report->detail_qwords[12]);
}

static iree_status_t
loom_amdgpu_sanitizer_race_pack_synchronization_capacity_report(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_race_synchronization_capacity_report_t* report,
    loom_value_id_t check_kind, loom_value_id_t flags,
    loom_value_id_t zero_detail_qword, loom_location_id_t location,
    loom_amdgpu_sanitizer_race_packed_report_t* out_report) {
  *out_report = (loom_amdgpu_sanitizer_race_packed_report_t){
      .check_kind = check_kind,
      .flags = flags,
  };
  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2, &vgpr_x2_type));
  out_report->detail_qwords[0] = report->current_site_id;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_concat_vgpr_words(
      builder, vgpr_x2_type, report->subgroup_epoch,
      report->subgroup_epoch_capacity, location,
      &out_report->detail_qwords[1]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_concat_vgpr_words(
      builder, vgpr_x2_type, report->wavefront_size, report->subgroup_id,
      location, &out_report->detail_qwords[2]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_concat_vgpr_words(
      builder, vgpr_x2_type, report->current_workgroup_ids[0],
      report->current_workgroup_ids[1], location,
      &out_report->detail_qwords[3]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_concat_vgpr_words(
      builder, vgpr_x2_type, report->current_workgroup_ids[2],
      report->current_workitem_linear_id, location,
      &out_report->detail_qwords[4]));
  out_report->detail_qwords[5] = report->subgroup_clock_address;
  for (uint32_t i = 6; i < LOOM_AMDGPU_SANITIZER_RACE_REPORT_DETAIL_QWORD_COUNT;
       ++i) {
    out_report->detail_qwords[i] = zero_detail_qword;
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_sanitizer_race_packed_report_payload(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    const loom_amdgpu_sanitizer_race_packed_report_t* report,
    loom_location_id_t location) {
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
  for (uint32_t i = 0; i < LOOM_AMDGPU_SANITIZER_RACE_REPORT_DETAIL_QWORD_COUNT;
       ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_packet_store_b64(
        builder, descriptor_set, packet_address,
        payload_base + LOOM_AMDGPU_TSAN_REPORT_MEMORY_SPACE_OFFSET +
            i * sizeof(uint64_t),
        report->detail_qwords[i], location));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_build_packed_report_payload_callback(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    const void* payload_context, loom_location_id_t location) {
  return loom_amdgpu_build_sanitizer_race_packed_report_payload(
      builder, descriptor_set, packet_address,
      (const loom_amdgpu_sanitizer_race_packed_report_t*)payload_context,
      location);
}

static iree_status_t
loom_amdgpu_sanitizer_build_packed_report_terminate_from_current_block(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t feedback_config_symbol,
    const loom_amdgpu_feedback_packet_source_t* source,
    const loom_amdgpu_sanitizer_race_packed_report_t* report,
    loom_location_id_t location, loom_block_t** out_terminal_block) {
  const loom_amdgpu_feedback_packet_producer_t producer = {
      .payload_byte_length = LOOM_AMDGPU_TSAN_REPORT_BYTE_LENGTH,
      .packet_kind = LOOM_AMDGPU_FEEDBACK_PACKET_KIND_TSAN,
      .packet_flags = LOOM_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC,
      .source = source,
      .build_payload =
          loom_amdgpu_sanitizer_build_packed_report_payload_callback,
      .payload_context = report,
  };
  return loom_amdgpu_build_feedback_packet_producer_terminate(
      builder, descriptor_set, feedback_config_symbol, &producer, location,
      out_terminal_block);
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

static iree_status_t loom_amdgpu_sanitizer_define_report_island_args(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_block_t* entry_block,
    loom_amdgpu_sanitizer_race_report_island_t* island) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2,
      &island->source_args.dispatch_ptr));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
      builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
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
  for (uint32_t i = 0; i < LOOM_AMDGPU_SANITIZER_RACE_REPORT_DETAIL_QWORD_COUNT;
       ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_define_register_block_arg(
        builder, descriptor_set, entry_block, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2,
        &island->report_args.detail_qwords[i]));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_sanitizer_race_report_island(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_block_t* after_block, loom_symbol_ref_t feedback_config_symbol,
    loom_location_id_t location,
    loom_amdgpu_sanitizer_race_report_island_t* out_island) {
  IREE_ASSERT_ARGUMENT(out_island);
  *out_island = (loom_amdgpu_sanitizer_race_report_island_t){0};
  IREE_ASSERT(after_block->parent_region != NULL,
              "AMDGPU sanitizer race report island requires a low region "
              "block");

  loom_amdgpu_sanitizer_race_report_island_t island = {0};
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      builder->module, after_block->parent_region,
      (uint16_t)(after_block->region_index + 1), &island.entry_block));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_define_report_island_args(
      builder, descriptor_set, island.entry_block, &island));
  loom_builder_set_block(builder, island.entry_block);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_build_packed_report_terminate_from_current_block(
          builder, descriptor_set, feedback_config_symbol, &island.source_args,
          &island.report_args, location, &island.terminal_block));

  *out_island = island;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_sanitizer_race_report_branch(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_race_report_island_t* island,
    const loom_amdgpu_feedback_packet_source_t* source,
    const loom_amdgpu_sanitizer_race_report_t* report,
    loom_location_id_t location) {
  loom_amdgpu_sanitizer_race_packed_report_t packed_report = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_pack_data_race_report(
      builder, descriptor_set, report, location, &packed_report));
  loom_value_id_t args[5 +
                       LOOM_AMDGPU_SANITIZER_RACE_REPORT_DETAIL_QWORD_COUNT] = {
      source->dispatch_ptr,     source->workgroup_id_x, source->workitem_id_x,
      packed_report.check_kind, packed_report.flags,
  };
  for (uint32_t i = 0; i < LOOM_AMDGPU_SANITIZER_RACE_REPORT_DETAIL_QWORD_COUNT;
       ++i) {
    args[5 + i] = packed_report.detail_qwords[i];
  }
  loom_op_t* branch_op = NULL;
  return loom_low_br_build(builder, island->entry_block, args,
                           IREE_ARRAYSIZE(args), location, &branch_op);
}

iree_status_t
loom_amdgpu_build_sanitizer_race_synchronization_capacity_report_branch(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_race_report_island_t* island,
    const loom_amdgpu_feedback_packet_source_t* source,
    const loom_amdgpu_sanitizer_race_synchronization_capacity_report_t* report,
    loom_value_id_t check_kind, loom_value_id_t flags,
    loom_value_id_t zero_detail_qword, loom_location_id_t location) {
  loom_amdgpu_sanitizer_race_packed_report_t packed_report = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_pack_synchronization_capacity_report(
          builder, descriptor_set, report, check_kind, flags, zero_detail_qword,
          location, &packed_report));
  loom_value_id_t args[5 +
                       LOOM_AMDGPU_SANITIZER_RACE_REPORT_DETAIL_QWORD_COUNT] = {
      source->dispatch_ptr,     source->workgroup_id_x, source->workitem_id_x,
      packed_report.check_kind, packed_report.flags,
  };
  for (uint32_t i = 0; i < LOOM_AMDGPU_SANITIZER_RACE_REPORT_DETAIL_QWORD_COUNT;
       ++i) {
    args[5 + i] = packed_report.detail_qwords[i];
  }
  loom_op_t* branch_op = NULL;
  return loom_low_br_build(builder, island->entry_block, args,
                           IREE_ARRAYSIZE(args), location, &branch_op);
}

iree_status_t loom_amdgpu_build_sanitizer_race_report_failure_mask_branch(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_race_report_island_t* island,
    loom_value_id_t failure_mask,
    const loom_amdgpu_feedback_packet_source_t* source,
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
  loom_amdgpu_feedback_failure_branch_t feedback_branch = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_failure_mask_split(
      builder, descriptor_set, failure_mask, location, &feedback_branch));
  *out_branch = (loom_amdgpu_sanitizer_race_report_failure_branch_t){
      .failure_block = feedback_branch.failure_block,
      .continuation_block = feedback_branch.continuation_block,
  };
  return iree_ok_status();
}
