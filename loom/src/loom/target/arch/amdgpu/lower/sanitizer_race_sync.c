// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/kernel/ops.h"
#include "loom/target/arch/amdgpu/abi/tsan.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/feedback.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_race.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_race_history_layout.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_race_lifetime.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_race_report.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_race_state.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_report.h"
#include "loom/target/arch/amdgpu/lower/sync.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static iree_status_t loom_amdgpu_sanitizer_race_sync_vgpr_u32_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t value, loom_value_id_t* out_value) {
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  return loom_amdgpu_emit_const_u32(context, source_op,
                                    LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, value,
                                    vgpr_type, out_value);
}

static iree_status_t
loom_amdgpu_sanitizer_race_emit_synchronization_capacity_failure(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_sync_plan_t* plan,
    const loom_amdgpu_sanitizer_race_config_values_t* config,
    const loom_amdgpu_sanitizer_race_history_layout_t* history_layout,
    loom_value_id_t subgroup_clock_offset, loom_value_id_t workitem_linear_id,
    loom_value_id_t capacity_mask, loom_value_id_t observed_epoch) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_location_id_t location = source_op->location;
  const loom_sanitizer_reporting_mode_t reporting_mode =
      loom_low_lower_context_sanitizer_reporting_mode(context);
  if (reporting_mode == LOOM_SANITIZER_REPORTING_MODE_TRAP) {
    const loom_amdgpu_sanitizer_trap_island_t* island = NULL;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_sanitizer_race_get_trap_island(context, location, &island));
    loom_amdgpu_sanitizer_access_report_failure_branch_t branch = {0};
    return loom_amdgpu_build_sanitizer_trap_failure_mask_branch_to_island(
        builder, descriptor_set, island, capacity_mask, location, &branch);
  }

  IREE_ASSERT(reporting_mode == LOOM_SANITIZER_REPORTING_MODE_DEFAULT ||
              reporting_mode == LOOM_SANITIZER_REPORTING_MODE_REPORT_ONLY);
  loom_amdgpu_feedback_failure_branch_t branch = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_failure_mask_split(
      builder, descriptor_set, capacity_mask, location, &branch));

  const loom_amdgpu_sanitizer_race_report_coordinates_t* coordinates = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_get_report_coordinates(context, &coordinates));
  const loom_amdgpu_feedback_packet_source_t source = {
      .dispatch_ptr = coordinates->dispatch_ptr,
      .workgroup_id_x = coordinates->workgroup_ids[LOOM_KERNEL_DIMENSION_X],
      .workitem_id_x = workitem_linear_id,
  };

  loom_amdgpu_sanitizer_race_synchronization_capacity_report_t report = {
      .subgroup_epoch = observed_epoch,
      .current_workgroup_ids =
          {
              coordinates->workgroup_ids[LOOM_KERNEL_DIMENSION_X],
              coordinates->workgroup_ids[LOOM_KERNEL_DIMENSION_Y],
              coordinates->workgroup_ids[LOOM_KERNEL_DIMENSION_Z],
          },
      .current_workitem_linear_id = workitem_linear_id,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_constant_u64(
      context, source_op, plan->site_id, &report.current_site_id));
  const uint32_t subgroup_epoch_capacity =
      1u << history_layout->subgroup_epoch.bit_count;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_sync_vgpr_u32_constant(
      context, source_op, subgroup_epoch_capacity,
      &report.subgroup_epoch_capacity));
  const uint32_t wavefront_size = 1u << history_layout->wavefront_size_log2;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_sync_vgpr_u32_constant(
      context, source_op, wavefront_size, &report.wavefront_size));
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      history_layout->wavefront_size_log2, workitem_linear_id, vgpr_type,
      &report.subgroup_id));
  loom_value_id_t shadow_base_vgpr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_vgpr_registers(
      builder, descriptor_set, config->shadow_base,
      /*expected_unit_count=*/2, location, &shadow_base_vgpr));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_add(
      context, source_op, shadow_base_vgpr, subgroup_clock_offset,
      &report.subgroup_clock_address));

  const loom_amdgpu_sanitizer_race_synchronization_capacity_report_island_t*
      island = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_get_synchronization_capacity_report_island(
          context, location, &island));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_build_sanitizer_race_synchronization_capacity_report_branch(
          builder, descriptor_set, island, &source, &report, location));
  loom_builder_set_block(builder, branch.continuation_block);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_sanitizer_race_sync(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_sync_plan_t* plan) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_amdgpu_sanitizer_race_config_values_t config = {0};
  loom_block_t* continuation_block = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_build_config_guard(
      context, source_op, &config, &continuation_block));

  const loom_amdgpu_sanitizer_race_history_layout_t* history_layout = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_get_history_layout(context, &history_layout));
  loom_value_id_t workitem_linear_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_get_workitem_linear_id(
      context, &workitem_linear_id));
  loom_value_id_t subgroup_clock_offset = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_sanitizer_race_history_layout_has_subgroups(history_layout)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_get_subgroup_clock_offset(
        context, &subgroup_clock_offset));
  }

  switch (plan->scope) {
    case LOOM_ATOMIC_SCOPE_SUBGROUP: {
      IREE_ASSERT(loom_amdgpu_sanitizer_race_history_layout_has_subgroups(
          history_layout));
      loom_value_id_t capacity_mask = LOOM_VALUE_ID_INVALID;
      loom_value_id_t observed_epoch = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_advance_subgroup_epoch(
          context, source_op, &config, history_layout, subgroup_clock_offset,
          workitem_linear_id, &capacity_mask, &observed_epoch));
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_sanitizer_race_emit_synchronization_capacity_failure(
              context, source_op, plan, &config, history_layout,
              subgroup_clock_offset, workitem_linear_id, capacity_mask,
              observed_epoch));
      break;
    }
    case LOOM_ATOMIC_SCOPE_WORKGROUP: {
      loom_value_id_t workgroup_shadow_record_offset = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_sanitizer_race_get_workgroup_shadow_record_offset(
              context, &workgroup_shadow_record_offset));
      loom_value_id_t workgroup_chunk_state_bitmap_size = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_sanitizer_race_get_workgroup_chunk_state_bitmap_size(
              context, &workgroup_chunk_state_bitmap_size));
      IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_advance_barrier_epoch(
          context, source_op, &config, workgroup_shadow_record_offset,
          workgroup_chunk_state_bitmap_size, workitem_linear_id));
      if (loom_amdgpu_sanitizer_race_history_layout_has_subgroups(
              history_layout)) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_reset_subgroup_epoch(
            context, source_op, &config, history_layout, subgroup_clock_offset,
            workitem_linear_id));
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_lower_workgroup_barrier_plan(
          context, source_op, &plan->barrier));
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("selected AMDGPU TSAN synchronization scope");
      IREE_BUILTIN_UNREACHABLE();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_branch_to_continuation(
      builder, continuation_block, source_op->location));
  return iree_ok_status();
}
