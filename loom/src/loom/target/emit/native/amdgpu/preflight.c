// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/preflight.h"

#include <inttypes.h>

#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/emit/native/amdgpu/register_class.h"
#include "loom/target/emit/native/fragment.h"

// AMDGPU unspillable physical classes model singleton architectural state such
// as SCC, EXEC, and M0. They constrain scheduling/allocation but do not
// contribute SGPR/VGPR high-water metadata.
static bool loom_amdgpu_native_preflight_metadata_free_register_class(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t reg_class_id) {
  if (reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
      reg_class_id >= descriptor_set->reg_class_count) {
    return false;
  }
  const loom_low_reg_class_t* reg_class =
      &descriptor_set->reg_classes[reg_class_id];
  return iree_all_bits_set(reg_class->flags,
                           LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE);
}

static iree_status_t loom_amdgpu_native_preflight_update_high_water(
    uint32_t location_base, uint32_t location_count, uint32_t* inout_value) {
  const uint64_t next_free = (uint64_t)location_base + location_count;
  if (next_free > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU native emission register high-water mark overflows");
  }
  if ((uint32_t)next_free > *inout_value) {
    *inout_value = (uint32_t)next_free;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_native_preflight_collect_register_usage(
    const loom_low_allocation_table_t* allocation,
    loom_amdgpu_native_preflight_t* preflight) {
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    if (assignment->value_class.type_kind != LOOM_TYPE_REGISTER) {
      continue;
    }
    if (assignment->location_kind !=
        LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
      continue;
    }
    if (assignment->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_native_preflight_update_high_water(
          assignment->location_base, assignment->location_count,
          &preflight->next_free_sgpr));
      continue;
    }
    if (assignment->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_native_preflight_update_high_water(
          assignment->location_base, assignment->location_count,
          &preflight->next_free_vgpr));
      continue;
    }
    if (loom_amdgpu_register_class_is_agpr(
            allocation->target.descriptor_set,
            assignment->descriptor_reg_class_id)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "AMDGPU native emission register class 'amdgpu.agpr' requires "
          "additional kernel descriptor metadata");
    }
    if (loom_amdgpu_native_preflight_metadata_free_register_class(
            allocation->target.descriptor_set,
            assignment->descriptor_reg_class_id)) {
      continue;
    }
    iree_string_view_t register_class = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_register_class_name(
        allocation, assignment, &register_class));
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU native emission register class '%.*s' requires additional "
        "kernel descriptor metadata",
        (int)register_class.size, register_class.data);
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_native_preflight_analyze(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    loom_amdgpu_native_preflight_t* out_preflight) {
  *out_preflight = (loom_amdgpu_native_preflight_t){0};
  IREE_RETURN_IF_ERROR(
      loom_native_fragment_validate_emission_inputs(schedule, allocation));
  *out_preflight = (loom_amdgpu_native_preflight_t){
      .schedule = schedule,
      .allocation = allocation,
  };
  return loom_amdgpu_native_preflight_collect_register_usage(allocation,
                                                             out_preflight);
}
