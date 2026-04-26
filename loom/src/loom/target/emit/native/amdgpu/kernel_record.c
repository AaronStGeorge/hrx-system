// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/kernel_record.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/packet.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/gfx950_descriptors.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/emit/native/amdgpu/slot_layout.h"
#include "loom/target/emit/native/fragment.h"

#define LOOM_AMDGPU_KERNEL_RECORD_KERNARG_USER_SGPR_COUNT 2u

typedef struct loom_amdgpu_kernel_record_register_usage_t {
  // Highest SGPR index used by the function body plus one.
  uint32_t next_free_sgpr;
  // Highest VGPR index used by the function body plus one.
  uint32_t next_free_vgpr;
} loom_amdgpu_kernel_record_register_usage_t;

typedef struct loom_amdgpu_kernel_record_workitem_id_t {
  // Predicate identifying the ABI live-in value.
  bool (*is_live_in)(const loom_module_t* module, loom_value_id_t value_id);
  // Physical VGPR base required by the AMDGPU kernel ABI.
  uint32_t expected_location_base;
  // Kernel descriptor flag that requests this initialized system VGPR.
  loom_amdgpu_kernel_descriptor_flags_t descriptor_flag;
  // Diagnostic label for this workitem-id dimension.
  iree_string_view_t label;
} loom_amdgpu_kernel_record_workitem_id_t;

typedef struct loom_amdgpu_kernel_record_packed_workitem_id_t {
  // Predicate identifying the packed ABI live-in value.
  bool (*is_live_in)(const loom_module_t* module, loom_value_id_t value_id);
  // Kernel descriptor flags that request the packed logical dimensions.
  loom_amdgpu_kernel_descriptor_flags_t descriptor_flags;
  // Diagnostic label for the packed live-in source.
  iree_string_view_t label;
} loom_amdgpu_kernel_record_packed_workitem_id_t;

typedef struct loom_amdgpu_kernel_record_workgroup_id_t {
  // Predicate identifying the ABI live-in value.
  bool (*is_live_in)(const loom_module_t* module, loom_value_id_t value_id);
  // SGPR offset from the first enabled system SGPR.
  uint32_t expected_location_offset;
  // Kernel descriptor flag that requests this initialized system SGPR.
  loom_amdgpu_kernel_descriptor_flags_t descriptor_flag;
  // Diagnostic label for this workgroup-id dimension.
  iree_string_view_t label;
} loom_amdgpu_kernel_record_workgroup_id_t;

static bool loom_amdgpu_kernel_record_symbol_ref_equal(loom_symbol_ref_t lhs,
                                                       loom_symbol_ref_t rhs) {
  return lhs.module_id == rhs.module_id && lhs.symbol_id == rhs.symbol_id;
}

static iree_status_t loom_amdgpu_kernel_record_concat3(
    iree_string_view_t a, iree_string_view_t b, iree_string_view_t c,
    iree_string_view_t* out_value, iree_arena_allocator_t* arena) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = iree_string_view_empty();
  iree_host_size_t prefix_length = 0;
  iree_host_size_t length = 0;
  if (!iree_host_size_checked_add(a.size, b.size, &prefix_length) ||
      !iree_host_size_checked_add(prefix_length, c.size, &length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU kernel record string length overflows");
  }
  char* data = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, length, (void**)&data));
  char* cursor = data;
  memcpy(cursor, a.data, a.size);
  cursor += a.size;
  memcpy(cursor, b.data, b.size);
  cursor += b.size;
  memcpy(cursor, c.data, c.size);
  *out_value = iree_make_string_view(data, length);
  return iree_ok_status();
}

static bool loom_amdgpu_kernel_record_symbol_start_char(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_' ||
         c == '$';
}

static bool loom_amdgpu_kernel_record_symbol_continue_char(char c) {
  return loom_amdgpu_kernel_record_symbol_start_char(c) ||
         (c >= '0' && c <= '9') || c == '.';
}

static iree_status_t loom_amdgpu_kernel_record_validate_symbol(
    iree_string_view_t symbol) {
  if (iree_string_view_is_empty(symbol)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel emission symbol is required");
  }
  if (!loom_amdgpu_kernel_record_symbol_start_char(symbol.data[0])) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel emission symbol '%.*s' has an invalid first character",
        (int)symbol.size, symbol.data);
  }
  for (iree_host_size_t i = 1; i < symbol.size; ++i) {
    if (!loom_amdgpu_kernel_record_symbol_continue_char(symbol.data[i])) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU kernel emission symbol '%.*s' contains an invalid character",
          (int)symbol.size, symbol.data);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_record_validate_target_id_component(
    iree_string_view_t value, iree_string_view_t field_name) {
  if (iree_string_view_is_empty(value)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel emission target-id field '%.*s' is required",
        (int)field_name.size, field_name.data);
  }
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    const unsigned char c = (unsigned char)value.data[i];
    if (c <= ' ' || c == '"' || c == '\\') {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU kernel emission target-id field '%.*s' contains an "
          "unsupported character",
          (int)field_name.size, field_name.data);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_record_symbol_name(
    const loom_module_t* module, const loom_op_t* function_op,
    iree_string_view_t* out_symbol) {
  IREE_ASSERT_ARGUMENT(out_symbol);
  *out_symbol = iree_string_view_empty();
  loom_symbol_ref_t symbol_ref = loom_low_func_def_callee(function_op);
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel emission function symbol is invalid");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id == LOOM_STRING_ID_INVALID ||
      symbol->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel emission function symbol has no "
                            "module string");
  }
  *out_symbol = module->strings.entries[symbol->name_id];
  return loom_amdgpu_kernel_record_validate_symbol(*out_symbol);
}

static iree_status_t loom_amdgpu_kernel_record_export_symbol(
    const loom_low_resolved_target_t* target, const loom_module_t* module,
    const loom_op_t* function_op, iree_string_view_t* out_symbol) {
  IREE_ASSERT_ARGUMENT(out_symbol);
  if (!iree_string_view_is_empty(
          target->bundle_storage.export_plan.export_symbol)) {
    *out_symbol = target->bundle_storage.export_plan.export_symbol;
    return loom_amdgpu_kernel_record_validate_symbol(*out_symbol);
  }
  return loom_amdgpu_kernel_record_symbol_name(module, function_op, out_symbol);
}

static iree_status_t loom_amdgpu_kernel_record_validate_target(
    const loom_low_resolved_target_t* target) {
  const loom_target_snapshot_t* snapshot = &target->bundle_storage.snapshot;
  const loom_target_export_plan_t* export_plan =
      &target->bundle_storage.export_plan;
  if (target->descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel emission target bundle is required");
  }
  if (snapshot->codegen_format != LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU kernel emission requires low_native codegen snapshots");
  }
  if (!iree_string_view_equal(snapshot->target_triple,
                              IREE_SV("amdgcn-amd-amdhsa"))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU kernel emission requires amdgcn-amd-amdhsa, got '%.*s'",
        (int)snapshot->target_triple.size, snapshot->target_triple.data);
  }
  if (!iree_string_view_is_empty(snapshot->target_features)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU kernel emission target-id feature strings are not encoded yet");
  }
  if (snapshot->artifact_format != LOOM_TARGET_ARTIFACT_FORMAT_ELF) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU kernel emission requires ELF artifacts");
  }
  if (export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU kernel emission requires a HAL kernel ABI");
  }
  if (export_plan->linkage != LOOM_TARGET_LINKAGE_DEFAULT) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU kernel emission requires default linkage");
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_validate_target_id_component(
      snapshot->target_triple, IREE_SV("target_triple")));
  return loom_amdgpu_kernel_record_validate_target_id_component(
      snapshot->target_cpu, IREE_SV("target_cpu"));
}

static iree_status_t loom_amdgpu_kernel_record_validate_function_shape(
    const loom_op_t* function_op) {
  if (!loom_low_func_def_isa(function_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel emission requires low.func.def");
  }
  loom_region_t* body = loom_low_func_def_body(function_op);
  if (body == NULL || body->block_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel emission function body is required");
  }
  const loom_block_t* entry_block = loom_region_const_entry_block(body);
  if (entry_block->arg_count != 0 || function_op->result_count != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU kernel emission requires ABI-lowered kernels with no low "
        "function arguments or results");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_record_update_high_water(
    uint32_t location_base, uint32_t location_count, uint32_t* inout_value) {
  IREE_ASSERT_ARGUMENT(inout_value);
  uint64_t next_free = (uint64_t)location_base + location_count;
  if (next_free > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU kernel emission register high-water mark overflows");
  }
  if ((uint32_t)next_free > *inout_value) {
    *inout_value = (uint32_t)next_free;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_record_collect_register_usage(
    const loom_low_allocation_sidecar_t* allocation,
    loom_amdgpu_kernel_record_register_usage_t* out_usage) {
  IREE_ASSERT_ARGUMENT(out_usage);
  *out_usage = (loom_amdgpu_kernel_record_register_usage_t){0};
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    if (assignment->value_class.type_kind != LOOM_TYPE_REGISTER) {
      continue;
    }
    if (assignment->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_update_high_water(
          assignment->location_base, assignment->location_count,
          &out_usage->next_free_sgpr));
      continue;
    }
    if (assignment->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_update_high_water(
          assignment->location_base, assignment->location_count,
          &out_usage->next_free_vgpr));
      continue;
    }
    if (assignment->descriptor_reg_class_id ==
        AMDGPU_GFX950_CORE_REG_CLASS_ID_AMDGPU_AGPR) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "AMDGPU kernel emission register class 'amdgpu.agpr' requires "
          "additional kernel descriptor metadata");
    }
    iree_string_view_t register_class = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_register_class_name(
        allocation, assignment, &register_class));
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU kernel emission register class '%.*s' requires additional "
        "kernel descriptor metadata",
        (int)register_class.size, register_class.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_record_collect_segment_usage(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_amdgpu_slot_layout_segment_sizes_t* out_usage) {
  IREE_ASSERT_ARGUMENT(out_usage);
  *out_usage = (loom_amdgpu_slot_layout_segment_sizes_t){0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_slot_layout_collect_segment_sizes(
      module, function_op, out_usage));
  if (out_usage->group_segment_fixed_size > UINT32_MAX ||
      out_usage->private_segment_fixed_size > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU kernel emission fixed segment sizes exceed metadata limits");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_record_validate_kernarg_live_in(
    const loom_low_allocation_sidecar_t* allocation,
    const loom_amdgpu_hal_kernel_abi_layout_t* abi_layout) {
  if (!abi_layout->uses_kernarg_segment_ptr) {
    return iree_ok_status();
  }
  bool found = false;
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    if (assignment->value_class.type_kind != LOOM_TYPE_REGISTER) {
      continue;
    }
    if (!loom_amdgpu_hal_kernel_abi_is_kernarg_segment_ptr_live_in(
            allocation->module, assignment->value_id)) {
      continue;
    }
    if (found) {
      return iree_make_status(
          IREE_STATUS_ALREADY_EXISTS,
          "AMDGPU kernel emission found duplicate kernarg segment pointer "
          "live-ins");
    }
    found = true;
    if (assignment->location_base != 0 ||
        assignment->location_count !=
            LOOM_AMDGPU_KERNEL_RECORD_KERNARG_USER_SGPR_COUNT) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU kernel emission requires the kernarg segment pointer live-in "
          "to be fixed to s[0:%u]",
          LOOM_AMDGPU_KERNEL_RECORD_KERNARG_USER_SGPR_COUNT - 1);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_record_collect_descriptor_flags(
    const loom_low_allocation_sidecar_t* allocation, uint32_t system_sgpr_base,
    bool target_has_packed_workitem_id,
    loom_amdgpu_kernel_descriptor_flags_t* out_flags) {
  IREE_ASSERT_ARGUMENT(out_flags);
  *out_flags = 0;
  static const loom_amdgpu_kernel_record_workgroup_id_t workgroup_ids[] = {
      {
          .is_live_in = loom_amdgpu_hal_kernel_abi_is_workgroup_id_x_live_in,
          .expected_location_offset = 0,
          .descriptor_flag =
              LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_X,
          .label = IREE_SVL("workgroup_id.x"),
      },
      {
          .is_live_in = loom_amdgpu_hal_kernel_abi_is_workgroup_id_y_live_in,
          .expected_location_offset = 1,
          .descriptor_flag =
              LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Y,
          .label = IREE_SVL("workgroup_id.y"),
      },
      {
          .is_live_in = loom_amdgpu_hal_kernel_abi_is_workgroup_id_z_live_in,
          .expected_location_offset = 2,
          .descriptor_flag =
              LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Z,
          .label = IREE_SVL("workgroup_id.z"),
      },
  };
  static const loom_amdgpu_kernel_record_workitem_id_t workitem_ids[] = {
      {
          .is_live_in = loom_amdgpu_hal_kernel_abi_is_workitem_id_x_live_in,
          .expected_location_base = 0,
          .descriptor_flag =
              LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X,
          .label = IREE_SVL("workitem_id.x"),
      },
      {
          .is_live_in = loom_amdgpu_hal_kernel_abi_is_workitem_id_y_live_in,
          .expected_location_base = 1,
          .descriptor_flag =
              LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y,
          .label = IREE_SVL("workitem_id.y"),
      },
      {
          .is_live_in = loom_amdgpu_hal_kernel_abi_is_workitem_id_z_live_in,
          .expected_location_base = 2,
          .descriptor_flag =
              LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Z,
          .label = IREE_SVL("workitem_id.z"),
      },
  };
  static const loom_amdgpu_kernel_record_packed_workitem_id_t
      packed_workitem_ids[] = {
          {
              .is_live_in =
                  loom_amdgpu_hal_kernel_abi_is_workitem_id_packed_xy_live_in,
              .descriptor_flags =
                  LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X |
                  LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y,
              .label = IREE_SVL("packed workitem_id.x/y"),
          },
          {
              .is_live_in =
                  loom_amdgpu_hal_kernel_abi_is_workitem_id_packed_xyz_live_in,
              .descriptor_flags =
                  LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X |
                  LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y |
                  LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Z,
              .label = IREE_SVL("packed workitem_id.x/y/z"),
          },
      };
  bool found_workgroup_ids[IREE_ARRAYSIZE(workgroup_ids)] = {0};
  bool found_workitem_ids[IREE_ARRAYSIZE(workitem_ids)] = {0};
  bool found_packed_workitem_id = false;
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    if (assignment->value_class.type_kind != LOOM_TYPE_REGISTER) {
      continue;
    }
    for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(workgroup_ids); ++j) {
      if (!workgroup_ids[j].is_live_in(allocation->module,
                                       assignment->value_id)) {
        continue;
      }
      if (found_workgroup_ids[j]) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "AMDGPU kernel emission found duplicate %.*s live-ins",
            (int)workgroup_ids[j].label.size, workgroup_ids[j].label.data);
      }
      found_workgroup_ids[j] = true;
      const uint32_t expected_location_base =
          system_sgpr_base + workgroup_ids[j].expected_location_offset;
      if (assignment->location_base != expected_location_base ||
          assignment->location_count != 1) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU kernel emission requires the %.*s live-in to be fixed to "
            "s%" PRIu32,
            (int)workgroup_ids[j].label.size, workgroup_ids[j].label.data,
            expected_location_base);
      }
      *out_flags |= workgroup_ids[j].descriptor_flag;
    }
    for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(packed_workitem_ids); ++j) {
      if (!packed_workitem_ids[j].is_live_in(allocation->module,
                                             assignment->value_id)) {
        continue;
      }
      if (!target_has_packed_workitem_id) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU kernel emission target does not support %.*s live-ins",
            (int)packed_workitem_ids[j].label.size,
            packed_workitem_ids[j].label.data);
      }
      if (found_packed_workitem_id) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "AMDGPU kernel emission found duplicate packed workitem-id "
            "live-ins");
      }
      for (iree_host_size_t k = 0; k < IREE_ARRAYSIZE(found_workitem_ids);
           ++k) {
        if (found_workitem_ids[k]) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "AMDGPU kernel emission cannot mix packed and unpacked "
              "workitem-id live-ins");
        }
      }
      found_packed_workitem_id = true;
      if (assignment->location_base != 0 || assignment->location_count != 1) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU kernel emission requires the %.*s live-in to be fixed to "
            "v0",
            (int)packed_workitem_ids[j].label.size,
            packed_workitem_ids[j].label.data);
      }
      *out_flags |= packed_workitem_ids[j].descriptor_flags;
    }
    for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(workitem_ids); ++j) {
      if (!workitem_ids[j].is_live_in(allocation->module,
                                      assignment->value_id)) {
        continue;
      }
      if (found_packed_workitem_id) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AMDGPU kernel emission cannot mix packed and unpacked "
            "workitem-id live-ins");
      }
      if (found_workitem_ids[j]) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "AMDGPU kernel emission found duplicate %.*s live-ins",
            (int)workitem_ids[j].label.size, workitem_ids[j].label.data);
      }
      found_workitem_ids[j] = true;
      if (assignment->location_base != workitem_ids[j].expected_location_base ||
          assignment->location_count != 1) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU kernel emission requires the %.*s live-in to be fixed to "
            "v%" PRIu32,
            (int)workitem_ids[j].label.size, workitem_ids[j].label.data,
            workitem_ids[j].expected_location_base);
      }
      *out_flags |= workitem_ids[j].descriptor_flag;
    }
  }
  if (target_has_packed_workitem_id &&
      (found_workitem_ids[1] || found_workitem_ids[2])) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU kernel emission requires packed workitem-id live-ins when "
        "workitem_id.y/z are used on this target");
  }
  if (iree_any_bit_set(
          *out_flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Z)) {
    *out_flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_X |
                  LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Y;
  } else if (iree_any_bit_set(
                 *out_flags,
                 LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Y)) {
    *out_flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_X;
  }
  if (iree_any_bit_set(
          *out_flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Z)) {
    *out_flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X |
                  LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y;
  } else if (iree_any_bit_set(
                 *out_flags,
                 LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y)) {
    *out_flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_record_build_metadata_arguments(
    const loom_amdgpu_hal_kernel_abi_layout_t* abi_layout,
    const loom_amdgpu_metadata_argument_t** out_arguments,
    iree_arena_allocator_t* arena) {
  IREE_ASSERT_ARGUMENT(out_arguments);
  *out_arguments = NULL;
  if (abi_layout->resource_count == 0) {
    return iree_ok_status();
  }
  loom_amdgpu_metadata_argument_t* arguments = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, abi_layout->resource_count,
                                sizeof(*arguments), (void**)&arguments));
  for (iree_host_size_t i = 0; i < abi_layout->resource_count; ++i) {
    const loom_amdgpu_hal_kernarg_resource_t* resource =
        &abi_layout->resources[i];
    arguments[i] = (loom_amdgpu_metadata_argument_t){
        .name = resource->name,
        .offset = resource->kernarg_offset,
        .size = resource->kernarg_size,
        .alignment = resource->kernarg_alignment,
        .kind = LOOM_AMDGPU_METADATA_ARGUMENT_GLOBAL_BUFFER,
        .address_space = IREE_SV("global"),
    };
  }
  *out_arguments = arguments;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_kernel_record_build(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation,
    const loom_amdgpu_kernel_record_options_t* options,
    loom_amdgpu_kernel_record_t* out_record,
    iree_arena_allocator_t* scratch_arena) {
  IREE_ASSERT_ARGUMENT(out_record);
  *out_record = (loom_amdgpu_kernel_record_t){0};
  if (scratch_arena == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel emission scratch arena is required");
  }
  IREE_RETURN_IF_ERROR(
      loom_native_fragment_validate_emission_inputs(schedule, allocation));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_kernel_record_validate_target(&schedule->target));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_kernel_record_validate_function_shape(schedule->function_op));

  iree_string_view_t symbol = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_export_symbol(
      &schedule->target, schedule->module, schedule->function_op, &symbol));

  loom_amdgpu_hal_kernel_abi_layout_t derived_abi_layout = {0};
  const loom_amdgpu_hal_kernel_abi_layout_t* abi_layout =
      options ? options->abi_layout : NULL;
  if (abi_layout != NULL) {
    if (abi_layout->function_op != schedule->function_op) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU kernel record ABI layout does not belong to the scheduled "
          "function");
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_layout_from_low(
        schedule->module, schedule->function_op,
        &schedule->target.bundle_storage.bundle,
        schedule->target.descriptor_set, &derived_abi_layout, scratch_arena));
    abi_layout = &derived_abi_layout;
  }

  loom_amdgpu_kernel_record_register_usage_t register_usage = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_collect_register_usage(
      allocation, &register_usage));
  loom_amdgpu_slot_layout_segment_sizes_t segment_usage = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_collect_segment_usage(
      schedule->module, schedule->function_op, &segment_usage));
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_validate_kernarg_live_in(
      allocation, abi_layout));

  const loom_target_snapshot_t* snapshot =
      &schedule->target.bundle_storage.snapshot;
  const loom_target_hal_kernel_abi_t* hal_kernel =
      &schedule->target.bundle_storage.export_plan.hal_kernel;
  const loom_amdgpu_processor_info_t* processor = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_info_lookup_processor(
      snapshot->target_cpu, &processor));

  const uint32_t user_sgpr_count =
      abi_layout->uses_kernarg_segment_ptr
          ? LOOM_AMDGPU_KERNEL_RECORD_KERNARG_USER_SGPR_COUNT
          : 0;
  loom_amdgpu_kernel_descriptor_flags_t descriptor_flags = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_collect_descriptor_flags(
      allocation, user_sgpr_count,
      processor->kernel_descriptor_has_packed_workitem_id, &descriptor_flags));
  descriptor_flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_X;
  uint32_t system_vgpr_workitem_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_kernel_descriptor_workitem_id_mode_from_flags(
          descriptor_flags, &system_vgpr_workitem_id));

  const uint32_t next_free_sgpr =
      register_usage.next_free_sgpr > user_sgpr_count
          ? register_usage.next_free_sgpr
          : user_sgpr_count;

  iree_string_view_t target_id = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_concat3(
      snapshot->target_triple, IREE_SV("--"), snapshot->target_cpu, &target_id,
      scratch_arena));
  iree_string_view_t descriptor_symbol = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_concat3(
      symbol, IREE_SV(".kd"), iree_string_view_empty(), &descriptor_symbol,
      scratch_arena));

  const loom_amdgpu_metadata_argument_t* arguments = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_build_metadata_arguments(
      abi_layout, &arguments, scratch_arena));

  *out_record = (loom_amdgpu_kernel_record_t){
      .symbol = symbol,
      .descriptor_symbol = descriptor_symbol,
      .target_id = target_id,
      .abi_layout = *abi_layout,
      .metadata =
          {
              .name = symbol,
              .descriptor_symbol = descriptor_symbol,
              .kernarg_segment_size = abi_layout->kernarg_segment_size,
              .kernarg_segment_alignment =
                  abi_layout->kernarg_segment_alignment,
              .wavefront_size = processor->default_wavefront_size,
              .group_segment_fixed_size =
                  (uint32_t)segment_usage.group_segment_fixed_size,
              .private_segment_fixed_size =
                  (uint32_t)segment_usage.private_segment_fixed_size,
              .sgpr_count = next_free_sgpr,
              .vgpr_count = register_usage.next_free_vgpr,
              .max_flat_workgroup_size = hal_kernel->flat_workgroup_size_max,
              .required_workgroup_size = hal_kernel->required_workgroup_size,
              .has_required_workgroup_size = true,
              .arguments = arguments,
              .argument_count = abi_layout->resource_count,
          },
      .descriptor_flags = descriptor_flags,
      .system_vgpr_workitem_id = system_vgpr_workitem_id,
      .user_sgpr_count = user_sgpr_count,
  };
  return iree_ok_status();
}
