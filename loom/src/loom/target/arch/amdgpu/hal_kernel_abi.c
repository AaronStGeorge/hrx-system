// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/hal_kernel_abi.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/function.h"
#include "loom/codegen/low/register_class_map.h"
#include "loom/error/error_defs.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/hal_binding_descriptor.h"
#include "loom/target/arch/amdgpu/target_info.h"

#define LOOM_AMDGPU_HAL_KERNEL_ABI_COORDINATE_DIMENSION_COUNT 3u
#define LOOM_AMDGPU_HAL_KERNEL_ABI_MAX_FIXED_VALUE_COUNT \
  (3u + 2u * LOOM_AMDGPU_HAL_KERNEL_ABI_COORDINATE_DIMENSION_COUNT)
#define LOOM_AMDGPU_HAL_KERNEL_ABI_MAX_RESOURCE_COUNT \
  (UINT32_MAX / LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_SIZE)

enum {
  LOOM_AMDGPU_ERROR_HAL_ABI_RESOURCE_COUNT_OVERFLOW = 6,
  LOOM_AMDGPU_ERROR_HAL_ABI_RESOURCE_IMPORT_KIND = 7,
  LOOM_AMDGPU_ERROR_HAL_ABI_RESOURCE_RESULT_TYPE = 8,
  LOOM_AMDGPU_ERROR_HAL_ABI_BINDING_INDEX_NEGATIVE = 9,
  LOOM_AMDGPU_ERROR_HAL_ABI_BINDING_INDEX_SPARSE = 10,
  LOOM_AMDGPU_ERROR_HAL_ABI_BINDING_INDEX_DUPLICATE = 11,
  LOOM_AMDGPU_ERROR_HAL_ABI_BINDING_INDEX_MISSING = 12,
  LOOM_AMDGPU_ERROR_HAL_ABI_DESCRIPTOR_ATTR = 13,
  LOOM_AMDGPU_ERROR_HAL_ABI_DESCRIPTOR_CACHE_SWIZZLE = 14,
  LOOM_AMDGPU_ERROR_HAL_ABI_LIVE_IN_DUPLICATE = 15,
  LOOM_AMDGPU_ERROR_HAL_ABI_WORKITEM_LIVE_IN_MIX = 16,
  LOOM_AMDGPU_ERROR_HAL_ABI_LIVE_IN_RESULT_TYPE = 17,
  LOOM_AMDGPU_ERROR_HAL_ABI_M0_RESULT_TYPE = 18,
};

typedef enum loom_amdgpu_hal_kernel_abi_source_kind_e {
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_UNKNOWN = 0,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_KERNARG_SEGMENT_PTR = 1,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKGROUP_ID_X = 2,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKGROUP_ID_Y = 3,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKGROUP_ID_Z = 4,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_X = 5,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_Y = 6,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_Z = 7,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_PACKED_XY = 8,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_PACKED_XYZ = 9,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_M0 = 10,
} loom_amdgpu_hal_kernel_abi_source_kind_t;

static iree_string_view_t loom_amdgpu_hal_kernel_abi_source_name(
    loom_amdgpu_hal_kernel_abi_source_kind_t source_kind) {
  switch (source_kind) {
    case LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_KERNARG_SEGMENT_PTR:
      return IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_KERNARG_SEGMENT_PTR_SOURCE);
    case LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKGROUP_ID_X:
      return IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKGROUP_ID_X_SOURCE);
    case LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKGROUP_ID_Y:
      return IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKGROUP_ID_Y_SOURCE);
    case LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKGROUP_ID_Z:
      return IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKGROUP_ID_Z_SOURCE);
    case LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_X:
      return IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_X_SOURCE);
    case LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_Y:
      return IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_Y_SOURCE);
    case LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_Z:
      return IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_Z_SOURCE);
    case LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_PACKED_XY:
      return IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_PACKED_XY_SOURCE);
    case LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_PACKED_XYZ:
      return IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_PACKED_XYZ_SOURCE);
    case LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_M0:
      return IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_M0_SOURCE);
    default:
      return iree_string_view_empty();
  }
}

static loom_amdgpu_hal_kernel_abi_source_kind_t
loom_amdgpu_hal_kernel_abi_source_kind_from_stable_id(uint64_t source_id) {
  switch (source_id) {
    case LOOM_AMDGPU_HAL_KERNEL_ABI_KERNARG_SEGMENT_PTR_SOURCE_ID:
      return LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_KERNARG_SEGMENT_PTR;
    case LOOM_AMDGPU_HAL_KERNEL_ABI_WORKGROUP_ID_X_SOURCE_ID:
      return LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKGROUP_ID_X;
    case LOOM_AMDGPU_HAL_KERNEL_ABI_WORKGROUP_ID_Y_SOURCE_ID:
      return LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKGROUP_ID_Y;
    case LOOM_AMDGPU_HAL_KERNEL_ABI_WORKGROUP_ID_Z_SOURCE_ID:
      return LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKGROUP_ID_Z;
    case LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_X_SOURCE_ID:
      return LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_X;
    case LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_Y_SOURCE_ID:
      return LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_Y;
    case LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_Z_SOURCE_ID:
      return LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_Z;
    case LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_PACKED_XY_SOURCE_ID:
      return LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_PACKED_XY;
    case LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_PACKED_XYZ_SOURCE_ID:
      return LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_PACKED_XYZ;
    case LOOM_AMDGPU_HAL_KERNEL_ABI_M0_SOURCE_ID:
      return LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_M0;
    default:
      return LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_UNKNOWN;
  }
}

static bool loom_amdgpu_hal_kernel_abi_is_workgroup_source(
    loom_amdgpu_hal_kernel_abi_source_kind_t source_kind,
    uint32_t* out_dimension) {
  if (source_kind < LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKGROUP_ID_X ||
      source_kind > LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKGROUP_ID_Z) {
    return false;
  }
  *out_dimension = (uint32_t)(source_kind -
                              LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKGROUP_ID_X);
  return true;
}

static bool loom_amdgpu_hal_kernel_abi_is_workitem_source(
    loom_amdgpu_hal_kernel_abi_source_kind_t source_kind,
    uint32_t* out_dimension) {
  if (source_kind < LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_X ||
      source_kind > LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_Z) {
    return false;
  }
  *out_dimension =
      (uint32_t)(source_kind - LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_X);
  return true;
}

static loom_type_t loom_amdgpu_hal_kernel_abi_type_attr(
    const loom_module_t* module, loom_type_id_t type_id) {
  if (type_id == LOOM_TYPE_ID_INVALID || type_id >= module->types.count) {
    return loom_type_none();
  }
  return module->types.entries[type_id];
}

static bool loom_amdgpu_hal_kernel_abi_is_live_in_source_kind(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_amdgpu_hal_kernel_abi_source_kind_t expected_source_kind) {
  if (module == NULL || value_id >= module->values.count) {
    return false;
  }
  if (expected_source_kind == LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_UNKNOWN) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* def_op = loom_value_def_op(value);
  if (def_op == NULL || !loom_low_live_in_isa(def_op)) {
    return false;
  }
  return loom_amdgpu_hal_kernel_abi_source_kind_from_stable_id(
             (uint64_t)loom_low_live_in_source_id(def_op)) ==
         expected_source_kind;
}

bool loom_amdgpu_hal_kernel_abi_is_kernarg_segment_ptr_live_in(
    const loom_module_t* module, loom_value_id_t value_id) {
  return loom_amdgpu_hal_kernel_abi_is_live_in_source_kind(
      module, value_id, LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_KERNARG_SEGMENT_PTR);
}

bool loom_amdgpu_hal_kernel_abi_is_workgroup_id_x_live_in(
    const loom_module_t* module, loom_value_id_t value_id) {
  return loom_amdgpu_hal_kernel_abi_is_live_in_source_kind(
      module, value_id, LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKGROUP_ID_X);
}

bool loom_amdgpu_hal_kernel_abi_is_workgroup_id_y_live_in(
    const loom_module_t* module, loom_value_id_t value_id) {
  return loom_amdgpu_hal_kernel_abi_is_live_in_source_kind(
      module, value_id, LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKGROUP_ID_Y);
}

bool loom_amdgpu_hal_kernel_abi_is_workgroup_id_z_live_in(
    const loom_module_t* module, loom_value_id_t value_id) {
  return loom_amdgpu_hal_kernel_abi_is_live_in_source_kind(
      module, value_id, LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKGROUP_ID_Z);
}

bool loom_amdgpu_hal_kernel_abi_is_workitem_id_x_live_in(
    const loom_module_t* module, loom_value_id_t value_id) {
  return loom_amdgpu_hal_kernel_abi_is_live_in_source_kind(
      module, value_id, LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_X);
}

bool loom_amdgpu_hal_kernel_abi_is_workitem_id_y_live_in(
    const loom_module_t* module, loom_value_id_t value_id) {
  return loom_amdgpu_hal_kernel_abi_is_live_in_source_kind(
      module, value_id, LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_Y);
}

bool loom_amdgpu_hal_kernel_abi_is_workitem_id_z_live_in(
    const loom_module_t* module, loom_value_id_t value_id) {
  return loom_amdgpu_hal_kernel_abi_is_live_in_source_kind(
      module, value_id, LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_Z);
}

bool loom_amdgpu_hal_kernel_abi_is_workitem_id_packed_xy_live_in(
    const loom_module_t* module, loom_value_id_t value_id) {
  return loom_amdgpu_hal_kernel_abi_is_live_in_source_kind(
      module, value_id,
      LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_PACKED_XY);
}

bool loom_amdgpu_hal_kernel_abi_is_workitem_id_packed_xyz_live_in(
    const loom_module_t* module, loom_value_id_t value_id) {
  return loom_amdgpu_hal_kernel_abi_is_live_in_source_kind(
      module, value_id,
      LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_PACKED_XYZ);
}

bool loom_amdgpu_hal_kernel_abi_is_m0_live_in(const loom_module_t* module,
                                              loom_value_id_t value_id) {
  return loom_amdgpu_hal_kernel_abi_is_live_in_source_kind(
      module, value_id, LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_M0);
}

typedef enum loom_amdgpu_hal_kernel_abi_reg_class_e {
  LOOM_AMDGPU_HAL_KERNEL_ABI_REG_CLASS_SGPR = 0,
  LOOM_AMDGPU_HAL_KERNEL_ABI_REG_CLASS_VGPR = 1,
} loom_amdgpu_hal_kernel_abi_reg_class_t;

static iree_status_t loom_amdgpu_hal_kernel_abi_reg_class_id(
    loom_amdgpu_hal_kernel_abi_reg_class_t reg_class,
    uint16_t* out_descriptor_reg_class_id) {
  switch (reg_class) {
    case LOOM_AMDGPU_HAL_KERNEL_ABI_REG_CLASS_SGPR:
      *out_descriptor_reg_class_id = LOOM_AMDGPU_REG_CLASS_ID_SGPR;
      return iree_ok_status();
    case LOOM_AMDGPU_HAL_KERNEL_ABI_REG_CLASS_VGPR:
      *out_descriptor_reg_class_id = LOOM_AMDGPU_REG_CLASS_ID_VGPR;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown AMDGPU HAL kernel ABI register class %u",
                              (unsigned)reg_class);
  }
}

static iree_status_t loom_amdgpu_hal_kernel_abi_descriptor_reg_class(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_hal_kernel_abi_reg_class_t reg_class,
    uint16_t* out_descriptor_reg_class_id,
    const loom_low_reg_class_t** out_descriptor_reg_class) {
  *out_descriptor_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  if (out_descriptor_reg_class) {
    *out_descriptor_reg_class = NULL;
  }
  uint16_t descriptor_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_reg_class_id(
      reg_class, &descriptor_reg_class_id));
  if (descriptor_reg_class_id >= descriptor_set->reg_class_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU HAL kernel ABI descriptor register-class ID %" PRIu16
        " is out of range",
        descriptor_reg_class_id);
  }
  *out_descriptor_reg_class_id = descriptor_reg_class_id;
  if (out_descriptor_reg_class) {
    *out_descriptor_reg_class =
        &descriptor_set->reg_classes[descriptor_reg_class_id];
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_register_value_matches(
    const loom_module_t* module, const loom_low_register_class_map_t* map,
    loom_value_id_t value_id, loom_amdgpu_hal_kernel_abi_reg_class_t reg_class,
    uint32_t unit_count, bool* out_matches) {
  *out_matches = false;
  uint16_t expected_class_id = LOOM_LOW_REG_CLASS_NONE;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_descriptor_reg_class(
      map->descriptor_set, reg_class, &expected_class_id, NULL));
  loom_type_t type = loom_module_value_type(module, value_id);
  uint16_t actual_class_id = LOOM_LOW_REG_CLASS_NONE;
  bool found_actual_class = false;
  IREE_RETURN_IF_ERROR(loom_low_register_class_map_try_resolve_type(
      map, type, &actual_class_id, NULL, &found_actual_class));
  *out_matches = found_actual_class && actual_class_id == expected_class_id &&
                 loom_type_register_unit_count(type) == unit_count;
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_hal_kernel_abi_single_physical_register_matches(
    const loom_module_t* module, const loom_low_register_class_map_t* map,
    loom_value_id_t value_id, bool* out_matches) {
  *out_matches = false;
  loom_type_t type = loom_module_value_type(module, value_id);
  uint16_t actual_class_id = LOOM_LOW_REG_CLASS_NONE;
  const loom_low_reg_class_t* actual_class = NULL;
  bool found_actual_class = false;
  IREE_RETURN_IF_ERROR(loom_low_register_class_map_try_resolve_type(
      map, type, &actual_class_id, &actual_class, &found_actual_class));
  *out_matches =
      found_actual_class && actual_class != NULL &&
      loom_type_register_unit_count(type) == 1 &&
      actual_class->physical_count == 1 &&
      iree_any_bit_set(actual_class->flags, LOOM_LOW_REG_CLASS_FLAG_PHYSICAL) &&
      iree_any_bit_set(actual_class->flags,
                       LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE);
  return iree_ok_status();
}

static bool loom_amdgpu_hal_kernel_abi_has_workitem_id_live_in(
    const loom_value_id_t* workitem_ids) {
  for (uint32_t i = 0;
       i < LOOM_AMDGPU_HAL_KERNEL_ABI_COORDINATE_DIMENSION_COUNT; ++i) {
    if (workitem_ids[i] != LOOM_VALUE_ID_INVALID) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_hal_kernel_abi_can_emit(
    uint32_t max_errors,
    const loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  return max_errors == 0 || result->error_count < max_errors;
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op, uint16_t code,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count,
    const loom_diagnostic_related_op_t* related_ops,
    iree_host_size_t related_op_count, uint32_t max_errors,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  if (!loom_amdgpu_hal_kernel_abi_can_emit(max_errors, result)) {
    return iree_ok_status();
  }
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_AMDGPU, code),
      .params = params,
      .param_count = param_count,
      .related_ops = related_ops,
      .related_op_count = related_op_count,
  };
  IREE_RETURN_IF_ERROR(iree_diagnostic_emit(emitter, &emission));
  ++result->error_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit_resource_count_overflow(
    const loom_op_t* function_op, iree_host_size_t resource_count,
    uint32_t max_errors, iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_u64(resource_count),
      loom_param_u64(LOOM_AMDGPU_HAL_KERNEL_ABI_MAX_RESOURCE_COUNT),
  };
  return loom_amdgpu_hal_kernel_abi_emit(
      emitter, function_op, LOOM_AMDGPU_ERROR_HAL_ABI_RESOURCE_COUNT_OVERFLOW,
      params, IREE_ARRAYSIZE(params), NULL, 0, max_errors, result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit_resource_import_kind_error(
    const loom_op_t* resource_op, uint8_t actual_import_kind,
    uint32_t max_errors, iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_u32(LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BINDING),
      loom_param_u32(actual_import_kind),
  };
  return loom_amdgpu_hal_kernel_abi_emit(
      emitter, resource_op, LOOM_AMDGPU_ERROR_HAL_ABI_RESOURCE_IMPORT_KIND,
      params, IREE_ARRAYSIZE(params), NULL, 0, max_errors, result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit_resource_result_type_error(
    const loom_module_t* module, const loom_op_t* resource_op,
    uint32_t max_errors, iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_type(loom_module_value_type(
          module, loom_low_resource_result(resource_op))),
      loom_param_u32(LOOM_AMDGPU_REG_CLASS_ID_SGPR),
      loom_param_u32(2),
  };
  return loom_amdgpu_hal_kernel_abi_emit(
      emitter, resource_op, LOOM_AMDGPU_ERROR_HAL_ABI_RESOURCE_RESULT_TYPE,
      params, IREE_ARRAYSIZE(params), NULL, 0, max_errors, result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit_binding_index_negative(
    const loom_op_t* resource_op, int64_t binding_index, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_i64(binding_index),
  };
  return loom_amdgpu_hal_kernel_abi_emit(
      emitter, resource_op, LOOM_AMDGPU_ERROR_HAL_ABI_BINDING_INDEX_NEGATIVE,
      params, IREE_ARRAYSIZE(params), NULL, 0, max_errors, result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit_binding_index_sparse(
    const loom_op_t* resource_op, uint64_t binding_index,
    iree_host_size_t resource_count, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_u64(binding_index),
      loom_param_u64(resource_count),
  };
  return loom_amdgpu_hal_kernel_abi_emit(
      emitter, resource_op, LOOM_AMDGPU_ERROR_HAL_ABI_BINDING_INDEX_SPARSE,
      params, IREE_ARRAYSIZE(params), NULL, 0, max_errors, result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit_binding_index_duplicate(
    const loom_op_t* resource_op, const loom_op_t* previous_op,
    uint64_t binding_index, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_u64(binding_index),
  };
  const loom_diagnostic_related_op_t related[] = {{
      .label = IREE_SV("previous binding"),
      .op = previous_op,
      .field_ref = loom_diagnostic_field_ref(
          LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, loom_low_resource_index_ATTR_INDEX),
  }};
  return loom_amdgpu_hal_kernel_abi_emit(
      emitter, resource_op, LOOM_AMDGPU_ERROR_HAL_ABI_BINDING_INDEX_DUPLICATE,
      params, IREE_ARRAYSIZE(params), related, IREE_ARRAYSIZE(related),
      max_errors, result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit_missing_binding(
    const loom_op_t* function_op, uint64_t binding_index, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_u64(binding_index),
  };
  return loom_amdgpu_hal_kernel_abi_emit(
      emitter, function_op, LOOM_AMDGPU_ERROR_HAL_ABI_BINDING_INDEX_MISSING,
      params, IREE_ARRAYSIZE(params), NULL, 0, max_errors, result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit_descriptor_attr_error(
    const loom_op_t* op, iree_string_view_t attr_name,
    loom_attr_kind_t actual_kind, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(attr_name),
      loom_param_u32(LOOM_ATTR_I64),
      loom_param_u32((uint32_t)actual_kind),
  };
  return loom_amdgpu_hal_kernel_abi_emit(
      emitter, op, LOOM_AMDGPU_ERROR_HAL_ABI_DESCRIPTOR_ATTR, params,
      IREE_ARRAYSIZE(params), NULL, 0, max_errors, result);
}

static iree_status_t
loom_amdgpu_hal_kernel_abi_emit_descriptor_cache_swizzle_error(
    const loom_op_t* op, iree_string_view_t descriptor_set_key,
    uint64_t cache_swizzle_stride, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(descriptor_set_key),
      loom_param_u64(cache_swizzle_stride),
  };
  return loom_amdgpu_hal_kernel_abi_emit(
      emitter, op, LOOM_AMDGPU_ERROR_HAL_ABI_DESCRIPTOR_CACHE_SWIZZLE, params,
      IREE_ARRAYSIZE(params), NULL, 0, max_errors, result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit_live_in_duplicate(
    const loom_op_t* live_in_op, const loom_op_t* previous_op,
    iree_string_view_t source_name, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(source_name),
  };
  const loom_diagnostic_related_op_t related[] = {{
      .label = IREE_SV("previous live-in"),
      .op = previous_op,
      .field_ref = loom_diagnostic_field_ref(
          LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, loom_low_live_in_source_ATTR_INDEX),
  }};
  return loom_amdgpu_hal_kernel_abi_emit(
      emitter, live_in_op, LOOM_AMDGPU_ERROR_HAL_ABI_LIVE_IN_DUPLICATE, params,
      IREE_ARRAYSIZE(params), related, IREE_ARRAYSIZE(related), max_errors,
      result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit_workitem_live_in_mix(
    const loom_op_t* live_in_op, const loom_op_t* conflicting_op,
    iree_string_view_t source_name, iree_string_view_t conflicting_source_name,
    uint32_t max_errors, iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(source_name),
      loom_param_string(conflicting_source_name),
  };
  const loom_diagnostic_related_op_t related[] = {{
      .label = IREE_SV("conflicting live-in"),
      .op = conflicting_op,
      .field_ref = loom_diagnostic_field_ref(
          LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, loom_low_live_in_source_ATTR_INDEX),
  }};
  return loom_amdgpu_hal_kernel_abi_emit(
      emitter, live_in_op, LOOM_AMDGPU_ERROR_HAL_ABI_WORKITEM_LIVE_IN_MIX,
      params, IREE_ARRAYSIZE(params), related, IREE_ARRAYSIZE(related),
      max_errors, result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit_live_in_type_error(
    const loom_module_t* module, const loom_op_t* live_in_op,
    iree_string_view_t source_name, uint16_t expected_reg_class_id,
    uint32_t expected_unit_count, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(source_name),
      loom_param_type(
          loom_module_value_type(module, loom_low_live_in_result(live_in_op))),
      loom_param_u32(expected_reg_class_id),
      loom_param_u32(expected_unit_count),
  };
  return loom_amdgpu_hal_kernel_abi_emit(
      emitter, live_in_op, LOOM_AMDGPU_ERROR_HAL_ABI_LIVE_IN_RESULT_TYPE,
      params, IREE_ARRAYSIZE(params), NULL, 0, max_errors, result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_emit_m0_type_error(
    const loom_module_t* module, const loom_op_t* live_in_op,
    iree_string_view_t source_name, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(source_name),
      loom_param_type(
          loom_module_value_type(module, loom_low_live_in_result(live_in_op))),
  };
  return loom_amdgpu_hal_kernel_abi_emit(
      emitter, live_in_op, LOOM_AMDGPU_ERROR_HAL_ABI_M0_RESULT_TYPE, params,
      IREE_ARRAYSIZE(params), NULL, 0, max_errors, result);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_format_resource_name(
    uint32_t binding_index, iree_string_view_t* out_name,
    iree_arena_allocator_t* arena) {
  char scratch[32];
  int length =
      iree_snprintf(scratch, sizeof(scratch), "binding%" PRIu32, binding_index);
  if (length < 0 || (iree_host_size_t)length >= sizeof(scratch)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "failed to format AMDGPU HAL resource name");
  }
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, (iree_host_size_t)length, (void**)&storage));
  memcpy(storage, scratch, (iree_host_size_t)length);
  *out_name = iree_make_string_view(storage, (iree_host_size_t)length);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_resource_from_import(
    const loom_module_t* module, const loom_op_t* resource_op,
    loom_amdgpu_hal_kernarg_resource_t* out_resource,
    iree_arena_allocator_t* arena) {
  *out_resource = (loom_amdgpu_hal_kernarg_resource_t){0};

  iree_string_view_t resource_name = iree_string_view_empty();
  int64_t binding_index = loom_low_resource_index(resource_op);
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_format_resource_name(
      (uint32_t)binding_index, &resource_name, arena));

  uint64_t kernarg_offset =
      (uint64_t)binding_index *
      LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_SIZE;

  *out_resource = (loom_amdgpu_hal_kernarg_resource_t){
      .resource_op = resource_op,
      .name = resource_name,
      .binding_index = (uint32_t)binding_index,
      .kernarg_offset = (uint32_t)kernarg_offset,
      .kernarg_size = LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_SIZE,
      .kernarg_alignment =
          LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_ALIGNMENT,
      .source_type = loom_amdgpu_hal_kernel_abi_type_attr(
          module, loom_low_resource_source_type(resource_op)),
      .abi_type =
          loom_module_value_type(module, loom_low_resource_result(resource_op)),
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_verify_resource_type(
    const loom_module_t* module, const loom_low_register_class_map_t* map,
    const loom_op_t* resource_op, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  bool type_matches = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_register_value_matches(
      module, map, loom_low_resource_result(resource_op),
      LOOM_AMDGPU_HAL_KERNEL_ABI_REG_CLASS_SGPR, 2, &type_matches));
  if (!type_matches) {
    return loom_amdgpu_hal_kernel_abi_emit_resource_result_type_error(
        module, resource_op, max_errors, emitter, result);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_verify_resources(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_low_register_class_map_t* map, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result,
    iree_arena_allocator_t* arena) {
  const loom_region_t* body = loom_low_function_const_body(function_op);
  iree_host_size_t resource_count = 0;
  if (body != NULL) {
    for (uint16_t block_index = 0; block_index < body->block_count;
         ++block_index) {
      const loom_block_t* block = loom_region_const_block(body, block_index);
      const loom_op_t* resource_op = NULL;
      loom_block_for_each_op(block, resource_op) {
        if (loom_low_resource_isa(resource_op)) {
          ++resource_count;
        }
      }
    }
  }
  if (resource_count == 0) {
    return iree_ok_status();
  }
  if (resource_count > LOOM_AMDGPU_HAL_KERNEL_ABI_MAX_RESOURCE_COUNT) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_hal_kernel_abi_emit_resource_count_overflow(
            function_op, resource_count, max_errors, emitter, result));
    return iree_ok_status();
  }

  const loom_op_t** resource_slots = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, resource_count, sizeof(*resource_slots), (void**)&resource_slots));
  memset(resource_slots, 0, resource_count * sizeof(*resource_slots));

  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(body, block_index);
    const loom_op_t* resource_op = NULL;
    loom_block_for_each_op(block, resource_op) {
      if (!loom_low_resource_isa(resource_op)) {
        continue;
      }

      if (loom_low_resource_import_kind(resource_op) !=
          LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BINDING) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_hal_kernel_abi_emit_resource_import_kind_error(
                resource_op, loom_low_resource_import_kind(resource_op),
                max_errors, emitter, result));
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_verify_resource_type(
          module, map, resource_op, max_errors, emitter, result));

      const int64_t binding_index = loom_low_resource_index(resource_op);
      if (binding_index < 0) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_hal_kernel_abi_emit_binding_index_negative(
                resource_op, binding_index, max_errors, emitter, result));
        continue;
      }
      if ((uint64_t)binding_index >= resource_count) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_hal_kernel_abi_emit_binding_index_sparse(
                resource_op, (uint64_t)binding_index, resource_count,
                max_errors, emitter, result));
        continue;
      }

      const iree_host_size_t slot_index = (iree_host_size_t)binding_index;
      if (resource_slots[slot_index] != NULL) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_hal_kernel_abi_emit_binding_index_duplicate(
                resource_op, resource_slots[slot_index],
                (uint64_t)binding_index, max_errors, emitter, result));
        continue;
      }
      resource_slots[slot_index] = resource_op;
    }
  }

  for (iree_host_size_t i = 0; i < resource_count; ++i) {
    if (resource_slots[i] != NULL) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_emit_missing_binding(
        function_op, i, max_errors, emitter, result));
  }
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_hal_kernel_abi_verify_hal_buffer_descriptor_pseudos(
    const loom_op_t* function_op,
    const loom_low_descriptor_set_t* descriptor_set, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  loom_amdgpu_buffer_resource_cache_swizzle_t cache_swizzle_kind =
      LOOM_AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_NONE;
  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_info_lookup_descriptor_set_by_id(
      descriptor_set->stable_id, &descriptor_set_info));
  cache_swizzle_kind = descriptor_set_info->buffer_resource_cache_swizzle;
  const bool supports_cache_swizzle =
      cache_swizzle_kind ==
      LOOM_AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT;

  const loom_region_t* body = loom_low_function_const_body(function_op);
  if (body == NULL) {
    return iree_ok_status();
  }
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(body, block_index);
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (!loom_low_op_isa(op) ||
          (uint64_t)loom_low_op_descriptor_id(op) !=
              LOOM_AMDGPU_DESCRIPTOR_ID_HAL_BUFFER_DESCRIPTOR) {
        continue;
      }

      loom_named_attr_slice_t attrs = loom_low_op_attrs(op);
      if (attrs.count <=
              LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_ATTR_CACHE_SWIZZLE_STRIDE ||
          attrs.entries
                  [LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_ATTR_CACHE_SWIZZLE_STRIDE]
                      .value.kind != LOOM_ATTR_I64) {
        loom_attr_kind_t actual_kind = LOOM_ATTR_ABSENT;
        if (attrs.count >
            LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_ATTR_CACHE_SWIZZLE_STRIDE) {
          actual_kind =
              (loom_attr_kind_t)attrs
                  .entries
                      [LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_ATTR_CACHE_SWIZZLE_STRIDE]
                  .value.kind;
        }
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_hal_kernel_abi_emit_descriptor_attr_error(
                op, IREE_SV("cache_swizzle_stride"), actual_kind, max_errors,
                emitter, result));
        continue;
      }

      const int64_t cache_swizzle_stride =
          attrs
              .entries
                  [LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_ATTR_CACHE_SWIZZLE_STRIDE]
              .value.i64;
      if (cache_swizzle_stride == 0 || supports_cache_swizzle) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_hal_kernel_abi_emit_descriptor_cache_swizzle_error(
              op, descriptor_set_info->descriptor_set_key,
              (uint64_t)cache_swizzle_stride, max_errors, emitter, result));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_verify_live_in_type(
    const loom_module_t* module, const loom_low_register_class_map_t* map,
    const loom_op_t* live_in_op,
    loom_amdgpu_hal_kernel_abi_reg_class_t reg_class, uint32_t unit_count,
    iree_string_view_t source_name, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  uint16_t expected_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_reg_class_id(
      reg_class, &expected_reg_class_id));
  bool type_matches = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_register_value_matches(
      module, map, loom_low_live_in_result(live_in_op), reg_class, unit_count,
      &type_matches));
  if (!type_matches) {
    return loom_amdgpu_hal_kernel_abi_emit_live_in_type_error(
        module, live_in_op, source_name, expected_reg_class_id, unit_count,
        max_errors, emitter, result);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_verify_m0_live_in_type(
    const loom_module_t* module, const loom_low_register_class_map_t* map,
    const loom_op_t* live_in_op, iree_string_view_t source_name,
    uint32_t max_errors, iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  bool type_matches = false;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_abi_single_physical_register_matches(
          module, map, loom_low_live_in_result(live_in_op), &type_matches));
  if (!type_matches) {
    return loom_amdgpu_hal_kernel_abi_emit_m0_type_error(
        module, live_in_op, source_name, max_errors, emitter, result);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_verify_live_ins(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_low_register_class_map_t* map, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* result) {
  const loom_region_t* body = loom_low_function_const_body(function_op);
  if (body == NULL || body->block_count == 0) {
    return iree_ok_status();
  }

  const loom_op_t* kernarg_ptr_op = NULL;
  const loom_op_t*
      workgroup_id_ops[LOOM_AMDGPU_HAL_KERNEL_ABI_COORDINATE_DIMENSION_COUNT] =
          {
              NULL,
              NULL,
              NULL,
          };
  const loom_op_t*
      workitem_id_ops[LOOM_AMDGPU_HAL_KERNEL_ABI_COORDINATE_DIMENSION_COUNT] = {
          NULL,
          NULL,
          NULL,
      };
  const loom_op_t* packed_workitem_op = NULL;
  iree_string_view_t packed_workitem_source_name = iree_string_view_empty();
  const loom_op_t* m0_op = NULL;

  const loom_block_t* entry_block = loom_region_const_entry_block(body);
  const loom_op_t* live_in_op = NULL;
  loom_block_for_each_op(entry_block, live_in_op) {
    if (!loom_low_live_in_isa(live_in_op)) {
      continue;
    }
    const loom_amdgpu_hal_kernel_abi_source_kind_t source_kind =
        loom_amdgpu_hal_kernel_abi_source_kind_from_stable_id(
            (uint64_t)loom_low_live_in_source_id(live_in_op));
    const iree_string_view_t source_name =
        loom_amdgpu_hal_kernel_abi_source_name(source_kind);
    if (source_kind == LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_KERNARG_SEGMENT_PTR) {
      if (kernarg_ptr_op != NULL) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_emit_live_in_duplicate(
            live_in_op, kernarg_ptr_op, source_name, max_errors, emitter,
            result));
      }
      kernarg_ptr_op = live_in_op;
      IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_verify_live_in_type(
          module, map, live_in_op, LOOM_AMDGPU_HAL_KERNEL_ABI_REG_CLASS_SGPR, 2,
          source_name, max_errors, emitter, result));
      continue;
    }

    uint32_t dimension = 0;
    if (loom_amdgpu_hal_kernel_abi_is_workgroup_source(source_kind,
                                                       &dimension)) {
      if (workgroup_id_ops[dimension] != NULL) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_emit_live_in_duplicate(
            live_in_op, workgroup_id_ops[dimension], source_name, max_errors,
            emitter, result));
      }
      workgroup_id_ops[dimension] = live_in_op;
      IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_verify_live_in_type(
          module, map, live_in_op, LOOM_AMDGPU_HAL_KERNEL_ABI_REG_CLASS_SGPR, 1,
          source_name, max_errors, emitter, result));
      continue;
    }

    if (loom_amdgpu_hal_kernel_abi_is_workitem_source(source_kind,
                                                      &dimension)) {
      if (packed_workitem_op != NULL) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_hal_kernel_abi_emit_workitem_live_in_mix(
                live_in_op, packed_workitem_op, source_name,
                packed_workitem_source_name, max_errors, emitter, result));
      }
      if (workitem_id_ops[dimension] != NULL) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_emit_live_in_duplicate(
            live_in_op, workitem_id_ops[dimension], source_name, max_errors,
            emitter, result));
      }
      workitem_id_ops[dimension] = live_in_op;
      IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_verify_live_in_type(
          module, map, live_in_op, LOOM_AMDGPU_HAL_KERNEL_ABI_REG_CLASS_VGPR, 1,
          source_name, max_errors, emitter, result));
      continue;
    }

    if (source_kind ==
            LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_PACKED_XY ||
        source_kind ==
            LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_PACKED_XYZ) {
      if (packed_workitem_op != NULL) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_emit_live_in_duplicate(
            live_in_op, packed_workitem_op, source_name, max_errors, emitter,
            result));
      }
      for (uint32_t i = 0;
           i < LOOM_AMDGPU_HAL_KERNEL_ABI_COORDINATE_DIMENSION_COUNT; ++i) {
        if (workitem_id_ops[i] == NULL) {
          continue;
        }
        const loom_amdgpu_hal_kernel_abi_source_kind_t unpacked_source_kind =
            (loom_amdgpu_hal_kernel_abi_source_kind_t)(LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_X +
                                                       i);
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_hal_kernel_abi_emit_workitem_live_in_mix(
                live_in_op, workitem_id_ops[i], source_name,
                loom_amdgpu_hal_kernel_abi_source_name(unpacked_source_kind),
                max_errors, emitter, result));
      }
      packed_workitem_op = live_in_op;
      packed_workitem_source_name = source_name;
      IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_verify_live_in_type(
          module, map, live_in_op, LOOM_AMDGPU_HAL_KERNEL_ABI_REG_CLASS_VGPR, 1,
          source_name, max_errors, emitter, result));
      continue;
    }

    if (source_kind == LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_M0) {
      if (m0_op != NULL) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_emit_live_in_duplicate(
            live_in_op, m0_op, source_name, max_errors, emitter, result));
      }
      m0_op = live_in_op;
      IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_verify_m0_live_in_type(
          module, map, live_in_op, source_name, max_errors, emitter, result));
      continue;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_hal_kernel_abi_verify_low(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_low_descriptor_set_t* descriptor_set, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* out_result,
    iree_arena_allocator_t* arena) {
  *out_result = (loom_amdgpu_hal_kernel_abi_verify_result_t){0};
  if (module == NULL || function_op == NULL || descriptor_set == NULL ||
      arena == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL kernel ABI verification requires a module, function, "
        "descriptor set, and arena");
  }
  if (!loom_low_function_def_isa(function_op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL kernel ABI verification requires low.func.def or "
        "low.kernel.def");
  }

  loom_low_register_class_map_t register_class_map = {0};
  IREE_RETURN_IF_ERROR(loom_low_register_class_map_initialize(
      module, descriptor_set, arena, &register_class_map));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_verify_resources(
      module, function_op, &register_class_map, max_errors, emitter, out_result,
      arena));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_abi_verify_hal_buffer_descriptor_pseudos(
          function_op, descriptor_set, max_errors, emitter, out_result));
  return loom_amdgpu_hal_kernel_abi_verify_live_ins(
      module, function_op, &register_class_map, max_errors, emitter,
      out_result);
}

iree_status_t loom_amdgpu_hal_kernel_abi_layout_from_low(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_amdgpu_hal_kernel_abi_layout_t* out_layout,
    iree_arena_allocator_t* arena) {
  *out_layout = (loom_amdgpu_hal_kernel_abi_layout_t){0};
  if (module == NULL || function_op == NULL || arena == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL kernel ABI layout requires a module, function, and arena");
  }
  if (!loom_low_function_def_isa(function_op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL kernel ABI layout requires low.func.def or "
        "low.kernel.def");
  }

  const loom_region_t* body = loom_low_function_const_body(function_op);
  iree_host_size_t resource_count = 0;
  if (body != NULL) {
    for (uint16_t block_index = 0; block_index < body->block_count;
         ++block_index) {
      const loom_block_t* block = loom_region_const_block(body, block_index);
      const loom_op_t* resource_op = NULL;
      loom_block_for_each_op(block, resource_op) {
        if (!loom_low_resource_isa(resource_op)) {
          continue;
        }
        ++resource_count;
      }
    }
  }
  if (resource_count > LOOM_AMDGPU_HAL_KERNEL_ABI_MAX_RESOURCE_COUNT) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU HAL kernel ABI layout called with an unverified resource set");
  }

  loom_amdgpu_hal_kernarg_resource_t* resources = NULL;
  if (resource_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, resource_count, sizeof(*resources), (void**)&resources));
    memset(resources, 0, resource_count * sizeof(*resources));
  }

  if (body != NULL) {
    for (uint16_t block_index = 0; block_index < body->block_count;
         ++block_index) {
      const loom_block_t* block = loom_region_const_block(body, block_index);
      const loom_op_t* resource_op = NULL;
      loom_block_for_each_op(block, resource_op) {
        if (!loom_low_resource_isa(resource_op)) {
          continue;
        }

        const int64_t binding_index = loom_low_resource_index(resource_op);
        if (binding_index < 0 || (uint64_t)binding_index >= resource_count) {
          return iree_make_status(
              IREE_STATUS_INTERNAL,
              "AMDGPU HAL kernel ABI layout called with an unverified binding "
              "index");
        }
        const iree_host_size_t slot_index = (iree_host_size_t)binding_index;
        if (resources[slot_index].resource_op != NULL) {
          return iree_make_status(
              IREE_STATUS_INTERNAL,
              "AMDGPU HAL kernel ABI layout called with duplicate binding "
              "indexes");
        }

        loom_amdgpu_hal_kernarg_resource_t resource = {0};
        IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_resource_from_import(
            module, resource_op, &resource, arena));
        resources[slot_index] = resource;
      }
    }
  }

  for (iree_host_size_t i = 0; i < resource_count; ++i) {
    if (resources[i].resource_op == NULL) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "AMDGPU HAL kernel ABI layout called with a sparse binding set");
    }
  }

  *out_layout = (loom_amdgpu_hal_kernel_abi_layout_t){
      .function_op = function_op,
      .kernarg_segment_size =
          (uint32_t)resource_count *
          LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_SIZE,
      .kernarg_segment_alignment =
          LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_ALIGNMENT,
      .uses_kernarg_segment_ptr = true,
      .resources = resources,
      .resource_count = resource_count,
  };

  return iree_ok_status();
}

iree_status_t loom_amdgpu_hal_kernel_abi_fixed_values_from_low(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_low_allocation_fixed_value_t** out_fixed_values,
    iree_host_size_t* out_fixed_value_count, iree_arena_allocator_t* arena) {
  *out_fixed_values = NULL;
  *out_fixed_value_count = 0;
  if (module == NULL || function_op == NULL || arena == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL kernel ABI fixed-value collection requires a module, "
        "function, and arena");
  }
  if (!loom_low_function_def_isa(function_op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL kernel ABI fixed-value collection requires low.func.def or "
        "low.kernel.def");
  }

  const loom_region_t* body = loom_low_function_const_body(function_op);
  if (body == NULL || body->block_count == 0) {
    return iree_ok_status();
  }

  loom_low_allocation_fixed_value_t
      local_fixed_values[LOOM_AMDGPU_HAL_KERNEL_ABI_MAX_FIXED_VALUE_COUNT] = {
          0};
  iree_host_size_t local_fixed_value_count = 0;
  loom_value_id_t kernarg_ptr = LOOM_VALUE_ID_INVALID;
  loom_value_id_t
      workgroup_ids[LOOM_AMDGPU_HAL_KERNEL_ABI_COORDINATE_DIMENSION_COUNT] = {
          LOOM_VALUE_ID_INVALID,
          LOOM_VALUE_ID_INVALID,
          LOOM_VALUE_ID_INVALID,
      };
  loom_value_id_t
      workitem_ids[LOOM_AMDGPU_HAL_KERNEL_ABI_COORDINATE_DIMENSION_COUNT] = {
          LOOM_VALUE_ID_INVALID,
          LOOM_VALUE_ID_INVALID,
          LOOM_VALUE_ID_INVALID,
      };
  loom_value_id_t packed_workitem_id = LOOM_VALUE_ID_INVALID;
  loom_value_id_t m0 = LOOM_VALUE_ID_INVALID;

  const loom_block_t* entry_block = loom_region_const_entry_block(body);
  const loom_op_t* op = NULL;
  loom_block_for_each_op(entry_block, op) {
    if (!loom_low_live_in_isa(op)) {
      continue;
    }
    const loom_amdgpu_hal_kernel_abi_source_kind_t source_kind =
        loom_amdgpu_hal_kernel_abi_source_kind_from_stable_id(
            (uint64_t)loom_low_live_in_source_id(op));
    if (source_kind == LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_KERNARG_SEGMENT_PTR) {
      if (kernarg_ptr != LOOM_VALUE_ID_INVALID) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "AMDGPU HAL kernel ABI fixed-value collection reached duplicate "
            "kernarg segment pointer live-ins");
      }
      kernarg_ptr = loom_low_live_in_result(op);
      continue;
    }

    uint32_t dimension = 0;
    if (loom_amdgpu_hal_kernel_abi_is_workgroup_source(source_kind,
                                                       &dimension)) {
      if (workgroup_ids[dimension] != LOOM_VALUE_ID_INVALID) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "AMDGPU HAL kernel ABI fixed-value collection reached duplicate "
            "workgroup-id live-ins");
      }
      workgroup_ids[dimension] = loom_low_live_in_result(op);
      continue;
    }

    if (loom_amdgpu_hal_kernel_abi_is_workitem_source(source_kind,
                                                      &dimension)) {
      if (packed_workitem_id != LOOM_VALUE_ID_INVALID) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "AMDGPU HAL kernel ABI fixed-value collection reached mixed "
            "packed and unpacked workitem-id live-ins");
      }
      if (workitem_ids[dimension] != LOOM_VALUE_ID_INVALID) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "AMDGPU HAL kernel ABI fixed-value collection reached duplicate "
            "workitem-id live-ins");
      }
      workitem_ids[dimension] = loom_low_live_in_result(op);
      continue;
    }

    if (source_kind ==
            LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_PACKED_XY ||
        source_kind ==
            LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_PACKED_XYZ) {
      if (loom_amdgpu_hal_kernel_abi_has_workitem_id_live_in(workitem_ids)) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "AMDGPU HAL kernel ABI fixed-value collection reached mixed "
            "packed and unpacked workitem-id live-ins");
      }
      if (packed_workitem_id != LOOM_VALUE_ID_INVALID) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "AMDGPU HAL kernel ABI fixed-value collection reached duplicate "
            "packed workitem-id live-ins");
      }
      packed_workitem_id = loom_low_live_in_result(op);
      continue;
    }

    if (source_kind == LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_M0) {
      if (m0 != LOOM_VALUE_ID_INVALID) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "AMDGPU HAL kernel ABI fixed-value collection reached duplicate "
            "M0 live-ins");
      }
      m0 = loom_low_live_in_result(op);
      continue;
    }
  }

  if (kernarg_ptr != LOOM_VALUE_ID_INVALID) {
    local_fixed_values[local_fixed_value_count++] =
        (loom_low_allocation_fixed_value_t){
            .value_id = kernarg_ptr,
            .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
            .location_base = 0,
            .location_count = 2,
        };
  }
  const uint32_t workgroup_id_base =
      kernarg_ptr != LOOM_VALUE_ID_INVALID ? 2u : 0u;
  for (uint32_t i = 0;
       i < LOOM_AMDGPU_HAL_KERNEL_ABI_COORDINATE_DIMENSION_COUNT; ++i) {
    if (workgroup_ids[i] == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    local_fixed_values[local_fixed_value_count++] =
        (loom_low_allocation_fixed_value_t){
            .value_id = workgroup_ids[i],
            .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
            .location_base = workgroup_id_base + i,
            .location_count = 1,
        };
  }
  for (uint32_t i = 0;
       i < LOOM_AMDGPU_HAL_KERNEL_ABI_COORDINATE_DIMENSION_COUNT; ++i) {
    if (workitem_ids[i] == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    local_fixed_values[local_fixed_value_count++] =
        (loom_low_allocation_fixed_value_t){
            .value_id = workitem_ids[i],
            .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
            .location_base = i,
            .location_count = 1,
        };
  }
  if (packed_workitem_id != LOOM_VALUE_ID_INVALID) {
    local_fixed_values[local_fixed_value_count++] =
        (loom_low_allocation_fixed_value_t){
            .value_id = packed_workitem_id,
            .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
            .location_base = 0,
            .location_count = 1,
        };
  }
  if (m0 != LOOM_VALUE_ID_INVALID) {
    local_fixed_values[local_fixed_value_count++] =
        (loom_low_allocation_fixed_value_t){
            .value_id = m0,
            .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
            .location_base = 0,
            .location_count = 1,
        };
  }

  if (local_fixed_value_count == 0) {
    return iree_ok_status();
  }
  loom_low_allocation_fixed_value_t* fixed_values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, local_fixed_value_count,
                                                 sizeof(*fixed_values),
                                                 (void**)&fixed_values));
  memcpy(fixed_values, local_fixed_values,
         local_fixed_value_count * sizeof(*fixed_values));
  *out_fixed_values = fixed_values;
  *out_fixed_value_count = local_fixed_value_count;
  return iree_ok_status();
}
