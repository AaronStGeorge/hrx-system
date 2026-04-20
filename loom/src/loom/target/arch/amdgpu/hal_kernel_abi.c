// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/hal_kernel_abi.h"

#include <inttypes.h>
#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

static bool loom_amdgpu_hal_kernel_abi_symbol_ref_equal(loom_symbol_ref_t lhs,
                                                        loom_symbol_ref_t rhs) {
  return lhs.module_id == rhs.module_id && lhs.symbol_id == rhs.symbol_id;
}

static const loom_symbol_t* loom_amdgpu_hal_kernel_abi_lookup_defined_symbol(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return NULL;
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->definition == NULL || symbol->defining_op == NULL) {
    return NULL;
  }
  return symbol;
}

static iree_status_t loom_amdgpu_hal_kernel_abi_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref,
    iree_string_view_t field_name, iree_string_view_t* out_name) {
  IREE_ASSERT_ARGUMENT(out_name);
  *out_name = iree_string_view_empty();
  const loom_symbol_t* symbol =
      loom_amdgpu_hal_kernel_abi_lookup_defined_symbol(module, symbol_ref);
  if (symbol == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HAL kernel ABI %.*s symbol is not defined",
                            (int)field_name.size, field_name.data);
  }
  if (symbol->name_id == LOOM_STRING_ID_INVALID ||
      symbol->name_id >= module->strings.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL kernel ABI %.*s symbol has no module string",
        (int)field_name.size, field_name.data);
  }
  *out_name = module->strings.entries[symbol->name_id];
  return iree_ok_status();
}

static loom_type_t loom_amdgpu_hal_kernel_abi_type_attr(
    const loom_module_t* module, loom_type_id_t type_id) {
  if (type_id == LOOM_TYPE_ID_INVALID || type_id >= module->types.count) {
    return loom_type_none();
  }
  return module->types.entries[type_id];
}

static iree_string_view_t loom_amdgpu_hal_kernel_abi_string_or_empty(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static bool loom_amdgpu_hal_kernel_abi_is_live_in_source(
    const loom_module_t* module, loom_value_id_t value_id,
    iree_string_view_t expected_source) {
  if (module == NULL || value_id >= module->values.count) {
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
  iree_string_view_t source = loom_amdgpu_hal_kernel_abi_string_or_empty(
      module, loom_low_live_in_source(def_op));
  return iree_string_view_equal(source, expected_source);
}

bool loom_amdgpu_hal_kernel_abi_is_kernarg_segment_ptr_live_in(
    const loom_module_t* module, loom_value_id_t value_id) {
  return loom_amdgpu_hal_kernel_abi_is_live_in_source(
      module, value_id,
      IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_KERNARG_SEGMENT_PTR_SOURCE));
}

bool loom_amdgpu_hal_kernel_abi_is_workitem_id_x_live_in(
    const loom_module_t* module, loom_value_id_t value_id) {
  return loom_amdgpu_hal_kernel_abi_is_live_in_source(
      module, value_id,
      IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_X_SOURCE));
}

static bool loom_amdgpu_hal_kernel_abi_type_is_register_range(
    const loom_module_t* module, loom_type_t type,
    iree_string_view_t register_class, uint32_t unit_count) {
  if (!loom_type_is_register(type) ||
      loom_type_register_unit_count(type) != unit_count) {
    return false;
  }
  iree_string_view_t actual_register_class =
      loom_amdgpu_hal_kernel_abi_string_or_empty(
          module, loom_type_register_class_id(type));
  return iree_string_view_equal(actual_register_class, register_class);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_verify_register_value(
    const loom_module_t* module, loom_value_id_t value_id,
    iree_string_view_t register_class, uint32_t unit_count,
    iree_string_view_t label) {
  if (value_id >= module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HAL kernel ABI %.*s value id %" PRIu32
                            " is invalid",
                            (int)label.size, label.data, value_id);
  }
  loom_type_t type = loom_module_value_type(module, value_id);
  if (!loom_amdgpu_hal_kernel_abi_type_is_register_range(
          module, type, register_class, unit_count)) {
    if (unit_count == 1) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU HAL kernel ABI %.*s must have type reg<%.*s>",
          (int)label.size, label.data, (int)register_class.size,
          register_class.data);
    }
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL kernel ABI %.*s must have type reg<%.*s x%" PRIu32 ">",
        (int)label.size, label.data, (int)register_class.size,
        register_class.data, unit_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_validate_target(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_target_bundle_t* target_bundle) {
  if (target_bundle == NULL || target_bundle->snapshot == NULL ||
      target_bundle->export_plan == NULL || target_bundle->config == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL kernel ABI requires a complete target bundle");
  }
  if (!iree_string_view_equal(target_bundle->snapshot->target_triple,
                              IREE_SV("amdgcn-amd-amdhsa"))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU HAL kernel ABI requires amdgcn-amd-amdhsa, got '%.*s'",
        (int)target_bundle->snapshot->target_triple.size,
        target_bundle->snapshot->target_triple.data);
  }
  if (target_bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU HAL kernel ABI requires a hal_kernel export plan");
  }

  iree_string_view_t target_symbol_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_symbol_name(
      module, loom_low_func_def_target(function_op), IREE_SV("function target"),
      &target_symbol_name));
  if (!iree_string_view_equal(target_symbol_name, target_bundle->name)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU HAL kernel ABI target bundle '%.*s' does not match low "
        "function target '%.*s'",
        (int)target_bundle->name.size, target_bundle->name.data,
        (int)target_symbol_name.size, target_symbol_name.data);
  }

  if (iree_string_view_is_empty(target_bundle->export_plan->source_symbol)) {
    return iree_ok_status();
  }
  iree_string_view_t function_symbol_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_symbol_name(
      module, loom_low_func_def_callee(function_op), IREE_SV("function"),
      &function_symbol_name));
  if (!iree_string_view_equal(target_bundle->export_plan->source_symbol,
                              function_symbol_name)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU HAL kernel ABI export plan source '%.*s' does not match low "
        "function '%.*s'",
        (int)target_bundle->export_plan->source_symbol.size,
        target_bundle->export_plan->source_symbol.data,
        (int)function_symbol_name.size, function_symbol_name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_validate_semantic_type(
    const loom_module_t* module, loom_type_t semantic_type,
    iree_string_view_t resource_name) {
  if (!loom_type_is_dialect(semantic_type) ||
      loom_type_dialect_param_count(semantic_type) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL resource '%.*s' semantic type must be hal.buffer",
        (int)resource_name.size, resource_name.data);
  }
  iree_string_view_t type_name = loom_amdgpu_hal_kernel_abi_string_or_empty(
      module, loom_type_dialect_name_id(semantic_type));
  if (!iree_string_view_equal(type_name, IREE_SV("hal.buffer"))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL resource '%.*s' semantic type must be hal.buffer, got "
        "'%.*s'",
        (int)resource_name.size, resource_name.data, (int)type_name.size,
        type_name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_validate_abi_type(
    const loom_module_t* module, loom_type_t abi_type,
    iree_string_view_t resource_name) {
  if (!loom_type_is_register(abi_type)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL resource '%.*s' ABI type must be reg<amdgpu.sgpr x4>",
        (int)resource_name.size, resource_name.data);
  }
  iree_string_view_t register_class =
      loom_amdgpu_hal_kernel_abi_string_or_empty(
          module, loom_type_register_class_id(abi_type));
  uint32_t unit_count = loom_type_register_unit_count(abi_type);
  if (!iree_string_view_equal(register_class, IREE_SV("amdgpu.sgpr")) ||
      unit_count != 4) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL resource '%.*s' ABI type must be reg<amdgpu.sgpr x4>, "
        "got reg<%.*s x%" PRIu32 ">",
        (int)resource_name.size, resource_name.data, (int)register_class.size,
        register_class.data, unit_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_validate_resource_record(
    const loom_module_t* module, const loom_op_t* resource_op,
    loom_symbol_ref_t function_symbol, iree_string_view_t resource_name) {
  if (!loom_amdgpu_hal_kernel_abi_symbol_ref_equal(
          loom_low_abi_resource_function(resource_op), function_symbol)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU HAL resource '%.*s' is not owned by the selected low function",
        (int)resource_name.size, resource_name.data);
  }
  if (loom_low_abi_resource_kind(resource_op) !=
      LOOM_LOW_ABI_RESOURCE_KIND_HAL_BUFFER_RESOURCE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL resource '%.*s' must use kind hal_buffer_resource",
        (int)resource_name.size, resource_name.data);
  }
  if (loom_low_abi_resource_index(resource_op) < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL resource '%.*s' binding index must be non-negative",
        (int)resource_name.size, resource_name.data);
  }
  loom_type_t semantic_type = loom_amdgpu_hal_kernel_abi_type_attr(
      module, loom_low_abi_resource_semantic_type(resource_op));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_validate_semantic_type(
      module, semantic_type, resource_name));
  loom_type_t abi_type = loom_amdgpu_hal_kernel_abi_type_attr(
      module, loom_low_abi_resource_abi_type(resource_op));
  return loom_amdgpu_hal_kernel_abi_validate_abi_type(module, abi_type,
                                                      resource_name);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_resource_from_record(
    const loom_module_t* module, const loom_op_t* resource_op,
    loom_symbol_ref_t function_symbol,
    loom_amdgpu_hal_kernarg_resource_t* out_resource) {
  IREE_ASSERT_ARGUMENT(out_resource);
  *out_resource = (loom_amdgpu_hal_kernarg_resource_t){0};

  loom_symbol_ref_t resource_symbol = loom_low_abi_resource_symbol(resource_op);
  iree_string_view_t resource_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_symbol_name(
      module, resource_symbol, IREE_SV("resource"), &resource_name));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_validate_resource_record(
      module, resource_op, function_symbol, resource_name));

  int64_t binding_index = loom_low_abi_resource_index(resource_op);
  if (binding_index > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU HAL resource '%.*s' binding index %" PRIi64 " exceeds uint32_t",
        (int)resource_name.size, resource_name.data, binding_index);
  }
  uint64_t kernarg_offset =
      (uint64_t)binding_index *
      LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_SIZE;
  if (kernarg_offset > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU HAL resource '%.*s' kernarg offset overflows uint32_t",
        (int)resource_name.size, resource_name.data);
  }

  *out_resource = (loom_amdgpu_hal_kernarg_resource_t){
      .resource_op = resource_op,
      .symbol = resource_symbol,
      .name = resource_name,
      .binding_index = (uint32_t)binding_index,
      .kernarg_offset = (uint32_t)kernarg_offset,
      .kernarg_size = LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_SIZE,
      .kernarg_alignment =
          LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_ALIGNMENT,
      .semantic_type = loom_amdgpu_hal_kernel_abi_type_attr(
          module, loom_low_abi_resource_semantic_type(resource_op)),
      .abi_type = loom_amdgpu_hal_kernel_abi_type_attr(
          module, loom_low_abi_resource_abi_type(resource_op)),
  };
  return iree_ok_status();
}

static const loom_op_t*
loom_amdgpu_hal_kernel_abi_lookup_resource_record_by_symbol(
    const loom_module_t* module, loom_symbol_ref_t resource_symbol) {
  const loom_symbol_t* symbol =
      loom_amdgpu_hal_kernel_abi_lookup_defined_symbol(module, resource_symbol);
  if (symbol == NULL || !loom_low_abi_resource_isa(symbol->defining_op)) {
    return NULL;
  }
  return symbol->defining_op;
}

static bool loom_amdgpu_hal_kernel_abi_layout_has_resource(
    const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    loom_symbol_ref_t resource_symbol) {
  for (iree_host_size_t i = 0; i < layout->resource_count; ++i) {
    if (loom_amdgpu_hal_kernel_abi_symbol_ref_equal(layout->resources[i].symbol,
                                                    resource_symbol)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_amdgpu_hal_kernel_abi_validate_body_imports(
    const loom_module_t* module, const loom_region_t* body,
    const loom_amdgpu_hal_kernel_abi_layout_t* layout) {
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(body, block_index);
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (!loom_low_resource_isa(op)) {
        continue;
      }
      loom_symbol_ref_t resource_symbol = loom_low_resource_resource(op);
      const loom_op_t* resource_record =
          loom_amdgpu_hal_kernel_abi_lookup_resource_record_by_symbol(
              module, resource_symbol);
      if (resource_record == NULL) {
        return iree_make_status(
            IREE_STATUS_NOT_FOUND,
            "AMDGPU HAL low.resource references a missing resource record");
      }
      iree_string_view_t resource_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_symbol_name(
          module, resource_symbol, IREE_SV("resource"), &resource_name));
      if (!loom_amdgpu_hal_kernel_abi_layout_has_resource(layout,
                                                          resource_symbol)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU HAL low.resource '%.*s' is not owned by the selected low "
            "function",
            (int)resource_name.size, resource_name.data);
      }

      loom_type_t record_abi_type = loom_amdgpu_hal_kernel_abi_type_attr(
          module, loom_low_abi_resource_abi_type(resource_record));
      loom_type_t result_type =
          loom_module_value_type(module, loom_low_resource_result(op));
      if (!loom_type_equal(record_abi_type, result_type)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU HAL low.resource '%.*s' result type does not match its "
            "resource ABI type",
            (int)resource_name.size, resource_name.data);
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_hal_kernel_abi_layout_from_low(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_target_bundle_t* target_bundle,
    loom_amdgpu_hal_kernel_abi_layout_t* out_layout,
    iree_arena_allocator_t* arena) {
  IREE_ASSERT_ARGUMENT(out_layout);
  *out_layout = (loom_amdgpu_hal_kernel_abi_layout_t){0};
  if (module == NULL || function_op == NULL || arena == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL kernel ABI requires a module, function, and arena");
  }
  if (!loom_low_func_def_isa(function_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HAL kernel ABI requires low.func.def");
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_validate_target(
      module, function_op, target_bundle));

  loom_symbol_ref_t function_symbol = loom_low_func_def_callee(function_op);
  uint64_t max_binding_index = 0;
  iree_host_size_t resource_count = 0;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_op_t* resource_op = module->symbols.entries[i].defining_op;
    if (resource_op == NULL || !loom_low_abi_resource_isa(resource_op)) {
      continue;
    }
    if (!loom_amdgpu_hal_kernel_abi_symbol_ref_equal(
            loom_low_abi_resource_function(resource_op), function_symbol)) {
      continue;
    }
    ++resource_count;
    int64_t binding_index = loom_low_abi_resource_index(resource_op);
    if (binding_index < 0) {
      iree_string_view_t resource_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_symbol_name(
          module, loom_low_abi_resource_symbol(resource_op),
          IREE_SV("resource"), &resource_name));
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU HAL resource '%.*s' binding index must be non-negative",
          (int)resource_name.size, resource_name.data);
    }
    if ((uint64_t)binding_index > max_binding_index) {
      max_binding_index = (uint64_t)binding_index;
    }
  }

  if (resource_count == 0) {
    *out_layout = (loom_amdgpu_hal_kernel_abi_layout_t){
        .function_op = function_op,
        .kernarg_segment_size = 0,
        .kernarg_segment_alignment =
            LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_ALIGNMENT,
        .uses_kernarg_segment_ptr = false,
    };
    loom_region_t* body = loom_low_func_def_body(function_op);
    if (body != NULL) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_validate_body_imports(
          module, body, out_layout));
    }
    return iree_ok_status();
  }

  if (max_binding_index >= resource_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL resource binding indexes must be dense from zero; got "
        "%" PRIhsz " resources with max index %" PRIu64,
        resource_count, max_binding_index);
  }

  loom_amdgpu_hal_kernarg_resource_t* resources = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, resource_count, sizeof(*resources), (void**)&resources));
  memset(resources, 0, resource_count * sizeof(*resources));

  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_op_t* resource_op = module->symbols.entries[i].defining_op;
    if (resource_op == NULL || !loom_low_abi_resource_isa(resource_op)) {
      continue;
    }
    if (!loom_amdgpu_hal_kernel_abi_symbol_ref_equal(
            loom_low_abi_resource_function(resource_op), function_symbol)) {
      continue;
    }

    loom_amdgpu_hal_kernarg_resource_t resource = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_resource_from_record(
        module, resource_op, function_symbol, &resource));
    if (resources[resource.binding_index].resource_op != NULL) {
      return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                              "AMDGPU HAL resource binding index %" PRIu32
                              " is defined more than once",
                              resource.binding_index);
    }
    resources[resource.binding_index] = resource;
  }

  for (iree_host_size_t i = 0; i < resource_count; ++i) {
    if (resources[i].resource_op == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU HAL resource binding index %" PRIhsz " is missing", i);
    }
  }

  if (resource_count >
      UINT32_MAX / LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_SIZE) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU HAL kernarg segment size overflows uint32_t");
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

  loom_region_t* body = loom_low_func_def_body(function_op);
  if (body != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_validate_body_imports(
        module, body, out_layout));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_hal_kernel_abi_fixed_values_from_low(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_low_allocation_fixed_value_t** out_fixed_values,
    iree_host_size_t* out_fixed_value_count, iree_arena_allocator_t* arena) {
  IREE_ASSERT_ARGUMENT(out_fixed_values);
  IREE_ASSERT_ARGUMENT(out_fixed_value_count);
  *out_fixed_values = NULL;
  *out_fixed_value_count = 0;
  if (module == NULL || function_op == NULL || arena == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL kernel ABI fixed-value collection requires a module, "
        "function, and arena");
  }
  if (!loom_low_func_def_isa(function_op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL kernel ABI fixed-value collection requires low.func.def");
  }

  loom_region_t* body = loom_low_func_def_body(function_op);
  if (body == NULL || body->block_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL kernel ABI fixed-value collection requires a function "
        "body");
  }

  loom_low_allocation_fixed_value_t local_fixed_values[2] = {0};
  iree_host_size_t local_fixed_value_count = 0;
  loom_value_id_t kernarg_ptr = LOOM_VALUE_ID_INVALID;
  loom_value_id_t workitem_id_x = LOOM_VALUE_ID_INVALID;

  const loom_block_t* entry_block = loom_region_const_entry_block(body);
  const loom_op_t* op = NULL;
  loom_block_for_each_op(entry_block, op) {
    if (!loom_low_live_in_isa(op)) {
      continue;
    }
    iree_string_view_t source = loom_amdgpu_hal_kernel_abi_string_or_empty(
        module, loom_low_live_in_source(op));
    if (iree_string_view_equal(
            source,
            IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_KERNARG_SEGMENT_PTR_SOURCE))) {
      if (kernarg_ptr != LOOM_VALUE_ID_INVALID) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "AMDGPU HAL kernel ABI fixed-value collection found duplicate "
            "kernarg segment pointer live-ins");
      }
      kernarg_ptr = loom_low_live_in_result(op);
      IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_verify_register_value(
          module, kernarg_ptr, IREE_SV("amdgpu.sgpr"), 2,
          IREE_SV("kernarg segment pointer live-in")));
      local_fixed_values[local_fixed_value_count++] =
          (loom_low_allocation_fixed_value_t){
              .value_id = kernarg_ptr,
              .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
              .location_base = 0,
              .location_count = 2,
          };
      continue;
    }
    if (iree_string_view_equal(
            source, IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_X_SOURCE))) {
      if (workitem_id_x != LOOM_VALUE_ID_INVALID) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "AMDGPU HAL kernel ABI fixed-value collection found duplicate "
            "workitem_id.x live-ins");
      }
      workitem_id_x = loom_low_live_in_result(op);
      IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_verify_register_value(
          module, workitem_id_x, IREE_SV("amdgpu.vgpr"), 1,
          IREE_SV("workitem_id.x live-in")));
      local_fixed_values[local_fixed_value_count++] =
          (loom_low_allocation_fixed_value_t){
              .value_id = workitem_id_x,
              .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
              .location_base = 0,
              .location_count = 1,
          };
      continue;
    }
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
