// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/hal_resource_materialization.h"

#include <inttypes.h>

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/arch/amdgpu/hal_kernel_abi.h"

#define LOOM_AMDGPU_HAL_RESOURCE_DESCRIPTOR_RANGE_WORD UINT32_MAX

static iree_string_view_t loom_amdgpu_hal_resource_string_or_empty(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (module == NULL || string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static iree_status_t loom_amdgpu_hal_resource_make_sgpr_type(
    loom_module_t* module, uint32_t unit_count, loom_type_t* out_type) {
  IREE_ASSERT_ARGUMENT(out_type);
  *out_type = loom_type_none();
  loom_string_id_t sgpr_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, IREE_SV("amdgpu.sgpr"), &sgpr_id));
  return loom_module_intern_type(
      module, loom_type_register(sgpr_id, unit_count), out_type);
}

static bool loom_amdgpu_hal_resource_type_is_sgpr_range(
    const loom_module_t* module, loom_type_t type, uint32_t unit_count) {
  if (!loom_type_is_register(type) ||
      loom_type_register_unit_count(type) != unit_count) {
    return false;
  }
  iree_string_view_t register_class = loom_amdgpu_hal_resource_string_or_empty(
      module, loom_type_register_class_id(type));
  return iree_string_view_equal(register_class, IREE_SV("amdgpu.sgpr"));
}

static iree_status_t loom_amdgpu_hal_resource_verify_sgpr_range_value(
    const loom_module_t* module, loom_value_id_t value_id, uint32_t unit_count,
    iree_string_view_t label) {
  if (value_id >= module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HAL resource %.*s value id %" PRIu32
                            " is invalid",
                            (int)label.size, label.data, value_id);
  }
  loom_type_t type = loom_module_value_type(module, value_id);
  if (!loom_amdgpu_hal_resource_type_is_sgpr_range(module, type, unit_count)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL resource %.*s must have type reg<amdgpu.sgpr x%" PRIu32 ">",
        (int)label.size, label.data, unit_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_resource_find_kernarg_live_in(
    loom_module_t* module, loom_op_t* function_op, loom_value_id_t* out_value,
    loom_op_t** out_insert_before) {
  IREE_ASSERT_ARGUMENT(out_value);
  IREE_ASSERT_ARGUMENT(out_insert_before);
  *out_value = LOOM_VALUE_ID_INVALID;
  *out_insert_before = NULL;

  loom_region_t* body = loom_low_func_def_body(function_op);
  if (body == NULL || body->block_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL resource materialization requires a function body");
  }
  loom_block_t* entry_block = loom_region_entry_block(body);
  bool in_live_in_prefix = true;
  loom_op_t* op = NULL;
  loom_block_for_each_op(entry_block, op) {
    if (loom_low_live_in_isa(op)) {
      iree_string_view_t source = loom_amdgpu_hal_resource_string_or_empty(
          module, loom_low_live_in_source(op));
      if (iree_string_view_equal(
              source,
              IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_KERNARG_SEGMENT_PTR_SOURCE))) {
        if (!in_live_in_prefix) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "AMDGPU HAL kernarg segment pointer live-in must be in the "
              "entry live-in prefix");
        }
        loom_value_id_t result = loom_low_live_in_result(op);
        IREE_RETURN_IF_ERROR(loom_amdgpu_hal_resource_verify_sgpr_range_value(
            module, result, 2, IREE_SV("kernarg live-in")));
        if (*out_value != LOOM_VALUE_ID_INVALID) {
          return iree_make_status(
              IREE_STATUS_ALREADY_EXISTS,
              "AMDGPU HAL resource materialization found duplicate kernarg "
              "segment pointer live-ins");
        }
        *out_value = result;
      }
      continue;
    }
    if (in_live_in_prefix) {
      in_live_in_prefix = false;
      *out_insert_before = op;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_resource_ensure_kernarg_live_in(
    loom_rewriter_t* rewriter, loom_op_t* function_op, loom_type_t sgpr_x2_type,
    loom_value_id_t* out_value, bool* out_inserted) {
  IREE_ASSERT_ARGUMENT(out_value);
  IREE_ASSERT_ARGUMENT(out_inserted);
  *out_value = LOOM_VALUE_ID_INVALID;
  *out_inserted = false;

  loom_op_t* insert_before = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_resource_find_kernarg_live_in(
      rewriter->module, function_op, out_value, &insert_before));
  if (*out_value != LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }

  loom_string_id_t source_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      rewriter->module,
      IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_KERNARG_SEGMENT_PTR_SOURCE),
      &source_id));
  if (insert_before != NULL) {
    loom_builder_set_before(&rewriter->builder, insert_before);
  } else {
    loom_builder_set_block(
        &rewriter->builder,
        loom_region_entry_block(loom_low_func_def_body(function_op)));
  }
  loom_op_t* live_in_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_live_in_build(
      &rewriter->builder, source_id, loom_make_named_attr_slice(NULL, 0),
      sgpr_x2_type, function_op->location, &live_in_op));
  *out_value = loom_low_live_in_result(live_in_op);
  *out_inserted = true;
  return iree_ok_status();
}

static const loom_amdgpu_hal_kernarg_resource_t*
loom_amdgpu_hal_resource_lookup_layout_resource(
    const loom_amdgpu_hal_kernel_abi_layout_t* layout, const loom_op_t* op) {
  for (iree_host_size_t i = 0; i < layout->resource_count; ++i) {
    const loom_amdgpu_hal_kernarg_resource_t* resource = &layout->resources[i];
    if (resource->resource_op == op) {
      return resource;
    }
  }
  return NULL;
}

static iree_status_t loom_amdgpu_hal_resource_i64_attr(
    loom_module_t* module, iree_string_view_t name, int64_t value,
    loom_named_attr_t* out_attr) {
  IREE_ASSERT_ARGUMENT(out_attr);
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, name, &name_id));
  *out_attr = (loom_named_attr_t){
      .name_id = name_id,
      .value = loom_attr_i64(value),
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_resource_build_s_mov_b32(
    loom_rewriter_t* rewriter, int64_t value, loom_type_t sgpr_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      rewriter->module, IREE_SV("amdgpu.s_mov_b32"), &opcode_id));
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_resource_i64_attr(
      rewriter->module, IREE_SV("imm32"), value, &attr));
  loom_op_t* const_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_const_build(
      &rewriter->builder, opcode_id, loom_make_named_attr_slice(&attr, 1),
      sgpr_type, location, &const_op));
  *out_value = loom_low_const_result(const_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_resource_build_pointer_load(
    loom_rewriter_t* rewriter, loom_value_id_t kernarg_ptr,
    loom_value_id_t soffset, uint32_t kernarg_offset, loom_type_t sgpr_x2_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      rewriter->module, IREE_SV("amdgpu.s_load_dwordx2"), &opcode_id));
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_resource_i64_attr(
      rewriter->module, IREE_SV("offset"), kernarg_offset, &attr));
  const loom_value_id_t operands[] = {kernarg_ptr, soffset};
  const loom_type_t result_types[] = {sgpr_x2_type};
  loom_op_t* load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_op_build(
      &rewriter->builder, opcode_id, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(&attr, 1), result_types,
      IREE_ARRAYSIZE(result_types), NULL, 0, location, &load_op));
  *out_value = loom_low_op_results(load_op).values[0];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_resource_materialize_one(
    loom_rewriter_t* rewriter, loom_op_t* op,
    const loom_amdgpu_hal_kernarg_resource_t* resource,
    const loom_target_bundle_t* target_bundle, loom_value_id_t kernarg_ptr,
    loom_type_t sgpr_type, loom_type_t sgpr_x2_type) {
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_builder_set_before(&rewriter->builder, op);

  loom_value_id_t soffset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_resource_build_s_mov_b32(
      rewriter, 0, sgpr_type, op->location, &soffset));
  loom_value_id_t pointer = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_resource_build_pointer_load(
      rewriter, kernarg_ptr, soffset, resource->kernarg_offset, sgpr_x2_type,
      op->location, &pointer));
  loom_value_id_t range = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_resource_build_s_mov_b32(
      rewriter, LOOM_AMDGPU_HAL_RESOURCE_DESCRIPTOR_RANGE_WORD, sgpr_type,
      op->location, &range));
  loom_value_id_t flags = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_resource_build_s_mov_b32(
      rewriter, target_bundle->export_plan->hal_kernel.buffer_resource_flags,
      sgpr_type, op->location, &flags));

  const loom_value_id_t sources[] = {pointer, range, flags};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      &rewriter->builder, sources, IREE_ARRAYSIZE(sources),
      loom_module_value_type(rewriter->module, loom_low_resource_result(op)),
      op->location, &concat_op));

  loom_value_id_t replacement = loom_low_concat_result(concat_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

iree_status_t loom_amdgpu_hal_resource_materialize(
    loom_module_t* module, loom_op_t* function_op,
    const loom_target_bundle_t* target_bundle,
    loom_amdgpu_hal_resource_materialization_result_t* out_result,
    iree_arena_allocator_t* scratch_arena) {
  IREE_ASSERT_ARGUMENT(out_result);
  *out_result = (loom_amdgpu_hal_resource_materialization_result_t){0};
  if (module == NULL || function_op == NULL || scratch_arena == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL resource materialization requires a module, function, and "
        "scratch arena");
  }
  if (!loom_low_func_def_isa(function_op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL resource materialization requires low.func.def");
  }

  loom_amdgpu_hal_kernel_abi_layout_t layout = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_layout_from_low(
      module, function_op, target_bundle, &layout, scratch_arena));
  out_result->abi_layout = layout;
  if (layout.resource_count == 0) {
    return iree_ok_status();
  }

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_resource_make_sgpr_type(module, 1, &sgpr_type));
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_resource_make_sgpr_type(module, 2, &sgpr_x2_type));

  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, scratch_arena));
  iree_status_t status = iree_ok_status();
  loom_value_id_t kernarg_ptr = LOOM_VALUE_ID_INVALID;
  bool inserted_live_in = false;

  loom_region_t* body = loom_low_func_def_body(function_op);
  for (uint16_t block_index = 0;
       iree_status_is_ok(status) && block_index < body->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(body, block_index);
    loom_op_t* op = block->first_op;
    while (iree_status_is_ok(status) && op != NULL) {
      loom_op_t* next_op = op->next_op;
      if (loom_low_resource_isa(op)) {
        const loom_amdgpu_hal_kernarg_resource_t* resource =
            loom_amdgpu_hal_resource_lookup_layout_resource(&layout, op);
        if (resource == NULL) {
          status = iree_make_status(
              IREE_STATUS_NOT_FOUND,
              "AMDGPU HAL resource materialization found low.resource "
              "without a matching ABI layout entry");
        } else {
          if (kernarg_ptr == LOOM_VALUE_ID_INVALID) {
            status = loom_amdgpu_hal_resource_ensure_kernarg_live_in(
                &rewriter, function_op, sgpr_x2_type, &kernarg_ptr,
                &inserted_live_in);
          }
          if (iree_status_is_ok(status)) {
            status = loom_amdgpu_hal_resource_materialize_one(
                &rewriter, op, resource, target_bundle, kernarg_ptr, sgpr_type,
                sgpr_x2_type);
          }
          if (iree_status_is_ok(status)) {
            ++out_result->materialized_resource_count;
          }
        }
      }
      op = next_op;
    }
  }

  out_result->inserted_kernarg_segment_ptr_live_in = inserted_live_in;
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
