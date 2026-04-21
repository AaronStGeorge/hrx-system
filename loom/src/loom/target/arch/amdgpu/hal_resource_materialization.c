// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/hal_resource_materialization.h"

#include <inttypes.h>

#include "loom/codegen/low/builder.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/hal_kernel_abi.h"

static iree_string_view_t loom_amdgpu_hal_resource_string_or_empty(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (module == NULL || string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static iree_status_t loom_amdgpu_hal_resource_make_sgpr_type(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t unit_count, loom_type_t* out_type) {
  IREE_ASSERT_ARGUMENT(out_type);
  *out_type = loom_type_none();
  loom_type_t type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      module, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, unit_count,
      &type));
  return loom_module_intern_type(module, type, out_type);
}

static iree_status_t loom_amdgpu_hal_resource_type_is_sgpr_range(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    loom_type_t type, uint32_t unit_count, bool* out_match) {
  IREE_ASSERT_ARGUMENT(out_match);
  *out_match = false;
  if (!loom_type_is_register(type) ||
      loom_type_register_unit_count(type) != unit_count) {
    return iree_ok_status();
  }
  loom_string_id_t expected_class_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_build_register_class_string_id(
      module, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
      &expected_class_id));
  *out_match = loom_type_register_class_id(type) == expected_class_id;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_resource_verify_sgpr_range_value(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t value_id, uint32_t unit_count, iree_string_view_t label) {
  if (value_id >= module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HAL resource %.*s value id %" PRIu32
                            " is invalid",
                            (int)label.size, label.data, value_id);
  }
  loom_type_t type = loom_module_value_type(module, value_id);
  bool is_sgpr_range = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_resource_type_is_sgpr_range(
      module, descriptor_set, type, unit_count, &is_sgpr_range));
  if (!is_sgpr_range) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL resource %.*s must have type reg<amdgpu.sgpr x%" PRIu32 ">",
        (int)label.size, label.data, unit_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_resource_descriptor_range_word(
    const loom_op_t* resource_op, uint32_t* out_range_word) {
  IREE_ASSERT_ARGUMENT(out_range_word);
  *out_range_word = UINT32_MAX;
  loom_attribute_t attr =
      loom_op_attrs(resource_op)[loom_low_resource_valid_byte_count_ATTR_INDEX];
  if (loom_attr_is_absent(attr)) {
    return iree_ok_status();
  }
  const int64_t valid_byte_count =
      loom_low_resource_valid_byte_count(resource_op);
  if (valid_byte_count < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL resource valid_byte_count must be non-negative");
  }
  if (valid_byte_count > UINT32_MAX) {
    return iree_ok_status();
  }
  *out_range_word = (uint32_t)valid_byte_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_resource_find_kernarg_live_in(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    loom_op_t* function_op, loom_value_id_t* out_value,
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
            module, descriptor_set, result, 2, IREE_SV("kernarg live-in")));
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
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    loom_op_t* function_op, loom_type_t sgpr_x2_type,
    loom_value_id_t* out_value, bool* out_inserted) {
  IREE_ASSERT_ARGUMENT(out_value);
  IREE_ASSERT_ARGUMENT(out_inserted);
  *out_value = LOOM_VALUE_ID_INVALID;
  *out_inserted = false;

  loom_op_t* insert_before = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_resource_find_kernarg_live_in(
      rewriter->module, descriptor_set, function_op, out_value,
      &insert_before));
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
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    int64_t value, loom_type_t sgpr_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_resource_i64_attr(
      rewriter->module, IREE_SV("imm32"), value, &attr));
  loom_op_t* const_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_descriptor_const(
      &rewriter->builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_ID_S_MOV_B32,
      loom_make_named_attr_slice(&attr, 1), sgpr_type, location, &const_op));
  *out_value = loom_low_const_result(const_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_resource_build_pointer_load(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t kernarg_ptr, loom_value_id_t soffset,
    uint32_t kernarg_offset, loom_type_t sgpr_x2_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_resource_i64_attr(
      rewriter->module, IREE_SV("offset"), kernarg_offset, &attr));
  const loom_value_id_t operands[] = {kernarg_ptr, soffset};
  const loom_type_t result_types[] = {sgpr_x2_type};
  loom_op_t* load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_descriptor_op(
      &rewriter->builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_ID_S_LOAD_DWORDX2, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(&attr, 1),
      result_types, IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &load_op));
  *out_value = loom_low_op_results(load_op).values[0];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_resource_materialize_one(
    loom_rewriter_t* rewriter, loom_op_t* op,
    const loom_amdgpu_hal_kernarg_resource_t* resource,
    const loom_target_bundle_t* target_bundle, loom_value_id_t kernarg_ptr,
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t sgpr_type,
    loom_type_t sgpr_x2_type) {
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_builder_set_before(&rewriter->builder, op);

  loom_value_id_t soffset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_resource_build_s_mov_b32(
      rewriter, descriptor_set, 0, sgpr_type, op->location, &soffset));
  loom_value_id_t pointer = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_resource_build_pointer_load(
      rewriter, descriptor_set, kernarg_ptr, soffset, resource->kernarg_offset,
      sgpr_x2_type, op->location, &pointer));
  uint32_t range_word = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_resource_descriptor_range_word(op, &range_word));
  loom_value_id_t range = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_resource_build_s_mov_b32(
      rewriter, descriptor_set, range_word, sgpr_type, op->location, &range));
  loom_value_id_t flags = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_resource_build_s_mov_b32(
      rewriter, descriptor_set,
      target_bundle->export_plan->hal_kernel.buffer_resource_flags, sgpr_type,
      op->location, &flags));

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
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_hal_resource_materialization_result_t* out_result,
    iree_arena_allocator_t* scratch_arena) {
  IREE_ASSERT_ARGUMENT(out_result);
  *out_result = (loom_amdgpu_hal_resource_materialization_result_t){0};
  if (module == NULL || function_op == NULL || target_bundle == NULL ||
      descriptor_set == NULL || scratch_arena == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL resource materialization requires a module, function, "
        "target bundle, descriptor set, and scratch arena");
  }
  if (!loom_low_func_def_isa(function_op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL resource materialization requires low.func.def");
  }

  loom_amdgpu_hal_kernel_abi_layout_t layout = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_layout_from_low(
      module, function_op, target_bundle, descriptor_set, &layout,
      scratch_arena));
  out_result->abi_layout = layout;
  if (layout.resource_count == 0) {
    return iree_ok_status();
  }

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_resource_make_sgpr_type(
      module, descriptor_set, 1, &sgpr_type));
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_resource_make_sgpr_type(
      module, descriptor_set, 2, &sgpr_x2_type));

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
                &rewriter, descriptor_set, function_op, sgpr_x2_type,
                &kernarg_ptr, &inserted_live_in);
          }
          if (iree_status_is_ok(status)) {
            status = loom_amdgpu_hal_resource_materialize_one(
                &rewriter, op, resource, target_bundle, kernarg_ptr,
                descriptor_set, sgpr_type, sgpr_x2_type);
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
