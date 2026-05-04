// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/hal_binding_materialization.h"

#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/hal_binding_descriptor.h"
#include "loom/target/arch/amdgpu/hal_kernel_abi.h"
#include "loom/target/arch/amdgpu/target_info.h"

#define LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_POINTER_HIGH_MASK UINT32_C(0x0000FFFF)
#define LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_CACHE_SWIZZLE_ENABLE_BIT \
  UINT32_C(0x4000)
#define LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_CACHE_SWIZZLE_WORD_SHIFT 16u

static iree_status_t loom_amdgpu_hal_binding_make_sgpr_type(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t unit_count, loom_type_t* out_type) {
  *out_type = loom_type_none();
  loom_type_t type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      module, descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, unit_count,
      &type));
  return loom_module_intern_type(module, type, out_type);
}

static uint32_t loom_amdgpu_hal_binding_descriptor_range_word(
    int64_t valid_byte_count) {
  if (valid_byte_count > UINT32_MAX) {
    return UINT32_MAX;
  }
  return (uint32_t)valid_byte_count;
}

static iree_status_t loom_amdgpu_hal_binding_cache_swizzle_kind(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_buffer_resource_cache_swizzle_t* out_kind) {
  *out_kind = LOOM_AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_NONE;
  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_info_lookup_descriptor_set_by_ordinal(
      descriptor_set->descriptor_set_ordinal, &descriptor_set_info));
  *out_kind = descriptor_set_info->buffer_resource_cache_swizzle;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_binding_insert_kernarg_live_in(
    loom_rewriter_t* rewriter, loom_op_t* function_op, loom_type_t sgpr_x2_type,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_string_id_t source_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      rewriter->module,
      IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_KERNARG_SEGMENT_PTR_SOURCE),
      &source_id));
  loom_block_t* entry_block =
      loom_region_entry_block(loom_low_function_body(function_op));
  if (entry_block->first_op != NULL) {
    loom_builder_set_before(&rewriter->builder, entry_block->first_op);
  } else {
    loom_builder_set_block(&rewriter->builder, entry_block);
  }
  loom_op_t* live_in_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_live_in_build(
      &rewriter->builder, source_id, loom_make_named_attr_slice(NULL, 0),
      sgpr_x2_type, function_op->location, &live_in_op));
  *out_value = loom_low_live_in_result(live_in_op);
  return iree_ok_status();
}

static const loom_amdgpu_hal_kernarg_resource_t*
loom_amdgpu_hal_binding_lookup_layout_resource(
    const loom_amdgpu_hal_kernel_abi_layout_t* layout, const loom_op_t* op) {
  for (iree_host_size_t i = 0; i < layout->resource_count; ++i) {
    const loom_amdgpu_hal_kernarg_resource_t* resource = &layout->resources[i];
    if (resource->resource_op == op) {
      return resource;
    }
  }
  return NULL;
}

static iree_status_t loom_amdgpu_hal_binding_i64_attr(
    loom_module_t* module, iree_string_view_t name, int64_t value,
    loom_named_attr_t* out_attr) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, name, &name_id));
  *out_attr = (loom_named_attr_t){
      .name_id = name_id,
      .value = loom_attr_i64(value),
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_binding_build_s_mov_b32(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    int64_t value, loom_type_t sgpr_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_i64_attr(
      rewriter->module, IREE_SV("imm32"), value, &attr));
  loom_op_t* const_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_descriptor_const(
      &rewriter->builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_ID_S_MOV_B32,
      loom_make_named_attr_slice(&attr, 1), sgpr_type, location, &const_op));
  *out_value = loom_low_const_result(const_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_binding_build_low_slice(
    loom_rewriter_t* rewriter, loom_value_id_t source, uint32_t offset,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(&rewriter->builder, source, offset,
                                            result_type, location, &slice_op));
  *out_value = loom_low_slice_result(slice_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_binding_build_s_binary_b32(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    uint64_t descriptor_id, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {lhs, rhs};
  const loom_type_t result_types[] = {result_type};
  loom_op_t* binary_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_descriptor_op(
      &rewriter->builder, descriptor_set, descriptor_id, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0),
      result_types, IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &binary_op));
  *out_value = loom_value_slice_get(loom_low_op_results(binary_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_binding_build_pointer_load(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t kernarg_ptr, loom_value_id_t soffset,
    uint32_t kernarg_offset, loom_type_t sgpr_x2_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_i64_attr(
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

static iree_status_t loom_amdgpu_hal_binding_build_descriptor_pointer(
    loom_rewriter_t* rewriter, bool has_cache_swizzle,
    uint32_t cache_swizzle_stride,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t loaded_pointer, loom_type_t sgpr_type,
    loom_type_t sgpr_x2_type, loom_location_id_t location,
    loom_value_id_t* out_pointer) {
  *out_pointer = LOOM_VALUE_ID_INVALID;
  loom_value_id_t pointer_low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_low_slice(
      rewriter, loaded_pointer, 0, sgpr_type, location, &pointer_low));
  loom_value_id_t pointer_high = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_low_slice(
      rewriter, loaded_pointer, 1, sgpr_type, location, &pointer_high));

  loom_value_id_t pointer_high_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_mov_b32(
      rewriter, descriptor_set,
      LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_POINTER_HIGH_MASK, sgpr_type, location,
      &pointer_high_mask));
  loom_value_id_t masked_pointer_high = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_binary_b32(
      rewriter, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_ID_S_AND_B32,
      pointer_high, pointer_high_mask, sgpr_type, location,
      &masked_pointer_high));

  loom_value_id_t descriptor_pointer_high = masked_pointer_high;
  if (has_cache_swizzle) {
    loom_amdgpu_buffer_resource_cache_swizzle_t cache_swizzle_kind =
        LOOM_AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_NONE;
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_cache_swizzle_kind(
        descriptor_set, &cache_swizzle_kind));
    if (cache_swizzle_kind !=
        LOOM_AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "AMDGPU HAL binding materialization reached an unverified cache "
          "swizzle descriptor");
    }

    const uint32_t cache_swizzle_word =
        ((cache_swizzle_stride |
          LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_CACHE_SWIZZLE_ENABLE_BIT)
         << LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_CACHE_SWIZZLE_WORD_SHIFT);
    loom_value_id_t cache_swizzle = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_mov_b32(
        rewriter, descriptor_set, cache_swizzle_word, sgpr_type, location,
        &cache_swizzle));
    loom_value_id_t swizzled_pointer_high = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_binary_b32(
        rewriter, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_ID_S_OR_B32,
        masked_pointer_high, cache_swizzle, sgpr_type, location,
        &swizzled_pointer_high));
    descriptor_pointer_high = swizzled_pointer_high;
  }

  const loom_value_id_t sources[] = {pointer_low, descriptor_pointer_high};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      &rewriter->builder, sources, IREE_ARRAYSIZE(sources), sgpr_x2_type,
      location, &concat_op));
  *out_pointer = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_binding_materialize_one(
    loom_rewriter_t* rewriter, loom_op_t* op,
    const loom_amdgpu_hal_kernarg_resource_t* resource,
    loom_value_id_t kernarg_ptr,
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t sgpr_type,
    loom_type_t sgpr_x2_type) {
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_builder_set_before(&rewriter->builder, op);

  loom_value_id_t soffset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_mov_b32(
      rewriter, descriptor_set, 0, sgpr_type, op->location, &soffset));
  loom_value_id_t pointer = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_pointer_load(
      rewriter, descriptor_set, kernarg_ptr, soffset, resource->kernarg_offset,
      sgpr_x2_type, op->location, &pointer));

  loom_value_id_t replacement = pointer;
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

static iree_status_t
loom_amdgpu_hal_binding_materialize_buffer_descriptor_pseudo(
    loom_rewriter_t* rewriter, loom_op_t* op,
    const loom_target_bundle_t* target_bundle,
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t sgpr_type,
    loom_type_t sgpr_x2_type) {
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_builder_set_before(&rewriter->builder, op);

  loom_value_slice_t operands = loom_low_op_operands(op);
  loom_value_slice_t results = loom_low_op_results(op);
  loom_named_attr_slice_t attrs = loom_low_op_attrs(op);
  const int64_t cache_swizzle_stride_attr =
      attrs.entries[LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_ATTR_CACHE_SWIZZLE_STRIDE]
          .value.i64;
  const int64_t valid_byte_count =
      attrs.entries[LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_ATTR_VALID_BYTE_COUNT]
          .value.i64;
  const uint32_t cache_swizzle_stride = (uint32_t)cache_swizzle_stride_attr;
  const bool has_cache_swizzle = cache_swizzle_stride != 0;
  const uint32_t range_word =
      loom_amdgpu_hal_binding_descriptor_range_word(valid_byte_count);

  loom_value_id_t pointer = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_descriptor_pointer(
      rewriter, has_cache_swizzle, cache_swizzle_stride, descriptor_set,
      operands.values[0], sgpr_type, sgpr_x2_type, op->location, &pointer));
  loom_value_id_t range = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_mov_b32(
      rewriter, descriptor_set, range_word, sgpr_type, op->location, &range));
  loom_value_id_t flags = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_mov_b32(
      rewriter, descriptor_set,
      target_bundle->export_plan->hal_kernel.buffer_resource_flags, sgpr_type,
      op->location, &flags));

  const loom_value_id_t sources[] = {pointer, range, flags};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      &rewriter->builder, sources, IREE_ARRAYSIZE(sources),
      loom_module_value_type(rewriter->module, results.values[0]), op->location,
      &concat_op));

  loom_value_id_t replacement = loom_low_concat_result(concat_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

iree_status_t loom_amdgpu_hal_binding_materialize(
    loom_module_t* module, loom_op_t* function_op,
    const loom_target_bundle_t* target_bundle,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_hal_binding_materialization_result_t* out_result,
    iree_arena_allocator_t* scratch_arena) {
  *out_result = (loom_amdgpu_hal_binding_materialization_result_t){0};
  if (module == NULL || function_op == NULL || target_bundle == NULL ||
      descriptor_set == NULL || scratch_arena == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL binding materialization requires a module, function, "
        "target bundle, descriptor set, and scratch arena");
  }
  if (!loom_low_function_def_isa(function_op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL binding materialization requires low.func.def or "
        "low.kernel.def");
  }

  loom_amdgpu_hal_kernel_abi_layout_t layout = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_layout_from_low(
      module, function_op, &layout, scratch_arena));
  out_result->abi_layout = layout;

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_make_sgpr_type(
      module, descriptor_set, 1, &sgpr_type));
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_make_sgpr_type(
      module, descriptor_set, 2, &sgpr_x2_type));

  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, scratch_arena));
  iree_status_t status = iree_ok_status();
  loom_value_id_t kernarg_ptr = LOOM_VALUE_ID_INVALID;
  bool inserted_live_in = false;

  loom_region_t* body = loom_low_function_body(function_op);
  if (body == NULL) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU HAL binding materialization reached a low function without a "
        "body");
  }
  for (uint16_t block_index = 0;
       iree_status_is_ok(status) && block_index < body->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(body, block_index);
    loom_op_t* op = block->first_op;
    while (iree_status_is_ok(status) && op != NULL) {
      loom_op_t* next_op = op->next_op;
      if (loom_low_resource_isa(op)) {
        const loom_amdgpu_hal_kernarg_resource_t* resource =
            loom_amdgpu_hal_binding_lookup_layout_resource(&layout, op);
        if (resource == NULL) {
          status = iree_make_status(
              IREE_STATUS_INTERNAL,
              "AMDGPU HAL binding materialization found low.resource "
              "without a matching ABI layout entry");
        } else {
          if (kernarg_ptr == LOOM_VALUE_ID_INVALID) {
            status = loom_amdgpu_hal_binding_insert_kernarg_live_in(
                &rewriter, function_op, sgpr_x2_type, &kernarg_ptr);
            if (iree_status_is_ok(status)) {
              inserted_live_in = true;
            }
          }
          if (iree_status_is_ok(status)) {
            status = loom_amdgpu_hal_binding_materialize_one(
                &rewriter, op, resource, kernarg_ptr, descriptor_set, sgpr_type,
                sgpr_x2_type);
          }
          if (iree_status_is_ok(status)) {
            ++out_result->materialized_binding_count;
          }
        }
      } else if (loom_low_op_isa(op) &&
                 (uint64_t)loom_low_op_descriptor_id(op) ==
                     LOOM_AMDGPU_DESCRIPTOR_ID_HAL_BUFFER_DESCRIPTOR) {
        status = loom_amdgpu_hal_binding_materialize_buffer_descriptor_pseudo(
            &rewriter, op, target_bundle, descriptor_set, sgpr_type,
            sgpr_x2_type);
        if (iree_status_is_ok(status)) {
          ++out_result->materialized_descriptor_count;
        }
      }
      op = next_op;
    }
  }

  out_result->inserted_kernarg_segment_ptr_live_in = inserted_live_in;
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
