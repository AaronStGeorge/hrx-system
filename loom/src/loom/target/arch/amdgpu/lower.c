// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower.h"

#include <inttypes.h>
#include <stdint.h>
#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/gfx11_descriptors.h"
#include "loom/target/arch/amdgpu/gfx950_descriptors.h"
#include "loom/target/arch/amdgpu/hal_kernel_abi.h"
#include "loom/util/fact_table.h"

#define LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES 4u

static bool loom_amdgpu_type_is_i32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
}

static bool loom_amdgpu_type_is_address_scalar(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         (loom_type_element_type(type) == LOOM_SCALAR_TYPE_INDEX ||
          loom_type_element_type(type) == LOOM_SCALAR_TYPE_OFFSET);
}

static bool loom_amdgpu_type_is_f32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32;
}

static bool loom_amdgpu_type_is_vector_1xi32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32 &&
         loom_type_dim_static_size_at(type, 0) == 1;
}

static bool loom_amdgpu_type_is_vector_4xi32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static bool loom_amdgpu_type_is_vector_1xf32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32 &&
         loom_type_dim_static_size_at(type, 0) == 1;
}

static bool loom_amdgpu_type_is_vector_4xf32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static bool loom_amdgpu_type_is_vector_1x32_or_4x32(loom_type_t type) {
  return loom_amdgpu_type_is_vector_1xi32(type) ||
         loom_amdgpu_type_is_vector_4xi32(type) ||
         loom_amdgpu_type_is_vector_1xf32(type) ||
         loom_amdgpu_type_is_vector_4xf32(type);
}

static uint32_t loom_amdgpu_vector_32bit_lane_count(loom_type_t type) {
  if (loom_amdgpu_type_is_vector_1xi32(type) ||
      loom_amdgpu_type_is_vector_1xf32(type)) {
    return 1;
  }
  if (loom_amdgpu_type_is_vector_4xi32(type) ||
      loom_amdgpu_type_is_vector_4xf32(type)) {
    return 4;
  }
  return 0;
}

static uint32_t loom_amdgpu_vector_i32_lane_count(loom_type_t type) {
  if (loom_amdgpu_type_is_vector_1xi32(type)) {
    return 1;
  }
  if (loom_amdgpu_type_is_vector_4xi32(type)) {
    return 4;
  }
  return 0;
}

static uint32_t loom_amdgpu_vector_f32_lane_count(loom_type_t type) {
  if (loom_amdgpu_type_is_vector_1xf32(type)) {
    return 1;
  }
  if (loom_amdgpu_type_is_vector_4xf32(type)) {
    return 4;
  }
  return 0;
}

static bool loom_amdgpu_type_is_32bit_view(loom_type_t type) {
  return loom_type_is_view(type) &&
         (loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32 ||
          loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32);
}

static bool loom_amdgpu_value_is_i32(loom_low_lower_context_t* context,
                                     loom_value_id_t value_id) {
  return loom_amdgpu_type_is_i32(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

static bool loom_amdgpu_value_is_address_scalar(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  return loom_amdgpu_type_is_address_scalar(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

static bool loom_amdgpu_value_is_f32(loom_low_lower_context_t* context,
                                     loom_value_id_t value_id) {
  return loom_amdgpu_type_is_f32(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

static bool loom_amdgpu_value_is_vector_1x32_or_4x32(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  return loom_amdgpu_type_is_vector_1x32_or_4x32(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

static bool loom_amdgpu_value_is_32bit_view(loom_low_lower_context_t* context,
                                            loom_value_id_t value_id) {
  return loom_amdgpu_type_is_32bit_view(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

static iree_status_t loom_amdgpu_make_register_type(
    loom_low_lower_context_t* context, uint16_t reg_class_id,
    uint32_t unit_count, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(context, reg_class_id, unit_count,
                                           out_type);
}

static iree_status_t loom_amdgpu_make_sgpr_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_amdgpu_make_register_type(
      context, AMDGPU_GFX11_CORE_REG_CLASS_ID_AMDGPU_SGPR, 1, out_type);
}

static iree_status_t loom_amdgpu_make_sgpr_range_type(
    loom_low_lower_context_t* context, uint32_t unit_count,
    loom_type_t* out_type) {
  return loom_amdgpu_make_register_type(
      context, AMDGPU_GFX11_CORE_REG_CLASS_ID_AMDGPU_SGPR, unit_count,
      out_type);
}

static iree_status_t loom_amdgpu_make_vgpr_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_amdgpu_make_register_type(
      context, AMDGPU_GFX11_CORE_REG_CLASS_ID_AMDGPU_VGPR, 1, out_type);
}

static iree_status_t loom_amdgpu_low_type_register_class_is(
    loom_low_lower_context_t* context, loom_type_t type, uint16_t reg_class_id,
    bool* out_match) {
  IREE_ASSERT_ARGUMENT(out_match);
  *out_match = false;
  if (!loom_type_is_register(type)) {
    return iree_ok_status();
  }
  loom_string_id_t expected_class_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_register_class_string_id(
      context, reg_class_id, &expected_class_id));
  *out_match = loom_type_register_class_id(type) == expected_class_id;
  return iree_ok_status();
}

static bool loom_amdgpu_value_prefers_vgpr(loom_low_lower_context_t* context,
                                           loom_value_id_t source_value_id) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (source_value_id >= module->values.count) {
    return false;
  }

  loom_type_t source_type = loom_module_value_type(module, source_value_id);
  if (loom_amdgpu_type_is_f32(source_type) ||
      loom_amdgpu_type_is_vector_1x32_or_4x32(source_type)) {
    return true;
  }

  const loom_value_t* value = loom_module_value(module, source_value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL) {
    return false;
  }

  switch (defining_op->kind) {
    case LOOM_OP_KERNEL_WORKITEM_ID:
      return loom_kernel_workitem_id_dimension(defining_op) ==
             LOOM_KERNEL_WORKITEM_ID_DIMENSION_X;
    case LOOM_OP_VECTOR_EXTRACT:
      return true;
    case LOOM_OP_SCALAR_ADDI:
      return loom_amdgpu_value_prefers_vgpr(
                 context, loom_scalar_addi_lhs(defining_op)) ||
             loom_amdgpu_value_prefers_vgpr(context,
                                            loom_scalar_addi_rhs(defining_op));
    case LOOM_OP_SCALAR_SUBI:
      return loom_amdgpu_value_prefers_vgpr(
                 context, loom_scalar_subi_lhs(defining_op)) ||
             loom_amdgpu_value_prefers_vgpr(context,
                                            loom_scalar_subi_rhs(defining_op));
    case LOOM_OP_SCALAR_MULI:
      return loom_amdgpu_value_prefers_vgpr(
                 context, loom_scalar_muli_lhs(defining_op)) ||
             loom_amdgpu_value_prefers_vgpr(context,
                                            loom_scalar_muli_rhs(defining_op));
    case LOOM_OP_SCALAR_ANDI:
      return loom_amdgpu_value_prefers_vgpr(
                 context, loom_scalar_andi_lhs(defining_op)) ||
             loom_amdgpu_value_prefers_vgpr(context,
                                            loom_scalar_andi_rhs(defining_op));
    case LOOM_OP_SCALAR_ORI:
      return loom_amdgpu_value_prefers_vgpr(context,
                                            loom_scalar_ori_lhs(defining_op)) ||
             loom_amdgpu_value_prefers_vgpr(context,
                                            loom_scalar_ori_rhs(defining_op));
    case LOOM_OP_SCALAR_XORI:
      return loom_amdgpu_value_prefers_vgpr(
                 context, loom_scalar_xori_lhs(defining_op)) ||
             loom_amdgpu_value_prefers_vgpr(context,
                                            loom_scalar_xori_rhs(defining_op));
    case LOOM_OP_SCALAR_SHLI:
      return loom_amdgpu_value_prefers_vgpr(
                 context, loom_scalar_shli_lhs(defining_op)) ||
             loom_amdgpu_value_prefers_vgpr(context,
                                            loom_scalar_shli_rhs(defining_op));
    case LOOM_OP_SCALAR_SHRSI:
      return loom_amdgpu_value_prefers_vgpr(
                 context, loom_scalar_shrsi_lhs(defining_op)) ||
             loom_amdgpu_value_prefers_vgpr(context,
                                            loom_scalar_shrsi_rhs(defining_op));
    case LOOM_OP_SCALAR_SHRUI:
      return loom_amdgpu_value_prefers_vgpr(
                 context, loom_scalar_shrui_lhs(defining_op)) ||
             loom_amdgpu_value_prefers_vgpr(context,
                                            loom_scalar_shrui_rhs(defining_op));
    default:
      return false;
  }
}

static iree_status_t loom_amdgpu_map_type(void* user_data,
                                          loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_type_t source_type,
                                          loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_amdgpu_type_is_i32(source_type)) {
    return loom_amdgpu_make_sgpr_type(context, out_low_type);
  }
  if (loom_amdgpu_type_is_address_scalar(source_type)) {
    return loom_amdgpu_make_sgpr_type(context, out_low_type);
  }
  if (loom_amdgpu_type_is_vector_1xi32(source_type) ||
      loom_amdgpu_type_is_vector_1xf32(source_type)) {
    return loom_amdgpu_make_vgpr_type(context, out_low_type);
  }
  if (loom_amdgpu_type_is_vector_4xi32(source_type) ||
      loom_amdgpu_type_is_vector_4xf32(source_type)) {
    return loom_amdgpu_make_register_type(
        context, AMDGPU_GFX11_CORE_REG_CLASS_ID_AMDGPU_VGPR, 4, out_low_type);
  }
  return loom_low_lower_emit_reject(
      context, source_op, IREE_SV("type"), IREE_SV("source"),
      IREE_SV("AMDGPU lowering currently supports only i32 scalar values, "
              "address scalar values, vector<1xi32>/vector<1xf32> VGPR "
              "values, and vector<4xi32>/vector<4xf32> VGPR ranges"));
}

static iree_status_t loom_amdgpu_map_value(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_value_id_t source_value_id,
                                           loom_type_t source_type,
                                           loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_amdgpu_type_is_f32(source_type)) {
    return loom_amdgpu_make_vgpr_type(context, out_low_type);
  }
  if ((loom_amdgpu_type_is_i32(source_type) ||
       loom_amdgpu_type_is_address_scalar(source_type)) &&
      loom_amdgpu_value_prefers_vgpr(context, source_value_id)) {
    return loom_amdgpu_make_vgpr_type(context, out_low_type);
  }
  return loom_amdgpu_map_type(user_data, context, source_op, source_type,
                              out_low_type);
}

static iree_status_t loom_amdgpu_make_hal_buffer_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  loom_string_id_t hal_buffer_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(loom_low_lower_context_module(context),
                                IREE_SV("hal.buffer"), &hal_buffer_id));
  *out_type = loom_type_dialect_opaque(hal_buffer_id);
  return iree_ok_status();
}

static uint32_t loom_amdgpu_hal_buffer_resource_index(
    loom_low_lower_context_t* context, uint16_t source_argument_index) {
  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids = loom_func_like_arg_ids(
      loom_low_lower_context_source_function(context), &argument_count);
  uint32_t resource_index = 0;
  for (uint16_t i = 0; i < source_argument_index && i < argument_count; ++i) {
    loom_type_t type = loom_module_value_type(
        loom_low_lower_context_module(context), argument_ids[i]);
    if (loom_type_is_buffer(type)) {
      ++resource_index;
    }
  }
  return resource_index;
}

static bool loom_amdgpu_module_value_as_exact_index_constant(
    const loom_module_t* module, loom_value_id_t value_id, int64_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op || !loom_index_constant_isa(defining_op)) {
    return false;
  }
  loom_attribute_t attr = loom_index_constant_value(defining_op);
  if (attr.kind != LOOM_ATTR_I64) {
    return false;
  }
  *out_value = attr.i64;
  return true;
}

typedef struct loom_amdgpu_buffer_argument_extent_t {
  // Module containing the source function being inspected.
  const loom_module_t* module;
  // True once a buffer.view derived from the source argument is found.
  bool found_view;
  // True when a derived view has no exact static dense byte extent.
  bool found_unbounded_view;
  // Maximum byte count required by all statically boundable derived views.
  int64_t valid_byte_count;
} loom_amdgpu_buffer_argument_extent_t;

static bool loom_amdgpu_view_static_dense_byte_extent(
    const loom_module_t* module, loom_type_t view_type,
    int64_t* out_byte_extent) {
  IREE_ASSERT_ARGUMENT(out_byte_extent);
  *out_byte_extent = 0;
  if (!loom_type_is_view(view_type)) {
    return false;
  }

  loom_value_facts_t stride_storage[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {
      0};
  loom_value_fact_address_layout_t layout = {0};
  if (!loom_encoding_query_type_address_layout(
          /*context=*/NULL, module, view_type, stride_storage,
          IREE_ARRAYSIZE(stride_storage), &layout) ||
      layout.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE) {
    return false;
  }

  const int32_t element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(view_type));
  if (element_bit_count <= 0 || (element_bit_count % 8) != 0) {
    return false;
  }

  int64_t element_count = 1;
  const uint8_t rank = loom_type_rank(view_type);
  for (uint8_t i = 0; i < rank; ++i) {
    if (loom_type_dim_is_dynamic_at(view_type, i)) {
      return false;
    }
    const int64_t dim_size = loom_type_dim_static_size_at(view_type, i);
    if (dim_size < 0 ||
        !iree_checked_mul_i64(element_count, dim_size, &element_count)) {
      return false;
    }
  }

  const int64_t element_byte_count = element_bit_count / 8;
  return iree_checked_mul_i64(element_count, element_byte_count,
                              out_byte_extent);
}

static void loom_amdgpu_buffer_argument_extent_include_view(
    loom_amdgpu_buffer_argument_extent_t* state,
    const loom_op_t* buffer_view_op) {
  if (state->found_unbounded_view) {
    return;
  }
  state->found_view = true;
  int64_t base_byte_offset = 0;
  int64_t view_byte_extent = 0;
  int64_t valid_byte_count = 0;
  if (!loom_amdgpu_module_value_as_exact_index_constant(
          state->module, loom_buffer_view_byte_offset(buffer_view_op),
          &base_byte_offset) ||
      base_byte_offset < 0 ||
      !loom_amdgpu_view_static_dense_byte_extent(
          state->module,
          loom_module_value_type(state->module,
                                 loom_buffer_view_result(buffer_view_op)),
          &view_byte_extent) ||
      !iree_checked_add_i64(base_byte_offset, view_byte_extent,
                            &valid_byte_count)) {
    state->found_unbounded_view = true;
    return;
  }
  state->valid_byte_count = iree_max(state->valid_byte_count, valid_byte_count);
}

static bool loom_amdgpu_source_buffer_argument_valid_byte_count(
    loom_low_lower_context_t* context, loom_value_id_t source_argument_id,
    int64_t* out_valid_byte_count) {
  IREE_ASSERT_ARGUMENT(out_valid_byte_count);
  *out_valid_byte_count = 0;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_t* source_argument =
      loom_module_value(module, source_argument_id);
  loom_amdgpu_buffer_argument_extent_t state = {
      .module = module,
  };
  const loom_use_t* use = NULL;
  loom_value_for_each_use(source_argument, use) {
    const loom_op_t* user_op = loom_use_user_op(*use);
    if (loom_use_operand_index(*use) == 0 && loom_buffer_view_isa(user_op)) {
      loom_amdgpu_buffer_argument_extent_include_view(&state, user_op);
    } else {
      state.found_unbounded_view = true;
    }
  }
  if (!state.found_view || state.found_unbounded_view) {
    return false;
  }
  *out_valid_byte_count = state.valid_byte_count;
  return true;
}

static iree_status_t loom_amdgpu_map_argument(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  (void)user_data;
  loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_argument_id);
  const loom_target_bundle_t* bundle = loom_low_lower_context_bundle(context);
  if (bundle->export_plan->abi_kind == LOOM_TARGET_ABI_HAL_KERNEL &&
      loom_type_is_buffer(source_type)) {
    loom_type_t resource_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_sgpr_range_type(context, 4, &resource_type));
    loom_type_t semantic_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_hal_buffer_type(context, &semantic_type));
    loom_low_resource_build_flags_t resource_build_flags = 0;
    int64_t resource_valid_byte_count = 0;
    if (loom_amdgpu_source_buffer_argument_valid_byte_count(
            context, source_argument_id, &resource_valid_byte_count)) {
      resource_build_flags |= LOOM_LOW_RESOURCE_BUILD_FLAG_HAS_VALID_BYTE_COUNT;
    }
    *out_argument = (loom_low_lower_abi_argument_t){
        .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE,
        .abi_type = resource_type,
        .resource_import_kind =
            LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BUFFER_RESOURCE,
        .resource_index = loom_amdgpu_hal_buffer_resource_index(
            context, source_argument_index),
        .resource_semantic_type = semantic_type,
        .resource_build_flags = resource_build_flags,
        .resource_valid_byte_count = resource_valid_byte_count,
    };
    return iree_ok_status();
  }

  *out_argument = (loom_low_lower_abi_argument_t){
      .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
      .abi_type = loom_type_none(),
      .resource_semantic_type = loom_type_none(),
  };
  return loom_amdgpu_map_value(user_data, context, source_function_op,
                               source_argument_id, source_type,
                               &out_argument->abi_type);
}

static bool loom_amdgpu_attr_is_i32_immediate(loom_attribute_t value) {
  return value.kind == LOOM_ATTR_I64 && value.i64 >= INT32_MIN &&
         value.i64 <= INT32_MAX;
}

static bool loom_amdgpu_attr_is_f32_immediate(loom_attribute_t value) {
  return value.kind == LOOM_ATTR_F64;
}

static uint32_t loom_amdgpu_attr_f32_bit_pattern(loom_attribute_t value) {
  const float f32_value = (float)loom_attr_as_f64(value);
  uint32_t bit_pattern = 0;
  memcpy(&bit_pattern, &f32_value, sizeof(bit_pattern));
  return bit_pattern;
}

static bool loom_amdgpu_value_as_i32_constant(loom_low_lower_context_t* context,
                                              loom_value_id_t value_id,
                                              int64_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;
  if (!loom_amdgpu_value_is_i32(context, value_id)) {
    return false;
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op || !loom_scalar_constant_isa(defining_op)) {
    return false;
  }
  loom_attribute_t attr = loom_scalar_constant_value(defining_op);
  if (!loom_amdgpu_attr_is_i32_immediate(attr)) {
    return false;
  }
  *out_value = attr.i64;
  return true;
}

static bool loom_amdgpu_value_can_materialize_as_vgpr_i32(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  int64_t unused_value = 0;
  return loom_amdgpu_value_prefers_vgpr(context, value_id) ||
         loom_amdgpu_value_as_i32_constant(context, value_id, &unused_value);
}

static bool loom_amdgpu_iota_i32_lane_value(int64_t base, int64_t step,
                                            uint32_t lane, int64_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;
  int64_t scaled_step = 0;
  if (!iree_checked_mul_i64((int64_t)lane, step, &scaled_step)) {
    return false;
  }
  int64_t value = 0;
  if (!iree_checked_add_i64(base, scaled_step, &value) || value < INT32_MIN ||
      value > INT32_MAX) {
    return false;
  }
  *out_value = value;
  return true;
}

static bool loom_amdgpu_iota_i32_lanes_fit(int64_t base, int64_t step,
                                           uint32_t lane_count) {
  for (uint32_t i = 0; i < lane_count; ++i) {
    int64_t unused = 0;
    if (!loom_amdgpu_iota_i32_lane_value(base, step, i, &unused)) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_can_lower_scalar_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_value_id_t result = loom_scalar_constant_result(source_op);
  const loom_attribute_t value = loom_scalar_constant_value(source_op);
  if (loom_amdgpu_value_is_i32(context, result)) {
    return loom_amdgpu_attr_is_i32_immediate(value);
  }
  if (loom_amdgpu_value_is_f32(context, result)) {
    return loom_amdgpu_attr_is_f32_immediate(value);
  }
  return false;
}

static bool loom_amdgpu_can_lower_index_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_value_is_address_scalar(
             context, loom_index_constant_result(source_op)) &&
         loom_amdgpu_attr_is_i32_immediate(
             loom_index_constant_value(source_op));
}

static bool loom_amdgpu_can_lower_vector_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t result_type =
      loom_module_value_type(module, loom_vector_constant_result(source_op));
  const loom_attribute_t value = loom_vector_constant_value(source_op);
  if (loom_amdgpu_vector_i32_lane_count(result_type) != 0) {
    return loom_amdgpu_attr_is_i32_immediate(value);
  }
  if (loom_amdgpu_vector_f32_lane_count(result_type) != 0) {
    return loom_amdgpu_attr_is_f32_immediate(value);
  }
  return false;
}

static bool loom_amdgpu_can_lower_vector_iota(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const uint32_t lane_count = loom_amdgpu_vector_i32_lane_count(
      loom_module_value_type(module, loom_vector_iota_result(source_op)));
  int64_t base = 0;
  int64_t step = 0;
  return lane_count != 0 &&
         loom_amdgpu_value_as_i32_constant(
             context, loom_vector_iota_base(source_op), &base) &&
         loom_amdgpu_value_as_i32_constant(
             context, loom_vector_iota_step(source_op), &step) &&
         loom_amdgpu_iota_i32_lanes_fit(base, step, lane_count);
}

static bool loom_amdgpu_can_lower_buffer_view(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op) {
  int64_t unused_byte_offset = 0;
  return loom_amdgpu_value_is_32bit_view(context,
                                         loom_buffer_view_result(source_op)) &&
         loom_amdgpu_module_value_as_exact_index_constant(
             loom_low_lower_context_module(context),
             loom_buffer_view_byte_offset(source_op), &unused_byte_offset) &&
         unused_byte_offset >= 0;
}

static bool loom_amdgpu_module_value_is_workitem_id_x(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  return defining_op && loom_kernel_workitem_id_isa(defining_op) &&
         loom_kernel_workitem_id_dimension(defining_op) ==
             LOOM_KERNEL_WORKITEM_ID_DIMENSION_X;
}

typedef uint32_t loom_amdgpu_memory_access_rejection_flags_t;

#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIBE_FAILED ((uint32_t)1u << 0)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_LAYOUT ((uint32_t)1u << 1)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_ELEMENT_WIDTH ((uint32_t)1u << 2)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_RANK ((uint32_t)1u << 3)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_TYPE ((uint32_t)1u << 4)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE \
  ((uint32_t)1u << 5)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_COUNT \
  ((uint32_t)1u << 6)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_AXIS ((uint32_t)1u << 7)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_SOURCE \
  ((uint32_t)1u << 8)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_STRIDE ((uint32_t)1u << 9)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_B128_DYNAMIC_ALIGNMENT \
  ((uint32_t)1u << 10)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_STATIC_OFFSET ((uint32_t)1u << 11)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_NEGATIVE_STATIC_OFFSET \
  ((uint32_t)1u << 12)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_B128_STATIC_ALIGNMENT \
  ((uint32_t)1u << 13)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_MISSING \
  ((uint32_t)1u << 14)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE \
  ((uint32_t)1u << 15)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE \
  ((uint32_t)1u << 16)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VIEW_SOURCE ((uint32_t)1u << 17)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VIEW_BASE ((uint32_t)1u << 18)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VIEW_BASE_OVERFLOW \
  ((uint32_t)1u << 19)

typedef struct loom_amdgpu_memory_access_diagnostic_t {
  // Rejection bits explaining why an access is not legal for this target.
  loom_amdgpu_memory_access_rejection_flags_t rejection_bits;
} loom_amdgpu_memory_access_diagnostic_t;

typedef struct loom_amdgpu_memory_access_t {
  // Dynamic view-axis index used to compute the VADDR register, or invalid for
  // a purely static access.
  loom_value_id_t dynamic_index;
  // Byte stride multiplied by dynamic_index to compute VADDR.
  uint32_t dynamic_index_byte_stride;
  // Total static byte offset selected from the source view access.
  int64_t static_byte_offset;
  // Static byte offset encoded in the descriptor offset immediate.
  uint32_t immediate_byte_offset;
  // Static byte offset materialized through the scalar SOFFSET operand.
  uint32_t scalar_byte_offset;
  // Number of 32-bit VGPR lanes moved by the selected memory packet.
  uint32_t vgpr_count;
  // Stable descriptor ID selected for the active descriptor set.
  uint64_t descriptor_id;
} loom_amdgpu_memory_access_t;

static iree_string_view_t loom_amdgpu_memory_access_rejection_detail(
    loom_amdgpu_memory_access_rejection_flags_t rejection_bits) {
  if (iree_any_bit_set(rejection_bits,
                       LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIBE_FAILED)) {
    return IREE_SV(
        "memory access shape is not representable as a vector "
        "footprint in the source view");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_LAYOUT)) {
    return IREE_SV(
        "AMDGPU buffer memory lowering currently requires a dense "
        "view layout");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_ELEMENT_WIDTH)) {
    return IREE_SV(
        "AMDGPU buffer memory lowering currently requires 32-bit "
        "view elements");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_RANK)) {
    return IREE_SV(
        "AMDGPU buffer memory lowering currently requires "
        "one-dimensional vectors");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_TYPE)) {
    return IREE_SV(
        "AMDGPU buffer memory lowering currently supports one-lane "
        "and four-lane 32-bit vectors");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE)) {
    return IREE_SV(
        "AMDGPU buffer memory lowering requires unit stride along "
        "the vector axis");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_COUNT)) {
    return IREE_SV(
        "AMDGPU buffer memory lowering currently supports at most "
        "one dynamic index");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_AXIS)) {
    return IREE_SV(
        "AMDGPU buffer memory lowering requires exactly one "
        "well-formed dynamic axis for dynamic accesses");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_SOURCE)) {
    return IREE_SV(
        "AMDGPU buffer memory lowering currently requires dynamic "
        "indices to come from kernel.workitem.id<x>");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_STRIDE)) {
    return IREE_SV(
        "AMDGPU buffer memory lowering requires a non-negative "
        "32-bit dynamic byte stride");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_B128_DYNAMIC_ALIGNMENT)) {
    return IREE_SV(
        "128-bit AMDGPU buffer memory accesses currently require "
        "16-byte aligned dynamic byte strides");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_STATIC_OFFSET)) {
    return IREE_SV(
        "AMDGPU buffer memory lowering requires a statically "
        "representable descriptor offset");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_NEGATIVE_STATIC_OFFSET)) {
    return IREE_SV(
        "AMDGPU buffer memory lowering requires non-negative "
        "static byte offsets");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_B128_STATIC_ALIGNMENT)) {
    return IREE_SV(
        "128-bit AMDGPU buffer memory accesses currently require "
        "16-byte aligned static byte offsets");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_MISSING)) {
    return IREE_SV(
        "selected AMDGPU descriptor set has no buffer memory "
        "descriptor for this access width");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE)) {
    return IREE_SV(
        "selected AMDGPU buffer memory descriptor does not expose "
        "one unsigned offset immediate");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE)) {
    return IREE_SV(
        "AMDGPU buffer memory static byte offset is outside the selected "
        "descriptor's immediate plus scalar SOFFSET range");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VIEW_SOURCE)) {
    return IREE_SV(
        "AMDGPU buffer memory lowering currently requires views to come from "
        "buffer.view");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VIEW_BASE)) {
    return IREE_SV(
        "AMDGPU HAL buffer views currently require exact non-negative static "
        "byte offsets");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VIEW_BASE_OVERFLOW)) {
    return IREE_SV(
        "AMDGPU HAL buffer view byte offset overflows the static memory "
        "access offset");
  }
  return IREE_SV("AMDGPU buffer memory access is not target-legal");
}

static bool loom_amdgpu_memory_access_find_dynamic_axis(
    loom_attribute_t static_indices, uint8_t* out_dynamic_axis) {
  IREE_ASSERT_ARGUMENT(out_dynamic_axis);
  *out_dynamic_axis = UINT8_MAX;
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY) {
    return false;
  }
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    if (static_indices.i64_array[i] != INT64_MIN) {
      continue;
    }
    if (*out_dynamic_axis != UINT8_MAX || i > UINT8_MAX) {
      return false;
    }
    *out_dynamic_axis = (uint8_t)i;
  }
  return true;
}

static bool loom_amdgpu_memory_access_static_byte_offset(
    const loom_vector_memory_access_t* vector_access,
    loom_attribute_t static_indices, uint8_t dynamic_axis,
    int64_t* out_static_byte_offset) {
  IREE_ASSERT_ARGUMENT(out_static_byte_offset);
  *out_static_byte_offset = 0;
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY) {
    return false;
  }
  if (static_indices.count > LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK) {
    return false;
  }

  int64_t static_origin[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {0};
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    if (i == dynamic_axis) {
      if (static_indices.i64_array[i] != INT64_MIN) {
        return false;
      }
      continue;
    }
    if (static_indices.i64_array[i] == INT64_MIN) {
      return false;
    }
    static_origin[i] = static_indices.i64_array[i];
  }
  loom_attribute_t static_origin_attr =
      loom_attr_i64_array(static_origin, static_indices.count);
  int64_t lane_indices[] = {0};
  return loom_vector_memory_access_static_lane_byte_offset(
      vector_access, static_origin_attr, lane_indices,
      IREE_ARRAYSIZE(lane_indices), out_static_byte_offset);
}

static bool loom_amdgpu_memory_access_static_byte_offset_is_usable(
    int64_t static_byte_offset, uint32_t vgpr_count,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (static_byte_offset < 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_NEGATIVE_STATIC_OFFSET;
    return false;
  }
  if (vgpr_count == 4 && (static_byte_offset & 15) != 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_B128_STATIC_ALIGNMENT;
    return false;
  }
  return true;
}

static bool loom_amdgpu_memory_access_view_base_byte_offset(
    const loom_module_t* module, loom_value_id_t view_value_id,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic,
    int64_t* out_byte_offset) {
  IREE_ASSERT_ARGUMENT(out_byte_offset);
  *out_byte_offset = 0;
  const loom_value_t* value = loom_module_value(module, view_value_id);
  if (loom_value_is_block_arg(value)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VIEW_SOURCE;
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL || !loom_buffer_view_isa(defining_op)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VIEW_SOURCE;
    return false;
  }
  if (!loom_amdgpu_module_value_as_exact_index_constant(
          module, loom_buffer_view_byte_offset(defining_op), out_byte_offset)) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VIEW_BASE;
    return false;
  }
  if (*out_byte_offset < 0) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VIEW_BASE;
    return false;
  }
  return true;
}

static bool loom_amdgpu_memory_access_add_view_base_byte_offset(
    const loom_module_t* module, loom_value_id_t view_value_id,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic,
    int64_t* inout_static_byte_offset) {
  IREE_ASSERT_ARGUMENT(inout_static_byte_offset);
  int64_t view_base_byte_offset = 0;
  if (!loom_amdgpu_memory_access_view_base_byte_offset(
          module, view_value_id, diagnostic, &view_base_byte_offset)) {
    return false;
  }
  int64_t static_byte_offset = 0;
  if (!iree_checked_add_i64(*inout_static_byte_offset, view_base_byte_offset,
                            &static_byte_offset)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VIEW_BASE_OVERFLOW;
    return false;
  }
  *inout_static_byte_offset = static_byte_offset;
  return true;
}

typedef enum loom_amdgpu_buffer_memory_kind_e {
  LOOM_AMDGPU_BUFFER_MEMORY_LOAD = 0,
  LOOM_AMDGPU_BUFFER_MEMORY_STORE = 1,
} loom_amdgpu_buffer_memory_kind_t;

typedef struct loom_amdgpu_buffer_memory_descriptor_family_t {
  // Number of VGPR lanes moved by the memory packet.
  uint32_t vgpr_count;
  // Direction of the memory packet.
  loom_amdgpu_buffer_memory_kind_t kind;
  // Candidate descriptor stable IDs ordered by preference.
  const uint64_t* descriptor_ids;
  // Number of entries in descriptor_ids.
  iree_host_size_t descriptor_id_count;
} loom_amdgpu_buffer_memory_descriptor_family_t;

static bool loom_amdgpu_select_buffer_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_buffer_memory_kind_t kind, uint64_t* out_descriptor_id,
    uint32_t* out_descriptor_ordinal) {
  IREE_ASSERT_ARGUMENT(out_descriptor_id);
  IREE_ASSERT_ARGUMENT(out_descriptor_ordinal);
  *out_descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE;
  *out_descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  // The GFX11 constants name descriptor-key stable IDs that are shared by
  // GFX11/GFX12/GFX1250 descriptor sets for the same low op key.
  static const uint64_t kLoadB32DescriptorIds[] = {
      AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_BUFFER_LOAD_DWORD,
  };
  static const uint64_t kLoadB64DescriptorIds[] = {
      AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_BUFFER_LOAD_B64,
      AMDGPU_GFX950_CORE_DESCRIPTOR_ID_AMDGPU_BUFFER_LOAD_DWORDX2,
  };
  static const uint64_t kLoadB128DescriptorIds[] = {
      AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_BUFFER_LOAD_B128,
      AMDGPU_GFX950_CORE_DESCRIPTOR_ID_AMDGPU_BUFFER_LOAD_DWORDX4,
  };
  static const uint64_t kStoreB32DescriptorIds[] = {
      AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_BUFFER_STORE_DWORD,
  };
  static const uint64_t kStoreB64DescriptorIds[] = {
      AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_BUFFER_STORE_B64,
      AMDGPU_GFX950_CORE_DESCRIPTOR_ID_AMDGPU_BUFFER_STORE_DWORDX2,
  };
  static const uint64_t kStoreB128DescriptorIds[] = {
      AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_BUFFER_STORE_B128,
      AMDGPU_GFX950_CORE_DESCRIPTOR_ID_AMDGPU_BUFFER_STORE_DWORDX4,
  };
  static const loom_amdgpu_buffer_memory_descriptor_family_t kFamilies[] = {
      {
          .vgpr_count = 1,
          .kind = LOOM_AMDGPU_BUFFER_MEMORY_LOAD,
          .descriptor_ids = kLoadB32DescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kLoadB32DescriptorIds),
      },
      {
          .vgpr_count = 2,
          .kind = LOOM_AMDGPU_BUFFER_MEMORY_LOAD,
          .descriptor_ids = kLoadB64DescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kLoadB64DescriptorIds),
      },
      {
          .vgpr_count = 4,
          .kind = LOOM_AMDGPU_BUFFER_MEMORY_LOAD,
          .descriptor_ids = kLoadB128DescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kLoadB128DescriptorIds),
      },
      {
          .vgpr_count = 1,
          .kind = LOOM_AMDGPU_BUFFER_MEMORY_STORE,
          .descriptor_ids = kStoreB32DescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kStoreB32DescriptorIds),
      },
      {
          .vgpr_count = 2,
          .kind = LOOM_AMDGPU_BUFFER_MEMORY_STORE,
          .descriptor_ids = kStoreB64DescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kStoreB64DescriptorIds),
      },
      {
          .vgpr_count = 4,
          .kind = LOOM_AMDGPU_BUFFER_MEMORY_STORE,
          .descriptor_ids = kStoreB128DescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kStoreB128DescriptorIds),
      },
  };

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kFamilies); ++i) {
    const loom_amdgpu_buffer_memory_descriptor_family_t* family = &kFamilies[i];
    if (family->vgpr_count != access->vgpr_count || family->kind != kind) {
      continue;
    }
    for (iree_host_size_t j = 0; j < family->descriptor_id_count; ++j) {
      const uint64_t descriptor_id = family->descriptor_ids[j];
      const uint32_t descriptor_ordinal =
          loom_low_descriptor_set_lookup_descriptor_by_id(descriptor_set,
                                                          descriptor_id);
      if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
        continue;
      }
      *out_descriptor_id = descriptor_id;
      *out_descriptor_ordinal = descriptor_ordinal;
      return true;
    }
    return false;
  }
  return false;
}

static bool loom_amdgpu_buffer_memory_offset_immediate_range(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal, uint64_t* out_unsigned_max) {
  IREE_ASSERT_ARGUMENT(out_unsigned_max);
  *out_unsigned_max = 0;
  if (descriptor_ordinal >= descriptor_set->descriptor_count) {
    return false;
  }
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[descriptor_ordinal];
  if (descriptor->immediate_count != 1 ||
      descriptor->immediate_start >= descriptor_set->immediate_count) {
    return false;
  }
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[descriptor->immediate_start];
  if (immediate->kind != LOOM_LOW_IMMEDIATE_KIND_UNSIGNED) {
    return false;
  }
  *out_unsigned_max = immediate->unsigned_max;
  return true;
}

static bool loom_amdgpu_memory_access_split_static_offset(
    loom_amdgpu_memory_access_t* access, uint64_t offset_unsigned_max,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (access->static_byte_offset < 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_NEGATIVE_STATIC_OFFSET;
    return false;
  }

  const uint64_t static_byte_offset = (uint64_t)access->static_byte_offset;
  uint64_t immediate_byte_offset =
      iree_min(static_byte_offset, offset_unsigned_max);
  if (access->vgpr_count == 4) {
    immediate_byte_offset &= ~UINT64_C(15);
  }
  const uint64_t scalar_byte_offset =
      static_byte_offset - immediate_byte_offset;
  if (scalar_byte_offset > UINT32_MAX) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE;
    return false;
  }
  access->immediate_byte_offset = (uint32_t)immediate_byte_offset;
  access->scalar_byte_offset = (uint32_t)scalar_byte_offset;
  return true;
}

static bool loom_amdgpu_memory_access_select(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t view_value_id, loom_value_slice_t dynamic_indices,
    loom_attribute_t static_indices, loom_type_t view_type,
    loom_type_t vector_type, loom_amdgpu_buffer_memory_kind_t kind,
    loom_amdgpu_memory_access_t* out_access,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic) {
  IREE_ASSERT_ARGUMENT(out_access);
  IREE_ASSERT_ARGUMENT(out_diagnostic);
  *out_access = (loom_amdgpu_memory_access_t){
      .dynamic_index = LOOM_VALUE_ID_INVALID,
      .descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE,
  };
  *out_diagnostic = (loom_amdgpu_memory_access_diagnostic_t){0};

  loom_vector_memory_access_t vector_access;
  if (!loom_vector_memory_access_describe(module, view_type, vector_type,
                                          &vector_access)) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIBE_FAILED;
    return false;
  }
  if (vector_access.layout_kind != LOOM_VECTOR_MEMORY_LAYOUT_DENSE) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_LAYOUT;
    return false;
  }
  if (vector_access.static_element_byte_count != 4) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_ELEMENT_WIDTH;
    return false;
  }
  if (vector_access.vector_rank != 1) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_RANK;
    return false;
  }

  out_access->vgpr_count = loom_amdgpu_vector_32bit_lane_count(vector_type);
  if (out_access->vgpr_count == 0) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_TYPE;
    return false;
  }

  int64_t vector_axis_stride = 0;
  if (!loom_vector_memory_access_static_axis_stride(
          &vector_access, vector_access.first_vector_axis,
          &vector_axis_stride) ||
      vector_axis_stride != 1) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
    return false;
  }

  if (dynamic_indices.count == 0) {
    int64_t lane_indices[] = {0};
    int64_t static_byte_offset = 0;
    if (!loom_vector_memory_access_static_lane_byte_offset(
            &vector_access, static_indices, lane_indices,
            IREE_ARRAYSIZE(lane_indices), &static_byte_offset)) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_STATIC_OFFSET;
      return false;
    }
    if (!loom_amdgpu_memory_access_add_view_base_byte_offset(
            module, view_value_id, out_diagnostic, &static_byte_offset)) {
      return false;
    }
    if (!loom_amdgpu_memory_access_static_byte_offset_is_usable(
            static_byte_offset, out_access->vgpr_count, out_diagnostic)) {
      return false;
    }
    out_access->static_byte_offset = static_byte_offset;
  } else {
    if (dynamic_indices.count != 1) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_COUNT;
      return false;
    }
    uint8_t dynamic_axis = UINT8_MAX;
    if (!loom_amdgpu_memory_access_find_dynamic_axis(static_indices,
                                                     &dynamic_axis) ||
        dynamic_axis == UINT8_MAX || dynamic_axis >= vector_access.view_rank) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_AXIS;
      return false;
    }
    if (!loom_amdgpu_module_value_is_workitem_id_x(module,
                                                   dynamic_indices.values[0])) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_SOURCE;
      return false;
    }

    int64_t axis_stride = 0;
    if (!loom_vector_memory_access_static_axis_stride(
            &vector_access, dynamic_axis, &axis_stride)) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_STRIDE;
      return false;
    }
    int64_t dynamic_index_byte_stride = 0;
    if (!iree_checked_mul_i64(axis_stride,
                              vector_access.static_element_byte_count,
                              &dynamic_index_byte_stride) ||
        dynamic_index_byte_stride < 0 ||
        dynamic_index_byte_stride > UINT32_MAX) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_STRIDE;
      return false;
    }
    if (out_access->vgpr_count == 4 && (dynamic_index_byte_stride & 15) != 0) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_B128_DYNAMIC_ALIGNMENT;
      return false;
    }
    int64_t static_byte_offset = 0;
    if (!loom_amdgpu_memory_access_static_byte_offset(
            &vector_access, static_indices, dynamic_axis,
            &static_byte_offset)) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_STATIC_OFFSET;
      return false;
    }
    if (!loom_amdgpu_memory_access_add_view_base_byte_offset(
            module, view_value_id, out_diagnostic, &static_byte_offset)) {
      return false;
    }
    if (!loom_amdgpu_memory_access_static_byte_offset_is_usable(
            static_byte_offset, out_access->vgpr_count, out_diagnostic)) {
      return false;
    }
    out_access->dynamic_index = dynamic_indices.values[0];
    out_access->dynamic_index_byte_stride = (uint32_t)dynamic_index_byte_stride;
    out_access->static_byte_offset = static_byte_offset;
  }

  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (!loom_amdgpu_select_buffer_memory_descriptor(
          descriptor_set, out_access, kind, &out_access->descriptor_id,
          &descriptor_ordinal)) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_MISSING;
    return false;
  }
  uint64_t offset_unsigned_max = 0;
  if (!loom_amdgpu_buffer_memory_offset_immediate_range(
          descriptor_set, descriptor_ordinal, &offset_unsigned_max)) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
    return false;
  }
  return loom_amdgpu_memory_access_split_static_offset(
      out_access, offset_unsigned_max, out_diagnostic);
}

static bool loom_amdgpu_load_memory_access_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_memory_access_t* out_access) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_amdgpu_memory_access_diagnostic_t diagnostic = {0};
  return loom_amdgpu_memory_access_select(
      module, loom_low_lower_context_descriptor_set(context),
      loom_vector_load_view(source_op), loom_vector_load_indices(source_op),
      loom_vector_load_static_indices(source_op),
      loom_module_value_type(module, loom_vector_load_view(source_op)),
      loom_module_value_type(module, loom_vector_load_result(source_op)),
      LOOM_AMDGPU_BUFFER_MEMORY_LOAD, out_access, &diagnostic);
}

static bool loom_amdgpu_store_memory_access_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_memory_access_t* out_access) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_amdgpu_memory_access_diagnostic_t diagnostic = {0};
  return loom_amdgpu_memory_access_select(
      module, loom_low_lower_context_descriptor_set(context),
      loom_vector_store_view(source_op), loom_vector_store_indices(source_op),
      loom_vector_store_static_indices(source_op),
      loom_module_value_type(module, loom_vector_store_view(source_op)),
      loom_module_value_type(module, loom_vector_store_value(source_op)),
      LOOM_AMDGPU_BUFFER_MEMORY_STORE, out_access, &diagnostic);
}

static bool loom_amdgpu_can_lower_vector_load(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op) {
  loom_amdgpu_memory_access_t access;
  return loom_amdgpu_value_is_32bit_view(context,
                                         loom_vector_load_view(source_op)) &&
         loom_amdgpu_value_is_vector_1x32_or_4x32(
             context, loom_vector_load_result(source_op)) &&
         loom_amdgpu_load_memory_access_select(context, source_op, &access);
}

static bool loom_amdgpu_can_lower_vector_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_amdgpu_memory_access_t access;
  return loom_amdgpu_value_is_vector_1x32_or_4x32(
             context, loom_vector_store_value(source_op)) &&
         loom_amdgpu_value_is_32bit_view(context,
                                         loom_vector_store_view(source_op)) &&
         loom_amdgpu_store_memory_access_select(context, source_op, &access);
}

static bool loom_amdgpu_can_lower_i32_binary(loom_low_lower_context_t* context,
                                             loom_value_id_t lhs,
                                             loom_value_id_t rhs,
                                             loom_value_id_t result,
                                             bool allow_uniform_result) {
  if (!loom_amdgpu_value_is_i32(context, lhs) ||
      !loom_amdgpu_value_is_i32(context, rhs) ||
      !loom_amdgpu_value_is_i32(context, result)) {
    return false;
  }
  const bool lhs_vgpr = loom_amdgpu_value_prefers_vgpr(context, lhs);
  const bool rhs_vgpr = loom_amdgpu_value_prefers_vgpr(context, rhs);
  const bool result_vgpr = loom_amdgpu_value_prefers_vgpr(context, result);
  if (result_vgpr) {
    return loom_amdgpu_value_can_materialize_as_vgpr_i32(context, lhs) &&
           loom_amdgpu_value_can_materialize_as_vgpr_i32(context, rhs);
  }
  return allow_uniform_result && !lhs_vgpr && !rhs_vgpr;
}

static bool loom_amdgpu_can_lower_vector_i32_binary(
    loom_low_lower_context_t* context, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_value_id_t result) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t result_type = loom_module_value_type(module, result);
  return loom_amdgpu_vector_i32_lane_count(result_type) != 0 &&
         loom_type_equal(loom_module_value_type(module, lhs), result_type) &&
         loom_type_equal(loom_module_value_type(module, rhs), result_type);
}

static bool loom_amdgpu_can_lower_f32_binary(loom_low_lower_context_t* context,
                                             loom_value_id_t lhs,
                                             loom_value_id_t rhs,
                                             loom_value_id_t result) {
  return loom_amdgpu_value_is_f32(context, lhs) &&
         loom_amdgpu_value_is_f32(context, rhs) &&
         loom_amdgpu_value_is_f32(context, result);
}

static bool loom_amdgpu_can_lower_vector_f32_binary(
    loom_low_lower_context_t* context, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_value_id_t result) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t result_type = loom_module_value_type(module, result);
  return (loom_amdgpu_type_is_vector_1xf32(result_type) ||
          loom_amdgpu_type_is_vector_4xf32(result_type)) &&
         loom_type_equal(loom_module_value_type(module, lhs), result_type) &&
         loom_type_equal(loom_module_value_type(module, rhs), result_type);
}

static bool loom_amdgpu_can_lower_vector_f32_ternary(
    loom_low_lower_context_t* context, loom_value_id_t a, loom_value_id_t b,
    loom_value_id_t c, loom_value_id_t result) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t result_type = loom_module_value_type(module, result);
  return (loom_amdgpu_type_is_vector_1xf32(result_type) ||
          loom_amdgpu_type_is_vector_4xf32(result_type)) &&
         loom_type_equal(loom_module_value_type(module, a), result_type) &&
         loom_type_equal(loom_module_value_type(module, b), result_type) &&
         loom_type_equal(loom_module_value_type(module, c), result_type);
}

static bool loom_amdgpu_can_lower_f32_ternary(loom_low_lower_context_t* context,
                                              loom_value_id_t a,
                                              loom_value_id_t b,
                                              loom_value_id_t c,
                                              loom_value_id_t result) {
  return loom_amdgpu_value_is_f32(context, a) &&
         loom_amdgpu_value_is_f32(context, b) &&
         loom_amdgpu_value_is_f32(context, c) &&
         loom_amdgpu_value_is_f32(context, result);
}

static bool loom_amdgpu_vector_extract_select(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              uint32_t* out_lane_offset) {
  IREE_ASSERT_ARGUMENT(out_lane_offset);
  *out_lane_offset = 0;
  if (loom_vector_extract_indices(source_op).count != 0) {
    return false;
  }
  loom_attribute_t static_indices =
      loom_vector_extract_static_indices(source_op);
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY || static_indices.count != 1 ||
      static_indices.i64_array[0] < 0 ||
      static_indices.i64_array[0] > UINT32_MAX) {
    return false;
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type =
      loom_module_value_type(module, loom_vector_extract_source(source_op));
  const uint32_t lane_count = loom_amdgpu_vector_32bit_lane_count(source_type);
  const uint32_t lane_offset = (uint32_t)static_indices.i64_array[0];
  if (lane_count == 0 || lane_offset >= lane_count) {
    return false;
  }

  const loom_type_t result_type =
      loom_module_value_type(module, loom_vector_extract_result(source_op));
  if (loom_type_element_type(source_type) == LOOM_SCALAR_TYPE_I32 &&
      !loom_amdgpu_type_is_i32(result_type)) {
    return false;
  }
  if (loom_type_element_type(source_type) == LOOM_SCALAR_TYPE_F32 &&
      !loom_amdgpu_type_is_f32(result_type)) {
    return false;
  }
  *out_lane_offset = lane_offset;
  return true;
}

static bool loom_amdgpu_can_lower_vector_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  uint32_t unused_lane_offset = 0;
  return loom_amdgpu_vector_extract_select(context, source_op,
                                           &unused_lane_offset);
}

static bool loom_amdgpu_can_lower_vector_from_elements(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t source_result =
      loom_vector_from_elements_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, source_result);
  const uint32_t lane_count = loom_amdgpu_vector_32bit_lane_count(result_type);
  if (lane_count == 0) {
    return false;
  }
  loom_value_slice_t elements = loom_vector_from_elements_elements(source_op);
  if (elements.count != lane_count) {
    return false;
  }
  const loom_scalar_type_t element_type = loom_type_element_type(result_type);
  for (uint32_t i = 0; i < elements.count; ++i) {
    const loom_value_id_t element = elements.values[i];
    const loom_type_t source_type = loom_module_value_type(module, element);
    if (!loom_type_is_scalar(source_type) ||
        loom_type_element_type(source_type) != element_type) {
      return false;
    }
    if (element_type == LOOM_SCALAR_TYPE_I32 &&
        !loom_amdgpu_value_can_materialize_as_vgpr_i32(context, element)) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_can_lower_workitem_id(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op) {
  return loom_kernel_workitem_id_dimension(source_op) ==
             LOOM_KERNEL_WORKITEM_ID_DIMENSION_X &&
         loom_amdgpu_value_is_address_scalar(
             context, loom_kernel_workitem_id_result(source_op));
}

static iree_status_t loom_amdgpu_can_lower_op(void* user_data,
                                              loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              bool* out_handled) {
  (void)user_data;
  switch (source_op->kind) {
    case LOOM_OP_INDEX_CONSTANT:
      *out_handled = loom_amdgpu_can_lower_index_constant(context, source_op);
      return iree_ok_status();
    case LOOM_OP_SCALAR_CONSTANT:
      *out_handled = loom_amdgpu_can_lower_scalar_constant(context, source_op);
      return iree_ok_status();
    case LOOM_OP_VECTOR_CONSTANT:
      *out_handled = loom_amdgpu_can_lower_vector_constant(context, source_op);
      return iree_ok_status();
    case LOOM_OP_VECTOR_IOTA:
      *out_handled = loom_amdgpu_can_lower_vector_iota(context, source_op);
      return iree_ok_status();
    case LOOM_OP_BUFFER_ASSUME_MEMORY_SPACE:
      *out_handled = true;
      return iree_ok_status();
    case LOOM_OP_BUFFER_VIEW:
      *out_handled = loom_amdgpu_can_lower_buffer_view(context, source_op);
      return iree_ok_status();
    case LOOM_OP_KERNEL_WORKITEM_ID:
      *out_handled = loom_amdgpu_can_lower_workitem_id(context, source_op);
      return iree_ok_status();
    case LOOM_OP_SCALAR_ADDI:
      *out_handled = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_addi_lhs(source_op),
          loom_scalar_addi_rhs(source_op), loom_scalar_addi_result(source_op),
          /*allow_uniform_result=*/true);
      return iree_ok_status();
    case LOOM_OP_SCALAR_SUBI:
      *out_handled = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_subi_lhs(source_op),
          loom_scalar_subi_rhs(source_op), loom_scalar_subi_result(source_op),
          /*allow_uniform_result=*/true);
      return iree_ok_status();
    case LOOM_OP_SCALAR_MULI:
      *out_handled = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_muli_lhs(source_op),
          loom_scalar_muli_rhs(source_op), loom_scalar_muli_result(source_op),
          /*allow_uniform_result=*/false);
      return iree_ok_status();
    case LOOM_OP_SCALAR_ANDI:
      *out_handled = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_andi_lhs(source_op),
          loom_scalar_andi_rhs(source_op), loom_scalar_andi_result(source_op),
          /*allow_uniform_result=*/true);
      return iree_ok_status();
    case LOOM_OP_SCALAR_ORI:
      *out_handled = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_ori_lhs(source_op),
          loom_scalar_ori_rhs(source_op), loom_scalar_ori_result(source_op),
          /*allow_uniform_result=*/true);
      return iree_ok_status();
    case LOOM_OP_SCALAR_XORI:
      *out_handled = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_xori_lhs(source_op),
          loom_scalar_xori_rhs(source_op), loom_scalar_xori_result(source_op),
          /*allow_uniform_result=*/true);
      return iree_ok_status();
    case LOOM_OP_SCALAR_SHLI:
      *out_handled = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_shli_lhs(source_op),
          loom_scalar_shli_rhs(source_op), loom_scalar_shli_result(source_op),
          /*allow_uniform_result=*/true);
      return iree_ok_status();
    case LOOM_OP_SCALAR_SHRSI:
      *out_handled = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_shrsi_lhs(source_op),
          loom_scalar_shrsi_rhs(source_op), loom_scalar_shrsi_result(source_op),
          /*allow_uniform_result=*/true);
      return iree_ok_status();
    case LOOM_OP_SCALAR_SHRUI:
      *out_handled = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_shrui_lhs(source_op),
          loom_scalar_shrui_rhs(source_op), loom_scalar_shrui_result(source_op),
          /*allow_uniform_result=*/true);
      return iree_ok_status();
    case LOOM_OP_SCALAR_ADDF:
      *out_handled = loom_amdgpu_can_lower_f32_binary(
          context, loom_scalar_addf_lhs(source_op),
          loom_scalar_addf_rhs(source_op), loom_scalar_addf_result(source_op));
      return iree_ok_status();
    case LOOM_OP_SCALAR_SUBF:
      *out_handled = loom_amdgpu_can_lower_f32_binary(
          context, loom_scalar_subf_lhs(source_op),
          loom_scalar_subf_rhs(source_op), loom_scalar_subf_result(source_op));
      return iree_ok_status();
    case LOOM_OP_SCALAR_MULF:
      *out_handled = loom_amdgpu_can_lower_f32_binary(
          context, loom_scalar_mulf_lhs(source_op),
          loom_scalar_mulf_rhs(source_op), loom_scalar_mulf_result(source_op));
      return iree_ok_status();
    case LOOM_OP_SCALAR_FMAF:
      *out_handled = loom_amdgpu_can_lower_f32_ternary(
          context, loom_scalar_fmaf_a(source_op), loom_scalar_fmaf_b(source_op),
          loom_scalar_fmaf_c(source_op), loom_scalar_fmaf_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_ADDI:
      *out_handled = loom_amdgpu_can_lower_vector_i32_binary(
          context, loom_vector_addi_lhs(source_op),
          loom_vector_addi_rhs(source_op), loom_vector_addi_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_SUBI:
      *out_handled = loom_amdgpu_can_lower_vector_i32_binary(
          context, loom_vector_subi_lhs(source_op),
          loom_vector_subi_rhs(source_op), loom_vector_subi_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_MULI:
      *out_handled = loom_amdgpu_can_lower_vector_i32_binary(
          context, loom_vector_muli_lhs(source_op),
          loom_vector_muli_rhs(source_op), loom_vector_muli_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_ANDI:
      *out_handled = loom_amdgpu_can_lower_vector_i32_binary(
          context, loom_vector_andi_lhs(source_op),
          loom_vector_andi_rhs(source_op), loom_vector_andi_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_ORI:
      *out_handled = loom_amdgpu_can_lower_vector_i32_binary(
          context, loom_vector_ori_lhs(source_op),
          loom_vector_ori_rhs(source_op), loom_vector_ori_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_XORI:
      *out_handled = loom_amdgpu_can_lower_vector_i32_binary(
          context, loom_vector_xori_lhs(source_op),
          loom_vector_xori_rhs(source_op), loom_vector_xori_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_SHLI:
      *out_handled = loom_amdgpu_can_lower_vector_i32_binary(
          context, loom_vector_shli_lhs(source_op),
          loom_vector_shli_rhs(source_op), loom_vector_shli_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_SHRSI:
      *out_handled = loom_amdgpu_can_lower_vector_i32_binary(
          context, loom_vector_shrsi_lhs(source_op),
          loom_vector_shrsi_rhs(source_op),
          loom_vector_shrsi_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_SHRUI:
      *out_handled = loom_amdgpu_can_lower_vector_i32_binary(
          context, loom_vector_shrui_lhs(source_op),
          loom_vector_shrui_rhs(source_op),
          loom_vector_shrui_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_ADDF:
      *out_handled = loom_amdgpu_can_lower_vector_f32_binary(
          context, loom_vector_addf_lhs(source_op),
          loom_vector_addf_rhs(source_op), loom_vector_addf_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_SUBF:
      *out_handled = loom_amdgpu_can_lower_vector_f32_binary(
          context, loom_vector_subf_lhs(source_op),
          loom_vector_subf_rhs(source_op), loom_vector_subf_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_MULF:
      *out_handled = loom_amdgpu_can_lower_vector_f32_binary(
          context, loom_vector_mulf_lhs(source_op),
          loom_vector_mulf_rhs(source_op), loom_vector_mulf_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_FMAF:
      *out_handled = loom_amdgpu_can_lower_vector_f32_ternary(
          context, loom_vector_fmaf_a(source_op), loom_vector_fmaf_b(source_op),
          loom_vector_fmaf_c(source_op), loom_vector_fmaf_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_EXTRACT:
      *out_handled = loom_amdgpu_can_lower_vector_extract(context, source_op);
      return iree_ok_status();
    case LOOM_OP_VECTOR_FROM_ELEMENTS:
      *out_handled =
          loom_amdgpu_can_lower_vector_from_elements(context, source_op);
      return iree_ok_status();
    case LOOM_OP_VECTOR_LOAD:
      *out_handled = loom_amdgpu_can_lower_vector_load(context, source_op);
      return iree_ok_status();
    case LOOM_OP_VECTOR_STORE:
      *out_handled = loom_amdgpu_can_lower_vector_store(context, source_op);
      return iree_ok_status();
    default:
      *out_handled = false;
      return iree_ok_status();
  }
}

static iree_status_t loom_amdgpu_intern(loom_low_lower_context_t* context,
                                        iree_string_view_t string,
                                        loom_string_id_t* out_string_id) {
  return loom_module_intern_string(loom_low_lower_context_module(context),
                                   string, out_string_id);
}

static iree_status_t loom_amdgpu_low_result_type(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, loom_type_t* out_low_type) {
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(context, source_op,
                                                source_result, out_low_type));
  if (!loom_type_is_register(*out_low_type)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU source type did not map to a register");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_low_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t descriptor_id, const loom_value_id_t* operands,
    iree_host_size_t operand_count, loom_named_attr_slice_t attrs,
    const loom_type_t* result_types, iree_host_size_t result_count,
    loom_op_t** out_op) {
  IREE_ASSERT_ARGUMENT(out_op);
  *out_op = NULL;
  return loom_low_lower_emit_descriptor_op(
      context, descriptor_id, operands, operand_count, attrs, result_types,
      result_count, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, out_op);
}

static iree_status_t loom_amdgpu_emit_const_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t descriptor_id, uint32_t value, loom_type_t result_type,
    loom_value_id_t* out_value_id) {
  IREE_ASSERT_ARGUMENT(out_value_id);
  *out_value_id = LOOM_VALUE_ID_INVALID;
  loom_string_id_t value_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_intern(context, IREE_SV("imm32"), &value_name_id));
  loom_named_attr_t attrs[] = {
      {
          .name_id = value_name_id,
          .value = loom_attr_i64(value),
      },
  };
  loom_op_t* low_const = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_descriptor_const(
      context, descriptor_id,
      loom_make_named_attr_slice(attrs, IREE_ARRAYSIZE(attrs)), result_type,
      source_op->location, &low_const));
  *out_value_id = loom_low_const_result(low_const);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lookup_or_materialize_vgpr_i32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value) {
  IREE_ASSERT_ARGUMENT(out_low_value);
  *out_low_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_value));

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  bool is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, AMDGPU_GFX11_CORE_REG_CLASS_ID_AMDGPU_VGPR, &is_vgpr));
  if (is_vgpr) {
    *out_low_value = low_value;
    return iree_ok_status();
  }

  int64_t value = 0;
  if (!loom_amdgpu_value_as_i32_constant(context, source_value, &value)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted AMDGPU i32 value that cannot "
                            "materialize as a VGPR operand");
  }
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  return loom_amdgpu_emit_const_u32(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_MOV_B32,
      (uint32_t)(int32_t)value, vgpr_type, out_low_value);
}

static iree_status_t loom_amdgpu_bind_vgpr_u32_lane_constants(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, const uint32_t* lane_bit_patterns,
    uint32_t lane_count) {
  IREE_ASSERT_ARGUMENT(lane_bit_patterns);
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU vector "
                            "constant lane count");
  }

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_value_id_t low_lane_values[LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_MOV_B32,
        lane_bit_patterns[i], lane_type, &low_lane_values[i]));
  }

  if (lane_count == 1) {
    return loom_low_lower_bind_value(context, source_result,
                                     low_lane_values[0]);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), low_lane_values, lane_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, source_result,
                                   loom_low_concat_result(concat_op));
}

static iree_status_t loom_amdgpu_bind_vgpr_i32_lane_constants(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, const int64_t* lane_values,
    uint32_t lane_count) {
  IREE_ASSERT_ARGUMENT(lane_values);
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU vector "
                            "constant lane count");
  }

  uint32_t lane_bit_patterns[LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    if (lane_values[i] < INT32_MIN || lane_values[i] > INT32_MAX) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "preflight accepted out-of-range AMDGPU vector "
                              "constant lane");
    }
    lane_bit_patterns[i] = (uint32_t)(int32_t)lane_values[i];
  }
  return loom_amdgpu_bind_vgpr_u32_lane_constants(
      context, source_op, source_result, lane_bit_patterns, lane_count);
}

static iree_status_t loom_amdgpu_bind_vgpr_f32_lane_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, uint32_t lane_bit_pattern,
    uint32_t lane_count) {
  uint32_t lane_bit_patterns[LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    lane_bit_patterns[i] = lane_bit_pattern;
  }
  return loom_amdgpu_bind_vgpr_u32_lane_constants(
      context, source_op, source_result, lane_bit_patterns, lane_count);
}

static iree_status_t loom_amdgpu_lower_u32_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t descriptor_id, uint32_t bit_pattern,
    loom_value_id_t source_result) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));

  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(context, source_op,
                                                  descriptor_id, bit_pattern,
                                                  result_type, &low_result));
  return loom_low_lower_bind_value(context, source_result, low_result);
}

static iree_status_t loom_amdgpu_lower_i32_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t descriptor_id, loom_attribute_t source_attr,
    loom_value_id_t source_result) {
  const int64_t source_value = source_attr.i64;
  return loom_amdgpu_lower_u32_constant(context, source_op, descriptor_id,
                                        (uint32_t)(int32_t)source_value,
                                        source_result);
}

static iree_status_t loom_amdgpu_lower_index_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_constant(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_S_MOV_B32,
      loom_index_constant_value(source_op),
      loom_index_constant_result(source_op));
}

static iree_status_t loom_amdgpu_lower_scalar_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t source_result = loom_scalar_constant_result(source_op);
  const loom_type_t source_type = loom_module_value_type(module, source_result);
  if (loom_amdgpu_type_is_f32(source_type)) {
    return loom_amdgpu_lower_u32_constant(
        context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_MOV_B32,
        loom_amdgpu_attr_f32_bit_pattern(loom_scalar_constant_value(source_op)),
        source_result);
  }

  const uint64_t descriptor_id =
      loom_amdgpu_value_prefers_vgpr(context, source_result)
          ? AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_MOV_B32
          : AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_S_MOV_B32;
  return loom_amdgpu_lower_i32_constant(context, source_op, descriptor_id,
                                        loom_scalar_constant_value(source_op),
                                        source_result);
}

static iree_status_t loom_amdgpu_lower_vector_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t source_result = loom_vector_constant_result(source_op);
  const loom_type_t source_type = loom_module_value_type(module, source_result);
  const uint32_t i32_lane_count =
      loom_amdgpu_vector_i32_lane_count(source_type);
  if (i32_lane_count == 1) {
    return loom_amdgpu_lower_i32_constant(
        context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_MOV_B32,
        loom_vector_constant_value(source_op), source_result);
  }
  if (i32_lane_count > 1) {
    const int64_t source_value = loom_vector_constant_value(source_op).i64;
    int64_t lane_values[LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES];
    for (uint32_t i = 0; i < i32_lane_count; ++i) {
      lane_values[i] = source_value;
    }
    return loom_amdgpu_bind_vgpr_i32_lane_constants(
        context, source_op, source_result, lane_values, i32_lane_count);
  }

  const uint32_t f32_lane_count =
      loom_amdgpu_vector_f32_lane_count(source_type);
  if (f32_lane_count != 0) {
    return loom_amdgpu_bind_vgpr_f32_lane_constant(
        context, source_op, source_result,
        loom_amdgpu_attr_f32_bit_pattern(loom_vector_constant_value(source_op)),
        f32_lane_count);
  }

  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "preflight accepted unsupported AMDGPU vector "
                          "constant lane count");
}

static iree_status_t loom_amdgpu_lower_vector_iota(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t source_result = loom_vector_iota_result(source_op);
  const uint32_t lane_count = loom_amdgpu_vector_i32_lane_count(
      loom_module_value_type(module, source_result));
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU vector "
                            "iota lane count");
  }
  int64_t base = 0;
  int64_t step = 0;
  if (!loom_amdgpu_value_as_i32_constant(
          context, loom_vector_iota_base(source_op), &base) ||
      !loom_amdgpu_value_as_i32_constant(
          context, loom_vector_iota_step(source_op), &step)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU vector "
                            "iota base/step");
  }

  int64_t lane_values[LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    if (!loom_amdgpu_iota_i32_lane_value(base, step, i, &lane_values[i])) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "preflight accepted out-of-range AMDGPU vector "
                              "iota lane");
    }
  }
  return loom_amdgpu_bind_vgpr_i32_lane_constants(
      context, source_op, source_result, lane_values, lane_count);
}

static iree_status_t loom_amdgpu_lower_binary_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t descriptor_id, loom_value_id_t source_lhs,
    loom_value_id_t source_rhs, loom_value_id_t source_result) {
  loom_value_id_t low_operands[2] = {LOOM_VALUE_ID_INVALID,
                                     LOOM_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_lhs, &low_operands[0]));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_rhs, &low_operands[1]));

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));

  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_id, low_operands,
      IREE_ARRAYSIZE(low_operands), loom_make_named_attr_slice(NULL, 0),
      &result_type, 1, &low_op));
  return loom_low_lower_bind_value(
      context, source_result,
      loom_value_slice_get(loom_low_op_results(low_op), 0));
}

static iree_status_t loom_amdgpu_lower_ternary_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t descriptor_id, loom_value_id_t source_a, loom_value_id_t source_b,
    loom_value_id_t source_c, loom_value_id_t source_result) {
  loom_value_id_t low_operands[3] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_a, &low_operands[0]));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_b, &low_operands[1]));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_c, &low_operands[2]));

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));

  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_id, low_operands,
      IREE_ARRAYSIZE(low_operands), loom_make_named_attr_slice(NULL, 0),
      &result_type, 1, &low_op));
  return loom_low_lower_bind_value(
      context, source_result,
      loom_value_slice_get(loom_low_op_results(low_op), 0));
}

static iree_status_t loom_amdgpu_emit_low_slice(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t offset, loom_type_t result_type,
    loom_value_id_t* out_value_id) {
  IREE_ASSERT_ARGUMENT(out_value_id);
  *out_value_id = LOOM_VALUE_ID_INVALID;
  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(
      loom_low_lower_context_builder(context), low_source, offset, result_type,
      source_op->location, &slice_op));
  *out_value_id = loom_low_slice_result(slice_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lower_vector_binary_op_ordered(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t descriptor_id, loom_value_id_t source_lhs,
    loom_value_id_t source_rhs, loom_value_id_t source_result,
    bool swap_operands) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const uint32_t lane_count = loom_amdgpu_vector_32bit_lane_count(
      loom_module_value_type(module, source_result));
  if (lane_count == 1) {
    if (swap_operands) {
      return loom_amdgpu_lower_binary_op(context, source_op, descriptor_id,
                                         source_rhs, source_lhs, source_result);
    }
    return loom_amdgpu_lower_binary_op(context, source_op, descriptor_id,
                                       source_lhs, source_rhs, source_result);
  }
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU vector "
                            "binary lane count");
  }

  loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_lhs, &low_lhs));
  loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_rhs, &low_rhs));

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_value_id_t lane_results[LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    loom_value_id_t lane_lhs = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, source_op, low_lhs,
                                                    i, lane_type, &lane_lhs));
    loom_value_id_t lane_rhs = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, source_op, low_rhs,
                                                    i, lane_type, &lane_rhs));
    loom_value_id_t operands[] = {
        swap_operands ? lane_rhs : lane_lhs,
        swap_operands ? lane_lhs : lane_rhs,
    };
    loom_op_t* lane_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, descriptor_id, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(NULL, 0), &lane_type, 1, &lane_op));
    lane_results[i] = loom_value_slice_get(loom_low_op_results(lane_op), 0);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lane_results, lane_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, source_result,
                                   loom_low_concat_result(concat_op));
}

static iree_status_t loom_amdgpu_lower_vector_binary_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t descriptor_id, loom_value_id_t source_lhs,
    loom_value_id_t source_rhs, loom_value_id_t source_result) {
  return loom_amdgpu_lower_vector_binary_op_ordered(
      context, source_op, descriptor_id, source_lhs, source_rhs, source_result,
      /*swap_operands=*/false);
}

static iree_status_t loom_amdgpu_lower_vector_ternary_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t descriptor_id, loom_value_id_t source_a, loom_value_id_t source_b,
    loom_value_id_t source_c, loom_value_id_t source_result) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const uint32_t lane_count = loom_amdgpu_vector_32bit_lane_count(
      loom_module_value_type(module, source_result));
  if (lane_count == 1) {
    return loom_amdgpu_lower_ternary_op(context, source_op, descriptor_id,
                                        source_a, source_b, source_c,
                                        source_result);
  }
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU vector "
                            "ternary lane count");
  }

  loom_value_id_t low_a = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, source_a, &low_a));
  loom_value_id_t low_b = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, source_b, &low_b));
  loom_value_id_t low_c = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, source_c, &low_c));

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_value_id_t lane_results[LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    loom_value_id_t lane_a = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, source_op, low_a,
                                                    i, lane_type, &lane_a));
    loom_value_id_t lane_b = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, source_op, low_b,
                                                    i, lane_type, &lane_b));
    loom_value_id_t lane_c = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, source_op, low_c,
                                                    i, lane_type, &lane_c));
    loom_value_id_t operands[] = {
        lane_a,
        lane_b,
        lane_c,
    };
    loom_op_t* lane_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, descriptor_id, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(NULL, 0), &lane_type, 1, &lane_op));
    lane_results[i] = loom_value_slice_get(loom_low_op_results(lane_op), 0);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lane_results, lane_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, source_result,
                                   loom_low_concat_result(concat_op));
}

static iree_status_t loom_amdgpu_lower_i32_binary_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t scalar_descriptor_id, uint64_t vector_descriptor_id,
    loom_value_id_t source_lhs, loom_value_id_t source_rhs,
    loom_value_id_t source_result, bool swap_vector_operands) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  bool result_is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, result_type, AMDGPU_GFX11_CORE_REG_CLASS_ID_AMDGPU_VGPR,
      &result_is_vgpr));
  if (result_is_vgpr) {
    const loom_value_id_t first_source =
        swap_vector_operands ? source_rhs : source_lhs;
    const loom_value_id_t second_source =
        swap_vector_operands ? source_lhs : source_rhs;
    loom_value_id_t low_operands[2] = {
        LOOM_VALUE_ID_INVALID,
        LOOM_VALUE_ID_INVALID,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
        context, source_op, first_source, &low_operands[0]));
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
        context, source_op, second_source, &low_operands[1]));
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, vector_descriptor_id, low_operands,
        IREE_ARRAYSIZE(low_operands), loom_make_named_attr_slice(NULL, 0),
        &result_type, 1, &low_op));
    return loom_low_lower_bind_value(
        context, source_result,
        loom_value_slice_get(loom_low_op_results(low_op), 0));
  }
  if (scalar_descriptor_id == LOOM_LOW_DESCRIPTOR_ID_NONE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted uniform AMDGPU scalar integer "
                            "op without a scalar descriptor");
  }
  return loom_amdgpu_lower_binary_op(context, source_op, scalar_descriptor_id,
                                     source_lhs, source_rhs, source_result);
}

static iree_status_t loom_amdgpu_lower_addi(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_S_ADD_U32,
      AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_ADD_U32,
      loom_scalar_addi_lhs(source_op), loom_scalar_addi_rhs(source_op),
      loom_scalar_addi_result(source_op), /*swap_vector_operands=*/false);
}

static iree_status_t loom_amdgpu_lower_subi(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_S_SUB_U32,
      AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_SUB_U32,
      loom_scalar_subi_lhs(source_op), loom_scalar_subi_rhs(source_op),
      loom_scalar_subi_result(source_op), /*swap_vector_operands=*/false);
}

static iree_status_t loom_amdgpu_lower_muli(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, LOOM_LOW_DESCRIPTOR_ID_NONE,
      AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_MUL_LO_U32,
      loom_scalar_muli_lhs(source_op), loom_scalar_muli_rhs(source_op),
      loom_scalar_muli_result(source_op), /*swap_vector_operands=*/false);
}

static iree_status_t loom_amdgpu_lower_andi(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_S_AND_B32,
      AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_AND_B32,
      loom_scalar_andi_lhs(source_op), loom_scalar_andi_rhs(source_op),
      loom_scalar_andi_result(source_op), /*swap_vector_operands=*/false);
}

static iree_status_t loom_amdgpu_lower_ori(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_S_OR_B32,
      AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_OR_B32,
      loom_scalar_ori_lhs(source_op), loom_scalar_ori_rhs(source_op),
      loom_scalar_ori_result(source_op), /*swap_vector_operands=*/false);
}

static iree_status_t loom_amdgpu_lower_xori(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_S_XOR_B32,
      AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_XOR_B32,
      loom_scalar_xori_lhs(source_op), loom_scalar_xori_rhs(source_op),
      loom_scalar_xori_result(source_op), /*swap_vector_operands=*/false);
}

static iree_status_t loom_amdgpu_lower_shli(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_S_LSHL_B32,
      AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_LSHLREV_B32,
      loom_scalar_shli_lhs(source_op), loom_scalar_shli_rhs(source_op),
      loom_scalar_shli_result(source_op), /*swap_vector_operands=*/true);
}

static iree_status_t loom_amdgpu_lower_shrsi(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_S_ASHR_I32,
      AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_ASHRREV_I32,
      loom_scalar_shrsi_lhs(source_op), loom_scalar_shrsi_rhs(source_op),
      loom_scalar_shrsi_result(source_op), /*swap_vector_operands=*/true);
}

static iree_status_t loom_amdgpu_lower_shrui(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_S_LSHR_B32,
      AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_LSHRREV_B32,
      loom_scalar_shrui_lhs(source_op), loom_scalar_shrui_rhs(source_op),
      loom_scalar_shrui_result(source_op), /*swap_vector_operands=*/true);
}

static iree_status_t loom_amdgpu_lower_vector_addi(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_ADD_U32,
      loom_vector_addi_lhs(source_op), loom_vector_addi_rhs(source_op),
      loom_vector_addi_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_subi(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_SUB_U32,
      loom_vector_subi_lhs(source_op), loom_vector_subi_rhs(source_op),
      loom_vector_subi_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_muli(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_MUL_LO_U32,
      loom_vector_muli_lhs(source_op), loom_vector_muli_rhs(source_op),
      loom_vector_muli_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_andi(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_AND_B32,
      loom_vector_andi_lhs(source_op), loom_vector_andi_rhs(source_op),
      loom_vector_andi_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_ori(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_OR_B32,
      loom_vector_ori_lhs(source_op), loom_vector_ori_rhs(source_op),
      loom_vector_ori_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_xori(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_XOR_B32,
      loom_vector_xori_lhs(source_op), loom_vector_xori_rhs(source_op),
      loom_vector_xori_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_shli(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op_ordered(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_LSHLREV_B32,
      loom_vector_shli_lhs(source_op), loom_vector_shli_rhs(source_op),
      loom_vector_shli_result(source_op), /*swap_operands=*/true);
}

static iree_status_t loom_amdgpu_lower_vector_shrsi(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op_ordered(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_ASHRREV_I32,
      loom_vector_shrsi_lhs(source_op), loom_vector_shrsi_rhs(source_op),
      loom_vector_shrsi_result(source_op), /*swap_operands=*/true);
}

static iree_status_t loom_amdgpu_lower_vector_shrui(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op_ordered(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_LSHRREV_B32,
      loom_vector_shrui_lhs(source_op), loom_vector_shrui_rhs(source_op),
      loom_vector_shrui_result(source_op), /*swap_operands=*/true);
}

static iree_status_t loom_amdgpu_lower_vector_addf(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_ADD_F32,
      loom_vector_addf_lhs(source_op), loom_vector_addf_rhs(source_op),
      loom_vector_addf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_subf(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_SUB_F32,
      loom_vector_subf_lhs(source_op), loom_vector_subf_rhs(source_op),
      loom_vector_subf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_mulf(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_MUL_F32,
      loom_vector_mulf_lhs(source_op), loom_vector_mulf_rhs(source_op),
      loom_vector_mulf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_fmaf(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_ternary_op(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_FMA_F32,
      loom_vector_fmaf_a(source_op), loom_vector_fmaf_b(source_op),
      loom_vector_fmaf_c(source_op), loom_vector_fmaf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_addf(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_binary_op(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_ADD_F32,
      loom_scalar_addf_lhs(source_op), loom_scalar_addf_rhs(source_op),
      loom_scalar_addf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_subf(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_binary_op(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_SUB_F32,
      loom_scalar_subf_lhs(source_op), loom_scalar_subf_rhs(source_op),
      loom_scalar_subf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_mulf(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_binary_op(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_MUL_F32,
      loom_scalar_mulf_lhs(source_op), loom_scalar_mulf_rhs(source_op),
      loom_scalar_mulf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_fmaf(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_ternary_op(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_FMA_F32,
      loom_scalar_fmaf_a(source_op), loom_scalar_fmaf_b(source_op),
      loom_scalar_fmaf_c(source_op), loom_scalar_fmaf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  uint32_t lane_offset = 0;
  if (!loom_amdgpu_vector_extract_select(context, source_op, &lane_offset)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU "
                            "vector.extract");
  }

  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_extract_source(source_op), &low_source));
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type =
      loom_module_value_type(module, loom_vector_extract_source(source_op));
  const uint32_t lane_count = loom_amdgpu_vector_32bit_lane_count(source_type);
  if (lane_count == 1) {
    return loom_low_lower_bind_value(
        context, loom_vector_extract_result(source_op), low_source);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, loom_vector_extract_result(source_op), &result_type));
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_source, lane_offset, result_type, &low_result));
  return loom_low_lower_bind_value(
      context, loom_vector_extract_result(source_op), low_result);
}

static iree_status_t loom_amdgpu_lower_vector_from_elements(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  if (!loom_amdgpu_can_lower_vector_from_elements(context, source_op)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU "
                            "vector.from_elements");
  }
  loom_value_slice_t elements = loom_vector_from_elements_elements(source_op);
  const loom_value_id_t source_result =
      loom_vector_from_elements_result(source_op);
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_result_type =
      loom_module_value_type(module, source_result);
  const bool result_is_i32_vector =
      loom_type_element_type(source_result_type) == LOOM_SCALAR_TYPE_I32;
  loom_value_id_t low_elements[LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES] = {0};
  for (uint32_t i = 0; i < elements.count; ++i) {
    if (result_is_i32_vector) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
          context, source_op, elements.values[i], &low_elements[i]));
    } else {
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
          context, elements.values[i], &low_elements[i]));
    }
  }
  if (elements.count == 1) {
    return loom_low_lower_bind_value(context, source_result, low_elements[0]);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), low_elements, elements.count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, source_result,
                                   loom_low_concat_result(concat_op));
}

static iree_status_t loom_amdgpu_lower_buffer_view(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_buffer = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_buffer_view_buffer(source_op), &low_buffer));
  return loom_low_lower_bind_value(context, loom_buffer_view_result(source_op),
                                   low_buffer);
}

static iree_status_t loom_amdgpu_lower_buffer_assume_memory_space(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_buffer = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_buffer_assume_memory_space_buffer(source_op), &low_buffer));
  return loom_low_lower_bind_value(
      context, loom_buffer_assume_memory_space_result(source_op), low_buffer);
}

static iree_status_t loom_amdgpu_emit_workitem_id_x_live_in(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t* out_low_value_id) {
  IREE_ASSERT_ARGUMENT(out_low_value_id);
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_string_id_t source_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_intern(
      context, IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_X_SOURCE),
      &source_id));
  loom_op_t* live_in_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_live_in_build(loom_low_lower_context_builder(context), source_id,
                             loom_make_named_attr_slice(NULL, 0), vgpr_type,
                             source_op->location, &live_in_op));
  *out_low_value_id = loom_low_live_in_result(live_in_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_preamble(
    void* user_data, loom_low_lower_context_t* context) {
  (void)user_data;
  loom_func_like_t source_function =
      loom_low_lower_context_source_function(context);
  loom_region_t* source_body = loom_func_like_body(source_function);
  if (source_body == NULL) {
    return iree_ok_status();
  }
  const loom_op_t* first_workitem_id_x_op = NULL;
  for (uint16_t block_index = 0; block_index < source_body->block_count;
       ++block_index) {
    const loom_block_t* source_block =
        loom_region_const_block(source_body, block_index);
    const loom_op_t* source_op = NULL;
    loom_block_for_each_op(source_block, source_op) {
      if (loom_kernel_workitem_id_isa(source_op) &&
          loom_kernel_workitem_id_dimension(source_op) ==
              LOOM_KERNEL_WORKITEM_ID_DIMENSION_X) {
        first_workitem_id_x_op = source_op;
        break;
      }
    }
    if (first_workitem_id_x_op != NULL) {
      break;
    }
  }
  if (first_workitem_id_x_op == NULL) {
    return iree_ok_status();
  }

  loom_value_id_t low_workitem_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workitem_id_x_live_in(
      context, first_workitem_id_x_op, &low_workitem_id));
  for (uint16_t block_index = 0; block_index < source_body->block_count;
       ++block_index) {
    const loom_block_t* source_block =
        loom_region_const_block(source_body, block_index);
    const loom_op_t* source_op = NULL;
    loom_block_for_each_op(source_block, source_op) {
      if (!loom_kernel_workitem_id_isa(source_op) ||
          loom_kernel_workitem_id_dimension(source_op) !=
              LOOM_KERNEL_WORKITEM_ID_DIMENSION_X) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
          context, loom_kernel_workitem_id_result(source_op), low_workitem_id));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lower_workitem_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  return loom_low_lower_lookup_value(
      context, loom_kernel_workitem_id_result(source_op), &low_result);
}

static iree_status_t loom_amdgpu_emit_memory_vaddr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_t* access, loom_value_id_t* out_low_vaddr) {
  IREE_ASSERT_ARGUMENT(out_low_vaddr);
  *out_low_vaddr = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  if (access->dynamic_index == LOOM_VALUE_ID_INVALID) {
    return loom_amdgpu_emit_const_u32(
        context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_MOV_B32, 0,
        vgpr_type, out_low_vaddr);
  }

  loom_value_id_t low_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, access->dynamic_index, &low_index));
  loom_value_id_t low_byte_stride = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_MOV_B32,
      access->dynamic_index_byte_stride, vgpr_type, &low_byte_stride));
  loom_value_id_t operands[] = {
      low_index,
      low_byte_stride,
  };
  loom_op_t* low_offset_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_V_MUL_LO_U32,
      operands, IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0),
      &vgpr_type, 1, &low_offset_op));
  *out_low_vaddr = loom_value_slice_get(loom_low_op_results(low_offset_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lower_vector_load(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_load_view(source_op), &low_resource));

  loom_amdgpu_memory_access_t access;
  if (!loom_amdgpu_load_memory_access_select(context, source_op, &access)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU vector.load "
                            "memory access");
  }
  loom_value_id_t low_vaddr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_memory_vaddr(context, source_op, &access, &low_vaddr));

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_S_MOV_B32,
      access.scalar_byte_offset, sgpr_type, &low_soffset));

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, loom_vector_load_result(source_op), &result_type));

  loom_string_id_t offset_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_intern(context, IREE_SV("offset"), &offset_id));
  loom_value_id_t operands[] = {
      low_resource,
      low_vaddr,
      low_soffset,
  };
  loom_named_attr_t attrs[] = {
      {
          .name_id = offset_id,
          .value = loom_attr_i64(access.immediate_byte_offset),
      },
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, access.descriptor_id, operands,
      IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, IREE_ARRAYSIZE(attrs)), &result_type, 1,
      &low_op));
  return loom_low_lower_bind_value(
      context, loom_vector_load_result(source_op),
      loom_value_slice_get(loom_low_op_results(low_op), 0));
}

static iree_status_t loom_amdgpu_lower_vector_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_store_value(source_op), &low_value));
  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_store_view(source_op), &low_resource));

  loom_amdgpu_memory_access_t access;
  if (!loom_amdgpu_store_memory_access_select(context, source_op, &access)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU "
                            "vector.store memory access");
  }
  loom_value_id_t low_vaddr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_memory_vaddr(context, source_op, &access, &low_vaddr));

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, AMDGPU_GFX11_CORE_DESCRIPTOR_ID_AMDGPU_S_MOV_B32,
      access.scalar_byte_offset, sgpr_type, &low_soffset));

  loom_string_id_t offset_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_intern(context, IREE_SV("offset"), &offset_id));
  loom_value_id_t operands[] = {
      low_value,
      low_resource,
      low_vaddr,
      low_soffset,
  };
  loom_named_attr_t attrs[] = {
      {
          .name_id = offset_id,
          .value = loom_attr_i64(access.immediate_byte_offset),
      },
  };
  loom_op_t* low_op = NULL;
  return loom_amdgpu_emit_low_op(
      context, source_op, access.descriptor_id, operands,
      IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, IREE_ARRAYSIZE(attrs)),
      /*result_types=*/NULL, /*result_count=*/0, &low_op);
}

static iree_status_t loom_amdgpu_try_lower_op(void* user_data,
                                              loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              bool* out_handled) {
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_can_lower_op(user_data, context, source_op, out_handled));
  if (!*out_handled) {
    return iree_ok_status();
  }

  switch (source_op->kind) {
    case LOOM_OP_INDEX_CONSTANT:
      return loom_amdgpu_lower_index_constant(context, source_op);
    case LOOM_OP_SCALAR_CONSTANT:
      return loom_amdgpu_lower_scalar_constant(context, source_op);
    case LOOM_OP_VECTOR_CONSTANT:
      return loom_amdgpu_lower_vector_constant(context, source_op);
    case LOOM_OP_VECTOR_IOTA:
      return loom_amdgpu_lower_vector_iota(context, source_op);
    case LOOM_OP_BUFFER_ASSUME_MEMORY_SPACE:
      return loom_amdgpu_lower_buffer_assume_memory_space(context, source_op);
    case LOOM_OP_BUFFER_VIEW:
      return loom_amdgpu_lower_buffer_view(context, source_op);
    case LOOM_OP_KERNEL_WORKITEM_ID:
      return loom_amdgpu_lower_workitem_id(context, source_op);
    case LOOM_OP_SCALAR_ADDI:
      return loom_amdgpu_lower_addi(context, source_op);
    case LOOM_OP_SCALAR_SUBI:
      return loom_amdgpu_lower_subi(context, source_op);
    case LOOM_OP_SCALAR_MULI:
      return loom_amdgpu_lower_muli(context, source_op);
    case LOOM_OP_SCALAR_ANDI:
      return loom_amdgpu_lower_andi(context, source_op);
    case LOOM_OP_SCALAR_ORI:
      return loom_amdgpu_lower_ori(context, source_op);
    case LOOM_OP_SCALAR_XORI:
      return loom_amdgpu_lower_xori(context, source_op);
    case LOOM_OP_SCALAR_SHLI:
      return loom_amdgpu_lower_shli(context, source_op);
    case LOOM_OP_SCALAR_SHRSI:
      return loom_amdgpu_lower_shrsi(context, source_op);
    case LOOM_OP_SCALAR_SHRUI:
      return loom_amdgpu_lower_shrui(context, source_op);
    case LOOM_OP_SCALAR_ADDF:
      return loom_amdgpu_lower_addf(context, source_op);
    case LOOM_OP_SCALAR_SUBF:
      return loom_amdgpu_lower_subf(context, source_op);
    case LOOM_OP_SCALAR_MULF:
      return loom_amdgpu_lower_mulf(context, source_op);
    case LOOM_OP_SCALAR_FMAF:
      return loom_amdgpu_lower_fmaf(context, source_op);
    case LOOM_OP_VECTOR_ADDI:
      return loom_amdgpu_lower_vector_addi(context, source_op);
    case LOOM_OP_VECTOR_SUBI:
      return loom_amdgpu_lower_vector_subi(context, source_op);
    case LOOM_OP_VECTOR_MULI:
      return loom_amdgpu_lower_vector_muli(context, source_op);
    case LOOM_OP_VECTOR_ANDI:
      return loom_amdgpu_lower_vector_andi(context, source_op);
    case LOOM_OP_VECTOR_ORI:
      return loom_amdgpu_lower_vector_ori(context, source_op);
    case LOOM_OP_VECTOR_XORI:
      return loom_amdgpu_lower_vector_xori(context, source_op);
    case LOOM_OP_VECTOR_SHLI:
      return loom_amdgpu_lower_vector_shli(context, source_op);
    case LOOM_OP_VECTOR_SHRSI:
      return loom_amdgpu_lower_vector_shrsi(context, source_op);
    case LOOM_OP_VECTOR_SHRUI:
      return loom_amdgpu_lower_vector_shrui(context, source_op);
    case LOOM_OP_VECTOR_ADDF:
      return loom_amdgpu_lower_vector_addf(context, source_op);
    case LOOM_OP_VECTOR_SUBF:
      return loom_amdgpu_lower_vector_subf(context, source_op);
    case LOOM_OP_VECTOR_MULF:
      return loom_amdgpu_lower_vector_mulf(context, source_op);
    case LOOM_OP_VECTOR_FMAF:
      return loom_amdgpu_lower_vector_fmaf(context, source_op);
    case LOOM_OP_VECTOR_EXTRACT:
      return loom_amdgpu_lower_vector_extract(context, source_op);
    case LOOM_OP_VECTOR_FROM_ELEMENTS:
      return loom_amdgpu_lower_vector_from_elements(context, source_op);
    case LOOM_OP_VECTOR_LOAD:
      return loom_amdgpu_lower_vector_load(context, source_op);
    case LOOM_OP_VECTOR_STORE:
      return loom_amdgpu_lower_vector_store(context, source_op);
    default:
      return iree_ok_status();
  }
}

static bool loom_amdgpu_low_legality_bundle_is_amdgpu(
    const loom_target_bundle_t* bundle) {
  return bundle != NULL && bundle->config != NULL &&
         iree_string_view_starts_with(bundle->config->contract_set_key,
                                      IREE_SV("amdgpu."));
}

static iree_status_t loom_amdgpu_low_legality_verify_buffer_view(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  if (!loom_amdgpu_type_is_32bit_view(
          loom_module_value_type(module, loom_buffer_view_result(op)))) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("memory"), loom_op_name(module, op),
        IREE_SV("AMDGPU buffer memory lowering currently requires typed views "
                "over 32-bit elements"));
  }
  int64_t unused_byte_offset = 0;
  if (!loom_amdgpu_module_value_as_exact_index_constant(
          module, loom_buffer_view_byte_offset(op), &unused_byte_offset) ||
      unused_byte_offset < 0) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("memory"), loom_op_name(module, op),
        IREE_SV("AMDGPU HAL buffer views currently require exact non-negative "
                "static byte offsets"));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_low_legality_verify_vector_memory(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_target_low_legality_descriptor_set(context);
  loom_value_slice_t dynamic_indices = {0};
  loom_attribute_t static_indices = {0};
  loom_value_id_t view_value_id = LOOM_VALUE_ID_INVALID;
  loom_type_t view_type = loom_type_none();
  loom_type_t vector_type = loom_type_none();
  loom_amdgpu_buffer_memory_kind_t kind = LOOM_AMDGPU_BUFFER_MEMORY_LOAD;
  switch (op->kind) {
    case LOOM_OP_VECTOR_LOAD:
      dynamic_indices = loom_vector_load_indices(op);
      static_indices = loom_vector_load_static_indices(op);
      view_value_id = loom_vector_load_view(op);
      view_type = loom_module_value_type(module, loom_vector_load_view(op));
      vector_type = loom_module_value_type(module, loom_vector_load_result(op));
      kind = LOOM_AMDGPU_BUFFER_MEMORY_LOAD;
      break;
    case LOOM_OP_VECTOR_STORE:
      dynamic_indices = loom_vector_store_indices(op);
      static_indices = loom_vector_store_static_indices(op);
      view_value_id = loom_vector_store_view(op);
      view_type = loom_module_value_type(module, loom_vector_store_view(op));
      vector_type = loom_module_value_type(module, loom_vector_store_value(op));
      kind = LOOM_AMDGPU_BUFFER_MEMORY_STORE;
      break;
    default:
      *out_handled = false;
      return iree_ok_status();
  }

  loom_amdgpu_memory_access_t access = {0};
  loom_amdgpu_memory_access_diagnostic_t diagnostic = {0};
  if (!loom_amdgpu_memory_access_select(
          module, descriptor_set, view_value_id, dynamic_indices,
          static_indices, view_type, vector_type, kind, &access, &diagnostic)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("memory"), loom_op_name(module, op),
        loom_amdgpu_memory_access_rejection_detail(diagnostic.rejection_bits));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_low_legality_try_verify_op(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  *out_handled = false;
  switch (op->kind) {
    case LOOM_OP_BUFFER_VIEW:
      return loom_amdgpu_low_legality_verify_buffer_view(provider, context, op,
                                                         out_handled);
    case LOOM_OP_VECTOR_LOAD:
    case LOOM_OP_VECTOR_STORE:
      return loom_amdgpu_low_legality_verify_vector_memory(provider, context,
                                                           op, out_handled);
    default:
      return iree_ok_status();
  }
}

static const loom_low_lower_policy_t kAmdgpuLowLowerPolicy = {
    .name = IREE_SVL("amdgpu-register-lower"),
    .map_type = {.fn = loom_amdgpu_map_type, .user_data = NULL},
    .map_value = {.fn = loom_amdgpu_map_value, .user_data = NULL},
    .map_argument = {.fn = loom_amdgpu_map_argument, .user_data = NULL},
    .emit_preamble = {.fn = loom_amdgpu_emit_preamble, .user_data = NULL},
    .can_lower_op = {.fn = loom_amdgpu_can_lower_op, .user_data = NULL},
    .try_lower_op = {.fn = loom_amdgpu_try_lower_op, .user_data = NULL},
};

const loom_target_low_legality_provider_t
    loom_amdgpu_low_legality_provider_storage = {
        .name = IREE_SVL("amdgpu"),
        .try_verify_op = loom_amdgpu_low_legality_try_verify_op,
};

const loom_low_lower_policy_t* loom_amdgpu_low_lower_policy(void) {
  return &kAmdgpuLowLowerPolicy;
}

const loom_target_low_legality_provider_t* loom_amdgpu_low_legality_provider(
    void) {
  return &loom_amdgpu_low_legality_provider_storage;
}

void loom_amdgpu_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("amdgpu.gfx950.core"),
          .policy = &kAmdgpuLowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("amdgpu.gfx11.core"),
          .policy = &kAmdgpuLowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("amdgpu.gfx12.core"),
          .policy = &kAmdgpuLowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("amdgpu.gfx1250.core"),
          .policy = &kAmdgpuLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
