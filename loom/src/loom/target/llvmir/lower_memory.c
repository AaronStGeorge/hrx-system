// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/facts.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/view/ops.h"
#include "loom/target/llvmir/intrinsics_builtin.h"
#include "loom/target/llvmir/lower_internal.h"

static loom_value_fact_memory_space_t
loom_llvmir_lowering_memory_space_from_buffer_attr(uint8_t value) {
  switch ((loom_buffer_memory_space_t)value) {
    case LOOM_BUFFER_MEMORY_SPACE_GLOBAL:
      return LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL;
    case LOOM_BUFFER_MEMORY_SPACE_WORKGROUP:
      return LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP;
    case LOOM_BUFFER_MEMORY_SPACE_PRIVATE:
      return LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE;
    case LOOM_BUFFER_MEMORY_SPACE_CONSTANT:
      return LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT;
    case LOOM_BUFFER_MEMORY_SPACE_HOST:
      return LOOM_VALUE_FACT_MEMORY_SPACE_HOST;
    case LOOM_BUFFER_MEMORY_SPACE_DESCRIPTOR:
      return LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR;
    case LOOM_BUFFER_MEMORY_SPACE_UNKNOWN:
    case LOOM_BUFFER_MEMORY_SPACE_COUNT_:
      return LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN;
  }
  return LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN;
}

static iree_status_t loom_llvmir_lowering_address_space_from_memory_space(
    const loom_llvmir_lowering_state_t* state,
    loom_value_fact_memory_space_t memory_space, uint32_t* out_address_space) {
  const loom_llvmir_target_address_spaces_t* address_spaces =
      &state->target_profile->target_env->address_spaces;
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
      *out_address_space = address_spaces->generic;
      return iree_ok_status();
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
      *out_address_space = address_spaces->global;
      return iree_ok_status();
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
      *out_address_space = address_spaces->local;
      return iree_ok_status();
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
      *out_address_space = address_spaces->private_memory;
      return iree_ok_status();
    case LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT:
      *out_address_space = address_spaces->constant;
      return iree_ok_status();
    case LOOM_VALUE_FACT_MEMORY_SPACE_HOST:
      if (state->target_profile->kind ==
          LOOM_LLVMIR_TARGET_PROFILE_HOST_OBJECT) {
        *out_address_space = address_spaces->generic;
        return iree_ok_status();
      }
      break;
    case LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR:
      break;
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "LLVMIR target profile %.*s has no pointer address "
                          "space mapping for Loom memory space %u",
                          (int)state->target_profile->name.size,
                          state->target_profile->name.data,
                          (unsigned)memory_space);
}

static loom_value_facts_t loom_llvmir_lowering_value_facts(
    const loom_llvmir_lowering_state_t* state, loom_value_id_t value_id) {
  if (!state->fact_table) return loom_value_facts_unknown();
  return loom_value_fact_table_lookup(state->fact_table, value_id);
}

static uint64_t loom_llvmir_lowering_min_nonzero_u64(uint64_t lhs,
                                                     uint64_t rhs) {
  if (lhs == 0) return rhs;
  if (rhs == 0) return lhs;
  return lhs < rhs ? lhs : rhs;
}

static uint64_t loom_llvmir_lowering_alignment_from_offset(
    uint64_t root_alignment, loom_value_facts_t offset_facts) {
  if (root_alignment == 0) return 0;
  if (loom_value_facts_is_exact(offset_facts) && offset_facts.range_lo == 0) {
    return root_alignment;
  }
  uint64_t offset_alignment = 0;
  if (offset_facts.known_divisor > 0) {
    offset_alignment = (uint64_t)offset_facts.known_divisor;
  }
  return loom_llvmir_lowering_min_nonzero_u64(root_alignment, offset_alignment);
}

static uint64_t loom_llvmir_lowering_alignment_from_view_reference(
    loom_value_fact_view_reference_t reference) {
  return loom_llvmir_lowering_alignment_from_offset(
      reference.root_minimum_alignment, reference.base_byte_offset);
}

static uint32_t loom_llvmir_lowering_load_store_alignment(
    uint64_t pointer_alignment, int64_t element_byte_count) {
  if (pointer_alignment == 0 || element_byte_count <= 1 ||
      element_byte_count > UINT32_MAX) {
    return 0;
  }
  uint64_t alignment = pointer_alignment < (uint64_t)element_byte_count
                           ? pointer_alignment
                           : (uint64_t)element_byte_count;
  return alignment <= 1 ? 0 : (uint32_t)alignment;
}

static iree_status_t loom_llvmir_lowering_lower_memory_space_pointer_type(
    loom_llvmir_lowering_state_t* state,
    loom_value_fact_memory_space_t memory_space, uint32_t* out_address_space,
    loom_llvmir_type_id_t* out_type_id) {
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_address_space_from_memory_space(
      state, memory_space, out_address_space));
  return loom_llvmir_lowering_get_pointer_type(state, *out_address_space,
                                               out_type_id);
}

static iree_status_t loom_llvmir_lowering_index_constant(
    loom_llvmir_lowering_state_t* state, int64_t value,
    loom_llvmir_value_id_t* out_value_id) {
  loom_llvmir_type_id_t index_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_scalar_type(
      state, LOOM_SCALAR_TYPE_INDEX, &index_type));
  return loom_llvmir_module_add_integer_constant(
      state->target_module, index_type, (uint64_t)value, out_value_id);
}

static iree_status_t loom_llvmir_lowering_index_binop(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    loom_llvmir_binop_t op, loom_llvmir_value_id_t lhs,
    loom_llvmir_value_id_t rhs, loom_llvmir_value_id_t* out_value_id) {
  loom_llvmir_type_id_t index_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_scalar_type(
      state, LOOM_SCALAR_TYPE_INDEX, &index_type));
  return loom_llvmir_build_binop(target_block,
                                 &(loom_llvmir_binop_desc_t){
                                     .result_type = index_type,
                                     .op = op,
                                     .lhs = lhs,
                                     .rhs = rhs,
                                 },
                                 out_value_id);
}

static iree_status_t loom_llvmir_lowering_scale_index(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    loom_llvmir_value_id_t value, int64_t scale,
    loom_llvmir_value_id_t* out_value_id) {
  if (scale < 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "negative view layout strides cannot lower to "
                            "LLVMIR GEP indices");
  }
  if (scale == 1) {
    *out_value_id = value;
    return iree_ok_status();
  }
  loom_llvmir_value_id_t scale_value = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_index_constant(state, scale, &scale_value));
  return loom_llvmir_lowering_index_binop(state, target_block,
                                          LOOM_LLVMIR_BINOP_MUL, value,
                                          scale_value, out_value_id);
}

static iree_status_t loom_llvmir_lowering_add_index_contribution(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    loom_llvmir_value_id_t contribution,
    loom_llvmir_value_id_t* inout_accumulator) {
  if (*inout_accumulator == LOOM_LLVMIR_VALUE_ID_INVALID) {
    *inout_accumulator = contribution;
    return iree_ok_status();
  }
  loom_llvmir_value_id_t sum = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_index_binop(
      state, target_block, LOOM_LLVMIR_BINOP_ADD, *inout_accumulator,
      contribution, &sum));
  *inout_accumulator = sum;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_lowering_static_or_dynamic_index(
    loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_attribute_t static_indices, loom_value_slice_t dynamic_indices,
    uint8_t axis, loom_llvmir_value_id_t* out_value_id) {
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      axis >= static_indices.count) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "memory op is missing full-rank static index metadata");
  }
  uint16_t dynamic_ordinal = 0;
  for (uint8_t i = 0; i <= axis; ++i) {
    int64_t static_index = static_indices.i64_array[i];
    if (static_index != INT64_MIN) {
      if (i == axis) {
        if (static_index < 0) {
          return loom_llvmir_lowering_unsupported_op(
              state, op, "negative static memory indices are unsupported");
        }
        return loom_llvmir_lowering_index_constant(state, static_index,
                                                   out_value_id);
      }
      continue;
    }
    if (i == axis) {
      if (dynamic_ordinal >= dynamic_indices.count) {
        return loom_llvmir_lowering_unsupported_op(
            state, op, "dynamic memory index metadata is inconsistent");
      }
      return loom_llvmir_lowering_lookup_value(
          state, dynamic_indices.values[dynamic_ordinal], out_value_id);
    }
    ++dynamic_ordinal;
  }
  return loom_llvmir_lowering_unsupported_op(
      state, op, "memory index metadata is inconsistent");
}

static iree_status_t loom_llvmir_lowering_view_dense_stride(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op, loom_type_t view_type, uint8_t axis,
    int64_t* inout_static_stride, loom_llvmir_value_id_t* inout_stride) {
  const uint8_t rank = loom_type_rank(view_type);
  *inout_static_stride = 1;
  *inout_stride = LOOM_LLVMIR_VALUE_ID_INVALID;
  for (uint8_t suffix_axis = (uint8_t)(axis + 1); suffix_axis < rank;
       ++suffix_axis) {
    if (!loom_type_dim_is_dynamic_at(view_type, suffix_axis)) {
      int64_t dim = loom_type_dim_static_size_at(view_type, suffix_axis);
      if (dim < 0 || (dim != 0 && *inout_static_stride > INT64_MAX / dim)) {
        return loom_llvmir_lowering_unsupported_op(
            state, op, "dense view stride is not representable");
      }
      *inout_static_stride = dim == 0 ? 0 : *inout_static_stride * dim;
      continue;
    }
    loom_llvmir_value_id_t dim_value = LOOM_LLVMIR_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_value(
        state, loom_type_dim_value_id_at(view_type, suffix_axis), &dim_value));
    if (*inout_stride == LOOM_LLVMIR_VALUE_ID_INVALID) {
      *inout_stride = dim_value;
    } else {
      loom_llvmir_value_id_t product = LOOM_LLVMIR_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_llvmir_lowering_index_binop(
          state, target_block, LOOM_LLVMIR_BINOP_MUL, *inout_stride, dim_value,
          &product));
      *inout_stride = product;
    }
  }
  if (*inout_stride != LOOM_LLVMIR_VALUE_ID_INVALID &&
      *inout_static_stride != 1) {
    IREE_RETURN_IF_ERROR(
        loom_llvmir_lowering_scale_index(state, target_block, *inout_stride,
                                         *inout_static_stride, inout_stride));
    *inout_static_stride = 1;
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_lowering_build_view_element_offset(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op, loom_type_t view_type, loom_attribute_t static_indices,
    loom_value_slice_t dynamic_indices,
    loom_llvmir_value_id_t* out_element_offset) {
  const uint8_t rank = loom_type_rank(view_type);
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      static_indices.count != rank) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "memory op must carry one index position per view axis");
  }
  if (rank == 0) {
    return loom_llvmir_lowering_index_constant(state, 0, out_element_offset);
  }

  loom_value_facts_t stride_storage[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK];
  loom_value_fact_address_layout_t layout = {0};
  const loom_fact_context_t* fact_context =
      state->fact_table ? &state->fact_table->context : NULL;
  if (!loom_encoding_query_type_address_layout(
          fact_context, state->source_module, view_type, stride_storage,
          IREE_ARRAYSIZE(stride_storage), &layout)) {
    return loom_llvmir_lowering_unsupported_type(
        state, view_type,
        "view.load/store need a dense or strided address layout");
  }

  loom_llvmir_value_id_t accumulator = LOOM_LLVMIR_VALUE_ID_INVALID;
  for (uint8_t axis = 0; axis < rank; ++axis) {
    loom_llvmir_value_id_t index = LOOM_LLVMIR_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_llvmir_lowering_static_or_dynamic_index(
        state, op, static_indices, dynamic_indices, axis, &index));

    int64_t static_stride = 1;
    loom_llvmir_value_id_t dynamic_stride = LOOM_LLVMIR_VALUE_ID_INVALID;
    if (layout.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE) {
      IREE_RETURN_IF_ERROR(loom_llvmir_lowering_view_dense_stride(
          state, target_block, op, view_type, axis, &static_stride,
          &dynamic_stride));
    } else if (layout.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED) {
      if (layout.rank != rank || !layout.strides ||
          !loom_value_facts_is_exact(layout.strides[axis])) {
        return loom_llvmir_lowering_unsupported_type(
            state, view_type,
            "dynamic strided view layouts need stride value lowering");
      }
      static_stride = layout.strides[axis].range_lo;
    } else {
      return loom_llvmir_lowering_unsupported_type(
          state, view_type,
          "view.load/store need a dense or strided address layout");
    }

    loom_llvmir_value_id_t contribution = index;
    if (dynamic_stride != LOOM_LLVMIR_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_llvmir_lowering_index_binop(
          state, target_block, LOOM_LLVMIR_BINOP_MUL, contribution,
          dynamic_stride, &contribution));
    }
    IREE_RETURN_IF_ERROR(loom_llvmir_lowering_scale_index(
        state, target_block, contribution, static_stride, &contribution));
    IREE_RETURN_IF_ERROR(loom_llvmir_lowering_add_index_contribution(
        state, target_block, contribution, &accumulator));
  }

  if (accumulator == LOOM_LLVMIR_VALUE_ID_INVALID) {
    return loom_llvmir_lowering_index_constant(state, 0, out_element_offset);
  }
  *out_element_offset = accumulator;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_lowering_build_view_element_pointer(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op, loom_value_id_t view_value_id, loom_type_t view_type,
    loom_attribute_t static_indices, loom_value_slice_t dynamic_indices,
    iree_string_view_t result_name, loom_llvmir_value_id_t* out_pointer,
    uint64_t* out_pointer_alignment) {
  loom_llvmir_value_id_t base_pointer = LOOM_LLVMIR_VALUE_ID_INVALID;
  uint32_t address_space = UINT32_MAX;
  uint64_t base_alignment = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_pointer(
      state, view_value_id, &base_pointer, &address_space, &base_alignment));

  loom_llvmir_value_id_t element_offset = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_build_view_element_offset(
      state, target_block, op, view_type, static_indices, dynamic_indices,
      &element_offset));

  loom_llvmir_type_id_t pointer_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_get_pointer_type(
      state, address_space, &pointer_type));
  loom_llvmir_type_id_t element_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_scalar_type(
      state, loom_type_element_type(view_type), &element_type));
  IREE_RETURN_IF_ERROR(loom_llvmir_build_gep(target_block,
                                             &(loom_llvmir_gep_desc_t){
                                                 .result_name = result_name,
                                                 .result_type = pointer_type,
                                                 .element_type = element_type,
                                                 .base = base_pointer,
                                                 .indices = &element_offset,
                                                 .index_count = 1,
                                             },
                                             out_pointer));
  if (out_pointer_alignment) {
    int32_t bit_width =
        loom_scalar_type_bitwidth(loom_type_element_type(view_type));
    int64_t element_byte_count =
        bit_width > 0 && (bit_width % 8) == 0 ? bit_width / 8 : -1;
    *out_pointer_alignment =
        element_byte_count > 0
            ? loom_llvmir_lowering_min_nonzero_u64(base_alignment,
                                                   (uint64_t)element_byte_count)
            : base_alignment;
  }
  return iree_ok_status();
}

iree_status_t loom_llvmir_lowering_lower_alloca(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_llvmir_value_id_t byte_length = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_value(
      state, loom_buffer_alloca_byte_length(op), &byte_length));

  loom_value_fact_memory_space_t memory_space =
      loom_llvmir_lowering_memory_space_from_buffer_attr(
          loom_buffer_alloca_memory_space(op));
  uint32_t address_space = UINT32_MAX;
  loom_llvmir_type_id_t pointer_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_memory_space_pointer_type(
      state, memory_space, &address_space, &pointer_type));

  loom_llvmir_type_id_t i8_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(state->target_module, 8, &i8_type));

  loom_value_id_t result_value_id = loom_buffer_alloca_result(op);
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  int64_t base_alignment = loom_buffer_alloca_base_alignment(op);
  IREE_RETURN_IF_ERROR(loom_llvmir_build_alloca(
      target_block,
      &(loom_llvmir_alloca_desc_t){
          .result_name =
              loom_llvmir_lowering_value_name(state, result_value_id),
          .result_type = pointer_type,
          .element_type = i8_type,
          .count = byte_length,
          .alignment = base_alignment > 0 ? (uint32_t)base_alignment : 0,
      },
      &result));
  return loom_llvmir_lowering_map_pointer_value(
      state, result_value_id, result, address_space,
      base_alignment > 0 ? (uint64_t)base_alignment : 0);
}

iree_status_t loom_llvmir_lowering_lower_assume_memory_space(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_value_id_t source_value_id = loom_buffer_assume_memory_space_buffer(op);
  loom_value_id_t result_value_id = loom_buffer_assume_memory_space_result(op);
  loom_llvmir_value_id_t source = LOOM_LLVMIR_VALUE_ID_INVALID;
  uint32_t source_address_space = UINT32_MAX;
  uint64_t source_alignment = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_pointer(
      state, source_value_id, &source, &source_address_space,
      &source_alignment));

  loom_value_fact_memory_space_t memory_space =
      loom_llvmir_lowering_memory_space_from_buffer_attr(
          loom_buffer_assume_memory_space_memory_space(op));
  uint32_t result_address_space = UINT32_MAX;
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_memory_space_pointer_type(
      state, memory_space, &result_address_space, &result_type));
  if (source_address_space == result_address_space) {
    return loom_llvmir_lowering_map_pointer_value(
        state, result_value_id, source, result_address_space, source_alignment);
  }

  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_cast(
      target_block,
      &(loom_llvmir_cast_desc_t){
          .result_name =
              loom_llvmir_lowering_value_name(state, result_value_id),
          .result_type = result_type,
          .op = LOOM_LLVMIR_CAST_ADDRESS_SPACE_CAST,
          .value = source,
      },
      &result));
  return loom_llvmir_lowering_map_pointer_value(
      state, result_value_id, result, result_address_space, source_alignment);
}

iree_status_t loom_llvmir_lowering_lower_buffer_view(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_value_id_t buffer_value_id = loom_buffer_view_buffer(op);
  loom_value_id_t result_value_id = loom_buffer_view_result(op);
  loom_llvmir_value_id_t buffer = LOOM_LLVMIR_VALUE_ID_INVALID;
  uint32_t address_space = UINT32_MAX;
  uint64_t buffer_alignment = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_pointer(
      state, buffer_value_id, &buffer, &address_space, &buffer_alignment));

  loom_llvmir_value_id_t byte_offset = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_value(
      state, loom_buffer_view_byte_offset(op), &byte_offset));

  loom_llvmir_type_id_t pointer_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_get_pointer_type(
      state, address_space, &pointer_type));
  loom_llvmir_type_id_t i8_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(state->target_module, 8, &i8_type));

  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_gep(
      target_block,
      &(loom_llvmir_gep_desc_t){
          .result_name =
              loom_llvmir_lowering_value_name(state, result_value_id),
          .result_type = pointer_type,
          .element_type = i8_type,
          .base = buffer,
          .indices = &byte_offset,
          .index_count = 1,
      },
      &result));

  uint64_t result_alignment = loom_llvmir_lowering_alignment_from_offset(
      buffer_alignment, loom_llvmir_lowering_value_facts(
                            state, loom_buffer_view_byte_offset(op)));
  loom_value_fact_view_reference_t view_reference = {0};
  if (state->fact_table &&
      loom_value_facts_query_view_reference(
          &state->fact_table->context,
          loom_value_fact_table_lookup(state->fact_table, result_value_id),
          &view_reference)) {
    result_alignment =
        loom_llvmir_lowering_alignment_from_view_reference(view_reference);
  }
  return loom_llvmir_lowering_map_pointer_value(
      state, result_value_id, result, address_space, result_alignment);
}

iree_status_t loom_llvmir_lowering_lower_subview(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_value_id_t source_value_id = loom_view_subview_source(op);
  loom_value_id_t result_value_id = loom_view_subview_result(op);
  loom_type_t source_type =
      loom_module_value_type(state->source_module, source_value_id);
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  uint64_t result_alignment = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_build_view_element_pointer(
      state, target_block, op, source_value_id, source_type,
      loom_view_subview_static_offsets(op), loom_view_subview_offsets(op),
      loom_llvmir_lowering_value_name(state, result_value_id), &result,
      &result_alignment));

  loom_llvmir_value_id_t source = LOOM_LLVMIR_VALUE_ID_INVALID;
  uint32_t address_space = UINT32_MAX;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_pointer(
      state, source_value_id, &source, &address_space, NULL));
  loom_value_fact_view_reference_t view_reference = {0};
  if (state->fact_table &&
      loom_value_facts_query_view_reference(
          &state->fact_table->context,
          loom_value_fact_table_lookup(state->fact_table, result_value_id),
          &view_reference)) {
    result_alignment =
        loom_llvmir_lowering_alignment_from_view_reference(view_reference);
  }
  return loom_llvmir_lowering_map_pointer_value(
      state, result_value_id, result, address_space, result_alignment);
}

iree_status_t loom_llvmir_lowering_lower_refine(
    loom_llvmir_lowering_state_t* state, const loom_op_t* op) {
  loom_value_id_t source_value_id = loom_view_refine_source(op);
  loom_value_id_t result_value_id = loom_view_refine_result(op);
  loom_llvmir_value_id_t source = LOOM_LLVMIR_VALUE_ID_INVALID;
  uint32_t address_space = UINT32_MAX;
  uint64_t alignment = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_pointer(
      state, source_value_id, &source, &address_space, &alignment));
  return loom_llvmir_lowering_map_pointer_value(state, result_value_id, source,
                                                address_space, alignment);
}

iree_status_t loom_llvmir_lowering_lower_view_load(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_value_id_t view_value_id = loom_view_load_view(op);
  loom_value_id_t result_value_id = loom_view_load_result(op);
  loom_type_t view_type =
      loom_module_value_type(state->source_module, view_value_id);
  loom_llvmir_value_id_t pointer = LOOM_LLVMIR_VALUE_ID_INVALID;
  uint64_t pointer_alignment = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_build_view_element_pointer(
      state, target_block, op, view_value_id, view_type,
      loom_view_load_static_indices(op), loom_view_load_indices(op),
      IREE_SV(""), &pointer, &pointer_alignment));

  loom_type_t result_source_type =
      loom_module_value_type(state->source_module, result_value_id);
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lower_type(state, result_source_type, &result_type));
  int32_t bit_width =
      loom_scalar_type_bitwidth(loom_type_element_type(view_type));
  int64_t element_byte_count =
      bit_width > 0 && (bit_width % 8) == 0 ? bit_width / 8 : -1;
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_load(
      target_block,
      &(loom_llvmir_load_desc_t){
          .result_name =
              loom_llvmir_lowering_value_name(state, result_value_id),
          .result_type = result_type,
          .pointer = pointer,
          .alignment = loom_llvmir_lowering_load_store_alignment(
              pointer_alignment, element_byte_count),
      },
      &result));
  return loom_llvmir_lowering_map_single_result(state, op, result);
}

iree_status_t loom_llvmir_lowering_lower_view_store(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_value_id_t value_id = loom_view_store_value(op);
  loom_value_id_t view_value_id = loom_view_store_view(op);
  loom_type_t view_type =
      loom_module_value_type(state->source_module, view_value_id);
  loom_llvmir_value_id_t value = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lookup_value(state, value_id, &value));

  loom_llvmir_value_id_t pointer = LOOM_LLVMIR_VALUE_ID_INVALID;
  uint64_t pointer_alignment = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_build_view_element_pointer(
      state, target_block, op, view_value_id, view_type,
      loom_view_store_static_indices(op), loom_view_store_indices(op),
      IREE_SV(""), &pointer, &pointer_alignment));

  int32_t bit_width =
      loom_scalar_type_bitwidth(loom_type_element_type(view_type));
  int64_t element_byte_count =
      bit_width > 0 && (bit_width % 8) == 0 ? bit_width / 8 : -1;
  return loom_llvmir_build_store(
      target_block, &(loom_llvmir_store_desc_t){
                        .value = value,
                        .pointer = pointer,
                        .alignment = loom_llvmir_lowering_load_store_alignment(
                            pointer_alignment, element_byte_count),
                    });
}

static iree_status_t loom_llvmir_lowering_prefetch_intent(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    uint32_t* out_rw) {
  switch ((loom_view_prefetch_intent_t)loom_view_prefetch_intent(op)) {
    case LOOM_VIEW_PREFETCH_INTENT_READ:
      *out_rw = 0;
      return iree_ok_status();
    case LOOM_VIEW_PREFETCH_INTENT_WRITE:
      *out_rw = 1;
      return iree_ok_status();
    case LOOM_VIEW_PREFETCH_INTENT_COUNT_:
      break;
  }
  return loom_llvmir_lowering_unsupported_op(state, op,
                                             "unknown view.prefetch intent");
}

static iree_status_t loom_llvmir_lowering_prefetch_locality(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    uint32_t* out_locality) {
  switch ((loom_view_prefetch_locality_t)loom_view_prefetch_locality(op)) {
    case LOOM_VIEW_PREFETCH_LOCALITY_NONE:
      *out_locality = 0;
      return iree_ok_status();
    case LOOM_VIEW_PREFETCH_LOCALITY_L1:
      *out_locality = 1;
      return iree_ok_status();
    case LOOM_VIEW_PREFETCH_LOCALITY_L2:
      *out_locality = 2;
      return iree_ok_status();
    case LOOM_VIEW_PREFETCH_LOCALITY_L3:
      *out_locality = 3;
      return iree_ok_status();
    case LOOM_VIEW_PREFETCH_LOCALITY_COUNT_:
      break;
  }
  return loom_llvmir_lowering_unsupported_op(state, op,
                                             "unknown view.prefetch locality");
}

static iree_status_t loom_llvmir_lowering_prefetch_function(
    loom_llvmir_lowering_state_t* state, uint32_t pointer_address_space,
    loom_llvmir_function_t** out_function) {
  loom_llvmir_function_t* function = NULL;
  for (iree_host_size_t i = 0; i < state->prefetch_function_count; ++i) {
    if (state->prefetch_function_address_spaces[i] == pointer_address_space) {
      function = state->prefetch_functions[i];
      break;
    }
  }
  if (function) {
    *out_function = function;
    return iree_ok_status();
  }
  if (state->prefetch_function_count >=
      IREE_ARRAYSIZE(state->prefetch_functions)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "too many llvm.prefetch address-space overloads");
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_declare_prefetch(
      state->target_module, pointer_address_space, &function));
  iree_host_size_t ordinal = state->prefetch_function_count++;
  state->prefetch_function_address_spaces[ordinal] = pointer_address_space;
  state->prefetch_functions[ordinal] = function;
  *out_function = function;
  return iree_ok_status();
}

iree_status_t loom_llvmir_lowering_lower_view_prefetch(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_value_id_t view_value_id = loom_view_prefetch_view(op);
  loom_type_t view_type =
      loom_module_value_type(state->source_module, view_value_id);

  loom_llvmir_value_id_t view = LOOM_LLVMIR_VALUE_ID_INVALID;
  uint32_t pointer_address_space = UINT32_MAX;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_pointer(
      state, view_value_id, &view, &pointer_address_space, NULL));
  (void)view;

  loom_llvmir_value_id_t pointer = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_build_view_element_pointer(
      state, target_block, op, view_value_id, view_type,
      loom_view_prefetch_static_indices(op), loom_view_prefetch_indices(op),
      IREE_SV(""), &pointer, NULL));

  loom_llvmir_type_id_t i32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(state->target_module, 32, &i32_type));
  uint32_t rw = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_prefetch_intent(state, op, &rw));
  uint32_t locality = 0;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_prefetch_locality(state, op, &locality));
  loom_llvmir_value_id_t rw_value = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t locality_value = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t data_cache_value = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_integer_constant(
      state->target_module, i32_type, rw, &rw_value));
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_integer_constant(
      state->target_module, i32_type, locality, &locality_value));
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_integer_constant(
      state->target_module, i32_type, 1, &data_cache_value));

  loom_llvmir_function_t* prefetch_function = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_prefetch_function(
      state, pointer_address_space, &prefetch_function));
  loom_llvmir_value_id_t args[] = {pointer, rw_value, locality_value,
                                   data_cache_value};
  return loom_llvmir_build_call(
      target_block,
      &(loom_llvmir_call_desc_t){
          .callee = loom_llvmir_function_id(prefetch_function),
          .args = args,
          .arg_count = IREE_ARRAYSIZE(args),
      },
      NULL);
}
