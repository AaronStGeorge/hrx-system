// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/llvmir/lower.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/llvmir/builder.h"
#include "loom/util/fact_table.h"

static_assert((int)LOOM_INDEX_CMP_PREDICATE_EQ ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_EQ,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_NE ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_NE,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_SLT ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_SLT,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_SLE ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_SLE,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_SGT ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_SGT,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_SGE ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_SGE,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_ULT ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_ULT,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_ULE ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_ULE,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_UGT ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_UGT,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_UGE ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_UGE,
              "index and scalar integer predicate enums must align");

typedef struct loom_llvmir_lowering_state_t {
  // Source Loom module being lowered.
  const loom_module_t* source_module;
  // Target structured LLVM IR module being populated.
  loom_llvmir_module_t* target_module;
  // Target/ABI profile selected for this lowering run.
  const loom_llvmir_target_profile_t* target_profile;
  // Host allocator used for temporary lowering maps.
  iree_allocator_t allocator;
  // Map from source Loom value id to target LLVMIR value id.
  loom_llvmir_value_id_t* value_map;
  // Number of entries in |value_map|.
  iree_host_size_t value_map_count;
  // Per-source-value pointer address space, or UINT32_MAX for non-pointers.
  uint32_t* value_pointer_address_spaces;
  // Per-source-value minimum pointer alignment, or zero when unknown.
  uint64_t* value_pointer_alignments;
  // Map from source module symbol id to target LLVMIR function object.
  loom_llvmir_function_t** symbol_functions;
  // Number of entries in |symbol_functions|.
  iree_host_size_t symbol_function_count;
  // Function-local fact table active while lowering a body, or NULL.
  const loom_value_fact_table_t* fact_table;
} loom_llvmir_lowering_state_t;

static iree_string_view_t loom_llvmir_lowering_module_name(
    const loom_module_t* module) {
  if (module->name_id == LOOM_STRING_ID_INVALID) return IREE_SV("");
  return module->strings.entries[module->name_id];
}

static iree_string_view_t loom_llvmir_lowering_symbol_name(
    const loom_llvmir_lowering_state_t* state, const loom_symbol_t* symbol) {
  if (symbol->name_id == LOOM_STRING_ID_INVALID) return IREE_SV("");
  return state->source_module->strings.entries[symbol->name_id];
}

static iree_string_view_t loom_llvmir_lowering_value_name(
    const loom_llvmir_lowering_state_t* state, loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(state->source_module, value_id);
  if (value->name_id == LOOM_STRING_ID_INVALID) return IREE_SV("");
  return state->source_module->strings.entries[value->name_id];
}

static iree_string_view_t loom_llvmir_lowering_block_name(
    const loom_llvmir_lowering_state_t* state, const loom_block_t* block) {
  if (block->label_id == LOOM_STRING_ID_INVALID) return IREE_SV("entry");
  return state->source_module->strings.entries[block->label_id];
}

static iree_status_t loom_llvmir_lowering_unsupported_op(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    const char* detail) {
  iree_string_view_t op_name = loom_op_name(state->source_module, op);
  iree_string_view_t profile_name = state->target_profile->name;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "LLVMIR lowering for op %.*s is unsupported for "
                          "target profile %.*s: %s",
                          (int)op_name.size, op_name.data,
                          (int)profile_name.size, profile_name.data, detail);
}

static iree_status_t loom_llvmir_lowering_unsupported_type(
    const loom_llvmir_lowering_state_t* state, loom_type_t type,
    const char* detail) {
  iree_string_view_t profile_name = state->target_profile->name;
  loom_type_kind_t kind = loom_type_kind(type);
  if (loom_type_is_scalar(type)) {
    loom_scalar_type_t scalar_type = loom_type_element_type(type);
    const char* type_name = loom_scalar_type_name(scalar_type);
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "LLVMIR lowering for scalar type %s is unsupported "
                            "for target profile %.*s: %s",
                            type_name ? type_name : "unknown",
                            (int)profile_name.size, profile_name.data, detail);
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "LLVMIR lowering for Loom type kind %u is "
                          "unsupported for target profile %.*s: %s",
                          (unsigned)kind, (int)profile_name.size,
                          profile_name.data, detail);
}

static iree_status_t loom_llvmir_lowering_get_integer_type(
    loom_llvmir_lowering_state_t* state, uint32_t bit_width,
    loom_llvmir_type_id_t* out_type_id) {
  return loom_llvmir_module_get_integer_type(state->target_module, bit_width,
                                             out_type_id);
}

static iree_status_t loom_llvmir_lowering_lower_scalar_type(
    loom_llvmir_lowering_state_t* state, loom_scalar_type_t scalar_type,
    loom_llvmir_type_id_t* out_type_id) {
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_INDEX:
      return loom_llvmir_lowering_get_integer_type(
          state, state->target_profile->target_env->index_bitwidth,
          out_type_id);
    case LOOM_SCALAR_TYPE_OFFSET:
      return loom_llvmir_lowering_get_integer_type(
          state, state->target_profile->target_env->offset_bitwidth,
          out_type_id);
    case LOOM_SCALAR_TYPE_I1:
      return loom_llvmir_lowering_get_integer_type(state, 1, out_type_id);
    case LOOM_SCALAR_TYPE_I8:
      return loom_llvmir_lowering_get_integer_type(state, 8, out_type_id);
    case LOOM_SCALAR_TYPE_I16:
      return loom_llvmir_lowering_get_integer_type(state, 16, out_type_id);
    case LOOM_SCALAR_TYPE_I32:
      return loom_llvmir_lowering_get_integer_type(state, 32, out_type_id);
    case LOOM_SCALAR_TYPE_I64:
      return loom_llvmir_lowering_get_integer_type(state, 64, out_type_id);
    case LOOM_SCALAR_TYPE_F16:
      return loom_llvmir_module_get_float_type(
          state->target_module, LOOM_LLVMIR_FLOAT_F16, out_type_id);
    case LOOM_SCALAR_TYPE_F32:
      return loom_llvmir_module_get_float_type(
          state->target_module, LOOM_LLVMIR_FLOAT_F32, out_type_id);
    case LOOM_SCALAR_TYPE_F64:
      return loom_llvmir_module_get_float_type(
          state->target_module, LOOM_LLVMIR_FLOAT_F64, out_type_id);
    case LOOM_SCALAR_TYPE_F8E4M3:
    case LOOM_SCALAR_TYPE_F8E5M2:
    case LOOM_SCALAR_TYPE_BF16:
    case LOOM_SCALAR_TYPE_COUNT_:
      return loom_llvmir_lowering_unsupported_type(
          state, loom_type_scalar(scalar_type),
          "the structured LLVMIR type model does not carry this scalar yet");
  }
  return loom_llvmir_lowering_unsupported_type(
      state, loom_type_scalar(scalar_type), "unknown scalar type");
}

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

static iree_status_t loom_llvmir_lowering_get_pointer_type(
    loom_llvmir_lowering_state_t* state, uint32_t address_space,
    loom_llvmir_type_id_t* out_type_id) {
  return loom_llvmir_module_get_pointer_type(state->target_module,
                                             address_space, out_type_id);
}

static iree_status_t loom_llvmir_lowering_lower_type(
    loom_llvmir_lowering_state_t* state, loom_type_t type,
    loom_llvmir_type_id_t* out_type_id) {
  if (loom_type_is_scalar(type)) {
    return loom_llvmir_lowering_lower_scalar_type(
        state, loom_type_element_type(type), out_type_id);
  }
  if (loom_type_is_buffer(type) || loom_type_is_view(type)) {
    return loom_llvmir_lowering_get_pointer_type(
        state, state->target_profile->target_env->address_spaces.generic,
        out_type_id);
  }
  if (loom_type_is_vector(type)) {
    if (!loom_type_is_all_static(type) || loom_type_rank(type) != 1) {
      return loom_llvmir_lowering_unsupported_type(
          state, type, "only static one-dimensional vectors lower today");
    }
    uint64_t element_count = 0;
    if (!loom_type_static_element_count(type, &element_count) ||
        element_count > UINT32_MAX) {
      return loom_llvmir_lowering_unsupported_type(
          state, type, "vector lane count is not representable");
    }
    loom_llvmir_type_id_t element_type_id = LOOM_LLVMIR_TYPE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_scalar_type(
        state, loom_type_element_type(type), &element_type_id));
    return loom_llvmir_module_get_vector_type(state->target_module,
                                              (uint32_t)element_count,
                                              element_type_id, out_type_id);
  }
  return loom_llvmir_lowering_unsupported_type(
      state, type, "no LLVMIR type mapping exists yet");
}

static iree_status_t loom_llvmir_lowering_lower_return_type(
    loom_llvmir_lowering_state_t* state, const loom_op_t* func_op,
    loom_llvmir_type_id_t* out_type_id) {
  if (func_op->result_count == 0) {
    return loom_llvmir_module_get_void_type(state->target_module, out_type_id);
  }
  if (func_op->result_count != 1) {
    return loom_llvmir_lowering_unsupported_op(
        state, func_op,
        "LLVM functions with multiple direct results need an "
        "aggregate or sret ABI decision");
  }
  const loom_value_id_t result_id = loom_op_const_results(func_op)[0];
  return loom_llvmir_lowering_lower_type(
      state, loom_module_value_type(state->source_module, result_id),
      out_type_id);
}

static iree_status_t loom_llvmir_lowering_map_value(
    loom_llvmir_lowering_state_t* state, loom_value_id_t source_value_id,
    loom_llvmir_value_id_t target_value_id) {
  if (source_value_id >= state->value_map_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source value id %u is out of range",
                            (unsigned)source_value_id);
  }
  state->value_map[source_value_id] = target_value_id;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_lowering_map_pointer_value(
    loom_llvmir_lowering_state_t* state, loom_value_id_t source_value_id,
    loom_llvmir_value_id_t target_value_id, uint32_t address_space,
    uint64_t minimum_alignment) {
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_map_value(state, source_value_id, target_value_id));
  state->value_pointer_address_spaces[source_value_id] = address_space;
  state->value_pointer_alignments[source_value_id] = minimum_alignment;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_lowering_lookup_value(
    const loom_llvmir_lowering_state_t* state, loom_value_id_t source_value_id,
    loom_llvmir_value_id_t* out_target_value_id) {
  if (source_value_id >= state->value_map_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source value id %u is out of range",
                            (unsigned)source_value_id);
  }
  loom_llvmir_value_id_t target_value_id = state->value_map[source_value_id];
  if (target_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "source value id %u has not been lowered",
                            (unsigned)source_value_id);
  }
  *out_target_value_id = target_value_id;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_lowering_lookup_pointer(
    const loom_llvmir_lowering_state_t* state, loom_value_id_t source_value_id,
    loom_llvmir_value_id_t* out_target_value_id, uint32_t* out_address_space,
    uint64_t* out_minimum_alignment) {
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_value(state, source_value_id,
                                                         out_target_value_id));
  uint32_t address_space = state->value_pointer_address_spaces[source_value_id];
  if (address_space == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "source value id %u is not lowered as a pointer",
                            (unsigned)source_value_id);
  }
  if (out_address_space) *out_address_space = address_space;
  if (out_minimum_alignment) {
    *out_minimum_alignment = state->value_pointer_alignments[source_value_id];
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_lowering_lookup_operands(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_llvmir_value_id_t* target_operands) {
  const loom_value_id_t* source_operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_value(
        state, source_operands[i], &target_operands[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_lowering_map_single_result(
    loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_llvmir_value_id_t target_value_id) {
  if (op->result_count != 1) {
    return loom_llvmir_lowering_unsupported_op(state, op,
                                               "expected exactly one result");
  }
  return loom_llvmir_lowering_map_value(state, loom_op_const_results(op)[0],
                                        target_value_id);
}

static loom_llvmir_linkage_t loom_llvmir_lowering_function_linkage(
    const loom_llvmir_lowering_state_t* state, loom_func_like_t func,
    loom_llvmir_function_kind_t function_kind) {
  if (function_kind == LOOM_LLVMIR_FUNCTION_DECLARATION) {
    return LOOM_LLVMIR_LINKAGE_DEFAULT;
  }
  if (loom_func_like_visibility(func) != 0) {
    return state->target_profile->exported_linkage;
  }
  return LOOM_LLVMIR_LINKAGE_INTERNAL;
}

static loom_llvmir_calling_convention_t
loom_llvmir_lowering_function_calling_convention(
    const loom_llvmir_lowering_state_t* state, loom_func_like_t func) {
  if (state->target_profile->kind == LOOM_LLVMIR_TARGET_PROFILE_HAL_KERNEL &&
      loom_func_like_cc(func) == LOOM_FUNC_CC_DEVICE) {
    return state->target_profile->kernel_calling_convention;
  }
  return LOOM_LLVMIR_CALLING_CONVENTION_DEFAULT;
}

static iree_status_t loom_llvmir_lowering_add_function_signature(
    loom_llvmir_lowering_state_t* state, uint16_t symbol_id,
    loom_func_like_t func) {
  loom_symbol_t* symbol = &state->source_module->symbols.entries[symbol_id];
  loom_region_t* body = loom_func_like_body(func);
  loom_llvmir_function_kind_t function_kind =
      body ? LOOM_LLVMIR_FUNCTION_DEFINITION : LOOM_LLVMIR_FUNCTION_DECLARATION;
  loom_llvmir_type_id_t return_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lower_return_type(state, func.op, &return_type));

  loom_llvmir_function_desc_t function_desc = {0};
  function_desc.kind = function_kind;
  function_desc.name = loom_llvmir_lowering_symbol_name(state, symbol);
  function_desc.return_type = return_type;
  function_desc.linkage =
      loom_llvmir_lowering_function_linkage(state, func, function_kind);
  function_desc.calling_convention =
      loom_llvmir_lowering_function_calling_convention(state, func);
  function_desc.attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID;

  loom_llvmir_function_t* target_function = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_function(
      state->target_module, &function_desc, &target_function));
  state->symbol_functions[symbol_id] = target_function;

  uint16_t arg_count = 0;
  const loom_value_id_t* arg_ids = loom_func_like_arg_ids(func, &arg_count);
  for (uint16_t i = 0; i < arg_count; ++i) {
    loom_type_t source_arg_type =
        loom_module_value_type(state->source_module, arg_ids[i]);
    uint32_t pointer_address_space = UINT32_MAX;
    loom_llvmir_type_id_t arg_type = LOOM_LLVMIR_TYPE_ID_INVALID;
    if (loom_type_is_buffer(source_arg_type) ||
        loom_type_is_view(source_arg_type)) {
      pointer_address_space =
          state->target_profile->target_env->address_spaces.generic;
      IREE_RETURN_IF_ERROR(loom_llvmir_lowering_get_pointer_type(
          state, pointer_address_space, &arg_type));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_llvmir_lowering_lower_type(state, source_arg_type, &arg_type));
    }

    loom_llvmir_parameter_desc_t parameter_desc = {0};
    parameter_desc.type_id = arg_type;
    parameter_desc.name = loom_llvmir_lowering_value_name(state, arg_ids[i]);
    loom_llvmir_value_id_t parameter_value = LOOM_LLVMIR_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_llvmir_function_add_parameter(
        target_function, &parameter_desc, &parameter_value));
    if (pointer_address_space == UINT32_MAX) {
      IREE_RETURN_IF_ERROR(
          loom_llvmir_lowering_map_value(state, arg_ids[i], parameter_value));
    } else {
      IREE_RETURN_IF_ERROR(loom_llvmir_lowering_map_pointer_value(
          state, arg_ids[i], parameter_value, pointer_address_space,
          /*minimum_alignment=*/0));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_lowering_lower_function_signatures(
    loom_llvmir_lowering_state_t* state) {
  for (iree_host_size_t i = 0; i < state->source_module->symbols.count; ++i) {
    loom_symbol_t* symbol = &state->source_module->symbols.entries[i];
    if (!loom_symbol_kind_is_function_like(symbol->kind)) continue;
    if (!symbol->defining_op) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "function-like symbol id %u has no defining op",
                              (unsigned)i);
    }
    if (symbol->kind != LOOM_SYMBOL_FUNC_DEF &&
        symbol->kind != LOOM_SYMBOL_FUNC_DECL) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "LLVMIR lowering only supports func.def and "
                              "func.decl symbols today");
    }
    loom_func_like_t func =
        loom_func_like_cast(state->source_module, symbol->defining_op);
    if (!loom_func_like_isa(func)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "function-like symbol id %u does not implement "
                              "the FuncLike interface",
                              (unsigned)i);
    }
    IREE_RETURN_IF_ERROR(
        loom_llvmir_lowering_add_function_signature(state, (uint16_t)i, func));
  }
  return iree_ok_status();
}

static loom_llvmir_integer_arithmetic_flags_t
loom_llvmir_lowering_integer_flags(uint8_t instance_flags) {
  loom_llvmir_integer_arithmetic_flags_t flags =
      LOOM_LLVMIR_INTEGER_ARITHMETIC_NONE;
  if (iree_any_bit_set(instance_flags, LOOM_SCALAR_INTOVERFLOWFLAGS_NUW)) {
    flags |= LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_UNSIGNED_WRAP;
  }
  if (iree_any_bit_set(instance_flags, LOOM_SCALAR_INTOVERFLOWFLAGS_NSW)) {
    flags |= LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_SIGNED_WRAP;
  }
  return flags;
}

static loom_llvmir_fast_math_flags_t loom_llvmir_lowering_fast_math_flags(
    uint8_t instance_flags) {
  loom_llvmir_fast_math_flags_t flags = LOOM_LLVMIR_FAST_MATH_NONE;
  if (iree_any_bit_set(instance_flags, LOOM_SCALAR_FASTMATHFLAGS_REASSOC)) {
    flags |= LOOM_LLVMIR_FAST_MATH_ALLOW_REASSOC;
  }
  if (iree_any_bit_set(instance_flags, LOOM_SCALAR_FASTMATHFLAGS_NNAN)) {
    flags |= LOOM_LLVMIR_FAST_MATH_NO_NANS;
  }
  if (iree_any_bit_set(instance_flags, LOOM_SCALAR_FASTMATHFLAGS_NINF)) {
    flags |= LOOM_LLVMIR_FAST_MATH_NO_INFS;
  }
  if (iree_any_bit_set(instance_flags, LOOM_SCALAR_FASTMATHFLAGS_NSZ)) {
    flags |= LOOM_LLVMIR_FAST_MATH_NO_SIGNED_ZEROS;
  }
  if (iree_any_bit_set(instance_flags, LOOM_SCALAR_FASTMATHFLAGS_ARCP)) {
    flags |= LOOM_LLVMIR_FAST_MATH_ALLOW_RECIPROCAL;
  }
  if (iree_any_bit_set(instance_flags, LOOM_SCALAR_FASTMATHFLAGS_CONTRACT)) {
    flags |= LOOM_LLVMIR_FAST_MATH_ALLOW_CONTRACT;
  }
  if (iree_any_bit_set(instance_flags, LOOM_SCALAR_FASTMATHFLAGS_AFN)) {
    flags |= LOOM_LLVMIR_FAST_MATH_APPROX_FUNC;
  }
  return flags;
}

static bool loom_llvmir_lowering_op_is_float_binop(const loom_op_t* op) {
  switch (op->kind) {
    case LOOM_OP_SCALAR_ADDF:
    case LOOM_OP_SCALAR_SUBF:
    case LOOM_OP_SCALAR_MULF:
    case LOOM_OP_SCALAR_DIVF:
    case LOOM_OP_SCALAR_REMF:
      return true;
    default:
      return false;
  }
}

static bool loom_llvmir_lowering_index_type_is_unsigned(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_OFFSET;
}

static iree_status_t loom_llvmir_lowering_binop_from_op(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_llvmir_binop_t* out_binop) {
  switch (op->kind) {
    case LOOM_OP_SCALAR_ADDI:
    case LOOM_OP_INDEX_ADD:
      *out_binop = LOOM_LLVMIR_BINOP_ADD;
      return iree_ok_status();
    case LOOM_OP_SCALAR_SUBI:
    case LOOM_OP_INDEX_SUB:
      *out_binop = LOOM_LLVMIR_BINOP_SUB;
      return iree_ok_status();
    case LOOM_OP_SCALAR_MULI:
    case LOOM_OP_INDEX_MUL:
      *out_binop = LOOM_LLVMIR_BINOP_MUL;
      return iree_ok_status();
    case LOOM_OP_SCALAR_DIVSI:
      *out_binop = LOOM_LLVMIR_BINOP_SDIV;
      return iree_ok_status();
    case LOOM_OP_SCALAR_DIVUI:
      *out_binop = LOOM_LLVMIR_BINOP_UDIV;
      return iree_ok_status();
    case LOOM_OP_INDEX_DIV: {
      loom_type_t result_type =
          loom_module_value_type(state->source_module, loom_op_results(op)[0]);
      *out_binop = loom_llvmir_lowering_index_type_is_unsigned(result_type)
                       ? LOOM_LLVMIR_BINOP_UDIV
                       : LOOM_LLVMIR_BINOP_SDIV;
      return iree_ok_status();
    }
    case LOOM_OP_SCALAR_REMSI:
      *out_binop = LOOM_LLVMIR_BINOP_SREM;
      return iree_ok_status();
    case LOOM_OP_SCALAR_REMUI:
      *out_binop = LOOM_LLVMIR_BINOP_UREM;
      return iree_ok_status();
    case LOOM_OP_INDEX_REM: {
      loom_type_t result_type =
          loom_module_value_type(state->source_module, loom_op_results(op)[0]);
      *out_binop = loom_llvmir_lowering_index_type_is_unsigned(result_type)
                       ? LOOM_LLVMIR_BINOP_UREM
                       : LOOM_LLVMIR_BINOP_SREM;
      return iree_ok_status();
    }
    case LOOM_OP_SCALAR_ADDF:
      *out_binop = LOOM_LLVMIR_BINOP_FADD;
      return iree_ok_status();
    case LOOM_OP_SCALAR_SUBF:
      *out_binop = LOOM_LLVMIR_BINOP_FSUB;
      return iree_ok_status();
    case LOOM_OP_SCALAR_MULF:
      *out_binop = LOOM_LLVMIR_BINOP_FMUL;
      return iree_ok_status();
    case LOOM_OP_SCALAR_DIVF:
      *out_binop = LOOM_LLVMIR_BINOP_FDIV;
      return iree_ok_status();
    case LOOM_OP_SCALAR_REMF:
      *out_binop = LOOM_LLVMIR_BINOP_FREM;
      return iree_ok_status();
    case LOOM_OP_SCALAR_ANDI:
      *out_binop = LOOM_LLVMIR_BINOP_AND;
      return iree_ok_status();
    case LOOM_OP_SCALAR_ORI:
      *out_binop = LOOM_LLVMIR_BINOP_OR;
      return iree_ok_status();
    case LOOM_OP_SCALAR_XORI:
      *out_binop = LOOM_LLVMIR_BINOP_XOR;
      return iree_ok_status();
    case LOOM_OP_SCALAR_SHLI:
      *out_binop = LOOM_LLVMIR_BINOP_SHL;
      return iree_ok_status();
    case LOOM_OP_SCALAR_SHRSI:
      *out_binop = LOOM_LLVMIR_BINOP_ASHR;
      return iree_ok_status();
    case LOOM_OP_SCALAR_SHRUI:
      *out_binop = LOOM_LLVMIR_BINOP_LSHR;
      return iree_ok_status();
    default:
      return loom_llvmir_lowering_unsupported_op(state, op,
                                                 "no binary opcode mapping");
  }
}

static iree_status_t loom_llvmir_lowering_lower_binop(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_llvmir_value_id_t operands[2] = {LOOM_LLVMIR_VALUE_ID_INVALID,
                                        LOOM_LLVMIR_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lookup_operands(state, op, operands));
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_type(
      state,
      loom_module_value_type(state->source_module,
                             loom_op_const_results(op)[0]),
      &result_type));
  loom_llvmir_binop_t binop = LOOM_LLVMIR_BINOP_ADD;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_binop_from_op(state, op, &binop));

  loom_llvmir_binop_desc_t desc = {0};
  desc.result_name =
      loom_llvmir_lowering_value_name(state, loom_op_const_results(op)[0]);
  desc.result_type = result_type;
  desc.op = binop;
  desc.lhs = operands[0];
  desc.rhs = operands[1];
  if (loom_llvmir_lowering_op_is_float_binop(op)) {
    desc.fast_math_flags =
        loom_llvmir_lowering_fast_math_flags(op->instance_flags);
  } else {
    desc.integer_flags = loom_llvmir_lowering_integer_flags(op->instance_flags);
  }
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_binop(target_block, &desc, &result));
  return loom_llvmir_lowering_map_single_result(state, op, result);
}

static iree_status_t loom_llvmir_lowering_lower_negf(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_llvmir_value_id_t value = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_value(
      state, loom_op_const_operands(op)[0], &value));
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_type(
      state,
      loom_module_value_type(state->source_module,
                             loom_op_const_results(op)[0]),
      &result_type));

  loom_llvmir_unop_desc_t desc = {0};
  desc.result_name =
      loom_llvmir_lowering_value_name(state, loom_op_const_results(op)[0]);
  desc.result_type = result_type;
  desc.op = LOOM_LLVMIR_UNOP_FNEG;
  desc.value = value;
  desc.fast_math_flags =
      loom_llvmir_lowering_fast_math_flags(op->instance_flags);
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_unop(target_block, &desc, &result));
  return loom_llvmir_lowering_map_single_result(state, op, result);
}

static iree_status_t loom_llvmir_lowering_lower_constant(
    loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_attribute_t value_attr) {
  loom_type_t result_source_type = loom_module_value_type(
      state->source_module, loom_op_const_results(op)[0]);
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lower_type(state, result_source_type, &result_type));

  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  if (loom_type_is_scalar(result_source_type) &&
      loom_scalar_type_is_integer(loom_type_element_type(result_source_type))) {
    if (value_attr.kind != LOOM_ATTR_I64 && value_attr.kind != LOOM_ATTR_BOOL) {
      return loom_llvmir_lowering_unsupported_op(
          state, op, "integer constants require an i64 or bool attribute");
    }
    uint64_t value = value_attr.kind == LOOM_ATTR_BOOL
                         ? value_attr.raw
                         : (uint64_t)value_attr.i64;
    IREE_RETURN_IF_ERROR(loom_llvmir_module_add_integer_constant(
        state->target_module, result_type, value, &result));
  } else if (loom_type_is_scalar(result_source_type) &&
             (loom_type_element_type(result_source_type) ==
                  LOOM_SCALAR_TYPE_INDEX ||
              loom_type_element_type(result_source_type) ==
                  LOOM_SCALAR_TYPE_OFFSET)) {
    if (value_attr.kind != LOOM_ATTR_I64) {
      return loom_llvmir_lowering_unsupported_op(
          state, op, "index/offset constants require an i64 attribute");
    }
    IREE_RETURN_IF_ERROR(loom_llvmir_module_add_integer_constant(
        state->target_module, result_type, (uint64_t)value_attr.i64, &result));
  } else if (loom_type_is_scalar(result_source_type) &&
             loom_type_element_type(result_source_type) ==
                 LOOM_SCALAR_TYPE_F64) {
    if (value_attr.kind != LOOM_ATTR_F64) {
      return loom_llvmir_lowering_unsupported_op(
          state, op, "f64 constants require an f64 attribute");
    }
    uint64_t bits = 0;
    memcpy(&bits, &value_attr.f64, sizeof(bits));
    IREE_RETURN_IF_ERROR(loom_llvmir_module_add_float_bits_constant(
        state->target_module, result_type, bits, &result));
  } else {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "constant attribute/type combination is not supported yet");
  }
  return loom_llvmir_lowering_map_single_result(state, op, result);
}

static iree_status_t loom_llvmir_lowering_lower_index_madd(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_llvmir_value_id_t operands[3] = {LOOM_LLVMIR_VALUE_ID_INVALID,
                                        LOOM_LLVMIR_VALUE_ID_INVALID,
                                        LOOM_LLVMIR_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lookup_operands(state, op, operands));
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_type(
      state,
      loom_module_value_type(state->source_module,
                             loom_op_const_results(op)[0]),
      &result_type));

  loom_llvmir_binop_desc_t multiply_desc = {0};
  multiply_desc.result_type = result_type;
  multiply_desc.op = LOOM_LLVMIR_BINOP_MUL;
  multiply_desc.lhs = operands[0];
  multiply_desc.rhs = operands[1];
  loom_llvmir_value_id_t product = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_binop(target_block, &multiply_desc, &product));

  loom_llvmir_binop_desc_t add_desc = {0};
  add_desc.result_name =
      loom_llvmir_lowering_value_name(state, loom_op_const_results(op)[0]);
  add_desc.result_type = result_type;
  add_desc.op = LOOM_LLVMIR_BINOP_ADD;
  add_desc.lhs = product;
  add_desc.rhs = operands[2];
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_binop(target_block, &add_desc, &result));
  return loom_llvmir_lowering_map_single_result(state, op, result);
}

static iree_status_t loom_llvmir_lowering_icmp_predicate(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    uint8_t source_predicate, loom_llvmir_icmp_predicate_t* out_predicate) {
  switch (source_predicate) {
    case LOOM_SCALAR_CMPI_PREDICATE_EQ:
      *out_predicate = LOOM_LLVMIR_ICMP_EQ;
      return iree_ok_status();
    case LOOM_SCALAR_CMPI_PREDICATE_NE:
      *out_predicate = LOOM_LLVMIR_ICMP_NE;
      return iree_ok_status();
    case LOOM_SCALAR_CMPI_PREDICATE_SLT:
      *out_predicate = LOOM_LLVMIR_ICMP_SLT;
      return iree_ok_status();
    case LOOM_SCALAR_CMPI_PREDICATE_SLE:
      *out_predicate = LOOM_LLVMIR_ICMP_SLE;
      return iree_ok_status();
    case LOOM_SCALAR_CMPI_PREDICATE_SGT:
      *out_predicate = LOOM_LLVMIR_ICMP_SGT;
      return iree_ok_status();
    case LOOM_SCALAR_CMPI_PREDICATE_SGE:
      *out_predicate = LOOM_LLVMIR_ICMP_SGE;
      return iree_ok_status();
    case LOOM_SCALAR_CMPI_PREDICATE_ULT:
      *out_predicate = LOOM_LLVMIR_ICMP_ULT;
      return iree_ok_status();
    case LOOM_SCALAR_CMPI_PREDICATE_ULE:
      *out_predicate = LOOM_LLVMIR_ICMP_ULE;
      return iree_ok_status();
    case LOOM_SCALAR_CMPI_PREDICATE_UGT:
      *out_predicate = LOOM_LLVMIR_ICMP_UGT;
      return iree_ok_status();
    case LOOM_SCALAR_CMPI_PREDICATE_UGE:
      *out_predicate = LOOM_LLVMIR_ICMP_UGE;
      return iree_ok_status();
    default:
      return loom_llvmir_lowering_unsupported_op(
          state, op, "unknown integer comparison predicate");
  }
}

static iree_status_t loom_llvmir_lowering_fcmp_predicate(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    uint8_t source_predicate, loom_llvmir_fcmp_predicate_t* out_predicate) {
  switch (source_predicate) {
    case LOOM_SCALAR_CMPF_PREDICATE_OEQ:
      *out_predicate = LOOM_LLVMIR_FCMP_OEQ;
      return iree_ok_status();
    case LOOM_SCALAR_CMPF_PREDICATE_OGT:
      *out_predicate = LOOM_LLVMIR_FCMP_OGT;
      return iree_ok_status();
    case LOOM_SCALAR_CMPF_PREDICATE_OGE:
      *out_predicate = LOOM_LLVMIR_FCMP_OGE;
      return iree_ok_status();
    case LOOM_SCALAR_CMPF_PREDICATE_OLT:
      *out_predicate = LOOM_LLVMIR_FCMP_OLT;
      return iree_ok_status();
    case LOOM_SCALAR_CMPF_PREDICATE_OLE:
      *out_predicate = LOOM_LLVMIR_FCMP_OLE;
      return iree_ok_status();
    case LOOM_SCALAR_CMPF_PREDICATE_ONE:
      *out_predicate = LOOM_LLVMIR_FCMP_ONE;
      return iree_ok_status();
    case LOOM_SCALAR_CMPF_PREDICATE_ORD:
      *out_predicate = LOOM_LLVMIR_FCMP_ORD;
      return iree_ok_status();
    case LOOM_SCALAR_CMPF_PREDICATE_UEQ:
      *out_predicate = LOOM_LLVMIR_FCMP_UEQ;
      return iree_ok_status();
    case LOOM_SCALAR_CMPF_PREDICATE_UGT:
      *out_predicate = LOOM_LLVMIR_FCMP_UGT;
      return iree_ok_status();
    case LOOM_SCALAR_CMPF_PREDICATE_UGE:
      *out_predicate = LOOM_LLVMIR_FCMP_UGE;
      return iree_ok_status();
    case LOOM_SCALAR_CMPF_PREDICATE_ULT:
      *out_predicate = LOOM_LLVMIR_FCMP_ULT;
      return iree_ok_status();
    case LOOM_SCALAR_CMPF_PREDICATE_ULE:
      *out_predicate = LOOM_LLVMIR_FCMP_ULE;
      return iree_ok_status();
    case LOOM_SCALAR_CMPF_PREDICATE_UNE:
      *out_predicate = LOOM_LLVMIR_FCMP_UNE;
      return iree_ok_status();
    case LOOM_SCALAR_CMPF_PREDICATE_UNO:
      *out_predicate = LOOM_LLVMIR_FCMP_UNO;
      return iree_ok_status();
    default:
      return loom_llvmir_lowering_unsupported_op(
          state, op, "unknown floating-point comparison predicate");
  }
}

static iree_status_t loom_llvmir_lowering_lower_icmp(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op, uint8_t source_predicate) {
  loom_llvmir_value_id_t operands[2] = {LOOM_LLVMIR_VALUE_ID_INVALID,
                                        LOOM_LLVMIR_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lookup_operands(state, op, operands));
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_type(
      state,
      loom_module_value_type(state->source_module,
                             loom_op_const_results(op)[0]),
      &result_type));
  loom_llvmir_icmp_predicate_t predicate = LOOM_LLVMIR_ICMP_EQ;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_icmp_predicate(
      state, op, source_predicate, &predicate));

  loom_llvmir_icmp_desc_t desc = {0};
  desc.result_name =
      loom_llvmir_lowering_value_name(state, loom_op_const_results(op)[0]);
  desc.result_type = result_type;
  desc.predicate = predicate;
  desc.lhs = operands[0];
  desc.rhs = operands[1];
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_icmp(target_block, &desc, &result));
  return loom_llvmir_lowering_map_single_result(state, op, result);
}

static iree_status_t loom_llvmir_lowering_lower_fcmp(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_llvmir_value_id_t operands[2] = {LOOM_LLVMIR_VALUE_ID_INVALID,
                                        LOOM_LLVMIR_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lookup_operands(state, op, operands));
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_type(
      state,
      loom_module_value_type(state->source_module,
                             loom_op_const_results(op)[0]),
      &result_type));
  loom_llvmir_fcmp_predicate_t predicate = LOOM_LLVMIR_FCMP_FALSE;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_fcmp_predicate(
      state, op, loom_scalar_cmpf_predicate(op), &predicate));

  loom_llvmir_fcmp_desc_t desc = {0};
  desc.result_name =
      loom_llvmir_lowering_value_name(state, loom_op_const_results(op)[0]);
  desc.result_type = result_type;
  desc.predicate = predicate;
  desc.lhs = operands[0];
  desc.rhs = operands[1];
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_fcmp(target_block, &desc, &result));
  return loom_llvmir_lowering_map_single_result(state, op, result);
}

static iree_status_t loom_llvmir_lowering_cast_op_from_kind(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_llvmir_cast_op_t* out_cast_op) {
  switch (op->kind) {
    case LOOM_OP_SCALAR_SITOFP:
      *out_cast_op = LOOM_LLVMIR_CAST_SIGNED_INT_TO_FP;
      return iree_ok_status();
    case LOOM_OP_SCALAR_UITOFP:
      *out_cast_op = LOOM_LLVMIR_CAST_UNSIGNED_INT_TO_FP;
      return iree_ok_status();
    case LOOM_OP_SCALAR_FPTOSI:
      *out_cast_op = LOOM_LLVMIR_CAST_FP_TO_SIGNED_INT;
      return iree_ok_status();
    case LOOM_OP_SCALAR_FPTOUI:
      *out_cast_op = LOOM_LLVMIR_CAST_FP_TO_UNSIGNED_INT;
      return iree_ok_status();
    case LOOM_OP_SCALAR_EXTF:
      *out_cast_op = LOOM_LLVMIR_CAST_FP_EXTEND;
      return iree_ok_status();
    case LOOM_OP_SCALAR_FPTRUNC:
      *out_cast_op = LOOM_LLVMIR_CAST_FP_TRUNCATE;
      return iree_ok_status();
    case LOOM_OP_SCALAR_EXTSI:
      *out_cast_op = LOOM_LLVMIR_CAST_SIGN_EXTEND;
      return iree_ok_status();
    case LOOM_OP_SCALAR_EXTUI:
      *out_cast_op = LOOM_LLVMIR_CAST_ZERO_EXTEND;
      return iree_ok_status();
    case LOOM_OP_SCALAR_TRUNCI:
      *out_cast_op = LOOM_LLVMIR_CAST_TRUNCATE;
      return iree_ok_status();
    case LOOM_OP_SCALAR_BITCAST:
      *out_cast_op = LOOM_LLVMIR_CAST_BITCAST;
      return iree_ok_status();
    default:
      return loom_llvmir_lowering_unsupported_op(state, op,
                                                 "no cast opcode mapping");
  }
}

static iree_status_t loom_llvmir_lowering_lower_cast(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_llvmir_value_id_t operand = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_value(
      state, loom_op_const_operands(op)[0], &operand));
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_type(
      state,
      loom_module_value_type(state->source_module,
                             loom_op_const_results(op)[0]),
      &result_type));
  loom_llvmir_cast_op_t cast_op = LOOM_LLVMIR_CAST_TRUNCATE;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_cast_op_from_kind(state, op, &cast_op));

  loom_llvmir_cast_desc_t desc = {0};
  desc.result_name =
      loom_llvmir_lowering_value_name(state, loom_op_const_results(op)[0]);
  desc.result_type = result_type;
  desc.op = cast_op;
  desc.value = operand;
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_cast(target_block, &desc, &result));
  return loom_llvmir_lowering_map_single_result(state, op, result);
}

static iree_status_t loom_llvmir_lowering_integer_bitwidth(
    loom_llvmir_lowering_state_t* state, loom_type_t type,
    uint32_t* out_bitwidth) {
  if (!loom_type_is_scalar(type)) {
    return loom_llvmir_lowering_unsupported_type(
        state, type, "index.cast only supports scalar integer-like types");
  }
  switch (loom_type_element_type(type)) {
    case LOOM_SCALAR_TYPE_INDEX:
      *out_bitwidth = state->target_profile->target_env->index_bitwidth;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_OFFSET:
      *out_bitwidth = state->target_profile->target_env->offset_bitwidth;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_I1:
    case LOOM_SCALAR_TYPE_I8:
    case LOOM_SCALAR_TYPE_I16:
    case LOOM_SCALAR_TYPE_I32:
    case LOOM_SCALAR_TYPE_I64:
      *out_bitwidth =
          (uint32_t)loom_scalar_type_bitwidth(loom_type_element_type(type));
      return iree_ok_status();
    default:
      return loom_llvmir_lowering_unsupported_type(
          state, type, "index.cast only supports integer-like scalar types");
  }
}

static iree_status_t loom_llvmir_lowering_lower_index_cast(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_value_id_t source_value_id = loom_op_const_operands(op)[0];
  loom_value_id_t result_value_id = loom_op_const_results(op)[0];
  loom_llvmir_value_id_t operand = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lookup_value(state, source_value_id, &operand));

  loom_type_t source_type =
      loom_module_value_type(state->source_module, source_value_id);
  loom_type_t result_source_type =
      loom_module_value_type(state->source_module, result_value_id);
  uint32_t source_bitwidth = 0;
  uint32_t result_bitwidth = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_integer_bitwidth(state, source_type,
                                                             &source_bitwidth));
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_integer_bitwidth(
      state, result_source_type, &result_bitwidth));
  if (source_bitwidth == result_bitwidth) {
    return loom_llvmir_lowering_map_value(state, result_value_id, operand);
  }

  loom_llvmir_cast_op_t cast_op = LOOM_LLVMIR_CAST_TRUNCATE;
  if (source_bitwidth < result_bitwidth) {
    cast_op = loom_llvmir_lowering_index_type_is_unsigned(source_type)
                  ? LOOM_LLVMIR_CAST_ZERO_EXTEND
                  : LOOM_LLVMIR_CAST_SIGN_EXTEND;
  }
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lower_type(state, result_source_type, &result_type));
  loom_llvmir_cast_desc_t desc = {0};
  desc.result_name = loom_llvmir_lowering_value_name(state, result_value_id);
  desc.result_type = result_type;
  desc.op = cast_op;
  desc.value = operand;
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_cast(target_block, &desc, &result));
  return loom_llvmir_lowering_map_value(state, result_value_id, result);
}

static iree_status_t loom_llvmir_lowering_lower_select(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_llvmir_value_id_t operands[3] = {LOOM_LLVMIR_VALUE_ID_INVALID,
                                        LOOM_LLVMIR_VALUE_ID_INVALID,
                                        LOOM_LLVMIR_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lookup_operands(state, op, operands));
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_type(
      state,
      loom_module_value_type(state->source_module,
                             loom_op_const_results(op)[0]),
      &result_type));

  loom_llvmir_select_desc_t desc = {0};
  desc.result_name =
      loom_llvmir_lowering_value_name(state, loom_op_const_results(op)[0]);
  desc.result_type = result_type;
  desc.condition = operands[0];
  desc.true_value = operands[1];
  desc.false_value = operands[2];
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_select(target_block, &desc, &result));
  return loom_llvmir_lowering_map_single_result(state, op, result);
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

static iree_status_t loom_llvmir_lowering_lower_alloca(
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

static iree_status_t loom_llvmir_lowering_lower_assume_memory_space(
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

static iree_status_t loom_llvmir_lowering_lower_buffer_view(
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

static iree_status_t loom_llvmir_lowering_lower_subview(
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

static iree_status_t loom_llvmir_lowering_lower_refine(
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

static iree_status_t loom_llvmir_lowering_lower_view_load(
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

static iree_status_t loom_llvmir_lowering_lower_view_store(
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

static iree_status_t loom_llvmir_lowering_lower_call(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  if (op->result_count > 1) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "calls with multiple direct results need tuple/sret policy");
  }
  loom_symbol_ref_t callee_ref = loom_func_call_callee(op);
  if (callee_ref.module_id != 0 ||
      callee_ref.symbol_id >= state->symbol_function_count) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "callee must be a module-local function symbol");
  }
  loom_llvmir_function_t* callee =
      state->symbol_functions[callee_ref.symbol_id];
  if (!callee) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "callee has not been declared in the LLVMIR module");
  }

  loom_llvmir_value_id_t* args = NULL;
  iree_status_t status = iree_ok_status();
  if (op->operand_count > 0) {
    status = iree_allocator_malloc(
        state->allocator, op->operand_count * sizeof(loom_llvmir_value_id_t),
        (void**)&args);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_lowering_lookup_operands(state, op, args);
  }
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  if (iree_status_is_ok(status)) {
    loom_llvmir_call_desc_t desc = {0};
    desc.callee = loom_llvmir_function_id(callee);
    desc.args = args;
    desc.arg_count = op->operand_count;
    if (op->result_count == 1) {
      desc.result_name =
          loom_llvmir_lowering_value_name(state, loom_op_const_results(op)[0]);
    }
    status = loom_llvmir_build_call(target_block, &desc, &result);
  }
  if (iree_status_is_ok(status) && op->result_count == 1) {
    status = loom_llvmir_lowering_map_single_result(state, op, result);
  }
  iree_allocator_free(state->allocator, args);
  return status;
}

static iree_status_t loom_llvmir_lowering_lower_return(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  if (op->operand_count == 0) {
    return loom_llvmir_build_ret_void(target_block);
  }
  if (op->operand_count != 1) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "returns with multiple values need aggregate/sret policy");
  }
  loom_llvmir_value_id_t value = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_value(
      state, loom_op_const_operands(op)[0], &value));
  return loom_llvmir_build_ret(target_block, value);
}

static iree_status_t loom_llvmir_lowering_lower_op(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  switch (op->kind) {
    case LOOM_OP_SCALAR_ADDI:
    case LOOM_OP_SCALAR_SUBI:
    case LOOM_OP_SCALAR_MULI:
    case LOOM_OP_SCALAR_DIVSI:
    case LOOM_OP_SCALAR_DIVUI:
    case LOOM_OP_SCALAR_REMSI:
    case LOOM_OP_SCALAR_REMUI:
    case LOOM_OP_SCALAR_ADDF:
    case LOOM_OP_SCALAR_SUBF:
    case LOOM_OP_SCALAR_MULF:
    case LOOM_OP_SCALAR_DIVF:
    case LOOM_OP_SCALAR_REMF:
    case LOOM_OP_SCALAR_ANDI:
    case LOOM_OP_SCALAR_ORI:
    case LOOM_OP_SCALAR_XORI:
    case LOOM_OP_SCALAR_SHLI:
    case LOOM_OP_SCALAR_SHRSI:
    case LOOM_OP_SCALAR_SHRUI:
    case LOOM_OP_INDEX_ADD:
    case LOOM_OP_INDEX_SUB:
    case LOOM_OP_INDEX_MUL:
    case LOOM_OP_INDEX_DIV:
    case LOOM_OP_INDEX_REM:
      return loom_llvmir_lowering_lower_binop(state, target_block, op);
    case LOOM_OP_SCALAR_NEGF:
      return loom_llvmir_lowering_lower_negf(state, target_block, op);
    case LOOM_OP_SCALAR_CONSTANT:
      return loom_llvmir_lowering_lower_constant(
          state, op, loom_scalar_constant_value(op));
    case LOOM_OP_INDEX_CONSTANT:
      return loom_llvmir_lowering_lower_constant(state, op,
                                                 loom_index_constant_value(op));
    case LOOM_OP_INDEX_MADD:
      return loom_llvmir_lowering_lower_index_madd(state, target_block, op);
    case LOOM_OP_SCALAR_CMPI:
      return loom_llvmir_lowering_lower_icmp(state, target_block, op,
                                             loom_scalar_cmpi_predicate(op));
    case LOOM_OP_INDEX_CMP:
      return loom_llvmir_lowering_lower_icmp(state, target_block, op,
                                             loom_index_cmp_predicate(op));
    case LOOM_OP_SCALAR_CMPF:
      return loom_llvmir_lowering_lower_fcmp(state, target_block, op);
    case LOOM_OP_SCALAR_SELECT:
    case LOOM_OP_INDEX_SELECT:
      return loom_llvmir_lowering_lower_select(state, target_block, op);
    case LOOM_OP_SCALAR_SITOFP:
    case LOOM_OP_SCALAR_UITOFP:
    case LOOM_OP_SCALAR_FPTOSI:
    case LOOM_OP_SCALAR_FPTOUI:
    case LOOM_OP_SCALAR_EXTF:
    case LOOM_OP_SCALAR_FPTRUNC:
    case LOOM_OP_SCALAR_EXTSI:
    case LOOM_OP_SCALAR_EXTUI:
    case LOOM_OP_SCALAR_TRUNCI:
    case LOOM_OP_SCALAR_BITCAST:
      return loom_llvmir_lowering_lower_cast(state, target_block, op);
    case LOOM_OP_INDEX_CAST:
      return loom_llvmir_lowering_lower_index_cast(state, target_block, op);
    case LOOM_OP_BUFFER_ALLOCA:
      return loom_llvmir_lowering_lower_alloca(state, target_block, op);
    case LOOM_OP_BUFFER_ASSUME_MEMORY_SPACE:
      return loom_llvmir_lowering_lower_assume_memory_space(state, target_block,
                                                            op);
    case LOOM_OP_BUFFER_VIEW:
      return loom_llvmir_lowering_lower_buffer_view(state, target_block, op);
    case LOOM_OP_VIEW_SUBVIEW:
      return loom_llvmir_lowering_lower_subview(state, target_block, op);
    case LOOM_OP_VIEW_REFINE:
      return loom_llvmir_lowering_lower_refine(state, op);
    case LOOM_OP_VIEW_LOAD:
      return loom_llvmir_lowering_lower_view_load(state, target_block, op);
    case LOOM_OP_VIEW_STORE:
      return loom_llvmir_lowering_lower_view_store(state, target_block, op);
    case LOOM_OP_FUNC_CALL:
      return loom_llvmir_lowering_lower_call(state, target_block, op);
    case LOOM_OP_FUNC_RETURN:
      return loom_llvmir_lowering_lower_return(state, target_block, op);
    default:
      return loom_llvmir_lowering_unsupported_op(
          state, op, "no lowering rule is registered");
  }
}

static iree_status_t loom_llvmir_lowering_lower_body(
    loom_llvmir_lowering_state_t* state,
    loom_llvmir_function_t* target_function, loom_func_like_t func) {
  loom_region_t* body = loom_func_like_body(func);
  if (!body) return iree_ok_status();
  if (body->block_count != 1) {
    return loom_llvmir_lowering_unsupported_op(
        state, func.op, "CFG lowering is not wired to LLVMIR blocks yet");
  }
  loom_block_t* source_block = loom_region_entry_block(body);
  loom_llvmir_block_t* target_block = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_function_add_block(
      target_function, loom_llvmir_lowering_block_name(state, source_block),
      &target_block));

  iree_arena_block_pool_t fact_block_pool;
  iree_arena_block_pool_initialize(4096, state->allocator, &fact_block_pool);
  iree_arena_allocator_t fact_arena;
  iree_arena_initialize(&fact_block_pool, &fact_arena);
  loom_value_fact_table_t fact_table;
  iree_status_t status = loom_value_fact_table_initialize(
      &fact_table, &fact_arena, state->source_module->values.count);
  if (iree_status_is_ok(status)) {
    status =
        loom_value_fact_table_compute(&fact_table, state->source_module, func);
  }
  const loom_value_fact_table_t* previous_fact_table = state->fact_table;
  if (iree_status_is_ok(status)) {
    state->fact_table = &fact_table;
    loom_op_t* source_op = NULL;
    loom_block_for_each_op(source_block, source_op) {
      if (!iree_status_is_ok(status)) break;
      status = loom_llvmir_lowering_lower_op(state, target_block, source_op);
    }
    state->fact_table = previous_fact_table;
  }
  iree_arena_deinitialize(&fact_arena);
  iree_arena_block_pool_deinitialize(&fact_block_pool);
  return status;
}

static iree_status_t loom_llvmir_lowering_lower_function_bodies(
    loom_llvmir_lowering_state_t* state) {
  for (iree_host_size_t i = 0; i < state->source_module->symbols.count; ++i) {
    loom_symbol_t* symbol = &state->source_module->symbols.entries[i];
    if (symbol->kind != LOOM_SYMBOL_FUNC_DEF) continue;
    loom_func_like_t func =
        loom_func_like_cast(state->source_module, symbol->defining_op);
    IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_body(
        state, state->symbol_functions[i], func));
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_lowering_state_initialize(
    loom_llvmir_lowering_state_t* state, const loom_module_t* source_module,
    const loom_llvmir_lowering_options_t* options, iree_allocator_t allocator) {
  memset(state, 0, sizeof(*state));
  state->source_module = source_module;
  state->target_profile = options->target_profile;
  state->allocator = allocator;
  state->value_map_count = source_module->values.count;
  state->symbol_function_count = source_module->symbols.count;

  iree_string_view_t source_name = options->source_name;
  if (iree_string_view_is_empty(source_name)) {
    source_name = loom_llvmir_lowering_module_name(source_module);
  }
  loom_llvmir_target_config_t target_config = {0};
  IREE_RETURN_IF_ERROR(loom_llvmir_target_profile_module_config(
      options->target_profile, source_name, &target_config));
  IREE_RETURN_IF_ERROR(loom_llvmir_module_allocate(&target_config, allocator,
                                                   &state->target_module));

  iree_status_t status = iree_ok_status();
  if (state->value_map_count > 0) {
    status = iree_allocator_malloc(
        allocator, state->value_map_count * sizeof(*state->value_map),
        (void**)&state->value_map);
  }
  if (iree_status_is_ok(status) && state->value_map_count > 0) {
    status = iree_allocator_malloc(
        allocator,
        state->value_map_count * sizeof(*state->value_pointer_address_spaces),
        (void**)&state->value_pointer_address_spaces);
  }
  if (iree_status_is_ok(status) && state->value_map_count > 0) {
    status = iree_allocator_malloc(
        allocator,
        state->value_map_count * sizeof(*state->value_pointer_alignments),
        (void**)&state->value_pointer_alignments);
  }
  if (iree_status_is_ok(status) && state->symbol_function_count > 0) {
    status = iree_allocator_malloc(
        allocator,
        state->symbol_function_count * sizeof(*state->symbol_functions),
        (void**)&state->symbol_functions);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < state->value_map_count; ++i) {
      state->value_map[i] = LOOM_LLVMIR_VALUE_ID_INVALID;
      state->value_pointer_address_spaces[i] = UINT32_MAX;
      state->value_pointer_alignments[i] = 0;
    }
    for (iree_host_size_t i = 0; i < state->symbol_function_count; ++i) {
      state->symbol_functions[i] = NULL;
    }
  }
  if (!iree_status_is_ok(status)) {
    if (state->target_module) {
      loom_llvmir_module_free(state->target_module);
      state->target_module = NULL;
    }
    iree_allocator_free(state->allocator, state->symbol_functions);
    state->symbol_functions = NULL;
    iree_allocator_free(state->allocator, state->value_pointer_alignments);
    state->value_pointer_alignments = NULL;
    iree_allocator_free(state->allocator, state->value_pointer_address_spaces);
    state->value_pointer_address_spaces = NULL;
    iree_allocator_free(state->allocator, state->value_map);
    state->value_map = NULL;
  }
  return status;
}

static void loom_llvmir_lowering_state_deinitialize(
    loom_llvmir_lowering_state_t* state) {
  iree_allocator_free(state->allocator, state->symbol_functions);
  iree_allocator_free(state->allocator, state->value_pointer_alignments);
  iree_allocator_free(state->allocator, state->value_pointer_address_spaces);
  iree_allocator_free(state->allocator, state->value_map);
}

iree_status_t loom_llvmir_lower_module(
    const loom_module_t* source_module,
    const loom_llvmir_lowering_options_t* options, iree_allocator_t allocator,
    loom_llvmir_module_t** out_module) {
  if (!source_module || !options || !options->target_profile ||
      !options->target_profile->target_env || !out_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source module, options, target profile, target "
                            "environment, and out_module are required");
  }
  *out_module = NULL;

  loom_llvmir_lowering_state_t state;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_state_initialize(
      &state, source_module, options, allocator));

  iree_status_t status = loom_llvmir_lowering_lower_function_signatures(&state);
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_lowering_lower_function_bodies(&state);
  }
  if (iree_status_is_ok(status)) {
    *out_module = state.target_module;
    state.target_module = NULL;
  }
  if (state.target_module) {
    loom_llvmir_module_free(state.target_module);
  }
  loom_llvmir_lowering_state_deinitialize(&state);
  return status;
}
