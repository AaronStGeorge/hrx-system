// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/ir/context.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/llvmir/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/emit/llvmir/lower_internal.h"
#include "loom/util/cfg_graph.h"

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

iree_string_view_t loom_llvmir_lowering_value_name(
    const loom_llvmir_lowering_state_t* state, loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(state->source_module, value_id);
  if (value->name_id == LOOM_STRING_ID_INVALID) return IREE_SV("");
  return state->source_module->strings.entries[value->name_id];
}

static iree_string_view_t loom_llvmir_lowering_block_name(
    const loom_llvmir_lowering_state_t* state, const loom_block_t* block,
    bool is_entry_block) {
  if (block->label_id == LOOM_STRING_ID_INVALID) {
    return is_entry_block ? IREE_SV("entry") : IREE_SV("");
  }
  return state->source_module->strings.entries[block->label_id];
}

iree_status_t loom_llvmir_lowering_unsupported_op(
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

iree_status_t loom_llvmir_lowering_unsupported_type(
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

bool loom_llvmir_lowering_lookup_provider_intrinsic(
    const loom_llvmir_lowering_state_t* state, const void* key,
    loom_llvmir_function_t** out_function) {
  for (iree_host_size_t i = 0; i < state->provider_intrinsic_function_count;
       ++i) {
    if (state->provider_intrinsic_keys[i] == key) {
      *out_function = state->provider_intrinsic_functions[i];
      return true;
    }
  }
  *out_function = NULL;
  return false;
}

iree_status_t loom_llvmir_lowering_cache_provider_intrinsic(
    loom_llvmir_lowering_state_t* state, const void* key,
    loom_llvmir_function_t* function) {
  if (state->provider_intrinsic_function_count >=
      IREE_ARRAYSIZE(state->provider_intrinsic_functions)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "LLVM target-provider intrinsic cache capacity exceeded");
  }
  iree_host_size_t cache_ordinal = state->provider_intrinsic_function_count++;
  state->provider_intrinsic_keys[cache_ordinal] = key;
  state->provider_intrinsic_functions[cache_ordinal] = function;
  return iree_ok_status();
}

iree_status_t loom_llvmir_lowering_declare_provider_intrinsic_cached(
    loom_llvmir_lowering_state_t* state, const void* key,
    loom_llvmir_lowering_provider_intrinsic_decl_fn_t declare_fn,
    loom_llvmir_function_t** out_function) {
  if (loom_llvmir_lowering_lookup_provider_intrinsic(state, key,
                                                     out_function)) {
    return iree_ok_status();
  }
  loom_llvmir_function_t* function = NULL;
  IREE_RETURN_IF_ERROR(declare_fn(state->target_module, &function));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_cache_provider_intrinsic(state, key, function));
  *out_function = function;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_lowering_get_integer_type(
    loom_llvmir_lowering_state_t* state, uint32_t bit_width,
    loom_llvmir_type_id_t* out_type_id) {
  return loom_llvmir_module_get_integer_type(state->target_module, bit_width,
                                             out_type_id);
}

iree_status_t loom_llvmir_lowering_lower_scalar_type(
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
    case LOOM_SCALAR_TYPE_BF16:
      return loom_llvmir_module_get_float_type(
          state->target_module, LOOM_LLVMIR_FLOAT_BF16, out_type_id);
    case LOOM_SCALAR_TYPE_F32:
      return loom_llvmir_module_get_float_type(
          state->target_module, LOOM_LLVMIR_FLOAT_F32, out_type_id);
    case LOOM_SCALAR_TYPE_F64:
      return loom_llvmir_module_get_float_type(
          state->target_module, LOOM_LLVMIR_FLOAT_F64, out_type_id);
    case LOOM_SCALAR_TYPE_F8E4M3:
    case LOOM_SCALAR_TYPE_F8E5M2:
    case LOOM_SCALAR_TYPE_COUNT_:
      return loom_llvmir_lowering_unsupported_type(
          state, loom_type_scalar(scalar_type),
          "the structured LLVMIR type model does not carry this scalar yet");
  }
  return loom_llvmir_lowering_unsupported_type(
      state, loom_type_scalar(scalar_type), "unknown scalar type");
}

iree_status_t loom_llvmir_lowering_get_pointer_type(
    loom_llvmir_lowering_state_t* state, uint32_t address_space,
    loom_llvmir_type_id_t* out_type_id) {
  return loom_llvmir_module_get_pointer_type(state->target_module,
                                             address_space, out_type_id);
}

iree_status_t loom_llvmir_lowering_lower_type(
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

iree_status_t loom_llvmir_lowering_map_value(
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

iree_status_t loom_llvmir_lowering_map_pointer_value(
    loom_llvmir_lowering_state_t* state, loom_value_id_t source_value_id,
    loom_llvmir_value_id_t target_value_id, uint32_t address_space,
    uint64_t minimum_alignment) {
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_map_value(state, source_value_id, target_value_id));
  state->value_pointer_address_spaces[source_value_id] = address_space;
  state->value_pointer_alignments[source_value_id] = minimum_alignment;
  return iree_ok_status();
}

iree_status_t loom_llvmir_lowering_lookup_value(
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

iree_status_t loom_llvmir_lowering_lookup_pointer(
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

iree_status_t loom_llvmir_lowering_lookup_operands(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_llvmir_value_id_t* target_operands) {
  const loom_value_id_t* source_operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_value(
        state, source_operands[i], &target_operands[i]));
  }
  return iree_ok_status();
}

iree_status_t loom_llvmir_lowering_map_single_result(
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

static bool loom_llvmir_lowering_function_is_kernel_entry(
    const loom_llvmir_lowering_state_t* state, loom_func_like_t func) {
  return state->target_profile->kind == LOOM_LLVMIR_TARGET_PROFILE_HAL_KERNEL &&
         loom_func_like_visibility(func) == LOOM_FUNC_VISIBILITY_PUBLIC &&
         loom_func_like_cc(func) == LOOM_FUNC_CC_DEVICE;
}

static loom_llvmir_calling_convention_t
loom_llvmir_lowering_function_calling_convention(
    const loom_llvmir_lowering_state_t* state, loom_func_like_t func) {
  if (loom_llvmir_lowering_function_is_kernel_entry(state, func)) {
    return state->target_profile->kernel_calling_convention;
  }
  return LOOM_LLVMIR_CALLING_CONVENTION_DEFAULT;
}

static iree_status_t loom_llvmir_lowering_kernel_attr_group(
    loom_llvmir_lowering_state_t* state,
    loom_llvmir_attr_group_id_t* out_attr_group_id) {
  if (state->kernel_attr_group_id != LOOM_LLVMIR_ATTR_GROUP_ID_INVALID) {
    *out_attr_group_id = state->kernel_attr_group_id;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_target_profile_add_kernel_attr_group(
      state->target_module, state->target_profile,
      &state->kernel_attr_group_id));
  *out_attr_group_id = state->kernel_attr_group_id;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_lowering_validate_function_abi(
    const loom_llvmir_lowering_state_t* state, loom_func_like_t func,
    loom_llvmir_function_kind_t function_kind, bool is_kernel_entry) {
  if (state->target_profile->kind == LOOM_LLVMIR_TARGET_PROFILE_HAL_KERNEL &&
      loom_func_like_visibility(func) == LOOM_FUNC_VISIBILITY_PUBLIC &&
      loom_func_like_cc(func) != LOOM_FUNC_CC_DEVICE) {
    return loom_llvmir_lowering_unsupported_op(
        state, func.op,
        "HAL kernel profile can only export public device entry points");
  }
  if (!is_kernel_entry) return iree_ok_status();
  if (function_kind != LOOM_LLVMIR_FUNCTION_DEFINITION) {
    return loom_llvmir_lowering_unsupported_op(
        state, func.op, "HAL kernel entry points must be function definitions");
  }
  if (func.op->result_count != 0) {
    return loom_llvmir_lowering_unsupported_op(
        state, func.op, "HAL kernel entry points must return void");
  }
  uint16_t arg_count = 0;
  const loom_value_id_t* arg_ids = loom_func_like_arg_ids(func, &arg_count);
  for (uint16_t i = 0; i < arg_count; ++i) {
    loom_type_t arg_type =
        loom_module_value_type(state->source_module, arg_ids[i]);
    if (loom_type_is_view(arg_type)) {
      return loom_llvmir_lowering_unsupported_op(
          state, func.op,
          "HAL kernel entry point view parameters need an explicit ABI "
          "adapter");
    }
  }
  return iree_ok_status();
}

static uint32_t loom_llvmir_lowering_default_pointer_argument_address_space(
    const loom_llvmir_lowering_state_t* state) {
  if (state->target_profile->kind == LOOM_LLVMIR_TARGET_PROFILE_HAL_KERNEL) {
    return state->target_profile->target_env->address_spaces.global;
  }
  return state->target_profile->target_env->address_spaces.generic;
}

static iree_status_t loom_llvmir_lowering_add_function_signature(
    loom_llvmir_lowering_state_t* state, uint16_t symbol_id,
    loom_func_like_t func) {
  loom_symbol_t* symbol = &state->source_module->symbols.entries[symbol_id];
  loom_region_t* body = loom_func_like_body(func);
  loom_llvmir_function_kind_t function_kind =
      body ? LOOM_LLVMIR_FUNCTION_DEFINITION : LOOM_LLVMIR_FUNCTION_DECLARATION;
  bool is_kernel_entry =
      loom_llvmir_lowering_function_is_kernel_entry(state, func);
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_validate_function_abi(
      state, func, function_kind, is_kernel_entry));
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
  if (is_kernel_entry) {
    IREE_RETURN_IF_ERROR(loom_llvmir_lowering_kernel_attr_group(
        state, &function_desc.attr_group_id));
  }

  loom_llvmir_function_t* target_function = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_function(
      state->target_module, &function_desc, &target_function));
  state->symbol_functions[symbol_id] = target_function;
  if (is_kernel_entry) {
    IREE_RETURN_IF_ERROR(loom_llvmir_target_profile_attach_kernel_metadata(
        target_function, state->target_profile));
  }

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
          loom_llvmir_lowering_default_pointer_argument_address_space(state);
      IREE_RETURN_IF_ERROR(loom_llvmir_lowering_get_pointer_type(
          state, pointer_address_space, &arg_type));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_llvmir_lowering_lower_type(state, source_arg_type, &arg_type));
    }

    loom_llvmir_parameter_desc_t parameter_desc = {0};
    parameter_desc.type_id = arg_type;
    parameter_desc.name = loom_llvmir_lowering_value_name(state, arg_ids[i]);
    uint64_t pointer_alignment = 0;
    loom_llvmir_attr_t kernel_binding_attrs
        [LOOM_LLVMIR_TARGET_PROFILE_MAX_KERNEL_BINDING_ATTR_COUNT];
    iree_host_size_t kernel_binding_attr_count = 0;
    if (is_kernel_entry && loom_type_is_buffer(source_arg_type)) {
      IREE_RETURN_IF_ERROR(loom_llvmir_target_profile_kernel_binding_attrs(
          state->target_profile, kernel_binding_attrs,
          IREE_ARRAYSIZE(kernel_binding_attrs), &kernel_binding_attr_count));
      parameter_desc.attrs = kernel_binding_attrs;
      parameter_desc.attr_count = kernel_binding_attr_count;
      pointer_alignment = state->target_profile->amdgpu_hal.binding_alignment;
    }
    loom_llvmir_value_id_t parameter_value = LOOM_LLVMIR_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_llvmir_function_add_parameter(
        target_function, &parameter_desc, &parameter_value));
    if (pointer_address_space == UINT32_MAX) {
      IREE_RETURN_IF_ERROR(
          loom_llvmir_lowering_map_value(state, arg_ids[i], parameter_value));
    } else {
      IREE_RETURN_IF_ERROR(loom_llvmir_lowering_map_pointer_value(
          state, arg_ids[i], parameter_value, pointer_address_space,
          pointer_alignment));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_lowering_lower_function_signatures(
    loom_llvmir_lowering_state_t* state) {
  for (iree_host_size_t i = 0; i < state->source_module->symbols.count; ++i) {
    loom_symbol_t* symbol = &state->source_module->symbols.entries[i];
    if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
      continue;
    }
    if (!symbol->defining_op) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "function-like symbol id %u has no defining op",
                              (unsigned)i);
    }
    loom_symbol_kind_t bytecode_kind = loom_symbol_bytecode_kind(symbol);
    if (bytecode_kind != LOOM_SYMBOL_FUNC_DEF &&
        bytecode_kind != LOOM_SYMBOL_FUNC_DECL) {
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

iree_status_t loom_llvmir_lowering_try_provider_op(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op, bool* out_handled) {
  *out_handled = false;
  for (iree_host_size_t i = 0; i < state->provider_count; ++i) {
    const loom_llvmir_lowering_provider_t* provider = state->providers[i];
    bool handled = false;
    IREE_RETURN_IF_ERROR(
        provider->try_lower_op(provider, state, target_block, op, &handled));
    if (handled) {
      *out_handled = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_lowering_lower_provider_op_or_unsupported(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op, const char* unsupported_detail) {
  bool handled = false;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_try_provider_op(state, target_block, op, &handled));
  if (handled) {
    return iree_ok_status();
  }
  return loom_llvmir_lowering_unsupported_op(state, op, unsupported_detail);
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
    case LOOM_OP_VECTOR_ADDF:
    case LOOM_OP_VECTOR_SUBF:
    case LOOM_OP_VECTOR_MULF:
    case LOOM_OP_VECTOR_DIVF:
    case LOOM_OP_VECTOR_REMF:
    case LOOM_OP_VECTOR_ADDI:
    case LOOM_OP_VECTOR_SUBI:
    case LOOM_OP_VECTOR_MULI:
    case LOOM_OP_VECTOR_DIVSI:
    case LOOM_OP_VECTOR_DIVUI:
    case LOOM_OP_VECTOR_REMSI:
    case LOOM_OP_VECTOR_REMUI:
    case LOOM_OP_VECTOR_ANDI:
    case LOOM_OP_VECTOR_ORI:
    case LOOM_OP_VECTOR_XORI:
    case LOOM_OP_VECTOR_SHLI:
    case LOOM_OP_VECTOR_SHRSI:
    case LOOM_OP_VECTOR_SHRUI:
      return loom_llvmir_lowering_lower_vector_binop(state, target_block, op);
    case LOOM_OP_VECTOR_DOT2F:
    case LOOM_OP_VECTOR_DOT4I:
      return loom_llvmir_lowering_lower_provider_op_or_unsupported(
          state, target_block, op,
          "vector dot op requires an explicit target lowering provider");
    case LOOM_OP_VECTOR_DOT8I4:
      return loom_llvmir_lowering_unsupported_op(
          state, op,
          "packed i4 dot needs explicit unpack/reference expansion or "
          "target-contract lowering");
    case LOOM_OP_VECTOR_DOT4F8:
      return loom_llvmir_lowering_unsupported_op(
          state, op,
          "packed fp8/bf8 dot needs explicit decode/reference expansion or "
          "target-contract lowering");
    case LOOM_OP_SCALAR_NEGF:
      return loom_llvmir_lowering_lower_negf(state, target_block, op);
    case LOOM_OP_VECTOR_NEGF:
      return loom_llvmir_lowering_lower_vector_negf(state, target_block, op);
    case LOOM_OP_SCALAR_CONSTANT:
      return loom_llvmir_lowering_lower_constant(
          state, op, loom_scalar_constant_value(op));
    case LOOM_OP_INDEX_CONSTANT:
      return loom_llvmir_lowering_lower_constant(state, op,
                                                 loom_index_constant_value(op));
    case LOOM_OP_VECTOR_CONSTANT:
      return loom_llvmir_lowering_lower_vector_constant(
          state, op, loom_vector_constant_value(op));
    case LOOM_OP_VECTOR_POISON:
      return loom_llvmir_lowering_lower_vector_poison(state, op);
    case LOOM_OP_VECTOR_SPLAT:
      return loom_llvmir_lowering_lower_vector_splat(state, target_block, op);
    case LOOM_OP_VECTOR_FROM_ELEMENTS:
      return loom_llvmir_lowering_lower_vector_from_elements(state,
                                                             target_block, op);
    case LOOM_OP_VECTOR_EXTRACT:
      return loom_llvmir_lowering_lower_vector_extract(state, target_block, op);
    case LOOM_OP_VECTOR_INSERT:
      return loom_llvmir_lowering_lower_vector_insert(state, target_block, op);
    case LOOM_OP_VECTOR_SHUFFLE:
      return loom_llvmir_lowering_lower_vector_shuffle(state, target_block, op);
    case LOOM_OP_INDEX_MADD:
      return loom_llvmir_lowering_lower_index_madd(state, target_block, op);
    case LOOM_OP_SCALAR_CMPI:
      return loom_llvmir_lowering_lower_icmp(state, target_block, op,
                                             loom_scalar_cmpi_predicate(op));
    case LOOM_OP_INDEX_CMP:
      return loom_llvmir_lowering_lower_icmp(state, target_block, op,
                                             loom_index_cmp_predicate(op));
    case LOOM_OP_VECTOR_CMPI:
      return loom_llvmir_lowering_lower_vector_icmp(state, target_block, op);
    case LOOM_OP_SCALAR_CMPF:
      return loom_llvmir_lowering_lower_fcmp(state, target_block, op);
    case LOOM_OP_VECTOR_CMPF:
      return loom_llvmir_lowering_lower_vector_fcmp(state, target_block, op);
    case LOOM_OP_SCF_SELECT:
      return loom_llvmir_lowering_lower_select(state, target_block, op);
    case LOOM_OP_VECTOR_SELECT:
      return loom_llvmir_lowering_lower_vector_select(state, target_block, op);
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
    case LOOM_OP_VECTOR_SITOFP:
    case LOOM_OP_VECTOR_UITOFP:
    case LOOM_OP_VECTOR_FPTOSI:
    case LOOM_OP_VECTOR_FPTOUI:
    case LOOM_OP_VECTOR_EXTF:
    case LOOM_OP_VECTOR_FPTRUNC:
    case LOOM_OP_VECTOR_EXTSI:
    case LOOM_OP_VECTOR_EXTUI:
    case LOOM_OP_VECTOR_TRUNCI:
    case LOOM_OP_VECTOR_BITCAST:
      return loom_llvmir_lowering_lower_vector_cast(state, target_block, op);
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
    case LOOM_OP_VECTOR_LOAD:
      return loom_llvmir_lowering_lower_vector_load(state, target_block, op);
    case LOOM_OP_VECTOR_STORE:
      return loom_llvmir_lowering_lower_vector_store(state, target_block, op);
    case LOOM_OP_VIEW_PREFETCH:
      return loom_llvmir_lowering_lower_view_prefetch(state, target_block, op);
    case LOOM_OP_LLVMIR_INLINE_ASM:
      return loom_llvmir_lowering_lower_inline_asm(state, target_block, op);
    case LOOM_OP_LLVMIR_INTRINSIC:
      return loom_llvmir_lowering_lower_intrinsic(state, target_block, op);
    case LOOM_OP_SCF_IF:
    case LOOM_OP_SCF_FOR:
    case LOOM_OP_SCF_WHILE:
    case LOOM_OP_SCF_SWITCH:
      return loom_llvmir_lowering_unsupported_op(
          state, op,
          "structured SCF control flow must be lowered to CFG before LLVMIR "
          "lowering");
    case LOOM_OP_SCF_CONDITION:
    case LOOM_OP_SCF_YIELD:
      return loom_llvmir_lowering_unsupported_op(
          state, op,
          "SCF terminators must be lowered with their parent structured "
          "control-flow op before LLVMIR lowering");
    case LOOM_OP_FUNC_CALL:
      return loom_llvmir_lowering_lower_call(state, target_block, op);
    case LOOM_OP_FUNC_RETURN:
      return loom_llvmir_lowering_lower_return(state, target_block, op);
    default:
      return loom_llvmir_lowering_unsupported_op(
          state, op, "no lowering rule is registered");
  }
}

static iree_status_t loom_llvmir_lowering_lower_source_block(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    loom_block_t* source_block) {
  loom_op_t* source_op = NULL;
  loom_block_for_each_op(source_block, source_op) {
    IREE_RETURN_IF_ERROR(
        loom_llvmir_lowering_lower_op(state, target_block, source_op));
  }
  return iree_ok_status();
}

typedef struct loom_llvmir_cfg_phi_accumulator_t {
  // Source block argument represented by the phi result.
  loom_value_id_t source_arg_id;
  // LLVM block containing the phi instruction.
  loom_llvmir_block_t* target_block;
  // LLVM phi instruction result value.
  loom_llvmir_value_id_t phi_value_id;
  // Incoming edge list allocated from the function-local arena.
  loom_llvmir_phi_incoming_t* incoming;
  // Number of incoming edges written so far.
  iree_host_size_t incoming_count;
  // Number of reachable predecessor edges expected by the CFG graph.
  iree_host_size_t incoming_capacity;
  // True when the block argument lowers as an LLVM pointer.
  bool is_pointer;
  // Phi pointer address space when |is_pointer| is true.
  uint32_t pointer_address_space;
} loom_llvmir_cfg_phi_accumulator_t;

typedef struct loom_llvmir_cfg_lowering_t {
  // Dense graph for the source CFG region.
  loom_cfg_graph_t graph;
  // LLVM block for each source block in graph order.
  loom_llvmir_block_t** target_blocks;
  // Offset of each block's first non-entry block-argument phi accumulator.
  iree_host_size_t* phi_starts;
  // Flat phi accumulator table for all non-entry block arguments.
  loom_llvmir_cfg_phi_accumulator_t* phis;
  // Number of entries in |phis|.
  iree_host_size_t phi_count;
} loom_llvmir_cfg_lowering_t;

static bool loom_llvmir_lowering_type_is_pointer_like(loom_type_t type) {
  return loom_type_is_buffer(type) || loom_type_is_view(type);
}

static loom_llvmir_cfg_phi_accumulator_t* loom_llvmir_lowering_cfg_phi_for_arg(
    loom_llvmir_cfg_lowering_t* cfg, uint16_t block_index, uint16_t arg_index) {
  IREE_ASSERT(block_index < cfg->graph.block_count);
  IREE_ASSERT(block_index > 0);
  const iree_host_size_t phi_index = cfg->phi_starts[block_index] + arg_index;
  IREE_ASSERT(phi_index < cfg->phi_count);
  return &cfg->phis[phi_index];
}

static iree_status_t loom_llvmir_lowering_cfg_append_incoming(
    loom_llvmir_cfg_phi_accumulator_t* phi,
    loom_llvmir_value_id_t incoming_value, loom_llvmir_block_id_t predecessor) {
  if (phi->incoming_count >= phi->incoming_capacity) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "CFG phi incoming edge count exceeds predecessor "
                            "edge count");
  }
  phi->incoming[phi->incoming_count++] = (loom_llvmir_phi_incoming_t){
      .value = incoming_value,
      .predecessor = predecessor,
  };
  return iree_ok_status();
}

static iree_status_t loom_llvmir_lowering_cfg_build_block_arg_phi(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_block_t* source_block, uint16_t arg_index,
    iree_host_size_t predecessor_count, iree_arena_allocator_t* arena,
    loom_llvmir_cfg_phi_accumulator_t* phi) {
  loom_value_id_t arg_id = loom_block_arg_id(source_block, arg_index);
  loom_type_t arg_type = loom_module_value_type(state->source_module, arg_id);
  loom_llvmir_type_id_t phi_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  bool is_pointer = loom_llvmir_lowering_type_is_pointer_like(arg_type);
  uint32_t pointer_address_space = UINT32_MAX;
  if (is_pointer) {
    pointer_address_space =
        state->target_profile->target_env->address_spaces.generic;
    IREE_RETURN_IF_ERROR(loom_llvmir_lowering_get_pointer_type(
        state, pointer_address_space, &phi_type));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_llvmir_lowering_lower_type(state, arg_type, &phi_type));
  }

  loom_llvmir_value_id_t phi_value_id = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_phi(
      target_block,
      &(loom_llvmir_phi_desc_t){
          .result_name = loom_llvmir_lowering_value_name(state, arg_id),
          .result_type = phi_type,
      },
      &phi_value_id));

  *phi = (loom_llvmir_cfg_phi_accumulator_t){
      .source_arg_id = arg_id,
      .target_block = target_block,
      .phi_value_id = phi_value_id,
      .incoming = NULL,
      .incoming_count = 0,
      .incoming_capacity = predecessor_count,
      .is_pointer = is_pointer,
      .pointer_address_space = pointer_address_space,
  };
  if (predecessor_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, predecessor_count,
                                                   sizeof(*phi->incoming),
                                                   (void**)&phi->incoming));
  }
  if (is_pointer) {
    return loom_llvmir_lowering_map_pointer_value(state, arg_id, phi_value_id,
                                                  pointer_address_space,
                                                  /*minimum_alignment=*/0);
  }
  return loom_llvmir_lowering_map_value(state, arg_id, phi_value_id);
}

static iree_host_size_t loom_llvmir_lowering_cfg_reachable_predecessor_count(
    const loom_cfg_graph_t* graph, uint16_t block_index) {
  loom_cfg_block_index_span_t predecessors =
      loom_cfg_graph_predecessors(graph, block_index);
  iree_host_size_t reachable_count = 0;
  for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
    if (loom_cfg_graph_block_is_reachable(graph, predecessors.values[i])) {
      ++reachable_count;
    }
  }
  return reachable_count;
}

static iree_status_t loom_llvmir_lowering_cfg_prepare_blocks(
    loom_llvmir_lowering_state_t* state,
    loom_llvmir_function_t* target_function, loom_region_t* body,
    const loom_op_t* owner_op, iree_arena_allocator_t* arena,
    loom_llvmir_cfg_lowering_t* cfg) {
  memset(cfg, 0, sizeof(*cfg));
  IREE_RETURN_IF_ERROR(loom_cfg_graph_build(body, arena, &cfg->graph));
  if (cfg->graph.malformed) {
    return loom_llvmir_lowering_unsupported_op(
        state, owner_op,
        "CFG graph is malformed; run Loom verification before lowering");
  }
  if (cfg->graph.block_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "function body CFG has no blocks");
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, cfg->graph.block_count,
                                                 sizeof(*cfg->target_blocks),
                                                 (void**)&cfg->target_blocks));
  for (uint16_t block_index = 0; block_index < cfg->graph.block_count;
       ++block_index) {
    const loom_block_t* source_block = cfg->graph.blocks[block_index].block;
    IREE_RETURN_IF_ERROR(loom_llvmir_function_add_block(
        target_function,
        loom_llvmir_lowering_block_name(state, source_block, block_index == 0),
        &cfg->target_blocks[block_index]));
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, cfg->graph.block_count + 1, sizeof(*cfg->phi_starts),
      (void**)&cfg->phi_starts));
  iree_host_size_t phi_count = 0;
  for (uint16_t block_index = 0; block_index < cfg->graph.block_count;
       ++block_index) {
    cfg->phi_starts[block_index] = phi_count;
    if (block_index == 0 ||
        !loom_cfg_graph_block_is_reachable(&cfg->graph, block_index)) {
      continue;
    }
    const loom_block_t* source_block = cfg->graph.blocks[block_index].block;
    phi_count += source_block->arg_count;
  }
  cfg->phi_starts[cfg->graph.block_count] = phi_count;
  cfg->phi_count = phi_count;
  if (phi_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, phi_count, sizeof(*cfg->phis), (void**)&cfg->phis));
  }

  for (uint16_t block_index = 1; block_index < cfg->graph.block_count;
       ++block_index) {
    if (!loom_cfg_graph_block_is_reachable(&cfg->graph, block_index)) {
      continue;
    }
    const loom_block_t* source_block = cfg->graph.blocks[block_index].block;
    iree_host_size_t predecessor_count =
        loom_llvmir_lowering_cfg_reachable_predecessor_count(&cfg->graph,
                                                             block_index);
    for (uint16_t arg_index = 0; arg_index < source_block->arg_count;
         ++arg_index) {
      loom_llvmir_cfg_phi_accumulator_t* phi =
          loom_llvmir_lowering_cfg_phi_for_arg(cfg, block_index, arg_index);
      IREE_RETURN_IF_ERROR(loom_llvmir_lowering_cfg_build_block_arg_phi(
          state, cfg->target_blocks[block_index], source_block, arg_index,
          predecessor_count, arena, phi));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_lowering_cfg_cast_pointer_to_phi_type(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    loom_value_id_t source_value_id, uint32_t phi_address_space,
    loom_llvmir_value_id_t* out_value) {
  loom_llvmir_value_id_t value = LOOM_LLVMIR_VALUE_ID_INVALID;
  uint32_t address_space = UINT32_MAX;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_pointer(
      state, source_value_id, &value, &address_space, NULL));
  if (address_space == phi_address_space) {
    *out_value = value;
    return iree_ok_status();
  }

  loom_llvmir_type_id_t phi_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_get_pointer_type(
      state, phi_address_space, &phi_type));
  return loom_llvmir_build_cast(target_block,
                                &(loom_llvmir_cast_desc_t){
                                    .result_type = phi_type,
                                    .op = LOOM_LLVMIR_CAST_ADDRESS_SPACE_CAST,
                                    .value = value,
                                },
                                out_value);
}

static iree_status_t loom_llvmir_lowering_cfg_edge_value(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_llvmir_cfg_phi_accumulator_t* phi,
    loom_value_id_t source_value_id, loom_llvmir_value_id_t* out_value) {
  if (phi->is_pointer) {
    return loom_llvmir_lowering_cfg_cast_pointer_to_phi_type(
        state, target_block, source_value_id, phi->pointer_address_space,
        out_value);
  }
  return loom_llvmir_lowering_lookup_value(state, source_value_id, out_value);
}

static iree_status_t loom_llvmir_lowering_lower_cfg_br(
    loom_llvmir_lowering_state_t* state, loom_llvmir_cfg_lowering_t* cfg,
    loom_llvmir_block_t* target_block, const loom_op_t* op) {
  loom_block_t* dest = loom_cfg_br_dest(op);
  iree_host_size_t dest_index = loom_cfg_graph_block_index(&cfg->graph, dest);
  if (dest_index == IREE_HOST_SIZE_MAX) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "cfg.br destination must belong to the function body CFG");
  }
  const loom_block_t* dest_block = cfg->graph.blocks[dest_index].block;
  loom_value_slice_t args = loom_cfg_br_args(op);
  if (args.count != dest_block->arg_count) {
    return loom_llvmir_lowering_unsupported_op(
        state, op,
        "cfg.br operand count must match destination block arguments");
  }

  loom_llvmir_block_id_t predecessor = loom_llvmir_block_id(target_block);
  for (uint16_t arg_index = 0; arg_index < dest_block->arg_count; ++arg_index) {
    if (dest_index == 0) {
      return loom_llvmir_lowering_unsupported_op(
          state, op,
          "cfg.br cannot pass arguments to the function entry block");
    }
    loom_llvmir_cfg_phi_accumulator_t* phi =
        loom_llvmir_lowering_cfg_phi_for_arg(cfg, (uint16_t)dest_index,
                                             arg_index);
    loom_llvmir_value_id_t incoming_value = LOOM_LLVMIR_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_llvmir_lowering_cfg_edge_value(
        state, target_block, phi, args.values[arg_index], &incoming_value));
    IREE_RETURN_IF_ERROR(loom_llvmir_lowering_cfg_append_incoming(
        phi, incoming_value, predecessor));
  }
  return loom_llvmir_build_br(
      target_block, loom_llvmir_block_id(cfg->target_blocks[dest_index]));
}

static iree_status_t loom_llvmir_lowering_cfg_cond_dest(
    const loom_llvmir_lowering_state_t* state,
    const loom_llvmir_cfg_lowering_t* cfg, const loom_op_t* op,
    loom_block_t* dest, loom_llvmir_block_id_t* out_block_id) {
  iree_host_size_t dest_index = loom_cfg_graph_block_index(&cfg->graph, dest);
  if (dest_index == IREE_HOST_SIZE_MAX) {
    return loom_llvmir_lowering_unsupported_op(
        state, op,
        "cfg.cond_br destination must belong to the function body CFG");
  }
  const loom_block_t* dest_block = cfg->graph.blocks[dest_index].block;
  if (dest_block->arg_count != 0) {
    return loom_llvmir_lowering_unsupported_op(
        state, op,
        "cfg.cond_br cannot target blocks with arguments because it carries no "
        "edge payloads");
  }
  *out_block_id = loom_llvmir_block_id(cfg->target_blocks[dest_index]);
  return iree_ok_status();
}

static iree_status_t loom_llvmir_lowering_lower_cfg_cond_br(
    loom_llvmir_lowering_state_t* state, loom_llvmir_cfg_lowering_t* cfg,
    loom_llvmir_block_t* target_block, const loom_op_t* op) {
  loom_llvmir_value_id_t condition = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_value(
      state, loom_cfg_cond_br_condition(op), &condition));
  loom_llvmir_block_id_t true_block = LOOM_LLVMIR_BLOCK_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_cfg_cond_dest(
      state, cfg, op, loom_cfg_cond_br_true_dest(op), &true_block));
  loom_llvmir_block_id_t false_block = LOOM_LLVMIR_BLOCK_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_cfg_cond_dest(
      state, cfg, op, loom_cfg_cond_br_false_dest(op), &false_block));
  return loom_llvmir_build_cond_br(target_block, condition, true_block,
                                   false_block);
}

static bool loom_llvmir_lowering_op_ends_cfg_block(const loom_op_t* op) {
  return loom_cfg_br_isa(op) || loom_cfg_cond_br_isa(op) ||
         loom_func_return_isa(op);
}

static iree_status_t loom_llvmir_lowering_lower_cfg_block(
    loom_llvmir_lowering_state_t* state,
    loom_llvmir_function_t* target_function, loom_llvmir_cfg_lowering_t* cfg,
    uint16_t block_index) {
  loom_block_t* source_block =
      (loom_block_t*)cfg->graph.blocks[block_index].block;
  loom_llvmir_block_t* current_block = cfg->target_blocks[block_index];
  bool saw_terminator = false;
  loom_op_t* source_op = NULL;
  loom_block_for_each_op(source_block, source_op) {
    if (saw_terminator) {
      return loom_llvmir_lowering_unsupported_op(
          state, source_op, "CFG terminator must be last in its block");
    }
    if (loom_cfg_br_isa(source_op)) {
      IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_cfg_br(
          state, cfg, current_block, source_op));
      saw_terminator = true;
    } else if (loom_cfg_cond_br_isa(source_op)) {
      IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_cfg_cond_br(
          state, cfg, current_block, source_op));
      saw_terminator = true;
    } else {
      IREE_RETURN_IF_ERROR(
          loom_llvmir_lowering_lower_op(state, current_block, source_op));
      saw_terminator = loom_llvmir_lowering_op_ends_cfg_block(source_op);
    }
  }
  if (!saw_terminator) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "CFG block is missing a terminator");
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_lowering_lower_cfg_body(
    loom_llvmir_lowering_state_t* state,
    loom_llvmir_function_t* target_function, loom_region_t* body,
    const loom_op_t* owner_op, iree_arena_allocator_t* arena) {
  loom_llvmir_cfg_lowering_t cfg = {0};
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_cfg_prepare_blocks(
      state, target_function, body, owner_op, arena, &cfg));
  for (uint16_t block_index = 0; block_index < cfg.graph.block_count;
       ++block_index) {
    if (!loom_cfg_graph_block_is_reachable(&cfg.graph, block_index)) continue;
    IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_cfg_block(
        state, target_function, &cfg, block_index));
  }
  for (uint16_t block_index = 0; block_index < cfg.graph.block_count;
       ++block_index) {
    if (loom_cfg_graph_block_is_reachable(&cfg.graph, block_index)) continue;
    IREE_RETURN_IF_ERROR(
        loom_llvmir_build_unreachable(cfg.target_blocks[block_index]));
  }
  for (iree_host_size_t i = 0; i < cfg.phi_count; ++i) {
    loom_llvmir_cfg_phi_accumulator_t* phi = &cfg.phis[i];
    if (phi->incoming_count != phi->incoming_capacity) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "CFG phi incoming edge count does not match "
                              "predecessor edge count");
    }
    IREE_RETURN_IF_ERROR(
        loom_llvmir_set_phi_incoming(phi->target_block, phi->phi_value_id,
                                     phi->incoming, phi->incoming_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_lowering_lower_body(
    loom_llvmir_lowering_state_t* state,
    loom_llvmir_function_t* target_function, loom_func_like_t func) {
  loom_region_t* body = loom_func_like_body(func);
  if (!body) return iree_ok_status();

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
    if (iree_any_bit_set(body->flags, LOOM_REGION_INSTANCE_FLAG_CFG)) {
      status = loom_llvmir_lowering_lower_cfg_body(state, target_function, body,
                                                   func.op, &fact_arena);
    } else if (body->block_count != 1) {
      status = loom_llvmir_lowering_unsupported_op(
          state, func.op,
          "multi-block function body must use CFG successor terminators before "
          "LLVMIR lowering");
    } else {
      loom_block_t* source_block = loom_region_entry_block(body);
      loom_llvmir_block_t* target_block = NULL;
      status = loom_llvmir_function_add_block(
          target_function,
          loom_llvmir_lowering_block_name(state, source_block,
                                          /*is_entry_block=*/true),
          &target_block);
      if (iree_status_is_ok(status)) {
        status = loom_llvmir_lowering_lower_source_block(state, target_block,
                                                         source_block);
      }
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
    if (loom_symbol_bytecode_kind(symbol) != LOOM_SYMBOL_FUNC_DEF) continue;
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
  state->providers = options->providers;
  state->provider_count = options->provider_count;
  state->value_map_count = source_module->values.count;
  state->symbol_function_count = source_module->symbols.count;
  state->kernel_attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID;
  state->nontemporal_metadata_id = LOOM_LLVMIR_METADATA_ID_INVALID;
  if (state->provider_count > 0 && state->providers == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVMIR lowering providers are required");
  }
  for (iree_host_size_t i = 0; i < state->provider_count; ++i) {
    if (state->providers[i] == NULL ||
        state->providers[i]->try_lower_op == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVMIR lowering provider is invalid");
    }
  }

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
