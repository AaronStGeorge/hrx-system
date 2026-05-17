// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/low_verify.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/spirv/abi.h"
#include "loom/target/arch/spirv/descriptors.h"
#include "loom/target/arch/spirv/error_catalog.h"
#include "loom/target/arch/spirv/packet_rows.h"
#include "loom/target/registers.h"

typedef struct loom_spirv_low_value_type_entry_t {
  // Value ID stored in this hash slot, or LOOM_VALUE_ID_INVALID when empty.
  loom_value_id_t value_id;
  // Exact SPIR-V value type attached to value_id.
  loom_spirv_value_type_t value_type;
} loom_spirv_low_value_type_entry_t;

typedef struct loom_spirv_low_value_type_table_t {
  // Open-addressed value-id table.
  loom_spirv_low_value_type_entry_t* entries;
  // Power-of-two entry capacity.
  iree_host_size_t capacity;
  // Number of occupied entries.
  iree_host_size_t count;
} loom_spirv_low_value_type_table_t;

typedef struct loom_spirv_low_verify_state_t {
  // Module being verified.
  const loom_module_t* module;
  // Function op being verified.
  const loom_op_t* function_op;
  // Function body being verified.
  const loom_region_t* body;
  // Entry block for single-block emission.
  const loom_block_t* entry_block;
  // Resolved low target for this function.
  const loom_low_resolved_target_t* target;
  // Scratch arena reset after the function.
  iree_arena_allocator_t* arena;
  // Borrowed function symbol name for diagnostics.
  iree_string_view_t function_name;
  // Exact SPIR-V value types indexed by compact hash table.
  loom_spirv_low_value_type_table_t value_types;
  // Exact ABI result value types.
  loom_spirv_value_type_t* result_value_types;
  // Number of entries in result_value_types.
  iree_host_size_t result_value_type_count;
  // True when the resolved function target uses HAL raw-BDA resources.
  bool raw_bda_hal_kernel;
  // True after a function-level diagnostic makes body-local checks unreliable.
  bool skip_body_checks;
} loom_spirv_low_verify_state_t;

static uint64_t loom_spirv_low_hash_value_id(loom_value_id_t value_id) {
  uint64_t value = value_id;
  value ^= value >> 33;
  value *= UINT64_C(0xff51afd7ed558ccd);
  value ^= value >> 33;
  return value;
}

static iree_host_size_t loom_spirv_low_next_power_of_two(
    iree_host_size_t value) {
  iree_host_size_t power = 1;
  while (power < value) {
    power <<= 1;
  }
  return power;
}

static void loom_spirv_low_initialize_value_type_entries(
    loom_spirv_low_value_type_entry_t* entries, iree_host_size_t count) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    entries[i].value_id = LOOM_VALUE_ID_INVALID;
  }
}

static iree_status_t loom_spirv_low_value_type_table_reserve(
    loom_spirv_low_value_type_table_t* table, iree_host_size_t minimum_capacity,
    iree_arena_allocator_t* arena) {
  if (table->capacity >= minimum_capacity) {
    return iree_ok_status();
  }

  const iree_host_size_t old_capacity = table->capacity;
  loom_spirv_low_value_type_entry_t* old_entries = table->entries;
  const iree_host_size_t new_capacity =
      loom_spirv_low_next_power_of_two(iree_max(minimum_capacity, 16));
  loom_spirv_low_value_type_entry_t* new_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, new_capacity, sizeof(*new_entries), (void**)&new_entries));
  loom_spirv_low_initialize_value_type_entries(new_entries, new_capacity);

  table->entries = new_entries;
  table->capacity = new_capacity;
  table->count = 0;
  for (iree_host_size_t i = 0; i < old_capacity; ++i) {
    const loom_spirv_low_value_type_entry_t* old_entry = &old_entries[i];
    if (old_entry->value_id == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    uint64_t cursor = loom_spirv_low_hash_value_id(old_entry->value_id);
    for (;;) {
      loom_spirv_low_value_type_entry_t* new_entry =
          &table->entries[cursor & (table->capacity - 1)];
      if (new_entry->value_id == LOOM_VALUE_ID_INVALID) {
        *new_entry = *old_entry;
        ++table->count;
        break;
      }
      ++cursor;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_low_value_type_table_define(
    loom_spirv_low_value_type_table_t* table, loom_value_id_t value_id,
    loom_spirv_value_type_t value_type, iree_arena_allocator_t* arena) {
  if ((table->count + 1) * 4 >= table->capacity * 3) {
    IREE_RETURN_IF_ERROR(loom_spirv_low_value_type_table_reserve(
        table, table->capacity == 0 ? 16 : table->capacity * 2, arena));
  }
  uint64_t cursor = loom_spirv_low_hash_value_id(value_id);
  for (;;) {
    loom_spirv_low_value_type_entry_t* entry =
        &table->entries[cursor & (table->capacity - 1)];
    if (entry->value_id == LOOM_VALUE_ID_INVALID) {
      entry->value_id = value_id;
      entry->value_type = value_type;
      ++table->count;
      return iree_ok_status();
    }
    if (entry->value_id == value_id) {
      entry->value_type = value_type;
      return iree_ok_status();
    }
    ++cursor;
  }
}

static bool loom_spirv_low_value_type_table_lookup(
    const loom_spirv_low_value_type_table_t* table, loom_value_id_t value_id,
    loom_spirv_value_type_t* out_value_type) {
  if (table->capacity == 0) {
    *out_value_type = (loom_spirv_value_type_t){0};
    return false;
  }
  uint64_t cursor = loom_spirv_low_hash_value_id(value_id);
  for (;;) {
    const loom_spirv_low_value_type_entry_t* entry =
        &table->entries[cursor & (table->capacity - 1)];
    if (entry->value_id == LOOM_VALUE_ID_INVALID) {
      *out_value_type = (loom_spirv_value_type_t){0};
      return false;
    }
    if (entry->value_id == value_id) {
      *out_value_type = entry->value_type;
      return true;
    }
    ++cursor;
  }
}

static bool loom_spirv_low_type_is_named_opaque(const loom_module_t* module,
                                                loom_type_t type,
                                                iree_string_view_t name) {
  if (!loom_type_is_dialect(type) || loom_type_dialect_param_count(type) != 0) {
    return false;
  }
  const loom_string_id_t name_id = loom_type_dialect_name_id(type);
  return name_id < module->strings.count &&
         iree_string_view_equal(module->strings.entries[name_id], name);
}

static loom_type_t loom_spirv_low_module_type_attr(const loom_module_t* module,
                                                   loom_type_id_t type_id) {
  if (type_id >= module->types.count) {
    return loom_type_none();
  }
  return module->types.entries[type_id];
}

static bool loom_spirv_low_type_is_spirv_id(loom_type_t type) {
  return loom_low_type_is_register(type) &&
         loom_low_register_type_descriptor_set_stable_id(type) ==
             SPIRV_LOGICAL_CORE_DESCRIPTOR_SET_ID &&
         loom_low_register_type_class_id(type) ==
             SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID;
}

static bool loom_spirv_low_register_value_type(
    loom_type_t type, loom_spirv_value_type_t* out_value_type) {
  *out_value_type = (loom_spirv_value_type_t){0};
  if (!loom_low_type_is_register(type) ||
      loom_low_register_type_descriptor_set_stable_id(type) !=
          SPIRV_LOGICAL_CORE_DESCRIPTOR_SET_ID) {
    return false;
  }
  switch (loom_low_register_type_class_id(type)) {
    case SPIRV_LOGICAL_CORE_REG_CLASS_ID_OFFSET64:
      *out_value_type = (loom_spirv_value_type_t){
          .value_class = LOOM_SPIRV_VALUE_CLASS_OFFSET64,
          .scalar_type = LOOM_SPIRV_SCALAR_TYPE_U64,
      };
      return true;
    case SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_STORAGE_BUFFER:
      *out_value_type = (loom_spirv_value_type_t){
          .value_class = LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS,
      };
      return true;
    case SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID:
      return false;
  }
  return false;
}

static iree_string_view_t loom_spirv_low_value_type_format(
    loom_spirv_low_verify_state_t* state, loom_spirv_value_type_t value_type) {
  (void)state;
  switch (value_type.value_class) {
    case LOOM_SPIRV_VALUE_CLASS_UNKNOWN:
      return IREE_SV("unknown");
    case LOOM_SPIRV_VALUE_CLASS_BOOL:
      return IREE_SV("bool");
    case LOOM_SPIRV_VALUE_CLASS_OFFSET64:
      return IREE_SV("offset64");
    case LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS:
      return IREE_SV("storage_buffer_address");
    case LOOM_SPIRV_VALUE_CLASS_SCALAR:
      return loom_spirv_scalar_type_name(value_type.scalar_type);
    case LOOM_SPIRV_VALUE_CLASS_PTR_PHYSICAL_STORAGE_BUFFER:
      return IREE_SV("ptr.physical_storage_buffer");
    case LOOM_SPIRV_VALUE_CLASS_COOPERATIVE_MATRIX:
      return IREE_SV("cooperative_matrix");
  }
  return IREE_SV("unknown");
}

static const loom_named_attr_t* loom_spirv_low_find_boundary_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t attr_name) {
  const loom_string_id_t name_id = loom_module_lookup_string(module, attr_name);
  if (name_id == LOOM_STRING_ID_INVALID) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    if (attrs.entries[i].name_id == name_id) {
      return &attrs.entries[i];
    }
  }
  return NULL;
}

static loom_named_attr_slice_t loom_spirv_low_boundary_attrs(
    const loom_op_t* function_op) {
  if (loom_low_kernel_def_isa(function_op)) {
    return loom_low_kernel_def_abi_layout(function_op);
  }
  if (loom_low_func_def_isa(function_op)) {
    return loom_low_func_def_abi_layout(function_op);
  }
  return loom_named_attr_slice_empty();
}

static iree_status_t loom_spirv_low_emit(loom_low_verify_context_t* context,
                                         const loom_op_t* op,
                                         const loom_error_def_t* error,
                                         const loom_diagnostic_param_t* params,
                                         iree_host_size_t param_count) {
  return loom_low_verify_context_emit(context, op, error, params, param_count);
}

static iree_status_t loom_spirv_low_emit_malformed_abi_attr(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state,
    const loom_op_t* op, iree_string_view_t attr_name,
    iree_host_size_t expected_count) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_string(attr_name),
      loom_param_u32((uint32_t)expected_count),
  };
  return loom_spirv_low_emit(context, op, LOOM_ERR_SPIRV_001, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_spirv_low_lookup_abi_value_type_attr(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state,
    iree_string_view_t attr_name, iree_host_size_t expected_count,
    const loom_attribute_t** out_attr, bool* out_malformed) {
  *out_attr = NULL;
  *out_malformed = false;
  const loom_named_attr_t* entry = loom_spirv_low_find_boundary_attr(
      state->module, loom_spirv_low_boundary_attrs(state->function_op),
      attr_name);
  if (entry == NULL) {
    return iree_ok_status();
  }
  const loom_attribute_t* attr = &entry->value;
  if (attr->kind != LOOM_ATTR_I64_ARRAY || attr->count != expected_count ||
      (attr->count != 0 && attr->i64_array == NULL)) {
    IREE_RETURN_IF_ERROR(loom_spirv_low_emit_malformed_abi_attr(
        context, state, state->function_op, attr_name, expected_count));
    *out_malformed = true;
    return iree_ok_status();
  }
  *out_attr = attr;
  return iree_ok_status();
}

static iree_status_t loom_spirv_low_emit_missing_abi_value_type(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state,
    const loom_op_t* op, iree_string_view_t attr_name,
    loom_value_id_t value_id) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_string(attr_name),
      loom_param_string(
          loom_low_diagnostic_value_name(state->module, value_id)),
  };
  return loom_spirv_low_emit(context, op, LOOM_ERR_SPIRV_002, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_spirv_low_emit_non_id_abi_value_type(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state,
    const loom_op_t* op, iree_string_view_t attr_name, loom_value_id_t value_id,
    int64_t value_code) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_string(attr_name),
      loom_param_string(
          loom_low_diagnostic_value_name(state->module, value_id)),
      loom_param_i64(value_code),
  };
  return loom_spirv_low_emit(context, op, LOOM_ERR_SPIRV_003, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_spirv_low_emit_invalid_abi_value_type(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state,
    const loom_op_t* op, iree_string_view_t attr_name, loom_value_id_t value_id,
    int64_t value_code) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_string(attr_name),
      loom_param_string(
          loom_low_diagnostic_value_name(state->module, value_id)),
      loom_param_i64(value_code),
  };
  return loom_spirv_low_emit(context, op, LOOM_ERR_SPIRV_004, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_spirv_low_emit_unsupported_register_type(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state,
    const loom_op_t* op, loom_value_id_t value_id, loom_type_t actual_type) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_string(
          loom_low_diagnostic_value_name(state->module, value_id)),
      loom_param_type(actual_type),
  };
  return loom_spirv_low_emit(context, op, LOOM_ERR_SPIRV_005, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_spirv_low_prepare_abi_value_type(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state,
    const loom_op_t* op, iree_string_view_t attr_name,
    const loom_attribute_t* attr, iree_host_size_t attr_index,
    loom_value_id_t value_id, loom_spirv_value_type_t* out_value_type) {
  loom_type_t low_type = loom_module_value_type(state->module, value_id);
  *out_value_type = (loom_spirv_value_type_t){0};
  if (!loom_spirv_low_type_is_spirv_id(low_type)) {
    if (attr != NULL && attr->i64_array[attr_index] != 0) {
      IREE_RETURN_IF_ERROR(loom_spirv_low_emit_non_id_abi_value_type(
          context, state, op, attr_name, value_id,
          attr->i64_array[attr_index]));
      state->skip_body_checks = true;
      return iree_ok_status();
    }
    if (!loom_spirv_low_register_value_type(low_type, out_value_type)) {
      IREE_RETURN_IF_ERROR(loom_spirv_low_emit_unsupported_register_type(
          context, state, op, value_id, low_type));
      state->skip_body_checks = true;
    }
    return iree_ok_status();
  }
  if (attr == NULL) {
    IREE_RETURN_IF_ERROR(loom_spirv_low_emit_missing_abi_value_type(
        context, state, op, attr_name, value_id));
    state->skip_body_checks = true;
    return iree_ok_status();
  }
  if (!loom_spirv_abi_value_type_decode(attr->i64_array[attr_index],
                                        out_value_type) ||
      out_value_type->value_class == LOOM_SPIRV_VALUE_CLASS_UNKNOWN) {
    IREE_RETURN_IF_ERROR(loom_spirv_low_emit_invalid_abi_value_type(
        context, state, op, attr_name, value_id, attr->i64_array[attr_index]));
    state->skip_body_checks = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_low_seed_entry_args(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state) {
  if (state->entry_block == NULL) {
    return iree_ok_status();
  }
  const loom_attribute_t* attr = NULL;
  bool malformed = false;
  IREE_RETURN_IF_ERROR(loom_spirv_low_lookup_abi_value_type_attr(
      context, state, IREE_SV(LOOM_SPIRV_ABI_ARG_VALUE_TYPES_ATTR_NAME),
      state->entry_block->arg_count, &attr, &malformed));
  if (malformed) {
    state->skip_body_checks = true;
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < state->entry_block->arg_count; ++i) {
    const loom_value_id_t value_id = loom_block_arg_id(state->entry_block, i);
    loom_spirv_value_type_t value_type = {0};
    IREE_RETURN_IF_ERROR(loom_spirv_low_prepare_abi_value_type(
        context, state, state->function_op,
        IREE_SV(LOOM_SPIRV_ABI_ARG_VALUE_TYPES_ATTR_NAME), attr, i, value_id,
        &value_type));
    if (value_type.value_class != LOOM_SPIRV_VALUE_CLASS_UNKNOWN) {
      IREE_RETURN_IF_ERROR(loom_spirv_low_value_type_table_define(
          &state->value_types, value_id, value_type, state->arena));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_low_prepare_result_value_types(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state) {
  const iree_host_size_t result_count = state->function_op->result_count;
  if (result_count == 0) {
    return iree_ok_status();
  }
  const loom_attribute_t* attr = NULL;
  bool malformed = false;
  IREE_RETURN_IF_ERROR(loom_spirv_low_lookup_abi_value_type_attr(
      context, state, IREE_SV(LOOM_SPIRV_ABI_RESULT_VALUE_TYPES_ATTR_NAME),
      result_count, &attr, &malformed));
  if (malformed) {
    state->skip_body_checks = true;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, result_count, sizeof(*state->result_value_types),
      (void**)&state->result_value_types));
  state->result_value_type_count = result_count;
  const loom_value_id_t* results = loom_op_const_results(state->function_op);
  for (iree_host_size_t i = 0; i < result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_spirv_low_prepare_abi_value_type(
        context, state, state->function_op,
        IREE_SV(LOOM_SPIRV_ABI_RESULT_VALUE_TYPES_ATTR_NAME), attr, i,
        results[i], &state->result_value_types[i]));
  }
  return iree_ok_status();
}

static bool loom_spirv_low_value_type_is_direct_scalar(
    loom_spirv_value_type_t value_type) {
  return value_type.value_class == LOOM_SPIRV_VALUE_CLASS_SCALAR ||
         value_type.value_class == LOOM_SPIRV_VALUE_CLASS_BOOL ||
         value_type.value_class == LOOM_SPIRV_VALUE_CLASS_OFFSET64;
}

static iree_status_t loom_spirv_low_emit_raw_bda_returns(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
  };
  return loom_spirv_low_emit(context, state->function_op, LOOM_ERR_SPIRV_006,
                             params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_spirv_low_emit_raw_bda_direct_value(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state,
    loom_value_id_t value_id, loom_spirv_value_type_t value_type) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_string(
          loom_low_diagnostic_value_name(state->module, value_id)),
      loom_param_string(loom_spirv_low_value_type_format(state, value_type)),
  };
  return loom_spirv_low_emit(context, state->function_op, LOOM_ERR_SPIRV_007,
                             params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_spirv_low_emit_shader_result_value(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state,
    loom_value_id_t value_id, loom_spirv_value_type_t value_type) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_string(
          loom_low_diagnostic_value_name(state->module, value_id)),
      loom_param_string(loom_spirv_low_value_type_format(state, value_type)),
  };
  return loom_spirv_low_emit(context, state->function_op, LOOM_ERR_SPIRV_016,
                             params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_spirv_low_verify_abi_shape(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state) {
  if (state->raw_bda_hal_kernel) {
    if (state->function_op->result_count != 0) {
      IREE_RETURN_IF_ERROR(loom_spirv_low_emit_raw_bda_returns(context, state));
    }
    if (state->entry_block != NULL) {
      for (uint16_t i = 0; i < state->entry_block->arg_count; ++i) {
        const loom_value_id_t value_id =
            loom_block_arg_id(state->entry_block, i);
        loom_spirv_value_type_t value_type = {0};
        if (!loom_spirv_low_value_type_table_lookup(&state->value_types,
                                                    value_id, &value_type)) {
          continue;
        }
        if (!loom_spirv_low_value_type_is_direct_scalar(value_type)) {
          IREE_RETURN_IF_ERROR(loom_spirv_low_emit_raw_bda_direct_value(
              context, state, value_id, value_type));
        }
      }
    }
    return iree_ok_status();
  }

  const loom_value_id_t* results = loom_op_const_results(state->function_op);
  for (iree_host_size_t i = 0; i < state->result_value_type_count; ++i) {
    if (!loom_spirv_low_value_type_is_direct_scalar(
            state->result_value_types[i])) {
      IREE_RETURN_IF_ERROR(loom_spirv_low_emit_shader_result_value(
          context, state, results[i], state->result_value_types[i]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_low_emit_multiblock(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_u32(state->body ? state->body->block_count : 0),
  };
  return loom_spirv_low_emit(context, state->function_op, LOOM_ERR_SPIRV_015,
                             params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_spirv_low_emit_resource_import_kind(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state,
    const loom_op_t* op) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_u32(loom_low_resource_import_kind(op)),
  };
  return loom_spirv_low_emit(context, op, LOOM_ERR_SPIRV_008, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_spirv_low_emit_resource_abi(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state,
    const loom_op_t* op) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
  };
  return loom_spirv_low_emit(context, op, LOOM_ERR_SPIRV_018, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_spirv_low_emit_resource_binding_range(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state,
    const loom_op_t* op) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_i64(loom_low_resource_index(op)),
  };
  return loom_spirv_low_emit(context, op, LOOM_ERR_SPIRV_009, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_spirv_low_emit_resource_source_type(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state,
    const loom_op_t* op, loom_type_t actual_type) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_type(actual_type),
  };
  return loom_spirv_low_emit(context, op, LOOM_ERR_SPIRV_010, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_spirv_low_emit_resource_result_type(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state,
    const loom_op_t* op, loom_value_id_t value_id,
    loom_spirv_value_type_t value_type) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_string(
          loom_low_diagnostic_value_name(state->module, value_id)),
      loom_param_string(loom_spirv_low_value_type_format(state, value_type)),
  };
  return loom_spirv_low_emit(context, op, LOOM_ERR_SPIRV_011, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_spirv_low_verify_resource(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state,
    const loom_op_t* op) {
  if (!state->raw_bda_hal_kernel) {
    return loom_spirv_low_emit_resource_abi(context, state, op);
  }
  if (loom_low_resource_import_kind(op) !=
      LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BINDING) {
    IREE_RETURN_IF_ERROR(
        loom_spirv_low_emit_resource_import_kind(context, state, op));
  }
  const int64_t index = loom_low_resource_index(op);
  if (index < 0 || index >= UINT16_MAX) {
    IREE_RETURN_IF_ERROR(
        loom_spirv_low_emit_resource_binding_range(context, state, op));
  }

  const loom_type_t source_type = loom_spirv_low_module_type_attr(
      state->module, loom_low_resource_source_type(op));
  if (!loom_spirv_low_type_is_named_opaque(state->module, source_type,
                                           IREE_SV("hal.buffer"))) {
    IREE_RETURN_IF_ERROR(loom_spirv_low_emit_resource_source_type(
        context, state, op, source_type));
  }

  const loom_value_id_t result = loom_low_resource_result(op);
  loom_spirv_value_type_t result_type = {0};
  if (!loom_spirv_low_register_value_type(
          loom_module_value_type(state->module, result), &result_type)) {
    IREE_RETURN_IF_ERROR(loom_spirv_low_emit_unsupported_register_type(
        context, state, op, result,
        loom_module_value_type(state->module, result)));
    return iree_ok_status();
  }
  if (result_type.value_class !=
      LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS) {
    IREE_RETURN_IF_ERROR(loom_spirv_low_emit_resource_result_type(
        context, state, op, result, result_type));
  }
  return loom_spirv_low_value_type_table_define(&state->value_types, result,
                                                result_type, state->arena);
}

static iree_status_t loom_spirv_low_emit_missing_packet_row(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_string(packet->key),
  };
  return loom_spirv_low_emit(context, packet->op, LOOM_ERR_SPIRV_012, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_spirv_low_emit_packet_type_mismatch(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    iree_string_view_t field_kind, uint32_t field_index,
    loom_value_id_t value_id, loom_spirv_value_type_t expected_value_type,
    loom_spirv_value_type_t actual_value_type) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_string(packet->key),
      loom_param_string(field_kind),
      loom_param_u32(field_index),
      loom_param_string(
          loom_low_diagnostic_value_name(state->module, value_id)),
      loom_param_string(
          loom_spirv_low_value_type_format(state, expected_value_type)),
      loom_param_string(
          loom_spirv_low_value_type_format(state, actual_value_type)),
  };
  return loom_spirv_low_emit(context, packet->op, LOOM_ERR_SPIRV_013, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_spirv_low_verify_packet_operands(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  const loom_value_id_t* operands = loom_op_const_operands(packet->op);
  for (uint8_t i = 0; i < row->operand_count; ++i) {
    loom_spirv_value_type_t actual_type = {0};
    if (!loom_spirv_low_value_type_table_lookup(&state->value_types,
                                                operands[i], &actual_type)) {
      continue;
    }
    if (!loom_spirv_value_type_equal(actual_type, row->operand_types[i])) {
      IREE_RETURN_IF_ERROR(loom_spirv_low_emit_packet_type_mismatch(
          context, state, packet, IREE_SV("operand"), i, operands[i],
          row->operand_types[i], actual_type));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_low_define_packet_results(
    loom_spirv_low_verify_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  if (row->result_count == 0) {
    return iree_ok_status();
  }
  const loom_value_id_t* results = loom_op_const_results(packet->op);
  return loom_spirv_low_value_type_table_define(&state->value_types, results[0],
                                                row->result_type, state->arena);
}

static iree_status_t loom_spirv_low_verify_packet(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet) {
  if (packet->descriptor == NULL) {
    return iree_ok_status();
  }
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_descriptor_ordinal(state->target->descriptor_set,
                                                 packet->descriptor);
  const loom_spirv_packet_row_t* row =
      loom_spirv_packet_row_for_descriptor_ordinal(descriptor_ordinal);
  if (row == NULL) {
    return loom_spirv_low_emit_missing_packet_row(context, state, packet);
  }
  if (packet->op->operand_count != row->operand_count ||
      packet->op->result_count != row->result_count) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_spirv_low_verify_packet_operands(context, state, packet, row));
  return loom_spirv_low_define_packet_results(state, packet, row);
}

static iree_status_t loom_spirv_low_verify_copy(
    loom_spirv_low_verify_state_t* state, const loom_op_t* op) {
  loom_spirv_value_type_t source_type = {0};
  if (!loom_spirv_low_value_type_table_lookup(
          &state->value_types, loom_low_copy_source(op), &source_type)) {
    return iree_ok_status();
  }
  return loom_spirv_low_value_type_table_define(
      &state->value_types, loom_low_copy_result(op), source_type, state->arena);
}

static iree_status_t loom_spirv_low_emit_return_type_mismatch(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state,
    const loom_op_t* op, uint32_t result_index, loom_value_id_t value_id,
    loom_spirv_value_type_t expected_value_type,
    loom_spirv_value_type_t actual_value_type) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_u32(result_index),
      loom_param_string(
          loom_low_diagnostic_value_name(state->module, value_id)),
      loom_param_string(
          loom_spirv_low_value_type_format(state, expected_value_type)),
      loom_param_string(
          loom_spirv_low_value_type_format(state, actual_value_type)),
  };
  return loom_spirv_low_emit(context, op, LOOM_ERR_SPIRV_014, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_spirv_low_verify_return(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state,
    const loom_op_t* op) {
  if (op->operand_count != state->result_value_type_count) {
    return iree_ok_status();
  }
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    loom_spirv_value_type_t actual_type = {0};
    if (!loom_spirv_low_value_type_table_lookup(&state->value_types,
                                                operands[i], &actual_type)) {
      continue;
    }
    if (!loom_spirv_value_type_equal(actual_type,
                                     state->result_value_types[i])) {
      IREE_RETURN_IF_ERROR(loom_spirv_low_emit_return_type_mismatch(
          context, state, op, i, operands[i], state->result_value_types[i],
          actual_type));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_low_emit_unsupported_structural_op(
    loom_low_verify_context_t* context, loom_spirv_low_verify_state_t* state,
    const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(state->module, op);
  const iree_string_view_t op_name =
      vtable != NULL ? loom_op_vtable_name(vtable) : IREE_SV("<unknown>");
  loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_string(op_name),
  };
  return loom_spirv_low_emit(context, op, LOOM_ERR_SPIRV_017, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_spirv_low_begin_function(
    const loom_low_verify_provider_t* provider,
    loom_low_verify_context_t* context, void** out_provider_state) {
  (void)provider;
  *out_provider_state = NULL;
  const loom_low_resolved_target_t* target =
      loom_low_verify_context_target(context);
  if (target->descriptor_set == NULL ||
      target->descriptor_set->stable_id !=
          SPIRV_LOGICAL_CORE_DESCRIPTOR_SET_ID) {
    return iree_ok_status();
  }

  loom_spirv_low_verify_state_t* state = NULL;
  iree_arena_allocator_t* arena = loom_low_verify_context_arena(context);
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*state), (void**)&state));
  *state = (loom_spirv_low_verify_state_t){
      .module = loom_low_verify_context_module(context),
      .function_op = loom_low_verify_context_function_op(context),
      .body = loom_low_verify_context_function_body(context),
      .target = target,
      .arena = arena,
      .function_name = loom_low_diagnostic_function_name(
          loom_low_verify_context_module(context),
          loom_low_verify_context_function_op(context)),
      .raw_bda_hal_kernel = target->bundle_storage.export_plan.abi_kind ==
                            LOOM_TARGET_ABI_HAL_KERNEL,
  };
  *out_provider_state = state;

  if (state->body != NULL && state->body->block_count != 1) {
    IREE_RETURN_IF_ERROR(loom_spirv_low_emit_multiblock(context, state));
    state->skip_body_checks = true;
  }
  if (state->body != NULL && state->body->block_count > 0) {
    state->entry_block = loom_region_const_entry_block(state->body);
  }
  IREE_RETURN_IF_ERROR(loom_spirv_low_seed_entry_args(context, state));
  IREE_RETURN_IF_ERROR(
      loom_spirv_low_prepare_result_value_types(context, state));
  if (state->skip_body_checks) {
    return iree_ok_status();
  }
  return loom_spirv_low_verify_abi_shape(context, state);
}

static iree_status_t loom_spirv_low_verify_op(
    const loom_low_verify_provider_t* provider,
    loom_low_verify_context_t* context, void* provider_state,
    const loom_low_resolved_descriptor_packet_t* packet) {
  (void)provider;
  loom_spirv_low_verify_state_t* state =
      (loom_spirv_low_verify_state_t*)provider_state;
  if (state == NULL || state->skip_body_checks ||
      loom_low_verify_context_should_stop(context)) {
    return iree_ok_status();
  }
  const loom_op_t* op = packet->op;
  if (loom_low_return_isa(op)) {
    return loom_spirv_low_verify_return(context, state, op);
  }
  if (loom_low_resource_isa(op)) {
    return loom_spirv_low_verify_resource(context, state, op);
  }
  if (loom_low_copy_isa(op)) {
    return loom_spirv_low_verify_copy(state, op);
  }
  if (packet->kind != LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
    return loom_spirv_low_verify_packet(context, state, packet);
  }
  return loom_spirv_low_emit_unsupported_structural_op(context, state, op);
}

const loom_low_verify_provider_t loom_spirv_low_verify_provider = {
    .name = IREE_SVL("spirv"),
    .begin_function = loom_spirv_low_begin_function,
    .verify_op = loom_spirv_low_verify_op,
};

static const loom_low_verify_provider_t* const kLoomSpirvLowVerifyProviders[] =
    {
        &loom_spirv_low_verify_provider,
};

loom_low_verify_provider_list_t loom_spirv_low_verify_provider_list(void) {
  return loom_low_verify_provider_list_make(
      kLoomSpirvLowVerifyProviders,
      IREE_ARRAYSIZE(kLoomSpirvLowVerifyProviders));
}
