// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/module_emitter.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/function.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/context.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/spirv/descriptors.h"
#include "loom/target/emit/spirv/binary_format.h"
#include "loom/target/emit/spirv/packet_rows.h"
#include "loom/target/registers.h"
#include "loom/target/types.h"

typedef enum loom_spirv_type_key_kind_e {
  LOOM_SPIRV_TYPE_KEY_VOID = 1,
  LOOM_SPIRV_TYPE_KEY_INT = 2,
  LOOM_SPIRV_TYPE_KEY_POINTER = 3,
  LOOM_SPIRV_TYPE_KEY_DESCRIPTOR_STRUCT = 4,
  LOOM_SPIRV_TYPE_KEY_FUNCTION = 5,
} loom_spirv_type_key_kind_t;

enum {
  LOOM_SPIRV_TYPE_CACHE_INITIAL_CAPACITY = 16,
  LOOM_SPIRV_TYPE_KEY_MAX_OPERAND_COUNT = 8,
  LOOM_SPIRV_INTEGER_CONSTANT_CACHE_INITIAL_CAPACITY = 16,
};

typedef struct loom_spirv_type_key_t {
  // SPIR-V type declaration family.
  loom_spirv_type_key_kind_t kind;
  // Pointer ArrayStride decoration, or zero when the type is undecorated.
  uint32_t pointer_array_stride;
  // Number of entries in |operands|.
  uint8_t operand_count;
  // Kind-specific SPIR-V type operands following the result ID.
  uint32_t operands[LOOM_SPIRV_TYPE_KEY_MAX_OPERAND_COUNT];
} loom_spirv_type_key_t;

typedef struct loom_spirv_type_cache_entry_t {
  // Numeric type key.
  loom_spirv_type_key_t key;
  // SPIR-V result ID assigned to |key|.
  uint32_t type_id;
} loom_spirv_type_cache_entry_t;

typedef struct loom_spirv_integer_constant_key_t {
  // SPIR-V integer type ID of the constant.
  uint32_t type_id;
  // Number of literal words in |words|.
  uint8_t word_count;
  // Little-endian integer literal words following the result ID.
  uint32_t words[2];
} loom_spirv_integer_constant_key_t;

typedef struct loom_spirv_integer_constant_cache_entry_t {
  // Numeric integer constant key.
  loom_spirv_integer_constant_key_t key;
  // SPIR-V result ID assigned to |key|.
  uint32_t id;
} loom_spirv_integer_constant_cache_entry_t;

typedef struct loom_spirv_abi_slot_type_info_t {
  // SPIR-V type ID used by packet emission after ABI materialization.
  uint32_t value_type_id;
  // SPIR-V type ID stored in the descriptor-backed slot field.
  uint32_t field_type_id;
  // StorageBuffer pointer-to-descriptor-block type ID for OpVariable.
  uint32_t variable_type_id;
  // StorageBuffer pointer-to-field type ID for OpAccessChain results.
  uint32_t field_pointer_type_id;
  // Natural byte alignment attached to loads/stores for this slot field.
  uint32_t field_alignment;
} loom_spirv_abi_slot_type_info_t;

typedef struct loom_spirv_abi_slot_t {
  // Low function value materialized by this ABI slot.
  loom_value_id_t value_id;
  // SPIR-V type category materialized for |value_id|.
  loom_spirv_value_kind_t type_kind;
  // Descriptor binding assigned to this shader descriptor slot.
  uint32_t binding;
  // Module-scope StorageBuffer variable ID for this descriptor slot.
  uint32_t variable_id;
} loom_spirv_abi_slot_t;

typedef struct loom_spirv_abi_plan_t {
  // Materialization slots for low function entry block arguments.
  loom_spirv_abi_slot_t* args;
  // Number of entries in |args|.
  iree_host_size_t arg_count;
  // Materialization slots for low function results.
  loom_spirv_abi_slot_t* results;
  // Number of entries in |results|.
  iree_host_size_t result_count;
} loom_spirv_abi_plan_t;

typedef struct loom_spirv_value_ref_t {
  // SPIR-V result ID carrying this low SSA value.
  uint32_t id;
  // SPIR-V type ID assigned to |id|.
  uint32_t type_id;
  // Target-local type category used by packet emitters.
  loom_spirv_value_kind_t type_kind;
} loom_spirv_value_ref_t;

typedef struct loom_spirv_emit_state_t {
  // Module containing the emitted low function.
  loom_module_t* module;
  // Target-low function definition being emitted.
  loom_op_t* function_op;
  // Resolved target record and descriptor set for |function_op|.
  const loom_low_resolved_target_t* target;
  // Case/function scratch arena.
  iree_arena_allocator_t* scratch_arena;
  // Sectioned SPIR-V module builder.
  loom_spirv_module_builder_t* builder;
  // Function-local value domain for dense value tables.
  loom_local_value_domain_t value_domain;
  // SPIR-V value refs indexed by function-local value ordinal.
  loom_spirv_value_ref_t* value_refs;
  // Module string IDs for descriptor-set immediate rows.
  loom_string_id_t* immediate_name_ids;
  // Number of entries in |immediate_name_ids|.
  iree_host_size_t immediate_name_id_count;
  // Cached SPIR-V type declaration rows.
  loom_spirv_type_cache_entry_t* type_cache_entries;
  // Number of entries in |type_cache_entries|.
  iree_host_size_t type_cache_count;
  // Allocated capacity of |type_cache_entries|.
  iree_host_size_t type_cache_capacity;
  // Cached emitter-owned integer constants.
  loom_spirv_integer_constant_cache_entry_t* integer_constant_cache_entries;
  // Number of entries in |integer_constant_cache_entries|.
  iree_host_size_t integer_constant_cache_count;
  // Allocated capacity of |integer_constant_cache_entries|.
  iree_host_size_t integer_constant_cache_capacity;
  // SPIR-V ID assigned to the function.
  uint32_t function_id;
  // Shader-entry ABI plan for materialized resources.
  loom_spirv_abi_plan_t abi_plan;
} loom_spirv_emit_state_t;

static loom_spirv_binary_writer_t* loom_spirv_emit_section(
    loom_spirv_emit_state_t* state, loom_spirv_module_section_t section) {
  return loom_spirv_module_builder_section(state->builder, section);
}

static uint32_t loom_spirv_emit_allocate_id(loom_spirv_emit_state_t* state) {
  return loom_spirv_module_builder_allocate_id(state->builder);
}

static iree_string_view_t loom_spirv_emit_string_or_empty(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static iree_string_view_t loom_spirv_emit_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  return loom_spirv_emit_string_or_empty(module, symbol->name_id);
}

static iree_string_view_t loom_spirv_emit_function_name(
    const loom_spirv_emit_state_t* state) {
  return loom_spirv_emit_symbol_name(
      state->module, loom_low_function_callee(state->function_op));
}

static iree_string_view_t loom_spirv_emit_export_name(
    const loom_spirv_emit_state_t* state) {
  const iree_string_view_t export_symbol =
      state->target->bundle_storage.export_plan.export_symbol;
  if (!iree_string_view_is_empty(export_symbol)) {
    return export_symbol;
  }
  return loom_spirv_emit_function_name(state);
}

static iree_status_t loom_spirv_emit_op_name(loom_spirv_emit_state_t* state,
                                             uint32_t id,
                                             iree_string_view_t name) {
  if (id == 0 || iree_string_view_is_empty(name)) {
    return iree_ok_status();
  }
  const uint32_t prefix_operands[] = {id};
  return loom_spirv_binary_write_string_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DEBUG),
      LOOM_SPIRV_OP_NAME, prefix_operands, IREE_ARRAYSIZE(prefix_operands),
      name, NULL, 0);
}

static iree_status_t loom_spirv_emit_value_name(loom_spirv_emit_state_t* state,
                                                loom_value_id_t value_id,
                                                uint32_t id) {
  if (value_id >= state->module->values.count) {
    return iree_ok_status();
  }
  const loom_value_t* value = loom_module_value(state->module, value_id);
  return loom_spirv_emit_op_name(
      state, id,
      loom_spirv_emit_string_or_empty(state->module, value->name_id));
}

static iree_status_t loom_spirv_emit_prepare_immediate_name_ids(
    loom_spirv_emit_state_t* state) {
  const iree_host_size_t immediate_count =
      state->target->descriptor_set->immediate_count;
  if (immediate_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, immediate_count, sizeof(*state->immediate_name_ids),
      (void**)&state->immediate_name_ids));
  state->immediate_name_id_count = immediate_count;
  for (iree_host_size_t i = 0; i < immediate_count; ++i) {
    const loom_low_immediate_t* immediate =
        &state->target->descriptor_set->immediates[i];
    const iree_string_view_t name = loom_low_descriptor_set_string(
        state->target->descriptor_set, immediate->field_name_string_offset);
    state->immediate_name_ids[i] =
        loom_module_lookup_string(state->module, name);
  }
  return iree_ok_status();
}

static bool loom_spirv_type_key_equal(const loom_spirv_type_key_t* lhs,
                                      const loom_spirv_type_key_t* rhs) {
  if (lhs->kind != rhs->kind ||
      lhs->pointer_array_stride != rhs->pointer_array_stride ||
      lhs->operand_count != rhs->operand_count) {
    return false;
  }
  for (uint8_t i = 0; i < lhs->operand_count; ++i) {
    if (lhs->operands[i] != rhs->operands[i]) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_spirv_emit_type_cache_reserve(
    loom_spirv_emit_state_t* state, iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= state->type_cache_capacity) {
    return iree_ok_status();
  }
  if (state->type_cache_capacity == 0) {
    minimum_capacity =
        iree_max(minimum_capacity, LOOM_SPIRV_TYPE_CACHE_INITIAL_CAPACITY);
  }
  return iree_arena_grow_array(
      state->scratch_arena, state->type_cache_count, minimum_capacity,
      sizeof(*state->type_cache_entries), &state->type_cache_capacity,
      (void**)&state->type_cache_entries);
}

static iree_status_t loom_spirv_emit_type_declaration(
    loom_spirv_emit_state_t* state, const loom_spirv_type_key_t* key,
    uint32_t type_id) {
  switch (key->kind) {
    case LOOM_SPIRV_TYPE_KEY_VOID: {
      const uint32_t operands[] = {type_id};
      return loom_spirv_binary_write_instruction(
          loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
          LOOM_SPIRV_OP_TYPE_VOID, operands, IREE_ARRAYSIZE(operands));
    }
    case LOOM_SPIRV_TYPE_KEY_INT: {
      const uint32_t operands[] = {type_id, key->operands[0], key->operands[1]};
      return loom_spirv_binary_write_instruction(
          loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
          LOOM_SPIRV_OP_TYPE_INT, operands, IREE_ARRAYSIZE(operands));
    }
    case LOOM_SPIRV_TYPE_KEY_POINTER: {
      const uint32_t operands[] = {
          type_id,
          key->operands[0],
          key->operands[1],
      };
      return loom_spirv_binary_write_instruction(
          loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
          LOOM_SPIRV_OP_TYPE_POINTER, operands, IREE_ARRAYSIZE(operands));
    }
    case LOOM_SPIRV_TYPE_KEY_DESCRIPTOR_STRUCT: {
      const uint32_t operands[] = {type_id, key->operands[0]};
      return loom_spirv_binary_write_instruction(
          loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
          LOOM_SPIRV_OP_TYPE_STRUCT, operands, IREE_ARRAYSIZE(operands));
    }
    case LOOM_SPIRV_TYPE_KEY_FUNCTION: {
      uint32_t operands[1 + LOOM_SPIRV_TYPE_KEY_MAX_OPERAND_COUNT] = {0};
      operands[0] = type_id;
      for (uint8_t i = 0; i < key->operand_count; ++i) {
        operands[1 + i] = key->operands[i];
      }
      return loom_spirv_binary_write_instruction(
          loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
          LOOM_SPIRV_OP_TYPE_FUNCTION, operands, 1 + key->operand_count);
    }
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "unknown SPIR-V type key kind %u",
                          (uint32_t)key->kind);
}

static iree_status_t loom_spirv_emit_type_id(loom_spirv_emit_state_t* state,
                                             const loom_spirv_type_key_t* key,
                                             uint32_t* out_type_id,
                                             bool* out_emitted) {
  for (iree_host_size_t i = 0; i < state->type_cache_count; ++i) {
    const loom_spirv_type_cache_entry_t* entry = &state->type_cache_entries[i];
    if (loom_spirv_type_key_equal(key, &entry->key)) {
      *out_type_id = entry->type_id;
      if (out_emitted != NULL) {
        *out_emitted = false;
      }
      return iree_ok_status();
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_cache_reserve(state, state->type_cache_count + 1));

  const uint32_t type_id = loom_spirv_emit_allocate_id(state);
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_declaration(state, key, type_id));
  state->type_cache_entries[state->type_cache_count++] =
      (loom_spirv_type_cache_entry_t){
          .key = *key,
          .type_id = type_id,
      };
  *out_type_id = type_id;
  if (out_emitted != NULL) {
    *out_emitted = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_type_void(loom_spirv_emit_state_t* state,
                                               uint32_t* out_type_id) {
  const loom_spirv_type_key_t key = {
      .kind = LOOM_SPIRV_TYPE_KEY_VOID,
  };
  return loom_spirv_emit_type_id(state, &key, out_type_id,
                                 /*out_emitted=*/NULL);
}

static iree_status_t loom_spirv_emit_type_int(loom_spirv_emit_state_t* state,
                                              uint32_t bit_width,
                                              uint32_t signedness,
                                              uint32_t* out_type_id) {
  const loom_spirv_type_key_t key = {
      .kind = LOOM_SPIRV_TYPE_KEY_INT,
      .operand_count = 2,
      .operands = {bit_width, signedness},
  };
  return loom_spirv_emit_type_id(state, &key, out_type_id,
                                 /*out_emitted=*/NULL);
}

static iree_status_t loom_spirv_emit_type_i32(loom_spirv_emit_state_t* state,
                                              uint32_t* out_type_id) {
  return loom_spirv_emit_type_int(state, 32, 1, out_type_id);
}

static iree_status_t loom_spirv_emit_type_u64(loom_spirv_emit_state_t* state,
                                              uint32_t* out_type_id) {
  return loom_spirv_emit_type_int(state, 64, 0, out_type_id);
}

static iree_status_t loom_spirv_emit_type_function(
    loom_spirv_emit_state_t* state, uint32_t result_type_id,
    const uint32_t* parameter_type_ids, uint8_t parameter_count,
    uint32_t* out_type_id) {
  if ((iree_host_size_t)parameter_count + 1 >
      LOOM_SPIRV_TYPE_KEY_MAX_OPERAND_COUNT) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "SPIR-V function type exceeds the emitter "
                            "type-key operand capacity");
  }
  loom_spirv_type_key_t key = {
      .kind = LOOM_SPIRV_TYPE_KEY_FUNCTION,
      .operand_count = (uint8_t)(parameter_count + 1),
      .operands = {result_type_id},
  };
  for (uint8_t i = 0; i < parameter_count; ++i) {
    key.operands[1 + i] = parameter_type_ids[i];
  }
  return loom_spirv_emit_type_id(state, &key, out_type_id,
                                 /*out_emitted=*/NULL);
}

static iree_status_t loom_spirv_emit_type_pointer(
    loom_spirv_emit_state_t* state, uint32_t storage_class,
    uint32_t pointee_type_id, uint32_t pointer_array_stride,
    uint32_t* out_type_id) {
  const loom_spirv_type_key_t key = {
      .kind = LOOM_SPIRV_TYPE_KEY_POINTER,
      .pointer_array_stride = pointer_array_stride,
      .operand_count = 2,
      .operands = {storage_class, pointee_type_id},
  };
  bool emitted = false;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_id(state, &key, out_type_id, &emitted));
  if (emitted && pointer_array_stride != 0) {
    const uint32_t decoration_operands[] = {
        *out_type_id,
        LOOM_SPIRV_DECORATION_ARRAY_STRIDE,
        pointer_array_stride,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_ANNOTATION),
        LOOM_SPIRV_OP_DECORATE, decoration_operands,
        IREE_ARRAYSIZE(decoration_operands)));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_type_descriptor_struct(
    loom_spirv_emit_state_t* state, uint32_t field_type_id,
    uint32_t* out_type_id) {
  const loom_spirv_type_key_t key = {
      .kind = LOOM_SPIRV_TYPE_KEY_DESCRIPTOR_STRUCT,
      .operand_count = 1,
      .operands = {field_type_id},
  };
  bool emitted = false;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_id(state, &key, out_type_id, &emitted));
  if (emitted) {
    const uint32_t member_decoration_operands[] = {
        *out_type_id,
        0,
        LOOM_SPIRV_DECORATION_OFFSET,
        0,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_ANNOTATION),
        LOOM_SPIRV_OP_MEMBER_DECORATE, member_decoration_operands,
        IREE_ARRAYSIZE(member_decoration_operands)));
    const uint32_t block_decoration_operands[] = {
        *out_type_id,
        LOOM_SPIRV_DECORATION_BLOCK,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_ANNOTATION),
        LOOM_SPIRV_OP_DECORATE, block_decoration_operands,
        IREE_ARRAYSIZE(block_decoration_operands)));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_type_descriptor_i32_struct(
    loom_spirv_emit_state_t* state, uint32_t* out_type_id) {
  uint32_t i32_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_i32(state, &i32_type_id));
  return loom_spirv_emit_type_descriptor_struct(state, i32_type_id,
                                                out_type_id);
}

static iree_status_t loom_spirv_emit_type_descriptor_u64_struct(
    loom_spirv_emit_state_t* state, uint32_t* out_type_id) {
  uint32_t u64_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_u64(state, &u64_type_id));
  return loom_spirv_emit_type_descriptor_struct(state, u64_type_id,
                                                out_type_id);
}

static iree_status_t
loom_spirv_emit_type_ptr_storage_buffer_descriptor_i32_struct(
    loom_spirv_emit_state_t* state, uint32_t* out_type_id) {
  uint32_t struct_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_descriptor_i32_struct(state, &struct_type_id));
  return loom_spirv_emit_type_pointer(
      state, LOOM_SPIRV_STORAGE_CLASS_STORAGE_BUFFER, struct_type_id,
      /*pointer_array_stride=*/0, out_type_id);
}

static iree_status_t
loom_spirv_emit_type_ptr_storage_buffer_descriptor_u64_struct(
    loom_spirv_emit_state_t* state, uint32_t* out_type_id) {
  uint32_t struct_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_descriptor_u64_struct(state, &struct_type_id));
  return loom_spirv_emit_type_pointer(
      state, LOOM_SPIRV_STORAGE_CLASS_STORAGE_BUFFER, struct_type_id,
      /*pointer_array_stride=*/0, out_type_id);
}

static iree_status_t
loom_spirv_emit_type_ptr_storage_buffer_descriptor_i32_field(
    loom_spirv_emit_state_t* state, uint32_t* out_type_id) {
  uint32_t i32_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_i32(state, &i32_type_id));
  return loom_spirv_emit_type_pointer(
      state, LOOM_SPIRV_STORAGE_CLASS_STORAGE_BUFFER, i32_type_id,
      /*pointer_array_stride=*/0, out_type_id);
}

static iree_status_t
loom_spirv_emit_type_ptr_storage_buffer_descriptor_u64_field(
    loom_spirv_emit_state_t* state, uint32_t* out_type_id) {
  uint32_t u64_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_u64(state, &u64_type_id));
  return loom_spirv_emit_type_pointer(
      state, LOOM_SPIRV_STORAGE_CLASS_STORAGE_BUFFER, u64_type_id,
      /*pointer_array_stride=*/0, out_type_id);
}

static iree_status_t loom_spirv_emit_type_ptr_physical_storage_buffer_i32(
    loom_spirv_emit_state_t* state, uint32_t* out_type_id) {
  uint32_t i32_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_i32(state, &i32_type_id));
  // OpPtrAccessChain scales its element operand by the base pointer's
  // ArrayStride. A stride of 1 preserves Loom's byte-offset register semantics.
  return loom_spirv_emit_type_pointer(
      state, LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER, i32_type_id,
      /*pointer_array_stride=*/1, out_type_id);
}

static bool loom_spirv_integer_constant_key_equal(
    const loom_spirv_integer_constant_key_t* lhs,
    const loom_spirv_integer_constant_key_t* rhs) {
  if (lhs->type_id != rhs->type_id || lhs->word_count != rhs->word_count) {
    return false;
  }
  for (uint8_t i = 0; i < lhs->word_count; ++i) {
    if (lhs->words[i] != rhs->words[i]) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_spirv_emit_integer_constant_cache_reserve(
    loom_spirv_emit_state_t* state, iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= state->integer_constant_cache_capacity) {
    return iree_ok_status();
  }
  if (state->integer_constant_cache_capacity == 0) {
    minimum_capacity = iree_max(
        minimum_capacity, LOOM_SPIRV_INTEGER_CONSTANT_CACHE_INITIAL_CAPACITY);
  }
  return iree_arena_grow_array(
      state->scratch_arena, state->integer_constant_cache_count,
      minimum_capacity, sizeof(*state->integer_constant_cache_entries),
      &state->integer_constant_cache_capacity,
      (void**)&state->integer_constant_cache_entries);
}

static iree_status_t loom_spirv_emit_cached_integer_constant(
    loom_spirv_emit_state_t* state,
    const loom_spirv_integer_constant_key_t* key, uint32_t* out_constant_id) {
  for (iree_host_size_t i = 0; i < state->integer_constant_cache_count; ++i) {
    const loom_spirv_integer_constant_cache_entry_t* entry =
        &state->integer_constant_cache_entries[i];
    if (loom_spirv_integer_constant_key_equal(key, &entry->key)) {
      *out_constant_id = entry->id;
      return iree_ok_status();
    }
  }
  IREE_RETURN_IF_ERROR(loom_spirv_emit_integer_constant_cache_reserve(
      state, state->integer_constant_cache_count + 1));

  const uint32_t constant_id = loom_spirv_emit_allocate_id(state);
  uint32_t operands[2 + IREE_ARRAYSIZE(key->words)] = {0};
  operands[0] = key->type_id;
  operands[1] = constant_id;
  for (uint8_t i = 0; i < key->word_count; ++i) {
    operands[2 + i] = key->words[i];
  }
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
      LOOM_SPIRV_OP_CONSTANT, operands, 2 + key->word_count));
  state->integer_constant_cache_entries[state->integer_constant_cache_count++] =
      (loom_spirv_integer_constant_cache_entry_t){
          .key = *key,
          .id = constant_id,
      };
  *out_constant_id = constant_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_i32_zero_constant(
    loom_spirv_emit_state_t* state, uint32_t* out_constant_id) {
  uint32_t i32_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_i32(state, &i32_type_id));
  const loom_spirv_integer_constant_key_t key = {
      .type_id = i32_type_id,
      .word_count = 1,
      .words = {0},
  };
  return loom_spirv_emit_cached_integer_constant(state, &key, out_constant_id);
}

static iree_status_t loom_spirv_emit_low_register_type_kind(
    loom_type_t type, loom_spirv_value_kind_t* out_type_kind) {
  if (!loom_low_type_is_register(type) ||
      loom_low_register_type_descriptor_set_stable_id(type) !=
          SPIRV_LOGICAL_CORE_DESCRIPTOR_SET_ID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V ABI value must be a "
                            "spirv.logical.core register type");
  }
  switch (loom_low_register_type_class_id(type)) {
    case SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID:
      *out_type_kind = LOOM_SPIRV_VALUE_I32;
      return iree_ok_status();
    case SPIRV_LOGICAL_CORE_REG_CLASS_ID_OFFSET64:
      *out_type_kind = LOOM_SPIRV_VALUE_U64;
      return iree_ok_status();
    case SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_STORAGE_BUFFER:
      *out_type_kind = LOOM_SPIRV_VALUE_PTR_PHYSICAL_STORAGE_BUFFER_I32;
      return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "SPIR-V ABI register class %u is not "
                          "supported by binary emission",
                          loom_low_register_type_class_id(type));
}

static iree_status_t loom_spirv_emit_define_value(
    loom_spirv_emit_state_t* state, loom_value_id_t value_id,
    loom_spirv_value_ref_t value_ref, bool emit_name) {
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(&state->value_domain, value_id);
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "SPIR-V value %" PRIu32
                            " is outside the emitted function domain",
                            value_id);
  }
  state->value_refs[value_ordinal] = value_ref;
  if (emit_name) {
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_value_name(state, value_id, value_ref.id));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_lookup_value(
    loom_spirv_emit_state_t* state, loom_value_id_t value_id,
    loom_spirv_value_ref_t* out_value_ref) {
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(&state->value_domain, value_id);
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "SPIR-V operand value %" PRIu32
                            " is outside the emitted function domain",
                            value_id);
  }
  const loom_spirv_value_ref_t value_ref = state->value_refs[value_ordinal];
  if (value_ref.id == 0 || value_ref.type_id == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "SPIR-V operand value %" PRIu32 " has no emitted definition", value_id);
  }
  *out_value_ref = value_ref;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_abi_slot_type_info(
    loom_spirv_emit_state_t* state, loom_spirv_value_kind_t type_kind,
    loom_spirv_abi_slot_type_info_t* out_type_info) {
  *out_type_info = (loom_spirv_abi_slot_type_info_t){0};
  switch (type_kind) {
    case LOOM_SPIRV_VALUE_I32: {
      IREE_RETURN_IF_ERROR(
          loom_spirv_emit_type_i32(state, &out_type_info->value_type_id));
      out_type_info->field_type_id = out_type_info->value_type_id;
      IREE_RETURN_IF_ERROR(
          loom_spirv_emit_type_ptr_storage_buffer_descriptor_i32_struct(
              state, &out_type_info->variable_type_id));
      IREE_RETURN_IF_ERROR(
          loom_spirv_emit_type_ptr_storage_buffer_descriptor_i32_field(
              state, &out_type_info->field_pointer_type_id));
      out_type_info->field_alignment = 4;
      return iree_ok_status();
    }
    case LOOM_SPIRV_VALUE_U64: {
      IREE_RETURN_IF_ERROR(
          loom_spirv_emit_type_u64(state, &out_type_info->value_type_id));
      out_type_info->field_type_id = out_type_info->value_type_id;
      IREE_RETURN_IF_ERROR(
          loom_spirv_emit_type_ptr_storage_buffer_descriptor_u64_struct(
              state, &out_type_info->variable_type_id));
      IREE_RETURN_IF_ERROR(
          loom_spirv_emit_type_ptr_storage_buffer_descriptor_u64_field(
              state, &out_type_info->field_pointer_type_id));
      out_type_info->field_alignment = 8;
      return iree_ok_status();
    }
    case LOOM_SPIRV_VALUE_PTR_PHYSICAL_STORAGE_BUFFER_I32: {
      IREE_RETURN_IF_ERROR(loom_spirv_emit_type_ptr_physical_storage_buffer_i32(
          state, &out_type_info->value_type_id));
      IREE_RETURN_IF_ERROR(
          loom_spirv_emit_type_u64(state, &out_type_info->field_type_id));
      IREE_RETURN_IF_ERROR(
          loom_spirv_emit_type_ptr_storage_buffer_descriptor_u64_struct(
              state, &out_type_info->variable_type_id));
      IREE_RETURN_IF_ERROR(
          loom_spirv_emit_type_ptr_storage_buffer_descriptor_u64_field(
              state, &out_type_info->field_pointer_type_id));
      out_type_info->field_alignment = 8;
      return iree_ok_status();
    }
    case LOOM_SPIRV_VALUE_UNKNOWN:
      break;
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "SPIR-V shader-entry ABI does not support value "
                          "type kind %u",
                          (uint32_t)type_kind);
}

static iree_status_t loom_spirv_emit_declare_abi_slot_variable(
    loom_spirv_emit_state_t* state, const loom_spirv_abi_slot_t* slot) {
  loom_spirv_abi_slot_type_info_t type_info = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_abi_slot_type_info(state, slot->type_kind, &type_info));

  const uint32_t descriptor_set_operands[] = {
      slot->variable_id,
      LOOM_SPIRV_DECORATION_DESCRIPTOR_SET,
      0,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_ANNOTATION),
      LOOM_SPIRV_OP_DECORATE, descriptor_set_operands,
      IREE_ARRAYSIZE(descriptor_set_operands)));
  const uint32_t binding_operands[] = {
      slot->variable_id,
      LOOM_SPIRV_DECORATION_BINDING,
      slot->binding,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_ANNOTATION),
      LOOM_SPIRV_OP_DECORATE, binding_operands,
      IREE_ARRAYSIZE(binding_operands)));

  const uint32_t variable_operands[] = {
      type_info.variable_type_id,
      slot->variable_id,
      LOOM_SPIRV_STORAGE_CLASS_STORAGE_BUFFER,
  };
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
      LOOM_SPIRV_OP_VARIABLE, variable_operands,
      IREE_ARRAYSIZE(variable_operands));
}

static iree_status_t loom_spirv_emit_build_abi_plan(
    loom_spirv_emit_state_t* state, const loom_block_t* entry_block) {
  const iree_host_size_t arg_count = entry_block->arg_count;
  const iree_host_size_t result_count = state->function_op->result_count;
  const iree_host_size_t descriptor_slot_count = arg_count + result_count;

  loom_spirv_abi_plan_t plan = {
      .arg_count = arg_count,
      .result_count = result_count,
  };
  if (arg_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->scratch_arena, arg_count,
                                  sizeof(*plan.args), (void**)&plan.args));
  }
  if (result_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->scratch_arena, result_count, sizeof(*plan.results),
        (void**)&plan.results));
  }
  if (descriptor_slot_count != 0) {
    uint32_t zero_id = 0;
    IREE_RETURN_IF_ERROR(loom_spirv_emit_i32_zero_constant(state, &zero_id));
  }

  uint32_t binding = 0;
  for (iree_host_size_t i = 0; i < arg_count; ++i) {
    loom_spirv_abi_slot_t* slot = &plan.args[i];
    slot->value_id = loom_block_arg_id(entry_block, (uint16_t)i);
    slot->binding = binding++;
    slot->variable_id = loom_spirv_emit_allocate_id(state);
    IREE_RETURN_IF_ERROR(loom_spirv_emit_low_register_type_kind(
        loom_module_value_type(state->module, slot->value_id),
        &slot->type_kind));
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_declare_abi_slot_variable(state, slot));
  }
  const loom_value_id_t* results = loom_op_const_results(state->function_op);
  for (iree_host_size_t i = 0; i < result_count; ++i) {
    loom_spirv_abi_slot_t* slot = &plan.results[i];
    slot->value_id = results[i];
    slot->binding = binding++;
    slot->variable_id = loom_spirv_emit_allocate_id(state);
    IREE_RETURN_IF_ERROR(loom_spirv_emit_low_register_type_kind(
        loom_module_value_type(state->module, slot->value_id),
        &slot->type_kind));
    if (slot->type_kind != LOOM_SPIRV_VALUE_I32 &&
        slot->type_kind != LOOM_SPIRV_VALUE_U64) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "SPIR-V shader-entry ABI does not support "
                              "returning pointer register classes");
    }
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_declare_abi_slot_variable(state, slot));
  }
  state->abi_plan = plan;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_abi_slot_field_pointer(
    loom_spirv_emit_state_t* state, const loom_spirv_abi_slot_t* slot,
    uint32_t field_pointer_type_id, uint32_t* out_field_pointer_id) {
  uint32_t zero_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_i32_zero_constant(state, &zero_id));
  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
  const uint32_t operands[] = {
      field_pointer_type_id,
      result_id,
      slot->variable_id,
      zero_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_ACCESS_CHAIN, operands, IREE_ARRAYSIZE(operands)));
  *out_field_pointer_id = result_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_load_abi_slot_field(
    loom_spirv_emit_state_t* state, const loom_spirv_abi_slot_t* slot,
    const loom_spirv_abi_slot_type_info_t* type_info,
    loom_spirv_value_ref_t* out_field_ref) {
  uint32_t field_pointer_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_abi_slot_field_pointer(
      state, slot, type_info->field_pointer_type_id, &field_pointer_id));
  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
  const uint32_t operands[] = {
      type_info->field_type_id,   result_id,
      field_pointer_id,           LOOM_SPIRV_MEMORY_ACCESS_ALIGNED_MASK,
      type_info->field_alignment,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_LOAD, operands, IREE_ARRAYSIZE(operands)));
  *out_field_ref = (loom_spirv_value_ref_t){
      .id = result_id,
      .type_id = type_info->field_type_id,
      .type_kind = LOOM_SPIRV_VALUE_UNKNOWN,
  };
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_materialize_abi_arg(
    loom_spirv_emit_state_t* state, const loom_spirv_abi_slot_t* slot) {
  loom_spirv_abi_slot_type_info_t type_info = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_abi_slot_type_info(state, slot->type_kind, &type_info));
  loom_spirv_value_ref_t field_ref = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_abi_slot_field(state, slot, &type_info, &field_ref));
  if (slot->type_kind == LOOM_SPIRV_VALUE_PTR_PHYSICAL_STORAGE_BUFFER_I32) {
    const uint32_t result_id = loom_spirv_emit_allocate_id(state);
    const uint32_t operands[] = {
        type_info.value_type_id,
        result_id,
        field_ref.id,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
        LOOM_SPIRV_OP_CONVERT_U_TO_PTR, operands, IREE_ARRAYSIZE(operands)));
    return loom_spirv_emit_define_value(state, slot->value_id,
                                        (loom_spirv_value_ref_t){
                                            .id = result_id,
                                            .type_id = type_info.value_type_id,
                                            .type_kind = slot->type_kind,
                                        },
                                        true);
  }
  field_ref.type_kind = slot->type_kind;
  return loom_spirv_emit_define_value(state, slot->value_id, field_ref, true);
}

static iree_status_t loom_spirv_emit_materialize_abi_args(
    loom_spirv_emit_state_t* state) {
  for (iree_host_size_t i = 0; i < state->abi_plan.arg_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_materialize_abi_arg(state, &state->abi_plan.args[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_store_abi_slot_value(
    loom_spirv_emit_state_t* state, const loom_spirv_abi_slot_t* slot,
    loom_spirv_value_ref_t value) {
  loom_spirv_abi_slot_type_info_t type_info = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_abi_slot_type_info(state, slot->type_kind, &type_info));
  if (value.type_kind != slot->type_kind ||
      value.type_id != type_info.value_type_id) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V shader-entry return value does not match "
                            "the result ABI slot type");
  }
  uint32_t field_pointer_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_abi_slot_field_pointer(
      state, slot, type_info.field_pointer_type_id, &field_pointer_id));
  const uint32_t operands[] = {
      field_pointer_id,
      value.id,
      LOOM_SPIRV_MEMORY_ACCESS_ALIGNED_MASK,
      type_info.field_alignment,
  };
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_STORE, operands, IREE_ARRAYSIZE(operands));
}

static iree_status_t loom_spirv_emit_type_id_for_value_kind(
    loom_spirv_emit_state_t* state, loom_spirv_value_kind_t kind,
    uint32_t* out_type_id) {
  switch (kind) {
    case LOOM_SPIRV_VALUE_I32:
      return loom_spirv_emit_type_i32(state, out_type_id);
    case LOOM_SPIRV_VALUE_U64:
      return loom_spirv_emit_type_u64(state, out_type_id);
    case LOOM_SPIRV_VALUE_PTR_PHYSICAL_STORAGE_BUFFER_I32:
      return loom_spirv_emit_type_ptr_physical_storage_buffer_i32(state,
                                                                  out_type_id);
    case LOOM_SPIRV_VALUE_UNKNOWN:
      break;
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "SPIR-V packet row does not select a result type");
}

static loom_named_attr_slice_t loom_spirv_emit_packet_attrs(
    const loom_low_resolved_descriptor_packet_t* packet) {
  switch (packet->kind) {
    case LOOM_LOW_DESCRIPTOR_PACKET_CONST:
      return loom_low_const_attrs(packet->op);
    case LOOM_LOW_DESCRIPTOR_PACKET_OP:
      return loom_low_op_attrs(packet->op);
    case LOOM_LOW_DESCRIPTOR_PACKET_NONE:
      break;
  }
  return loom_make_named_attr_slice(NULL, 0);
}

static const loom_low_immediate_t* loom_spirv_emit_descriptor_immediate(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    uint8_t descriptor_immediate_index) {
  if (descriptor_immediate_index >= packet->descriptor->immediate_count) {
    return NULL;
  }
  const uint32_t immediate_row =
      packet->descriptor->immediate_start + descriptor_immediate_index;
  if (immediate_row >= state->target->descriptor_set->immediate_count) {
    return NULL;
  }
  return &state->target->descriptor_set->immediates[immediate_row];
}

static iree_string_view_t loom_spirv_emit_immediate_name(
    loom_spirv_emit_state_t* state, const loom_low_immediate_t* immediate) {
  if (immediate == NULL) {
    return IREE_SV("<unknown>");
  }
  return loom_low_descriptor_set_string(state->target->descriptor_set,
                                        immediate->field_name_string_offset);
}

static iree_status_t loom_spirv_emit_lookup_i64_immediate(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row, int64_t* out_value) {
  const loom_low_immediate_t* immediate =
      loom_spirv_emit_descriptor_immediate(state, packet, row->immediate_index);
  if (immediate == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "SPIR-V packet row references missing immediate %" PRIu8,
        row->immediate_index);
  }
  const uint32_t immediate_row =
      packet->descriptor->immediate_start + row->immediate_index;
  if (immediate_row >= state->immediate_name_id_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "SPIR-V immediate name table is incomplete");
  }
  const loom_string_id_t expected_name_id =
      state->immediate_name_ids[immediate_row];
  const loom_named_attr_slice_t attrs = loom_spirv_emit_packet_attrs(packet);
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (attr->name_id != expected_name_id) {
      continue;
    }
    if (attr->value.kind != LOOM_ATTR_I64) {
      const iree_string_view_t immediate_name =
          loom_spirv_emit_immediate_name(state, immediate);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "SPIR-V low descriptor '%.*s' immediate '%.*s' must be an i64 "
          "attribute",
          (int)packet->key.size, packet->key.data, (int)immediate_name.size,
          immediate_name.data);
    }
    *out_value = loom_attr_as_i64(attr->value);
    return iree_ok_status();
  }
  if (iree_any_bit_set(immediate->flags,
                       LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE)) {
    *out_value = immediate->default_value;
    return iree_ok_status();
  }
  const iree_string_view_t immediate_name =
      loom_spirv_emit_immediate_name(state, immediate);
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "SPIR-V low descriptor '%.*s' is missing "
                          "immediate '%.*s'",
                          (int)packet->key.size, packet->key.data,
                          (int)immediate_name.size, immediate_name.data);
}

static iree_status_t loom_spirv_emit_validate_packet_shape(
    const loom_op_t* op, const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  if (op->result_count == row->result_count &&
      op->operand_count == row->operand_count) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "SPIR-V low descriptor '%.*s' expects %" PRIu8 " results and %" PRIu8
      " operands but op has %u "
      "results and %u operands",
      (int)packet->key.size, packet->key.data, row->result_count,
      row->operand_count, op->result_count, op->operand_count);
}

static iree_status_t loom_spirv_emit_load_packet_operands(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row, loom_spirv_value_ref_t* out_operands) {
  const loom_value_id_t* operand_values = loom_op_const_operands(packet->op);
  for (uint8_t i = 0; i < row->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_value(state, operand_values[i],
                                                      &out_operands[i]));
    if (out_operands[i].type_kind != row->operand_kinds[i]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "SPIR-V low descriptor '%.*s' operand %" PRIu8
                              " type does not match the packet row",
                              (int)packet->key.size, packet->key.data, i);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_integer_constant_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_i64_immediate(state, packet, row, &value));
  uint32_t type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_kind(
      state, row->result_kind, &type_id));
  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
  const uint64_t literal = (uint64_t)value;
  uint32_t operands[] = {
      type_id,
      result_id,
      (uint32_t)literal,
      (uint32_t)(literal >> 32),
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
      row->opcode, operands, 2 + row->literal_word_count));
  return loom_spirv_emit_define_value(state, loom_low_const_result(packet->op),
                                      (loom_spirv_value_ref_t){
                                          .id = result_id,
                                          .type_id = type_id,
                                          .type_kind = row->result_kind,
                                      },
                                      true);
}

static iree_status_t loom_spirv_emit_binary_same_type_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_value_ref_t operands[2] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  if (operands[0].type_id != operands[1].type_id) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V low descriptor '%.*s' operand types do "
                            "not match",
                            (int)packet->key.size, packet->key.data);
  }
  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
  const uint32_t instruction_operands[] = {
      operands[0].type_id,
      result_id,
      operands[0].id,
      operands[1].id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_value(state,
                                      loom_op_const_results(packet->op)[0],
                                      (loom_spirv_value_ref_t){
                                          .id = result_id,
                                          .type_id = operands[0].type_id,
                                          .type_kind = row->result_kind,
                                      },
                                      true);
}

static iree_status_t loom_spirv_emit_ptr_access_chain_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_value_ref_t operands[2] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_kind(
      state, row->result_kind, &result_type_id));
  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
  const uint32_t instruction_operands[] = {
      result_type_id,
      result_id,
      operands[0].id,
      operands[1].id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_value(state,
                                      loom_op_const_results(packet->op)[0],
                                      (loom_spirv_value_ref_t){
                                          .id = result_id,
                                          .type_id = result_type_id,
                                          .type_kind = row->result_kind,
                                      },
                                      true);
}

static iree_status_t loom_spirv_emit_load_aligned_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_value_ref_t operands[1] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_kind(
      state, row->result_kind, &result_type_id));
  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
  const uint32_t instruction_operands[] = {
      result_type_id,        result_id,
      operands[0].id,        LOOM_SPIRV_MEMORY_ACCESS_ALIGNED_MASK,
      row->memory_alignment,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_value(state,
                                      loom_op_const_results(packet->op)[0],
                                      (loom_spirv_value_ref_t){
                                          .id = result_id,
                                          .type_id = result_type_id,
                                          .type_kind = row->result_kind,
                                      },
                                      true);
}

static iree_status_t loom_spirv_emit_store_aligned_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_value_ref_t operands[2] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  const uint32_t instruction_operands[] = {
      operands[0].id,
      operands[1].id,
      LOOM_SPIRV_MEMORY_ACCESS_ALIGNED_MASK,
      row->memory_alignment,
  };
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands));
}

static iree_status_t loom_spirv_emit_copy(loom_spirv_emit_state_t* state,
                                          const loom_op_t* op) {
  loom_spirv_value_ref_t source = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_value(state, loom_low_copy_source(op), &source));
  return loom_spirv_emit_define_value(state, loom_low_copy_result(op), source,
                                      false);
}

static iree_status_t loom_spirv_emit_descriptor_packet(
    loom_spirv_emit_state_t* state, const loom_op_t* op,
    const loom_low_resolved_descriptor_packet_t* packet) {
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_descriptor_ordinal(state->target->descriptor_set,
                                                 packet->descriptor);
  const loom_spirv_packet_row_t* row =
      loom_spirv_packet_row_for_descriptor_ordinal(descriptor_ordinal);
  if (row == NULL) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "SPIR-V low descriptor '%.*s' is not supported by "
                            "binary emission",
                            (int)packet->key.size, packet->key.data);
  }
  IREE_RETURN_IF_ERROR(loom_spirv_emit_validate_packet_shape(op, packet, row));
  switch (row->form) {
    case LOOM_SPIRV_PACKET_FORM_INTEGER_CONSTANT:
      return loom_spirv_emit_integer_constant_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_BINARY_SAME_TYPE:
      return loom_spirv_emit_binary_same_type_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_PTR_ACCESS_CHAIN:
      return loom_spirv_emit_ptr_access_chain_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_LOAD_ALIGNED:
      return loom_spirv_emit_load_aligned_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_STORE_ALIGNED:
      return loom_spirv_emit_store_aligned_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_UNSUPPORTED:
      break;
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "SPIR-V low descriptor '%.*s' selects an "
                          "unsupported packet row",
                          (int)packet->key.size, packet->key.data);
}

static iree_status_t loom_spirv_emit_low_op(loom_spirv_emit_state_t* state,
                                            const loom_op_t* op) {
  if (loom_low_copy_isa(op)) {
    return loom_spirv_emit_copy(state, op);
  }

  loom_low_resolved_descriptor_packet_t packet = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_descriptor_packet(
      state->module, state->target, op, &packet));
  if (packet.kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
    const loom_op_vtable_t* vtable = loom_op_vtable(state->module, op);
    const iree_string_view_t op_name =
        vtable ? loom_op_vtable_name(vtable) : IREE_SV("<unknown>");
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "SPIR-V emission does not support op '%.*s'",
                            (int)op_name.size, op_name.data);
  }
  if (packet.descriptor == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V low descriptor '%.*s' is unresolved",
                            (int)packet.key.size, packet.key.data);
  }
  return loom_spirv_emit_descriptor_packet(state, op, &packet);
}

static iree_status_t loom_spirv_emit_return(loom_spirv_emit_state_t* state,
                                            const loom_op_t* op) {
  if (op->operand_count != state->abi_plan.result_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V low return operand count does not match "
                            "the shader-entry result ABI");
  }
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (iree_host_size_t i = 0; i < state->abi_plan.result_count; ++i) {
    loom_spirv_value_ref_t value = {0};
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_lookup_value(state, operands[i], &value));
    IREE_RETURN_IF_ERROR(loom_spirv_emit_store_abi_slot_value(
        state, &state->abi_plan.results[i], value));
  }
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_RETURN, NULL, 0);
}

static iree_status_t loom_spirv_emit_entry_point(
    loom_spirv_emit_state_t* state) {
  const uint32_t prefix_operands[] = {
      LOOM_SPIRV_EXECUTION_MODEL_GL_COMPUTE,
      state->function_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_string_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_ENTRY_POINT),
      LOOM_SPIRV_OP_ENTRY_POINT, prefix_operands,
      IREE_ARRAYSIZE(prefix_operands), loom_spirv_emit_export_name(state), NULL,
      0));
  const uint32_t execution_mode_operands[] = {
      state->function_id, LOOM_SPIRV_EXECUTION_MODE_LOCAL_SIZE, 1, 1, 1,
  };
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_EXECUTION_MODE),
      LOOM_SPIRV_OP_EXECUTION_MODE, execution_mode_operands,
      IREE_ARRAYSIZE(execution_mode_operands));
}

static iree_status_t loom_spirv_emit_function_signature(
    loom_spirv_emit_state_t* state, uint32_t* out_result_type_id,
    uint32_t* out_function_type_id) {
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_void(state, &result_type_id));
  uint32_t function_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_function(
      state, result_type_id, /*parameter_type_ids=*/NULL,
      /*parameter_count=*/0, &function_type_id));
  *out_result_type_id = result_type_id;
  *out_function_type_id = function_type_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_function_body(
    loom_spirv_emit_state_t* state, const loom_block_t* entry_block) {
  const uint32_t label_id = loom_spirv_emit_allocate_id(state);
  const uint32_t label_operands[] = {label_id};
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_LABEL, label_operands, IREE_ARRAYSIZE(label_operands)));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_materialize_abi_args(state));
  for (const loom_op_t* op = entry_block->first_op; op != NULL;
       op = op->next_op) {
    if (loom_low_return_isa(op)) {
      IREE_RETURN_IF_ERROR(loom_spirv_emit_return(state, op));
    } else {
      IREE_RETURN_IF_ERROR(loom_spirv_emit_low_op(state, op));
    }
  }
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_FUNCTION_END, NULL, 0);
}

static iree_status_t loom_spirv_emit_function(loom_spirv_emit_state_t* state) {
  const loom_region_t* body = loom_low_function_const_body(state->function_op);
  if (body == NULL || body->block_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V emission requires a low function body");
  }
  if (body->block_count != 1) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "SPIR-V emission currently supports single-block "
                            "low functions");
  }
  const loom_block_t* entry_block = loom_region_const_entry_block(body);
  state->function_id = loom_spirv_emit_allocate_id(state);
  IREE_RETURN_IF_ERROR(loom_spirv_emit_op_name(
      state, state->function_id, loom_spirv_emit_function_name(state)));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_build_abi_plan(state, entry_block));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_entry_point(state));

  uint32_t result_type_id = 0;
  uint32_t function_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_function_signature(
      state, &result_type_id, &function_type_id));

  const uint32_t function_operands[] = {
      result_type_id,
      state->function_id,
      LOOM_SPIRV_FUNCTION_CONTROL_NONE,
      function_type_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_FUNCTION, function_operands,
      IREE_ARRAYSIZE(function_operands)));
  return loom_spirv_emit_function_body(state, entry_block);
}

iree_status_t loom_spirv_emit_low_function_module(
    loom_module_t* module, loom_op_t* low_function_op,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* scratch_arena,
    loom_spirv_module_binary_t* out_module, iree_allocator_t allocator) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(low_function_op);
  IREE_ASSERT_ARGUMENT(descriptor_registry);
  IREE_ASSERT_ARGUMENT(scratch_arena);
  IREE_ASSERT_ARGUMENT(out_module);

  *out_module = (loom_spirv_module_binary_t){0};
  if (!loom_low_function_def_isa(low_function_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V emission requires a low function "
                            "definition");
  }

  loom_low_resolved_target_t target = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      module, low_function_op, descriptor_registry, diagnostic_emitter,
      &target));
  if (target.descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "SPIR-V low function target did not resolve to a "
                            "descriptor set");
  }
  if (target.descriptor_set->stable_id !=
      SPIRV_LOGICAL_CORE_DESCRIPTOR_SET_ID) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "SPIR-V emission requires descriptor set 'spirv.logical.core'");
  }

  loom_spirv_module_builder_t builder = {0};
  iree_status_t status = loom_spirv_module_builder_initialize(
      &target.bundle_storage.bundle, allocator, &builder);

  loom_spirv_emit_state_t state = {
      .module = module,
      .function_op = low_function_op,
      .target = &target,
      .scratch_arena = scratch_arena,
      .builder = &builder,
  };
  const loom_region_t* body = loom_low_function_const_body(low_function_op);
  if (iree_status_is_ok(status) && body == NULL) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "SPIR-V emission requires a low function body");
  }
  if (iree_status_is_ok(status)) {
    status = loom_local_value_domain_acquire_for_region(
        module, body, scratch_arena, &state.value_domain);
  }
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(
        scratch_arena, state.value_domain.value_count,
        sizeof(*state.value_refs), (void**)&state.value_refs);
  }
  if (iree_status_is_ok(status)) {
    status = loom_spirv_emit_prepare_immediate_name_ids(&state);
  }
  if (iree_status_is_ok(status)) {
    memset(state.value_refs, 0,
           state.value_domain.value_count * sizeof(*state.value_refs));
    status = loom_spirv_emit_function(&state);
  }
  if (iree_status_is_ok(status)) {
    status = loom_spirv_module_builder_finalize(&builder, out_module);
  }

  if (loom_local_value_domain_is_acquired(&state.value_domain)) {
    loom_local_value_domain_release(&state.value_domain);
  }
  loom_spirv_module_builder_deinitialize(&builder);
  if (!iree_status_is_ok(status)) {
    loom_spirv_module_binary_deinitialize(out_module, allocator);
  }
  return status;
}
