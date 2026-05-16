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
#include "loom/target/registers.h"
#include "loom/target/types.h"

typedef enum loom_spirv_emitted_type_kind_e {
  LOOM_SPIRV_EMITTED_TYPE_UNKNOWN = 0,
  LOOM_SPIRV_EMITTED_TYPE_I32 = 1,
  LOOM_SPIRV_EMITTED_TYPE_U64 = 2,
  LOOM_SPIRV_EMITTED_TYPE_PTR_PHYSICAL_STORAGE_BUFFER_I32 = 3,
} loom_spirv_emitted_type_kind_t;

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
  loom_spirv_emitted_type_kind_t type_kind;
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
  loom_spirv_emitted_type_kind_t type_kind;
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
  // SPIR-V ID assigned to the function.
  uint32_t function_id;
  // Cached OpTypeVoid result ID.
  uint32_t void_type_id;
  // Cached signed OpTypeInt 32 result ID.
  uint32_t i32_type_id;
  // Cached unsigned OpTypeInt 64 result ID.
  uint32_t u64_type_id;
  // Cached OpTypeStruct {i32} storage-buffer descriptor block result ID.
  uint32_t descriptor_i32_struct_type_id;
  // Cached OpTypeStruct {u64} storage-buffer descriptor block result ID.
  uint32_t descriptor_u64_struct_type_id;
  // Cached StorageBuffer pointer-to-descriptor-i32-struct result ID.
  uint32_t ptr_storage_buffer_descriptor_i32_struct_type_id;
  // Cached StorageBuffer pointer-to-descriptor-u64-struct result ID.
  uint32_t ptr_storage_buffer_descriptor_u64_struct_type_id;
  // Cached StorageBuffer pointer-to-descriptor-i32-field result ID.
  uint32_t ptr_storage_buffer_descriptor_i32_field_type_id;
  // Cached StorageBuffer pointer-to-descriptor-u64-field result ID.
  uint32_t ptr_storage_buffer_descriptor_u64_field_type_id;
  // Cached byte-addressed PhysicalStorageBuffer pointer-to-i32 result ID.
  uint32_t ptr_physical_storage_buffer_i32_type_id;
  // Cached i32 zero constant used for descriptor-block field indexing.
  uint32_t i32_zero_constant_id;
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

static iree_status_t loom_spirv_emit_type_void(loom_spirv_emit_state_t* state,
                                               uint32_t* out_type_id) {
  if (state->void_type_id == 0) {
    state->void_type_id = loom_spirv_emit_allocate_id(state);
    const uint32_t operands[] = {state->void_type_id};
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
        LOOM_SPIRV_OP_TYPE_VOID, operands, IREE_ARRAYSIZE(operands)));
  }
  *out_type_id = state->void_type_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_type_int(loom_spirv_emit_state_t* state,
                                              uint32_t bit_width,
                                              uint32_t signedness,
                                              uint32_t* inout_type_id) {
  if (*inout_type_id == 0) {
    *inout_type_id = loom_spirv_emit_allocate_id(state);
    const uint32_t operands[] = {*inout_type_id, bit_width, signedness};
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
        LOOM_SPIRV_OP_TYPE_INT, operands, IREE_ARRAYSIZE(operands)));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_type_i32(loom_spirv_emit_state_t* state,
                                              uint32_t* out_type_id) {
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_int(state, 32, 1, &state->i32_type_id));
  *out_type_id = state->i32_type_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_type_u64(loom_spirv_emit_state_t* state,
                                              uint32_t* out_type_id) {
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_int(state, 64, 0, &state->u64_type_id));
  *out_type_id = state->u64_type_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_type_pointer(
    loom_spirv_emit_state_t* state, uint32_t storage_class,
    uint32_t pointee_type_id, uint32_t* inout_type_id) {
  if (*inout_type_id == 0) {
    *inout_type_id = loom_spirv_emit_allocate_id(state);
    const uint32_t operands[] = {
        *inout_type_id,
        storage_class,
        pointee_type_id,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
        LOOM_SPIRV_OP_TYPE_POINTER, operands, IREE_ARRAYSIZE(operands)));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_type_descriptor_struct(
    loom_spirv_emit_state_t* state, uint32_t field_type_id,
    uint32_t* inout_struct_type_id) {
  if (*inout_struct_type_id == 0) {
    *inout_struct_type_id = loom_spirv_emit_allocate_id(state);
    const uint32_t type_operands[] = {*inout_struct_type_id, field_type_id};
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
        LOOM_SPIRV_OP_TYPE_STRUCT, type_operands,
        IREE_ARRAYSIZE(type_operands)));
    const uint32_t member_decoration_operands[] = {
        *inout_struct_type_id,
        0,
        LOOM_SPIRV_DECORATION_OFFSET,
        0,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_ANNOTATION),
        LOOM_SPIRV_OP_MEMBER_DECORATE, member_decoration_operands,
        IREE_ARRAYSIZE(member_decoration_operands)));
    const uint32_t block_decoration_operands[] = {
        *inout_struct_type_id,
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
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_descriptor_struct(
      state, i32_type_id, &state->descriptor_i32_struct_type_id));
  *out_type_id = state->descriptor_i32_struct_type_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_type_descriptor_u64_struct(
    loom_spirv_emit_state_t* state, uint32_t* out_type_id) {
  uint32_t u64_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_u64(state, &u64_type_id));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_descriptor_struct(
      state, u64_type_id, &state->descriptor_u64_struct_type_id));
  *out_type_id = state->descriptor_u64_struct_type_id;
  return iree_ok_status();
}

static iree_status_t
loom_spirv_emit_type_ptr_storage_buffer_descriptor_i32_struct(
    loom_spirv_emit_state_t* state, uint32_t* out_type_id) {
  uint32_t struct_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_descriptor_i32_struct(state, &struct_type_id));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_pointer(
      state, LOOM_SPIRV_STORAGE_CLASS_STORAGE_BUFFER, struct_type_id,
      &state->ptr_storage_buffer_descriptor_i32_struct_type_id));
  *out_type_id = state->ptr_storage_buffer_descriptor_i32_struct_type_id;
  return iree_ok_status();
}

static iree_status_t
loom_spirv_emit_type_ptr_storage_buffer_descriptor_u64_struct(
    loom_spirv_emit_state_t* state, uint32_t* out_type_id) {
  uint32_t struct_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_descriptor_u64_struct(state, &struct_type_id));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_pointer(
      state, LOOM_SPIRV_STORAGE_CLASS_STORAGE_BUFFER, struct_type_id,
      &state->ptr_storage_buffer_descriptor_u64_struct_type_id));
  *out_type_id = state->ptr_storage_buffer_descriptor_u64_struct_type_id;
  return iree_ok_status();
}

static iree_status_t
loom_spirv_emit_type_ptr_storage_buffer_descriptor_i32_field(
    loom_spirv_emit_state_t* state, uint32_t* out_type_id) {
  uint32_t i32_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_i32(state, &i32_type_id));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_pointer(
      state, LOOM_SPIRV_STORAGE_CLASS_STORAGE_BUFFER, i32_type_id,
      &state->ptr_storage_buffer_descriptor_i32_field_type_id));
  *out_type_id = state->ptr_storage_buffer_descriptor_i32_field_type_id;
  return iree_ok_status();
}

static iree_status_t
loom_spirv_emit_type_ptr_storage_buffer_descriptor_u64_field(
    loom_spirv_emit_state_t* state, uint32_t* out_type_id) {
  uint32_t u64_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_u64(state, &u64_type_id));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_pointer(
      state, LOOM_SPIRV_STORAGE_CLASS_STORAGE_BUFFER, u64_type_id,
      &state->ptr_storage_buffer_descriptor_u64_field_type_id));
  *out_type_id = state->ptr_storage_buffer_descriptor_u64_field_type_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_type_ptr_physical_storage_buffer_i32(
    loom_spirv_emit_state_t* state, uint32_t* out_type_id) {
  uint32_t i32_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_i32(state, &i32_type_id));
  const bool emit_array_stride =
      state->ptr_physical_storage_buffer_i32_type_id == 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_pointer(
      state, LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER, i32_type_id,
      &state->ptr_physical_storage_buffer_i32_type_id));
  if (emit_array_stride) {
    // OpPtrAccessChain scales its element operand by the base pointer's
    // ArrayStride. A stride of 1 preserves Loom's byte-offset register
    // semantics.
    const uint32_t decoration_operands[] = {
        state->ptr_physical_storage_buffer_i32_type_id,
        LOOM_SPIRV_DECORATION_ARRAY_STRIDE,
        1,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_ANNOTATION),
        LOOM_SPIRV_OP_DECORATE, decoration_operands,
        IREE_ARRAYSIZE(decoration_operands)));
  }
  *out_type_id = state->ptr_physical_storage_buffer_i32_type_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_i32_zero_constant(
    loom_spirv_emit_state_t* state, uint32_t* out_constant_id) {
  if (state->i32_zero_constant_id == 0) {
    uint32_t i32_type_id = 0;
    IREE_RETURN_IF_ERROR(loom_spirv_emit_type_i32(state, &i32_type_id));
    state->i32_zero_constant_id = loom_spirv_emit_allocate_id(state);
    const uint32_t operands[] = {i32_type_id, state->i32_zero_constant_id, 0};
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
        LOOM_SPIRV_OP_CONSTANT, operands, IREE_ARRAYSIZE(operands)));
  }
  *out_constant_id = state->i32_zero_constant_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_low_register_type_kind(
    loom_type_t type, loom_spirv_emitted_type_kind_t* out_type_kind) {
  if (!loom_low_type_is_register(type) ||
      loom_low_register_type_descriptor_set_stable_id(type) !=
          SPIRV_LOGICAL_CORE_DESCRIPTOR_SET_ID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V ABI value must be a "
                            "spirv.logical.core register type");
  }
  switch (loom_low_register_type_class_id(type)) {
    case SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID:
      *out_type_kind = LOOM_SPIRV_EMITTED_TYPE_I32;
      return iree_ok_status();
    case SPIRV_LOGICAL_CORE_REG_CLASS_ID_OFFSET64:
      *out_type_kind = LOOM_SPIRV_EMITTED_TYPE_U64;
      return iree_ok_status();
    case SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_STORAGE_BUFFER:
      *out_type_kind = LOOM_SPIRV_EMITTED_TYPE_PTR_PHYSICAL_STORAGE_BUFFER_I32;
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
    loom_spirv_emit_state_t* state, loom_spirv_emitted_type_kind_t type_kind,
    loom_spirv_abi_slot_type_info_t* out_type_info) {
  *out_type_info = (loom_spirv_abi_slot_type_info_t){0};
  switch (type_kind) {
    case LOOM_SPIRV_EMITTED_TYPE_I32: {
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
    case LOOM_SPIRV_EMITTED_TYPE_U64: {
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
    case LOOM_SPIRV_EMITTED_TYPE_PTR_PHYSICAL_STORAGE_BUFFER_I32: {
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
    case LOOM_SPIRV_EMITTED_TYPE_UNKNOWN:
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
    if (slot->type_kind != LOOM_SPIRV_EMITTED_TYPE_I32 &&
        slot->type_kind != LOOM_SPIRV_EMITTED_TYPE_U64) {
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
      .type_kind = LOOM_SPIRV_EMITTED_TYPE_UNKNOWN,
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
  if (slot->type_kind ==
      LOOM_SPIRV_EMITTED_TYPE_PTR_PHYSICAL_STORAGE_BUFFER_I32) {
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

static bool loom_spirv_emit_attr_name_equals(const loom_module_t* module,
                                             loom_string_id_t name_id,
                                             iree_string_view_t expected) {
  return iree_string_view_equal(
      loom_spirv_emit_string_or_empty(module, name_id), expected);
}

static bool loom_spirv_emit_lookup_attr(const loom_module_t* module,
                                        loom_named_attr_slice_t attrs,
                                        iree_string_view_t name,
                                        loom_attribute_t* out_attr) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    if (loom_spirv_emit_attr_name_equals(module, attrs.entries[i].name_id,
                                         name)) {
      *out_attr = attrs.entries[i].value;
      return true;
    }
  }
  return false;
}

static iree_status_t loom_spirv_emit_lookup_i64_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, int64_t* out_value) {
  loom_attribute_t attr = {0};
  if (!loom_spirv_emit_lookup_attr(module, attrs, name, &attr) ||
      attr.kind != LOOM_ATTR_I64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V low constant is missing i64 attribute "
                            "'%.*s'",
                            (int)name.size, name.data);
  }
  *out_value = loom_attr_as_i64(attr);
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_constant_i32(
    loom_spirv_emit_state_t* state, const loom_op_t* op) {
  if (op->result_count != 1 || op->operand_count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "OpConstant.i32 expects one result and no operands");
  }
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_i64_attr(
      state->module, loom_low_const_attrs(op), IREE_SV("i32_value"), &value));

  uint32_t i32_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_i32(state, &i32_type_id));
  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
  const uint32_t operands[] = {
      i32_type_id,
      result_id,
      (uint32_t)value,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
      LOOM_SPIRV_OP_CONSTANT, operands, IREE_ARRAYSIZE(operands)));
  return loom_spirv_emit_define_value(
      state, loom_low_const_result(op),
      (loom_spirv_value_ref_t){
          .id = result_id,
          .type_id = i32_type_id,
          .type_kind = LOOM_SPIRV_EMITTED_TYPE_I32,
      },
      true);
}

static iree_status_t loom_spirv_emit_constant_offset64(
    loom_spirv_emit_state_t* state, const loom_op_t* op) {
  if (op->result_count != 1 || op->operand_count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "OpConstant.offset64 expects one result and no operands");
  }
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_i64_attr(state->module, loom_low_const_attrs(op),
                                      IREE_SV("offset64_value"), &value));

  uint32_t u64_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_u64(state, &u64_type_id));
  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
  const uint64_t literal = (uint64_t)value;
  const uint32_t operands[] = {
      u64_type_id,
      result_id,
      (uint32_t)literal,
      (uint32_t)(literal >> 32),
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
      LOOM_SPIRV_OP_CONSTANT, operands, IREE_ARRAYSIZE(operands)));
  return loom_spirv_emit_define_value(
      state, loom_low_const_result(op),
      (loom_spirv_value_ref_t){
          .id = result_id,
          .type_id = u64_type_id,
          .type_kind = LOOM_SPIRV_EMITTED_TYPE_U64,
      },
      true);
}

static iree_status_t loom_spirv_emit_iadd(loom_spirv_emit_state_t* state,
                                          const loom_op_t* op,
                                          loom_spirv_emitted_type_kind_t kind) {
  if (op->result_count != 1 || op->operand_count != 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "OpIAdd expects one result and two operands");
  }
  loom_spirv_value_ref_t lhs = {0};
  loom_spirv_value_ref_t rhs = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_value(state, loom_op_const_operands(op)[0], &lhs));
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_value(state, loom_op_const_operands(op)[1], &rhs));
  if (lhs.type_kind != kind || rhs.type_kind != kind ||
      lhs.type_id != rhs.type_id) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "OpIAdd operand types do not match the selected "
                            "SPIR-V descriptor");
  }

  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
  const uint32_t operands[] = {lhs.type_id, result_id, lhs.id, rhs.id};
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_I_ADD, operands, IREE_ARRAYSIZE(operands)));
  return loom_spirv_emit_define_value(state, loom_op_const_results(op)[0],
                                      (loom_spirv_value_ref_t){
                                          .id = result_id,
                                          .type_id = lhs.type_id,
                                          .type_kind = kind,
                                      },
                                      true);
}

static iree_status_t loom_spirv_emit_ptr_access_chain_storage_buffer_i32(
    loom_spirv_emit_state_t* state, const loom_op_t* op) {
  if (op->result_count != 1 || op->operand_count != 2) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "OpPtrAccessChain.storage_buffer.byte_offset expects one result and "
        "two operands");
  }
  loom_spirv_value_ref_t base = {0};
  loom_spirv_value_ref_t byte_offset = {0};
  IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_value(
      state, loom_op_const_operands(op)[0], &base));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_value(
      state, loom_op_const_operands(op)[1], &byte_offset));
  if (base.type_kind !=
          LOOM_SPIRV_EMITTED_TYPE_PTR_PHYSICAL_STORAGE_BUFFER_I32 ||
      byte_offset.type_kind != LOOM_SPIRV_EMITTED_TYPE_U64) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "OpPtrAccessChain.storage_buffer.byte_offset operand types do not "
        "match the selected SPIR-V descriptor");
  }

  uint32_t ptr_i32_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_ptr_physical_storage_buffer_i32(
      state, &ptr_i32_type_id));
  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
  const uint32_t operands[] = {
      ptr_i32_type_id,
      result_id,
      base.id,
      byte_offset.id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_PTR_ACCESS_CHAIN, operands, IREE_ARRAYSIZE(operands)));
  return loom_spirv_emit_define_value(
      state, loom_op_const_results(op)[0],
      (loom_spirv_value_ref_t){
          .id = result_id,
          .type_id = ptr_i32_type_id,
          .type_kind = LOOM_SPIRV_EMITTED_TYPE_PTR_PHYSICAL_STORAGE_BUFFER_I32,
      },
      true);
}

static iree_status_t loom_spirv_emit_load_storage_buffer_i32(
    loom_spirv_emit_state_t* state, const loom_op_t* op) {
  if (op->result_count != 1 || op->operand_count != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "OpLoad.storage_buffer.i32 expects one result and one operand");
  }
  loom_spirv_value_ref_t ptr = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_value(state, loom_op_const_operands(op)[0], &ptr));
  if (ptr.type_kind !=
      LOOM_SPIRV_EMITTED_TYPE_PTR_PHYSICAL_STORAGE_BUFFER_I32) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "OpLoad.storage_buffer.i32 requires a "
                            "storage-buffer i32 pointer");
  }

  uint32_t i32_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_i32(state, &i32_type_id));
  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
  const uint32_t operands[] = {
      i32_type_id, result_id, ptr.id, LOOM_SPIRV_MEMORY_ACCESS_ALIGNED_MASK, 4,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_LOAD, operands, IREE_ARRAYSIZE(operands)));
  return loom_spirv_emit_define_value(
      state, loom_op_const_results(op)[0],
      (loom_spirv_value_ref_t){
          .id = result_id,
          .type_id = i32_type_id,
          .type_kind = LOOM_SPIRV_EMITTED_TYPE_I32,
      },
      true);
}

static iree_status_t loom_spirv_emit_store_storage_buffer_i32(
    loom_spirv_emit_state_t* state, const loom_op_t* op) {
  if (op->result_count != 0 || op->operand_count != 2) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "OpStore.storage_buffer.i32 expects no results and two operands");
  }
  loom_spirv_value_ref_t ptr = {0};
  loom_spirv_value_ref_t value = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_value(state, loom_op_const_operands(op)[0], &ptr));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_value(
      state, loom_op_const_operands(op)[1], &value));
  if (ptr.type_kind !=
          LOOM_SPIRV_EMITTED_TYPE_PTR_PHYSICAL_STORAGE_BUFFER_I32 ||
      value.type_kind != LOOM_SPIRV_EMITTED_TYPE_I32) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "OpStore.storage_buffer.i32 operand types do not "
                            "match the selected SPIR-V descriptor");
  }

  const uint32_t operands[] = {
      ptr.id,
      value.id,
      LOOM_SPIRV_MEMORY_ACCESS_ALIGNED_MASK,
      4,
  };
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_STORE, operands, IREE_ARRAYSIZE(operands));
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
  switch (descriptor_ordinal) {
    case SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_CONSTANT_I32:
      return loom_spirv_emit_constant_i32(state, op);
    case SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_CONSTANT_OFFSET64:
      return loom_spirv_emit_constant_offset64(state, op);
    case SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_IADD_I32:
      return loom_spirv_emit_iadd(state, op, LOOM_SPIRV_EMITTED_TYPE_I32);
    case SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_IADD_OFFSET64:
      return loom_spirv_emit_iadd(state, op, LOOM_SPIRV_EMITTED_TYPE_U64);
    case SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_PTR_ACCESS_CHAIN_STORAGE_BUFFER_BYTE_OFFSET:
      return loom_spirv_emit_ptr_access_chain_storage_buffer_i32(state, op);
    case SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_LOAD_STORAGE_BUFFER_I32:
      return loom_spirv_emit_load_storage_buffer_i32(state, op);
    case SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_STORE_STORAGE_BUFFER_I32:
      return loom_spirv_emit_store_storage_buffer_i32(state, op);
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "SPIR-V low descriptor '%.*s' is not supported by "
                          "binary emission",
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
  const uint32_t function_type_id = loom_spirv_emit_allocate_id(state);
  const uint32_t function_type_operands[] = {
      function_type_id,
      result_type_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
      LOOM_SPIRV_OP_TYPE_FUNCTION, function_type_operands,
      IREE_ARRAYSIZE(function_type_operands)));
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
