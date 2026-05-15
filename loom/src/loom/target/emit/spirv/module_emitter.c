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
  LOOM_SPIRV_EMITTED_TYPE_PTR_STORAGE_BUFFER_I32 = 3,
  LOOM_SPIRV_EMITTED_TYPE_PTR_STORAGE_BUFFER_RUNTIME_ARRAY_I32 = 4,
} loom_spirv_emitted_type_kind_t;

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
  // Cached OpTypeRuntimeArray i32 result ID.
  uint32_t runtime_array_i32_type_id;
  // Cached PhysicalStorageBuffer pointer-to-i32 result ID.
  uint32_t ptr_storage_buffer_i32_type_id;
  // Cached PhysicalStorageBuffer pointer-to-runtime-array-i32 result ID.
  uint32_t ptr_storage_buffer_runtime_array_i32_type_id;
  // Cached i32 zero constant used for storage-buffer runtime-array indexing.
  uint32_t i32_zero_constant_id;
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

static iree_status_t loom_spirv_emit_type_runtime_array_i32(
    loom_spirv_emit_state_t* state, uint32_t* out_type_id) {
  if (state->runtime_array_i32_type_id == 0) {
    uint32_t i32_type_id = 0;
    IREE_RETURN_IF_ERROR(loom_spirv_emit_type_i32(state, &i32_type_id));
    state->runtime_array_i32_type_id = loom_spirv_emit_allocate_id(state);
    const uint32_t operands[] = {state->runtime_array_i32_type_id, i32_type_id};
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
        LOOM_SPIRV_OP_TYPE_RUNTIME_ARRAY, operands, IREE_ARRAYSIZE(operands)));
  }
  *out_type_id = state->runtime_array_i32_type_id;
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

static iree_status_t loom_spirv_emit_type_ptr_storage_buffer_i32(
    loom_spirv_emit_state_t* state, uint32_t* out_type_id) {
  uint32_t i32_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_i32(state, &i32_type_id));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_pointer(
      state, LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER, i32_type_id,
      &state->ptr_storage_buffer_i32_type_id));
  *out_type_id = state->ptr_storage_buffer_i32_type_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_type_ptr_storage_buffer_runtime_array_i32(
    loom_spirv_emit_state_t* state, uint32_t* out_type_id) {
  uint32_t runtime_array_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_runtime_array_i32(state, &runtime_array_type_id));
  if (state->ptr_storage_buffer_runtime_array_i32_type_id == 0) {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_type_pointer(
        state, LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER,
        runtime_array_type_id,
        &state->ptr_storage_buffer_runtime_array_i32_type_id));
    const uint32_t decoration_operands[] = {
        runtime_array_type_id,
        LOOM_SPIRV_DECORATION_ARRAY_STRIDE,
        4,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_ANNOTATION),
        LOOM_SPIRV_OP_DECORATE, decoration_operands,
        IREE_ARRAYSIZE(decoration_operands)));
  }
  *out_type_id = state->ptr_storage_buffer_runtime_array_i32_type_id;
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

static iree_status_t loom_spirv_emit_function_parameter_type(
    loom_spirv_emit_state_t* state, loom_type_t type, uint32_t* out_type_id,
    loom_spirv_emitted_type_kind_t* out_type_kind) {
  if (!loom_low_type_is_register(type) ||
      loom_low_register_type_descriptor_set_stable_id(type) !=
          SPIRV_LOGICAL_CORE_DESCRIPTOR_SET_ID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V function parameter must be a "
                            "spirv.logical.core register type");
  }
  switch (loom_low_register_type_class_id(type)) {
    case SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID:
      *out_type_kind = LOOM_SPIRV_EMITTED_TYPE_I32;
      return loom_spirv_emit_type_i32(state, out_type_id);
    case SPIRV_LOGICAL_CORE_REG_CLASS_ID_OFFSET64:
      *out_type_kind = LOOM_SPIRV_EMITTED_TYPE_U64;
      return loom_spirv_emit_type_u64(state, out_type_id);
    case SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_STORAGE_BUFFER:
      *out_type_kind =
          LOOM_SPIRV_EMITTED_TYPE_PTR_STORAGE_BUFFER_RUNTIME_ARRAY_I32;
      return loom_spirv_emit_type_ptr_storage_buffer_runtime_array_i32(
          state, out_type_id);
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "SPIR-V function parameter register class %u is not "
                          "supported by binary emission",
                          loom_low_register_type_class_id(type));
}

static iree_status_t loom_spirv_emit_function_result_type(
    loom_spirv_emit_state_t* state, loom_type_t type, uint32_t* out_type_id) {
  loom_spirv_emitted_type_kind_t unused_type_kind =
      LOOM_SPIRV_EMITTED_TYPE_UNKNOWN;
  return loom_spirv_emit_function_parameter_type(state, type, out_type_id,
                                                 &unused_type_kind);
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
          LOOM_SPIRV_EMITTED_TYPE_PTR_STORAGE_BUFFER_RUNTIME_ARRAY_I32 ||
      byte_offset.type_kind != LOOM_SPIRV_EMITTED_TYPE_U64) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "OpPtrAccessChain.storage_buffer.byte_offset operand types do not "
        "match the selected SPIR-V descriptor");
  }

  uint32_t ptr_i32_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_ptr_storage_buffer_i32(state, &ptr_i32_type_id));
  uint32_t zero_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_i32_zero_constant(state, &zero_id));
  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
  const uint32_t operands[] = {
      ptr_i32_type_id, result_id, base.id, byte_offset.id, zero_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_PTR_ACCESS_CHAIN, operands, IREE_ARRAYSIZE(operands)));
  return loom_spirv_emit_define_value(
      state, loom_op_const_results(op)[0],
      (loom_spirv_value_ref_t){
          .id = result_id,
          .type_id = ptr_i32_type_id,
          .type_kind = LOOM_SPIRV_EMITTED_TYPE_PTR_STORAGE_BUFFER_I32,
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
  if (ptr.type_kind != LOOM_SPIRV_EMITTED_TYPE_PTR_STORAGE_BUFFER_I32) {
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
  if (ptr.type_kind != LOOM_SPIRV_EMITTED_TYPE_PTR_STORAGE_BUFFER_I32 ||
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
  if (op->operand_count == 0) {
    return loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
        LOOM_SPIRV_OP_RETURN, NULL, 0);
  }
  if (op->operand_count != 1) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "SPIR-V emission supports at most one low return "
                            "value");
  }
  loom_spirv_value_ref_t value = {0};
  IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_value(
      state, loom_op_const_operands(op)[0], &value));
  const uint32_t operands[] = {value.id};
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_RETURN_VALUE, operands, IREE_ARRAYSIZE(operands));
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
    loom_spirv_emit_state_t* state, const loom_block_t* entry_block,
    uint32_t* out_result_type_id, uint32_t* out_function_type_id) {
  uint32_t result_type_id = 0;
  if (state->function_op->result_count == 0) {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_type_void(state, &result_type_id));
  } else if (state->function_op->result_count == 1) {
    const loom_value_id_t result_value =
        loom_op_const_results(state->function_op)[0];
    IREE_RETURN_IF_ERROR(loom_spirv_emit_function_result_type(
        state, loom_module_value_type(state->module, result_value),
        &result_type_id));
  } else {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "SPIR-V emission supports at most one function "
                            "result");
  }

  uint32_t* function_type_operands = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, (iree_host_size_t)entry_block->arg_count + 2,
      sizeof(uint32_t), (void**)&function_type_operands));
  const uint32_t function_type_id = loom_spirv_emit_allocate_id(state);
  function_type_operands[0] = function_type_id;
  function_type_operands[1] = result_type_id;
  for (uint16_t i = 0; i < entry_block->arg_count; ++i) {
    uint32_t parameter_type_id = 0;
    loom_spirv_emitted_type_kind_t parameter_type_kind =
        LOOM_SPIRV_EMITTED_TYPE_UNKNOWN;
    const loom_value_id_t arg_id = loom_block_arg_id(entry_block, i);
    IREE_RETURN_IF_ERROR(loom_spirv_emit_function_parameter_type(
        state, loom_module_value_type(state->module, arg_id),
        &parameter_type_id, &parameter_type_kind));
    function_type_operands[i + 2] = parameter_type_id;
  }
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
      LOOM_SPIRV_OP_TYPE_FUNCTION, function_type_operands,
      (iree_host_size_t)entry_block->arg_count + 2));
  *out_result_type_id = result_type_id;
  *out_function_type_id = function_type_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_function_parameters(
    loom_spirv_emit_state_t* state, const loom_block_t* entry_block) {
  for (uint16_t i = 0; i < entry_block->arg_count; ++i) {
    const loom_value_id_t arg_id = loom_block_arg_id(entry_block, i);
    uint32_t parameter_type_id = 0;
    loom_spirv_emitted_type_kind_t parameter_type_kind =
        LOOM_SPIRV_EMITTED_TYPE_UNKNOWN;
    IREE_RETURN_IF_ERROR(loom_spirv_emit_function_parameter_type(
        state, loom_module_value_type(state->module, arg_id),
        &parameter_type_id, &parameter_type_kind));
    const uint32_t parameter_id = loom_spirv_emit_allocate_id(state);
    const uint32_t operands[] = {parameter_type_id, parameter_id};
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
        LOOM_SPIRV_OP_FUNCTION_PARAMETER, operands, IREE_ARRAYSIZE(operands)));
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_define_value(state, arg_id,
                                     (loom_spirv_value_ref_t){
                                         .id = parameter_id,
                                         .type_id = parameter_type_id,
                                         .type_kind = parameter_type_kind,
                                     },
                                     true));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_function_body(
    loom_spirv_emit_state_t* state, const loom_block_t* entry_block) {
  const uint32_t label_id = loom_spirv_emit_allocate_id(state);
  const uint32_t label_operands[] = {label_id};
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_LABEL, label_operands, IREE_ARRAYSIZE(label_operands)));
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
  IREE_RETURN_IF_ERROR(loom_spirv_emit_entry_point(state));

  uint32_t result_type_id = 0;
  uint32_t function_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_function_signature(
      state, entry_block, &result_type_id, &function_type_id));

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
  IREE_RETURN_IF_ERROR(loom_spirv_emit_function_parameters(state, entry_block));
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
