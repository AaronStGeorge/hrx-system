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
#include "loom/target/arch/spirv/packet_rows.h"
#include "loom/target/arch/spirv/structured_control_flow.h"
#include "loom/target/emit/spirv/binary_format.h"
#include "loom/target/emit/spirv/module_abi.h"
#include "loom/target/emit/spirv/module_instructions.h"
#include "loom/target/emit/spirv/module_storage.h"
#include "loom/target/emit/spirv/module_types.h"
#include "loom/target/emit/spirv/module_values.h"
#include "loom/target/types.h"

enum {
  LOOM_SPIRV_BUILTIN_VARIABLE_COUNT = 3,
};

typedef struct loom_spirv_phi_incoming_t {
  // Low branch payload value forwarded into a block argument.
  loom_value_id_t value_id;
  // SPIR-V label ID of the predecessor block.
  uint32_t predecessor_label_id;
} loom_spirv_phi_incoming_t;

typedef struct loom_spirv_block_plan_t {
  // SPIR-V label ID assigned to the low block.
  uint32_t label_id;
  // SPIR-V label ID of a structured selection merge, or zero.
  uint32_t selection_merge_label_id;
  // SPIR-V label ID of a structured loop merge, or zero.
  uint32_t loop_merge_label_id;
  // SPIR-V label ID of a structured loop continue block, or zero.
  uint32_t loop_continue_label_id;
  // Synthetic continue label emitted after this block's backedge, or zero.
  uint32_t synthetic_continue_label_id;
  // First incoming branch-payload record for this block.
  uint32_t incoming_start;
  // Number of incoming branch-payload records for this block.
  uint32_t incoming_count;
} loom_spirv_block_plan_t;

typedef struct loom_spirv_emit_state_t {
  // Module containing the emitted low function.
  loom_module_t* module;
  // Target-low function definition being emitted.
  loom_op_t* function_op;
  // Target-low function body being emitted.
  const loom_region_t* body;
  // Resolved target record and descriptor set for |function_op|.
  const loom_low_resolved_target_t* target;
  // Case/function scratch arena.
  iree_arena_allocator_t* scratch_arena;
  // Sectioned SPIR-V module builder.
  loom_spirv_module_builder_t* builder;
  // Function-local value domain for dense value tables.
  loom_local_value_domain_t value_domain;
  // Function-local Loom value to SPIR-V value-ref table.
  loom_spirv_module_value_table_t value_table;
  // Module string IDs for descriptor-set immediate rows.
  loom_string_id_t* immediate_name_ids;
  // Number of entries in |immediate_name_ids|.
  iree_host_size_t immediate_name_id_count;
  // SPIR-V type and constant emission cache.
  loom_spirv_type_context_t type_context;
  // SPIR-V ID assigned to the function.
  uint32_t function_id;
  // Selected ABI plan for entry materialization.
  loom_spirv_module_abi_plan_t abi_plan;
  // Function-local Workgroup storage materialization state.
  loom_spirv_module_workgroup_storage_state_t workgroup_storage;
  // Per-block function emission plan, indexed by region block index.
  loom_spirv_block_plan_t* block_plans;
  // Branch payload records consumed by block-argument OpPhi emission.
  loom_spirv_phi_incoming_t* phi_incomings;
  // Number of entries in |phi_incomings|.
  uint32_t phi_incoming_count;
  // Cached Input variables for workgroup/local/global invocation builtins.
  uint32_t builtin_variable_ids[LOOM_SPIRV_BUILTIN_VARIABLE_COUNT];
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

static iree_status_t loom_spirv_emit_define_value(
    loom_spirv_emit_state_t* state, loom_value_id_t value_id,
    loom_spirv_module_value_ref_t value_ref, bool emit_name) {
  loom_spirv_module_value_table_define(&state->value_table, value_id,
                                       value_ref);
  if (emit_name) {
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_value_name(state, value_id, value_ref.id));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_reserve_value_ref(
    loom_spirv_emit_state_t* state, loom_value_id_t value_id, uint32_t type_id,
    loom_spirv_value_type_t value_type, uint32_t* out_result_id) {
  *out_result_id = loom_spirv_module_value_table_reserve(
      &state->value_table, state->builder, value_id, type_id, value_type);
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_prepare_packet_result(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet, uint32_t type_id,
    loom_spirv_value_type_t value_type, uint32_t* out_result_id) {
  return loom_spirv_emit_reserve_value_ref(state,
                                           loom_op_const_results(packet->op)[0],
                                           type_id, value_type, out_result_id);
}

static iree_status_t loom_spirv_emit_define_packet_result(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet, uint32_t id,
    uint32_t type_id, loom_spirv_value_type_t value_type) {
  const loom_spirv_module_value_ref_t value_ref = {
      .id = id, .type_id = type_id, .value_type = value_type};
  return loom_spirv_emit_define_value(
      state, loom_op_const_results(packet->op)[0], value_ref, true);
}

static iree_status_t loom_spirv_emit_lookup_value(
    loom_spirv_emit_state_t* state, loom_value_id_t value_id,
    loom_spirv_module_value_ref_t* out_value_ref) {
  *out_value_ref =
      loom_spirv_module_value_table_lookup(&state->value_table, value_id);
  return iree_ok_status();
}

static loom_spirv_module_abi_context_t loom_spirv_emit_abi_context(
    loom_spirv_emit_state_t* state) {
  return (loom_spirv_module_abi_context_t){
      .module = state->module,
      .function_op = state->function_op,
      .target = state->target,
      .scratch_arena = state->scratch_arena,
      .builder = state->builder,
      .type_context = &state->type_context,
      .value_table = &state->value_table,
  };
}

static bool loom_spirv_emit_builtin_variable_info(
    uint32_t builtin, uint8_t* out_slot, iree_string_view_t* out_name) {
  switch (builtin) {
    case LOOM_SPIRV_BUILT_IN_WORKGROUP_ID:
      *out_slot = 0;
      *out_name = IREE_SV("workgroup_id");
      return true;
    case LOOM_SPIRV_BUILT_IN_LOCAL_INVOCATION_ID:
      *out_slot = 1;
      *out_name = IREE_SV("local_invocation_id");
      return true;
    case LOOM_SPIRV_BUILT_IN_GLOBAL_INVOCATION_ID:
      *out_slot = 2;
      *out_name = IREE_SV("global_invocation_id");
      return true;
  }
  return false;
}

static iree_status_t loom_spirv_emit_builtin_variable(
    loom_spirv_emit_state_t* state, uint32_t builtin,
    uint32_t* out_variable_id) {
  uint8_t slot = 0;
  iree_string_view_t name = iree_string_view_empty();
  if (!loom_spirv_emit_builtin_variable_info(builtin, &slot, &name)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "unknown SPIR-V builtin packet row %" PRIu32,
                            builtin);
  }
  if (state->builtin_variable_ids[slot] != 0) {
    *out_variable_id = state->builtin_variable_ids[slot];
    return iree_ok_status();
  }

  uint32_t u32_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_u32(&state->type_context, &u32_type_id));
  uint32_t vector_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_vector(
      &state->type_context, u32_type_id, 3, &vector_type_id));
  uint32_t pointer_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_pointer(
      &state->type_context, LOOM_SPIRV_STORAGE_CLASS_INPUT, vector_type_id,
      /*pointer_array_stride=*/0, &pointer_type_id));

  const uint32_t variable_id = loom_spirv_emit_allocate_id(state);
  const uint32_t decoration_operands[] = {
      variable_id,
      LOOM_SPIRV_DECORATION_BUILT_IN,
      builtin,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_ANNOTATION),
      LOOM_SPIRV_OP_DECORATE, decoration_operands,
      IREE_ARRAYSIZE(decoration_operands)));
  const uint32_t variable_operands[] = {
      pointer_type_id,
      variable_id,
      LOOM_SPIRV_STORAGE_CLASS_INPUT,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
      LOOM_SPIRV_OP_VARIABLE, variable_operands,
      IREE_ARRAYSIZE(variable_operands)));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_op_name(state, variable_id, name));
  state->builtin_variable_ids[slot] = variable_id;
  *out_variable_id = variable_id;
  return iree_ok_status();
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
  IREE_ASSERT(immediate != NULL);
  const uint32_t immediate_row =
      packet->descriptor->immediate_start + row->immediate_index;
  IREE_ASSERT_LT(immediate_row, state->immediate_name_id_count);
  const loom_string_id_t expected_name_id =
      state->immediate_name_ids[immediate_row];
  const loom_named_attr_slice_t attrs = loom_spirv_emit_packet_attrs(packet);
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (attr->name_id != expected_name_id) {
      continue;
    }
    IREE_ASSERT_EQ(attr->value.kind, LOOM_ATTR_I64);
    *out_value = loom_attr_as_i64(attr->value);
    return iree_ok_status();
  }
  if (iree_any_bit_set(immediate->flags,
                       LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE)) {
    *out_value = immediate->default_value;
    return iree_ok_status();
  }
  IREE_CHECK_UNREACHABLE("verified low descriptor immediate attribute");
  return iree_ok_status();
}

static void loom_spirv_emit_validate_packet_shape(
    const loom_op_t* op, const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  (void)packet;
  IREE_ASSERT_EQ(op->result_count, row->result_count);
  IREE_ASSERT_EQ(op->operand_count, row->operand_count);
}

static iree_status_t loom_spirv_emit_load_packet_operands(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row,
    loom_spirv_module_value_ref_t* out_operands) {
  const loom_value_id_t* operand_values = loom_op_const_operands(packet->op);
  for (uint8_t i = 0; i < row->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_value(state, operand_values[i],
                                                      &out_operands[i]));
    IREE_ASSERT(loom_spirv_value_type_equal(out_operands[i].value_type,
                                            row->operand_types[i]));
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
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      &state->type_context, row->result_type, &type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, type_id, row->result_type, &result_id));
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
  return loom_spirv_emit_define_packet_result(state, packet, result_id, type_id,
                                              row->result_type);
}

static iree_status_t loom_spirv_emit_binary_same_type_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[2] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  IREE_ASSERT_EQ(operands[0].type_id, operands[1].type_id);
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, operands[0].type_id, row->result_type, &result_id));
  const uint32_t instruction_operands[] = {
      operands[0].type_id,
      result_id,
      operands[0].id,
      operands[1].id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(
      state, packet, result_id, operands[0].type_id, row->result_type);
}

static iree_status_t loom_spirv_emit_unary_convert_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[1] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      &state->type_context, row->result_type, &result_type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, row->result_type, &result_id));
  const uint32_t instruction_operands[] = {
      result_type_id,
      result_id,
      operands[0].id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(state, packet, result_id,
                                              result_type_id, row->result_type);
}

static iree_status_t loom_spirv_emit_load_builtin_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  uint32_t variable_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_builtin_variable(state, row->builtin, &variable_id));
  uint32_t u32_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_u32(&state->type_context, &u32_type_id));
  uint32_t vector_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_vector(
      &state->type_context, u32_type_id, 3, &vector_type_id));

  uint32_t vector_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_unary_result(
      state->builder, LOOM_SPIRV_OP_LOAD, vector_type_id, variable_id,
      &vector_id));

  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      &state->type_context, row->result_type, &result_type_id));
  uint32_t result_id = 0;
  uint32_t component_id = 0;
  if (result_type_id == u32_type_id) {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
        state, packet, result_type_id, row->result_type, &result_id));
    component_id = result_id;
  } else {
    component_id = loom_spirv_emit_allocate_id(state);
  }
  const uint32_t extract_operands[] = {
      u32_type_id,
      component_id,
      vector_id,
      row->component_index,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_COMPOSITE_EXTRACT, extract_operands,
      IREE_ARRAYSIZE(extract_operands)));

  if (result_type_id != u32_type_id) {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
        state, packet, result_type_id, row->result_type, &result_id));
    const uint32_t bitcast_operands[] = {
        result_type_id,
        result_id,
        component_id,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
        LOOM_SPIRV_OP_BITCAST, bitcast_operands,
        IREE_ARRAYSIZE(bitcast_operands)));
  }
  return loom_spirv_emit_define_packet_result(state, packet, result_id,
                                              result_type_id, row->result_type);
}

static iree_status_t loom_spirv_emit_integer_mul_add_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[3] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  IREE_ASSERT_EQ(operands[0].type_id, operands[1].type_id);
  IREE_ASSERT_EQ(operands[0].type_id, operands[2].type_id);
  uint32_t product_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_binary_result(
      state->builder, LOOM_SPIRV_OP_I_MUL, operands[0].type_id, operands[0].id,
      operands[1].id, &product_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, operands[0].type_id, row->result_type, &result_id));
  const uint32_t instruction_operands[] = {
      operands[0].type_id,
      result_id,
      product_id,
      operands[2].id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_I_ADD, instruction_operands,
      IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(
      state, packet, result_id, operands[0].type_id, row->result_type);
}

static iree_status_t loom_spirv_emit_compare_same_type_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[2] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  IREE_ASSERT_EQ(operands[0].type_id, operands[1].type_id);
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      &state->type_context, row->result_type, &result_type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, row->result_type, &result_id));
  const uint32_t instruction_operands[] = {
      result_type_id,
      result_id,
      operands[0].id,
      operands[1].id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(state, packet, result_id,
                                              result_type_id, row->result_type);
}

static iree_status_t loom_spirv_emit_select_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[3] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  IREE_ASSERT_EQ(operands[1].type_id, operands[2].type_id);
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      &state->type_context, row->result_type, &result_type_id));
  IREE_ASSERT_EQ(result_type_id, operands[1].type_id);
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, row->result_type, &result_id));
  const uint32_t instruction_operands[] = {
      result_type_id, result_id, operands[0].id, operands[1].id, operands[2].id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(state, packet, result_id,
                                              result_type_id, row->result_type);
}

static iree_status_t loom_spirv_emit_typed_physical_storage_buffer_pointer(
    loom_spirv_emit_state_t* state, loom_spirv_module_value_ref_t address,
    loom_spirv_value_type_t pointer_type, uint32_t pointer_type_id,
    uint32_t* out_pointer_id) {
  if (address.value_type.value_class ==
      LOOM_SPIRV_VALUE_CLASS_PTR_PHYSICAL_STORAGE_BUFFER) {
    IREE_ASSERT(loom_spirv_value_type_equal(address.value_type, pointer_type));
    *out_pointer_id = address.id;
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(address.value_type.value_class,
                 LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS);
  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
  const uint32_t operands[] = {
      pointer_type_id,
      result_id,
      address.id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_CONVERT_U_TO_PTR, operands, IREE_ARRAYSIZE(operands)));
  *out_pointer_id = result_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_ptr_access_chain_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[2] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      &state->type_context, row->result_type, &result_type_id));
  uint32_t base_pointer_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_typed_physical_storage_buffer_pointer(
      state, operands[0], row->result_type, result_type_id, &base_pointer_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, row->result_type, &result_id));
  const uint32_t instruction_operands[] = {
      result_type_id,
      result_id,
      base_pointer_id,
      operands[1].id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(state, packet, result_id,
                                              result_type_id, row->result_type);
}

static iree_status_t loom_spirv_emit_access_chain_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[2] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      &state->type_context, row->result_type, &result_type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, row->result_type, &result_id));
  const uint32_t instruction_operands[] = {
      result_type_id,
      result_id,
      operands[0].id,
      operands[1].id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(state, packet, result_id,
                                              result_type_id, row->result_type);
}

static iree_status_t loom_spirv_emit_load_aligned_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[1] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      &state->type_context, row->result_type, &result_type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, row->result_type, &result_id));
  const uint32_t instruction_operands[] = {
      result_type_id,        result_id,
      operands[0].id,        LOOM_SPIRV_MEMORY_ACCESS_ALIGNED_MASK,
      row->memory_alignment,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(state, packet, result_id,
                                              result_type_id, row->result_type);
}

static iree_status_t loom_spirv_emit_store_aligned_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[2] = {0};
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

static iree_status_t loom_spirv_emit_cooperative_matrix_layout_operands(
    loom_spirv_emit_state_t* state, const loom_spirv_packet_row_t* row,
    uint32_t* out_layout_id, uint32_t* out_stride_id) {
  IREE_RETURN_IF_ERROR(loom_spirv_emit_u32_constant(
      &state->type_context, row->cooperative_matrix_layout, out_layout_id));
  return loom_spirv_emit_u32_constant(
      &state->type_context, row->cooperative_matrix_stride, out_stride_id);
}

static iree_status_t loom_spirv_emit_cooperative_matrix_load_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[1] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      &state->type_context, row->result_type, &result_type_id));
  uint32_t layout_id = 0;
  uint32_t stride_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_cooperative_matrix_layout_operands(
      state, row, &layout_id, &stride_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, row->result_type, &result_id));
  const uint32_t instruction_operands[] = {
      result_type_id,
      result_id,
      operands[0].id,
      layout_id,
      stride_id,
      LOOM_SPIRV_MEMORY_ACCESS_ALIGNED_MASK,
      row->memory_alignment,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(state, packet, result_id,
                                              result_type_id, row->result_type);
}

static iree_status_t loom_spirv_emit_cooperative_matrix_store_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[2] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t layout_id = 0;
  uint32_t stride_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_cooperative_matrix_layout_operands(
      state, row, &layout_id, &stride_id));
  const uint32_t instruction_operands[] = {
      operands[0].id,
      operands[1].id,
      layout_id,
      stride_id,
      LOOM_SPIRV_MEMORY_ACCESS_ALIGNED_MASK,
      row->memory_alignment,
  };
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands));
}

static iree_status_t loom_spirv_emit_cooperative_matrix_mul_add_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[3] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      &state->type_context, row->result_type, &result_type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, row->result_type, &result_id));
  uint32_t instruction_operands[6] = {
      result_type_id, result_id, operands[0].id, operands[1].id, operands[2].id,
  };
  uint8_t operand_count = 5;
  if (row->cooperative_matrix_operands != 0) {
    instruction_operands[operand_count++] = row->cooperative_matrix_operands;
  }
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, operand_count));
  return loom_spirv_emit_define_packet_result(state, packet, result_id,
                                              result_type_id, row->result_type);
}

static iree_status_t loom_spirv_emit_control_barrier_packet(
    loom_spirv_emit_state_t* state, const loom_spirv_packet_row_t* row) {
  uint32_t execution_scope_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_u32_constant(
      &state->type_context, row->execution_scope, &execution_scope_id));
  uint32_t memory_scope_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_u32_constant(
      &state->type_context, row->memory_scope, &memory_scope_id));
  uint32_t memory_semantics_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_u32_constant(
      &state->type_context, row->memory_semantics, &memory_semantics_id));
  const uint32_t instruction_operands[] = {
      execution_scope_id,
      memory_scope_id,
      memory_semantics_id,
  };
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands));
}

static iree_status_t loom_spirv_emit_copy(loom_spirv_emit_state_t* state,
                                          const loom_op_t* op) {
  loom_spirv_module_value_ref_t source = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_value(state, loom_low_copy_source(op), &source));
  return loom_spirv_emit_define_value(state, loom_low_copy_result(op), source,
                                      false);
}

static iree_status_t loom_spirv_emit_descriptor_packet(
    loom_spirv_emit_state_t* state, const loom_op_t* op,
    const loom_low_resolved_descriptor_packet_t* packet) {
  for (uint16_t i = 0; i < packet->descriptor->feature_mask_word_count; ++i) {
    const uint32_t feature_mask_row =
        packet->descriptor->feature_mask_word_start + i;
    const uint64_t feature_bits =
        state->target->descriptor_set->feature_mask_words[feature_mask_row];
    IREE_ASSERT(i == 0 || feature_bits == 0);
    loom_spirv_module_builder_require_feature_bits(
        state->builder, (loom_spirv_feature_bits_t)feature_bits);
  }

  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_descriptor_ordinal(state->target->descriptor_set,
                                                 packet->descriptor);
  const loom_spirv_packet_row_t* row =
      loom_spirv_packet_row_for_descriptor_ordinal(descriptor_ordinal);
  IREE_ASSERT(row != NULL);
  loom_spirv_emit_validate_packet_shape(op, packet, row);
  switch (row->form) {
    case LOOM_SPIRV_PACKET_FORM_INTEGER_CONSTANT:
      return loom_spirv_emit_integer_constant_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_BINARY_SAME_TYPE:
      return loom_spirv_emit_binary_same_type_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_UNARY_CONVERT:
      return loom_spirv_emit_unary_convert_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_LOAD_BUILTIN:
      return loom_spirv_emit_load_builtin_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_INTEGER_MUL_ADD:
      return loom_spirv_emit_integer_mul_add_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_COMPARE_SAME_TYPE:
      return loom_spirv_emit_compare_same_type_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_SELECT:
      return loom_spirv_emit_select_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_PTR_ACCESS_CHAIN:
      return loom_spirv_emit_ptr_access_chain_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_ACCESS_CHAIN:
      return loom_spirv_emit_access_chain_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_LOAD_ALIGNED:
      return loom_spirv_emit_load_aligned_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_STORE_ALIGNED:
      return loom_spirv_emit_store_aligned_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_COOPERATIVE_MATRIX_LOAD:
      return loom_spirv_emit_cooperative_matrix_load_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_COOPERATIVE_MATRIX_STORE:
      return loom_spirv_emit_cooperative_matrix_store_packet(state, packet,
                                                             row);
    case LOOM_SPIRV_PACKET_FORM_COOPERATIVE_MATRIX_MUL_ADD:
      return loom_spirv_emit_cooperative_matrix_mul_add_packet(state, packet,
                                                               row);
    case LOOM_SPIRV_PACKET_FORM_CONTROL_BARRIER:
      return loom_spirv_emit_control_barrier_packet(state, row);
    case LOOM_SPIRV_PACKET_FORM_UNSUPPORTED:
      break;
  }
  IREE_CHECK_UNREACHABLE("verified SPIR-V binary packet row");
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_low_op(loom_spirv_emit_state_t* state,
                                            const loom_op_t* op) {
  if (loom_low_resource_isa(op)) {
    IREE_ASSERT_EQ(state->abi_plan.kind,
                   LOOM_SPIRV_MODULE_ABI_PLAN_HAL_KERNEL_RAW_BDA);
    loom_spirv_module_abi_context_t context =
        loom_spirv_emit_abi_context(state);
    return loom_spirv_module_abi_materialize_resource(&context,
                                                      &state->abi_plan, op);
  }
  if (loom_low_storage_reserve_isa(op)) {
    return loom_spirv_module_workgroup_storage_emit_reserve(
        &state->value_domain, &state->workgroup_storage, op);
  }
  if (loom_low_storage_address_isa(op)) {
    loom_spirv_module_value_ref_t value_ref = {0};
    IREE_RETURN_IF_ERROR(loom_spirv_module_workgroup_storage_emit_address(
        &state->value_domain, &state->workgroup_storage, op,
        &state->type_context, state->builder, &value_ref));
    return loom_spirv_emit_define_value(
        state, loom_low_storage_address_result(op), value_ref, true);
  }
  if (loom_low_copy_isa(op)) {
    return loom_spirv_emit_copy(state, op);
  }

  loom_low_resolved_descriptor_packet_t packet = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_descriptor_packet(
      state->module, state->target, op, &packet));
  if (packet.kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
    IREE_CHECK_UNREACHABLE("verified SPIR-V structural op");
    return iree_ok_status();
  }
  IREE_ASSERT(packet.descriptor != NULL);
  return loom_spirv_emit_descriptor_packet(state, op, &packet);
}

static iree_status_t loom_spirv_emit_return(loom_spirv_emit_state_t* state,
                                            const loom_op_t* op) {
  loom_spirv_module_abi_context_t context = loom_spirv_emit_abi_context(state);
  IREE_RETURN_IF_ERROR(loom_spirv_module_abi_store_return_values(
      &context, &state->abi_plan, op));
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_RETURN, NULL, 0);
}

static uint16_t loom_spirv_emit_block_index(const loom_block_t* block) {
  return loom_block_region_index(block);
}

static uint32_t loom_spirv_emit_br_predecessor_label(
    loom_spirv_emit_state_t* state, const loom_block_t* block,
    const loom_op_t* terminator) {
  const uint16_t block_index = loom_spirv_emit_block_index(block);
  loom_spirv_block_plan_t* block_plan = &state->block_plans[block_index];
  if (block_plan->synthetic_continue_label_id != 0 &&
      loom_spirv_emit_block_index(loom_low_br_dest(terminator)) <=
          block_index) {
    return block_plan->synthetic_continue_label_id;
  }
  return block_plan->label_id;
}

static iree_status_t loom_spirv_emit_prepare_block_labels(
    loom_spirv_emit_state_t* state) {
  for (uint16_t i = 0; i < state->body->block_count; ++i) {
    state->block_plans[i].label_id = loom_spirv_emit_allocate_id(state);
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_count_phi_incomings(
    loom_spirv_emit_state_t* state) {
  uint32_t incoming_count = 0;
  const loom_block_t* block = NULL;
  loom_region_for_each_block(state->body, block) {
    const loom_op_t* terminator = block->last_op;
    if (!terminator || !loom_low_br_isa(terminator)) {
      continue;
    }
    const loom_block_t* dest = loom_low_br_dest(terminator);
    const uint16_t dest_index = loom_spirv_emit_block_index(dest);
    loom_value_slice_t args = loom_low_br_args(terminator);
    state->block_plans[dest_index].incoming_count += args.count;
    incoming_count += args.count;
  }
  state->phi_incoming_count = incoming_count;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_prepare_phi_incomings(
    loom_spirv_emit_state_t* state) {
  uint32_t incoming_offset = 0;
  for (uint16_t i = 0; i < state->body->block_count; ++i) {
    state->block_plans[i].incoming_start = incoming_offset;
    incoming_offset += state->block_plans[i].incoming_count;
    state->block_plans[i].incoming_count = 0;
  }
  if (state->phi_incoming_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, state->phi_incoming_count,
      sizeof(*state->phi_incomings), (void**)&state->phi_incomings));

  const loom_block_t* block = NULL;
  loom_region_for_each_block(state->body, block) {
    const loom_op_t* terminator = block->last_op;
    if (!terminator || !loom_low_br_isa(terminator)) {
      continue;
    }
    const loom_block_t* dest = loom_low_br_dest(terminator);
    const uint16_t dest_index = loom_spirv_emit_block_index(dest);
    loom_spirv_block_plan_t* dest_plan = &state->block_plans[dest_index];
    loom_value_slice_t args = loom_low_br_args(terminator);
    for (uint16_t i = 0; i < args.count; ++i) {
      const uint32_t incoming_index =
          dest_plan->incoming_start + dest_plan->incoming_count++;
      state->phi_incomings[incoming_index] = (loom_spirv_phi_incoming_t){
          .value_id = args.values[i],
          .predecessor_label_id =
              loom_spirv_emit_br_predecessor_label(state, block, terminator),
      };
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_prepare_structured_control_flow(
    loom_spirv_emit_state_t* state) {
  const loom_block_t* block = NULL;
  loom_region_for_each_block(state->body, block) {
    const loom_op_t* terminator = block->last_op;
    if (!terminator || !loom_low_cond_br_isa(terminator)) {
      continue;
    }
    const uint16_t block_index = loom_spirv_emit_block_index(block);
    loom_spirv_block_plan_t* block_plan = &state->block_plans[block_index];
    loom_spirv_low_loop_shape_t loop = {0};
    if (loom_spirv_low_loop_shape(terminator, &loop)) {
      const uint32_t continue_label_id = loom_spirv_emit_allocate_id(state);
      block_plan->loop_merge_label_id =
          state->block_plans[loom_spirv_emit_block_index(loop.merge_block)]
              .label_id;
      block_plan->loop_continue_label_id = continue_label_id;
      state->block_plans[loom_spirv_emit_block_index(loop.body_block)]
          .synthetic_continue_label_id = continue_label_id;
      continue;
    }
    const loom_block_t* merge_block = NULL;
    if (!loom_spirv_low_select_merge_block(terminator, &merge_block)) {
      IREE_CHECK_UNREACHABLE("verified SPIR-V structured conditional branch");
      return iree_ok_status();
    }
    block_plan->selection_merge_label_id =
        state->block_plans[loom_spirv_emit_block_index(merge_block)].label_id;
  }
  return iree_ok_status();
}

static bool loom_spirv_emit_value_ref_exists(loom_spirv_emit_state_t* state,
                                             loom_value_id_t value_id) {
  return loom_spirv_module_value_table_exists(&state->value_table, value_id);
}

static iree_status_t loom_spirv_emit_reserve_forward_value(
    loom_spirv_emit_state_t* state, loom_value_id_t value_id);

static iree_status_t loom_spirv_emit_reserve_packet_result_value(
    loom_spirv_emit_state_t* state, const loom_op_t* op) {
  loom_low_resolved_descriptor_packet_t packet = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_descriptor_packet(
      state->module, state->target, op, &packet));
  if (packet.kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE ||
      packet.descriptor == NULL) {
    IREE_CHECK_UNREACHABLE("verified SPIR-V forward phi value producer");
    return iree_ok_status();
  }
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_descriptor_ordinal(state->target->descriptor_set,
                                                 packet.descriptor);
  const loom_spirv_packet_row_t* row =
      loom_spirv_packet_row_for_descriptor_ordinal(descriptor_ordinal);
  if (row == NULL || row->result_count != 1) {
    IREE_CHECK_UNREACHABLE("verified SPIR-V forward phi packet row");
    return iree_ok_status();
  }
  loom_spirv_emit_validate_packet_shape(op, &packet, row);
  uint32_t type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      &state->type_context, row->result_type, &type_id));
  uint32_t result_id = 0;
  return loom_spirv_emit_prepare_packet_result(state, &packet, type_id,
                                               row->result_type, &result_id);
}

static iree_status_t loom_spirv_emit_reserve_copy_value(
    loom_spirv_emit_state_t* state, const loom_op_t* op) {
  const loom_value_id_t source_value_id = loom_low_copy_source(op);
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_reserve_forward_value(state, source_value_id));
  loom_spirv_module_value_ref_t source_ref = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_value(state, source_value_id, &source_ref));
  return loom_spirv_emit_define_value(state, loom_low_copy_result(op),
                                      source_ref, false);
}

static iree_status_t loom_spirv_emit_reserve_forward_value(
    loom_spirv_emit_state_t* state, loom_value_id_t value_id) {
  if (loom_spirv_emit_value_ref_exists(state, value_id)) {
    return iree_ok_status();
  }
  const loom_value_t* value = loom_module_value(state->module, value_id);
  if (loom_value_is_block_arg(value)) {
    return iree_ok_status();
  }
  const loom_op_t* op = loom_value_def_op(value);
  if (loom_low_copy_isa(op)) {
    return loom_spirv_emit_reserve_copy_value(state, op);
  }
  return loom_spirv_emit_reserve_packet_result_value(state, op);
}

static iree_status_t loom_spirv_emit_reserve_forward_phi_incomings(
    loom_spirv_emit_state_t* state) {
  const loom_block_t* block = NULL;
  loom_region_for_each_block(state->body, block) {
    const uint16_t block_index = loom_spirv_emit_block_index(block);
    const loom_spirv_block_plan_t* block_plan =
        &state->block_plans[block_index];
    if (block->arg_count == 0 || block_plan->incoming_count == 0) {
      continue;
    }
    for (uint32_t incoming_index = 0;
         incoming_index < block_plan->incoming_count; ++incoming_index) {
      const loom_spirv_phi_incoming_t* incoming =
          &state->phi_incomings[block_plan->incoming_start + incoming_index];
      if (loom_spirv_emit_value_ref_exists(state, incoming->value_id)) {
        continue;
      }
      const loom_value_t* value =
          loom_module_value(state->module, incoming->value_id);
      if (loom_value_is_block_arg(value)) {
        continue;
      }
      const loom_op_t* defining_op = loom_value_def_op(value);
      const uint16_t defining_block_index =
          loom_spirv_emit_block_index(defining_op->parent_block);
      if (defining_block_index >= block_index) {
        IREE_RETURN_IF_ERROR(
            loom_spirv_emit_reserve_forward_value(state, incoming->value_id));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_prepare_cfg_plan(
    loom_spirv_emit_state_t* state) {
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, state->body->block_count,
      sizeof(*state->block_plans), (void**)&state->block_plans));
  memset(state->block_plans, 0,
         state->body->block_count * sizeof(*state->block_plans));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_block_labels(state));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_structured_control_flow(state));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_count_phi_incomings(state));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_phi_incomings(state));
  return loom_spirv_emit_reserve_forward_phi_incomings(state);
}

static iree_status_t loom_spirv_emit_block_label(
    loom_spirv_emit_state_t* state, const loom_spirv_block_plan_t* block_plan) {
  const uint32_t operands[] = {block_plan->label_id};
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_LABEL, operands, IREE_ARRAYSIZE(operands));
}

static iree_status_t loom_spirv_emit_block_phis(
    loom_spirv_emit_state_t* state, const loom_block_t* block,
    const loom_spirv_block_plan_t* block_plan) {
  if (block->arg_count == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT_NE(block_plan->incoming_count, 0u);
  IREE_ASSERT_EQ(block_plan->incoming_count % block->arg_count, 0u);
  const uint32_t incoming_edge_count =
      block_plan->incoming_count / block->arg_count;
  for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
    const loom_value_id_t block_arg_id = loom_block_arg_id(block, arg_index);
    const loom_spirv_phi_incoming_t* first_incoming = NULL;
    for (uint32_t incoming_index = 0; incoming_index < incoming_edge_count;
         ++incoming_index) {
      const loom_spirv_phi_incoming_t* incoming =
          &state->phi_incomings[block_plan->incoming_start +
                                incoming_index * block->arg_count + arg_index];
      if (incoming->value_id != block_arg_id) {
        first_incoming = incoming;
        break;
      }
    }
    if (first_incoming == NULL) {
      IREE_CHECK_UNREACHABLE("verified SPIR-V block argument incoming value");
      return iree_ok_status();
    }
    loom_spirv_module_value_ref_t first_value_ref = {0};
    IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_value(
        state, first_incoming->value_id, &first_value_ref));
    const uint32_t result_id = loom_spirv_emit_allocate_id(state);
    const uint32_t operand_count = 2 + incoming_edge_count * 2;
    uint32_t* operands = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->scratch_arena, operand_count,
                                  sizeof(*operands), (void**)&operands));
    operands[0] = first_value_ref.type_id;
    operands[1] = result_id;
    for (uint32_t incoming_index = 0; incoming_index < incoming_edge_count;
         ++incoming_index) {
      const loom_spirv_phi_incoming_t* incoming =
          &state->phi_incomings[block_plan->incoming_start +
                                incoming_index * block->arg_count + arg_index];
      loom_spirv_module_value_ref_t value_ref = first_value_ref;
      if (incoming->value_id == block_arg_id) {
        value_ref.id = result_id;
      } else {
        IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_value(
            state, incoming->value_id, &value_ref));
      }
      IREE_ASSERT_EQ(value_ref.type_id, first_value_ref.type_id);
      IREE_ASSERT(loom_spirv_value_type_equal(value_ref.value_type,
                                              first_value_ref.value_type));
      operands[2 + incoming_index * 2] = value_ref.id;
      operands[3 + incoming_index * 2] = incoming->predecessor_label_id;
    }
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
        LOOM_SPIRV_OP_PHI, operands, operand_count));
    const loom_spirv_module_value_ref_t block_arg_ref = {
        .id = result_id,
        .type_id = first_value_ref.type_id,
        .value_type = first_value_ref.value_type,
    };
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_define_value(state, block_arg_id, block_arg_ref, true));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_br(loom_spirv_emit_state_t* state,
                                        const loom_op_t* op) {
  const loom_block_t* dest = loom_low_br_dest(op);
  const loom_spirv_block_plan_t* source_plan =
      &state->block_plans[loom_spirv_emit_block_index(op->parent_block)];
  const uint32_t dest_label_id =
      source_plan->synthetic_continue_label_id != 0 &&
              loom_spirv_emit_block_index(dest) <=
                  loom_spirv_emit_block_index(op->parent_block)
          ? source_plan->synthetic_continue_label_id
          : state->block_plans[loom_spirv_emit_block_index(dest)].label_id;
  const uint32_t operands[] = {
      dest_label_id,
  };
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_BRANCH, operands, IREE_ARRAYSIZE(operands));
}

static iree_status_t loom_spirv_emit_cond_br(loom_spirv_emit_state_t* state,
                                             const loom_op_t* op) {
  const uint16_t block_index = loom_spirv_emit_block_index(op->parent_block);
  const loom_spirv_block_plan_t* block_plan = &state->block_plans[block_index];
  if (block_plan->loop_merge_label_id != 0) {
    const uint32_t loop_merge_operands[] = {
        block_plan->loop_merge_label_id,
        block_plan->loop_continue_label_id,
        LOOM_SPIRV_LOOP_CONTROL_NONE,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
        LOOM_SPIRV_OP_LOOP_MERGE, loop_merge_operands,
        IREE_ARRAYSIZE(loop_merge_operands)));
  } else {
    const uint32_t selection_merge_operands[] = {
        block_plan->selection_merge_label_id,
        LOOM_SPIRV_SELECTION_CONTROL_NONE,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
        LOOM_SPIRV_OP_SELECTION_MERGE, selection_merge_operands,
        IREE_ARRAYSIZE(selection_merge_operands)));
  }

  loom_spirv_module_value_ref_t condition = {0};
  IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_value(
      state, loom_low_cond_br_condition(op), &condition));
  const uint32_t branch_operands[] = {
      condition.id,
      state
          ->block_plans[loom_spirv_emit_block_index(
              loom_low_cond_br_true_dest(op))]
          .label_id,
      state
          ->block_plans[loom_spirv_emit_block_index(
              loom_low_cond_br_false_dest(op))]
          .label_id,
  };
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_BRANCH_CONDITIONAL, branch_operands,
      IREE_ARRAYSIZE(branch_operands));
}

static iree_status_t loom_spirv_emit_synthetic_continue(
    loom_spirv_emit_state_t* state, const loom_block_t* block,
    const loom_spirv_block_plan_t* block_plan) {
  if (block_plan->synthetic_continue_label_id == 0) {
    return iree_ok_status();
  }
  const loom_op_t* terminator = block->last_op;
  const uint32_t label_operands[] = {block_plan->synthetic_continue_label_id};
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_LABEL, label_operands, IREE_ARRAYSIZE(label_operands)));
  const uint32_t branch_operands[] = {
      state
          ->block_plans[loom_spirv_emit_block_index(
              loom_low_br_dest(terminator))]
          .label_id,
  };
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_BRANCH, branch_operands, IREE_ARRAYSIZE(branch_operands));
}

static iree_status_t loom_spirv_emit_entry_point(
    loom_spirv_emit_state_t* state) {
  const uint32_t prefix_operands[] = {
      LOOM_SPIRV_EXECUTION_MODEL_GL_COMPUTE,
      state->function_id,
  };
  uint32_t interface_operands[LOOM_SPIRV_BUILTIN_VARIABLE_COUNT] = {0};
  iree_host_size_t interface_operand_count = 0;
  for (iree_host_size_t i = 0; i < LOOM_SPIRV_BUILTIN_VARIABLE_COUNT; ++i) {
    if (state->builtin_variable_ids[i] != 0) {
      interface_operands[interface_operand_count++] =
          state->builtin_variable_ids[i];
    }
  }
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_string_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_ENTRY_POINT),
      LOOM_SPIRV_OP_ENTRY_POINT, prefix_operands,
      IREE_ARRAYSIZE(prefix_operands), loom_spirv_emit_export_name(state),
      interface_operands, interface_operand_count));
  uint32_t workgroup_size_x = 1;
  uint32_t workgroup_size_y = 1;
  uint32_t workgroup_size_z = 1;
  if (loom_low_kernel_def_isa(state->function_op)) {
    const int64_t x = loom_low_kernel_def_workgroup_size_x(state->function_op);
    const int64_t y = loom_low_kernel_def_workgroup_size_y(state->function_op);
    const int64_t z = loom_low_kernel_def_workgroup_size_z(state->function_op);
    workgroup_size_x = x > 0 ? (uint32_t)x : 1u;
    workgroup_size_y = y > 0 ? (uint32_t)y : 1u;
    workgroup_size_z = z > 0 ? (uint32_t)z : 1u;
  }
  const uint32_t execution_mode_operands[] = {
      state->function_id, LOOM_SPIRV_EXECUTION_MODE_LOCAL_SIZE,
      workgroup_size_x,   workgroup_size_y,
      workgroup_size_z,
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
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_void(&state->type_context, &result_type_id));
  uint32_t function_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_function(
      &state->type_context, result_type_id, /*parameter_type_ids=*/NULL,
      /*parameter_count=*/0, &function_type_id));
  *out_result_type_id = result_type_id;
  *out_function_type_id = function_type_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_function_block(
    loom_spirv_emit_state_t* state, const loom_block_t* block) {
  const uint16_t block_index = loom_spirv_emit_block_index(block);
  const loom_spirv_block_plan_t* block_plan = &state->block_plans[block_index];
  IREE_RETURN_IF_ERROR(loom_spirv_emit_block_label(state, block_plan));
  if (block_index == 0) {
    loom_spirv_module_abi_context_t context =
        loom_spirv_emit_abi_context(state);
    IREE_RETURN_IF_ERROR(loom_spirv_module_abi_materialize_entry_args(
        &context, &state->abi_plan));
  } else {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_block_phis(state, block, block_plan));
  }
  for (const loom_op_t* op = block->first_op; op != NULL; op = op->next_op) {
    if (loom_low_return_isa(op)) {
      IREE_RETURN_IF_ERROR(loom_spirv_emit_return(state, op));
    } else if (loom_low_br_isa(op)) {
      IREE_RETURN_IF_ERROR(loom_spirv_emit_br(state, op));
    } else if (loom_low_cond_br_isa(op)) {
      IREE_RETURN_IF_ERROR(loom_spirv_emit_cond_br(state, op));
    } else {
      IREE_RETURN_IF_ERROR(loom_spirv_emit_low_op(state, op));
    }
  }
  return loom_spirv_emit_synthetic_continue(state, block, block_plan);
}

static iree_status_t loom_spirv_emit_function_body(
    loom_spirv_emit_state_t* state) {
  const loom_block_t* block = NULL;
  loom_region_for_each_block(state->body, block) {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_function_block(state, block));
  }
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_FUNCTION_END, NULL, 0);
}

static iree_status_t loom_spirv_emit_function(loom_spirv_emit_state_t* state) {
  const loom_region_t* body = loom_low_function_const_body(state->function_op);
  IREE_ASSERT(body != NULL);
  IREE_ASSERT_NE(body->block_count, 0u);
  state->body = body;
  const loom_block_t* entry_block = loom_region_const_entry_block(body);
  state->function_id = loom_spirv_emit_allocate_id(state);
  IREE_RETURN_IF_ERROR(loom_spirv_emit_op_name(
      state, state->function_id, loom_spirv_emit_function_name(state)));
  loom_spirv_module_abi_context_t context = loom_spirv_emit_abi_context(state);
  IREE_RETURN_IF_ERROR(loom_spirv_module_abi_build_plan(&context, entry_block,
                                                        &state->abi_plan));

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
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_cfg_plan(state));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_function_body(state));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_entry_point(state));
  return loom_spirv_module_abi_emit_metadata(&context, &state->abi_plan);
}

iree_status_t loom_spirv_emit_low_function_module(
    loom_module_t* module, loom_op_t* low_function_op,
    const loom_low_descriptor_registry_t* descriptor_registry,
    loom_target_selection_t target_selection,
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
      module, low_function_op, descriptor_registry, target_selection,
      diagnostic_emitter, &target));
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
  loom_spirv_emit_state_t state = {
      .module = module,
      .function_op = low_function_op,
      .target = &target,
      .scratch_arena = scratch_arena,
      .builder = &builder,
  };
  const loom_region_t* body = loom_low_function_const_body(low_function_op);
  iree_status_t status = iree_ok_status();
  if (body == NULL || body->block_count == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "SPIR-V emission requires a low function body");
  }
  if (iree_status_is_ok(status)) {
    status = loom_spirv_module_builder_initialize(&target.bundle_storage.bundle,
                                                  allocator, &builder);
  }
  if (iree_status_is_ok(status)) {
    loom_spirv_type_context_initialize(&builder, scratch_arena,
                                       &state.type_context);
  }
  if (iree_status_is_ok(status)) {
    status = loom_local_value_domain_acquire_for_region(
        module, body, scratch_arena, &state.value_domain);
  }
  if (iree_status_is_ok(status)) {
    status = loom_spirv_module_workgroup_storage_initialize(
        &state.value_domain, &state.workgroup_storage, scratch_arena);
  }
  if (iree_status_is_ok(status)) {
    status = loom_spirv_module_value_table_initialize(
        &state.value_domain, &state.value_table, scratch_arena);
  }
  if (iree_status_is_ok(status)) {
    loom_spirv_module_abi_context_t context =
        loom_spirv_emit_abi_context(&state);
    status = loom_spirv_module_abi_prepare_value_types(
        &context, loom_region_const_entry_block(body), &state.abi_plan);
  }
  if (iree_status_is_ok(status)) {
    status = loom_spirv_emit_prepare_immediate_name_ids(&state);
  }
  if (iree_status_is_ok(status)) {
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
