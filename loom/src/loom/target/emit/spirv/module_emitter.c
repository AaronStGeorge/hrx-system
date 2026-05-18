// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/module_emitter.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "loom/codegen/low/function.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/context.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/spirv/abi.h"
#include "loom/target/arch/spirv/descriptors.h"
#include "loom/target/arch/spirv/packet_rows.h"
#include "loom/target/arch/spirv/scalar_types.h"
#include "loom/target/arch/spirv/structured_control_flow.h"
#include "loom/target/emit/spirv/binary_format.h"
#include "loom/target/emit/spirv/module_types.h"
#include "loom/target/registers.h"
#include "loom/target/types.h"

enum {
  LOOM_SPIRV_BUILTIN_VARIABLE_COUNT = 3,
  LOOM_SPIRV_ACCESS_CHAIN_MAX_INDEX_COUNT = 8,
};

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
  // SPIR-V value type materialized for |value_id|.
  loom_spirv_value_type_t value_type;
  // Descriptor binding assigned to this shader descriptor slot.
  uint32_t binding;
  // Module-scope StorageBuffer variable ID for this descriptor slot.
  uint32_t variable_id;
  // Raw-BDA inline constant word offset for direct scalar arguments.
  uint16_t constant_word_offset;
  // Number of 32-bit inline constant words consumed by this argument.
  uint8_t constant_word_count;
} loom_spirv_abi_slot_t;

typedef enum loom_spirv_abi_plan_kind_e {
  // Descriptor-slot shader-entry ABI used by low.func.def fixtures.
  LOOM_SPIRV_ABI_PLAN_SHADER_ENTRY = 0,
  // Vulkan raw-BDA HAL kernel ABI used by low.kernel.def entries.
  LOOM_SPIRV_ABI_PLAN_HAL_KERNEL_RAW_BDA = 1,
} loom_spirv_abi_plan_kind_t;

typedef struct loom_spirv_bda_root_t {
  // Module-scope PushConstant root variable ID.
  uint32_t variable_id;
  // Function-local PhysicalStorageBuffer pointer to the binding table.
  uint32_t binding_table_pointer_id;
  // Function-local uint32_t binding-base value loaded from the root.
  uint32_t binding_base_id;
} loom_spirv_bda_root_t;

typedef struct loom_spirv_abi_plan_t {
  // ABI materialization algorithm selected for this function.
  loom_spirv_abi_plan_kind_t kind;
  // Materialization slots for low function entry block arguments.
  loom_spirv_abi_slot_t* args;
  // Number of entries in |args|.
  iree_host_size_t arg_count;
  // Materialization slots for low function results.
  loom_spirv_abi_slot_t* results;
  // Number of entries in |results|.
  iree_host_size_t result_count;
  // Number of HAL binding-table entries required by raw-BDA resources.
  uint16_t bda_binding_count;
  // Number of 32-bit HAL inline constants consumed by direct scalar arguments.
  uint16_t bda_constant_word_count;
  // Hidden raw-BDA PushConstant root state.
  loom_spirv_bda_root_t bda_root;
} loom_spirv_abi_plan_t;

typedef struct loom_spirv_value_ref_t {
  // SPIR-V result ID carrying this low SSA value.
  uint32_t id;
  // SPIR-V type ID assigned to |id|.
  uint32_t type_id;
  // Target-local value type used by packet emitters.
  loom_spirv_value_type_t value_type;
} loom_spirv_value_ref_t;

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
  // SPIR-V value refs indexed by function-local value ordinal.
  loom_spirv_value_ref_t* value_refs;
  // Module string IDs for descriptor-set immediate rows.
  loom_string_id_t* immediate_name_ids;
  // Number of entries in |immediate_name_ids|.
  iree_host_size_t immediate_name_id_count;
  // SPIR-V type and constant emission cache.
  loom_spirv_type_context_t type_context;
  // SPIR-V ID assigned to the function.
  uint32_t function_id;
  // Selected ABI plan for entry materialization.
  loom_spirv_abi_plan_t abi_plan;
  // Per-block function emission plan, indexed by region block index.
  loom_spirv_block_plan_t* block_plans;
  // Branch payload records consumed by block-argument OpPhi emission.
  loom_spirv_phi_incoming_t* phi_incomings;
  // Number of entries in |phi_incomings|.
  uint32_t phi_incoming_count;
  // Decoded ABI value types for entry block arguments.
  loom_spirv_value_type_t* abi_arg_value_types;
  // Number of decoded argument value types.
  iree_host_size_t abi_arg_value_type_count;
  // Decoded ABI value types for function results.
  loom_spirv_value_type_t* abi_result_value_types;
  // Number of decoded result value types.
  iree_host_size_t abi_result_value_type_count;
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

static bool loom_spirv_emit_low_register_is_id(loom_type_t type) {
  return loom_low_type_is_register(type) &&
         loom_low_register_type_descriptor_set_stable_id(type) ==
             SPIRV_LOGICAL_CORE_DESCRIPTOR_SET_ID &&
         loom_low_register_type_class_id(type) ==
             SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID;
}

static iree_status_t loom_spirv_emit_low_register_value_type(
    loom_type_t type, loom_spirv_value_type_t* out_value_type) {
  *out_value_type = (loom_spirv_value_type_t){0};
  if (!loom_low_type_is_register(type) ||
      loom_low_register_type_descriptor_set_stable_id(type) !=
          SPIRV_LOGICAL_CORE_DESCRIPTOR_SET_ID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V ABI value must be a "
                            "spirv.logical.core register type");
  }
  switch (loom_low_register_type_class_id(type)) {
    case SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "SPIR-V ABI value type metadata is required for "
                              "spirv.id register values");
    case SPIRV_LOGICAL_CORE_REG_CLASS_ID_OFFSET64:
      *out_value_type = (loom_spirv_value_type_t){
          .value_class = LOOM_SPIRV_VALUE_CLASS_OFFSET64,
          .scalar_type = LOOM_SPIRV_SCALAR_TYPE_U64,
      };
      return iree_ok_status();
    case SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_STORAGE_BUFFER:
      *out_value_type = (loom_spirv_value_type_t){
          .value_class = LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS,
      };
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

static iree_status_t loom_spirv_emit_define_packet_result(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet, uint32_t id,
    uint32_t type_id, loom_spirv_value_type_t value_type) {
  const loom_spirv_value_ref_t value_ref = {
      .id = id, .type_id = type_id, .value_type = value_type};
  return loom_spirv_emit_define_value(
      state, loom_op_const_results(packet->op)[0], value_ref, true);
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

static bool loom_spirv_emit_target_uses_raw_bda(
    const loom_spirv_emit_state_t* state) {
  return state->target->bundle_storage.export_plan.abi_kind ==
         LOOM_TARGET_ABI_HAL_KERNEL;
}

static bool loom_spirv_emit_type_is_named_opaque(
    const loom_spirv_emit_state_t* state, loom_type_t type,
    iree_string_view_t expected_name) {
  if (loom_type_kind(type) != LOOM_TYPE_DIALECT ||
      loom_type_dialect_param_count(type) != 0) {
    return false;
  }
  const iree_string_view_t actual_name = loom_spirv_emit_string_or_empty(
      state->module, loom_type_dialect_name_id(type));
  return iree_string_view_equal(actual_name, expected_name);
}

static loom_type_t loom_spirv_emit_module_type_attr(
    const loom_spirv_emit_state_t* state, loom_type_id_t type_id) {
  if (type_id >= state->module->types.count) {
    return loom_type_none();
  }
  return state->module->types.entries[type_id];
}

static iree_status_t loom_spirv_emit_low_resource_value_type(
    loom_spirv_emit_state_t* state, const loom_op_t* op,
    loom_spirv_value_type_t* out_value_type) {
  loom_type_t result_type =
      loom_module_value_type(state->module, loom_low_resource_result(op));
  return loom_spirv_emit_low_register_value_type(result_type, out_value_type);
}

static loom_named_attr_slice_t loom_spirv_emit_function_boundary_attrs(
    const loom_spirv_emit_state_t* state) {
  if (loom_low_kernel_def_isa(state->function_op)) {
    return loom_low_kernel_def_abi_layout(state->function_op);
  }
  return loom_low_func_def_abi_layout(state->function_op);
}

static const loom_attribute_t* loom_spirv_emit_find_boundary_attr(
    loom_named_attr_slice_t attrs, loom_string_id_t name_id) {
  if (name_id == LOOM_STRING_ID_INVALID) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    if (attrs.entries[i].name_id == name_id) {
      return &attrs.entries[i].value;
    }
  }
  return NULL;
}

static iree_status_t loom_spirv_emit_lookup_abi_value_type_array(
    const loom_spirv_emit_state_t* state, iree_string_view_t attr_name,
    iree_host_size_t expected_count, const loom_attribute_t** out_attr) {
  *out_attr = NULL;
  const loom_string_id_t name_id =
      loom_module_lookup_string(state->module, attr_name);
  const loom_attribute_t* attr = loom_spirv_emit_find_boundary_attr(
      loom_spirv_emit_function_boundary_attrs(state), name_id);
  if (attr == NULL) {
    return iree_ok_status();
  }
  if (attr->kind != LOOM_ATTR_I64_ARRAY || attr->count != expected_count ||
      (attr->count != 0 && attr->i64_array == NULL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V ABI value-type attr '%.*s' is malformed",
                            (int)attr_name.size, attr_name.data);
  }
  *out_attr = attr;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_prepare_abi_value_type(
    loom_type_t low_type, const loom_attribute_t* attr,
    iree_host_size_t attr_index, loom_spirv_value_type_t* out_value_type) {
  *out_value_type = (loom_spirv_value_type_t){0};
  if (!loom_spirv_emit_low_register_is_id(low_type)) {
    if (attr != NULL && attr->i64_array[attr_index] != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "SPIR-V ABI value-type attr annotates a non-spirv.id register");
    }
    return loom_spirv_emit_low_register_value_type(low_type, out_value_type);
  }
  if (attr == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V ABI is missing value-type metadata for a "
                            "spirv.id register");
  }
  if (!loom_spirv_abi_value_type_decode(attr->i64_array[attr_index],
                                        out_value_type) ||
      out_value_type->value_class == LOOM_SPIRV_VALUE_CLASS_UNKNOWN) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "SPIR-V ABI value-type attr has an invalid spirv.id payload code");
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_prepare_abi_arg_value_types(
    loom_spirv_emit_state_t* state, const loom_block_t* entry_block) {
  state->abi_arg_value_type_count = entry_block->arg_count;
  if (entry_block->arg_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->scratch_arena, entry_block->arg_count,
                                sizeof(*state->abi_arg_value_types),
                                (void**)&state->abi_arg_value_types));

  const loom_attribute_t* attr = NULL;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_abi_value_type_array(
      state, IREE_SV(LOOM_SPIRV_ABI_ARG_VALUE_TYPES_ATTR_NAME),
      entry_block->arg_count, &attr));
  for (uint16_t i = 0; i < entry_block->arg_count; ++i) {
    const loom_value_id_t value_id = loom_block_arg_id(entry_block, i);
    IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_abi_value_type(
        loom_module_value_type(state->module, value_id), attr, i,
        &state->abi_arg_value_types[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_prepare_abi_result_value_types(
    loom_spirv_emit_state_t* state) {
  state->abi_result_value_type_count = state->function_op->result_count;
  if (state->function_op->result_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, state->function_op->result_count,
      sizeof(*state->abi_result_value_types),
      (void**)&state->abi_result_value_types));

  const loom_attribute_t* attr = NULL;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_abi_value_type_array(
      state, IREE_SV(LOOM_SPIRV_ABI_RESULT_VALUE_TYPES_ATTR_NAME),
      state->function_op->result_count, &attr));
  const loom_value_id_t* results = loom_op_const_results(state->function_op);
  for (uint16_t i = 0; i < state->function_op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_abi_value_type(
        loom_module_value_type(state->module, results[i]), attr, i,
        &state->abi_result_value_types[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_prepare_abi_value_types(
    loom_spirv_emit_state_t* state, const loom_block_t* entry_block) {
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_prepare_abi_arg_value_types(state, entry_block));
  return loom_spirv_emit_prepare_abi_result_value_types(state);
}

static iree_status_t loom_spirv_emit_validate_bda_resource(
    loom_spirv_emit_state_t* state, const loom_op_t* op,
    uint16_t* out_binding_ordinal) {
  *out_binding_ordinal = 0;
  if (loom_low_resource_import_kind(op) !=
      LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BINDING) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V raw-BDA ABI requires low.resource "
                            "hal_binding imports");
  }
  const int64_t index = loom_low_resource_index(op);
  if (index < 0 || index >= UINT16_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V raw-BDA binding index %" PRId64
                            " is outside the 16-bit HAL binding-count range",
                            index);
  }

  const loom_type_t source_type = loom_spirv_emit_module_type_attr(
      state, loom_low_resource_source_type(op));
  if (!loom_spirv_emit_type_is_named_opaque(state, source_type,
                                            IREE_SV("hal.buffer"))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "SPIR-V raw-BDA ABI currently requires hal.buffer resources");
  }

  loom_spirv_value_type_t value_type = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_low_resource_value_type(state, op, &value_type));
  if (value_type.value_class != LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "SPIR-V raw-BDA ABI currently materializes HAL bindings as "
        "reg<spirv.ptr.storage_buffer>");
  }

  *out_binding_ordinal = (uint16_t)index;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_module_processed(
    loom_spirv_emit_state_t* state, iree_string_view_t value) {
  return loom_spirv_binary_write_string_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DEBUG),
      LOOM_SPIRV_OP_MODULE_PROCESSED, NULL, 0, value, NULL, 0);
}

static iree_status_t loom_spirv_emit_bda_module_processed_u32(
    loom_spirv_emit_state_t* state, const char* format, uint32_t value) {
  char text[64] = {0};
  const int length = snprintf(text, sizeof(text), format, value);
  if (length < 0 || (iree_host_size_t)length >= sizeof(text)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "failed to format SPIR-V raw-BDA metadata");
  }
  return loom_spirv_emit_module_processed(
      state, iree_make_string_view(text, (iree_host_size_t)length));
}

static iree_status_t loom_spirv_emit_bda_module_processed_u32_pair(
    loom_spirv_emit_state_t* state, const char* format, uint32_t lhs,
    uint32_t rhs) {
  char text[64] = {0};
  const int length = snprintf(text, sizeof(text), format, lhs, rhs);
  if (length < 0 || (iree_host_size_t)length >= sizeof(text)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "failed to format SPIR-V raw-BDA metadata");
  }
  return loom_spirv_emit_module_processed(
      state, iree_make_string_view(text, (iree_host_size_t)length));
}

static iree_status_t loom_spirv_emit_bda_metadata(
    loom_spirv_emit_state_t* state, uint16_t binding_count,
    uint16_t constant_word_count) {
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_module_processed(state, IREE_SV("iree.vulkan.bda.v1")));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_bda_module_processed_u32_pair(
      state, "iree.vulkan.bda.v1.root=%" PRIu32 ",%" PRIu32, 0,
      LOOM_SPIRV_BDA_ROOT_BYTE_LENGTH));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_bda_module_processed_u32(
      state, "iree.vulkan.bda.v1.constant_offset=%" PRIu32,
      LOOM_SPIRV_BDA_ROOT_CONSTANT_BYTE_OFFSET));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_bda_module_processed_u32(
      state, "iree.vulkan.bda.v1.constants=%" PRIu32, constant_word_count));
  return loom_spirv_emit_bda_module_processed_u32(
      state, "iree.vulkan.bda.v1.bindings=%" PRIu32, binding_count);
}

static iree_status_t loom_spirv_emit_declare_bda_root(
    loom_spirv_emit_state_t* state) {
  uint32_t root_pointer_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_ptr_push_constant_bda_root(
      &state->type_context, state->abi_plan.bda_constant_word_count,
      &root_pointer_type_id));
  const uint32_t root_variable_id = loom_spirv_emit_allocate_id(state);
  const uint32_t variable_operands[] = {
      root_pointer_type_id,
      root_variable_id,
      LOOM_SPIRV_STORAGE_CLASS_PUSH_CONSTANT,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
      LOOM_SPIRV_OP_VARIABLE, variable_operands,
      IREE_ARRAYSIZE(variable_operands)));
  state->abi_plan.bda_root.variable_id = root_variable_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_abi_slot_type_info(
    loom_spirv_emit_state_t* state, loom_spirv_value_type_t value_type,
    loom_spirv_abi_slot_type_info_t* out_type_info) {
  *out_type_info = (loom_spirv_abi_slot_type_info_t){0};
  switch (value_type.value_class) {
    case LOOM_SPIRV_VALUE_CLASS_SCALAR: {
      const loom_spirv_scalar_type_descriptor_t* descriptor =
          loom_spirv_scalar_type_descriptor(value_type.scalar_type);
      if (descriptor == NULL) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "SPIR-V ABI scalar type is unknown");
      }
      IREE_RETURN_IF_ERROR(loom_spirv_emit_type_scalar(
          &state->type_context, value_type.scalar_type,
          &out_type_info->value_type_id));
      out_type_info->field_type_id = out_type_info->value_type_id;
      IREE_RETURN_IF_ERROR(
          loom_spirv_emit_type_ptr_storage_buffer_descriptor_struct(
              &state->type_context, out_type_info->field_type_id,
              &out_type_info->variable_type_id));
      IREE_RETURN_IF_ERROR(
          loom_spirv_emit_type_ptr_storage_buffer_descriptor_field(
              &state->type_context, out_type_info->field_type_id,
              &out_type_info->field_pointer_type_id));
      out_type_info->field_alignment = iree_max(1u, descriptor->bit_width / 8);
      return iree_ok_status();
    }
    case LOOM_SPIRV_VALUE_CLASS_BOOL: {
      IREE_RETURN_IF_ERROR(loom_spirv_emit_type_bool(
          &state->type_context, &out_type_info->value_type_id));
      IREE_RETURN_IF_ERROR(loom_spirv_emit_type_i32(
          &state->type_context, &out_type_info->field_type_id));
      IREE_RETURN_IF_ERROR(
          loom_spirv_emit_type_ptr_storage_buffer_descriptor_struct(
              &state->type_context, out_type_info->field_type_id,
              &out_type_info->variable_type_id));
      IREE_RETURN_IF_ERROR(
          loom_spirv_emit_type_ptr_storage_buffer_descriptor_field(
              &state->type_context, out_type_info->field_type_id,
              &out_type_info->field_pointer_type_id));
      out_type_info->field_alignment = 4;
      return iree_ok_status();
    }
    case LOOM_SPIRV_VALUE_CLASS_OFFSET64:
    case LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS: {
      IREE_RETURN_IF_ERROR(loom_spirv_emit_type_u64(
          &state->type_context, &out_type_info->value_type_id));
      out_type_info->field_type_id = out_type_info->value_type_id;
      IREE_RETURN_IF_ERROR(
          loom_spirv_emit_type_ptr_storage_buffer_descriptor_struct(
              &state->type_context, out_type_info->field_type_id,
              &out_type_info->variable_type_id));
      IREE_RETURN_IF_ERROR(
          loom_spirv_emit_type_ptr_storage_buffer_descriptor_field(
              &state->type_context, out_type_info->field_type_id,
              &out_type_info->field_pointer_type_id));
      out_type_info->field_alignment = 8;
      return iree_ok_status();
    }
    case LOOM_SPIRV_VALUE_CLASS_PTR_PHYSICAL_STORAGE_BUFFER:
    case LOOM_SPIRV_VALUE_CLASS_COOPERATIVE_MATRIX:
    case LOOM_SPIRV_VALUE_CLASS_UNKNOWN:
      break;
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "SPIR-V shader-entry ABI does not support value "
                          "class %u",
                          (uint32_t)value_type.value_class);
}

static iree_status_t loom_spirv_emit_declare_abi_slot_variable(
    loom_spirv_emit_state_t* state, const loom_spirv_abi_slot_t* slot) {
  loom_spirv_abi_slot_type_info_t type_info = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_abi_slot_type_info(state, slot->value_type, &type_info));

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

static iree_status_t loom_spirv_emit_abi_slot_constant_word_count(
    loom_spirv_value_type_t value_type, uint8_t* out_word_count) {
  *out_word_count = 0;
  switch (value_type.value_class) {
    case LOOM_SPIRV_VALUE_CLASS_BOOL:
      *out_word_count = 1;
      return iree_ok_status();
    case LOOM_SPIRV_VALUE_CLASS_SCALAR: {
      const loom_spirv_scalar_type_descriptor_t* descriptor =
          loom_spirv_scalar_type_descriptor(value_type.scalar_type);
      if (descriptor == NULL) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "SPIR-V raw-BDA ABI scalar type is unknown");
      }
      if (descriptor->bit_width <= 32) {
        *out_word_count = 1;
        return iree_ok_status();
      }
      if (descriptor->bit_width == 64) {
        *out_word_count = 2;
        return iree_ok_status();
      }
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "SPIR-V raw-BDA ABI scalar type has an unsupported bit width");
    }
    case LOOM_SPIRV_VALUE_CLASS_OFFSET64:
      *out_word_count = 2;
      return iree_ok_status();
    case LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS:
    case LOOM_SPIRV_VALUE_CLASS_PTR_PHYSICAL_STORAGE_BUFFER:
    case LOOM_SPIRV_VALUE_CLASS_COOPERATIVE_MATRIX:
    case LOOM_SPIRV_VALUE_CLASS_UNKNOWN:
      break;
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "SPIR-V raw-BDA HAL ABI direct arguments must be scalar registers");
}

static iree_status_t loom_spirv_emit_build_shader_entry_abi_plan(
    loom_spirv_emit_state_t* state, const loom_block_t* entry_block) {
  const iree_host_size_t arg_count = entry_block->arg_count;
  const iree_host_size_t result_count = state->function_op->result_count;
  const iree_host_size_t descriptor_slot_count = arg_count + result_count;

  loom_spirv_abi_plan_t plan = {
      .kind = LOOM_SPIRV_ABI_PLAN_SHADER_ENTRY,
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
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_i32_constant(&state->type_context, 0, &zero_id));
  }

  uint32_t binding = 0;
  for (iree_host_size_t i = 0; i < arg_count; ++i) {
    loom_spirv_abi_slot_t* slot = &plan.args[i];
    slot->value_id = loom_block_arg_id(entry_block, (uint16_t)i);
    slot->binding = binding++;
    slot->variable_id = loom_spirv_emit_allocate_id(state);
    IREE_ASSERT_EQ(state->abi_arg_value_type_count, arg_count);
    slot->value_type = state->abi_arg_value_types[i];
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_declare_abi_slot_variable(state, slot));
  }
  const loom_value_id_t* results = loom_op_const_results(state->function_op);
  for (iree_host_size_t i = 0; i < result_count; ++i) {
    loom_spirv_abi_slot_t* slot = &plan.results[i];
    slot->value_id = results[i];
    slot->binding = binding++;
    slot->variable_id = loom_spirv_emit_allocate_id(state);
    IREE_ASSERT_EQ(state->abi_result_value_type_count, result_count);
    slot->value_type = state->abi_result_value_types[i];
    if (slot->value_type.value_class != LOOM_SPIRV_VALUE_CLASS_SCALAR &&
        slot->value_type.value_class != LOOM_SPIRV_VALUE_CLASS_BOOL &&
        slot->value_type.value_class != LOOM_SPIRV_VALUE_CLASS_OFFSET64) {
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

static iree_status_t loom_spirv_emit_build_raw_bda_abi_plan(
    loom_spirv_emit_state_t* state, const loom_block_t* entry_block) {
  if (state->function_op->result_count != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V raw-BDA HAL kernels cannot return values");
  }

  loom_spirv_abi_slot_t* args = NULL;
  if (entry_block->arg_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->scratch_arena, entry_block->arg_count,
                                  sizeof(*args), (void**)&args));
  }
  uint16_t constant_word_count = 0;
  for (uint16_t i = 0; i < entry_block->arg_count; ++i) {
    loom_spirv_abi_slot_t* slot = &args[i];
    slot->value_id = loom_block_arg_id(entry_block, i);
    IREE_ASSERT_EQ(state->abi_arg_value_type_count, entry_block->arg_count);
    slot->value_type = state->abi_arg_value_types[i];
    switch (slot->value_type.value_class) {
      case LOOM_SPIRV_VALUE_CLASS_SCALAR:
      case LOOM_SPIRV_VALUE_CLASS_BOOL:
      case LOOM_SPIRV_VALUE_CLASS_OFFSET64: {
        IREE_RETURN_IF_ERROR(loom_spirv_emit_abi_slot_constant_word_count(
            slot->value_type, &slot->constant_word_count));
        break;
      }
      case LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS:
      case LOOM_SPIRV_VALUE_CLASS_PTR_PHYSICAL_STORAGE_BUFFER:
      case LOOM_SPIRV_VALUE_CLASS_COOPERATIVE_MATRIX:
      case LOOM_SPIRV_VALUE_CLASS_UNKNOWN:
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "SPIR-V raw-BDA HAL ABI direct arguments must be scalar "
            "registers");
    }
    if (constant_word_count >
        (uint16_t)(UINT16_MAX - slot->constant_word_count)) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "SPIR-V raw-BDA HAL ABI direct constant count exceeds 16-bit range");
    }
    slot->constant_word_offset = constant_word_count;
    constant_word_count =
        (uint16_t)(constant_word_count + slot->constant_word_count);
  }

  uint16_t binding_count = 0;
  const loom_op_t* op = entry_block->first_op;
  while (op != NULL &&
         (loom_low_resource_isa(op) || loom_low_live_in_isa(op))) {
    if (loom_low_resource_isa(op)) {
      uint16_t binding_ordinal = 0;
      IREE_RETURN_IF_ERROR(
          loom_spirv_emit_validate_bda_resource(state, op, &binding_ordinal));
      binding_count = iree_max(binding_count, (uint16_t)(binding_ordinal + 1));
    }
    op = op->next_op;
  }

  state->abi_plan = (loom_spirv_abi_plan_t){
      .kind = LOOM_SPIRV_ABI_PLAN_HAL_KERNEL_RAW_BDA,
      .args = args,
      .arg_count = entry_block->arg_count,
      .bda_binding_count = binding_count,
      .bda_constant_word_count = constant_word_count,
  };
  return loom_spirv_emit_declare_bda_root(state);
}

static iree_status_t loom_spirv_emit_build_abi_plan(
    loom_spirv_emit_state_t* state, const loom_block_t* entry_block) {
  if (loom_spirv_emit_target_uses_raw_bda(state)) {
    return loom_spirv_emit_build_raw_bda_abi_plan(state, entry_block);
  }
  return loom_spirv_emit_build_shader_entry_abi_plan(state, entry_block);
}

static iree_status_t loom_spirv_emit_abi_slot_field_pointer(
    loom_spirv_emit_state_t* state, const loom_spirv_abi_slot_t* slot,
    uint32_t field_pointer_type_id, uint32_t* out_field_pointer_id) {
  uint32_t zero_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_i32_constant(&state->type_context, 0, &zero_id));
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
      .value_type = {0},
  };
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_materialize_abi_arg(
    loom_spirv_emit_state_t* state, const loom_spirv_abi_slot_t* slot) {
  loom_spirv_abi_slot_type_info_t type_info = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_abi_slot_type_info(state, slot->value_type, &type_info));
  loom_spirv_value_ref_t field_ref = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_abi_slot_field(state, slot, &type_info, &field_ref));
  if (slot->value_type.value_class == LOOM_SPIRV_VALUE_CLASS_BOOL) {
    uint32_t zero_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_i32_constant(&state->type_context, 0, &zero_id));
    const uint32_t result_id = loom_spirv_emit_allocate_id(state);
    const uint32_t operands[] = {
        type_info.value_type_id,
        result_id,
        field_ref.id,
        zero_id,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
        LOOM_SPIRV_OP_I_NOT_EQUAL, operands, IREE_ARRAYSIZE(operands)));
    return loom_spirv_emit_define_value(state, slot->value_id,
                                        (loom_spirv_value_ref_t){
                                            .id = result_id,
                                            .type_id = type_info.value_type_id,
                                            .value_type = slot->value_type,
                                        },
                                        true);
  }
  field_ref.value_type = slot->value_type;
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

static iree_status_t loom_spirv_emit_access_chain(
    loom_spirv_emit_state_t* state, uint32_t result_type_id,
    uint32_t base_pointer_id, const uint32_t* index_ids, uint8_t index_count,
    uint32_t* out_result_id) {
  if (index_count > LOOM_SPIRV_ACCESS_CHAIN_MAX_INDEX_COUNT) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "SPIR-V access chain exceeds the emitter operand "
                            "capacity");
  }
  uint32_t operands[3 + LOOM_SPIRV_ACCESS_CHAIN_MAX_INDEX_COUNT] = {0};
  operands[0] = result_type_id;
  operands[1] = loom_spirv_emit_allocate_id(state);
  operands[2] = base_pointer_id;
  for (uint8_t i = 0; i < index_count; ++i) {
    operands[3 + i] = index_ids[i];
  }
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_ACCESS_CHAIN, operands, 3 + index_count));
  *out_result_id = operands[1];
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_load_aligned(
    loom_spirv_emit_state_t* state, uint32_t result_type_id,
    uint32_t pointer_id, uint32_t alignment, uint32_t* out_result_id) {
  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
  const uint32_t operands[] = {
      result_type_id, result_id,
      pointer_id,     LOOM_SPIRV_MEMORY_ACCESS_ALIGNED_MASK,
      alignment,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_LOAD, operands, IREE_ARRAYSIZE(operands)));
  *out_result_id = result_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_unary_result(
    loom_spirv_emit_state_t* state, uint32_t opcode, uint32_t result_type_id,
    uint32_t operand_id, uint32_t* out_result_id) {
  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
  const uint32_t operands[] = {
      result_type_id,
      result_id,
      operand_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      opcode, operands, IREE_ARRAYSIZE(operands)));
  *out_result_id = result_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_binary_result(
    loom_spirv_emit_state_t* state, uint32_t opcode, uint32_t result_type_id,
    uint32_t lhs_id, uint32_t rhs_id, uint32_t* out_result_id) {
  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
  const uint32_t operands[] = {
      result_type_id,
      result_id,
      lhs_id,
      rhs_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      opcode, operands, IREE_ARRAYSIZE(operands)));
  *out_result_id = result_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_materialize_bda_constant_word(
    loom_spirv_emit_state_t* state, uint16_t word_offset,
    uint32_t* out_word_id) {
  uint32_t u32_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_u32(&state->type_context, &u32_type_id));
  uint32_t u32_pointer_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_ptr_push_constant_u32(
      &state->type_context, &u32_pointer_type_id));
  uint32_t constants_member_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_i32_constant(
      &state->type_context, LOOM_SPIRV_BDA_ROOT_CONSTANT_MEMBER_INDEX,
      &constants_member_id));
  uint32_t word_index_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_i32_constant(
      &state->type_context, word_offset, &word_index_id));
  const uint32_t index_ids[] = {
      constants_member_id,
      word_index_id,
  };
  uint32_t word_pointer_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_access_chain(
      state, u32_pointer_type_id, state->abi_plan.bda_root.variable_id,
      index_ids, IREE_ARRAYSIZE(index_ids), &word_pointer_id));
  return loom_spirv_emit_load_aligned(state, u32_type_id, word_pointer_id, 4,
                                      out_word_id);
}

static iree_status_t loom_spirv_emit_materialize_bda_u64_bits(
    loom_spirv_emit_state_t* state, const loom_spirv_abi_slot_t* slot,
    uint32_t* out_u64_id) {
  uint32_t low_word_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_materialize_bda_constant_word(
      state, slot->constant_word_offset, &low_word_id));
  uint32_t high_word_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_materialize_bda_constant_word(
      state, (uint16_t)(slot->constant_word_offset + 1), &high_word_id));

  uint32_t u64_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_u64(&state->type_context, &u64_type_id));
  uint32_t low_u64_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_unary_result(
      state, LOOM_SPIRV_OP_U_CONVERT, u64_type_id, low_word_id, &low_u64_id));
  uint32_t high_u64_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_unary_result(
      state, LOOM_SPIRV_OP_U_CONVERT, u64_type_id, high_word_id, &high_u64_id));
  uint32_t shift_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_u64_constant(&state->type_context, 32, &shift_id));
  uint32_t shifted_high_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_binary_result(
      state, LOOM_SPIRV_OP_SHIFT_LEFT_LOGICAL, u64_type_id, high_u64_id,
      shift_id, &shifted_high_id));
  return loom_spirv_emit_binary_result(state, LOOM_SPIRV_OP_BITWISE_OR,
                                       u64_type_id, shifted_high_id, low_u64_id,
                                       out_u64_id);
}

static iree_status_t loom_spirv_emit_bitcast_if_needed(
    loom_spirv_emit_state_t* state, uint32_t result_type_id,
    uint32_t operand_type_id, uint32_t operand_id, uint32_t* out_result_id) {
  if (result_type_id == operand_type_id) {
    *out_result_id = operand_id;
    return iree_ok_status();
  }
  return loom_spirv_emit_unary_result(
      state, LOOM_SPIRV_OP_BITCAST, result_type_id, operand_id, out_result_id);
}

static iree_status_t loom_spirv_emit_materialize_bda_scalar_arg(
    loom_spirv_emit_state_t* state, const loom_spirv_abi_slot_t* slot) {
  const loom_spirv_scalar_type_descriptor_t* descriptor =
      loom_spirv_scalar_type_descriptor(slot->value_type.scalar_type);
  if (descriptor == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V raw-BDA ABI scalar type is unknown");
  }
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_scalar(
      &state->type_context, slot->value_type.scalar_type, &result_type_id));

  uint32_t bits_type_id = 0;
  uint32_t bits_id = 0;
  if (descriptor->bit_width < 32) {
    uint32_t word_id = 0;
    IREE_RETURN_IF_ERROR(loom_spirv_emit_materialize_bda_constant_word(
        state, slot->constant_word_offset, &word_id));
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_type_int(&state->type_context, descriptor->bit_width,
                                 /*signedness=*/0, &bits_type_id));
    IREE_RETURN_IF_ERROR(loom_spirv_emit_unary_result(
        state, LOOM_SPIRV_OP_U_CONVERT, bits_type_id, word_id, &bits_id));
  } else if (descriptor->bit_width == 32) {
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_type_u32(&state->type_context, &bits_type_id));
    IREE_RETURN_IF_ERROR(loom_spirv_emit_materialize_bda_constant_word(
        state, slot->constant_word_offset, &bits_id));
  } else if (descriptor->bit_width == 64) {
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_type_u64(&state->type_context, &bits_type_id));
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_materialize_bda_u64_bits(state, slot, &bits_id));
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "SPIR-V raw-BDA ABI scalar type has an unsupported bit width");
  }

  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_bitcast_if_needed(
      state, result_type_id, bits_type_id, bits_id, &result_id));
  return loom_spirv_emit_define_value(state, slot->value_id,
                                      (loom_spirv_value_ref_t){
                                          .id = result_id,
                                          .type_id = result_type_id,
                                          .value_type = slot->value_type,
                                      },
                                      true);
}

static iree_status_t loom_spirv_emit_materialize_bda_bool_arg(
    loom_spirv_emit_state_t* state, const loom_spirv_abi_slot_t* slot) {
  uint32_t word_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_materialize_bda_constant_word(
      state, slot->constant_word_offset, &word_id));
  uint32_t zero_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_i32_constant(&state->type_context, 0, &zero_id));
  uint32_t bool_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_bool(&state->type_context, &bool_type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_binary_result(
      state, LOOM_SPIRV_OP_I_NOT_EQUAL, bool_type_id, word_id, zero_id,
      &result_id));
  return loom_spirv_emit_define_value(state, slot->value_id,
                                      (loom_spirv_value_ref_t){
                                          .id = result_id,
                                          .type_id = bool_type_id,
                                          .value_type = slot->value_type,
                                      },
                                      true);
}

static iree_status_t loom_spirv_emit_materialize_bda_u64_arg(
    loom_spirv_emit_state_t* state, const loom_spirv_abi_slot_t* slot) {
  uint32_t u64_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_u64(&state->type_context, &u64_type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_materialize_bda_u64_bits(state, slot, &result_id));
  return loom_spirv_emit_define_value(
      state, slot->value_id,
      (loom_spirv_value_ref_t){
          .id = result_id,
          .type_id = u64_type_id,
          .value_type =
              {
                  .value_class = LOOM_SPIRV_VALUE_CLASS_OFFSET64,
                  .scalar_type = LOOM_SPIRV_SCALAR_TYPE_U64,
              },
      },
      true);
}

static iree_status_t loom_spirv_emit_materialize_bda_arg(
    loom_spirv_emit_state_t* state, const loom_spirv_abi_slot_t* slot) {
  switch (slot->value_type.value_class) {
    case LOOM_SPIRV_VALUE_CLASS_SCALAR:
      return loom_spirv_emit_materialize_bda_scalar_arg(state, slot);
    case LOOM_SPIRV_VALUE_CLASS_BOOL:
      return loom_spirv_emit_materialize_bda_bool_arg(state, slot);
    case LOOM_SPIRV_VALUE_CLASS_OFFSET64:
      return loom_spirv_emit_materialize_bda_u64_arg(state, slot);
    case LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS:
    case LOOM_SPIRV_VALUE_CLASS_PTR_PHYSICAL_STORAGE_BUFFER:
    case LOOM_SPIRV_VALUE_CLASS_COOPERATIVE_MATRIX:
    case LOOM_SPIRV_VALUE_CLASS_UNKNOWN:
      break;
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "SPIR-V raw-BDA HAL ABI direct argument has unsupported value kind");
}

static iree_status_t loom_spirv_emit_materialize_bda_args(
    loom_spirv_emit_state_t* state) {
  for (iree_host_size_t i = 0; i < state->abi_plan.arg_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_materialize_bda_arg(state, &state->abi_plan.args[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_materialize_bda_root(
    loom_spirv_emit_state_t* state) {
  if (state->abi_plan.bda_root.binding_table_pointer_id != 0) {
    return iree_ok_status();
  }

  uint32_t u64_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_u64(&state->type_context, &u64_type_id));
  uint32_t u32_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_u32(&state->type_context, &u32_type_id));
  uint32_t root_u64_pointer_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_ptr_push_constant_u64(
      &state->type_context, &root_u64_pointer_type_id));
  uint32_t root_u32_pointer_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_ptr_push_constant_u32(
      &state->type_context, &root_u32_pointer_type_id));
  uint32_t address_table_pointer_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_ptr_physical_storage_buffer_bda_address_table(
          &state->type_context, &address_table_pointer_type_id));

  uint32_t binding_table_address_index_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_i32_constant(
      &state->type_context, 0, &binding_table_address_index_id));
  uint32_t binding_base_index_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_i32_constant(&state->type_context, 2,
                                                    &binding_base_index_id));

  uint32_t binding_table_address_pointer_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_access_chain(
      state, root_u64_pointer_type_id, state->abi_plan.bda_root.variable_id,
      &binding_table_address_index_id, 1, &binding_table_address_pointer_id));
  uint32_t binding_table_address_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_load_aligned(
      state, u64_type_id, binding_table_address_pointer_id, 8,
      &binding_table_address_id));

  const uint32_t convert_operands[] = {
      address_table_pointer_type_id,
      loom_spirv_emit_allocate_id(state),
      binding_table_address_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_CONVERT_U_TO_PTR, convert_operands,
      IREE_ARRAYSIZE(convert_operands)));

  uint32_t binding_base_pointer_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_access_chain(
      state, root_u32_pointer_type_id, state->abi_plan.bda_root.variable_id,
      &binding_base_index_id, 1, &binding_base_pointer_id));
  uint32_t binding_base_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_load_aligned(
      state, u32_type_id, binding_base_pointer_id, 4, &binding_base_id));

  state->abi_plan.bda_root.binding_table_pointer_id = convert_operands[1];
  state->abi_plan.bda_root.binding_base_id = binding_base_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_bda_binding_index(
    loom_spirv_emit_state_t* state, uint16_t binding_ordinal,
    uint32_t* out_binding_index_id) {
  IREE_RETURN_IF_ERROR(loom_spirv_emit_materialize_bda_root(state));
  if (binding_ordinal == 0) {
    *out_binding_index_id = state->abi_plan.bda_root.binding_base_id;
    return iree_ok_status();
  }

  uint32_t u32_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_u32(&state->type_context, &u32_type_id));
  uint32_t binding_ordinal_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_u32_constant(
      &state->type_context, binding_ordinal, &binding_ordinal_id));
  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
  const uint32_t operands[] = {
      u32_type_id,
      result_id,
      state->abi_plan.bda_root.binding_base_id,
      binding_ordinal_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_I_ADD, operands, IREE_ARRAYSIZE(operands)));
  *out_binding_index_id = result_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_materialize_bda_resource(
    loom_spirv_emit_state_t* state, const loom_op_t* op) {
  uint16_t binding_ordinal = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_validate_bda_resource(state, op, &binding_ordinal));

  uint32_t binding_index_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_bda_binding_index(state, binding_ordinal,
                                                         &binding_index_id));

  uint32_t zero_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_i32_constant(&state->type_context, 0, &zero_id));
  uint32_t address_pointer_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_ptr_physical_storage_buffer_u64(
      &state->type_context, &address_pointer_type_id));
  const uint32_t index_ids[] = {zero_id, binding_index_id};
  uint32_t address_pointer_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_access_chain(
      state, address_pointer_type_id,
      state->abi_plan.bda_root.binding_table_pointer_id, index_ids,
      IREE_ARRAYSIZE(index_ids), &address_pointer_id));

  uint32_t u64_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_u64(&state->type_context, &u64_type_id));
  uint32_t address_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_load_aligned(
      state, u64_type_id, address_pointer_id, 8, &address_id));

  return loom_spirv_emit_define_value(
      state, loom_low_resource_result(op),
      (loom_spirv_value_ref_t){
          .id = address_id,
          .type_id = u64_type_id,
          .value_type =
              {
                  .value_class = LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS,
              },
      },
      true);
}

static iree_status_t loom_spirv_emit_store_abi_slot_value(
    loom_spirv_emit_state_t* state, const loom_spirv_abi_slot_t* slot,
    loom_spirv_value_ref_t value) {
  loom_spirv_abi_slot_type_info_t type_info = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_abi_slot_type_info(state, slot->value_type, &type_info));
  if (!loom_spirv_value_type_equal(value.value_type, slot->value_type) ||
      value.type_id != type_info.value_type_id) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V shader-entry return value does not match "
                            "the result ABI slot type");
  }
  uint32_t store_value_id = value.id;
  if (slot->value_type.value_class == LOOM_SPIRV_VALUE_CLASS_BOOL) {
    uint32_t one_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_i32_constant(&state->type_context, 1, &one_id));
    uint32_t zero_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_i32_constant(&state->type_context, 0, &zero_id));
    store_value_id = loom_spirv_emit_allocate_id(state);
    const uint32_t select_operands[] = {
        type_info.field_type_id, store_value_id, value.id, one_id, zero_id,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
        LOOM_SPIRV_OP_SELECT, select_operands,
        IREE_ARRAYSIZE(select_operands)));
  }
  uint32_t field_pointer_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_abi_slot_field_pointer(
      state, slot, type_info.field_pointer_type_id, &field_pointer_id));
  const uint32_t operands[] = {
      field_pointer_id,
      store_value_id,
      LOOM_SPIRV_MEMORY_ACCESS_ALIGNED_MASK,
      type_info.field_alignment,
  };
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_STORE, operands, IREE_ARRAYSIZE(operands));
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
    if (!loom_spirv_value_type_equal(out_operands[i].value_type,
                                     row->operand_types[i])) {
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
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      &state->type_context, row->result_type, &type_id));
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
  return loom_spirv_emit_define_packet_result(state, packet, result_id, type_id,
                                              row->result_type);
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
  return loom_spirv_emit_define_packet_result(
      state, packet, result_id, operands[0].type_id, row->result_type);
}

static iree_status_t loom_spirv_emit_unary_convert_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_value_ref_t operands[1] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      &state->type_context, row->result_type, &result_type_id));
  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
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
  IREE_RETURN_IF_ERROR(loom_spirv_emit_unary_result(
      state, LOOM_SPIRV_OP_LOAD, vector_type_id, variable_id, &vector_id));
  const uint32_t component_id = loom_spirv_emit_allocate_id(state);
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

  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      &state->type_context, row->result_type, &result_type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_bitcast_if_needed(
      state, result_type_id, u32_type_id, component_id, &result_id));
  return loom_spirv_emit_define_packet_result(state, packet, result_id,
                                              result_type_id, row->result_type);
}

static iree_status_t loom_spirv_emit_integer_mul_add_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_value_ref_t operands[3] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  if (operands[0].type_id != operands[1].type_id ||
      operands[0].type_id != operands[2].type_id) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V low descriptor '%.*s' operand types do "
                            "not match",
                            (int)packet->key.size, packet->key.data);
  }
  uint32_t product_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_binary_result(
      state, LOOM_SPIRV_OP_I_MUL, operands[0].type_id, operands[0].id,
      operands[1].id, &product_id));
  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
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
  loom_spirv_value_ref_t operands[2] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  if (operands[0].type_id != operands[1].type_id) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V low descriptor '%.*s' operand types do "
                            "not match",
                            (int)packet->key.size, packet->key.data);
  }
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      &state->type_context, row->result_type, &result_type_id));
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
  return loom_spirv_emit_define_packet_result(state, packet, result_id,
                                              result_type_id, row->result_type);
}

static iree_status_t loom_spirv_emit_select_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_value_ref_t operands[3] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  if (operands[1].type_id != operands[2].type_id) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V low descriptor '%.*s' select value types "
                            "do not match",
                            (int)packet->key.size, packet->key.data);
  }
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      &state->type_context, row->result_type, &result_type_id));
  if (result_type_id != operands[1].type_id) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V low descriptor '%.*s' select row result "
                            "type does not match operand types",
                            (int)packet->key.size, packet->key.data);
  }
  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
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
    loom_spirv_emit_state_t* state, loom_spirv_value_ref_t address,
    loom_spirv_value_type_t pointer_type, uint32_t pointer_type_id,
    uint32_t* out_pointer_id) {
  if (address.value_type.value_class ==
      LOOM_SPIRV_VALUE_CLASS_PTR_PHYSICAL_STORAGE_BUFFER) {
    if (!loom_spirv_value_type_equal(address.value_type, pointer_type)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "SPIR-V storage-buffer pointer type does not "
                              "match access-chain result type");
    }
    *out_pointer_id = address.id;
    return iree_ok_status();
  }
  if (address.value_type.value_class !=
      LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V storage-buffer access requires a raw "
                            "storage-buffer address");
  }
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
  loom_spirv_value_ref_t operands[2] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      &state->type_context, row->result_type, &result_type_id));
  uint32_t base_pointer_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_typed_physical_storage_buffer_pointer(
      state, operands[0], row->result_type, result_type_id, &base_pointer_id));
  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
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

static iree_status_t loom_spirv_emit_load_aligned_packet(
    loom_spirv_emit_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_value_ref_t operands[1] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      &state->type_context, row->result_type, &result_type_id));
  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
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
  loom_spirv_value_ref_t operands[1] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      &state->type_context, row->result_type, &result_type_id));
  uint32_t layout_id = 0;
  uint32_t stride_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_cooperative_matrix_layout_operands(
      state, row, &layout_id, &stride_id));
  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
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
  loom_spirv_value_ref_t operands[2] = {0};
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
  loom_spirv_value_ref_t operands[3] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      &state->type_context, row->result_type, &result_type_id));
  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
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
  for (uint16_t i = 0; i < packet->descriptor->feature_mask_word_count; ++i) {
    const uint32_t feature_mask_row =
        packet->descriptor->feature_mask_word_start + i;
    const uint64_t feature_bits =
        state->target->descriptor_set->feature_mask_words[feature_mask_row];
    if (i != 0 && feature_bits != 0) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "SPIR-V emission only supports descriptor feature bits in word 0");
    }
    loom_spirv_module_builder_require_feature_bits(
        state->builder, (loom_spirv_feature_bits_t)feature_bits);
  }

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
  if (loom_low_resource_isa(op)) {
    if (state->abi_plan.kind != LOOM_SPIRV_ABI_PLAN_HAL_KERNEL_RAW_BDA) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "SPIR-V low.resource emission requires a raw-BDA HAL ABI plan");
    }
    return loom_spirv_emit_materialize_bda_resource(state, op);
  }
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
                            "the selected result ABI");
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

static uint16_t loom_spirv_emit_block_index(const loom_block_t* block) {
  return loom_block_region_index(block);
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
              state->block_plans[loom_spirv_emit_block_index(block)].label_id,
      };
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_prepare_selection_merges(
    loom_spirv_emit_state_t* state) {
  const loom_block_t* block = NULL;
  loom_region_for_each_block(state->body, block) {
    const loom_op_t* terminator = block->last_op;
    if (!terminator || !loom_low_cond_br_isa(terminator)) {
      continue;
    }
    const loom_block_t* merge_block = NULL;
    if (!loom_spirv_low_select_merge_block(terminator, &merge_block)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "SPIR-V low conditional branch cannot be emitted as a structured "
          "selection");
    }
    state->block_plans[loom_spirv_emit_block_index(block)]
        .selection_merge_label_id =
        state->block_plans[loom_spirv_emit_block_index(merge_block)].label_id;
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
  IREE_RETURN_IF_ERROR(loom_spirv_emit_count_phi_incomings(state));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_phi_incomings(state));
  return loom_spirv_emit_prepare_selection_merges(state);
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
  if (block_plan->incoming_count == 0 ||
      block_plan->incoming_count % block->arg_count != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V block argument has no complete incoming "
                            "branch payload set");
  }
  const uint32_t incoming_edge_count =
      block_plan->incoming_count / block->arg_count;
  for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
    const loom_spirv_phi_incoming_t* first_incoming =
        &state->phi_incomings[block_plan->incoming_start + arg_index];
    loom_spirv_value_ref_t first_value_ref = {0};
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
      loom_spirv_value_ref_t value_ref = {0};
      IREE_RETURN_IF_ERROR(
          loom_spirv_emit_lookup_value(state, incoming->value_id, &value_ref));
      if (value_ref.type_id != first_value_ref.type_id ||
          !loom_spirv_value_type_equal(value_ref.value_type,
                                       first_value_ref.value_type)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "SPIR-V block argument incoming value types do not match");
      }
      operands[2 + incoming_index * 2] = value_ref.id;
      operands[3 + incoming_index * 2] = incoming->predecessor_label_id;
    }
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
        LOOM_SPIRV_OP_PHI, operands, operand_count));
    const loom_spirv_value_ref_t block_arg_ref = {
        .id = result_id,
        .type_id = first_value_ref.type_id,
        .value_type = first_value_ref.value_type,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_emit_define_value(
        state, loom_block_arg_id(block, arg_index), block_arg_ref, true));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_br(loom_spirv_emit_state_t* state,
                                        const loom_op_t* op) {
  const loom_block_t* dest = loom_low_br_dest(op);
  const uint32_t operands[] = {
      state->block_plans[loom_spirv_emit_block_index(dest)].label_id,
  };
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_BRANCH, operands, IREE_ARRAYSIZE(operands));
}

static iree_status_t loom_spirv_emit_cond_br(loom_spirv_emit_state_t* state,
                                             const loom_op_t* op) {
  const uint16_t block_index = loom_spirv_emit_block_index(op->parent_block);
  const loom_spirv_block_plan_t* block_plan = &state->block_plans[block_index];
  const uint32_t merge_operands[] = {
      block_plan->selection_merge_label_id,
      LOOM_SPIRV_SELECTION_CONTROL_NONE,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_SELECTION_MERGE, merge_operands,
      IREE_ARRAYSIZE(merge_operands)));

  loom_spirv_value_ref_t condition = {0};
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

static iree_status_t loom_spirv_emit_entry_point(
    loom_spirv_emit_state_t* state) {
  const uint32_t prefix_operands[] = {
      LOOM_SPIRV_EXECUTION_MODEL_GL_COMPUTE,
      state->function_id,
  };
  uint32_t interface_operands[LOOM_SPIRV_BUILTIN_VARIABLE_COUNT] = {0};
  iree_host_size_t interface_operand_count = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(interface_operands); ++i) {
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
    if (state->abi_plan.kind == LOOM_SPIRV_ABI_PLAN_HAL_KERNEL_RAW_BDA) {
      IREE_RETURN_IF_ERROR(loom_spirv_emit_materialize_bda_args(state));
    } else {
      IREE_RETURN_IF_ERROR(loom_spirv_emit_materialize_abi_args(state));
    }
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
  return iree_ok_status();
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
  if (body == NULL || body->block_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V emission requires a low function body");
  }
  state->body = body;
  const loom_block_t* entry_block = loom_region_const_entry_block(body);
  state->function_id = loom_spirv_emit_allocate_id(state);
  IREE_RETURN_IF_ERROR(loom_spirv_emit_op_name(
      state, state->function_id, loom_spirv_emit_function_name(state)));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_build_abi_plan(state, entry_block));

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
  if (state->abi_plan.kind == LOOM_SPIRV_ABI_PLAN_HAL_KERNEL_RAW_BDA) {
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_bda_metadata(state, state->abi_plan.bda_binding_count,
                                     state->abi_plan.bda_constant_word_count));
  }
  return iree_ok_status();
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
    status = loom_spirv_emit_prepare_abi_value_types(
        &state, loom_region_const_entry_block(body));
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
