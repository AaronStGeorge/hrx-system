// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/verify.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/error/error_defs.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/util/walk.h"

typedef struct loom_low_verify_state_t {
  const loom_module_t* module;
  const loom_low_descriptor_registry_t* registry;
  iree_diagnostic_emitter_t emitter;
  loom_low_verify_result_t* result;
  uint32_t max_errors;
  iree_arena_allocator_t arena;
} loom_low_verify_state_t;

typedef struct loom_low_function_verify_state_t {
  loom_low_verify_state_t* state;
  const loom_low_resolved_target_t* target;
  iree_string_view_t function_name;
} loom_low_function_verify_state_t;

static bool loom_low_verify_should_stop(const loom_low_verify_state_t* state) {
  return state->max_errors != 0 &&
         state->result->error_count >= state->max_errors;
}

static iree_status_t loom_low_verify_emit(
    loom_low_verify_state_t* state, const loom_op_t* op,
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count,
    const loom_diagnostic_related_op_t* related_ops,
    iree_host_size_t related_op_count) {
  if (loom_low_verify_should_stop(state)) return iree_ok_status();
  if (error->severity == LOOM_DIAGNOSTIC_ERROR) {
    ++state->result->error_count;
  } else if (error->severity == LOOM_DIAGNOSTIC_WARNING) {
    ++state->result->warning_count;
  }
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
      .related_ops = related_ops,
      .related_op_count = related_op_count,
  };
  return iree_diagnostic_emit(state->emitter, &emission);
}

static iree_status_t loom_low_verify_counting_emitter(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  loom_low_verify_state_t* state = (loom_low_verify_state_t*)user_data;
  return loom_low_verify_emit(
      state, emission->op, emission->error, emission->params,
      emission->param_count, emission->related_ops, emission->related_op_count);
}

static iree_string_view_t loom_low_verify_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t ref) {
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (symbol->name_id < module->strings.count) {
    return module->strings.entries[symbol->name_id];
  }
  return IREE_SV("<unnamed>");
}

static iree_string_view_t loom_low_verify_function_name(
    const loom_module_t* module, const loom_op_t* low_func_op) {
  if (loom_low_func_def_isa(low_func_op)) {
    return loom_low_verify_symbol_name(module,
                                       loom_low_func_def_callee(low_func_op));
  }
  if (loom_low_func_decl_isa(low_func_op)) {
    return loom_low_verify_symbol_name(module,
                                       loom_low_func_decl_callee(low_func_op));
  }
  return IREE_SV("<unnamed>");
}

static iree_string_view_t loom_low_verify_string_or_empty(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static bool loom_low_verify_get_packet_opcode(const loom_module_t* module,
                                              const loom_op_t* op,
                                              iree_string_view_t* out_opcode,
                                              uint16_t* out_opcode_attr_index) {
  if (loom_low_op_isa(op)) {
    *out_opcode =
        loom_low_verify_string_or_empty(module, loom_low_op_opcode(op));
    *out_opcode_attr_index = loom_low_op_opcode_ATTR_INDEX;
    return true;
  }
  if (loom_low_const_isa(op)) {
    *out_opcode =
        loom_low_verify_string_or_empty(module, loom_low_const_opcode(op));
    *out_opcode_attr_index = loom_low_const_opcode_ATTR_INDEX;
    return true;
  }
  return false;
}

static iree_status_t loom_low_verify_emit_missing_descriptor(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    iree_string_view_t opcode, uint16_t opcode_attr_index) {
  const loom_low_resolved_target_t* target = function_state->target;
  loom_diagnostic_param_t params[] = {
      loom_param_string(function_state->function_name),
      loom_param_with_field_ref(
          loom_param_string(opcode),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    opcode_attr_index)),
      loom_param_string(target->descriptor_set_key),
  };
  return loom_low_verify_emit(
      function_state->state, op,
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 4), params,
      IREE_ARRAYSIZE(params), NULL, 0);
}

static iree_status_t loom_low_verify_emit_count_mismatch(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    const loom_error_def_t* error, iree_string_view_t opcode,
    uint16_t opcode_attr_index, uint32_t actual_count,
    uint32_t expected_count) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(opcode),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    opcode_attr_index)),
      loom_param_u32(actual_count),
      loom_param_u32(expected_count),
  };
  return loom_low_verify_emit(function_state->state, op, error, params,
                              IREE_ARRAYSIZE(params), NULL, 0);
}

static iree_status_t loom_low_verify_emit_missing_features(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    iree_string_view_t opcode, uint16_t opcode_attr_index,
    uint32_t feature_word_index, uint64_t missing_bits) {
  const loom_low_resolved_target_t* target = function_state->target;
  loom_diagnostic_param_t params[] = {
      loom_param_string(function_state->function_name),
      loom_param_with_field_ref(
          loom_param_string(opcode),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    opcode_attr_index)),
      loom_param_string(target->descriptor_set_key),
      loom_param_u32(feature_word_index),
      loom_param_u64(missing_bits),
  };
  return loom_low_verify_emit(
      function_state->state, op,
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 5), params,
      IREE_ARRAYSIZE(params), NULL, 0);
}

static iree_status_t loom_low_verify_format_expected_register_classes(
    loom_low_function_verify_state_t* function_state,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_operand_t* descriptor_operand,
    iree_string_view_t* out_expected_reg_classes) {
  *out_expected_reg_classes = iree_string_view_empty();
  iree_host_size_t byte_count = 0;
  uint32_t reg_class_count = 0;
  for (uint16_t i = 0; i < descriptor_operand->reg_class_alt_count; ++i) {
    const uint16_t alt_index = descriptor_operand->reg_class_alt_start + i;
    const loom_low_reg_class_alt_t* alt =
        &descriptor_set->reg_class_alts[alt_index];
    if (iree_all_bits_set(alt->flags, LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE)) {
      continue;
    }
    const loom_low_reg_class_t* reg_class =
        &descriptor_set->reg_classes[alt->reg_class_id];
    iree_string_view_t reg_class_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
        descriptor_set, reg_class->name_string_offset, &reg_class_name));
    if (reg_class_count > 0) byte_count += 3;
    byte_count += reg_class_name.size;
    ++reg_class_count;
  }
  if (reg_class_count == 0) {
    *out_expected_reg_classes = IREE_SV("<none>");
    return iree_ok_status();
  }

  char* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&function_state->state->arena,
                                           byte_count, (void**)&storage));
  char* cursor = storage;
  uint32_t appended_count = 0;
  for (uint16_t i = 0; i < descriptor_operand->reg_class_alt_count; ++i) {
    const uint16_t alt_index = descriptor_operand->reg_class_alt_start + i;
    const loom_low_reg_class_alt_t* alt =
        &descriptor_set->reg_class_alts[alt_index];
    if (iree_all_bits_set(alt->flags, LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE)) {
      continue;
    }
    const loom_low_reg_class_t* reg_class =
        &descriptor_set->reg_classes[alt->reg_class_id];
    iree_string_view_t reg_class_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
        descriptor_set, reg_class->name_string_offset, &reg_class_name));
    if (appended_count > 0) {
      memcpy(cursor, " | ", 3);
      cursor += 3;
    }
    memcpy(cursor, reg_class_name.data, reg_class_name.size);
    cursor += reg_class_name.size;
    ++appended_count;
  }
  *out_expected_reg_classes = iree_make_string_view(storage, byte_count);
  return iree_ok_status();
}

static iree_status_t loom_low_verify_operand_accepts_register_class(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_operand_t* descriptor_operand,
    iree_string_view_t actual_reg_class, bool* out_accepted) {
  *out_accepted = false;
  for (uint16_t i = 0; i < descriptor_operand->reg_class_alt_count; ++i) {
    const uint16_t alt_index = descriptor_operand->reg_class_alt_start + i;
    const loom_low_reg_class_alt_t* alt =
        &descriptor_set->reg_class_alts[alt_index];
    if (iree_all_bits_set(alt->flags, LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE)) {
      continue;
    }
    const loom_low_reg_class_t* reg_class =
        &descriptor_set->reg_classes[alt->reg_class_id];
    iree_string_view_t expected_reg_class = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
        descriptor_set, reg_class->name_string_offset, &expected_reg_class));
    if (iree_string_view_equal(actual_reg_class, expected_reg_class)) {
      *out_accepted = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_emit_register_type_mismatch(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    iree_string_view_t opcode, uint16_t opcode_attr_index,
    iree_string_view_t field_kind, loom_diagnostic_field_ref_t field_ref,
    iree_string_view_t field_name, loom_type_t actual_type,
    iree_string_view_t expected_reg_classes, uint32_t expected_unit_count) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(function_state->function_name),
      loom_param_with_field_ref(
          loom_param_string(opcode),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    opcode_attr_index)),
      loom_param_string(field_kind),
      loom_param_with_field_ref(loom_param_string(field_name), field_ref),
      loom_param_type(actual_type),
      loom_param_string(expected_reg_classes),
      loom_param_u32(expected_unit_count),
  };
  return loom_low_verify_emit(
      function_state->state, op,
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 6), params,
      IREE_ARRAYSIZE(params), NULL, 0);
}

static iree_status_t loom_low_verify_descriptor_register_field(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    iree_string_view_t opcode, uint16_t opcode_attr_index,
    const loom_low_descriptor_t* descriptor,
    uint16_t descriptor_operand_index) {
  const loom_module_t* module = function_state->state->module;
  const loom_low_descriptor_set_t* descriptor_set =
      function_state->target->descriptor_set;
  const uint32_t operand_row =
      descriptor->operand_start + descriptor_operand_index;
  const loom_low_operand_t* descriptor_operand =
      &descriptor_set->operands[operand_row];

  const bool is_result = descriptor_operand_index < descriptor->result_count;
  const uint16_t field_index =
      is_result
          ? descriptor_operand_index
          : (uint16_t)(descriptor_operand_index - descriptor->result_count);
  const loom_value_id_t value_id =
      is_result ? loom_op_const_results(op)[field_index]
                : loom_op_const_operands(op)[field_index];
  const loom_type_t actual_type = loom_module_value_type(module, value_id);

  bool accepted = false;
  if (loom_type_is_register(actual_type) &&
      loom_type_register_unit_count(actual_type) ==
          descriptor_operand->unit_count) {
    iree_string_view_t actual_reg_class = loom_low_verify_string_or_empty(
        module, loom_type_register_class_id(actual_type));
    IREE_RETURN_IF_ERROR(loom_low_verify_operand_accepts_register_class(
        descriptor_set, descriptor_operand, actual_reg_class, &accepted));
  }
  if (accepted) return iree_ok_status();

  iree_string_view_t expected_reg_classes = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_verify_format_expected_register_classes(
      function_state, descriptor_set, descriptor_operand,
      &expected_reg_classes));
  iree_string_view_t field_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
      descriptor_set, descriptor_operand->field_name_string_offset,
      &field_name));
  const loom_diagnostic_field_ref_t field_ref = loom_diagnostic_field_ref(
      is_result ? LOOM_DIAGNOSTIC_FIELD_RESULT : LOOM_DIAGNOSTIC_FIELD_OPERAND,
      field_index);
  return loom_low_verify_emit_register_type_mismatch(
      function_state, op, opcode, opcode_attr_index,
      is_result ? IREE_SV("result") : IREE_SV("operand"), field_ref, field_name,
      actual_type, expected_reg_classes, descriptor_operand->unit_count);
}

static iree_status_t loom_low_verify_descriptor_registers(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    iree_string_view_t opcode, uint16_t opcode_attr_index,
    const loom_low_descriptor_t* descriptor) {
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_register_field(
        function_state, op, opcode, opcode_attr_index, descriptor, i));
    if (loom_low_verify_should_stop(function_state->state)) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_descriptor_features(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    iree_string_view_t opcode, uint16_t opcode_attr_index,
    const loom_low_descriptor_t* descriptor) {
  const loom_low_resolved_target_t* target = function_state->target;
  const loom_low_descriptor_set_t* descriptor_set = target->descriptor_set;
  for (uint16_t i = 0; i < descriptor->feature_mask_word_count; ++i) {
    const uint32_t word_index = descriptor->feature_mask_word_start + i;
    const uint64_t required_bits =
        descriptor_set->feature_mask_words[word_index];
    const uint64_t available_bits = word_index == 0 ? target->feature_bits : 0;
    const uint64_t missing_bits = required_bits & ~available_bits;
    if (missing_bits == 0) continue;
    IREE_RETURN_IF_ERROR(loom_low_verify_emit_missing_features(
        function_state, op, opcode, opcode_attr_index, word_index,
        missing_bits));
    if (loom_low_verify_should_stop(function_state->state)) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_packet(
    loom_low_function_verify_state_t* function_state, const loom_op_t* op,
    iree_string_view_t opcode, uint16_t opcode_attr_index) {
  const loom_low_descriptor_set_t* descriptor_set =
      function_state->target->descriptor_set;
  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  iree_status_t lookup_status = loom_low_descriptor_set_lookup_descriptor(
      descriptor_set, opcode, &descriptor_ordinal);
  if (!iree_status_is_ok(lookup_status)) {
    iree_status_code_t status_code = iree_status_code(lookup_status);
    if (status_code == IREE_STATUS_NOT_FOUND) {
      iree_status_free(lookup_status);
      return loom_low_verify_emit_missing_descriptor(function_state, op, opcode,
                                                     opcode_attr_index);
    }
    return iree_status_annotate_f(lookup_status,
                                  "failed to look up low descriptor '%.*s'",
                                  (int)opcode.size, opcode.data);
  }

  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, descriptor_ordinal);
  if (!descriptor) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low descriptor ordinal %" PRIu32
                            " is out of range",
                            descriptor_ordinal);
  }
  if (descriptor->result_count > descriptor->operand_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor '%.*s' has %" PRIu16
                            " results but only %" PRIu16 " operand rows",
                            (int)opcode.size, opcode.data,
                            descriptor->result_count,
                            descriptor->operand_count);
  }

  const uint32_t expected_result_count = descriptor->result_count;
  const uint32_t expected_operand_count =
      descriptor->operand_count - descriptor->result_count;
  if (op->result_count != expected_result_count) {
    IREE_RETURN_IF_ERROR(loom_low_verify_emit_count_mismatch(
        function_state, op,
        loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 2), opcode,
        opcode_attr_index, op->result_count, expected_result_count));
  }
  if (op->operand_count != expected_operand_count) {
    IREE_RETURN_IF_ERROR(loom_low_verify_emit_count_mismatch(
        function_state, op,
        loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 1), opcode,
        opcode_attr_index, op->operand_count, expected_operand_count));
  }
  if (loom_low_verify_should_stop(function_state->state)) {
    return iree_ok_status();
  }
  if (op->result_count == expected_result_count &&
      op->operand_count == expected_operand_count) {
    IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_registers(
        function_state, op, opcode, opcode_attr_index, descriptor));
  }
  if (loom_low_verify_should_stop(function_state->state)) {
    return iree_ok_status();
  }
  return loom_low_verify_descriptor_features(function_state, op, opcode,
                                             opcode_attr_index, descriptor);
}

static iree_status_t loom_low_verify_walk_op(void* user_data, loom_op_t* op,
                                             const loom_walk_context_t* context,
                                             loom_walk_result_t* out_result) {
  (void)context;
  loom_low_function_verify_state_t* function_state =
      (loom_low_function_verify_state_t*)user_data;
  *out_result = LOOM_WALK_CONTINUE;
  if (loom_low_verify_should_stop(function_state->state)) {
    *out_result = LOOM_WALK_ABORT;
    return iree_ok_status();
  }

  iree_string_view_t opcode = iree_string_view_empty();
  uint16_t opcode_attr_index = 0;
  if (!loom_low_verify_get_packet_opcode(function_state->state->module, op,
                                         &opcode, &opcode_attr_index)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_low_verify_packet(function_state, op, opcode, opcode_attr_index));
  if (loom_low_verify_should_stop(function_state->state)) {
    *out_result = LOOM_WALK_ABORT;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_function(loom_low_verify_state_t* state,
                                              const loom_op_t* low_func_op) {
  iree_diagnostic_emitter_t counting_emitter = {
      .fn = loom_low_verify_counting_emitter,
      .user_data = state,
  };
  loom_low_resolved_target_t target = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      state->module, low_func_op, state->registry, counting_emitter, &target));
  if (target.descriptor_set == NULL || !loom_low_func_def_isa(low_func_op) ||
      loom_low_verify_should_stop(state)) {
    return iree_ok_status();
  }

  loom_low_function_verify_state_t function_state = {
      .state = state,
      .target = &target,
      .function_name =
          loom_low_verify_function_name(state->module, low_func_op),
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  return loom_walk_region(
      state->module, loom_low_func_def_body(low_func_op), LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_low_verify_walk_op, &function_state},
      &state->arena, &walk_result);
}

iree_status_t loom_low_verify_module(const loom_module_t* module,
                                     const loom_low_verify_options_t* options,
                                     loom_low_verify_result_t* out_result) {
  if (!module || !options || !out_result) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module, options, and out_result are required");
  }
  if (options->descriptor_registry == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor registry is required");
  }
  if (iree_any_bit_set(options->flags,
                       LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY)) {
    IREE_RETURN_IF_ERROR(
        loom_low_descriptor_registry_verify(options->descriptor_registry));
  }

  *out_result = (loom_low_verify_result_t){0};
  loom_low_verify_state_t state = {
      .module = module,
      .registry = options->descriptor_registry,
      .emitter = options->emitter,
      .result = out_result,
      .max_errors = options->max_errors,
  };
  iree_arena_initialize(module->arena.block_pool, &state.arena);

  iree_status_t status = iree_ok_status();
  if (module->body && module->body->block_count > 0) {
    loom_block_t* entry_block = loom_region_entry_block(module->body);
    const loom_op_t* op = NULL;
    loom_block_for_each_op(entry_block, op) {
      if (!loom_low_func_def_isa(op) && !loom_low_func_decl_isa(op)) continue;
      status = loom_low_verify_function(&state, op);
      if (!iree_status_is_ok(status) || loom_low_verify_should_stop(&state)) {
        break;
      }
    }
  }

  iree_arena_deinitialize(&state.arena);
  return status;
}
