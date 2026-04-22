// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/x86/assembly.h"

#include <inttypes.h>

#include "loom/codegen/low/packet.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/x86/avx512_descriptors.h"
#include "loom/target/arch/x86/packed_dot_descriptors.h"
#include "loom/target/emit/native/assembly.h"

static const char* const kX86Gpr64Names[] = {
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
    "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15",
};

typedef enum loom_x86_descriptor_set_kind_e {
  LOOM_X86_DESCRIPTOR_SET_AVX512_CORE = 0,
  LOOM_X86_DESCRIPTOR_SET_PACKED_DOT_CORE = 1,
} loom_x86_descriptor_set_kind_t;

typedef enum loom_x86_register_class_kind_e {
  LOOM_X86_REGISTER_CLASS_GPR64 = 0,
  LOOM_X86_REGISTER_CLASS_XMM = 1,
  LOOM_X86_REGISTER_CLASS_YMM = 2,
  LOOM_X86_REGISTER_CLASS_ZMM = 3,
  LOOM_X86_REGISTER_CLASS_K = 4,
} loom_x86_register_class_kind_t;

typedef struct loom_x86_assembly_state_t {
  // Descriptor-set family selected for this fragment.
  loom_x86_descriptor_set_kind_t descriptor_set_kind;
} loom_x86_assembly_state_t;

static iree_status_t loom_x86_resolve_descriptor_set_kind(
    const loom_low_schedule_sidecar_t* schedule,
    loom_x86_descriptor_set_kind_t* out_kind) {
  if (iree_string_view_equal(schedule->target.descriptor_set_key,
                             IREE_SV("x86.avx512.core"))) {
    *out_kind = LOOM_X86_DESCRIPTOR_SET_AVX512_CORE;
    return iree_ok_status();
  }
  if (iree_string_view_equal(schedule->target.descriptor_set_key,
                             IREE_SV("x86.packed_dot.core"))) {
    *out_kind = LOOM_X86_DESCRIPTOR_SET_PACKED_DOT_CORE;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "x86 assembly descriptor set '%.*s' is unsupported",
                          (int)schedule->target.descriptor_set_key.size,
                          schedule->target.descriptor_set_key.data);
}

static iree_status_t loom_x86_descriptor_key(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t* out_key) {
  return loom_native_assembly_descriptor_string(
      context->schedule->target.descriptor_set,
      context->packet->descriptor->key_string_offset, out_key);
}

static iree_status_t loom_x86_append_mnemonic(
    const loom_native_assembly_packet_context_t* context) {
  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
      context->schedule->target.descriptor_set,
      context->packet->descriptor->mnemonic_string_offset, &mnemonic));
  return iree_string_builder_append_string(context->builder, mnemonic);
}

static iree_status_t loom_x86_find_assignment(
    const loom_native_assembly_packet_context_t* context,
    loom_value_id_t value_id,
    const loom_low_allocation_assignment_t** out_assignment) {
  const loom_low_allocation_assignment_t* assignment =
      loom_low_packet_find_assignment(context->allocation, value_id, NULL);
  if (assignment == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "x86 assembly value %" PRIu32
                            " has no allocation assignment",
                            value_id);
  }
  *out_assignment = assignment;
  return iree_ok_status();
}

static bool loom_x86_assignments_match(
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  return lhs->location_kind == rhs->location_kind &&
         lhs->descriptor_reg_class_id == rhs->descriptor_reg_class_id &&
         lhs->location_base == rhs->location_base &&
         lhs->location_count == rhs->location_count;
}

static iree_status_t loom_x86_register_class_kind(
    const loom_x86_assembly_state_t* state,
    const loom_native_assembly_packet_context_t* context,
    const loom_low_allocation_assignment_t* assignment,
    loom_x86_register_class_kind_t* out_kind) {
  switch (state->descriptor_set_kind) {
    case LOOM_X86_DESCRIPTOR_SET_AVX512_CORE:
      switch (assignment->descriptor_reg_class_id) {
        case X86_AVX512_CORE_REG_CLASS_ID_X86_GPR64:
          *out_kind = LOOM_X86_REGISTER_CLASS_GPR64;
          return iree_ok_status();
        case X86_AVX512_CORE_REG_CLASS_ID_X86_ZMM:
          *out_kind = LOOM_X86_REGISTER_CLASS_ZMM;
          return iree_ok_status();
        case X86_AVX512_CORE_REG_CLASS_ID_X86_K:
          *out_kind = LOOM_X86_REGISTER_CLASS_K;
          return iree_ok_status();
        default:
          break;
      }
      break;
    case LOOM_X86_DESCRIPTOR_SET_PACKED_DOT_CORE:
      switch (assignment->descriptor_reg_class_id) {
        case X86_PACKED_DOT_CORE_REG_CLASS_ID_X86_XMM:
          *out_kind = LOOM_X86_REGISTER_CLASS_XMM;
          return iree_ok_status();
        case X86_PACKED_DOT_CORE_REG_CLASS_ID_X86_YMM:
          *out_kind = LOOM_X86_REGISTER_CLASS_YMM;
          return iree_ok_status();
        case X86_PACKED_DOT_CORE_REG_CLASS_ID_X86_ZMM:
          *out_kind = LOOM_X86_REGISTER_CLASS_ZMM;
          return iree_ok_status();
        default:
          break;
      }
      break;
  }
  iree_string_view_t register_class = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_register_class_name(
      context->allocation, assignment, &register_class));
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "x86 assembly register class '%.*s' is unsupported",
                          (int)register_class.size, register_class.data);
}

static iree_status_t loom_x86_append_assignment(
    const loom_x86_assembly_state_t* state,
    const loom_native_assembly_packet_context_t* context,
    const loom_low_allocation_assignment_t* assignment) {
  if (assignment->location_kind !=
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "x86 assembly value %" PRIu32
                            " is not physically allocated",
                            assignment->value_id);
  }
  if (assignment->location_count != 1) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 assembly multi-register value %" PRIu32
                            " is not supported",
                            assignment->value_id);
  }
  loom_x86_register_class_kind_t register_class_kind = 0;
  IREE_RETURN_IF_ERROR(loom_x86_register_class_kind(state, context, assignment,
                                                    &register_class_kind));
  if (register_class_kind == LOOM_X86_REGISTER_CLASS_GPR64) {
    if (assignment->location_base >= IREE_ARRAYSIZE(kX86Gpr64Names)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "x86 GPR index %" PRIu32 " is out of range",
                              assignment->location_base);
    }
    return iree_string_builder_append_cstring(
        context->builder, kX86Gpr64Names[assignment->location_base]);
  }
  if (register_class_kind == LOOM_X86_REGISTER_CLASS_XMM) {
    return iree_string_builder_append_format(context->builder, "xmm%" PRIu32,
                                             assignment->location_base);
  }
  if (register_class_kind == LOOM_X86_REGISTER_CLASS_YMM) {
    return iree_string_builder_append_format(context->builder, "ymm%" PRIu32,
                                             assignment->location_base);
  }
  if (register_class_kind == LOOM_X86_REGISTER_CLASS_ZMM) {
    return iree_string_builder_append_format(context->builder, "zmm%" PRIu32,
                                             assignment->location_base);
  }
  if (register_class_kind == LOOM_X86_REGISTER_CLASS_K) {
    return iree_string_builder_append_format(context->builder, "k%" PRIu32,
                                             assignment->location_base);
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "x86 assembly register class kind %u is unsupported",
                          (unsigned)register_class_kind);
}

static iree_status_t loom_x86_append_value(
    const loom_x86_assembly_state_t* state,
    const loom_native_assembly_packet_context_t* context,
    loom_value_id_t value_id) {
  const loom_low_allocation_assignment_t* assignment = NULL;
  IREE_RETURN_IF_ERROR(
      loom_x86_find_assignment(context, value_id, &assignment));
  return loom_x86_append_assignment(state, context, assignment);
}

static iree_status_t loom_x86_append_result(
    const loom_x86_assembly_state_t* state,
    const loom_native_assembly_packet_context_t* context,
    iree_host_size_t result_index) {
  const loom_op_t* op = context->packet->node->op;
  if (result_index >= op->result_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 assembly result index is out of range");
  }
  return loom_x86_append_value(state, context,
                               loom_op_const_results(op)[result_index]);
}

static iree_status_t loom_x86_append_operand(
    const loom_x86_assembly_state_t* state,
    const loom_native_assembly_packet_context_t* context,
    iree_host_size_t operand_index) {
  const loom_op_t* op = context->packet->node->op;
  if (operand_index >= op->operand_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 assembly operand index is out of range");
  }
  return loom_x86_append_value(state, context,
                               loom_op_const_operands(op)[operand_index]);
}

static loom_named_attr_slice_t loom_x86_packet_attrs(
    const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  if (loom_low_op_isa(op)) {
    return loom_low_op_attrs(op);
  }
  if (loom_low_const_isa(op)) {
    return loom_low_const_attrs(op);
  }
  return loom_make_named_attr_slice(NULL, 0);
}

static iree_status_t loom_x86_read_packet_i64_attr(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t name, int64_t* out_value) {
  return loom_native_assembly_read_i64_attr(context->schedule->module,
                                            loom_x86_packet_attrs(context),
                                            name, out_value);
}

static iree_status_t loom_x86_read_packet_address_scale_attr(
    const loom_native_assembly_packet_context_t* context, int64_t* out_scale) {
  IREE_ASSERT_ARGUMENT(out_scale);
  IREE_RETURN_IF_ERROR(
      loom_x86_read_packet_i64_attr(context, IREE_SV("scale"), out_scale));
  switch (*out_scale) {
    case 1:
    case 2:
    case 4:
    case 8:
      return iree_ok_status();
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "x86 indexed memory scale %" PRId64 " is unsupported", *out_scale);
  }
}

static iree_status_t loom_x86_append_memory_operand(
    const loom_x86_assembly_state_t* state,
    const loom_native_assembly_packet_context_t* context,
    loom_value_id_t base_value_id, loom_value_id_t index_value_id,
    int64_t scale, int64_t displacement) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "["));
  IREE_RETURN_IF_ERROR(loom_x86_append_value(state, context, base_value_id));
  if (index_value_id != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, " + "));
    IREE_RETURN_IF_ERROR(loom_x86_append_value(state, context, index_value_id));
    if (scale != 1) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          context->builder, " * %" PRId64, scale));
    }
  }
  if (displacement > 0) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        context->builder, " + %" PRId64, displacement));
  } else if (displacement < 0) {
    const uint64_t magnitude = (uint64_t)(-(displacement + 1)) + 1u;
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        context->builder, " - %" PRIu64, magnitude));
  }
  return iree_string_builder_append_cstring(context->builder, "]");
}

static iree_status_t loom_x86_append_binary_vector_packet(
    const loom_x86_assembly_state_t* state,
    const loom_native_assembly_packet_context_t* context) {
  IREE_RETURN_IF_ERROR(loom_x86_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_x86_append_result(state, context, 0));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  IREE_RETURN_IF_ERROR(loom_x86_append_operand(state, context, 0));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  return loom_x86_append_operand(state, context, 1);
}

static iree_status_t loom_x86_append_tied_ternary_packet(
    const loom_x86_assembly_state_t* state,
    const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  const loom_low_allocation_assignment_t* result_assignment = NULL;
  const loom_low_allocation_assignment_t* accumulator_assignment = NULL;
  IREE_RETURN_IF_ERROR(loom_x86_find_assignment(
      context, loom_op_const_results(op)[0], &result_assignment));
  IREE_RETURN_IF_ERROR(loom_x86_find_assignment(
      context, loom_op_const_operands(op)[0], &accumulator_assignment));
  if (!loom_x86_assignments_match(result_assignment, accumulator_assignment)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "x86 tied ternary result must share the accumulator physical register");
  }
  IREE_RETURN_IF_ERROR(loom_x86_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_x86_append_result(state, context, 0));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  IREE_RETURN_IF_ERROR(loom_x86_append_operand(state, context, 1));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  return loom_x86_append_operand(state, context, 2);
}

static iree_status_t loom_x86_append_tied_binary_packet(
    const loom_x86_assembly_state_t* state,
    const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  const loom_low_allocation_assignment_t* result_assignment = NULL;
  const loom_low_allocation_assignment_t* lhs_assignment = NULL;
  IREE_RETURN_IF_ERROR(loom_x86_find_assignment(
      context, loom_op_const_results(op)[0], &result_assignment));
  IREE_RETURN_IF_ERROR(loom_x86_find_assignment(
      context, loom_op_const_operands(op)[0], &lhs_assignment));
  if (!loom_x86_assignments_match(result_assignment, lhs_assignment)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "x86 tied binary result must share the left-hand physical register");
  }
  IREE_RETURN_IF_ERROR(loom_x86_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_x86_append_result(state, context, 0));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  return loom_x86_append_operand(state, context, 1);
}

static iree_status_t loom_x86_append_lea_add_packet(
    const loom_x86_assembly_state_t* state,
    const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  if (op->result_count != 1 || op->operand_count != 2) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_x86_descriptor_key(context, &key));
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 assembly LEA descriptor '%.*s' has an "
                            "unsupported operand shape",
                            (int)key.size, key.data);
  }
  IREE_RETURN_IF_ERROR(loom_x86_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_x86_append_result(state, context, 0));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  return loom_x86_append_memory_operand(state, context,
                                        loom_op_const_operands(op)[0],
                                        loom_op_const_operands(op)[1], 1, 0);
}

static iree_status_t loom_x86_descriptor_has_constraint(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_low_constraint_kind_t kind,
    uint16_t lhs_operand_index, uint16_t rhs_operand_index,
    bool* out_has_constraint) {
  *out_has_constraint = false;
  if (descriptor->constraint_count == 0) {
    return iree_ok_status();
  }
  if (descriptor->constraint_start > descriptor_set->constraint_count ||
      descriptor->constraint_count >
          descriptor_set->constraint_count - descriptor->constraint_start) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 descriptor constraint range is out of range");
  }
  if (descriptor_set->constraints == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "x86 descriptor constraints table is missing");
  }
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (constraint->kind == kind &&
        constraint->lhs_operand_index == lhs_operand_index &&
        constraint->rhs_operand_index == rhs_operand_index) {
      *out_has_constraint = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_descriptor_has_effect(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_low_effect_kind_t kind,
    bool* out_has_effect) {
  *out_has_effect = false;
  if (descriptor->effect_count == 0) {
    return iree_ok_status();
  }
  if (descriptor->effect_start > descriptor_set->effect_count ||
      descriptor->effect_count >
          descriptor_set->effect_count - descriptor->effect_start) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 descriptor effect range is out of range");
  }
  if (descriptor_set->effects == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "x86 descriptor effects table is missing");
  }
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor->effect_start + i];
    if (effect->kind == kind) {
      *out_has_effect = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_descriptor_uses_tied_binary_form(
    const loom_native_assembly_packet_context_t* context,
    bool* out_uses_tied_binary_form) {
  *out_uses_tied_binary_form = false;
  const loom_op_t* op = context->packet->node->op;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  if (descriptor->result_count != 1 || descriptor->operand_count != 3 ||
      op->result_count != 1 || op->operand_count != 2) {
    return iree_ok_status();
  }
  const uint16_t result_operand_index = 0;
  const uint16_t lhs_operand_index = descriptor->result_count;
  bool is_tied = false;
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_has_constraint(
      descriptor_set, descriptor, LOOM_LOW_CONSTRAINT_KIND_TIED,
      result_operand_index, lhs_operand_index, &is_tied));
  if (!is_tied) {
    return iree_ok_status();
  }
  bool is_destructive = false;
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_has_constraint(
      descriptor_set, descriptor, LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE,
      result_operand_index, lhs_operand_index, &is_destructive));
  *out_uses_tied_binary_form = is_destructive;
  return iree_ok_status();
}

static iree_status_t loom_x86_descriptor_uses_tied_ternary_form(
    const loom_native_assembly_packet_context_t* context,
    bool* out_uses_tied_ternary_form) {
  *out_uses_tied_ternary_form = false;
  const loom_op_t* op = context->packet->node->op;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  if (descriptor->result_count != 1 || descriptor->operand_count != 4 ||
      op->result_count != 1 || op->operand_count != 3) {
    return iree_ok_status();
  }
  const uint16_t result_operand_index = 0;
  const uint16_t accumulator_operand_index = descriptor->result_count;
  bool is_tied = false;
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_has_constraint(
      descriptor_set, descriptor, LOOM_LOW_CONSTRAINT_KIND_TIED,
      result_operand_index, accumulator_operand_index, &is_tied));
  if (!is_tied) {
    return iree_ok_status();
  }
  bool is_destructive = false;
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_has_constraint(
      descriptor_set, descriptor, LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE,
      result_operand_index, accumulator_operand_index, &is_destructive));
  *out_uses_tied_ternary_form = is_destructive;
  return iree_ok_status();
}

static iree_status_t loom_x86_append_load_packet(
    const loom_x86_assembly_state_t* state,
    const loom_native_assembly_packet_context_t* context) {
  int64_t displacement = 0;
  IREE_RETURN_IF_ERROR(
      loom_x86_read_packet_i64_attr(context, IREE_SV("disp32"), &displacement));
  const loom_op_t* op = context->packet->node->op;
  loom_value_id_t index_value_id = LOOM_VALUE_ID_INVALID;
  int64_t scale = 1;
  if (op->operand_count == 2) {
    index_value_id = loom_op_const_operands(op)[1];
    IREE_RETURN_IF_ERROR(
        loom_x86_read_packet_address_scale_attr(context, &scale));
  } else if (op->operand_count != 1) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_x86_descriptor_key(context, &key));
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 assembly load descriptor '%.*s' has an "
                            "unsupported operand shape",
                            (int)key.size, key.data);
  }
  IREE_RETURN_IF_ERROR(loom_x86_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_x86_append_result(state, context, 0));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  return loom_x86_append_memory_operand(state, context,
                                        loom_op_const_operands(op)[0],
                                        index_value_id, scale, displacement);
}

static iree_status_t loom_x86_append_store_packet(
    const loom_x86_assembly_state_t* state,
    const loom_native_assembly_packet_context_t* context) {
  int64_t displacement = 0;
  IREE_RETURN_IF_ERROR(
      loom_x86_read_packet_i64_attr(context, IREE_SV("disp32"), &displacement));
  const loom_op_t* op = context->packet->node->op;
  loom_value_id_t base_value_id = LOOM_VALUE_ID_INVALID;
  loom_value_id_t index_value_id = LOOM_VALUE_ID_INVALID;
  int64_t scale = 1;
  if (op->operand_count == 2) {
    base_value_id = loom_op_const_operands(op)[1];
  } else if (op->operand_count == 3) {
    base_value_id = loom_op_const_operands(op)[1];
    index_value_id = loom_op_const_operands(op)[2];
    IREE_RETURN_IF_ERROR(
        loom_x86_read_packet_address_scale_attr(context, &scale));
  } else {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_x86_descriptor_key(context, &key));
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 assembly store descriptor '%.*s' has an "
                            "unsupported operand shape",
                            (int)key.size, key.data);
  }
  IREE_RETURN_IF_ERROR(loom_x86_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_x86_append_memory_operand(
      state, context, base_value_id, index_value_id, scale, displacement));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  return loom_x86_append_operand(state, context, 0);
}

static iree_status_t loom_x86_append_move_packet(
    const loom_x86_assembly_state_t* state,
    const loom_native_assembly_packet_context_t* context) {
  IREE_RETURN_IF_ERROR(loom_x86_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_x86_append_result(state, context, 0));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  return loom_x86_append_operand(state, context, 0);
}

static iree_status_t loom_x86_append_const_packet(
    const loom_x86_assembly_state_t* state,
    const loom_native_assembly_packet_context_t* context) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  if (descriptor->result_count != 1 || descriptor->operand_count != 1 ||
      descriptor->immediate_count != 1) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_x86_descriptor_key(context, &key));
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 assembly const descriptor '%.*s' is "
                            "unsupported",
                            (int)key.size, key.data);
  }
  if (descriptor->immediate_start >= descriptor_set->immediate_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 assembly const immediate is out of range");
  }
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[descriptor->immediate_start];
  if (immediate->kind != LOOM_LOW_IMMEDIATE_KIND_SIGNED) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_x86_descriptor_key(context, &key));
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 assembly const descriptor '%.*s' has an "
                            "unsupported immediate kind",
                            (int)key.size, key.data);
  }
  iree_string_view_t immediate_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
      descriptor_set, immediate->field_name_string_offset, &immediate_name));
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(
      loom_x86_read_packet_i64_attr(context, immediate_name, &value));

  IREE_RETURN_IF_ERROR(loom_x86_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_x86_append_result(state, context, 0));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  return iree_string_builder_append_format(context->builder, "%" PRId64, value);
}

static iree_status_t loom_x86_append_copy_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  const loom_x86_assembly_state_t* state =
      (const loom_x86_assembly_state_t*)user_data;
  const loom_op_t* op = context->packet->node->op;
  const loom_low_allocation_assignment_t* source_assignment = NULL;
  IREE_RETURN_IF_ERROR(loom_x86_find_assignment(
      context, loom_low_copy_source(op), &source_assignment));
  const loom_low_allocation_assignment_t* result_assignment = NULL;
  IREE_RETURN_IF_ERROR(loom_x86_find_assignment(
      context, loom_low_copy_result(op), &result_assignment));
  if (loom_x86_assignments_match(source_assignment, result_assignment)) {
    return iree_ok_status();
  }

  if (source_assignment->descriptor_reg_class_id !=
      result_assignment->descriptor_reg_class_id) {
    iree_string_view_t source_register_class = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_register_class_name(
        context->allocation, source_assignment, &source_register_class));
    iree_string_view_t result_register_class = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_register_class_name(
        context->allocation, result_assignment, &result_register_class));
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "x86 assembly copy between register classes '%.*s' and '%.*s' is "
        "unsupported",
        (int)source_register_class.size, source_register_class.data,
        (int)result_register_class.size, result_register_class.data);
  }

  loom_x86_register_class_kind_t register_class_kind = 0;
  IREE_RETURN_IF_ERROR(loom_x86_register_class_kind(
      state, context, result_assignment, &register_class_kind));
  switch (register_class_kind) {
    case LOOM_X86_REGISTER_CLASS_XMM:
    case LOOM_X86_REGISTER_CLASS_YMM:
    case LOOM_X86_REGISTER_CLASS_ZMM: {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(context->builder, "vmovdqa32 "));
      break;
    }
    case LOOM_X86_REGISTER_CLASS_GPR64: {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(context->builder, "mov "));
      break;
    }
    case LOOM_X86_REGISTER_CLASS_K: {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(context->builder, "kmovq "));
      break;
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_x86_append_assignment(state, context, result_assignment));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  return loom_x86_append_assignment(state, context, source_assignment);
}

static iree_status_t loom_x86_append_descriptor_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  const loom_x86_assembly_state_t* state =
      (const loom_x86_assembly_state_t*)user_data;
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  if (loom_low_const_isa(context->packet->node->op)) {
    return loom_x86_append_const_packet(state, context);
  }
  bool has_read_effect = false;
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_has_effect(
      descriptor_set, descriptor, LOOM_LOW_EFFECT_KIND_READ, &has_read_effect));
  bool has_write_effect = false;
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_has_effect(
      descriptor_set, descriptor, LOOM_LOW_EFFECT_KIND_WRITE,
      &has_write_effect));
  if (has_read_effect && has_write_effect) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_x86_descriptor_key(context, &key));
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 assembly descriptor '%.*s' has both read and "
                            "write effects",
                            (int)key.size, key.data);
  }
  if (has_read_effect) {
    return loom_x86_append_load_packet(state, context);
  }
  if (has_write_effect) {
    return loom_x86_append_store_packet(state, context);
  }
  bool uses_tied_binary_form = false;
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_uses_tied_binary_form(
      context, &uses_tied_binary_form));
  if (uses_tied_binary_form) {
    return loom_x86_append_tied_binary_packet(state, context);
  }
  bool uses_tied_ternary_form = false;
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_uses_tied_ternary_form(
      context, &uses_tied_ternary_form));
  if (uses_tied_ternary_form) {
    return loom_x86_append_tied_ternary_packet(state, context);
  }
  if (descriptor->stable_id ==
      X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_LEA_ADD_GPR64) {
    return loom_x86_append_lea_add_packet(state, context);
  }
  const loom_op_t* op = context->packet->node->op;
  if (descriptor->result_count == 1 && descriptor->operand_count == 3 &&
      op->result_count == 1 && op->operand_count == 2) {
    return loom_x86_append_binary_vector_packet(state, context);
  }
  if (descriptor->result_count == 1 && descriptor->operand_count == 2 &&
      op->result_count == 1 && op->operand_count == 1) {
    return loom_x86_append_move_packet(state, context);
  }
  iree_string_view_t key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_key(context, &key));
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "x86 assembly descriptor '%.*s' is unsupported",
                          (int)key.size, key.data);
}

static iree_status_t loom_x86_append_return_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  (void)user_data;
  loom_value_slice_t values = loom_low_return_values(context->packet->node->op);
  if (values.count != 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 assembly return values require ABI lowering");
  }
  return iree_string_builder_append_cstring(context->builder, "ret");
}

static iree_status_t loom_x86_append_branch_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  (void)user_data;
  const loom_op_t* op = context->packet->node->op;
  loom_value_slice_t args = loom_low_br_args(op);
  if (args.count != 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 assembly branch arguments require block "
                            "argument lowering");
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "jmp "));
  return loom_native_assembly_append_block_label(
      context->schedule, loom_low_br_dest(op), context->builder);
}

iree_status_t loom_x86_emit_assembly_fragment(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation,
    iree_string_builder_t* builder) {
  if (schedule == NULL || schedule->target.descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "x86 assembly schedule target is required");
  }
  iree_string_view_t target_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
      schedule->target.descriptor_set,
      schedule->target.descriptor_set->target_key_string_offset, &target_key));
  if (!iree_string_view_equal(target_key, IREE_SV("x86"))) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "x86 assembly emitter received target '%.*s'",
                            (int)target_key.size, target_key.data);
  }
  loom_x86_assembly_state_t state = {0};
  IREE_RETURN_IF_ERROR(loom_x86_resolve_descriptor_set_kind(
      schedule, &state.descriptor_set_kind));
  const loom_native_assembly_structural_packet_callback_t
      structural_packet_callbacks[] = {
          {
              .op_kind = LOOM_OP_LOW_COPY,
              .append_packet =
                  {
                      .fn = loom_x86_append_copy_packet,
                      .user_data = &state,
                  },
          },
          {
              .op_kind = LOOM_OP_LOW_RETURN,
              .append_packet =
                  {
                      .fn = loom_x86_append_return_packet,
                      .user_data = NULL,
                  },
          },
          {
              .op_kind = LOOM_OP_LOW_BR,
              .append_packet =
                  {
                      .fn = loom_x86_append_branch_packet,
                      .user_data = NULL,
                  },
          },
      };
  const loom_native_assembly_format_options_t options = {
      .append_descriptor_packet =
          {
              .fn = loom_x86_append_descriptor_packet,
              .user_data = &state,
          },
      .structural_packet_callbacks = structural_packet_callbacks,
      .structural_packet_callback_count =
          IREE_ARRAYSIZE(structural_packet_callbacks),
  };
  return loom_native_assembly_format_fragment(schedule, allocation, &options,
                                              builder);
}
