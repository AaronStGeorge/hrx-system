// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/x86/assembly.h"

#include <inttypes.h>

#include "loom/codegen/low/packet.h"
#include "loom/ops/low/ops.h"
#include "loom/target/emit/native/assembly.h"

static const char* const kX86Gpr64Names[] = {
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
    "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15",
};

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
         loom_liveness_value_class_equal(lhs->value_class, rhs->value_class) &&
         lhs->location_base == rhs->location_base &&
         lhs->location_count == rhs->location_count;
}

static iree_status_t loom_x86_append_assignment(
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
  iree_string_view_t register_class = loom_native_assembly_module_string(
      context->allocation->module, assignment->value_class.register_class_id);
  if (iree_string_view_equal(register_class, IREE_SV("x86.gpr64"))) {
    if (assignment->location_base >= IREE_ARRAYSIZE(kX86Gpr64Names)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "x86 GPR index %" PRIu32 " is out of range",
                              assignment->location_base);
    }
    return iree_string_builder_append_cstring(
        context->builder, kX86Gpr64Names[assignment->location_base]);
  }
  if (iree_string_view_equal(register_class, IREE_SV("x86.xmm"))) {
    return iree_string_builder_append_format(context->builder, "xmm%" PRIu32,
                                             assignment->location_base);
  }
  if (iree_string_view_equal(register_class, IREE_SV("x86.ymm"))) {
    return iree_string_builder_append_format(context->builder, "ymm%" PRIu32,
                                             assignment->location_base);
  }
  if (iree_string_view_equal(register_class, IREE_SV("x86.zmm"))) {
    return iree_string_builder_append_format(context->builder, "zmm%" PRIu32,
                                             assignment->location_base);
  }
  if (iree_string_view_equal(register_class, IREE_SV("x86.k"))) {
    return iree_string_builder_append_format(context->builder, "k%" PRIu32,
                                             assignment->location_base);
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "x86 assembly register class '%.*s' is unsupported",
                          (int)register_class.size, register_class.data);
}

static iree_status_t loom_x86_append_value(
    const loom_native_assembly_packet_context_t* context,
    loom_value_id_t value_id) {
  const loom_low_allocation_assignment_t* assignment = NULL;
  IREE_RETURN_IF_ERROR(
      loom_x86_find_assignment(context, value_id, &assignment));
  return loom_x86_append_assignment(context, assignment);
}

static iree_status_t loom_x86_append_result(
    const loom_native_assembly_packet_context_t* context,
    iree_host_size_t result_index) {
  const loom_op_t* op = context->packet->node->op;
  if (result_index >= op->result_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 assembly result index is out of range");
  }
  return loom_x86_append_value(context,
                               loom_op_const_results(op)[result_index]);
}

static iree_status_t loom_x86_append_operand(
    const loom_native_assembly_packet_context_t* context,
    iree_host_size_t operand_index) {
  const loom_op_t* op = context->packet->node->op;
  if (operand_index >= op->operand_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 assembly operand index is out of range");
  }
  return loom_x86_append_value(context,
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

static iree_status_t loom_x86_append_memory_operand(
    const loom_native_assembly_packet_context_t* context,
    loom_value_id_t base_value_id, int64_t displacement) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "["));
  IREE_RETURN_IF_ERROR(loom_x86_append_value(context, base_value_id));
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
    const loom_native_assembly_packet_context_t* context) {
  IREE_RETURN_IF_ERROR(loom_x86_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_x86_append_result(context, 0));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  IREE_RETURN_IF_ERROR(loom_x86_append_operand(context, 0));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  return loom_x86_append_operand(context, 1);
}

static iree_status_t loom_x86_append_tied_ternary_packet(
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
  IREE_RETURN_IF_ERROR(loom_x86_append_result(context, 0));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  IREE_RETURN_IF_ERROR(loom_x86_append_operand(context, 1));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  return loom_x86_append_operand(context, 2);
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
    const loom_native_assembly_packet_context_t* context) {
  int64_t displacement = 0;
  IREE_RETURN_IF_ERROR(
      loom_x86_read_packet_i64_attr(context, IREE_SV("disp32"), &displacement));
  const loom_op_t* op = context->packet->node->op;
  IREE_RETURN_IF_ERROR(loom_x86_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_x86_append_result(context, 0));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  return loom_x86_append_memory_operand(context, loom_op_const_operands(op)[0],
                                        displacement);
}

static iree_status_t loom_x86_append_store_packet(
    const loom_native_assembly_packet_context_t* context) {
  int64_t displacement = 0;
  IREE_RETURN_IF_ERROR(
      loom_x86_read_packet_i64_attr(context, IREE_SV("disp32"), &displacement));
  const loom_op_t* op = context->packet->node->op;
  IREE_RETURN_IF_ERROR(loom_x86_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_x86_append_memory_operand(
      context, loom_op_const_operands(op)[1], displacement));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  return loom_x86_append_operand(context, 0);
}

static iree_status_t loom_x86_append_move_packet(
    const loom_native_assembly_packet_context_t* context) {
  IREE_RETURN_IF_ERROR(loom_x86_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_x86_append_result(context, 0));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  return loom_x86_append_operand(context, 0);
}

static iree_status_t loom_x86_append_copy_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  (void)user_data;
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

  iree_string_view_t source_register_class = loom_native_assembly_module_string(
      context->allocation->module,
      source_assignment->value_class.register_class_id);
  iree_string_view_t result_register_class = loom_native_assembly_module_string(
      context->allocation->module,
      result_assignment->value_class.register_class_id);
  if (!iree_string_view_equal(source_register_class, result_register_class)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "x86 assembly copy between register classes '%.*s' and '%.*s' is "
        "unsupported",
        (int)source_register_class.size, source_register_class.data,
        (int)result_register_class.size, result_register_class.data);
  }

  if (iree_string_view_equal(result_register_class, IREE_SV("x86.xmm")) ||
      iree_string_view_equal(result_register_class, IREE_SV("x86.ymm")) ||
      iree_string_view_equal(result_register_class, IREE_SV("x86.zmm"))) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "vmovdqa32 "));
  } else if (iree_string_view_equal(result_register_class,
                                    IREE_SV("x86.gpr64"))) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "mov "));
  } else if (iree_string_view_equal(result_register_class, IREE_SV("x86.k"))) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "kmovq "));
  } else {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 assembly copy register class '%.*s' is "
                            "unsupported",
                            (int)result_register_class.size,
                            result_register_class.data);
  }
  IREE_RETURN_IF_ERROR(loom_x86_append_assignment(context, result_assignment));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, ", "));
  return loom_x86_append_assignment(context, source_assignment);
}

static iree_status_t loom_x86_append_descriptor_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  (void)user_data;
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
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
    return loom_x86_append_load_packet(context);
  }
  if (has_write_effect) {
    return loom_x86_append_store_packet(context);
  }
  bool uses_tied_ternary_form = false;
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_uses_tied_ternary_form(
      context, &uses_tied_ternary_form));
  if (uses_tied_ternary_form) {
    return loom_x86_append_tied_ternary_packet(context);
  }
  const loom_op_t* op = context->packet->node->op;
  if (descriptor->result_count == 1 && descriptor->operand_count == 3 &&
      op->result_count == 1 && op->operand_count == 2) {
    return loom_x86_append_binary_vector_packet(context);
  }
  if (descriptor->result_count == 1 && descriptor->operand_count == 2 &&
      op->result_count == 1 && op->operand_count == 1) {
    return loom_x86_append_move_packet(context);
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

static const loom_native_assembly_structural_packet_callback_t
    kLoomX86AssemblyStructuralPacketCallbacks[] = {
        {
            .op_kind = LOOM_OP_LOW_COPY,
            .append_packet =
                {
                    .fn = loom_x86_append_copy_packet,
                    .user_data = NULL,
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
  const loom_native_assembly_format_options_t options = {
      .append_descriptor_packet =
          {
              .fn = loom_x86_append_descriptor_packet,
              .user_data = NULL,
          },
      .structural_packet_callbacks = kLoomX86AssemblyStructuralPacketCallbacks,
      .structural_packet_callback_count =
          IREE_ARRAYSIZE(kLoomX86AssemblyStructuralPacketCallbacks),
  };
  return loom_native_assembly_format_fragment(schedule, allocation, &options,
                                              builder);
}
