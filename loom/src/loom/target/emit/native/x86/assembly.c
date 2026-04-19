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

static iree_status_t loom_x86_append_dot_packet(
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
        "x86 dot result must share the accumulator physical register");
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

static iree_status_t loom_x86_append_descriptor_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  (void)user_data;
  iree_string_view_t key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_key(context, &key));
  if (iree_string_view_equal(key, IREE_SV("x86.avx512.vpaddd.zmm")) ||
      iree_string_view_equal(key, IREE_SV("x86.avx512.kandq"))) {
    return loom_x86_append_binary_vector_packet(context);
  }
  if (iree_string_view_equal(key, IREE_SV("x86.avx512.vpdpbusd.zmm")) ||
      iree_string_view_equal(key, IREE_SV("x86.avx512.vdpbf16ps.zmm"))) {
    return loom_x86_append_dot_packet(context);
  }
  if (iree_string_view_equal(key, IREE_SV("x86.avx512.vmovdqu32.load.zmm"))) {
    return loom_x86_append_load_packet(context);
  }
  if (iree_string_view_equal(key, IREE_SV("x86.avx512.vmovdqu32.store.zmm"))) {
    return loom_x86_append_store_packet(context);
  }
  if (iree_string_view_equal(key, IREE_SV("x86.avx512.mov.gpr64"))) {
    return loom_x86_append_move_packet(context);
  }
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
  const loom_native_assembly_format_options_t options = {
      .append_descriptor_packet =
          {
              .fn = loom_x86_append_descriptor_packet,
              .user_data = NULL,
          },
      .append_return_packet =
          {
              .fn = loom_x86_append_return_packet,
              .user_data = NULL,
          },
      .append_branch_packet =
          {
              .fn = loom_x86_append_branch_packet,
              .user_data = NULL,
          },
  };
  return loom_native_assembly_format_fragment(schedule, allocation, &options,
                                              builder);
}
