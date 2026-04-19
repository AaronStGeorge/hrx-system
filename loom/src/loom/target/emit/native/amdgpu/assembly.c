// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/assembly.h"

#include <inttypes.h>

#include "loom/codegen/low/packet.h"
#include "loom/ops/low/ops.h"
#include "loom/target/emit/native/assembly.h"

static iree_status_t loom_amdgpu_descriptor_key(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t* out_key) {
  return loom_native_assembly_descriptor_string(
      context->schedule->target.descriptor_set,
      context->packet->descriptor->key_string_offset, out_key);
}

static iree_status_t loom_amdgpu_append_mnemonic(
    const loom_native_assembly_packet_context_t* context) {
  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
      context->schedule->target.descriptor_set,
      context->packet->descriptor->mnemonic_string_offset, &mnemonic));
  return iree_string_builder_append_string(context->builder, mnemonic);
}

static iree_status_t loom_amdgpu_find_assignment(
    const loom_native_assembly_packet_context_t* context,
    loom_value_id_t value_id,
    const loom_low_allocation_assignment_t** out_assignment) {
  const loom_low_allocation_assignment_t* assignment =
      loom_low_packet_find_assignment(context->allocation, value_id, NULL);
  if (assignment == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU assembly value %" PRIu32
                            " has no allocation assignment",
                            value_id);
  }
  *out_assignment = assignment;
  return iree_ok_status();
}

static bool loom_amdgpu_assignments_match(
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  return lhs->location_kind == rhs->location_kind &&
         loom_liveness_value_class_equal(lhs->value_class, rhs->value_class) &&
         lhs->location_base == rhs->location_base &&
         lhs->location_count == rhs->location_count;
}

static iree_status_t loom_amdgpu_append_register_range(
    const loom_native_assembly_packet_context_t* context, const char* prefix,
    const loom_low_allocation_assignment_t* assignment) {
  if (assignment->location_count == 1) {
    return iree_string_builder_append_format(context->builder, "%s%" PRIu32,
                                             prefix, assignment->location_base);
  }
  const uint32_t last_register =
      assignment->location_base + assignment->location_count - 1;
  return iree_string_builder_append_format(
      context->builder, "%s[%" PRIu32 ":%" PRIu32 "]", prefix,
      assignment->location_base, last_register);
}

static iree_status_t loom_amdgpu_append_assignment(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_allocation_assignment_t* assignment) {
  if (assignment->location_kind !=
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU assembly value %" PRIu32
                            " is not physically allocated",
                            assignment->value_id);
  }
  if (assignment->location_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU assembly value %" PRIu32
                            " has an empty physical register range",
                            assignment->value_id);
  }
  iree_string_view_t register_class = loom_native_assembly_module_string(
      context->allocation->module, assignment->value_class.register_class_id);
  if (iree_string_view_equal(register_class, IREE_SV("amdgpu.sgpr"))) {
    return loom_amdgpu_append_register_range(context, "s", assignment);
  }
  if (iree_string_view_equal(register_class, IREE_SV("amdgpu.vgpr"))) {
    return loom_amdgpu_append_register_range(context, "v", assignment);
  }
  if (iree_string_view_equal(register_class, IREE_SV("amdgpu.agpr"))) {
    return loom_amdgpu_append_register_range(context, "acc", assignment);
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "AMDGPU assembly register class '%.*s' is unsupported",
      (int)register_class.size, register_class.data);
}

static iree_status_t loom_amdgpu_append_value(
    const loom_native_assembly_packet_context_t* context,
    loom_value_id_t value_id) {
  const loom_low_allocation_assignment_t* assignment = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_find_assignment(context, value_id, &assignment));
  return loom_amdgpu_append_assignment(context, assignment);
}

static iree_status_t loom_amdgpu_append_result(
    const loom_native_assembly_packet_context_t* context,
    iree_host_size_t result_index) {
  const loom_op_t* op = context->packet->node->op;
  if (result_index >= op->result_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU assembly result index is out of range");
  }
  return loom_amdgpu_append_value(context,
                                  loom_op_const_results(op)[result_index]);
}

static iree_status_t loom_amdgpu_append_operand(
    const loom_native_assembly_packet_context_t* context,
    iree_host_size_t operand_index) {
  const loom_op_t* op = context->packet->node->op;
  if (operand_index >= op->operand_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU assembly operand index is out of range");
  }
  return loom_amdgpu_append_value(context,
                                  loom_op_const_operands(op)[operand_index]);
}

static loom_named_attr_slice_t loom_amdgpu_packet_attrs(
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

static iree_status_t loom_amdgpu_read_packet_i64_attr(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t name, int64_t* out_value) {
  return loom_native_assembly_read_i64_attr(context->schedule->module,
                                            loom_amdgpu_packet_attrs(context),
                                            name, out_value);
}

static iree_status_t loom_amdgpu_append_comma(
    const loom_native_assembly_packet_context_t* context) {
  return iree_string_builder_append_cstring(context->builder, ", ");
}

static iree_status_t loom_amdgpu_append_result_operand_list(
    const loom_native_assembly_packet_context_t* context,
    iree_host_size_t result_count, iree_host_size_t operand_count) {
  bool needs_comma = false;
  for (iree_host_size_t i = 0; i < result_count; ++i) {
    if (needs_comma) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_result(context, i));
    needs_comma = true;
  }
  for (iree_host_size_t i = 0; i < operand_count; ++i) {
    if (needs_comma) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, i));
    needs_comma = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_basic_packet(
    const loom_native_assembly_packet_context_t* context,
    iree_host_size_t result_count, iree_host_size_t operand_count) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  if (result_count + operand_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  return loom_amdgpu_append_result_operand_list(context, result_count,
                                                operand_count);
}

static iree_status_t loom_amdgpu_append_offset_suffix(
    const loom_native_assembly_packet_context_t* context) {
  int64_t offset = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_read_packet_i64_attr(context, IREE_SV("offset"), &offset));
  return iree_string_builder_append_format(context->builder, " offset:%" PRId64,
                                           offset);
}

static iree_status_t loom_amdgpu_append_memory_packet(
    const loom_native_assembly_packet_context_t* context,
    iree_host_size_t result_count, iree_host_size_t operand_count) {
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_basic_packet(context, result_count, operand_count));
  return loom_amdgpu_append_offset_suffix(context);
}

static iree_status_t loom_amdgpu_append_s_mov_b32_packet(
    const loom_native_assembly_packet_context_t* context) {
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_read_packet_i64_attr(context, IREE_SV("imm32"), &value));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_result(context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  return iree_string_builder_append_format(context->builder, "%" PRId64, value);
}

static iree_status_t loom_amdgpu_append_waitcnt_packet(
    const loom_native_assembly_packet_context_t* context) {
  int64_t vmcnt = 0;
  int64_t lgkmcnt = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_read_packet_i64_attr(context, IREE_SV("vmcnt"), &vmcnt));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_read_packet_i64_attr(context, IREE_SV("lgkmcnt"), &lgkmcnt));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  return iree_string_builder_append_format(
      context->builder, " vmcnt(%" PRId64 ") lgkmcnt(%" PRId64 ")", vmcnt,
      lgkmcnt);
}

static iree_status_t loom_amdgpu_append_named_wait_packet(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t attr_name) {
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_read_packet_i64_attr(context, attr_name, &value));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  return iree_string_builder_append_format(
      context->builder, " %.*s(%" PRId64 ")", (int)attr_name.size,
      attr_name.data, value);
}

static iree_status_t loom_amdgpu_append_matrix_packet(
    const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  const loom_low_allocation_assignment_t* result_assignment = NULL;
  const loom_low_allocation_assignment_t* accumulator_assignment = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_find_assignment(
      context, loom_op_const_results(op)[0], &result_assignment));
  IREE_RETURN_IF_ERROR(loom_amdgpu_find_assignment(
      context, loom_op_const_operands(op)[2], &accumulator_assignment));
  if (!loom_amdgpu_assignments_match(result_assignment,
                                     accumulator_assignment)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU matrix result must share the accumulator physical register");
  }
  return loom_amdgpu_append_basic_packet(context, 1, 3);
}

static iree_status_t loom_amdgpu_append_descriptor_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  (void)user_data;
  iree_string_view_t key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_key(context, &key));
  if (iree_string_view_equal(key, IREE_SV("amdgpu.s_mov_b32"))) {
    return loom_amdgpu_append_s_mov_b32_packet(context);
  }
  if (iree_string_view_equal(key, IREE_SV("amdgpu.s_add_u32")) ||
      iree_string_view_equal(key, IREE_SV("amdgpu.v_add_u32")) ||
      iree_string_view_equal(key, IREE_SV("amdgpu.v_mul_lo_u32"))) {
    return loom_amdgpu_append_basic_packet(context, 1, 2);
  }
  if (iree_string_view_equal(key, IREE_SV("amdgpu.s_buffer_load_dword"))) {
    return loom_amdgpu_append_memory_packet(context, 1, 2);
  }
  if (iree_string_view_equal(key, IREE_SV("amdgpu.buffer_load_dword"))) {
    return loom_amdgpu_append_memory_packet(context, 1, 3);
  }
  if (iree_string_view_equal(key, IREE_SV("amdgpu.buffer_store_dword"))) {
    return loom_amdgpu_append_memory_packet(context, 0, 4);
  }
  if (iree_string_view_equal(key, IREE_SV("amdgpu.v_wmma_f32_16x16x16_f16")) ||
      iree_string_view_equal(key, IREE_SV("amdgpu.v_mfma_f32_16x16x16_f16"))) {
    return loom_amdgpu_append_matrix_packet(context);
  }
  if (iree_string_view_equal(key, IREE_SV("amdgpu.s_waitcnt"))) {
    return loom_amdgpu_append_waitcnt_packet(context);
  }
  if (iree_string_view_equal(key, IREE_SV("amdgpu.s_waitcnt_depctr")) ||
      iree_string_view_equal(key, IREE_SV("amdgpu.s_wait_alu"))) {
    return loom_amdgpu_append_named_wait_packet(context, IREE_SV("depctr"));
  }
  if (iree_string_view_equal(key, IREE_SV("amdgpu.s_wait_loadcnt"))) {
    return loom_amdgpu_append_named_wait_packet(context, IREE_SV("loadcnt"));
  }
  if (iree_string_view_equal(key, IREE_SV("amdgpu.s_wait_storecnt"))) {
    return loom_amdgpu_append_named_wait_packet(context, IREE_SV("storecnt"));
  }
  if (iree_string_view_equal(key, IREE_SV("amdgpu.s_wait_idle"))) {
    return loom_amdgpu_append_mnemonic(context);
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU assembly descriptor '%.*s' is unsupported",
                          (int)key.size, key.data);
}

static iree_status_t loom_amdgpu_append_return_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  (void)user_data;
  loom_value_slice_t values = loom_low_return_values(context->packet->node->op);
  if (values.count != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU assembly return values require ABI lowering");
  }
  return iree_string_builder_append_cstring(context->builder, "s_endpgm");
}

iree_status_t loom_amdgpu_emit_assembly_fragment(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation,
    iree_string_builder_t* builder) {
  if (schedule == NULL || schedule->target.descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU assembly schedule target is required");
  }
  iree_string_view_t target_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
      schedule->target.descriptor_set,
      schedule->target.descriptor_set->target_key_string_offset, &target_key));
  if (!iree_string_view_equal(target_key, IREE_SV("amdgpu"))) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU assembly emitter received target '%.*s'",
                            (int)target_key.size, target_key.data);
  }
  const loom_native_assembly_format_options_t options = {
      .append_descriptor_packet =
          {
              .fn = loom_amdgpu_append_descriptor_packet,
              .user_data = NULL,
          },
      .append_return_packet =
          {
              .fn = loom_amdgpu_append_return_packet,
              .user_data = NULL,
          },
  };
  return loom_native_assembly_format_fragment(schedule, allocation, &options,
                                              builder);
}
