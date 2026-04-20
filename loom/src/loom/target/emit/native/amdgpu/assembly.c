// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/assembly.h"

#include <inttypes.h>

#include "loom/codegen/low/move_sequence.h"
#include "loom/codegen/low/packet.h"
#include "loom/ops/low/ops.h"
#include "loom/target/emit/native/assembly.h"

#define LOOM_AMDGPU_INLINE_MOVE_COUNT 16u

typedef struct loom_amdgpu_wait_packet_emit_state_t {
  // Wait-packet plan consumed in scheduled insertion order.
  const loom_amdgpu_wait_packet_plan_t* wait_packets;
  // Next wait-packet row to compare with the current scheduled packet.
  iree_host_size_t next_packet_index;
} loom_amdgpu_wait_packet_emit_state_t;

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

static iree_status_t loom_amdgpu_append_move_location(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_move_location_t* location) {
  if (location->location_kind !=
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU assembly move location is not physically allocated");
  }
  iree_string_view_t register_class = loom_native_assembly_module_string(
      context->allocation->module, location->value_class.register_class_id);
  if (iree_string_view_equal(register_class, IREE_SV("amdgpu.sgpr"))) {
    return iree_string_builder_append_format(context->builder, "s%" PRIu32,
                                             location->location);
  }
  if (iree_string_view_equal(register_class, IREE_SV("amdgpu.vgpr"))) {
    return iree_string_builder_append_format(context->builder, "v%" PRIu32,
                                             location->location);
  }
  if (iree_string_view_equal(register_class, IREE_SV("amdgpu.agpr"))) {
    return iree_string_builder_append_format(context->builder, "acc%" PRIu32,
                                             location->location);
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

static iree_status_t loom_amdgpu_append_mubuf_load_packet(
    const loom_native_assembly_packet_context_t* context) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_result(context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 1));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 2));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " offen"));
  return loom_amdgpu_append_offset_suffix(context);
}

static iree_status_t loom_amdgpu_append_mubuf_store_packet(
    const loom_native_assembly_packet_context_t* context) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 2));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 1));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 3));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " offen"));
  return loom_amdgpu_append_offset_suffix(context);
}

static iree_status_t loom_amdgpu_append_mov_b32_const_packet(
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

static iree_status_t loom_amdgpu_append_materialized_wait_packet(
    const loom_native_assembly_packet_context_t* context,
    const loom_amdgpu_wait_packet_t* wait_packet,
    const loom_amdgpu_wait_packet_plan_t* wait_packets) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  if (wait_packet->descriptor_ordinal >= descriptor_set->descriptor_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU wait packet descriptor ordinal %" PRIu32
                            " is out of range",
                            wait_packet->descriptor_ordinal);
  }
  if (wait_packet->immediate_start > wait_packets->immediate_count ||
      wait_packet->immediate_count >
          wait_packets->immediate_count - wait_packet->immediate_start) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU wait packet immediate range is out of range");
  }
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[wait_packet->descriptor_ordinal];
  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
      descriptor_set, descriptor->mnemonic_string_offset, &mnemonic));

  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "  "));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(context->builder, mnemonic));
  for (iree_host_size_t i = 0; i < wait_packet->immediate_count; ++i) {
    const iree_host_size_t immediate_index = wait_packet->immediate_start + i;
    const loom_amdgpu_wait_packet_immediate_t* immediate =
        &wait_packets->immediates[immediate_index];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        context->builder, " %.*s(%" PRIu16 ")", (int)immediate->name.size,
        immediate->name.data, immediate->value));
  }
  return iree_string_builder_append_cstring(context->builder, "\n");
}

static bool loom_amdgpu_wait_packet_is_before_node(
    const loom_amdgpu_wait_packet_t* wait_packet,
    const loom_low_schedule_node_t* node) {
  return wait_packet->block_index < node->block_index ||
         (wait_packet->block_index == node->block_index &&
          wait_packet->scheduled_ordinal < node->scheduled_ordinal);
}

static bool loom_amdgpu_wait_packet_matches_packet(
    const loom_amdgpu_wait_packet_t* wait_packet,
    const loom_low_packet_view_t* packet) {
  const loom_low_schedule_node_t* node = packet->node;
  return wait_packet->block_index == node->block_index &&
         wait_packet->scheduled_ordinal == node->scheduled_ordinal &&
         wait_packet->node_index == packet->node_index;
}

static iree_status_t loom_amdgpu_append_wait_packets_before_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  loom_amdgpu_wait_packet_emit_state_t* state =
      (loom_amdgpu_wait_packet_emit_state_t*)user_data;
  const loom_low_schedule_node_t* node = context->packet->node;
  while (state->next_packet_index < state->wait_packets->packet_count) {
    const loom_amdgpu_wait_packet_t* wait_packet =
        &state->wait_packets->packets[state->next_packet_index];
    if (loom_amdgpu_wait_packet_is_before_node(wait_packet, node)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU wait packet plan contains an insertion point before the "
          "current scheduled packet");
    }
    if (!loom_amdgpu_wait_packet_matches_packet(wait_packet, context->packet)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_materialized_wait_packet(
        context, wait_packet, state->wait_packets));
    ++state->next_packet_index;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_copy_mnemonic(
    iree_string_view_t register_class, iree_string_view_t* out_mnemonic) {
  if (iree_string_view_equal(register_class, IREE_SV("amdgpu.sgpr"))) {
    *out_mnemonic = IREE_SV("s_mov_b32");
    return iree_ok_status();
  }
  if (iree_string_view_equal(register_class, IREE_SV("amdgpu.vgpr"))) {
    *out_mnemonic = IREE_SV("v_mov_b32");
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU assembly copy register class '%.*s' is "
                          "unsupported",
                          (int)register_class.size, register_class.data);
}

typedef struct loom_amdgpu_assembly_move_state_t {
  // Packet context receiving assembly text.
  const loom_native_assembly_packet_context_t* context;
  // Target move mnemonic used for each emitted unit move.
  iree_string_view_t mnemonic;
  // Number of non-identity moves emitted so far.
  uint32_t emitted_count;
} loom_amdgpu_assembly_move_state_t;

static iree_status_t loom_amdgpu_append_move(
    void* user_data, const loom_low_move_location_t* destination,
    const loom_low_move_location_t* source) {
  loom_amdgpu_assembly_move_state_t* state =
      (loom_amdgpu_assembly_move_state_t*)user_data;
  const loom_native_assembly_packet_context_t* context = state->context;
  if (state->emitted_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "\n  "));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(context->builder, state->mnemonic));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_move_location(context, destination));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_move_location(context, source));
  ++state->emitted_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_move_sequence(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t mnemonic, loom_low_move_t* moves,
    iree_host_size_t move_count) {
  loom_amdgpu_assembly_move_state_t move_state = {
      .context = context,
      .mnemonic = mnemonic,
  };
  loom_low_move_sequence_options_t options = {
      .emit_move =
          {
              .fn = loom_amdgpu_append_move,
              .user_data = &move_state,
          },
  };
  return loom_low_move_sequence_emit(moves, move_count, &options);
}

static iree_status_t loom_amdgpu_append_copy_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  (void)user_data;
  const loom_op_t* op = context->packet->node->op;
  const loom_low_allocation_assignment_t* source_assignment = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_find_assignment(
      context, loom_low_copy_source(op), &source_assignment));
  const loom_low_allocation_assignment_t* result_assignment = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_find_assignment(
      context, loom_low_copy_result(op), &result_assignment));
  if (loom_amdgpu_assignments_match(source_assignment, result_assignment)) {
    return iree_ok_status();
  }
  if (source_assignment->location_kind !=
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
      result_assignment->location_kind !=
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU assembly copy requires physical register assignments");
  }
  if (source_assignment->location_count == 0 ||
      source_assignment->location_count != result_assignment->location_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU assembly copy requires non-empty matching register ranges");
  }
  const uint32_t register_count = source_assignment->location_count;
  const uint32_t last_register_offset = register_count - 1;
  if (source_assignment->location_base > UINT32_MAX - last_register_offset ||
      result_assignment->location_base > UINT32_MAX - last_register_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU assembly copy register range exceeds uint32_t");
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
        "AMDGPU assembly copy between register classes '%.*s' and '%.*s' is "
        "unsupported",
        (int)source_register_class.size, source_register_class.data,
        (int)result_register_class.size, result_register_class.data);
  }

  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_copy_mnemonic(result_register_class, &mnemonic));

  loom_low_move_t inline_moves[LOOM_AMDGPU_INLINE_MOVE_COUNT];
  loom_low_move_t* moves = inline_moves;
  iree_status_t status = iree_ok_status();
  if (register_count > IREE_ARRAYSIZE(inline_moves)) {
    status =
        iree_allocator_malloc_array(context->builder->allocator, register_count,
                                    sizeof(*moves), (void**)&moves);
  }
  if (iree_status_is_ok(status)) {
    for (uint32_t i = 0; i < register_count; ++i) {
      status = loom_low_move_location_from_assignment_unit(
          result_assignment, i, &moves[i].destination);
      if (!iree_status_is_ok(status)) {
        break;
      }
      status = loom_low_move_location_from_assignment_unit(source_assignment, i,
                                                           &moves[i].source);
      if (!iree_status_is_ok(status)) {
        break;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_emit_move_sequence(context, mnemonic, moves,
                                            register_count);
  }
  if (moves != inline_moves) {
    iree_allocator_free(context->builder->allocator, moves);
  }
  return status;
}

static iree_status_t loom_amdgpu_append_concat_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  (void)user_data;
  const loom_op_t* op = context->packet->node->op;
  const loom_low_allocation_assignment_t* result_assignment = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_find_assignment(
      context, loom_low_concat_result(op), &result_assignment));
  if (result_assignment->location_kind !=
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
      result_assignment->location_count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU assembly concat requires a non-empty physical result range");
  }

  iree_string_view_t result_register_class = loom_native_assembly_module_string(
      context->allocation->module,
      result_assignment->value_class.register_class_id);
  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_copy_mnemonic(result_register_class, &mnemonic));

  const uint32_t move_count = result_assignment->location_count;
  loom_low_move_t inline_moves[LOOM_AMDGPU_INLINE_MOVE_COUNT];
  loom_low_move_t* moves = inline_moves;
  iree_status_t status = iree_ok_status();
  if (move_count > IREE_ARRAYSIZE(inline_moves)) {
    status =
        iree_allocator_malloc_array(context->builder->allocator, move_count,
                                    sizeof(*moves), (void**)&moves);
  }

  uint32_t result_register_index = 0;
  loom_value_slice_t sources = loom_low_concat_sources(op);
  for (uint16_t i = 0; i < sources.count && iree_status_is_ok(status); ++i) {
    const loom_low_allocation_assignment_t* source_assignment = NULL;
    status = loom_amdgpu_find_assignment(context, sources.values[i],
                                         &source_assignment);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (source_assignment->location_kind !=
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
        source_assignment->location_count == 0) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU assembly concat requires non-empty physical source ranges");
      break;
    }
    iree_string_view_t source_register_class =
        loom_native_assembly_module_string(
            context->allocation->module,
            source_assignment->value_class.register_class_id);
    if (!iree_string_view_equal(source_register_class, result_register_class)) {
      status = iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "AMDGPU assembly concat between register classes '%.*s' and '%.*s' "
          "is unsupported",
          (int)source_register_class.size, source_register_class.data,
          (int)result_register_class.size, result_register_class.data);
      break;
    }
    if (result_register_index > result_assignment->location_count ||
        source_assignment->location_count >
            result_assignment->location_count - result_register_index) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU assembly concat source ranges exceed the result range");
      break;
    }
    for (uint32_t source_register_index = 0;
         source_register_index < source_assignment->location_count;
         ++source_register_index) {
      status = loom_low_move_location_from_assignment_unit(
          result_assignment, result_register_index,
          &moves[result_register_index].destination);
      if (!iree_status_is_ok(status)) {
        break;
      }
      status = loom_low_move_location_from_assignment_unit(
          source_assignment, source_register_index,
          &moves[result_register_index].source);
      if (!iree_status_is_ok(status)) {
        break;
      }
      ++result_register_index;
    }
  }
  if (iree_status_is_ok(status) &&
      result_register_index != result_assignment->location_count) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU assembly concat source ranges do not fill the result range");
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_amdgpu_emit_move_sequence(context, mnemonic, moves, move_count);
  }
  if (moves != inline_moves) {
    iree_allocator_free(context->builder->allocator, moves);
  }
  return status;
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

typedef enum loom_amdgpu_assembly_descriptor_kind_e {
  LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_BASIC,
  LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_MEMORY,
  LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_MNEMONIC,
  LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_MUBUF_LOAD,
  LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_MUBUF_STORE,
  LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_MATRIX,
  LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_NAMED_WAIT,
  LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_MOV_B32_CONST,
  LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_WAITCNT,
} loom_amdgpu_assembly_descriptor_kind_t;

typedef struct loom_amdgpu_assembly_descriptor_dispatch_t {
  // Stable descriptor key handled by this row.
  const char* key;
  // Formatting strategy used for this descriptor.
  loom_amdgpu_assembly_descriptor_kind_t kind;
  // Number of leading result values printed by basic/memory strategies.
  iree_host_size_t result_count;
  // Number of operand values printed by basic/memory strategies.
  iree_host_size_t operand_count;
  // Attribute name used by named wait descriptors.
  const char* named_wait_attr;
} loom_amdgpu_assembly_descriptor_dispatch_t;

static const loom_amdgpu_assembly_descriptor_dispatch_t
    kLoomAmdgpuAssemblyDescriptorDispatch[] = {
        {"amdgpu.buffer_load_dword", LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_MUBUF_LOAD,
         0, 0, NULL},
        {"amdgpu.buffer_store_dword",
         LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_MUBUF_STORE, 0, 0, NULL},
        {"amdgpu.s_add_u32", LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_BASIC, 1, 2, NULL},
        {"amdgpu.s_buffer_load_dword", LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_MEMORY,
         1, 2, NULL},
        {"amdgpu.s_load_dwordx2", LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_MEMORY, 1, 2,
         NULL},
        {"amdgpu.s_mov_b32", LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_MOV_B32_CONST, 0,
         0, NULL},
        {"amdgpu.s_wait_alu", LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_NAMED_WAIT, 0, 0,
         "depctr"},
        {"amdgpu.s_wait_idle", LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_MNEMONIC, 0, 0,
         NULL},
        {"amdgpu.s_wait_loadcnt", LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_NAMED_WAIT, 0,
         0, "loadcnt"},
        {"amdgpu.s_wait_storecnt", LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_NAMED_WAIT,
         0, 0, "storecnt"},
        {"amdgpu.s_waitcnt", LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_WAITCNT, 0, 0,
         NULL},
        {"amdgpu.s_waitcnt_depctr", LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_NAMED_WAIT,
         0, 0, "depctr"},
        {"amdgpu.v_add_u32", LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_BASIC, 1, 2, NULL},
        {"amdgpu.v_mov_b32", LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_MOV_B32_CONST, 0,
         0, NULL},
        {"amdgpu.v_mfma_f32_16x16x16_f16",
         LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_MATRIX, 0, 0, NULL},
        {"amdgpu.v_mul_lo_u32", LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_BASIC, 1, 2,
         NULL},
        {"amdgpu.v_wmma_f32_16x16x16_f16",
         LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_MATRIX, 0, 0, NULL},
};

static const loom_amdgpu_assembly_descriptor_dispatch_t*
loom_amdgpu_lookup_assembly_descriptor_dispatch(iree_string_view_t key) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuAssemblyDescriptorDispatch); ++i) {
    const loom_amdgpu_assembly_descriptor_dispatch_t* row =
        &kLoomAmdgpuAssemblyDescriptorDispatch[i];
    if (iree_string_view_equal(key, iree_make_cstring_view(row->key))) {
      return row;
    }
  }
  return NULL;
}

static iree_status_t loom_amdgpu_append_descriptor_dispatch_packet(
    const loom_amdgpu_assembly_descriptor_dispatch_t* row,
    const loom_native_assembly_packet_context_t* context) {
  switch (row->kind) {
    case LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_BASIC:
      return loom_amdgpu_append_basic_packet(context, row->result_count,
                                             row->operand_count);
    case LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_MEMORY:
      return loom_amdgpu_append_memory_packet(context, row->result_count,
                                              row->operand_count);
    case LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_MNEMONIC:
      return loom_amdgpu_append_mnemonic(context);
    case LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_MUBUF_LOAD:
      return loom_amdgpu_append_mubuf_load_packet(context);
    case LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_MUBUF_STORE:
      return loom_amdgpu_append_mubuf_store_packet(context);
    case LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_MATRIX:
      return loom_amdgpu_append_matrix_packet(context);
    case LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_NAMED_WAIT:
      return loom_amdgpu_append_named_wait_packet(
          context, iree_make_cstring_view(row->named_wait_attr));
    case LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_MOV_B32_CONST:
      return loom_amdgpu_append_mov_b32_const_packet(context);
    case LOOM_AMDGPU_ASSEMBLY_DESCRIPTOR_WAITCNT:
      return loom_amdgpu_append_waitcnt_packet(context);
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "AMDGPU assembly descriptor row kind is invalid");
  }
}

static iree_status_t loom_amdgpu_append_descriptor_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  (void)user_data;
  iree_string_view_t key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_key(context, &key));
  const loom_amdgpu_assembly_descriptor_dispatch_t* row =
      loom_amdgpu_lookup_assembly_descriptor_dispatch(key);
  if (row != NULL) {
    return loom_amdgpu_append_descriptor_dispatch_packet(row, context);
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

static iree_status_t loom_amdgpu_verify_assembly_target(
    const loom_low_schedule_sidecar_t* schedule) {
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
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_verify_wait_packet_plan(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_amdgpu_wait_packet_plan_t* wait_packets) {
  if (wait_packets == NULL || wait_packets->wait_plan == NULL ||
      wait_packets->wait_plan->schedule != schedule) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU assembly wait packets must be derived from the emitted "
        "schedule");
  }
  if (wait_packets->packet_count != 0 && wait_packets->packets == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU assembly wait packet rows are required");
  }
  if (wait_packets->immediate_count != 0 && wait_packets->immediates == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU assembly wait packet immediate rows are required");
  }
  return iree_ok_status();
}

static const loom_native_assembly_structural_packet_callback_t
    kLoomAmdgpuAssemblyStructuralPacketCallbacks[] = {
        {
            .op_kind = LOOM_OP_LOW_COPY,
            .append_packet =
                {
                    .fn = loom_amdgpu_append_copy_packet,
                    .user_data = NULL,
                },
        },
        {
            .op_kind = LOOM_OP_LOW_CONCAT,
            .append_packet =
                {
                    .fn = loom_amdgpu_append_concat_packet,
                    .user_data = NULL,
                },
        },
        {
            .op_kind = LOOM_OP_LOW_RETURN,
            .append_packet =
                {
                    .fn = loom_amdgpu_append_return_packet,
                    .user_data = NULL,
                },
        },
};

static loom_native_assembly_format_options_t loom_amdgpu_assembly_options(
    loom_native_assembly_append_packet_callback_t append_before_packet) {
  return (loom_native_assembly_format_options_t){
      .append_before_packet = append_before_packet,
      .append_descriptor_packet =
          {
              .fn = loom_amdgpu_append_descriptor_packet,
              .user_data = NULL,
          },
      .structural_packet_callbacks =
          kLoomAmdgpuAssemblyStructuralPacketCallbacks,
      .structural_packet_callback_count =
          IREE_ARRAYSIZE(kLoomAmdgpuAssemblyStructuralPacketCallbacks),
  };
}

iree_status_t loom_amdgpu_emit_assembly_fragment(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_verify_assembly_target(schedule));
  const loom_native_assembly_format_options_t options =
      loom_amdgpu_assembly_options(
          (loom_native_assembly_append_packet_callback_t){0});
  return loom_native_assembly_format_fragment(schedule, allocation, &options,
                                              builder);
}

iree_status_t loom_amdgpu_emit_assembly_fragment_with_wait_packets(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation,
    const loom_amdgpu_wait_packet_plan_t* wait_packets,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_verify_assembly_target(schedule));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_verify_wait_packet_plan(schedule, wait_packets));

  loom_amdgpu_wait_packet_emit_state_t wait_state = {
      .wait_packets = wait_packets,
  };
  const loom_native_assembly_format_options_t options =
      loom_amdgpu_assembly_options(
          (loom_native_assembly_append_packet_callback_t){
              .fn = loom_amdgpu_append_wait_packets_before_packet,
              .user_data = &wait_state,
          });
  IREE_RETURN_IF_ERROR(loom_native_assembly_format_fragment(
      schedule, allocation, &options, builder));
  if (wait_state.next_packet_index != wait_packets->packet_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU assembly wait packet plan contains an unmatched insertion "
        "point");
  }
  return iree_ok_status();
}
