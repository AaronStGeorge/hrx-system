// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/ireevm/function_bytecode.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/packet.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/emit/ireevm/descriptors.h"

enum {
  // VM bytecode opcodes from
  // runtime/src/iree/vm/bytecode/utils/generated/op_table.h.
  LOOM_IREEVM_OP_CONST_I32 = 0x0D,
  LOOM_IREEVM_OP_ADD_I32 = 0x22,
  LOOM_IREEVM_OP_SUB_I32 = 0x23,
  LOOM_IREEVM_OP_CMP_EQ_I32 = 0x49,
  LOOM_IREEVM_OP_BRANCH = 0x56,
  LOOM_IREEVM_OP_COND_BRANCH = 0x57,
  LOOM_IREEVM_OP_CALL = 0x58,
  LOOM_IREEVM_OP_RETURN = 0x5A,
  LOOM_IREEVM_OP_BLOCK = 0x79,
};

#define LOOM_IREEVM_IMPORT_ORDINAL_BIT UINT32_C(0x80000000)
#define LOOM_IREEVM_MAX_BLOCK_BYTE_LENGTH UINT32_C(0x00FFFFFF)

typedef struct loom_ireevm_branch_fixup_t {
  // Target schedule block whose bytecode offset patches |offset|.
  uint32_t block_index;
  // Byte offset of the little-endian u32 branch-target field.
  iree_host_size_t offset;
} loom_ireevm_branch_fixup_t;

typedef struct loom_ireevm_bytecode_writer_t {
  // Allocator used for bytecode and side arrays.
  iree_allocator_t allocator;
  // Mutable bytecode storage.
  uint8_t* data;
  // Number of initialized bytes in |data|.
  iree_host_size_t length;
  // Allocated byte capacity of |data|.
  iree_host_size_t capacity;
  // Bytecode offsets for each schedule block.
  iree_host_size_t* block_offsets;
  // Number of entries in |block_offsets|.
  iree_host_size_t block_offset_count;
  // Branch target fixups recorded while emitting.
  loom_ireevm_branch_fixup_t* fixups;
  // Number of initialized fixup entries.
  iree_host_size_t fixup_count;
  // Allocated fixup entry capacity.
  iree_host_size_t fixup_capacity;
} loom_ireevm_bytecode_writer_t;

typedef struct loom_ireevm_emit_state_t {
  // Schedule table being emitted.
  const loom_low_schedule_table_t* schedule;
  // Allocation table supplying VM register ordinals.
  const loom_low_allocation_table_t* allocation;
  // Mutable bytecode writer.
  loom_ireevm_bytecode_writer_t writer;
  // Maximum i32 register ordinal plus one.
  uint32_t i32_register_count;
} loom_ireevm_emit_state_t;

static iree_string_view_t loom_ireevm_module_string(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static void loom_ireevm_bytecode_writer_deinitialize(
    loom_ireevm_bytecode_writer_t* writer) {
  iree_allocator_free(writer->allocator, writer->data);
  iree_allocator_free(writer->allocator, writer->block_offsets);
  iree_allocator_free(writer->allocator, writer->fixups);
  *writer = (loom_ireevm_bytecode_writer_t){0};
}

static iree_status_t loom_ireevm_bytecode_writer_initialize(
    iree_host_size_t block_count, iree_host_size_t fixup_capacity,
    iree_allocator_t allocator, loom_ireevm_bytecode_writer_t* out_writer) {
  *out_writer = (loom_ireevm_bytecode_writer_t){
      .allocator = allocator,
      .block_offset_count = block_count,
      .fixup_capacity = fixup_capacity,
  };
  iree_status_t status = iree_ok_status();
  if (block_count > 0) {
    status = iree_allocator_malloc_array(allocator, block_count,
                                         sizeof(*out_writer->block_offsets),
                                         (void**)&out_writer->block_offsets);
  }
  if (iree_status_is_ok(status) && fixup_capacity > 0) {
    status = iree_allocator_malloc_array(allocator, fixup_capacity,
                                         sizeof(*out_writer->fixups),
                                         (void**)&out_writer->fixups);
  }
  if (!iree_status_is_ok(status)) {
    loom_ireevm_bytecode_writer_deinitialize(out_writer);
  }
  return status;
}

static iree_status_t loom_ireevm_bytecode_writer_reserve(
    loom_ireevm_bytecode_writer_t* writer, iree_host_size_t additional_length) {
  iree_host_size_t minimum_capacity = 0;
  if (!iree_host_size_checked_add(writer->length, additional_length,
                                  &minimum_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM bytecode length overflow");
  }
  if (minimum_capacity <= writer->capacity) {
    return iree_ok_status();
  }

  iree_host_size_t doubled_capacity =
      writer->capacity ? writer->capacity * 2 : 128;
  if (doubled_capacity < writer->capacity ||
      doubled_capacity < minimum_capacity) {
    doubled_capacity = minimum_capacity;
  }
  IREE_RETURN_IF_ERROR(iree_allocator_realloc(
      writer->allocator, doubled_capacity, (void**)&writer->data));
  writer->capacity = doubled_capacity;
  return iree_ok_status();
}

static iree_status_t loom_ireevm_bytecode_write_u8(
    loom_ireevm_bytecode_writer_t* writer, uint8_t value) {
  IREE_RETURN_IF_ERROR(loom_ireevm_bytecode_writer_reserve(writer, 1));
  writer->data[writer->length++] = value;
  return iree_ok_status();
}

static iree_status_t loom_ireevm_bytecode_write_u16(
    loom_ireevm_bytecode_writer_t* writer, uint16_t value) {
  IREE_RETURN_IF_ERROR(loom_ireevm_bytecode_writer_reserve(writer, 2));
  writer->data[writer->length++] = (uint8_t)(value & 0xFFu);
  writer->data[writer->length++] = (uint8_t)(value >> 8);
  return iree_ok_status();
}

static iree_status_t loom_ireevm_bytecode_write_u32(
    loom_ireevm_bytecode_writer_t* writer, uint32_t value) {
  IREE_RETURN_IF_ERROR(loom_ireevm_bytecode_writer_reserve(writer, 4));
  writer->data[writer->length++] = (uint8_t)(value & 0xFFu);
  writer->data[writer->length++] = (uint8_t)((value >> 8) & 0xFFu);
  writer->data[writer->length++] = (uint8_t)((value >> 16) & 0xFFu);
  writer->data[writer->length++] = (uint8_t)(value >> 24);
  return iree_ok_status();
}

static iree_status_t loom_ireevm_bytecode_align(
    loom_ireevm_bytecode_writer_t* writer, iree_host_size_t alignment) {
  iree_host_size_t aligned_length = 0;
  if (!iree_host_size_checked_align(writer->length, alignment,
                                    &aligned_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM bytecode alignment overflow");
  }
  iree_host_size_t padding_length = aligned_length - writer->length;
  if (padding_length == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_ireevm_bytecode_writer_reserve(writer, padding_length));
  memset(writer->data + writer->length, 0, padding_length);
  writer->length = aligned_length;
  return iree_ok_status();
}

static void loom_ireevm_bytecode_patch_u32(
    loom_ireevm_bytecode_writer_t* writer, iree_host_size_t offset,
    uint32_t value) {
  writer->data[offset + 0] = (uint8_t)(value & 0xFFu);
  writer->data[offset + 1] = (uint8_t)((value >> 8) & 0xFFu);
  writer->data[offset + 2] = (uint8_t)((value >> 16) & 0xFFu);
  writer->data[offset + 3] = (uint8_t)(value >> 24);
}

static iree_status_t loom_ireevm_bytecode_record_branch_fixup(
    loom_ireevm_bytecode_writer_t* writer, uint32_t block_index) {
  if (writer->fixup_count >= writer->fixup_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many VM bytecode branch fixups");
  }
  writer->fixups[writer->fixup_count++] = (loom_ireevm_branch_fixup_t){
      .block_index = block_index,
      .offset = writer->length,
  };
  return iree_ok_status();
}

static iree_status_t loom_ireevm_bytecode_fixup_branch_targets(
    loom_ireevm_bytecode_writer_t* writer) {
  for (iree_host_size_t i = 0; i < writer->fixup_count; ++i) {
    const loom_ireevm_branch_fixup_t* fixup = &writer->fixups[i];
    if (fixup->block_index >= writer->block_offset_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "VM bytecode branch target block is invalid");
    }
    iree_host_size_t block_offset = writer->block_offsets[fixup->block_index];
    if (block_offset > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "VM bytecode branch target offset exceeds u32");
    }
    loom_ireevm_bytecode_patch_u32(writer, fixup->offset,
                                   (uint32_t)block_offset);
  }
  return iree_ok_status();
}

static iree_status_t loom_ireevm_validate_i32_register_assignment(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_assignment_t* assignment,
    uint16_t* out_register) {
  if (assignment->location_kind != LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "VM bytecode value %u is not allocated to a VM target id",
        (unsigned)assignment->value_id);
  }
  if (assignment->location_count != 1 || assignment->unit_count != 1) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "VM bytecode value %u uses a multi-unit register assignment",
        (unsigned)assignment->value_id);
  }
  if (assignment->value_class.type_kind != LOOM_TYPE_REGISTER) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "VM bytecode value %u is not allocated as a register value",
        (unsigned)assignment->value_id);
  }
  if (assignment->descriptor_reg_class_id != IREE_VM_CORE_REG_CLASS_ID_VM_I32) {
    iree_string_view_t register_class = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_register_class_name(
        allocation, assignment, &register_class));
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "VM bytecode emission only supports vm.i32 values, got '%.*s'",
        (int)register_class.size, register_class.data);
  }
  if (assignment->location_base > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM bytecode register ordinal exceeds u16");
  }
  *out_register = (uint16_t)assignment->location_base;
  return iree_ok_status();
}

static iree_status_t loom_ireevm_analyze_register_metadata(
    const loom_low_allocation_table_t* allocation,
    uint32_t* out_i32_register_count) {
  uint32_t i32_register_count = 0;
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    uint16_t reg = 0;
    IREE_RETURN_IF_ERROR(loom_ireevm_validate_i32_register_assignment(
        allocation, assignment, &reg));
    uint32_t register_limit = (uint32_t)reg + 1;
    if (register_limit > i32_register_count) {
      i32_register_count = register_limit;
    }
  }
  *out_i32_register_count = i32_register_count;
  return iree_ok_status();
}

static iree_status_t loom_ireevm_lookup_i32_register(
    loom_ireevm_emit_state_t* state, loom_value_id_t value_id,
    uint16_t* out_register) {
  IREE_ASSERT_ARGUMENT(out_register);
  const loom_low_allocation_assignment_t* assignment =
      loom_low_allocation_map_active_value_assignment(state->allocation,
                                                      value_id, NULL);
  return loom_ireevm_validate_i32_register_assignment(state->allocation,
                                                      assignment, out_register);
}

static const loom_named_attr_t* loom_ireevm_find_named_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (iree_string_view_equal(loom_ireevm_module_string(module, attr->name_id),
                               name)) {
      return attr;
    }
  }
  return NULL;
}

static iree_status_t loom_ireevm_read_i64_attr(const loom_module_t* module,
                                               loom_named_attr_slice_t attrs,
                                               iree_string_view_t name,
                                               int64_t* out_value) {
  const loom_named_attr_t* attr =
      loom_ireevm_find_named_attr(module, attrs, name);
  if (!attr) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "VM bytecode missing required '%.*s' attribute",
                            (int)name.size, name.data);
  }
  if (attr->value.kind != LOOM_ATTR_I64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM bytecode attribute '%.*s' must be i64",
                            (int)name.size, name.data);
  }
  *out_value = attr->value.i64;
  return iree_ok_status();
}

static iree_status_t loom_ireevm_write_register_list(
    loom_ireevm_emit_state_t* state, const loom_value_id_t* values,
    iree_host_size_t value_count) {
  if (value_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM register list exceeds u16 entries");
  }
  loom_ireevm_bytecode_writer_t* writer = &state->writer;
  IREE_RETURN_IF_ERROR(loom_ireevm_bytecode_align(writer, 2));
  IREE_RETURN_IF_ERROR(
      loom_ireevm_bytecode_write_u16(writer, (uint16_t)value_count));
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    uint16_t reg = 0;
    IREE_RETURN_IF_ERROR(
        loom_ireevm_lookup_i32_register(state, values[i], &reg));
    IREE_RETURN_IF_ERROR(loom_ireevm_bytecode_write_u16(writer, reg));
  }
  return iree_ok_status();
}

static iree_status_t loom_ireevm_write_branch_target(
    loom_ireevm_emit_state_t* state, const loom_block_t* target,
    const loom_value_id_t* values, iree_host_size_t value_count) {
  uint32_t block_index = loom_low_packet_block_index(state->schedule, target);
  if (block_index == LOOM_LOW_PACKET_INDEX_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM bytecode branch target is outside function");
  }
  if (value_count != target->arg_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VM bytecode branch operand count does not match target block args");
  }

  loom_ireevm_bytecode_writer_t* writer = &state->writer;
  IREE_RETURN_IF_ERROR(
      loom_ireevm_bytecode_record_branch_fixup(writer, block_index));
  IREE_RETURN_IF_ERROR(loom_ireevm_bytecode_write_u32(writer, 0));

  uint16_t remap_count = 0;
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    uint16_t source_reg = 0;
    uint16_t target_reg = 0;
    IREE_RETURN_IF_ERROR(
        loom_ireevm_lookup_i32_register(state, values[i], &source_reg));
    IREE_RETURN_IF_ERROR(loom_ireevm_lookup_i32_register(
        state, loom_block_arg_id(target, (uint16_t)i), &target_reg));
    if (source_reg != target_reg) {
      ++remap_count;
    }
  }

  IREE_RETURN_IF_ERROR(loom_ireevm_bytecode_align(writer, 2));
  IREE_RETURN_IF_ERROR(loom_ireevm_bytecode_write_u16(writer, remap_count));
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    uint16_t source_reg = 0;
    uint16_t target_reg = 0;
    IREE_RETURN_IF_ERROR(
        loom_ireevm_lookup_i32_register(state, values[i], &source_reg));
    IREE_RETURN_IF_ERROR(loom_ireevm_lookup_i32_register(
        state, loom_block_arg_id(target, (uint16_t)i), &target_reg));
    if (source_reg != target_reg) {
      IREE_RETURN_IF_ERROR(loom_ireevm_bytecode_write_u16(writer, source_reg));
      IREE_RETURN_IF_ERROR(loom_ireevm_bytecode_write_u16(writer, target_reg));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_ireevm_emit_const_i32(
    loom_ireevm_emit_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  if (!loom_low_const_isa(op) || op->result_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "iree.vm.const.i32 must be a unary low.const");
  }
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(loom_ireevm_read_i64_attr(state->schedule->module,
                                                 loom_low_const_attrs(op),
                                                 IREE_SV("i32_value"), &value));
  if (value < INT32_MIN || value > INT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "iree.vm.const.i32 value is outside i32 range");
  }
  uint16_t result_reg = 0;
  IREE_RETURN_IF_ERROR(loom_ireevm_lookup_i32_register(
      state, loom_low_const_result(op), &result_reg));
  IREE_RETURN_IF_ERROR(loom_ireevm_bytecode_write_u8(
      &state->writer, (uint8_t)descriptor->encoding_id));
  IREE_RETURN_IF_ERROR(
      loom_ireevm_bytecode_write_u32(&state->writer, (uint32_t)value));
  return loom_ireevm_bytecode_write_u16(&state->writer, result_reg);
}

static iree_status_t loom_ireevm_emit_binary_i32(
    loom_ireevm_emit_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  if (!loom_low_op_isa(op) || op->operand_count != 2 || op->result_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM binary i32 packet shape is invalid");
  }
  loom_value_slice_t operands = loom_low_op_operands(op);
  loom_value_slice_t results = loom_low_op_results(op);
  uint16_t lhs_reg = 0;
  uint16_t rhs_reg = 0;
  uint16_t result_reg = 0;
  IREE_RETURN_IF_ERROR(
      loom_ireevm_lookup_i32_register(state, operands.values[0], &lhs_reg));
  IREE_RETURN_IF_ERROR(
      loom_ireevm_lookup_i32_register(state, operands.values[1], &rhs_reg));
  IREE_RETURN_IF_ERROR(
      loom_ireevm_lookup_i32_register(state, results.values[0], &result_reg));
  IREE_RETURN_IF_ERROR(loom_ireevm_bytecode_write_u8(
      &state->writer, (uint8_t)descriptor->encoding_id));
  IREE_RETURN_IF_ERROR(loom_ireevm_bytecode_write_u16(&state->writer, lhs_reg));
  IREE_RETURN_IF_ERROR(loom_ireevm_bytecode_write_u16(&state->writer, rhs_reg));
  return loom_ireevm_bytecode_write_u16(&state->writer, result_reg);
}

static iree_status_t loom_ireevm_emit_call_import_i32(
    loom_ireevm_emit_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  if (!loom_low_op_isa(op) || op->operand_count != 1 || op->result_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "iree.vm.call.import.i32 packet shape is invalid");
  }
  int64_t callee_ordinal = 0;
  IREE_RETURN_IF_ERROR(
      loom_ireevm_read_i64_attr(state->schedule->module, loom_low_op_attrs(op),
                                IREE_SV("callee_ordinal"), &callee_ordinal));
  if (callee_ordinal < 0 || callee_ordinal > INT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM import callee ordinal is outside i31 range");
  }

  IREE_RETURN_IF_ERROR(loom_ireevm_bytecode_write_u8(
      &state->writer, (uint8_t)descriptor->encoding_id));
  IREE_RETURN_IF_ERROR(loom_ireevm_bytecode_write_u32(
      &state->writer,
      ((uint32_t)callee_ordinal) | LOOM_IREEVM_IMPORT_ORDINAL_BIT));
  loom_value_slice_t operands = loom_low_op_operands(op);
  loom_value_slice_t results = loom_low_op_results(op);
  IREE_RETURN_IF_ERROR(
      loom_ireevm_write_register_list(state, operands.values, operands.count));
  return loom_ireevm_write_register_list(state, results.values, results.count);
}

static iree_status_t loom_ireevm_emit_descriptor_packet(
    loom_ireevm_emit_state_t* state, const loom_low_packet_view_t* packet) {
  switch (packet->descriptor->encoding_id) {
    case LOOM_IREEVM_OP_CONST_I32:
      return loom_ireevm_emit_const_i32(state, packet->node->op,
                                        packet->descriptor);
    case LOOM_IREEVM_OP_ADD_I32:
    case LOOM_IREEVM_OP_SUB_I32:
    case LOOM_IREEVM_OP_CMP_EQ_I32:
      return loom_ireevm_emit_binary_i32(state, packet->node->op,
                                         packet->descriptor);
    case LOOM_IREEVM_OP_CALL:
      return loom_ireevm_emit_call_import_i32(state, packet->node->op,
                                              packet->descriptor);
    default: {
      iree_string_view_t key = IREE_SV("<unknown>");
      (void)loom_low_descriptor_set_string(
          state->schedule->target.descriptor_set,
          packet->descriptor->key_string_offset, &key);
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "VM bytecode descriptor '%.*s' is unsupported",
                              (int)key.size, key.data);
    }
  }
}

static iree_status_t loom_ireevm_emit_low_return(
    loom_ireevm_emit_state_t* state, const loom_op_t* op) {
  IREE_RETURN_IF_ERROR(
      loom_ireevm_bytecode_write_u8(&state->writer, LOOM_IREEVM_OP_RETURN));
  loom_value_slice_t values = loom_low_return_values(op);
  return loom_ireevm_write_register_list(state, values.values, values.count);
}

static iree_status_t loom_ireevm_emit_low_br(loom_ireevm_emit_state_t* state,
                                             const loom_op_t* op) {
  IREE_RETURN_IF_ERROR(
      loom_ireevm_bytecode_write_u8(&state->writer, LOOM_IREEVM_OP_BRANCH));
  loom_value_slice_t args = loom_low_br_args(op);
  return loom_ireevm_write_branch_target(state, loom_low_br_dest(op),
                                         args.values, args.count);
}

static iree_status_t loom_ireevm_emit_low_cond_br(
    loom_ireevm_emit_state_t* state, const loom_op_t* op) {
  uint16_t condition_reg = 0;
  IREE_RETURN_IF_ERROR(loom_ireevm_lookup_i32_register(
      state, loom_low_cond_br_condition(op), &condition_reg));
  IREE_RETURN_IF_ERROR(loom_ireevm_bytecode_write_u8(
      &state->writer, LOOM_IREEVM_OP_COND_BRANCH));
  IREE_RETURN_IF_ERROR(
      loom_ireevm_bytecode_write_u16(&state->writer, condition_reg));
  IREE_RETURN_IF_ERROR(loom_ireevm_write_branch_target(
      state, loom_low_cond_br_true_dest(op), NULL, 0));
  return loom_ireevm_write_branch_target(state, loom_low_cond_br_false_dest(op),
                                         NULL, 0);
}

static iree_status_t loom_ireevm_emit_structural_packet(
    loom_ireevm_emit_state_t* state, const loom_op_t* op) {
  if (loom_low_return_isa(op)) {
    return loom_ireevm_emit_low_return(state, op);
  }
  if (loom_low_br_isa(op)) {
    return loom_ireevm_emit_low_br(state, op);
  }
  if (loom_low_cond_br_isa(op)) {
    return loom_ireevm_emit_low_cond_br(state, op);
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "unsupported structural low op in VM bytecode");
}

static iree_status_t loom_ireevm_emit_packet(
    loom_ireevm_emit_state_t* state, const loom_low_packet_view_t* packet) {
  if (packet->descriptor) {
    return loom_ireevm_emit_descriptor_packet(state, packet);
  }
  return loom_ireevm_emit_structural_packet(state, packet->node->op);
}

static iree_status_t loom_ireevm_validate_tables(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation) {
  if (!schedule || !allocation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "schedule and allocation tables are required");
  }
  IREE_RETURN_IF_ERROR(loom_low_packet_validate_tables(schedule, allocation));
  if (schedule->target.descriptor_set != loom_ireevm_core_descriptor_set()) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM bytecode emission requires iree.vm.core");
  }
  if (schedule->block_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM bytecode block count exceeds u16");
  }
  if (allocation->spill_count != 0 || allocation->spill_plan_count != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "VM bytecode emission requires unspilled allocation tables");
  }
  return iree_ok_status();
}

static iree_status_t loom_ireevm_emit_function_body(
    loom_ireevm_emit_state_t* state) {
  loom_ireevm_bytecode_writer_t* writer = &state->writer;
  for (iree_host_size_t block_index = 0;
       block_index < state->schedule->block_count; ++block_index) {
    const loom_low_schedule_block_t* block =
        &state->schedule->blocks[block_index];
    if (writer->length > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "VM bytecode block offset exceeds u32");
    }
    writer->block_offsets[block_index] = writer->length;
    IREE_RETURN_IF_ERROR(
        loom_ireevm_bytecode_write_u8(writer, LOOM_IREEVM_OP_BLOCK));
    for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
      iree_host_size_t packet_index =
          (iree_host_size_t)block->scheduled_node_start + i;
      loom_low_packet_view_t packet = {0};
      IREE_RETURN_IF_ERROR(loom_low_packet_view_at(
          state->schedule, state->allocation, packet_index, &packet));
      IREE_RETURN_IF_ERROR(loom_ireevm_emit_packet(state, &packet));
    }
    iree_host_size_t block_length =
        writer->length - writer->block_offsets[block_index];
    if (block_length > LOOM_IREEVM_MAX_BLOCK_BYTE_LENGTH) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "VM bytecode block exceeds maximum length");
    }
  }
  return iree_ok_status();
}

void loom_ireevm_function_bytecode_deinitialize(
    loom_ireevm_function_bytecode_t* bytecode, iree_allocator_t allocator) {
  if (!bytecode) {
    return;
  }
  iree_allocator_free(allocator, bytecode->data);
  *bytecode = (loom_ireevm_function_bytecode_t){0};
}

iree_status_t loom_ireevm_emit_function_bytecode(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation, iree_allocator_t allocator,
    loom_ireevm_function_bytecode_t* out_bytecode) {
  if (!out_bytecode) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM bytecode output is required");
  }
  *out_bytecode = (loom_ireevm_function_bytecode_t){0};
  IREE_RETURN_IF_ERROR(loom_ireevm_validate_tables(schedule, allocation));

  loom_ireevm_emit_state_t state = {
      .schedule = schedule,
      .allocation = allocation,
  };
  loom_low_allocation_value_scratch_t scratch = {0};
  iree_status_t status =
      loom_low_allocation_acquire_value_scratch(allocation, &scratch);
  if (iree_status_is_ok(status)) {
    status = loom_ireevm_analyze_register_metadata(allocation,
                                                   &state.i32_register_count);
  }
  iree_host_size_t fixup_capacity = 0;
  if (iree_status_is_ok(status) &&
      !iree_host_size_checked_mul(schedule->scheduled_node_count, 2,
                                  &fixup_capacity)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "VM bytecode branch fixup capacity overflow");
  }
  if (iree_status_is_ok(status)) {
    status = loom_ireevm_bytecode_writer_initialize(
        schedule->block_count, fixup_capacity, allocator, &state.writer);
  }

  iree_host_size_t bytecode_length = 0;
  if (iree_status_is_ok(status)) {
    status = loom_ireevm_emit_function_body(&state);
  }
  if (iree_status_is_ok(status)) {
    status = loom_ireevm_bytecode_fixup_branch_targets(&state.writer);
  }
  if (iree_status_is_ok(status)) {
    bytecode_length = state.writer.length;
    status = loom_ireevm_bytecode_align(&state.writer, 8);
  }
  if (iree_status_is_ok(status)) {
    if (state.i32_register_count > UINT16_MAX) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "VM bytecode i32 register count exceeds u16");
    }
  }
  if (iree_status_is_ok(status)) {
    *out_bytecode = (loom_ireevm_function_bytecode_t){
        .data = state.writer.data,
        .data_length = state.writer.length,
        .bytecode_length = bytecode_length,
        .block_count = (uint16_t)schedule->block_count,
        .i32_register_count = (uint16_t)state.i32_register_count,
        .ref_register_count = 0,
    };
    state.writer.data = NULL;
  }

  loom_low_allocation_release_value_scratch(&scratch);
  loom_ireevm_bytecode_writer_deinitialize(&state.writer);
  return status;
}
