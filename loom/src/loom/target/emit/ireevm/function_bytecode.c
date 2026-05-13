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
#include "loom/target/arch/ireevm/descriptors.h"
#include "loom/target/emit/ireevm/module_plan.h"

enum {
  // VM bytecode opcodes from
  // runtime/src/iree/vm/bytecode/utils/generated/op_table.h.
  LOOM_IREEVM_OP_PREFIX_F32 = 0xE0,
  LOOM_IREEVM_OP_PREFIX_F64 = 0xE1,
  LOOM_IREEVM_OP_CONST_I32 = 0x0D,
  LOOM_IREEVM_OP_CONST_I64 = 0x0F,
  LOOM_IREEVM_OP_CONST_F32 = (LOOM_IREEVM_OP_PREFIX_F32 << 8) | 0x05,
  LOOM_IREEVM_OP_CONST_F64 = (LOOM_IREEVM_OP_PREFIX_F64 << 8) | 0x05,
  LOOM_IREEVM_OP_BRANCH = 0x56,
  LOOM_IREEVM_OP_COND_BRANCH = 0x57,
  LOOM_IREEVM_OP_CALL = 0x58,
  LOOM_IREEVM_OP_RETURN = 0x5A,
  LOOM_IREEVM_OP_BLOCK = 0x79,
};

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
  // Module-wide ordinal plan for local and imported call targets.
  const loom_ireevm_module_plan_t* module_plan;
  // Mutable bytecode writer.
  loom_ireevm_bytecode_writer_t writer;
  // Maximum i32 register ordinal plus one.
  uint32_t i32_register_count;
  // Maximum ref register ordinal plus one.
  uint32_t ref_register_count;
  // IREE VM FeatureBits required by emitted descriptor-backed opcodes.
  uint32_t feature_requirements;
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

static iree_status_t loom_ireevm_bytecode_write_u64(
    loom_ireevm_bytecode_writer_t* writer, uint64_t value) {
  IREE_RETURN_IF_ERROR(loom_ireevm_bytecode_writer_reserve(writer, 8));
  writer->data[writer->length++] = (uint8_t)(value & 0xFFu);
  writer->data[writer->length++] = (uint8_t)((value >> 8) & 0xFFu);
  writer->data[writer->length++] = (uint8_t)((value >> 16) & 0xFFu);
  writer->data[writer->length++] = (uint8_t)((value >> 24) & 0xFFu);
  writer->data[writer->length++] = (uint8_t)((value >> 32) & 0xFFu);
  writer->data[writer->length++] = (uint8_t)((value >> 40) & 0xFFu);
  writer->data[writer->length++] = (uint8_t)((value >> 48) & 0xFFu);
  writer->data[writer->length++] = (uint8_t)(value >> 56);
  return iree_ok_status();
}

static iree_status_t loom_ireevm_bytecode_write_opcode(
    loom_ireevm_bytecode_writer_t* writer, uint32_t encoding_id) {
  if (encoding_id <= UINT8_MAX) {
    return loom_ireevm_bytecode_write_u8(writer, (uint8_t)encoding_id);
  }
  if (encoding_id <= UINT16_MAX) {
    IREE_RETURN_IF_ERROR(
        loom_ireevm_bytecode_write_u8(writer, (uint8_t)(encoding_id >> 8)));
    return loom_ireevm_bytecode_write_u8(writer, (uint8_t)encoding_id);
  }
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "VM bytecode opcode exceeds u16");
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

typedef enum loom_ireevm_register_bank_e {
  LOOM_IREEVM_REGISTER_BANK_I32 = 0,
  LOOM_IREEVM_REGISTER_BANK_REF = 1,
} loom_ireevm_register_bank_t;

typedef struct loom_ireevm_register_ref_t {
  // VM register bank used by the bytecode operand.
  loom_ireevm_register_bank_t bank;
  // Base VM register ordinal.
  uint16_t base;
  // Number of VM register slots occupied by the value.
  uint16_t unit_count;
} loom_ireevm_register_ref_t;

static iree_status_t loom_ireevm_register_assignment_metadata(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_assignment_t* assignment,
    loom_ireevm_register_ref_t* out_register) {
  *out_register = (loom_ireevm_register_ref_t){0};
  if (!assignment) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "VM bytecode value has no active allocation");
  }
  if (assignment->location_kind != LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "VM bytecode value %u is not allocated to a VM target id",
        (unsigned)assignment->value_id);
  }
  if (assignment->value_class.type_kind != LOOM_TYPE_REGISTER) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "VM bytecode value %u is not allocated as a register value",
        (unsigned)assignment->value_id);
  }
  if (assignment->location_base > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM bytecode register ordinal exceeds u16");
  }
  if (assignment->location_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM bytecode register span exceeds u16");
  }

  loom_ireevm_register_ref_t register_ref = {
      .bank = LOOM_IREEVM_REGISTER_BANK_I32,
      .base = (uint16_t)assignment->location_base,
      .unit_count = (uint16_t)assignment->location_count,
  };
  uint32_t expected_unit_count = 0;
  switch (assignment->descriptor_reg_class_id) {
    case IREEVM_CORE_REG_CLASS_ID_I32:
    case IREEVM_CORE_REG_CLASS_ID_F32:
      expected_unit_count = 1;
      break;
    case IREEVM_CORE_REG_CLASS_ID_I64:
    case IREEVM_CORE_REG_CLASS_ID_F64:
      expected_unit_count = 2;
      if ((assignment->location_base & 1u) != 0) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "VM bytecode value %u uses an unaligned i64/f64 register ordinal",
            (unsigned)assignment->value_id);
      }
      break;
    case IREEVM_CORE_REG_CLASS_ID_REF:
      register_ref.bank = LOOM_IREEVM_REGISTER_BANK_REF;
      expected_unit_count = 1;
      break;
    default: {
      iree_string_view_t register_class = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_register_class_name(
          allocation, assignment, &register_class));
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "VM bytecode emission does not support register class '%.*s'",
          (int)register_class.size, register_class.data);
    }
  }
  if (assignment->unit_count != expected_unit_count ||
      assignment->location_count != expected_unit_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "VM bytecode value %u uses %" PRIu32
                            " units but its register class requires %" PRIu32,
                            (unsigned)assignment->value_id,
                            assignment->unit_count, expected_unit_count);
  }

  *out_register = register_ref;
  return iree_ok_status();
}

static iree_status_t loom_ireevm_analyze_register_metadata(
    const loom_low_allocation_table_t* allocation,
    uint32_t* out_i32_register_count, uint32_t* out_ref_register_count) {
  uint32_t i32_register_count = 0;
  uint32_t ref_register_count = 0;
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    loom_ireevm_register_ref_t register_ref = {0};
    IREE_RETURN_IF_ERROR(loom_ireevm_register_assignment_metadata(
        allocation, assignment, &register_ref));
    uint32_t register_limit =
        (uint32_t)register_ref.base + register_ref.unit_count;
    if (register_ref.bank == LOOM_IREEVM_REGISTER_BANK_REF) {
      if (register_limit > ref_register_count) {
        ref_register_count = register_limit;
      }
    } else if (register_limit > i32_register_count) {
      i32_register_count = register_limit;
    }
  }
  *out_i32_register_count = i32_register_count;
  *out_ref_register_count = ref_register_count;
  return iree_ok_status();
}

static iree_status_t loom_ireevm_lookup_register(
    loom_ireevm_emit_state_t* state, loom_value_id_t value_id,
    loom_ireevm_register_ref_t* out_register) {
  const loom_low_allocation_assignment_t* assignment =
      loom_low_allocation_map_active_value_assignment(state->allocation,
                                                      value_id, NULL);
  return loom_ireevm_register_assignment_metadata(state->allocation, assignment,
                                                  out_register);
}

static iree_status_t loom_ireevm_lookup_i32_bank_register(
    loom_ireevm_emit_state_t* state, loom_value_id_t value_id,
    loom_ireevm_register_ref_t* out_register) {
  IREE_RETURN_IF_ERROR(
      loom_ireevm_lookup_register(state, value_id, out_register));
  if (out_register->bank != LOOM_IREEVM_REGISTER_BANK_I32) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "VM bytecode scalar packet requires an i32-bank "
                            "register");
  }
  return iree_ok_status();
}

static iree_status_t loom_ireevm_lookup_condition_register(
    loom_ireevm_emit_state_t* state, loom_value_id_t value_id,
    uint16_t* out_register) {
  loom_ireevm_register_ref_t register_ref = {0};
  IREE_RETURN_IF_ERROR(
      loom_ireevm_lookup_i32_bank_register(state, value_id, &register_ref));
  if (register_ref.unit_count != 1) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "VM bytecode condition must use one i32 register");
  }
  *out_register = register_ref.base;
  return iree_ok_status();
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

static iree_status_t loom_ireevm_record_descriptor_requirements(
    loom_ireevm_emit_state_t* state, const loom_low_descriptor_t* descriptor) {
  if (descriptor->feature_mask_word_count == 0) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptor_set =
      state->schedule->target.descriptor_set;
  if (!descriptor_set->feature_mask_words) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "VM bytecode descriptor set has no feature-mask table");
  }
  for (uint16_t i = 0; i < descriptor->feature_mask_word_count; ++i) {
    const uint32_t word_index = descriptor->feature_mask_word_start + i;
    if (word_index >= descriptor_set->feature_mask_word_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "VM bytecode descriptor feature-mask row is out of range");
    }
    const uint64_t feature_bits =
        descriptor_set->feature_mask_words[word_index];
    if ((i != 0 && feature_bits != 0) || feature_bits > UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "VM bytecode feature requirements exceed FeatureBits width");
    }
    state->feature_requirements |= (uint32_t)feature_bits;
  }
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
    loom_ireevm_register_ref_t register_ref = {0};
    IREE_RETURN_IF_ERROR(
        loom_ireevm_lookup_register(state, values[i], &register_ref));
    IREE_RETURN_IF_ERROR(
        loom_ireevm_bytecode_write_u16(writer, register_ref.base));
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

  uint32_t remap_count = 0;
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    loom_ireevm_register_ref_t source_ref = {0};
    loom_ireevm_register_ref_t target_ref = {0};
    IREE_RETURN_IF_ERROR(
        loom_ireevm_lookup_register(state, values[i], &source_ref));
    IREE_RETURN_IF_ERROR(loom_ireevm_lookup_register(
        state, loom_block_arg_id(target, (uint16_t)i), &target_ref));
    if (source_ref.bank != target_ref.bank ||
        source_ref.unit_count != target_ref.unit_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "VM bytecode branch remap changes register bank or width");
    }
    for (uint16_t unit_index = 0; unit_index < source_ref.unit_count;
         ++unit_index) {
      if (source_ref.base + unit_index != target_ref.base + unit_index) {
        if (++remap_count > UINT16_MAX) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "VM bytecode branch remap count exceeds u16");
        }
      }
    }
  }

  IREE_RETURN_IF_ERROR(loom_ireevm_bytecode_align(writer, 2));
  IREE_RETURN_IF_ERROR(
      loom_ireevm_bytecode_write_u16(writer, (uint16_t)remap_count));
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    loom_ireevm_register_ref_t source_ref = {0};
    loom_ireevm_register_ref_t target_ref = {0};
    IREE_RETURN_IF_ERROR(
        loom_ireevm_lookup_register(state, values[i], &source_ref));
    IREE_RETURN_IF_ERROR(loom_ireevm_lookup_register(
        state, loom_block_arg_id(target, (uint16_t)i), &target_ref));
    for (uint16_t unit_index = 0; unit_index < source_ref.unit_count;
         ++unit_index) {
      const uint16_t source_reg = source_ref.base + unit_index;
      const uint16_t target_reg = target_ref.base + unit_index;
      if (source_reg != target_reg) {
        IREE_RETURN_IF_ERROR(
            loom_ireevm_bytecode_write_u16(writer, source_reg));
        IREE_RETURN_IF_ERROR(
            loom_ireevm_bytecode_write_u16(writer, target_reg));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_ireevm_emit_const(
    loom_ireevm_emit_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  if (!loom_low_const_isa(op) || op->result_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ireevm.const must be a unary low.const");
  }
  loom_ireevm_register_ref_t result_ref = {0};
  IREE_RETURN_IF_ERROR(loom_ireevm_lookup_i32_bank_register(
      state, loom_low_const_result(op), &result_ref));
  IREE_RETURN_IF_ERROR(loom_ireevm_bytecode_write_opcode(
      &state->writer, descriptor->encoding_id));
  switch (descriptor->encoding_id) {
    case LOOM_IREEVM_OP_CONST_I32: {
      int64_t value = 0;
      IREE_RETURN_IF_ERROR(loom_ireevm_read_i64_attr(
          state->schedule->module, loom_low_const_attrs(op),
          IREE_SV("i32_value"), &value));
      if (value < INT32_MIN || value > INT32_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "ireevm.const.i32 value is outside i32 range");
      }
      IREE_RETURN_IF_ERROR(
          loom_ireevm_bytecode_write_u32(&state->writer, (uint32_t)value));
      break;
    }
    case LOOM_IREEVM_OP_CONST_I64: {
      int64_t value = 0;
      IREE_RETURN_IF_ERROR(loom_ireevm_read_i64_attr(
          state->schedule->module, loom_low_const_attrs(op),
          IREE_SV("i64_value"), &value));
      IREE_RETURN_IF_ERROR(
          loom_ireevm_bytecode_write_u64(&state->writer, (uint64_t)value));
      break;
    }
    case LOOM_IREEVM_OP_CONST_F32: {
      int64_t value = 0;
      IREE_RETURN_IF_ERROR(loom_ireevm_read_i64_attr(
          state->schedule->module, loom_low_const_attrs(op),
          IREE_SV("f32_bits"), &value));
      if (value < 0 || value > UINT32_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "ireevm.const.f32 bits are outside u32 range");
      }
      IREE_RETURN_IF_ERROR(
          loom_ireevm_bytecode_write_u32(&state->writer, (uint32_t)value));
      break;
    }
    case LOOM_IREEVM_OP_CONST_F64: {
      int64_t value = 0;
      IREE_RETURN_IF_ERROR(loom_ireevm_read_i64_attr(
          state->schedule->module, loom_low_const_attrs(op),
          IREE_SV("f64_bits"), &value));
      IREE_RETURN_IF_ERROR(
          loom_ireevm_bytecode_write_u64(&state->writer, (uint64_t)value));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported VM constant descriptor");
  }
  return loom_ireevm_bytecode_write_u16(&state->writer, result_ref.base);
}

static iree_status_t loom_ireevm_emit_generic_low_op(
    loom_ireevm_emit_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  if (!loom_low_op_isa(op) || op->result_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM descriptor packet shape is invalid");
  }
  loom_value_slice_t operands = loom_low_op_operands(op);
  loom_value_slice_t results = loom_low_op_results(op);
  IREE_RETURN_IF_ERROR(loom_ireevm_bytecode_write_opcode(
      &state->writer, descriptor->encoding_id));
  for (iree_host_size_t i = 0; i < operands.count; ++i) {
    loom_ireevm_register_ref_t operand_ref = {0};
    IREE_RETURN_IF_ERROR(loom_ireevm_lookup_i32_bank_register(
        state, operands.values[i], &operand_ref));
    IREE_RETURN_IF_ERROR(
        loom_ireevm_bytecode_write_u16(&state->writer, operand_ref.base));
  }
  loom_ireevm_register_ref_t result_ref = {0};
  IREE_RETURN_IF_ERROR(loom_ireevm_lookup_i32_bank_register(
      state, results.values[0], &result_ref));
  return loom_ireevm_bytecode_write_u16(&state->writer, result_ref.base);
}

static iree_status_t loom_ireevm_emit_descriptor_packet(
    loom_ireevm_emit_state_t* state, const loom_low_packet_view_t* packet) {
  IREE_RETURN_IF_ERROR(
      loom_ireevm_record_descriptor_requirements(state, packet->descriptor));
  switch (packet->descriptor->encoding_id) {
    case LOOM_IREEVM_OP_CONST_I32:
    case LOOM_IREEVM_OP_CONST_I64:
    case LOOM_IREEVM_OP_CONST_F32:
    case LOOM_IREEVM_OP_CONST_F64:
      return loom_ireevm_emit_const(state, packet->node->op,
                                    packet->descriptor);
    default: {
      return loom_ireevm_emit_generic_low_op(state, packet->node->op,
                                             packet->descriptor);
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
  IREE_RETURN_IF_ERROR(loom_ireevm_lookup_condition_register(
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

static iree_status_t loom_ireevm_emit_low_func_call(
    loom_ireevm_emit_state_t* state, const loom_op_t* op) {
  uint32_t encoded_ordinal = 0;
  IREE_RETURN_IF_ERROR(loom_ireevm_module_plan_resolve_callee(
      state->module_plan, loom_low_func_call_callee(op), &encoded_ordinal));

  IREE_RETURN_IF_ERROR(
      loom_ireevm_bytecode_write_u8(&state->writer, LOOM_IREEVM_OP_CALL));
  IREE_RETURN_IF_ERROR(
      loom_ireevm_bytecode_write_u32(&state->writer, encoded_ordinal));
  loom_value_slice_t operands = loom_low_func_call_operands(op);
  loom_value_slice_t results = loom_low_func_call_results(op);
  IREE_RETURN_IF_ERROR(
      loom_ireevm_write_register_list(state, operands.values, operands.count));
  return loom_ireevm_write_register_list(state, results.values, results.count);
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
  if (loom_low_func_call_isa(op)) {
    return loom_ireevm_emit_low_func_call(state, op);
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
  IREE_RETURN_IF_ERROR(loom_low_packet_validate_tables(schedule, allocation));
  if (schedule->target.descriptor_set != loom_ireevm_core_descriptor_set()) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM bytecode emission requires ireevm.core");
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
    const loom_low_allocation_table_t* allocation,
    const loom_ireevm_module_plan_t* module_plan, iree_allocator_t allocator,
    loom_ireevm_function_bytecode_t* out_bytecode) {
  *out_bytecode = (loom_ireevm_function_bytecode_t){0};
  if (!module_plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM bytecode emission requires a module plan");
  }
  IREE_RETURN_IF_ERROR(loom_ireevm_validate_tables(schedule, allocation));

  loom_ireevm_emit_state_t state = {
      .schedule = schedule,
      .allocation = allocation,
      .module_plan = module_plan,
  };
  loom_low_allocation_value_scratch_t scratch = {0};
  iree_status_t status =
      loom_low_allocation_acquire_value_scratch(allocation, &scratch);
  if (iree_status_is_ok(status)) {
    status = loom_ireevm_analyze_register_metadata(
        allocation, &state.i32_register_count, &state.ref_register_count);
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
    if (state.ref_register_count > UINT16_MAX) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "VM bytecode ref register count exceeds u16");
    }
  }
  if (iree_status_is_ok(status)) {
    *out_bytecode = (loom_ireevm_function_bytecode_t){
        .data = state.writer.data,
        .data_length = state.writer.length,
        .bytecode_length = bytecode_length,
        .block_count = (uint16_t)schedule->block_count,
        .i32_register_count = (uint16_t)state.i32_register_count,
        .ref_register_count = (uint16_t)state.ref_register_count,
        .feature_requirements = state.feature_requirements,
    };
    state.writer.data = NULL;
  }

  loom_low_allocation_release_value_scratch(&scratch);
  loom_ireevm_bytecode_writer_deinitialize(&state.writer);
  return status;
}
