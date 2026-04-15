// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/llvmir/bitcode_writer.h"

#include <string.h>

#include "loom/target/llvmir/bitcode_format.h"
#include "loom/target/llvmir/bitcode_record_writer.h"
#include "loom/target/llvmir/types.h"

#define LOOM_LLVMIR_BITCODE_VALUE_ID_INVALID UINT64_MAX

static uint64_t loom_llvmir_bitcode_function_type_id(
    const loom_llvmir_module_t* module, iree_host_size_t function_ordinal) {
  return module->type_count + function_ordinal;
}

static iree_status_t loom_llvmir_bitcode_write_record_u64(
    loom_llvmir_bitcode_record_writer_t* writer, uint64_t code,
    uint64_t operand) {
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(writer, code,
                                                                 &operand, 1);
}

static iree_status_t loom_llvmir_bitcode_write_empty_record(
    loom_llvmir_bitcode_record_writer_t* writer, uint64_t code) {
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(writer, code,
                                                                 NULL, 0);
}

static uint64_t loom_llvmir_bitcode_encode_signed(int64_t value) {
  if (value >= 0) return ((uint64_t)value) << 1;
  return (((uint64_t)(-value)) << 1) | 1;
}

static iree_status_t loom_llvmir_bitcode_write_function_type(
    const loom_llvmir_function_t* function,
    loom_llvmir_bitcode_record_writer_t* writer) {
  iree_allocator_t allocator = iree_allocator_system();
  if (function->parameter_count > IREE_HOST_SIZE_MAX - 2) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM function type operand count overflows");
  }
  iree_host_size_t operand_count = function->parameter_count + 2;
  uint64_t stack_operands[16];
  uint64_t* operands = stack_operands;
  if (operand_count > IREE_ARRAYSIZE(stack_operands)) {
    if (operand_count > IREE_HOST_SIZE_MAX / sizeof(*operands)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "LLVM function type operand storage overflows");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        allocator, operand_count * sizeof(*operands), (void**)&operands));
  }

  operands[0] = 0;
  operands[1] = function->return_type;
  for (iree_host_size_t i = 0; i < function->parameter_count; ++i) {
    operands[i + 2] = function->parameters[i].type_id;
  }
  iree_status_t status =
      loom_llvmir_bitcode_record_writer_write_unabbrev_record(
          writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_FUNCTION, operands,
          operand_count);

  if (operands != stack_operands) {
    iree_allocator_free(allocator, operands);
  }
  return status;
}

static iree_status_t loom_llvmir_bitcode_write_type_block(
    const loom_llvmir_module_t* module,
    loom_llvmir_bitcode_record_writer_t* writer) {
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_record_writer_enter_subblock(
      writer, LOOM_LLVMIR_BITCODE_TYPE_BLOCK,
      LOOM_LLVMIR_BITCODE_TYPE_ABBREV_WIDTH));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_write_record_u64(
      writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_NUMENTRY,
      module->type_count + module->function_count));

  for (iree_host_size_t i = 0; i < module->type_count; ++i) {
    const loom_llvmir_type_t* type = &module->types[i];
    switch (type->kind) {
      case LOOM_LLVMIR_TYPE_VOID: {
        IREE_RETURN_IF_ERROR(
            loom_llvmir_bitcode_record_writer_write_unabbrev_record(
                writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_VOID, NULL, 0));
        break;
      }
      case LOOM_LLVMIR_TYPE_INTEGER: {
        IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_write_record_u64(
            writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_INTEGER, type->bit_width));
        break;
      }
      case LOOM_LLVMIR_TYPE_FLOAT: {
        switch (type->float_kind) {
          case LOOM_LLVMIR_FLOAT_F16: {
            IREE_RETURN_IF_ERROR(
                loom_llvmir_bitcode_record_writer_write_unabbrev_record(
                    writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_HALF, NULL, 0));
            break;
          }
          case LOOM_LLVMIR_FLOAT_F32: {
            IREE_RETURN_IF_ERROR(
                loom_llvmir_bitcode_record_writer_write_unabbrev_record(
                    writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_FLOAT, NULL, 0));
            break;
          }
          case LOOM_LLVMIR_FLOAT_F64: {
            IREE_RETURN_IF_ERROR(
                loom_llvmir_bitcode_record_writer_write_unabbrev_record(
                    writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_DOUBLE, NULL, 0));
            break;
          }
        }
        break;
      }
      case LOOM_LLVMIR_TYPE_POINTER: {
        IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_write_record_u64(
            writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_OPAQUE_POINTER,
            type->address_space));
        break;
      }
      case LOOM_LLVMIR_TYPE_VECTOR: {
        uint64_t operands[] = {type->element_count, type->element_type};
        IREE_RETURN_IF_ERROR(
            loom_llvmir_bitcode_record_writer_write_unabbrev_record(
                writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_VECTOR, operands,
                IREE_ARRAYSIZE(operands)));
        break;
      }
    }
  }
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_llvmir_bitcode_write_function_type(module->functions[i], writer));
  }

  return loom_llvmir_bitcode_record_writer_exit_block(writer);
}

typedef struct loom_llvmir_bitcode_function_value_map_t {
  // Module-global value ID to LLVM bitcode value ID map.
  uint64_t* value_ids;
  // Number of entries in |value_ids|.
  iree_host_size_t value_count;
  // First instruction value ID after this function's parameters.
  uint64_t first_instruction_value_id;
} loom_llvmir_bitcode_function_value_map_t;

static iree_status_t loom_llvmir_bitcode_check_value_id_range(
    uint64_t value_id) {
  if (value_id > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM bitcode value id exceeds 32-bit range");
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_map_set_value(
    loom_llvmir_bitcode_function_value_map_t* map,
    loom_llvmir_value_id_t value_id, uint64_t bitcode_value_id) {
  if (value_id >= map->value_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode value map saw unknown value");
  }
  if (map->value_ids[value_id] != LOOM_LLVMIR_BITCODE_VALUE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode value map saw duplicate value");
  }
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_check_value_id_range(bitcode_value_id));
  map->value_ids[value_id] = bitcode_value_id;
  return iree_ok_status();
}

static void loom_llvmir_bitcode_function_value_map_deinitialize(
    loom_llvmir_bitcode_function_value_map_t* map);

static iree_status_t loom_llvmir_bitcode_function_value_map_initialize(
    const loom_llvmir_function_t* function,
    loom_llvmir_bitcode_function_value_map_t* out_map) {
  memset(out_map, 0, sizeof(*out_map));
  const loom_llvmir_module_t* module = function->module;
  if (module->value_count == 0) {
    out_map->first_instruction_value_id = module->function_count;
    return iree_ok_status();
  }
  if (module->value_count > IREE_HOST_SIZE_MAX / sizeof(*out_map->value_ids)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM bitcode value map storage overflows");
  }
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(iree_allocator_system(),
                            module->value_count * sizeof(*out_map->value_ids),
                            (void**)&out_map->value_ids));
  out_map->value_count = module->value_count;
  for (iree_host_size_t i = 0; i < module->value_count; ++i) {
    out_map->value_ids[i] = LOOM_LLVMIR_BITCODE_VALUE_ID_INVALID;
  }

  iree_status_t status = iree_ok_status();
  uint64_t next_value_id = module->function_count;
  for (iree_host_size_t i = 0;
       i < function->parameter_count && iree_status_is_ok(status); ++i) {
    status = loom_llvmir_bitcode_map_set_value(
        out_map, function->parameters[i].value_id, next_value_id++);
  }
  out_map->first_instruction_value_id = next_value_id;
  for (iree_host_size_t i = 0;
       i < function->block_count && iree_status_is_ok(status); ++i) {
    const loom_llvmir_block_t* block = function->blocks[i];
    for (iree_host_size_t j = 0;
         j < block->instruction_count && iree_status_is_ok(status); ++j) {
      const loom_llvmir_instruction_t* instruction = &block->instructions[j];
      if (instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
        continue;
      }
      status = loom_llvmir_bitcode_map_set_value(
          out_map, instruction->result_value_id, next_value_id++);
    }
  }
  if (!iree_status_is_ok(status)) {
    loom_llvmir_bitcode_function_value_map_deinitialize(out_map);
  }
  return status;
}

static void loom_llvmir_bitcode_function_value_map_deinitialize(
    loom_llvmir_bitcode_function_value_map_t* map) {
  if (map->value_ids != NULL) {
    iree_allocator_free(iree_allocator_system(), map->value_ids);
  }
}

static iree_status_t loom_llvmir_bitcode_map_value(
    const loom_llvmir_bitcode_function_value_map_t* map,
    loom_llvmir_value_id_t value_id, uint64_t* out_bitcode_value_id) {
  if (value_id >= map->value_count ||
      map->value_ids[value_id] == LOOM_LLVMIR_BITCODE_VALUE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode instruction references unmapped "
                            "function value");
  }
  *out_bitcode_value_id = map->value_ids[value_id];
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_push_value(
    const loom_llvmir_bitcode_function_value_map_t* map,
    loom_llvmir_value_id_t value_id, uint64_t instruction_value_id,
    uint64_t* operands, iree_host_size_t* inout_operand_count) {
  uint64_t bitcode_value_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_map_value(map, value_id, &bitcode_value_id));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_check_value_id_range(instruction_value_id));
  if (bitcode_value_id >= instruction_value_id) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LLVM bitcode instruction operand references a forward value");
  }
  uint32_t relative_value_id =
      (uint32_t)instruction_value_id - (uint32_t)bitcode_value_id;
  operands[(*inout_operand_count)++] = relative_value_id;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_push_value_type_pair(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* map,
    loom_llvmir_value_id_t value_id, uint64_t instruction_value_id,
    uint64_t* operands, iree_host_size_t* inout_operand_count) {
  uint64_t bitcode_value_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_map_value(map, value_id, &bitcode_value_id));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_check_value_id_range(instruction_value_id));
  uint32_t relative_value_id =
      (uint32_t)instruction_value_id - (uint32_t)bitcode_value_id;
  operands[(*inout_operand_count)++] = relative_value_id;
  if (bitcode_value_id >= instruction_value_id) {
    if (value_id >= module->value_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVM bitcode instruction references unknown "
                              "typed value");
    }
    operands[(*inout_operand_count)++] = module->values[value_id].type_id;
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_push_signed_value(
    const loom_llvmir_bitcode_function_value_map_t* map,
    loom_llvmir_value_id_t value_id, uint64_t instruction_value_id,
    uint64_t* operands, iree_host_size_t* inout_operand_count) {
  uint64_t bitcode_value_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_map_value(map, value_id, &bitcode_value_id));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_check_value_id_range(instruction_value_id));
  int64_t relative_value_id =
      (int64_t)instruction_value_id - (int64_t)bitcode_value_id;
  operands[(*inout_operand_count)++] =
      loom_llvmir_bitcode_encode_signed(relative_value_id);
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_check_instruction(
    const loom_llvmir_instruction_t* instruction) {
  switch (instruction->kind) {
    case LOOM_LLVMIR_INST_RET:
      return iree_ok_status();
    case LOOM_LLVMIR_INST_BR:
    case LOOM_LLVMIR_INST_UNREACHABLE:
      return iree_ok_status();
    case LOOM_LLVMIR_INST_PHI:
      return iree_ok_status();
    case LOOM_LLVMIR_INST_BINOP:
      return iree_ok_status();
    case LOOM_LLVMIR_INST_GEP:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "LLVM bitcode gep emission is not implemented");
    case LOOM_LLVMIR_INST_LOAD:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "LLVM bitcode load emission is not implemented");
    case LOOM_LLVMIR_INST_STORE:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "LLVM bitcode store emission is not implemented");
    case LOOM_LLVMIR_INST_CALL:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "LLVM bitcode call emission is not implemented");
    case LOOM_LLVMIR_INST_INLINE_ASM:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "LLVM bitcode inline asm emission is not implemented");
    case LOOM_LLVMIR_INST_COND_BR:
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM instruction kind");
  }
}

static iree_status_t loom_llvmir_bitcode_check_function(
    const loom_llvmir_function_t* function) {
  if (function->kind != LOOM_LLVMIR_FUNCTION_DECLARATION &&
      function->kind != LOOM_LLVMIR_FUNCTION_DEFINITION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown LLVM function kind");
  }
  if (function->attr_group_id != LOOM_LLVMIR_ATTR_GROUP_ID_INVALID ||
      function->metadata_attachment_count != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "LLVM bitcode function attribute/metadata emission is not implemented");
  }
  for (iree_host_size_t i = 0; i < function->parameter_count; ++i) {
    if (function->parameters[i].attrs.attr_count != 0) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "LLVM bitcode parameter attribute emission is not implemented");
    }
  }
  if (function->kind == LOOM_LLVMIR_FUNCTION_DECLARATION) {
    if (function->block_count != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVM declaration contains blocks");
    }
    return iree_ok_status();
  }
  if (function->block_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM function definition has no blocks");
  }
  for (iree_host_size_t i = 0; i < function->block_count; ++i) {
    const loom_llvmir_block_t* block = function->blocks[i];
    for (iree_host_size_t j = 0; j < block->instruction_count; ++j) {
      IREE_RETURN_IF_ERROR(
          loom_llvmir_bitcode_check_instruction(&block->instructions[j]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_check_module(
    const loom_llvmir_module_t* module) {
  if (module->function_count > IREE_HOST_SIZE_MAX - module->type_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM bitcode type table is too large");
  }
  if (module->attr_group_count != 0 || module->metadata_node_count != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "LLVM bitcode attribute/metadata emission is not implemented");
  }
  for (iree_host_size_t i = 0; i < module->value_count; ++i) {
    switch (module->values[i].kind) {
      case LOOM_LLVMIR_VALUE_PARAMETER:
        break;
      case LOOM_LLVMIR_VALUE_CONSTANT_INTEGER:
      case LOOM_LLVMIR_VALUE_CONSTANT_FLOAT_BITS:
      case LOOM_LLVMIR_VALUE_CONSTANT_NULL:
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "LLVM bitcode constant emission is not implemented");
      case LOOM_LLVMIR_VALUE_INSTRUCTION:
        break;
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown LLVM value kind");
    }
  }
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_llvmir_bitcode_check_function(module->functions[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_function_linkage(
    const loom_llvmir_function_t* function, uint64_t* out_linkage,
    uint64_t* out_dso_local) {
  switch (function->linkage) {
    case LOOM_LLVMIR_LINKAGE_DEFAULT:
      *out_linkage = 0;
      *out_dso_local = 0;
      return iree_ok_status();
    case LOOM_LLVMIR_LINKAGE_DSO_LOCAL:
      *out_linkage = 0;
      *out_dso_local = 1;
      return iree_ok_status();
    case LOOM_LLVMIR_LINKAGE_INTERNAL:
      *out_linkage = 3;
      *out_dso_local = 0;
      return iree_ok_status();
    case LOOM_LLVMIR_LINKAGE_PRIVATE:
      *out_linkage = 9;
      *out_dso_local = 0;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM function linkage");
  }
}

static iree_status_t loom_llvmir_bitcode_function_calling_convention(
    loom_llvmir_calling_convention_t calling_convention,
    uint64_t* out_calling_convention) {
  switch (calling_convention) {
    case LOOM_LLVMIR_CALLING_CONVENTION_DEFAULT:
      *out_calling_convention = 0;
      return iree_ok_status();
    case LOOM_LLVMIR_CALLING_CONVENTION_AMDGPU_KERNEL:
      *out_calling_convention = 91;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM calling convention");
  }
}

static iree_status_t loom_llvmir_bitcode_binary_opcode(loom_llvmir_binop_t op,
                                                       uint64_t* out_opcode) {
  switch (op) {
    case LOOM_LLVMIR_BINOP_ADD:
    case LOOM_LLVMIR_BINOP_FADD:
      *out_opcode = LOOM_LLVMIR_BITCODE_BINOP_ADD;
      return iree_ok_status();
    case LOOM_LLVMIR_BINOP_SUB:
    case LOOM_LLVMIR_BINOP_FSUB:
      *out_opcode = LOOM_LLVMIR_BITCODE_BINOP_SUB;
      return iree_ok_status();
    case LOOM_LLVMIR_BINOP_MUL:
    case LOOM_LLVMIR_BINOP_FMUL:
      *out_opcode = LOOM_LLVMIR_BITCODE_BINOP_MUL;
      return iree_ok_status();
    case LOOM_LLVMIR_BINOP_AND:
      *out_opcode = LOOM_LLVMIR_BITCODE_BINOP_AND;
      return iree_ok_status();
    case LOOM_LLVMIR_BINOP_OR:
      *out_opcode = LOOM_LLVMIR_BITCODE_BINOP_OR;
      return iree_ok_status();
    case LOOM_LLVMIR_BINOP_XOR:
      *out_opcode = LOOM_LLVMIR_BITCODE_BINOP_XOR;
      return iree_ok_status();
    case LOOM_LLVMIR_BINOP_SHL:
      *out_opcode = LOOM_LLVMIR_BITCODE_BINOP_SHL;
      return iree_ok_status();
    case LOOM_LLVMIR_BINOP_LSHR:
      *out_opcode = LOOM_LLVMIR_BITCODE_BINOP_LSHR;
      return iree_ok_status();
    case LOOM_LLVMIR_BINOP_ASHR:
      *out_opcode = LOOM_LLVMIR_BITCODE_BINOP_ASHR;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM binary op");
  }
}

static iree_status_t loom_llvmir_bitcode_write_function_record(
    const loom_llvmir_module_t* module, const loom_llvmir_function_t* function,
    uint64_t name_offset, loom_llvmir_bitcode_record_writer_t* writer) {
  uint64_t linkage = 0;
  uint64_t dso_local = 0;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_function_linkage(function, &linkage, &dso_local));
  uint64_t calling_convention = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_function_calling_convention(
      function->calling_convention, &calling_convention));

  uint64_t operands[] = {
      name_offset,
      function->name.size,
      loom_llvmir_bitcode_function_type_id(module, function->id),
      calling_convention,
      function->kind == LOOM_LLVMIR_FUNCTION_DECLARATION,
      linkage,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      dso_local,
      0,
  };
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
      writer, LOOM_LLVMIR_BITCODE_MODULE_CODE_FUNCTION, operands,
      IREE_ARRAYSIZE(operands));
}

static iree_status_t loom_llvmir_bitcode_write_return(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (!instruction->ret.has_value) {
    return loom_llvmir_bitcode_write_empty_record(
        writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_RET);
  }
  uint64_t operands[2];
  iree_host_size_t operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value_type_pair(
      module, value_map, instruction->ret.value, instruction_value_id, operands,
      &operand_count));
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
      writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_RET, operands,
      operand_count);
}

static iree_status_t loom_llvmir_bitcode_write_binop(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  uint64_t opcode = 0;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_binary_opcode(instruction->binop.op, &opcode));
  uint64_t operands[4];
  iree_host_size_t operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value_type_pair(
      module, value_map, instruction->binop.lhs, instruction_value_id, operands,
      &operand_count));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value(
      value_map, instruction->binop.rhs, instruction_value_id, operands,
      &operand_count));
  operands[operand_count++] = opcode;
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
      writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_BINOP, operands,
      operand_count);
}

static iree_status_t loom_llvmir_bitcode_write_phi(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (instruction->result_value_id >= module->value_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode phi references unknown result");
  }
  if (instruction->phi.incoming_count > (IREE_HOST_SIZE_MAX - 1) / 2) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM bitcode phi operand count overflows");
  }
  iree_host_size_t operand_count = 1 + instruction->phi.incoming_count * 2;
  uint64_t stack_operands[17];
  uint64_t* operands = stack_operands;
  if (operand_count > IREE_ARRAYSIZE(stack_operands)) {
    if (operand_count > IREE_HOST_SIZE_MAX / sizeof(*operands)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "LLVM bitcode phi operand storage overflows");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        iree_allocator_system(), operand_count * sizeof(*operands),
        (void**)&operands));
  }

  iree_host_size_t written_operand_count = 0;
  operands[written_operand_count++] =
      module->values[instruction->result_value_id].type_id;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < instruction->phi.incoming_count && iree_status_is_ok(status); ++i) {
    status = loom_llvmir_bitcode_push_signed_value(
        value_map, instruction->phi.incoming[i].value, instruction_value_id,
        operands, &written_operand_count);
    if (iree_status_is_ok(status)) {
      operands[written_operand_count++] =
          instruction->phi.incoming[i].predecessor;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_write_unabbrev_record(
        writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_PHI, operands,
        written_operand_count);
  }
  if (operands != stack_operands) {
    iree_allocator_free(iree_allocator_system(), operands);
  }
  return status;
}

static iree_status_t loom_llvmir_bitcode_write_instruction(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  switch (instruction->kind) {
    case LOOM_LLVMIR_INST_PHI:
      return loom_llvmir_bitcode_write_phi(module, value_map, instruction,
                                           instruction_value_id, writer);
    case LOOM_LLVMIR_INST_BINOP:
      return loom_llvmir_bitcode_write_binop(module, value_map, instruction,
                                             instruction_value_id, writer);
    case LOOM_LLVMIR_INST_RET:
      return loom_llvmir_bitcode_write_return(module, value_map, instruction,
                                              instruction_value_id, writer);
    case LOOM_LLVMIR_INST_BR:
      return loom_llvmir_bitcode_write_record_u64(
          writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_BR,
          instruction->br.target);
    case LOOM_LLVMIR_INST_COND_BR: {
      uint64_t operands[3];
      iree_host_size_t operand_count = 0;
      operands[operand_count++] = instruction->cond_br.true_block;
      operands[operand_count++] = instruction->cond_br.false_block;
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value(
          value_map, instruction->cond_br.condition, instruction_value_id,
          operands, &operand_count));
      return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
          writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_BR, operands,
          operand_count);
    }
    case LOOM_LLVMIR_INST_UNREACHABLE:
      return loom_llvmir_bitcode_write_empty_record(
          writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_UNREACHABLE);
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "unsupported LLVM instruction passed bitcode "
                              "preflight");
  }
}

static iree_status_t loom_llvmir_bitcode_write_function_body(
    const loom_llvmir_function_t* function,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (function->kind == LOOM_LLVMIR_FUNCTION_DECLARATION) {
    return iree_ok_status();
  }
  loom_llvmir_bitcode_function_value_map_t value_map;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_function_value_map_initialize(function, &value_map));

  iree_status_t status = loom_llvmir_bitcode_record_writer_enter_subblock(
      writer, LOOM_LLVMIR_BITCODE_FUNCTION_BLOCK,
      LOOM_LLVMIR_BITCODE_FUNCTION_ABBREV_WIDTH);
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_write_record_u64(
        writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_DECLAREBLOCKS,
        function->block_count);
  }
  uint64_t instruction_value_id = value_map.first_instruction_value_id;
  for (iree_host_size_t i = 0;
       i < function->block_count && iree_status_is_ok(status); ++i) {
    const loom_llvmir_block_t* block = function->blocks[i];
    for (iree_host_size_t j = 0;
         j < block->instruction_count && iree_status_is_ok(status); ++j) {
      const loom_llvmir_instruction_t* instruction = &block->instructions[j];
      status = loom_llvmir_bitcode_write_instruction(
          function->module, &value_map, instruction, instruction_value_id,
          writer);
      if (iree_status_is_ok(status) &&
          instruction->result_value_id != LOOM_LLVMIR_VALUE_ID_INVALID) {
        uint64_t result_value_id = 0;
        status = loom_llvmir_bitcode_map_value(
            &value_map, instruction->result_value_id, &result_value_id);
        if (iree_status_is_ok(status) &&
            result_value_id != instruction_value_id) {
          status = iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "LLVM bitcode instruction value numbering is inconsistent");
        }
        instruction_value_id += 1;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_exit_block(writer);
  }
  loom_llvmir_bitcode_function_value_map_deinitialize(&value_map);
  return status;
}

static iree_status_t loom_llvmir_bitcode_write_function_records_and_bodies(
    const loom_llvmir_module_t* module,
    loom_llvmir_bitcode_record_writer_t* writer) {
  uint64_t name_offset = 0;
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    const loom_llvmir_function_t* function = module->functions[i];
    IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_write_function_record(
        module, function, name_offset, writer));
    name_offset += function->name.size;
  }
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_llvmir_bitcode_write_function_body(module->functions[i], writer));
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_build_function_strtab(
    const loom_llvmir_module_t* module, char** out_storage,
    iree_host_size_t* out_storage_size) {
  *out_storage = NULL;
  *out_storage_size = 0;
  iree_host_size_t storage_size = 0;
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    if (module->functions[i]->name.size > IREE_HOST_SIZE_MAX - storage_size) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "LLVM bitcode string table is too large");
    }
    storage_size += module->functions[i]->name.size;
  }
  if (storage_size == 0) return iree_ok_status();

  iree_allocator_t allocator = iree_allocator_system();
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, storage_size, (void**)&storage));
  iree_host_size_t offset = 0;
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    iree_string_view_t name = module->functions[i]->name;
    memcpy(storage + offset, name.data, name.size);
    offset += name.size;
  }
  *out_storage = storage;
  *out_storage_size = storage_size;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_write_strtab_block(
    iree_string_view_t string_table,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (iree_string_view_is_empty(string_table)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_record_writer_enter_subblock(
      writer, LOOM_LLVMIR_BITCODE_STRTAB_BLOCK,
      LOOM_LLVMIR_BITCODE_STRTAB_ABBREV_WIDTH));
  uint32_t blob_abbrev = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_record_writer_define_blob_abbrev(
      writer, LOOM_LLVMIR_BITCODE_STRTAB_CODE_BLOB, &blob_abbrev));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_record_writer_write_blob_record(
      writer, blob_abbrev, string_table));
  return loom_llvmir_bitcode_record_writer_exit_block(writer);
}

iree_status_t loom_llvmir_bitcode_write_module(
    const loom_llvmir_module_t* module, iree_io_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_check_module(module));

  char* string_table_storage = NULL;
  iree_host_size_t string_table_size = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_build_function_strtab(
      module, &string_table_storage, &string_table_size));

  static const uint8_t magic[] = {
      LOOM_LLVMIR_BITCODE_MAGIC_0,
      LOOM_LLVMIR_BITCODE_MAGIC_1,
      LOOM_LLVMIR_BITCODE_MAGIC_2,
      LOOM_LLVMIR_BITCODE_MAGIC_3,
  };

  iree_status_t status = iree_io_stream_write(stream, sizeof(magic), magic);
  loom_llvmir_bitstream_writer_t bitstream;
  loom_llvmir_bitcode_record_writer_t record_writer;
  if (iree_status_is_ok(status)) {
    loom_llvmir_bitstream_writer_initialize(stream, &bitstream);
    loom_llvmir_bitcode_record_writer_initialize(&bitstream, &record_writer);

    status = loom_llvmir_bitcode_record_writer_enter_subblock(
        &record_writer, LOOM_LLVMIR_BITCODE_MODULE_BLOCK,
        LOOM_LLVMIR_BITCODE_MODULE_ABBREV_WIDTH);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_write_record_u64(
        &record_writer, LOOM_LLVMIR_BITCODE_MODULE_CODE_VERSION,
        LOOM_LLVMIR_BITCODE_MODULE_VERSION);
  }
  if (iree_status_is_ok(status) &&
      !iree_string_view_is_empty(module->target_config.source_name)) {
    status = loom_llvmir_bitcode_record_writer_write_string_record(
        &record_writer, LOOM_LLVMIR_BITCODE_MODULE_CODE_SOURCE_FILENAME,
        module->target_config.source_name);
  }
  if (iree_status_is_ok(status) &&
      !iree_string_view_is_empty(module->target_config.target_triple)) {
    status = loom_llvmir_bitcode_record_writer_write_string_record(
        &record_writer, LOOM_LLVMIR_BITCODE_MODULE_CODE_TRIPLE,
        module->target_config.target_triple);
  }
  if (iree_status_is_ok(status) &&
      !iree_string_view_is_empty(module->target_config.data_layout)) {
    status = loom_llvmir_bitcode_record_writer_write_string_record(
        &record_writer, LOOM_LLVMIR_BITCODE_MODULE_CODE_DATALAYOUT,
        module->target_config.data_layout);
  }
  if (iree_status_is_ok(status) &&
      (module->type_count != 0 || module->function_count != 0)) {
    status = loom_llvmir_bitcode_write_type_block(module, &record_writer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_write_function_records_and_bodies(
        module, &record_writer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_exit_block(&record_writer);
  }
  if (iree_status_is_ok(status) && string_table_size != 0) {
    status = loom_llvmir_bitcode_write_strtab_block(
        iree_make_string_view(string_table_storage, string_table_size),
        &record_writer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitstream_writer_finish(&bitstream);
  }
  if (string_table_storage != NULL) {
    iree_allocator_free(iree_allocator_system(), string_table_storage);
  }
  return status;
}
