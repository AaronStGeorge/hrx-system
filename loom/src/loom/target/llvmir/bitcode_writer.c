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
#define LOOM_LLVMIR_BITCODE_VALUE_ID_PENDING_CONSTANT (UINT64_MAX - 1)

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

static bool loom_llvmir_bitcode_is_constant_value_kind(
    loom_llvmir_value_kind_t kind) {
  return kind == LOOM_LLVMIR_VALUE_CONSTANT_INTEGER ||
         kind == LOOM_LLVMIR_VALUE_CONSTANT_FLOAT_BITS ||
         kind == LOOM_LLVMIR_VALUE_CONSTANT_NULL;
}

static uint64_t loom_llvmir_bitcode_sign_extend_integer_constant(
    uint64_t value, uint32_t bit_width) {
  if (bit_width == 0 || bit_width >= 64) return value;
  uint64_t sign_bit = UINT64_C(1) << (bit_width - 1);
  uint64_t value_mask = (UINT64_C(1) << bit_width) - 1;
  uint64_t masked_value = value & value_mask;
  if ((masked_value & sign_bit) == 0) return masked_value;
  return masked_value | ~value_mask;
}

static bool loom_llvmir_bitcode_is_power_of_two_u64(uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

static iree_status_t loom_llvmir_bitcode_check_attr_alignment(
    uint64_t alignment) {
  static const uint64_t kMaxAlignment = UINT64_C(1) << 32;
  if (!loom_llvmir_bitcode_is_power_of_two_u64(alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode align attribute is not a power of "
                            "two");
  }
  if (alignment > kMaxAlignment) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode align attribute is too large");
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_encode_alignment(
    uint32_t alignment, uint64_t* out_encoded_alignment) {
  if (alignment == 0) {
    *out_encoded_alignment = 0;
    return iree_ok_status();
  }
  if (!loom_llvmir_bitcode_is_power_of_two_u64(alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode memory alignment is not a power of "
                            "two");
  }
  uint64_t exponent = 0;
  uint32_t remaining_alignment = alignment;
  while (remaining_alignment > 1) {
    remaining_alignment >>= 1;
    exponent += 1;
  }
  *out_encoded_alignment = exponent + 1;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_memory_volatile(
    uint32_t flags, uint64_t* out_is_volatile) {
  if ((flags & ~LOOM_LLVMIR_MEMORY_VOLATILE) != 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "LLVM bitcode memory flag emission is not "
                            "implemented");
  }
  *out_is_volatile = (flags & LOOM_LLVMIR_MEMORY_VOLATILE) ? 1 : 0;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_attr_encoding(
    const loom_llvmir_attr_t* attr, uint64_t* out_record_kind,
    uint64_t* out_attr_kind, uint64_t* out_value) {
  *out_value = 0;
  switch (attr->kind) {
    case LOOM_LLVMIR_ATTR_ALIGN: {
      IREE_RETURN_IF_ERROR(
          loom_llvmir_bitcode_check_attr_alignment(attr->value));
      *out_record_kind = 1;
      *out_attr_kind = LOOM_LLVMIR_BITCODE_ATTR_KIND_ALIGNMENT;
      *out_value = attr->value;
      return iree_ok_status();
    }
    case LOOM_LLVMIR_ATTR_NOALIAS:
      *out_record_kind = 0;
      *out_attr_kind = LOOM_LLVMIR_BITCODE_ATTR_KIND_NO_ALIAS;
      return iree_ok_status();
    case LOOM_LLVMIR_ATTR_READONLY:
      *out_record_kind = 0;
      *out_attr_kind = LOOM_LLVMIR_BITCODE_ATTR_KIND_READ_ONLY;
      return iree_ok_status();
    case LOOM_LLVMIR_ATTR_WRITEONLY:
      *out_record_kind = 0;
      *out_attr_kind = LOOM_LLVMIR_BITCODE_ATTR_KIND_WRITEONLY;
      return iree_ok_status();
    case LOOM_LLVMIR_ATTR_READNONE:
      *out_record_kind = 0;
      *out_attr_kind = LOOM_LLVMIR_BITCODE_ATTR_KIND_READ_NONE;
      return iree_ok_status();
    case LOOM_LLVMIR_ATTR_NOUNDEF:
      *out_record_kind = 0;
      *out_attr_kind = LOOM_LLVMIR_BITCODE_ATTR_KIND_NOUNDEF;
      return iree_ok_status();
    case LOOM_LLVMIR_ATTR_NONNULL:
      *out_record_kind = 0;
      *out_attr_kind = LOOM_LLVMIR_BITCODE_ATTR_KIND_NON_NULL;
      return iree_ok_status();
    case LOOM_LLVMIR_ATTR_INREG:
      *out_record_kind = 0;
      *out_attr_kind = LOOM_LLVMIR_BITCODE_ATTR_KIND_IN_REG;
      return iree_ok_status();
    case LOOM_LLVMIR_ATTR_ALWAYSINLINE:
      *out_record_kind = 0;
      *out_attr_kind = LOOM_LLVMIR_BITCODE_ATTR_KIND_ALWAYS_INLINE;
      return iree_ok_status();
    case LOOM_LLVMIR_ATTR_RANGE:
    case LOOM_LLVMIR_ATTR_STRING_KEY:
    case LOOM_LLVMIR_ATTR_STRING_KEY_VALUE:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "LLVM bitcode attribute emission is not "
                              "implemented");
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM attribute kind");
  }
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

static bool loom_llvmir_bitcode_parameter_has_attrs(
    const loom_llvmir_parameter_t* parameter) {
  return parameter->attrs.attr_count != 0;
}

static bool loom_llvmir_bitcode_function_has_parameter_attrs(
    const loom_llvmir_function_t* function) {
  for (iree_host_size_t i = 0; i < function->parameter_count; ++i) {
    if (loom_llvmir_bitcode_parameter_has_attrs(&function->parameters[i])) {
      return true;
    }
  }
  return false;
}

static bool loom_llvmir_bitcode_module_has_parameter_attrs(
    const loom_llvmir_module_t* module) {
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    if (loom_llvmir_bitcode_function_has_parameter_attrs(
            module->functions[i])) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_llvmir_bitcode_function_attr_list_id(
    const loom_llvmir_module_t* module, const loom_llvmir_function_t* function,
    uint64_t* out_attr_list_id) {
  if (!loom_llvmir_bitcode_function_has_parameter_attrs(function)) {
    *out_attr_list_id = 0;
    return iree_ok_status();
  }
  uint64_t attr_list_id = 1;
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    if (module->functions[i] == function) {
      *out_attr_list_id = attr_list_id;
      return iree_ok_status();
    }
    if (loom_llvmir_bitcode_function_has_parameter_attrs(
            module->functions[i])) {
      attr_list_id += 1;
    }
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "LLVM bitcode function is not owned by module");
}

static iree_status_t loom_llvmir_bitcode_write_parameter_attr_group_record(
    const loom_llvmir_parameter_t* parameter, uint64_t group_id,
    uint64_t attribute_index, loom_llvmir_bitcode_record_writer_t* writer) {
  if (parameter->attrs.attr_count > (IREE_HOST_SIZE_MAX - 2) / 3) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM bitcode attribute group operand count "
                            "overflows");
  }
  iree_host_size_t operand_capacity = 2 + parameter->attrs.attr_count * 3;
  uint64_t stack_operands[17];
  uint64_t* operands = stack_operands;
  if (operand_capacity > IREE_ARRAYSIZE(stack_operands)) {
    if (operand_capacity > IREE_HOST_SIZE_MAX / sizeof(*operands)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "LLVM bitcode attribute group operand storage "
                              "overflows");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        iree_allocator_system(), operand_capacity * sizeof(*operands),
        (void**)&operands));
  }

  iree_host_size_t operand_count = 0;
  operands[operand_count++] = group_id;
  operands[operand_count++] = attribute_index;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < parameter->attrs.attr_count && iree_status_is_ok(status); ++i) {
    uint64_t record_kind = 0;
    uint64_t attr_kind = 0;
    uint64_t attr_value = 0;
    status = loom_llvmir_bitcode_attr_encoding(
        &parameter->attrs.attrs[i], &record_kind, &attr_kind, &attr_value);
    if (iree_status_is_ok(status)) {
      operands[operand_count++] = record_kind;
      operands[operand_count++] = attr_kind;
      if (record_kind == 1) {
        operands[operand_count++] = attr_value;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_write_unabbrev_record(
        writer, LOOM_LLVMIR_BITCODE_PARAMETER_ATTR_GROUP_CODE_ENTRY, operands,
        operand_count);
  }
  if (operands != stack_operands) {
    iree_allocator_free(iree_allocator_system(), operands);
  }
  return status;
}

static iree_status_t loom_llvmir_bitcode_write_parameter_attr_group_block(
    const loom_llvmir_module_t* module,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (!loom_llvmir_bitcode_module_has_parameter_attrs(module)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_record_writer_enter_subblock(
      writer, LOOM_LLVMIR_BITCODE_PARAMETERATTR_GROUP_BLOCK,
      LOOM_LLVMIR_BITCODE_PARAMETER_ATTR_GROUP_ABBREV_WIDTH));

  iree_status_t status = iree_ok_status();
  uint64_t group_id = 1;
  for (iree_host_size_t i = 0;
       i < module->function_count && iree_status_is_ok(status); ++i) {
    const loom_llvmir_function_t* function = module->functions[i];
    for (iree_host_size_t j = 0;
         j < function->parameter_count && iree_status_is_ok(status); ++j) {
      const loom_llvmir_parameter_t* parameter = &function->parameters[j];
      if (!loom_llvmir_bitcode_parameter_has_attrs(parameter)) continue;
      status = loom_llvmir_bitcode_write_parameter_attr_group_record(
          parameter, group_id, j + 1, writer);
      if (iree_status_is_ok(status)) {
        group_id += 1;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_exit_block(writer);
  }
  return status;
}

static iree_host_size_t loom_llvmir_bitcode_count_parameter_attr_groups(
    const loom_llvmir_function_t* function) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < function->parameter_count; ++i) {
    if (loom_llvmir_bitcode_parameter_has_attrs(&function->parameters[i])) {
      count += 1;
    }
  }
  return count;
}

static iree_status_t loom_llvmir_bitcode_write_parameter_attr_record(
    const loom_llvmir_function_t* function, uint64_t* inout_group_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  iree_host_size_t operand_count =
      loom_llvmir_bitcode_count_parameter_attr_groups(function);
  uint64_t stack_operands[16];
  uint64_t* operands = stack_operands;
  if (operand_count > IREE_ARRAYSIZE(stack_operands)) {
    if (operand_count > IREE_HOST_SIZE_MAX / sizeof(*operands)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "LLVM bitcode attribute list operand storage "
                              "overflows");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        iree_allocator_system(), operand_count * sizeof(*operands),
        (void**)&operands));
  }

  iree_host_size_t written_operand_count = 0;
  for (iree_host_size_t i = 0; i < function->parameter_count; ++i) {
    if (!loom_llvmir_bitcode_parameter_has_attrs(&function->parameters[i])) {
      continue;
    }
    operands[written_operand_count++] = (*inout_group_id)++;
  }
  iree_status_t status =
      loom_llvmir_bitcode_record_writer_write_unabbrev_record(
          writer, LOOM_LLVMIR_BITCODE_PARAMETER_ATTR_CODE_ENTRY, operands,
          written_operand_count);
  if (operands != stack_operands) {
    iree_allocator_free(iree_allocator_system(), operands);
  }
  return status;
}

static iree_status_t loom_llvmir_bitcode_write_parameter_attr_block(
    const loom_llvmir_module_t* module,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (!loom_llvmir_bitcode_module_has_parameter_attrs(module)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_record_writer_enter_subblock(
      writer, LOOM_LLVMIR_BITCODE_PARAMETERATTR_BLOCK,
      LOOM_LLVMIR_BITCODE_PARAMETER_ATTR_ABBREV_WIDTH));

  iree_status_t status = iree_ok_status();
  uint64_t group_id = 1;
  for (iree_host_size_t i = 0;
       i < module->function_count && iree_status_is_ok(status); ++i) {
    const loom_llvmir_function_t* function = module->functions[i];
    if (!loom_llvmir_bitcode_function_has_parameter_attrs(function)) continue;
    status = loom_llvmir_bitcode_write_parameter_attr_record(function,
                                                             &group_id, writer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_exit_block(writer);
  }
  return status;
}

typedef struct loom_llvmir_bitcode_function_value_map_t {
  // Module-global value ID to LLVM bitcode value ID map.
  uint64_t* value_ids;
  // Number of entries in |value_ids|.
  iree_host_size_t value_count;
  // First function-local constant value ID after function parameters.
  uint64_t first_constant_value_id;
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

static iree_status_t loom_llvmir_bitcode_map_mark_constant(
    const loom_llvmir_module_t* module,
    loom_llvmir_bitcode_function_value_map_t* map,
    loom_llvmir_value_id_t value_id) {
  if (value_id >= module->value_count || value_id >= map->value_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode constant scan saw unknown value");
  }
  if (!loom_llvmir_bitcode_is_constant_value_kind(
          module->values[value_id].kind)) {
    return iree_ok_status();
  }
  if (map->value_ids[value_id] == LOOM_LLVMIR_BITCODE_VALUE_ID_INVALID) {
    map->value_ids[value_id] = LOOM_LLVMIR_BITCODE_VALUE_ID_PENDING_CONSTANT;
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitcode_map_mark_instruction_constants(
    const loom_llvmir_module_t* module,
    loom_llvmir_bitcode_function_value_map_t* map,
    const loom_llvmir_instruction_t* instruction) {
  switch (instruction->kind) {
    case LOOM_LLVMIR_INST_PHI: {
      for (iree_host_size_t i = 0; i < instruction->phi.incoming_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
            module, map, instruction->phi.incoming[i].value));
      }
      return iree_ok_status();
    }
    case LOOM_LLVMIR_INST_BINOP: {
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->binop.lhs));
      return loom_llvmir_bitcode_map_mark_constant(module, map,
                                                   instruction->binop.rhs);
    }
    case LOOM_LLVMIR_INST_GEP: {
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->gep.base));
      for (iree_host_size_t i = 0; i < instruction->gep.index_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
            module, map, instruction->gep.indices[i]));
      }
      return iree_ok_status();
    }
    case LOOM_LLVMIR_INST_LOAD:
      return loom_llvmir_bitcode_map_mark_constant(module, map,
                                                   instruction->load.pointer);
    case LOOM_LLVMIR_INST_STORE: {
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->store.value));
      return loom_llvmir_bitcode_map_mark_constant(module, map,
                                                   instruction->store.pointer);
    }
    case LOOM_LLVMIR_INST_CALL: {
      for (iree_host_size_t i = 0; i < instruction->call.arg_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
            module, map, instruction->call.args[i]));
      }
      return iree_ok_status();
    }
    case LOOM_LLVMIR_INST_INLINE_ASM: {
      for (iree_host_size_t i = 0; i < instruction->inline_asm.arg_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_map_mark_constant(
            module, map, instruction->inline_asm.args[i]));
      }
      return iree_ok_status();
    }
    case LOOM_LLVMIR_INST_RET:
      if (!instruction->ret.has_value) return iree_ok_status();
      return loom_llvmir_bitcode_map_mark_constant(module, map,
                                                   instruction->ret.value);
    case LOOM_LLVMIR_INST_BR:
    case LOOM_LLVMIR_INST_UNREACHABLE:
      return iree_ok_status();
    case LOOM_LLVMIR_INST_COND_BR:
      return loom_llvmir_bitcode_map_mark_constant(
          module, map, instruction->cond_br.condition);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM instruction kind");
  }
}

static void loom_llvmir_bitcode_function_value_map_deinitialize(
    loom_llvmir_bitcode_function_value_map_t* map) {
  if (map->value_ids != NULL) {
    iree_allocator_free(iree_allocator_system(), map->value_ids);
  }
}

static iree_status_t loom_llvmir_bitcode_function_value_map_initialize(
    const loom_llvmir_function_t* function,
    loom_llvmir_bitcode_function_value_map_t* out_map) {
  memset(out_map, 0, sizeof(*out_map));
  const loom_llvmir_module_t* module = function->module;
  if (module->value_count == 0) {
    uint64_t first_local_value_id = module->function_count;
    out_map->first_constant_value_id = first_local_value_id;
    out_map->first_instruction_value_id = first_local_value_id;
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
  out_map->first_constant_value_id = next_value_id;
  for (iree_host_size_t i = 0;
       i < function->block_count && iree_status_is_ok(status); ++i) {
    const loom_llvmir_block_t* block = function->blocks[i];
    for (iree_host_size_t j = 0;
         j < block->instruction_count && iree_status_is_ok(status); ++j) {
      status = loom_llvmir_bitcode_map_mark_instruction_constants(
          module, out_map, &block->instructions[j]);
    }
  }
  for (iree_host_size_t i = 0;
       i < module->value_count && iree_status_is_ok(status); ++i) {
    if (out_map->value_ids[i] !=
        LOOM_LLVMIR_BITCODE_VALUE_ID_PENDING_CONSTANT) {
      continue;
    }
    status = loom_llvmir_bitcode_check_value_id_range(next_value_id);
    if (iree_status_is_ok(status)) {
      out_map->value_ids[i] = next_value_id++;
    }
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

static bool loom_llvmir_bitcode_function_has_constants(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* map) {
  for (iree_host_size_t i = 0; i < module->value_count; ++i) {
    if (!loom_llvmir_bitcode_is_constant_value_kind(module->values[i].kind)) {
      continue;
    }
    if (map->value_ids[i] != LOOM_LLVMIR_BITCODE_VALUE_ID_INVALID &&
        map->value_ids[i] != LOOM_LLVMIR_BITCODE_VALUE_ID_PENDING_CONSTANT) {
      return true;
    }
  }
  return false;
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

static iree_status_t loom_llvmir_bitcode_push_function(
    const loom_llvmir_module_t* module, loom_llvmir_function_id_t function_id,
    uint64_t instruction_value_id, uint64_t* operands,
    iree_host_size_t* inout_operand_count) {
  if (function_id >= module->function_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode instruction references unknown "
                            "function");
  }
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_check_value_id_range(instruction_value_id));
  if (function_id >= instruction_value_id) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LLVM bitcode instruction operand references a forward function");
  }
  uint32_t relative_value_id =
      (uint32_t)instruction_value_id - (uint32_t)function_id;
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

static iree_status_t loom_llvmir_bitcode_check_constant(
    const loom_llvmir_module_t* module, const loom_llvmir_value_t* constant) {
  if (constant->type_id >= module->type_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode constant references unknown type");
  }
  const loom_llvmir_type_t* type = &module->types[constant->type_id];
  switch (constant->kind) {
    case LOOM_LLVMIR_VALUE_CONSTANT_INTEGER:
      if (type->kind != LOOM_LLVMIR_TYPE_INTEGER || type->bit_width > 64) {
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "LLVM bitcode integer constant emission is not "
                                "implemented for this type");
      }
      return iree_ok_status();
    case LOOM_LLVMIR_VALUE_CONSTANT_FLOAT_BITS:
      if (type->kind != LOOM_LLVMIR_TYPE_FLOAT) {
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "LLVM bitcode floating constant emission is "
                                "not implemented for this type");
      }
      return iree_ok_status();
    case LOOM_LLVMIR_VALUE_CONSTANT_NULL:
      if (type->kind == LOOM_LLVMIR_TYPE_VOID) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM bitcode null constant has invalid type");
      }
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVM bitcode value is not a constant");
  }
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
      if (instruction->gep.index_count > (IREE_HOST_SIZE_MAX - 4) / 2) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "LLVM bitcode gep operand count overflows");
      }
      return iree_ok_status();
    case LOOM_LLVMIR_INST_LOAD: {
      uint64_t encoded_alignment = 0;
      uint64_t is_volatile = 0;
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_encode_alignment(
          instruction->load.alignment, &encoded_alignment));
      return loom_llvmir_bitcode_memory_volatile(instruction->load.flags,
                                                 &is_volatile);
    }
    case LOOM_LLVMIR_INST_STORE: {
      uint64_t encoded_alignment = 0;
      uint64_t is_volatile = 0;
      IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_encode_alignment(
          instruction->store.alignment, &encoded_alignment));
      return loom_llvmir_bitcode_memory_volatile(instruction->store.flags,
                                                 &is_volatile);
    }
    case LOOM_LLVMIR_INST_CALL:
      if (instruction->call.result_attrs.attr_count != 0) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "LLVM bitcode call result attribute emission is not implemented");
      }
      return iree_ok_status();
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

static iree_status_t loom_llvmir_bitcode_check_attr_list(
    loom_llvmir_attr_list_t attrs) {
  for (iree_host_size_t i = 0; i < attrs.attr_count; ++i) {
    uint64_t record_kind = 0;
    uint64_t attr_kind = 0;
    uint64_t attr_value = 0;
    IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_attr_encoding(
        &attrs.attrs[i], &record_kind, &attr_kind, &attr_value));
  }
  return iree_ok_status();
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
    IREE_RETURN_IF_ERROR(
        loom_llvmir_bitcode_check_attr_list(function->parameters[i].attrs));
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
      case LOOM_LLVMIR_VALUE_CONSTANT_NULL: {
        IREE_RETURN_IF_ERROR(
            loom_llvmir_bitcode_check_constant(module, &module->values[i]));
        break;
      }
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
  uint64_t attr_list_id = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_function_attr_list_id(
      module, function, &attr_list_id));

  uint64_t operands[] = {
      name_offset,
      function->name.size,
      loom_llvmir_bitcode_function_type_id(module, function->id),
      calling_convention,
      function->kind == LOOM_LLVMIR_FUNCTION_DECLARATION,
      linkage,
      attr_list_id,
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

static iree_status_t loom_llvmir_bitcode_write_constant(
    const loom_llvmir_module_t* module, const loom_llvmir_value_t* constant,
    loom_llvmir_type_id_t* inout_current_type,
    loom_llvmir_bitcode_record_writer_t* writer) {
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_check_constant(module, constant));
  if (*inout_current_type != constant->type_id) {
    IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_write_record_u64(
        writer, LOOM_LLVMIR_BITCODE_CONSTANT_CODE_SETTYPE, constant->type_id));
    *inout_current_type = constant->type_id;
  }

  const loom_llvmir_type_t* type = &module->types[constant->type_id];
  switch (constant->kind) {
    case LOOM_LLVMIR_VALUE_CONSTANT_INTEGER: {
      uint64_t value = loom_llvmir_bitcode_sign_extend_integer_constant(
          constant->integer_value, type->bit_width);
      return loom_llvmir_bitcode_write_record_u64(
          writer, LOOM_LLVMIR_BITCODE_CONSTANT_CODE_INTEGER,
          loom_llvmir_bitcode_encode_signed((int64_t)value));
    }
    case LOOM_LLVMIR_VALUE_CONSTANT_FLOAT_BITS:
      return loom_llvmir_bitcode_write_record_u64(
          writer, LOOM_LLVMIR_BITCODE_CONSTANT_CODE_FLOAT,
          constant->float_bits);
    case LOOM_LLVMIR_VALUE_CONSTANT_NULL:
      return loom_llvmir_bitcode_write_empty_record(
          writer, LOOM_LLVMIR_BITCODE_CONSTANT_CODE_NULL);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVM bitcode value is not a constant");
  }
}

static iree_status_t loom_llvmir_bitcode_write_constants_block(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (!loom_llvmir_bitcode_function_has_constants(module, value_map)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_record_writer_enter_subblock(
      writer, LOOM_LLVMIR_BITCODE_CONSTANTS_BLOCK,
      LOOM_LLVMIR_BITCODE_CONSTANT_ABBREV_WIDTH));

  iree_status_t status = iree_ok_status();
  loom_llvmir_type_id_t current_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  uint64_t expected_value_id = value_map->first_constant_value_id;
  for (iree_host_size_t i = 0;
       i < module->value_count && iree_status_is_ok(status); ++i) {
    const loom_llvmir_value_t* value = &module->values[i];
    if (!loom_llvmir_bitcode_is_constant_value_kind(value->kind)) continue;
    if (value_map->value_ids[i] == LOOM_LLVMIR_BITCODE_VALUE_ID_INVALID) {
      continue;
    }
    if (value_map->value_ids[i] != expected_value_id) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "LLVM bitcode constant value numbering is inconsistent");
    } else {
      status = loom_llvmir_bitcode_write_constant(module, value, &current_type,
                                                  writer);
      expected_value_id += 1;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_exit_block(writer);
  }
  return status;
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

static iree_status_t loom_llvmir_bitcode_write_gep(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (instruction->gep.index_count > (IREE_HOST_SIZE_MAX - 4) / 2) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM bitcode gep operand count overflows");
  }
  iree_host_size_t operand_capacity = 4 + instruction->gep.index_count * 2;
  uint64_t stack_operands[16];
  uint64_t* operands = stack_operands;
  if (operand_capacity > IREE_ARRAYSIZE(stack_operands)) {
    if (operand_capacity > IREE_HOST_SIZE_MAX / sizeof(*operands)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "LLVM bitcode gep operand storage overflows");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        iree_allocator_system(), operand_capacity * sizeof(*operands),
        (void**)&operands));
  }

  iree_host_size_t operand_count = 0;
  operands[operand_count++] = 0;
  operands[operand_count++] = instruction->gep.element_type;
  iree_status_t status = loom_llvmir_bitcode_push_value_type_pair(
      module, value_map, instruction->gep.base, instruction_value_id, operands,
      &operand_count);
  for (iree_host_size_t i = 0;
       i < instruction->gep.index_count && iree_status_is_ok(status); ++i) {
    status = loom_llvmir_bitcode_push_value_type_pair(
        module, value_map, instruction->gep.indices[i], instruction_value_id,
        operands, &operand_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_write_unabbrev_record(
        writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_GEP, operands,
        operand_count);
  }
  if (operands != stack_operands) {
    iree_allocator_free(iree_allocator_system(), operands);
  }
  return status;
}

static iree_status_t loom_llvmir_bitcode_write_load(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  uint64_t encoded_alignment = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_encode_alignment(
      instruction->load.alignment, &encoded_alignment));
  uint64_t is_volatile = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_memory_volatile(
      instruction->load.flags, &is_volatile));

  uint64_t operands[5];
  iree_host_size_t operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value_type_pair(
      module, value_map, instruction->load.pointer, instruction_value_id,
      operands, &operand_count));
  operands[operand_count++] = instruction->load.result_type;
  operands[operand_count++] = encoded_alignment;
  operands[operand_count++] = is_volatile;
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
      writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_LOAD, operands,
      operand_count);
}

static iree_status_t loom_llvmir_bitcode_write_store(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  uint64_t encoded_alignment = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_encode_alignment(
      instruction->store.alignment, &encoded_alignment));
  uint64_t is_volatile = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_memory_volatile(
      instruction->store.flags, &is_volatile));

  uint64_t operands[6];
  iree_host_size_t operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value_type_pair(
      module, value_map, instruction->store.pointer, instruction_value_id,
      operands, &operand_count));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_push_value_type_pair(
      module, value_map, instruction->store.value, instruction_value_id,
      operands, &operand_count));
  operands[operand_count++] = encoded_alignment;
  operands[operand_count++] = is_volatile;
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(
      writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_STORE, operands,
      operand_count);
}

static iree_status_t loom_llvmir_bitcode_write_call(
    const loom_llvmir_module_t* module,
    const loom_llvmir_bitcode_function_value_map_t* value_map,
    const loom_llvmir_instruction_t* instruction, uint64_t instruction_value_id,
    loom_llvmir_bitcode_record_writer_t* writer) {
  if (instruction->call.callee >= module->function_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode call references unknown callee");
  }
  if (instruction->call.result_attrs.attr_count != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "LLVM bitcode call result attribute emission is not implemented");
  }
  const loom_llvmir_function_t* callee =
      module->functions[instruction->call.callee];
  if (instruction->call.arg_count != callee->parameter_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bitcode call arg count mismatch");
  }
  if (instruction->call.arg_count > IREE_HOST_SIZE_MAX - 5) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM bitcode call operand count overflows");
  }
  iree_host_size_t operand_capacity = 5 + instruction->call.arg_count;
  uint64_t stack_operands[16];
  uint64_t* operands = stack_operands;
  if (operand_capacity > IREE_ARRAYSIZE(stack_operands)) {
    if (operand_capacity > IREE_HOST_SIZE_MAX / sizeof(*operands)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "LLVM bitcode call operand storage overflows");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        iree_allocator_system(), operand_capacity * sizeof(*operands),
        (void**)&operands));
  }

  uint64_t calling_convention = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_function_calling_convention(
      callee->calling_convention, &calling_convention));

  iree_host_size_t operand_count = 0;
  operands[operand_count++] = 0;
  operands[operand_count++] =
      (calling_convention << LOOM_LLVMIR_BITCODE_CALL_FLAG_CCONV) |
      (UINT64_C(1) << LOOM_LLVMIR_BITCODE_CALL_FLAG_EXPLICIT_TYPE);
  operands[operand_count++] =
      loom_llvmir_bitcode_function_type_id(module, callee->id);
  iree_status_t status = loom_llvmir_bitcode_push_function(
      module, instruction->call.callee, instruction_value_id, operands,
      &operand_count);
  for (iree_host_size_t i = 0;
       i < instruction->call.arg_count && iree_status_is_ok(status); ++i) {
    status = loom_llvmir_bitcode_push_value(
        value_map, instruction->call.args[i], instruction_value_id, operands,
        &operand_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_record_writer_write_unabbrev_record(
        writer, LOOM_LLVMIR_BITCODE_FUNCTION_CODE_INST_CALL, operands,
        operand_count);
  }
  if (operands != stack_operands) {
    iree_allocator_free(iree_allocator_system(), operands);
  }
  return status;
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
    case LOOM_LLVMIR_INST_GEP:
      return loom_llvmir_bitcode_write_gep(module, value_map, instruction,
                                           instruction_value_id, writer);
    case LOOM_LLVMIR_INST_LOAD:
      return loom_llvmir_bitcode_write_load(module, value_map, instruction,
                                            instruction_value_id, writer);
    case LOOM_LLVMIR_INST_STORE:
      return loom_llvmir_bitcode_write_store(module, value_map, instruction,
                                             instruction_value_id, writer);
    case LOOM_LLVMIR_INST_CALL:
      return loom_llvmir_bitcode_write_call(module, value_map, instruction,
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
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_write_constants_block(function->module,
                                                       &value_map, writer);
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
    status = loom_llvmir_bitcode_write_parameter_attr_group_block(
        module, &record_writer);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_llvmir_bitcode_write_parameter_attr_block(module, &record_writer);
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
