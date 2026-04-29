// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/builder.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/descriptor_traits.h"
#include "loom/ir/module.h"

iree_status_t loom_low_build_register_class_string_id(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    uint16_t reg_class_id, loom_string_id_t* out_string_id) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(out_string_id);
  *out_string_id = LOOM_STRING_ID_INVALID;
  if (reg_class_id >= descriptor_set->reg_class_count) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "target-low register class ID %" PRIu16
                            " is not present in the selected descriptor set",
                            reg_class_id);
  }

  const loom_low_reg_class_t* reg_class =
      &descriptor_set->reg_classes[reg_class_id];
  iree_string_view_t reg_class_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
      descriptor_set, reg_class->name_string_offset, &reg_class_name));
  if (iree_string_view_is_empty(reg_class_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low register class ID %" PRIu16
                            " has no register class name",
                            reg_class_id);
  }
  return loom_module_intern_string(module, reg_class_name, out_string_id);
}

iree_status_t loom_low_build_register_type(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    uint16_t reg_class_id, uint32_t unit_count, loom_type_t* out_type) {
  IREE_ASSERT_ARGUMENT(out_type);
  *out_type = loom_type_none();
  if (unit_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "register type unit count must be non-zero");
  }
  loom_string_id_t reg_class_string_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_build_register_class_string_id(
      module, descriptor_set, reg_class_id, &reg_class_string_id));
  *out_type = loom_type_register(reg_class_string_id, unit_count);
  return iree_ok_status();
}

static iree_status_t loom_low_build_resolved_descriptor_opcode_id(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_string_id_t* out_opcode_id) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(descriptor);
  IREE_ASSERT_ARGUMENT(out_opcode_id);
  *out_opcode_id = LOOM_STRING_ID_INVALID;
  iree_string_view_t key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
      descriptor_set, descriptor->key_string_offset, &key));
  if (iree_string_view_is_empty(key)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low descriptor ID 0x%016" PRIx64
                            " has no descriptor key",
                            descriptor->stable_id);
  }
  return loom_module_intern_string(builder->module, key, out_opcode_id);
}

static iree_status_t loom_low_build_lookup_descriptor(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint64_t descriptor_id, const loom_low_descriptor_t** out_descriptor,
    loom_string_id_t* out_opcode_id) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(out_descriptor);
  IREE_ASSERT_ARGUMENT(out_opcode_id);
  *out_descriptor = NULL;
  *out_opcode_id = LOOM_STRING_ID_INVALID;
  if (descriptor_id == LOOM_LOW_DESCRIPTOR_ID_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low descriptor ID is required");
  }

  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_lookup_descriptor_by_id(descriptor_set,
                                                      descriptor_id);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "target-low descriptor ID 0x%016" PRIx64
                            " is not present in the selected descriptor set",
                            descriptor_id);
  }
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, descriptor_ordinal);
  if (descriptor == NULL) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "target-low descriptor ID 0x%016" PRIx64
        " mapped to an out-of-range descriptor ordinal %" PRIu32,
        descriptor_id, descriptor_ordinal);
  }
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_opcode_id(
      builder, descriptor_set, descriptor, out_opcode_id));
  *out_descriptor = descriptor;
  return iree_ok_status();
}

iree_status_t loom_low_build_resolved_descriptor_op(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_string_id_t opcode_id,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attrs, const loom_type_t* result_types,
    iree_host_size_t result_count, const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count, loom_location_id_t location,
    loom_op_t** out_op) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(descriptor);
  IREE_ASSERT(opcode_id != LOOM_STRING_ID_INVALID);
  IREE_ASSERT_ARGUMENT(out_op);
  IREE_ASSERT_ARGUMENT(operand_count == 0 || operands);
  IREE_ASSERT_ARGUMENT(result_count == 0 || result_types);
  IREE_ASSERT_ARGUMENT(tied_result_count == 0 || tied_results);
  *out_op = NULL;
  if (operand_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "low.op operand count exceeds uint16_t range");
  }
  if (result_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "low.op result count exceeds uint16_t range");
  }
  if (tied_result_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "low.op tied result count exceeds uint16_t range");
  }

  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
      builder, LOOM_OP_LOW_OP, (uint16_t)operand_count, (uint16_t)result_count,
      /*region_count=*/0, (uint16_t)tied_result_count, /*attribute_count=*/3,
      location, out_op));
  (*out_op)->traits =
      loom_low_descriptor_effective_traits(descriptor_set, descriptor);
  if (operand_count > 0) {
    memcpy(loom_op_operands(*out_op), operands,
           operand_count * sizeof(loom_value_id_t));
  }
  loom_op_attrs(*out_op)[loom_low_op_opcode_ATTR_INDEX] =
      loom_attr_string(opcode_id);
  loom_op_attrs(*out_op)[loom_low_op_descriptor_id_ATTR_INDEX] =
      loom_attr_i64((int64_t)descriptor->stable_id);
  if (attrs.count > 0) {
    IREE_RETURN_IF_ERROR(loom_module_make_canonical_attr_dict(
        builder->module, attrs,
        &loom_op_attrs(*out_op)[loom_low_op_attrs_ATTR_INDEX]));
  }
  for (iree_host_size_t i = 0; i < result_count; ++i) {
    loom_value_id_t result_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_builder_define_value(builder, result_types[i], &result_id));
    loom_op_results(*out_op)[i] = result_id;
  }
  if (tied_result_count > 0) {
    memcpy(loom_op_tied_results(*out_op), tied_results,
           tied_result_count * sizeof(loom_tied_result_t));
  }
  return loom_builder_finalize_op(builder, *out_op);
}

iree_status_t loom_low_build_resolved_descriptor_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_string_id_t opcode_id,
    loom_named_attr_slice_t attrs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(descriptor);
  IREE_ASSERT(opcode_id != LOOM_STRING_ID_INVALID);
  IREE_ASSERT_ARGUMENT(out_op);
  *out_op = NULL;

  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
      builder, LOOM_OP_LOW_CONST, /*operand_count=*/0, /*result_count=*/1,
      /*region_count=*/0, /*tied_result_count=*/0, /*attribute_count=*/3,
      location, out_op));
  (*out_op)->traits =
      (loom_low_descriptor_effective_traits(descriptor_set, descriptor) &
       ~LOOM_TRAIT_PURE) |
      LOOM_TRAIT_UNIQUE_IDENTITY;
  loom_op_attrs(*out_op)[loom_low_const_opcode_ATTR_INDEX] =
      loom_attr_string(opcode_id);
  loom_op_attrs(*out_op)[loom_low_const_descriptor_id_ATTR_INDEX] =
      loom_attr_i64((int64_t)descriptor->stable_id);
  if (attrs.count > 0) {
    IREE_RETURN_IF_ERROR(loom_module_make_canonical_attr_dict(
        builder->module, attrs,
        &loom_op_attrs(*out_op)[loom_low_const_attrs_ATTR_INDEX]));
  }
  loom_value_id_t result_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_builder_define_value(builder, result_type, &result_id));
  loom_op_results(*out_op)[0] = result_id;
  return loom_builder_finalize_op(builder, *out_op);
}

iree_status_t loom_low_build_descriptor_op(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint64_t descriptor_id, const loom_value_id_t* operands,
    iree_host_size_t operand_count, loom_named_attr_slice_t attrs,
    const loom_type_t* result_types, iree_host_size_t result_count,
    const loom_tied_result_t* tied_results, iree_host_size_t tied_result_count,
    loom_location_id_t location, loom_op_t** out_op) {
  IREE_ASSERT_ARGUMENT(out_op);
  *out_op = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  const loom_low_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_lookup_descriptor(
      builder, descriptor_set, descriptor_id, &descriptor, &opcode_id));
  return loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, opcode_id, operands, operand_count,
      attrs, result_types, result_count, tied_results, tied_result_count,
      location, out_op);
}

iree_status_t loom_low_build_descriptor_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint64_t descriptor_id, loom_named_attr_slice_t attrs,
    loom_type_t result_type, loom_location_id_t location, loom_op_t** out_op) {
  IREE_ASSERT_ARGUMENT(out_op);
  *out_op = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  const loom_low_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_lookup_descriptor(
      builder, descriptor_set, descriptor_id, &descriptor, &opcode_id));
  return loom_low_build_resolved_descriptor_const(
      builder, descriptor_set, descriptor, opcode_id, attrs, result_type,
      location, out_op);
}
