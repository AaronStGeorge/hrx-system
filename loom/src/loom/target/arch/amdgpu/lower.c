// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower.h"

#include <stdint.h>

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scalar/ops.h"

static bool loom_amdgpu_type_is_i32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
}

static bool loom_amdgpu_value_is_i32(loom_low_lower_context_t* context,
                                     loom_value_id_t value_id) {
  return loom_amdgpu_type_is_i32(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

static iree_status_t loom_amdgpu_make_sgpr_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  loom_string_id_t register_class_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(loom_low_lower_context_module(context),
                                IREE_SV("amdgpu.sgpr"), &register_class_id));
  *out_type = loom_type_register(register_class_id, 1);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_map_type(void* user_data,
                                          loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_type_t source_type,
                                          loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_amdgpu_type_is_i32(source_type)) {
    return loom_amdgpu_make_sgpr_type(context, out_low_type);
  }
  return loom_low_lower_emit_reject(
      context, source_op, IREE_SV("type"), IREE_SV("source"),
      IREE_SV("AMDGPU SGPR lowering currently supports only i32 scalar "
              "values"));
}

static bool loom_amdgpu_can_lower_constant(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op) {
  if (!loom_amdgpu_value_is_i32(context,
                                loom_scalar_constant_result(source_op))) {
    return false;
  }
  const loom_attribute_t value = loom_scalar_constant_value(source_op);
  return value.kind == LOOM_ATTR_I64 && value.i64 >= INT32_MIN &&
         value.i64 <= INT32_MAX;
}

static bool loom_amdgpu_can_lower_i32_binary(loom_low_lower_context_t* context,
                                             loom_value_id_t lhs,
                                             loom_value_id_t rhs,
                                             loom_value_id_t result) {
  return loom_amdgpu_value_is_i32(context, lhs) &&
         loom_amdgpu_value_is_i32(context, rhs) &&
         loom_amdgpu_value_is_i32(context, result);
}

static iree_status_t loom_amdgpu_can_lower_op(void* user_data,
                                              loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              bool* out_handled) {
  (void)user_data;
  switch (source_op->kind) {
    case LOOM_OP_SCALAR_CONSTANT:
      *out_handled = loom_amdgpu_can_lower_constant(context, source_op);
      return iree_ok_status();
    case LOOM_OP_SCALAR_ADDI:
      *out_handled = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_addi_lhs(source_op),
          loom_scalar_addi_rhs(source_op), loom_scalar_addi_result(source_op));
      return iree_ok_status();
    default:
      *out_handled = false;
      return iree_ok_status();
  }
}

static iree_status_t loom_amdgpu_intern(loom_low_lower_context_t* context,
                                        iree_string_view_t string,
                                        loom_string_id_t* out_string_id) {
  return loom_module_intern_string(loom_low_lower_context_module(context),
                                   string, out_string_id);
}

static iree_status_t loom_amdgpu_low_result_type(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, loom_type_t* out_low_type) {
  IREE_RETURN_IF_ERROR(loom_low_lower_map_type(
      context, source_op,
      loom_module_value_type(loom_low_lower_context_module(context),
                             source_result),
      out_low_type));
  if (!loom_type_is_register(*out_low_type)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU source type did not map to a register");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lower_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_intern(context, IREE_SV("amdgpu.s_mov_b32"), &opcode_id));
  loom_string_id_t value_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_intern(context, IREE_SV("imm32"), &value_name_id));

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, loom_scalar_constant_result(source_op),
      &result_type));

  const int64_t source_value = loom_scalar_constant_value(source_op).i64;
  const uint32_t bit_pattern = (uint32_t)(int32_t)source_value;
  loom_named_attr_t attrs[] = {
      {
          .name_id = value_name_id,
          .value = loom_attr_i64(bit_pattern),
      },
  };
  loom_op_t* low_const = NULL;
  IREE_RETURN_IF_ERROR(loom_low_const_build(
      loom_low_lower_context_builder(context), opcode_id,
      loom_make_named_attr_slice(attrs, IREE_ARRAYSIZE(attrs)), result_type,
      source_op->location, &low_const));
  return loom_low_lower_bind_value(context,
                                   loom_scalar_constant_result(source_op),
                                   loom_low_const_result(low_const));
}

static iree_status_t loom_amdgpu_lower_addi(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  loom_value_id_t low_operands[2] = {LOOM_VALUE_ID_INVALID,
                                     LOOM_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_scalar_addi_lhs(source_op), &low_operands[0]));
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_scalar_addi_rhs(source_op), &low_operands[1]));

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, loom_scalar_addi_result(source_op), &result_type));

  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_intern(context, IREE_SV("amdgpu.s_add_u32"), &opcode_id));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_op_build(
      loom_low_lower_context_builder(context), opcode_id, low_operands,
      IREE_ARRAYSIZE(low_operands), loom_make_named_attr_slice(NULL, 0),
      &result_type, 1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, &low_op));
  return loom_low_lower_bind_value(
      context, loom_scalar_addi_result(source_op),
      loom_value_slice_get(loom_low_op_results(low_op), 0));
}

static iree_status_t loom_amdgpu_try_lower_op(void* user_data,
                                              loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              bool* out_handled) {
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_can_lower_op(user_data, context, source_op, out_handled));
  if (!*out_handled) {
    return iree_ok_status();
  }

  switch (source_op->kind) {
    case LOOM_OP_SCALAR_CONSTANT:
      return loom_amdgpu_lower_constant(context, source_op);
    case LOOM_OP_SCALAR_ADDI:
      return loom_amdgpu_lower_addi(context, source_op);
    default:
      return iree_ok_status();
  }
}

static const loom_low_lower_policy_t kAmdgpuLowLowerPolicy = {
    .name = IREE_SVL("amdgpu-sgpr-lower"),
    .map_type = {.fn = loom_amdgpu_map_type, .user_data = NULL},
    .can_lower_op = {.fn = loom_amdgpu_can_lower_op, .user_data = NULL},
    .try_lower_op = {.fn = loom_amdgpu_try_lower_op, .user_data = NULL},
};

const loom_low_lower_policy_t* loom_amdgpu_low_lower_policy(void) {
  return &kAmdgpuLowLowerPolicy;
}

void loom_amdgpu_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("amdgpu.gfx950.core"),
          .policy = &kAmdgpuLowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("amdgpu.gfx11.core"),
          .policy = &kAmdgpuLowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("amdgpu.gfx12.core"),
          .policy = &kAmdgpuLowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("amdgpu.gfx1250.core"),
          .policy = &kAmdgpuLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
