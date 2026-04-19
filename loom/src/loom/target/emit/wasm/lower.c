// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/wasm/lower.h"

#include <stdint.h>

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"

static bool loom_wasm_type_is_i32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
}

static bool loom_wasm_type_is_vector_4xi32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static bool loom_wasm_value_is_i32(loom_low_lower_context_t* context,
                                   loom_value_id_t value_id) {
  return loom_wasm_type_is_i32(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

static bool loom_wasm_value_is_vector_4xi32(loom_low_lower_context_t* context,
                                            loom_value_id_t value_id) {
  return loom_wasm_type_is_vector_4xi32(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

static iree_status_t loom_wasm_make_i32_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  loom_string_id_t register_class_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(loom_low_lower_context_module(context),
                                IREE_SV("wasm.i32"), &register_class_id));
  *out_type = loom_type_register(register_class_id, 1);
  return iree_ok_status();
}

static iree_status_t loom_wasm_make_v128_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  loom_string_id_t register_class_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(loom_low_lower_context_module(context),
                                IREE_SV("wasm.v128"), &register_class_id));
  *out_type = loom_type_register(register_class_id, 1);
  return iree_ok_status();
}

static iree_status_t loom_wasm_map_type(void* user_data,
                                        loom_low_lower_context_t* context,
                                        const loom_op_t* source_op,
                                        loom_type_t source_type,
                                        loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_wasm_type_is_i32(source_type)) {
    return loom_wasm_make_i32_register_type(context, out_low_type);
  }
  if (loom_wasm_type_is_vector_4xi32(source_type)) {
    return loom_wasm_make_v128_register_type(context, out_low_type);
  }
  return loom_low_lower_emit_reject(
      context, source_op, IREE_SV("type"), IREE_SV("source"),
      IREE_SV("Wasm lowering currently supports only i32 scalar values and "
              "vector<4xi32> SIMD values"));
}

static bool loom_wasm_can_lower_constant(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op) {
  if (!loom_wasm_value_is_i32(context,
                              loom_scalar_constant_result(source_op))) {
    return false;
  }
  const loom_attribute_t value = loom_scalar_constant_value(source_op);
  return value.kind == LOOM_ATTR_I64 && value.i64 >= INT32_MIN &&
         value.i64 <= INT32_MAX;
}

static bool loom_wasm_can_lower_i32_binary(loom_low_lower_context_t* context,
                                           loom_value_id_t lhs,
                                           loom_value_id_t rhs,
                                           loom_value_id_t result) {
  return loom_wasm_value_is_i32(context, lhs) &&
         loom_wasm_value_is_i32(context, rhs) &&
         loom_wasm_value_is_i32(context, result);
}

static bool loom_wasm_can_lower_vector_4xi32_binary(
    loom_low_lower_context_t* context, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_value_id_t result) {
  return loom_wasm_value_is_vector_4xi32(context, lhs) &&
         loom_wasm_value_is_vector_4xi32(context, rhs) &&
         loom_wasm_value_is_vector_4xi32(context, result);
}

static iree_status_t loom_wasm_can_lower_op(void* user_data,
                                            loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            bool* out_handled) {
  (void)user_data;
  switch (source_op->kind) {
    case LOOM_OP_SCALAR_CONSTANT:
      *out_handled = loom_wasm_can_lower_constant(context, source_op);
      return iree_ok_status();
    case LOOM_OP_SCALAR_ADDI:
      *out_handled = loom_wasm_can_lower_i32_binary(
          context, loom_scalar_addi_lhs(source_op),
          loom_scalar_addi_rhs(source_op), loom_scalar_addi_result(source_op));
      return iree_ok_status();
    case LOOM_OP_SCALAR_SUBI:
      *out_handled = loom_wasm_can_lower_i32_binary(
          context, loom_scalar_subi_lhs(source_op),
          loom_scalar_subi_rhs(source_op), loom_scalar_subi_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_ADDI:
      *out_handled = loom_wasm_can_lower_vector_4xi32_binary(
          context, loom_vector_addi_lhs(source_op),
          loom_vector_addi_rhs(source_op), loom_vector_addi_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_MULI:
      *out_handled = loom_wasm_can_lower_vector_4xi32_binary(
          context, loom_vector_muli_lhs(source_op),
          loom_vector_muli_rhs(source_op), loom_vector_muli_result(source_op));
      return iree_ok_status();
    default:
      *out_handled = false;
      return iree_ok_status();
  }
}

static iree_status_t loom_wasm_intern(loom_low_lower_context_t* context,
                                      iree_string_view_t string,
                                      loom_string_id_t* out_string_id) {
  return loom_module_intern_string(loom_low_lower_context_module(context),
                                   string, out_string_id);
}

static iree_status_t loom_wasm_low_result_type(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, loom_type_t* out_low_type) {
  IREE_RETURN_IF_ERROR(loom_low_lower_map_type(
      context, source_op,
      loom_module_value_type(loom_low_lower_context_module(context),
                             source_result),
      out_low_type));
  if (!loom_type_is_register(*out_low_type)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm source type did not map to a register type");
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_lower_constant(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op) {
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_wasm_intern(context, IREE_SV("wasm.i32.const"), &opcode_id));
  loom_string_id_t value_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_wasm_intern(context, IREE_SV("i32_value"), &value_name_id));

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_wasm_low_result_type(
      context, source_op, loom_scalar_constant_result(source_op),
      &result_type));

  loom_named_attr_t attrs[] = {
      {
          .name_id = value_name_id,
          .value = loom_scalar_constant_value(source_op),
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

static iree_status_t loom_wasm_lower_binary(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            iree_string_view_t descriptor_key,
                                            loom_value_id_t source_lhs,
                                            loom_value_id_t source_rhs,
                                            loom_value_id_t source_result) {
  loom_value_id_t low_operands[2] = {LOOM_VALUE_ID_INVALID,
                                     LOOM_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_lhs, &low_operands[0]));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_rhs, &low_operands[1]));

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_wasm_low_result_type(context, source_op,
                                                 source_result, &result_type));

  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_wasm_intern(context, descriptor_key, &opcode_id));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_op_build(
      loom_low_lower_context_builder(context), opcode_id, low_operands,
      IREE_ARRAYSIZE(low_operands), loom_make_named_attr_slice(NULL, 0),
      &result_type, 1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, &low_op));
  return loom_low_lower_bind_value(
      context, source_result,
      loom_value_slice_get(loom_low_op_results(low_op), 0));
}

static iree_status_t loom_wasm_try_lower_op(void* user_data,
                                            loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            bool* out_handled) {
  (void)user_data;
  IREE_RETURN_IF_ERROR(
      loom_wasm_can_lower_op(user_data, context, source_op, out_handled));
  if (!*out_handled) {
    return iree_ok_status();
  }

  switch (source_op->kind) {
    case LOOM_OP_SCALAR_CONSTANT:
      return loom_wasm_lower_constant(context, source_op);
    case LOOM_OP_SCALAR_ADDI:
      return loom_wasm_lower_binary(context, source_op, IREE_SV("wasm.i32.add"),
                                    loom_scalar_addi_lhs(source_op),
                                    loom_scalar_addi_rhs(source_op),
                                    loom_scalar_addi_result(source_op));
    case LOOM_OP_SCALAR_SUBI:
      return loom_wasm_lower_binary(context, source_op, IREE_SV("wasm.i32.sub"),
                                    loom_scalar_subi_lhs(source_op),
                                    loom_scalar_subi_rhs(source_op),
                                    loom_scalar_subi_result(source_op));
    case LOOM_OP_VECTOR_ADDI:
      return loom_wasm_lower_binary(
          context, source_op, IREE_SV("wasm.i32x4.add"),
          loom_vector_addi_lhs(source_op), loom_vector_addi_rhs(source_op),
          loom_vector_addi_result(source_op));
    case LOOM_OP_VECTOR_MULI:
      return loom_wasm_lower_binary(
          context, source_op, IREE_SV("wasm.i32x4.mul"),
          loom_vector_muli_lhs(source_op), loom_vector_muli_rhs(source_op),
          loom_vector_muli_result(source_op));
    default:
      return iree_ok_status();
  }
}

static const loom_low_lower_policy_t kWasmLowLowerPolicy = {
    .name = IREE_SVL("wasm-lower"),
    .map_type = {.fn = loom_wasm_map_type, .user_data = NULL},
    .can_lower_op = {.fn = loom_wasm_can_lower_op, .user_data = NULL},
    .try_lower_op = {.fn = loom_wasm_try_lower_op, .user_data = NULL},
};

const loom_low_lower_policy_t* loom_wasm_low_lower_policy(void) {
  return &kWasmLowLowerPolicy;
}

void loom_wasm_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("wasm.core.simd128"),
          .policy = &kWasmLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
