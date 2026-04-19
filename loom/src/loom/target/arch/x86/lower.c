// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/lower.h"

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/ops.h"

static bool loom_x86_type_is_vector_16xi32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32 &&
         loom_type_dim_static_size_at(type, 0) == 16;
}

static bool loom_x86_value_is_vector_16xi32(loom_low_lower_context_t* context,
                                            loom_value_id_t value_id) {
  return loom_x86_type_is_vector_16xi32(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

static iree_status_t loom_x86_make_zmm_type(loom_low_lower_context_t* context,
                                            loom_type_t* out_type) {
  loom_string_id_t register_class_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(loom_low_lower_context_module(context),
                                IREE_SV("x86.zmm"), &register_class_id));
  *out_type = loom_type_register(register_class_id, 1);
  return iree_ok_status();
}

static iree_status_t loom_x86_map_type(void* user_data,
                                       loom_low_lower_context_t* context,
                                       const loom_op_t* source_op,
                                       loom_type_t source_type,
                                       loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_x86_type_is_vector_16xi32(source_type)) {
    return loom_x86_make_zmm_type(context, out_low_type);
  }
  return loom_low_lower_emit_reject(
      context, source_op, IREE_SV("type"), IREE_SV("source"),
      IREE_SV("x86 AVX512 lowering currently supports only vector<16xi32> "
              "values"));
}

static bool loom_x86_can_lower_vector_addi(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op) {
  return loom_x86_value_is_vector_16xi32(context,
                                         loom_vector_addi_lhs(source_op)) &&
         loom_x86_value_is_vector_16xi32(context,
                                         loom_vector_addi_rhs(source_op)) &&
         loom_x86_value_is_vector_16xi32(context,
                                         loom_vector_addi_result(source_op));
}

static iree_status_t loom_x86_can_lower_op(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           bool* out_handled) {
  (void)user_data;
  switch (source_op->kind) {
    case LOOM_OP_VECTOR_ADDI:
      *out_handled = loom_x86_can_lower_vector_addi(context, source_op);
      return iree_ok_status();
    default:
      *out_handled = false;
      return iree_ok_status();
  }
}

static iree_status_t loom_x86_intern(loom_low_lower_context_t* context,
                                     iree_string_view_t string,
                                     loom_string_id_t* out_string_id) {
  return loom_module_intern_string(loom_low_lower_context_module(context),
                                   string, out_string_id);
}

static iree_status_t loom_x86_low_result_type(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_value_id_t source_result,
                                              loom_type_t* out_low_type) {
  IREE_RETURN_IF_ERROR(loom_low_lower_map_type(
      context, source_op,
      loom_module_value_type(loom_low_lower_context_module(context),
                             source_result),
      out_low_type));
  if (!loom_type_is_register(*out_low_type)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "x86 source type did not map to a register");
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_lower_vector_addi(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_operands[2] = {LOOM_VALUE_ID_INVALID,
                                     LOOM_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_addi_lhs(source_op), &low_operands[0]));
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_addi_rhs(source_op), &low_operands[1]));

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_x86_low_result_type(
      context, source_op, loom_vector_addi_result(source_op), &result_type));

  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_x86_intern(context, IREE_SV("x86.avx512.vpaddd.zmm"), &opcode_id));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_op_build(
      loom_low_lower_context_builder(context), opcode_id, low_operands,
      IREE_ARRAYSIZE(low_operands), loom_make_named_attr_slice(NULL, 0),
      &result_type, 1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, &low_op));
  return loom_low_lower_bind_value(
      context, loom_vector_addi_result(source_op),
      loom_value_slice_get(loom_low_op_results(low_op), 0));
}

static iree_status_t loom_x86_try_lower_op(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           bool* out_handled) {
  IREE_RETURN_IF_ERROR(
      loom_x86_can_lower_op(user_data, context, source_op, out_handled));
  if (!*out_handled) {
    return iree_ok_status();
  }

  switch (source_op->kind) {
    case LOOM_OP_VECTOR_ADDI:
      return loom_x86_lower_vector_addi(context, source_op);
    default:
      return iree_ok_status();
  }
}

static const loom_low_lower_policy_t kX86LowLowerPolicy = {
    .name = IREE_SVL("x86-avx512-lower"),
    .map_type = {.fn = loom_x86_map_type, .user_data = NULL},
    .can_lower_op = {.fn = loom_x86_can_lower_op, .user_data = NULL},
    .try_lower_op = {.fn = loom_x86_try_lower_op, .user_data = NULL},
};

const loom_low_lower_policy_t* loom_x86_low_lower_policy(void) {
  return &kX86LowLowerPolicy;
}
