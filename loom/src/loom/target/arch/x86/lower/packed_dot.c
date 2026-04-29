// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// x86 packed-dot source-to-low lowering.

#include <stdint.h>

#include "loom/ops/vector/ops.h"
#include "loom/target/arch/x86/lower/internal.h"
#include "loom/target/arch/x86/packed_dot_contract.h"
#include "loom/target/arch/x86/packed_dot_descriptors.h"
#include "loom/target/arch/x86/packed_dot_vector.h"

#define LOOM_X86_CONTRACT_SET_AVX512_CORE IREE_SV("x86.avx512.core")
#define LOOM_X86_CONTRACT_SET_PACKED_DOT_CORE IREE_SV("x86.packed_dot.core")
#define LOOM_X86_CONTRACT_SET_AVX512_PACKED_DOT_CORE \
  IREE_SV("x86.avx512_packed_dot.core")

static bool loom_x86_contract_set_key_is_avx512_core(
    iree_string_view_t contract_set_key) {
  return iree_string_view_equal(contract_set_key,
                                LOOM_X86_CONTRACT_SET_AVX512_CORE);
}

static bool loom_x86_contract_set_key_is_packed_dot_core(
    iree_string_view_t contract_set_key) {
  return iree_string_view_equal(contract_set_key,
                                LOOM_X86_CONTRACT_SET_PACKED_DOT_CORE);
}

static bool loom_x86_contract_set_key_is_avx512_packed_dot_core(
    iree_string_view_t contract_set_key) {
  return iree_string_view_equal(contract_set_key,
                                LOOM_X86_CONTRACT_SET_AVX512_PACKED_DOT_CORE);
}

static bool loom_x86_contract_set_key_is_x86(
    iree_string_view_t contract_set_key) {
  return loom_x86_contract_set_key_is_avx512_core(contract_set_key) ||
         loom_x86_contract_set_key_is_packed_dot_core(contract_set_key) ||
         loom_x86_contract_set_key_is_avx512_packed_dot_core(contract_set_key);
}

static bool loom_x86_scalar_type_has_packed_dot_register_width(
    loom_scalar_type_t scalar_type, uint32_t* out_bit_width) {
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_I8:
      *out_bit_width = 8;
      return true;
    case LOOM_SCALAR_TYPE_I16:
    case LOOM_SCALAR_TYPE_F16:
    case LOOM_SCALAR_TYPE_BF16:
      *out_bit_width = 16;
      return true;
    case LOOM_SCALAR_TYPE_I32:
    case LOOM_SCALAR_TYPE_F32:
      *out_bit_width = 32;
      return true;
    default:
      *out_bit_width = 0;
      return false;
  }
}

bool loom_x86_packed_dot_type_static_vector_bit_width(loom_type_t type,
                                                      uint32_t* out_bit_width) {
  *out_bit_width = 0;
  if (!loom_type_is_vector(type) || loom_type_rank(type) != 1 ||
      !loom_type_is_all_static(type)) {
    return false;
  }
  uint32_t element_bit_width = 0;
  if (!loom_x86_scalar_type_has_packed_dot_register_width(
          loom_type_element_type(type), &element_bit_width)) {
    return false;
  }
  const int64_t lane_count = loom_type_dim_static_size_at(type, 0);
  if (lane_count <= 0 ||
      (uint64_t)lane_count > UINT32_MAX / element_bit_width) {
    return false;
  }
  *out_bit_width = (uint32_t)lane_count * element_bit_width;
  return true;
}

iree_status_t loom_x86_map_packed_dot_type(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_type_t source_type,
                                           loom_type_t* out_low_type) {
  (void)user_data;
  uint32_t vector_bit_width = 0;
  if (loom_x86_packed_dot_type_static_vector_bit_width(source_type,
                                                       &vector_bit_width)) {
    switch (vector_bit_width) {
      case 128:
        return loom_low_lower_make_register_type(
            context, X86_PACKED_DOT_CORE_REG_CLASS_ID_X86_XMM, 1, out_low_type);
      case 256:
        return loom_low_lower_make_register_type(
            context, X86_PACKED_DOT_CORE_REG_CLASS_ID_X86_YMM, 1, out_low_type);
      case 512:
        return loom_low_lower_make_register_type(
            context, X86_PACKED_DOT_CORE_REG_CLASS_ID_X86_ZMM, 1, out_low_type);
      default:
        break;
    }
  }
  return loom_low_lower_emit_reject(
      context, source_op, IREE_SV("type"), IREE_SV("source"),
      IREE_SV("x86 packed-dot lowering requires a static 128-, 256-, or "
              "512-bit i8/i16/f16/bf16/i32/f32 vector"));
}

static iree_string_view_t loom_x86_packed_dot_rejection_name(
    loom_x86_packed_dot_rejection_bits_t rejection_bits) {
  if (iree_any_bit_set(rejection_bits,
                       LOOM_X86_PACKED_DOT_REJECTION_FEATURES)) {
    return IREE_SV("features");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_PAYLOAD)) {
    return IREE_SV("payload");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_SHAPE)) {
    return IREE_SV("shape");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_FAMILY)) {
    return IREE_SV("family");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_FLAGS)) {
    return IREE_SV("flags");
  }
  return IREE_SV("invalid-request");
}

static iree_string_view_t loom_x86_packed_dot_rejection_detail(
    loom_x86_packed_dot_rejection_bits_t rejection_bits) {
  if (iree_any_bit_set(rejection_bits,
                       LOOM_X86_PACKED_DOT_REJECTION_FEATURES)) {
    return IREE_SV(
        "target profile does not enable a matching x86 packed-dot feature");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_PAYLOAD)) {
    return IREE_SV(
        "no x86 packed-dot descriptor matches the vector dot payload types");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_SHAPE)) {
    return IREE_SV("no x86 packed-dot descriptor matches the vector dot shape");
  }
  return IREE_SV("no x86 packed-dot descriptor matches this vector dot op");
}

static iree_status_t loom_x86_select_packed_dot_descriptor(
    const loom_module_t* module, const loom_target_bundle_t* bundle,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* op,
    loom_x86_packed_dot_match_diagnostic_t* out_diagnostic,
    const loom_x86_packed_dot_descriptor_t** out_descriptor,
    const loom_low_descriptor_t** out_low_descriptor) {
  if (out_descriptor != NULL) {
    *out_descriptor = NULL;
  }
  if (out_low_descriptor != NULL) {
    *out_low_descriptor = NULL;
  }
  loom_x86_packed_dot_match_request_t request = {0};
  if (!loom_x86_packed_dot_match_request_from_vector_op(module, op, &request)) {
    if (out_diagnostic != NULL) {
      *out_diagnostic = (loom_x86_packed_dot_match_diagnostic_t){
          .rejection_bits = LOOM_X86_PACKED_DOT_REJECTION_INVALID_REQUEST,
      };
    }
    return iree_ok_status();
  }
  if (bundle == NULL || bundle->config == NULL || descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "x86 packed-dot selection requires a target bundle "
                            "and descriptor set");
  }
  request.feature_bits =
      (loom_x86_packed_dot_feature_bits_t)bundle->config->contract_feature_bits;

  loom_x86_packed_dot_match_diagnostic_t diagnostic = {0};
  const loom_x86_packed_dot_descriptor_t* descriptor =
      loom_x86_packed_dot_select(&request, &diagnostic);
  if (out_diagnostic != NULL) {
    *out_diagnostic = diagnostic;
  }
  if (descriptor == NULL) {
    return iree_ok_status();
  }

  uint32_t descriptor_ordinal = loom_low_descriptor_set_lookup_descriptor_by_id(
      descriptor_set, descriptor->stable_id);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "x86 packed-dot descriptor '%.*s' is missing",
                            (int)descriptor->name.size, descriptor->name.data);
  }
  if (out_descriptor != NULL) {
    *out_descriptor = descriptor;
  }
  if (out_low_descriptor != NULL) {
    *out_low_descriptor = loom_low_descriptor_set_descriptor_at(
        descriptor_set, descriptor_ordinal);
    IREE_ASSERT(*out_low_descriptor != NULL);
  }
  return iree_ok_status();
}

bool loom_x86_op_is_vector_dot(loom_op_kind_t kind) {
  switch (kind) {
    case LOOM_OP_VECTOR_DOT2F:
    case LOOM_OP_VECTOR_DOT4F8:
    case LOOM_OP_VECTOR_DOT4I:
    case LOOM_OP_VECTOR_DOT8I4:
      return true;
    default:
      return false;
  }
}

typedef struct loom_x86_packed_dot_plan_t {
  // Descriptor row selected for the packed-dot instruction.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Left-hand payload vector value.
  loom_value_id_t lhs;
  // Right-hand payload vector value.
  loom_value_id_t rhs;
  // Accumulator vector value.
  loom_value_id_t accumulator;
  // Result vector value.
  loom_value_id_t result;
} loom_x86_packed_dot_plan_t;

static bool loom_x86_init_packed_dot_plan(
    const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_x86_packed_dot_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(descriptor);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (loom_x86_packed_dot_plan_t){0};
  switch (source_op->kind) {
    case LOOM_OP_VECTOR_DOT2F:
      *out_plan = (loom_x86_packed_dot_plan_t){
          .descriptor = *descriptor,
          .lhs = loom_vector_dot2f_lhs(source_op),
          .rhs = loom_vector_dot2f_rhs(source_op),
          .accumulator = loom_vector_dot2f_acc(source_op),
          .result = loom_vector_dot2f_result(source_op),
      };
      return true;
    case LOOM_OP_VECTOR_DOT4I:
      *out_plan = (loom_x86_packed_dot_plan_t){
          .descriptor = *descriptor,
          .lhs = loom_vector_dot4i_lhs(source_op),
          .rhs = loom_vector_dot4i_rhs(source_op),
          .accumulator = loom_vector_dot4i_acc(source_op),
          .result = loom_vector_dot4i_result(source_op),
      };
      return true;
    default:
      return false;
  }
}

iree_status_t loom_x86_select_packed_dot_op(void* user_data,
                                            loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            loom_low_lower_plan_t* out_plan) {
  (void)user_data;
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = loom_low_lower_plan_empty();
  switch (source_op->kind) {
    case LOOM_OP_VECTOR_DOT2F:
    case LOOM_OP_VECTOR_DOT4I: {
      const loom_low_descriptor_t* descriptor = NULL;
      IREE_RETURN_IF_ERROR(loom_x86_select_packed_dot_descriptor(
          loom_low_lower_context_module(context),
          loom_low_lower_context_bundle(context),
          loom_low_lower_context_descriptor_set(context), source_op,
          /*out_diagnostic=*/NULL, /*out_descriptor=*/NULL, &descriptor));
      if (descriptor != NULL) {
        loom_low_lower_resolved_descriptor_t resolved_descriptor = {0};
        IREE_RETURN_IF_ERROR(loom_low_lower_resolve_descriptor_row(
            context, descriptor, &resolved_descriptor));
        loom_x86_packed_dot_plan_t* plan_data = NULL;
        IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
            context, sizeof(*plan_data), (void**)&plan_data));
        if (loom_x86_init_packed_dot_plan(source_op, &resolved_descriptor,
                                          plan_data)) {
          *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);
        }
      }
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_x86_lower_tied_ternary_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t source_lhs, loom_value_id_t source_rhs,
    loom_value_id_t source_accumulator, loom_value_id_t source_result) {
  IREE_ASSERT_ARGUMENT(descriptor);
  loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_lhs, &low_lhs));
  loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_rhs, &low_rhs));
  loom_value_id_t low_accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, source_accumulator,
                                                   &low_accumulator));

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(context, source_op,
                                                source_result, &result_type));
  IREE_ASSERT(loom_type_is_register(result_type));

  loom_op_t* accumulator_copy_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_copy_build(
      loom_low_lower_context_builder(context), low_accumulator, result_type,
      source_op->location, &accumulator_copy_op));
  const loom_value_id_t copied_accumulator =
      loom_low_copy_result(accumulator_copy_op);

  loom_value_id_t low_operands[3] = {copied_accumulator, low_lhs, low_rhs};
  loom_tied_result_t tied_result = {
      .result_index = 0,
      .operand_index = 0,
      .has_type_change = false,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, low_operands, IREE_ARRAYSIZE(low_operands),
      loom_make_named_attr_slice(NULL, 0), &result_type, 1, &tied_result, 1,
      source_op->location, &low_op));
  return loom_low_lower_bind_value(
      context, source_result,
      loom_value_slice_get(loom_low_op_results(low_op), 0));
}

static iree_status_t loom_x86_lower_packed_dot_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_x86_packed_dot_plan_t* plan) {
  IREE_ASSERT_ARGUMENT(plan);
  return loom_x86_lower_tied_ternary_op(context, source_op, &plan->descriptor,
                                        plan->lhs, plan->rhs, plan->accumulator,
                                        plan->result);
}

iree_status_t loom_x86_emit_packed_dot_op(void* user_data,
                                          loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_low_lower_plan_t plan) {
  (void)user_data;
  switch (plan.id) {
    case LOOM_OP_VECTOR_DOT2F:
    case LOOM_OP_VECTOR_DOT4I:
      return loom_x86_lower_packed_dot_op(
          context, source_op,
          (const loom_x86_packed_dot_plan_t*)plan.target_data);
    default:
      IREE_CHECK_UNREACHABLE();
  }
}

static iree_string_view_t loom_x86_legality_contract_set_key(
    const loom_target_low_legality_context_t* context) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (bundle == NULL || bundle->config == NULL) {
    return iree_string_view_empty();
  }
  return bundle->config->contract_set_key;
}

iree_status_t loom_x86_low_legality_verify_packed_dot(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const iree_string_view_t contract_set_key =
      loom_x86_legality_contract_set_key(context);
  if (!loom_x86_contract_set_key_is_x86(contract_set_key)) {
    return iree_ok_status();
  }
  *out_handled = true;

  if (!loom_x86_contract_set_key_is_packed_dot_core(contract_set_key) &&
      !loom_x86_contract_set_key_is_avx512_packed_dot_core(contract_set_key)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("contract"), contract_set_key,
        IREE_SV("x86 vector dot ops require an x86 packed-dot target-low "
                "contract set"));
  }

  loom_x86_packed_dot_match_diagnostic_t diagnostic = {0};
  const loom_x86_packed_dot_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_x86_select_packed_dot_descriptor(
      loom_target_low_legality_module(context),
      loom_target_low_legality_bundle(context),
      loom_target_low_legality_descriptor_set(context), op, &diagnostic,
      &descriptor, /*out_low_descriptor=*/NULL));
  if (descriptor == NULL) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("contract"),
        loom_x86_packed_dot_rejection_name(diagnostic.rejection_bits),
        loom_x86_packed_dot_rejection_detail(diagnostic.rejection_bits));
  }

  return loom_target_low_legality_record_contract(
      context, provider, op, descriptor->name, IREE_SV("selected"),
      IREE_SV("selected x86 packed-dot descriptor"));
}
