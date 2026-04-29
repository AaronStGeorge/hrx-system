// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/contract_storage.h"

#include "loom/ops/encoding/storage.h"

static bool loom_contract_storage_fail(
    loom_contract_rejection_bits_t rejection_bits,
    loom_contract_diagnostic_t* out_diagnostic) {
  if (out_diagnostic) {
    out_diagnostic->rejection_bits = rejection_bits;
  }
  return false;
}

bool loom_contract_numeric_type_from_encoded_format(
    loom_value_fact_numeric_format_flags_t format,
    loom_contract_numeric_type_t* out_numeric_type) {
  IREE_ASSERT_ARGUMENT(out_numeric_type);
  *out_numeric_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
  switch (format) {
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_FP8;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_BF8:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_BF8;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F6_E3M2:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F6_E2M3:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_FP6;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_BF6:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_BF6;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F4_E2M1:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_FP4;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_I4:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_I4;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_U4:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_U4;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_I8:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I8:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_I8;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_U8:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_U8;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_I16:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_I16;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_U16:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_U16;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_I32:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_I32;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_U32:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_U32;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F16:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_F16;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_BF16:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_BF16;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F32:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_F32;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F64:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_F64;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_UNKNOWN:
    default:
      return false;
  }
}

bool loom_contract_scale_kind_from_storage_schema(
    loom_value_fact_storage_schema_t schema,
    loom_contract_scale_kind_t* out_scale_kind) {
  IREE_ASSERT_ARGUMENT(out_scale_kind);
  *out_scale_kind = LOOM_CONTRACT_SCALE_UNKNOWN;
  loom_value_fact_encoded_operand_schema_t operand = schema.encoded_operand;
  if (!loom_value_fact_encoded_operand_schema_has_scale(operand)) {
    *out_scale_kind = LOOM_CONTRACT_SCALE_NONE;
    return true;
  }
  if (operand.scale_topology == 0 || operand.scale_operand_count == 0) {
    return false;
  }
  switch (operand.scale_group_element_count) {
    case 32:
      *out_scale_kind = LOOM_CONTRACT_SCALE_32;
      return true;
    case 16:
      *out_scale_kind = LOOM_CONTRACT_SCALE_16;
      return true;
    default:
      return false;
  }
}

loom_contract_capability_flags_t
loom_contract_capability_flags_from_storage_schema(
    loom_value_fact_storage_schema_t schema) {
  loom_contract_capability_flags_t flags = 0;
  loom_contract_numeric_type_t numeric_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
  loom_value_fact_encoded_operand_schema_t operand = schema.encoded_operand;
  if (loom_contract_numeric_type_from_encoded_format(operand.element_format,
                                                     &numeric_type)) {
    flags |= LOOM_CONTRACT_CAPABILITY_FORMAT_SELECTORS;
  }
  if (loom_value_fact_encoded_operand_schema_has_scale(operand) &&
      operand.scale_operand_count != 0) {
    flags |= LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS;
  }
  if (operand.scale_format != 0 || operand.secondary_scale_format != 0) {
    flags |= LOOM_CONTRACT_CAPABILITY_SCALE_FORMAT_SELECTORS;
  }
  if (iree_any_bit_set(
          operand.flags,
          LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_ZERO_SCALE_FALLBACK)) {
    flags |= LOOM_CONTRACT_CAPABILITY_ZERO_SCALE_FALLBACK;
  }
  if (iree_any_bit_set(operand.sparsity_policy,
                       LOOM_VALUE_FACT_SPARSITY_POLICY_ALL)) {
    flags |= LOOM_CONTRACT_CAPABILITY_SPARSE_METADATA;
  }
  return flags;
}

bool loom_contract_operand_from_storage_schema(
    loom_contract_operand_role_t role, loom_value_fact_storage_schema_t schema,
    loom_contract_operand_t* out_operand) {
  IREE_ASSERT_ARGUMENT(out_operand);
  *out_operand = (loom_contract_operand_t){0};
  loom_value_fact_encoded_operand_schema_t operand = schema.encoded_operand;
  if (!iree_any_bit_set(operand.payload_packing,
                        LOOM_VALUE_FACT_PAYLOAD_PACKING_TARGET_FRAGMENT) ||
      operand.payload_register_count == 0 ||
      operand.payload_element_count == 0) {
    return false;
  }
  loom_contract_numeric_type_t numeric_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
  if (!loom_contract_numeric_type_from_encoded_format(operand.element_format,
                                                      &numeric_type)) {
    return false;
  }
  *out_operand = (loom_contract_operand_t){
      .role = role,
      .numeric_type = numeric_type,
      .payload_register_count = operand.payload_register_count,
      .payload_element_count = operand.payload_element_count,
  };
  return true;
}

bool loom_contract_view_payload_from_type(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_type_t view_type, loom_contract_operand_role_t role,
    bool plain_integer_is_unsigned, loom_contract_view_payload_t* out_payload) {
  IREE_ASSERT_ARGUMENT(out_payload);
  *out_payload = (loom_contract_view_payload_t){
      .kind = LOOM_CONTRACT_VIEW_PAYLOAD_UNKNOWN,
      .operand =
          (loom_contract_operand_t){
              .role = role,
              .numeric_type = LOOM_CONTRACT_NUMERIC_UNKNOWN,
          },
      .scale_kind = LOOM_CONTRACT_SCALE_NONE,
  };
  if (!loom_type_is_view(view_type)) {
    return false;
  }

  loom_value_fact_storage_schema_t storage_schema = {0};
  if (loom_encoding_query_type_storage_schema(context, module, view_type,
                                              &storage_schema)) {
    out_payload->kind = LOOM_CONTRACT_VIEW_PAYLOAD_UNSUPPORTED_STORAGE_SCHEMA;
    out_payload->storage_schema = storage_schema;
    if (!loom_contract_operand_from_storage_schema(role, storage_schema,
                                                   &out_payload->operand)) {
      return true;
    }
    if (!loom_contract_scale_kind_from_storage_schema(
            storage_schema, &out_payload->scale_kind)) {
      out_payload->operand.numeric_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
      return true;
    }
    out_payload->kind = LOOM_CONTRACT_VIEW_PAYLOAD_ENCODED_OPERAND_SCHEMA;
    out_payload->available_capability_flags =
        loom_contract_capability_flags_from_storage_schema(storage_schema);
    return true;
  }

  loom_contract_numeric_type_t numeric_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
  if (!loom_contract_numeric_type_from_scalar(loom_type_element_type(view_type),
                                              plain_integer_is_unsigned,
                                              &numeric_type)) {
    return false;
  }
  out_payload->kind = LOOM_CONTRACT_VIEW_PAYLOAD_PLAIN_ELEMENT;
  out_payload->operand.numeric_type = numeric_type;
  return true;
}

static bool loom_contract_payload_is_supported(
    const loom_contract_view_payload_t* payload) {
  return payload->kind == LOOM_CONTRACT_VIEW_PAYLOAD_PLAIN_ELEMENT ||
         payload->kind == LOOM_CONTRACT_VIEW_PAYLOAD_ENCODED_OPERAND_SCHEMA;
}

static bool loom_contract_payload_scale_kind_matches(
    const loom_contract_view_payload_t* lhs,
    const loom_contract_view_payload_t* rhs,
    loom_contract_scale_kind_t* out_scale_kind) {
  *out_scale_kind = lhs->scale_kind;
  return lhs->scale_kind == rhs->scale_kind;
}

bool loom_contract_request_from_matrix_payloads(
    const loom_contract_matrix_request_options_t* options,
    loom_contract_request_t* out_request,
    loom_contract_diagnostic_t* out_diagnostic) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_request);
  loom_contract_request_initialize(out_request);
  if (out_diagnostic) {
    *out_diagnostic = (loom_contract_diagnostic_t){0};
  }

  if (!loom_contract_payload_is_supported(&options->lhs) ||
      !loom_contract_payload_is_supported(&options->rhs)) {
    return loom_contract_storage_fail(LOOM_CONTRACT_REJECTION_NUMERIC,
                                      out_diagnostic);
  }

  loom_contract_scale_kind_t scale_kind = LOOM_CONTRACT_SCALE_UNKNOWN;
  if (!loom_contract_payload_scale_kind_matches(&options->lhs, &options->rhs,
                                                &scale_kind)) {
    return loom_contract_storage_fail(LOOM_CONTRACT_REJECTION_CAPABILITY,
                                      out_diagnostic);
  }

  out_request->kind = LOOM_CONTRACT_KIND_MATRIX_MULTIPLY;
  out_request->arithmetic = options->arithmetic;
  out_request->shape = options->shape;
  out_request->k_group_size = options->k_group_size;
  out_request->lhs = options->lhs.operand;
  out_request->lhs.role = LOOM_CONTRACT_OPERAND_ROLE_LHS;
  out_request->rhs = options->rhs.operand;
  out_request->rhs.role = LOOM_CONTRACT_OPERAND_ROLE_RHS;
  out_request->accumulator = (loom_contract_operand_t){
      .role = LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR,
      .numeric_type = options->accumulator_numeric_type,
  };
  out_request->result = (loom_contract_operand_t){
      .role = LOOM_CONTRACT_OPERAND_ROLE_RESULT,
      .numeric_type = options->result_numeric_type,
  };
  out_request->fragment = options->fragment;
  out_request->capability_class = options->capability_class;
  out_request->available_capability_flags =
      options->lhs.available_capability_flags |
      options->rhs.available_capability_flags;
  out_request->required_capability_flags = options->required_capability_flags;
  out_request->scale_kind = scale_kind;
  out_request->policy = options->policy;
  return loom_contract_request_validate(out_request, out_diagnostic);
}
