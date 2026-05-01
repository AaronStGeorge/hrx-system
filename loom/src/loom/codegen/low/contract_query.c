// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/contract_query.h"

#include "iree/base/internal/arena.h"
#include "loom/ops/vector/ops.h"

static iree_status_t loom_low_lower_contract_query_make_rejection(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_diagnostic_t* diagnostic,
    const loom_target_contract_rejection_t** out_rejection) {
  *out_rejection = NULL;
  if (diagnostic == NULL) {
    return iree_ok_status();
  }
  loom_target_contract_rejection_t* rejection = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      environment->arena, sizeof(*rejection), (void**)&rejection));
  *rejection = (loom_target_contract_rejection_t){
      .subject_kind = diagnostic->subject_kind,
      .subject_name = diagnostic->subject_name,
      .reason = diagnostic->reason,
  };
  *out_rejection = rejection;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_contract_query_make_direct_rejection(
    const loom_target_contract_query_environment_t* environment,
    iree_string_view_t subject_kind, iree_string_view_t subject_name,
    iree_string_view_t reason,
    const loom_target_contract_rejection_t** out_rejection) {
  *out_rejection = NULL;
  loom_target_contract_rejection_t* rejection = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      environment->arena, sizeof(*rejection), (void**)&rejection));
  *rejection = (loom_target_contract_rejection_t){
      .subject_kind = subject_kind,
      .subject_name = subject_name,
      .reason = reason,
  };
  *out_rejection = rejection;
  return iree_ok_status();
}

static void loom_low_lower_contract_query_adopt_case(
    const loom_target_contract_case_t* contract_case, uint16_t case_index,
    loom_target_contract_query_result_t* result) {
  if (result->binding_index == UINT16_MAX) {
    result->binding_index = contract_case->binding_index;
  }
  if (result->case_index == UINT16_MAX) {
    result->case_index = case_index;
  }
  if (result->rule_index == UINT16_MAX) {
    result->rule_index = contract_case->row_index;
  }
}

static iree_string_view_t loom_low_lower_descriptor_matrix_source_name(
    loom_target_contract_descriptor_matrix_source_t source) {
  switch (source) {
    case LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_SOURCE_VECTOR_MMA:
      return IREE_SV("vector.mma");
    case LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_SOURCE_NONE:
    default:
      return IREE_SV("<unknown>");
  }
}

static iree_string_view_t loom_low_lower_descriptor_matrix_rejection_reason(
    loom_target_contract_descriptor_matrix_source_t source,
    loom_contract_diagnostic_t diagnostic) {
  const loom_contract_rejection_bits_t bits = diagnostic.rejection_bits;
  if (source == LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_SOURCE_VECTOR_MMA) {
    if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_ROLE)) {
      return IREE_SV(
          "vector.mma operands must carry lhs, rhs, and init fragment facts");
    }
    if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_SHAPE)) {
      return IREE_SV(
          "vector.mma fragment shapes must resolve to exact compatible M/N/K "
          "values");
    }
    if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_SCHEMA)) {
      return IREE_SV(
          "encoded vector.mma fragments need a resolved target fragment "
          "schema");
    }
    if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_NUMERIC)) {
      return IREE_SV("vector.mma operands use unsupported numeric types");
    }
    if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_FRAGMENT)) {
      return IREE_SV(
          "vector.mma lowering requires compatible fragment ownership facts");
    }
    if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_CAPABILITY)) {
      return IREE_SV(
          "vector.mma requires matrix capabilities unavailable in the target "
          "projection");
    }
    if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_INVALID_REQUEST)) {
      return IREE_SV("vector.mma did not form a valid matrix contract request");
    }
  }
  return IREE_SV("source op did not form a valid matrix contract request");
}

static iree_status_t loom_low_lower_descriptor_matrix_make_unsupported(
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_contract_diagnostic_t* diagnostic,
    loom_target_contract_query_result_t* out_result) {
  const loom_target_contract_rejection_t* rejection = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_contract_query_make_direct_rejection(
      environment, IREE_SV("op"),
      loom_low_lower_descriptor_matrix_source_name(rule->source),
      loom_low_lower_descriptor_matrix_rejection_reason(rule->source,
                                                        *diagnostic),
      &rejection));
  *out_result = loom_target_contract_query_result_empty();
  out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED;
  out_result->source_rejection_bits = diagnostic->rejection_bits;
  out_result->rejection = rejection;
  return iree_ok_status();
}

iree_status_t loom_low_lower_query_descriptor_matrix_contract(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_descriptor_matrix_t* descriptor_matrix,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  *out_result = loom_target_contract_query_result_empty();
  if (descriptor_matrix->options == NULL || descriptor_matrix->query == NULL) {
    return iree_ok_status();
  }

  loom_contract_request_t request = {0};
  loom_contract_diagnostic_t diagnostic = {0};
  switch (rule->source) {
    case LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_SOURCE_VECTOR_MMA: {
      if (!loom_vector_mma_isa(source_op)) {
        return iree_ok_status();
      }
      loom_contract_vector_mma_options_t options = {0};
      IREE_RETURN_IF_ERROR(descriptor_matrix->options(
          descriptor_matrix->user_data, environment, rule, &options));
      if (!loom_contract_request_from_vector_mma_op(
              environment->module, environment->fact_table, source_op, &options,
              &request, &diagnostic)) {
        return loom_low_lower_descriptor_matrix_make_unsupported(
            environment, rule, &diagnostic, out_result);
      }
      break;
    }
    case LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_SOURCE_NONE:
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "unknown descriptor-matrix source");
  }

  return descriptor_matrix->query(descriptor_matrix->user_data, environment,
                                  rule, &request, out_result);
}

static iree_status_t loom_low_lower_query_target_contract_index(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_contract_query_options_t* options,
    const loom_op_t* source_op,
    const loom_low_lower_rule_match_context_t* match_context,
    loom_target_contract_query_result_t* out_result) {
  const loom_target_contract_index_t* index = options->contract_index;
  const loom_target_contract_op_entry_t op_entry =
      loom_target_contract_index_lookup_kind(index, source_op->kind);
  if (loom_target_contract_op_entry_is_empty(op_entry)) {
    return iree_ok_status();
  }

  const loom_low_lower_rule_set_t* failed_rule_set = NULL;
  loom_low_lower_rule_selection_t failed_selection = {0};
  uint16_t failed_binding_index = UINT16_MAX;
  uint16_t failed_case_index = UINT16_MAX;
  uint16_t failed_rule_set_index = UINT16_MAX;
  for (uint16_t i = 0; i < op_entry.case_count; ++i) {
    const uint16_t case_index = (uint16_t)(op_entry.case_start + i);
    const loom_target_contract_case_t* contract_case =
        &index->cases[case_index];
    const loom_target_contract_binding_t* binding =
        &index->bindings[contract_case->binding_index];
    if (!loom_target_contract_fragment_queries_target(binding->fragment)) {
      continue;
    }
    if (contract_case->system ==
        LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_MATRIX) {
      const loom_target_contract_descriptor_matrix_rule_t* matrix_rule =
          &binding->fragment->descriptor_matrices[contract_case->row_index];
      IREE_RETURN_IF_ERROR(loom_low_lower_query_descriptor_matrix_contract(
          environment, &options->descriptor_matrix, matrix_rule, source_op,
          out_result));
      if (out_result->outcome != LOOM_TARGET_CONTRACT_QUERY_UNHANDLED) {
        loom_low_lower_contract_query_adopt_case(contract_case, case_index,
                                                 out_result);
        return iree_ok_status();
      }
      continue;
    }
    uint16_t rule_index = UINT16_MAX;
    if (!loom_low_lower_contract_case_lower_rule_index(index, contract_case,
                                                       &rule_index)) {
      continue;
    }
    const loom_low_lower_rule_set_t* rule_set =
        options->rule_sets.values[binding->rule_set_index];
    loom_low_lower_rule_selection_t selection = {0};
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_set_select_rule_range_with_match_context(
            match_context, rule_set, source_op, rule_index, 1, &selection));
    if (selection.rule != NULL) {
      *out_result = (loom_target_contract_query_result_t){
          .outcome = LOOM_TARGET_CONTRACT_QUERY_LEGAL,
          .binding_index = contract_case->binding_index,
          .case_index = case_index,
          .rule_set_index = binding->rule_set_index,
          .rule_index = selection.rule_index,
          .diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE,
          .matched_guard_count = selection.rule->guard_count,
          .selected_descriptor_id =
              loom_low_lower_rule_first_descriptor_id(rule_set, selection.rule),
          .source_rejection_bits = 0,
          .target_rejection_bits = 0,
          .missing_feature_bits = 0,
          .missing_fact_bits = 0,
          .rejection = NULL,
      };
      return iree_ok_status();
    }
    if (selection.has_source_op_span &&
        (failed_rule_set == NULL || selection.matched_guard_count >
                                        failed_selection.matched_guard_count)) {
      failed_rule_set = rule_set;
      failed_selection = selection;
      failed_binding_index = contract_case->binding_index;
      failed_case_index = case_index;
      failed_rule_set_index = binding->rule_set_index;
    }
  }

  if (failed_rule_set == NULL) {
    return iree_ok_status();
  }

  const loom_low_lower_diagnostic_t* diagnostic =
      loom_low_lower_rule_set_selection_diagnostic(failed_rule_set,
                                                   failed_selection);
  const loom_target_contract_rejection_t* rejection = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_contract_query_make_rejection(
      environment, diagnostic, &rejection));
  *out_result = (loom_target_contract_query_result_t){
      .outcome = LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED,
      .binding_index = failed_binding_index,
      .case_index = failed_case_index,
      .rule_set_index = failed_rule_set_index,
      .rule_index = UINT16_MAX,
      .diagnostic_index = failed_selection.diagnostic_index,
      .matched_guard_count = failed_selection.matched_guard_count,
      .selected_descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE,
      .source_rejection_bits = 0,
      .target_rejection_bits = 0,
      .missing_feature_bits = 0,
      .missing_fact_bits = 0,
      .rejection = rejection,
  };
  return iree_ok_status();
}

iree_status_t loom_low_lower_query_target_contract(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_contract_query_options_t* options,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  *out_result = loom_target_contract_query_result_empty();

  if (options->contract_index == NULL ||
      options->contract_index->case_count == 0 ||
      (loom_low_lower_rule_set_list_is_empty(options->rule_sets) &&
       options->descriptor_matrix.query == NULL)) {
    return iree_ok_status();
  }

  const loom_low_lower_rule_match_context_t match_context = {
      .module = environment->module,
      .descriptor_set = environment->descriptor_set,
      .feature_bits = environment->bundle->config->contract_feature_bits,
      .map_value = options->map_value,
      .register_class = options->register_class,
      .can_materialize = options->can_materialize,
      .fact_table = environment->fact_table,
  };

  return loom_low_lower_query_target_contract_index(
      environment, options, source_op, &match_context, out_result);
}
