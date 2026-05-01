// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/contract_query.h"

#include "iree/base/internal/arena.h"

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

static iree_status_t loom_low_lower_query_custom_family(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_contract_query_options_t* options,
    const loom_target_contract_case_t* contract_case, uint16_t case_index,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  const loom_low_lower_contract_family_t* family =
      &options->contract_families[contract_case->row_index];
  IREE_RETURN_IF_ERROR(
      family->query(family->user_data, environment, source_op, out_result));
  if (out_result->outcome != LOOM_TARGET_CONTRACT_QUERY_UNHANDLED) {
    loom_low_lower_contract_query_adopt_case(contract_case, case_index,
                                             out_result);
  }
  return iree_ok_status();
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
    if (contract_case->system == LOOM_TARGET_CONTRACT_SYSTEM_CUSTOM_FAMILY) {
      IREE_RETURN_IF_ERROR(loom_low_lower_query_custom_family(
          environment, options, contract_case, case_index, source_op,
          out_result));
      if (out_result->outcome != LOOM_TARGET_CONTRACT_QUERY_UNHANDLED) {
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
       options->contract_family_count == 0)) {
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
