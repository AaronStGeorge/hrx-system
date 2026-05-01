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
  IREE_ASSERT_ARGUMENT(out_rejection);
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

static iree_status_t loom_low_lower_query_target_contract_table(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_contract_query_options_t* options,
    const loom_op_t* source_op,
    const loom_low_lower_rule_match_context_t* match_context,
    loom_target_contract_query_result_t* out_result) {
  const loom_target_contract_table_t* table = options->contract_table;
  const loom_target_contract_op_entry_t op_entry =
      loom_target_contract_table_lookup_kind(table, source_op->kind);
  if (loom_target_contract_op_entry_is_empty(op_entry)) {
    return iree_ok_status();
  }

  const loom_low_lower_rule_set_t* failed_rule_set = NULL;
  loom_low_lower_rule_selection_t failed_selection = {0};
  uint16_t failed_case_index = UINT16_MAX;
  for (uint16_t i = 0; i < op_entry.case_count; ++i) {
    const uint16_t case_index = (uint16_t)(op_entry.case_start + i);
    const loom_target_contract_case_t* contract_case =
        &table->cases[case_index];
    if (contract_case->system != LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE) {
      continue;
    }
    const loom_target_contract_descriptor_rule_t* descriptor_rule =
        &table->descriptor_rules[contract_case->row_index];
    const loom_low_lower_rule_set_t* rule_set =
        options->rule_sets.values[descriptor_rule->rule_set_index];
    loom_low_lower_rule_selection_t selection = {0};
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_set_select_rule_range_with_match_context(
            match_context, rule_set, source_op, descriptor_rule->rule_index, 1,
            &selection));
    if (selection.rule != NULL) {
      *out_result = (loom_target_contract_query_result_t){
          .outcome = LOOM_TARGET_CONTRACT_QUERY_LEGAL,
          .table_index = table->table_index,
          .rule_index = case_index,
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
      failed_case_index = case_index;
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
      .table_index = table->table_index,
      .rule_index = failed_case_index,
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
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(source_op);
  IREE_ASSERT_ARGUMENT(out_result);
  *out_result = loom_target_contract_query_result_empty();

  if (loom_low_lower_rule_set_list_is_empty(options->rule_sets)) {
    return iree_ok_status();
  }

  const loom_low_lower_rule_match_context_t match_context = {
      .module = environment->module,
      .descriptor_set = environment->descriptor_set,
      .map_value = options->map_value,
      .register_class = options->register_class,
      .can_materialize = options->can_materialize,
      .fact_table = environment->fact_table,
  };

  if (options->contract_table != NULL) {
    IREE_RETURN_IF_ERROR(loom_low_lower_query_target_contract_table(
        environment, options, source_op, &match_context, out_result));
    if (out_result->outcome != LOOM_TARGET_CONTRACT_QUERY_UNHANDLED) {
      return iree_ok_status();
    }
  }

  const loom_low_lower_rule_set_t* failed_rule_set = NULL;
  loom_low_lower_rule_selection_t failed_selection = {0};
  uint16_t failed_rule_set_index = UINT16_MAX;
  for (uint16_t i = 0; i < options->rule_sets.count; ++i) {
    const loom_low_lower_rule_set_t* rule_set = options->rule_sets.values[i];
    if (!iree_any_bit_set(rule_set->flags,
                          LOOM_LOW_LOWER_RULE_SET_FLAG_TARGET_CONTRACT_QUERY)) {
      continue;
    }
    loom_low_lower_rule_selection_t selection = {0};
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_set_select_with_match_context(
        &match_context, rule_set, source_op, &selection));
    if (selection.rule != NULL) {
      *out_result = (loom_target_contract_query_result_t){
          .outcome = LOOM_TARGET_CONTRACT_QUERY_LEGAL,
          .table_index = i,
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
      break;
    }
    if (selection.has_source_op_span &&
        (failed_rule_set == NULL || selection.matched_guard_count >
                                        failed_selection.matched_guard_count)) {
      failed_rule_set = rule_set;
      failed_selection = selection;
      failed_rule_set_index = i;
    }
  }

  if (out_result->outcome == LOOM_TARGET_CONTRACT_QUERY_UNHANDLED &&
      failed_rule_set != NULL) {
    const loom_low_lower_diagnostic_t* diagnostic =
        loom_low_lower_rule_set_selection_diagnostic(failed_rule_set,
                                                     failed_selection);
    const loom_target_contract_rejection_t* rejection = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_contract_query_make_rejection(
        environment, diagnostic, &rejection));
    *out_result = (loom_target_contract_query_result_t){
        .outcome = LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED,
        .table_index = failed_rule_set_index,
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
  }

  return iree_ok_status();
}
