// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/passes/target_legalize.h"

#include <string.h>

#include "loom/analysis/contract.h"
#include "loom/codegen/low/pass_environment.h"
#include "loom/codegen/low/source_selection.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ops/op_defs.h"
#include "loom/pass/pipeline.h"
#include "loom/pass/registry.h"
#include "loom/pass/value_facts.h"
#include "loom/passes/scalar_target_legalization.h"
#include "loom/passes/scf_to_cfg.h"
#include "loom/passes/vector/target_legalization.h"
#include "loom/rewrite/greedy.h"
#include "loom/target/compile_report.h"
#include "loom/target/legalization.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/low_legality.h"

typedef struct loom_low_target_legalize_pass_state_t {
  // Maximum number of fixed-point rewrite iterations.
  uint32_t max_iterations;
  // Maximum number of final legality diagnostics emitted before stopping.
  uint32_t max_errors;
  // Current target legalization phase.
  loom_target_legalization_mode_t mode;
  // True when max_iterations was explicitly provided.
  bool has_max_iterations_option;
  // True when max_errors was explicitly provided.
  bool has_max_errors_option;
  // True when mode was explicitly provided.
  bool has_mode_option;
} loom_low_target_legalize_pass_state_t;

typedef struct loom_low_target_legalize_parse_context_t {
  // Mutable pass state being populated.
  loom_low_target_legalize_pass_state_t* state;
} loom_low_target_legalize_parse_context_t;

static const loom_pass_option_def_t kLowTargetLegalizeOptions[] = {
    {IREE_SVL("max-errors"),
     IREE_SVL("Maximum number of final legality diagnostics to emit; zero "
              "means no limit.")},
    {IREE_SVL("max-iterations"),
     IREE_SVL("Maximum number of target legalization worklist iterations.")},
    {IREE_SVL("mode"), IREE_SVL("Legalization phase: eager or final.")},
};

enum {
  LOOM_LOW_TARGET_LEGALIZE_STAT_OPS_REWRITTEN = 0,
  LOOM_LOW_TARGET_LEGALIZE_STAT_LOOPS_CREATED = 1,
  LOOM_LOW_TARGET_LEGALIZE_STAT_LANES_MATERIALIZED = 2,
  LOOM_LOW_TARGET_LEGALIZE_STAT_OPS_LEGAL = 3,
  LOOM_LOW_TARGET_LEGALIZE_STAT_OPS_DEFERRED = 4,
  LOOM_LOW_TARGET_LEGALIZE_STAT_FUNCTIONS = 5,
  LOOM_LOW_TARGET_LEGALIZE_STAT_ERRORS = 6,
};

static const loom_pass_statistic_def_t kLowTargetLegalizeStatistics[] = {
    {IREE_SVL("ops-rewritten"),
     IREE_SVL("Number of source ops rewritten by target legalizers.")},
    {IREE_SVL("loops-created"),
     IREE_SVL("Number of structured loops created by reference lowerings.")},
    {IREE_SVL("lanes-materialized"),
     IREE_SVL("Number of scalar lane programs materialized by reference "
              "lowerings.")},
    {IREE_SVL("ops-legal"),
     IREE_SVL("Number of legalizer-rooted ops already accepted by the target "
              "contract.")},
    {IREE_SVL("ops-deferred"),
     IREE_SVL("Number of legalizer-rooted ops left for a later phase.")},
    {IREE_SVL("functions"),
     IREE_SVL("Number of target-bound source functions visited.")},
    {IREE_SVL("errors"), IREE_SVL("Number of final legality errors emitted.")},
};

static const loom_pass_info_t loom_low_target_legalize_pass_info_storage = {
    .name = IREE_SVL("target-legalize"),
    .description =
        IREE_SVL("Rewrite unsupported target-bound source ops toward a legal "
                 "target-low contract."),
    .kind = LOOM_PASS_MODULE,
    .option_defs = kLowTargetLegalizeOptions,
    .option_count = IREE_ARRAYSIZE(kLowTargetLegalizeOptions),
    .statistic_defs = kLowTargetLegalizeStatistics,
    .statistic_count = IREE_ARRAYSIZE(kLowTargetLegalizeStatistics),
};

const loom_pass_info_t* loom_low_target_legalize_pass_info(void) {
  return &loom_low_target_legalize_pass_info_storage;
}

static iree_status_t loom_low_target_legalize_parse_max_errors(
    uint32_t max_errors, loom_low_target_legalize_parse_context_t* context) {
  if (context->state->has_max_errors_option) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "duplicate option 'max-errors' for pass 'target-legalize'");
  }
  context->state->max_errors = max_errors;
  context->state->has_max_errors_option = true;
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_parse_max_iterations(
    uint32_t max_iterations,
    loom_low_target_legalize_parse_context_t* context) {
  if (context->state->has_max_iterations_option) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "duplicate option 'max-iterations' for pass 'target-legalize'");
  }
  if (max_iterations == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass 'target-legalize' option "
                            "'max-iterations' must be greater than 0");
  }
  context->state->max_iterations = max_iterations;
  context->state->has_max_iterations_option = true;
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_parse_mode(
    iree_string_view_t value,
    loom_low_target_legalize_parse_context_t* context) {
  if (context->state->has_mode_option) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate option 'mode' for pass "
                            "'target-legalize'");
  }
  if (iree_string_view_equal(value, IREE_SV("eager"))) {
    context->state->mode = LOOM_TARGET_LEGALIZATION_MODE_EAGER;
  } else if (iree_string_view_equal(value, IREE_SV("final"))) {
    context->state->mode = LOOM_TARGET_LEGALIZATION_MODE_FINAL;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-legalize option 'mode' expected 'eager' "
                            "or 'final', got '%.*s'",
                            (int)value.size, value.data);
  }
  context->state->has_mode_option = true;
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_parse_option(
    void* user_data, iree_string_view_t name, iree_string_view_t value) {
  loom_low_target_legalize_parse_context_t* context =
      (loom_low_target_legalize_parse_context_t*)user_data;
  if (iree_string_view_equal(name, IREE_SV("max-errors"))) {
    uint32_t max_errors = 0;
    IREE_RETURN_IF_ERROR(loom_pass_option_parse_uint32(
        IREE_SV("target-legalize"), name, value, &max_errors));
    return loom_low_target_legalize_parse_max_errors(max_errors, context);
  }
  if (iree_string_view_equal(name, IREE_SV("max-iterations"))) {
    uint32_t max_iterations = 0;
    IREE_RETURN_IF_ERROR(loom_pass_option_parse_uint32(
        IREE_SV("target-legalize"), name, value, &max_iterations));
    return loom_low_target_legalize_parse_max_iterations(max_iterations,
                                                         context);
  }
  if (iree_string_view_equal(name, IREE_SV("mode"))) {
    return loom_low_target_legalize_parse_mode(value, context);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown option '%.*s' for pass 'target-legalize'",
                          (int)name.size, name.data);
}

iree_status_t loom_low_target_legalize_create(loom_pass_t* pass,
                                              iree_string_view_t options) {
  loom_low_target_legalize_pass_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(pass->instance_arena, sizeof(*state),
                                           (void**)&state));
  memset(state, 0, sizeof(*state));
  state->max_errors = 20;
  state->max_iterations = LOOM_GREEDY_REWRITE_DEFAULT_MAX_ITERATIONS;
  state->mode = LOOM_TARGET_LEGALIZATION_MODE_EAGER;

  loom_low_target_legalize_parse_context_t context = {
      .state = state,
  };
  if (pass->decoded_options) {
    for (uint16_t i = 0; i < pass->decoded_options->option_count; ++i) {
      const loom_pass_decoded_option_t* option =
          &pass->decoded_options->options[i];
      if (!option->present) {
        continue;
      }
      if (iree_string_view_equal(option->schema->name, IREE_SV("max-errors"))) {
        IREE_RETURN_IF_ERROR(loom_low_target_legalize_parse_max_errors(
            option->uint32_value, &context));
        continue;
      }
      if (iree_string_view_equal(option->schema->name,
                                 IREE_SV("max-iterations"))) {
        IREE_RETURN_IF_ERROR(loom_low_target_legalize_parse_max_iterations(
            option->uint32_value, &context));
        continue;
      }
      if (iree_string_view_equal(option->schema->name, IREE_SV("mode"))) {
        IREE_RETURN_IF_ERROR(loom_low_target_legalize_parse_mode(
            option->schema->enum_values[option->enum_value_index].value,
            &context));
        continue;
      }
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown decoded option '%.*s' for pass "
                              "'target-legalize'",
                              (int)option->schema->name.size,
                              option->schema->name.data);
    }
  } else {
    IREE_RETURN_IF_ERROR(
        loom_pass_options_parse(pass->info->name, options,
                                (loom_pass_option_parse_callback_t){
                                    .fn = loom_low_target_legalize_parse_option,
                                    .user_data = &context,
                                }));
  }

  pass->state = state;
  return iree_ok_status();
}

typedef struct loom_low_target_legalize_final_rejection_t {
  // Op rejected by a final legalizer.
  const loom_op_t* op;
  // Legalizer rejection result copied for final verification.
  loom_target_legalizer_result_t result;
  // Next final rejection record in function-local insertion order.
  struct loom_low_target_legalize_final_rejection_t* next;
} loom_low_target_legalize_final_rejection_t;

typedef struct loom_low_target_legalize_function_state_t {
  // Pass invocation being executed.
  loom_pass_t* pass;
  // Module being legalized.
  loom_module_t* module;
  // Selected target-bound source function.
  const loom_low_source_selection_t* selection;
  // Source-to-low query options shared by eager and final legality checks.
  loom_low_lower_options_t lower_options;
  // Low descriptor set selected by selection->target_bundle.
  const loom_low_descriptor_set_t* descriptor_set;
  // Active source-to-low contract query scope, rebuilt after IR mutation.
  loom_low_lower_source_query_scope_t* query_scope;
  // Fact table used to initialize query_scope.
  const loom_value_fact_table_t* query_scope_fact_table;
  // Arena receiving query_scope storage for this function run.
  iree_arena_allocator_t* query_scope_arena;
  // Target legalizer registry shared across functions in this pass run.
  const loom_target_legalizer_registry_t* legalizer_registry;
  // Target-low legality providers available to the final verifier.
  loom_target_low_legality_provider_list_t legality_provider_list;
  // Current legalization context passed to legalizer callbacks.
  loom_target_legalization_context_t legalization_context;
  // Optional compile report receiving cold legalization decision rows.
  loom_target_compile_report_t* compile_report;
  // First final legalizer rejection retained for final legality diagnostics.
  loom_low_target_legalize_final_rejection_t* final_rejections;
  // True when contract queries should surface retained legalizer rejections.
  bool use_final_rejections;
} loom_low_target_legalize_function_state_t;

static loom_target_compile_report_legalization_mode_t
loom_low_target_legalize_report_mode(loom_target_legalization_mode_t mode) {
  switch (mode) {
    case LOOM_TARGET_LEGALIZATION_MODE_EAGER:
      return LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_MODE_EAGER;
    case LOOM_TARGET_LEGALIZATION_MODE_FINAL:
      return LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_MODE_FINAL;
    default:
      return LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_MODE_NONE;
  }
}

static loom_target_compile_report_contract_outcome_t
loom_low_target_legalize_report_contract_outcome(
    loom_target_contract_query_outcome_t outcome) {
  switch (outcome) {
    case LOOM_TARGET_CONTRACT_QUERY_UNHANDLED:
      return LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_UNHANDLED;
    case LOOM_TARGET_CONTRACT_QUERY_LEGAL:
      return LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_LEGAL;
    case LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED:
      return LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_UNSUPPORTED;
    case LOOM_TARGET_CONTRACT_QUERY_INVALID_IR:
      return LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_INVALID_IR;
    default:
      return LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_NONE;
  }
}

static loom_target_compile_report_legalizer_strategy_t
loom_low_target_legalize_report_legalizer_strategy(
    loom_target_legalizer_strategy_t strategy) {
  switch (strategy) {
    case LOOM_TARGET_LEGALIZER_STRATEGY_TARGET:
      return LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_TARGET;
    case LOOM_TARGET_LEGALIZER_STRATEGY_REFERENCE:
      return LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_REFERENCE;
    case LOOM_TARGET_LEGALIZER_STRATEGY_UNKNOWN:
    default:
      return LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_NONE;
  }
}

static iree_string_view_t loom_low_target_legalize_function_name(
    const loom_low_target_legalize_function_state_t* state) {
  const loom_symbol_ref_t callee =
      loom_func_like_callee(state->selection->func);
  if (!loom_symbol_ref_is_valid(callee) || callee.module_id != 0 ||
      callee.symbol_id >= state->module->symbols.count) {
    return iree_string_view_empty();
  }
  const loom_symbol_t* symbol =
      &state->module->symbols.entries[callee.symbol_id];
  if (symbol->name_id >= state->module->strings.count) {
    return iree_string_view_empty();
  }
  return state->module->strings.entries[symbol->name_id];
}

static iree_string_view_t loom_low_target_legalize_nonempty(
    iree_string_view_t value, iree_string_view_t placeholder) {
  return iree_string_view_is_empty(value) ? placeholder : value;
}

static uint64_t loom_low_target_legalize_report_descriptor_id(
    const loom_target_contract_query_result_t* query_result) {
  return query_result->selected_descriptor
             ? query_result->selected_descriptor->stable_id
             : UINT64_MAX;
}

static bool loom_low_target_legalize_report_wants_rows(
    const loom_low_target_legalize_function_state_t* state) {
  return state->compile_report != NULL &&
         state->compile_report->target_legalization_rows != NULL;
}

static void loom_low_target_legalize_record_report_row(
    loom_low_target_legalize_function_state_t* state,
    iree_string_view_t source_op_name, uint32_t source_op_kind,
    loom_target_compile_report_legalization_action_t action,
    const loom_target_contract_query_result_t* query_result,
    const loom_target_legalizer_entry_t* legalizer_entry,
    uint64_t created_op_count, uint64_t erased_op_count) {
  if (state->compile_report == NULL) {
    return;
  }
  const loom_target_compile_report_legalizer_strategy_t legalizer_strategy =
      legalizer_entry ? loom_low_target_legalize_report_legalizer_strategy(
                            legalizer_entry->provider_strategy)
                      : LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_NONE;
  if (!loom_low_target_legalize_report_wants_rows(state)) {
    loom_target_compile_report_record_legalization_summary(
        state->compile_report, action, legalizer_strategy);
    return;
  }

  const loom_target_bundle_t* bundle = state->selection->target_bundle;
  const loom_target_config_t* config = bundle ? bundle->config : NULL;
  const iree_string_view_t legalizer_name = legalizer_entry
                                                ? legalizer_entry->provider_name
                                                : iree_string_view_empty();
  const loom_target_compile_report_legalization_row_t row = {
      .function_name = loom_low_target_legalize_function_name(state),
      .source_op_name = source_op_name,
      .source_op_kind = source_op_kind,
      .target_bundle_name = bundle ? bundle->name : iree_string_view_empty(),
      .target_config_name = config ? config->name : iree_string_view_empty(),
      .legalizer_name = legalizer_name,
      .legalizer_strategy = legalizer_strategy,
      .mode = loom_low_target_legalize_report_mode(
          state->legalization_context.mode),
      .action = action,
      .contract_outcome = loom_low_target_legalize_report_contract_outcome(
          query_result->outcome),
      .binding_index = query_result->binding_index,
      .case_index = query_result->case_index,
      .rule_set_index = query_result->rule_set_index,
      .rule_index = query_result->rule_index,
      .diagnostic_index = query_result->diagnostic_index,
      .descriptor_id =
          loom_low_target_legalize_report_descriptor_id(query_result),
      .source_rejection_bits = query_result->source_rejection_bits,
      .source_rejection_detail = query_result->source_rejection_detail,
      .target_rejection_bits = query_result->target_rejection_bits,
      .missing_feature_bits = query_result->missing_feature_bits,
      .missing_fact_bits = query_result->missing_fact_bits,
      .created_op_count = created_op_count,
      .erased_op_count = erased_op_count,
  };
  loom_target_compile_report_record_legalization_row(state->compile_report,
                                                     &row);
}

static loom_target_contract_query_result_t
loom_low_target_legalize_result_for_legalizer_report(
    const loom_target_contract_query_result_t* query_result,
    const loom_target_legalizer_result_t* legalizer_result) {
  loom_target_contract_query_result_t report_result = *query_result;
  report_result.source_rejection_bits |=
      legalizer_result->source_rejection_bits;
  if (legalizer_result->source_rejection_detail !=
      LOOM_CONTRACT_REJECTION_DETAIL_NONE) {
    report_result.source_rejection_detail =
        legalizer_result->source_rejection_detail;
  }
  report_result.target_rejection_bits |=
      legalizer_result->target_rejection_bits;
  report_result.missing_feature_bits |= legalizer_result->missing_feature_bits;
  report_result.missing_fact_bits |= legalizer_result->missing_fact_bits;
  return report_result;
}

static bool loom_low_target_legalize_should_record_final_rejection(
    const loom_low_target_legalize_function_state_t* state,
    const loom_target_legalizer_entry_t* entry) {
  return state->legalization_context.mode ==
             LOOM_TARGET_LEGALIZATION_MODE_FINAL &&
         entry->provider_strategy == LOOM_TARGET_LEGALIZER_STRATEGY_REFERENCE;
}

static iree_status_t loom_low_target_legalize_record_final_rejection(
    loom_low_target_legalize_function_state_t* state, const loom_op_t* op,
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalizer_result_t result) {
  if (!loom_low_target_legalize_should_record_final_rejection(state, entry)) {
    return iree_ok_status();
  }
  loom_low_target_legalize_final_rejection_t* rejection = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      state->query_scope_arena, sizeof(*rejection), (void**)&rejection));
  *rejection = (loom_low_target_legalize_final_rejection_t){
      .op = op,
      .result = result,
      .next = state->final_rejections,
  };
  state->final_rejections = rejection;
  return iree_ok_status();
}

static const loom_low_target_legalize_final_rejection_t*
loom_low_target_legalize_find_final_rejection(
    const loom_low_target_legalize_function_state_t* state,
    const loom_op_t* op) {
  if (!state->use_final_rejections) {
    return NULL;
  }
  const loom_low_target_legalize_final_rejection_t* rejection =
      state->final_rejections;
  while (rejection != NULL) {
    if (rejection->op == op) {
      return rejection;
    }
    rejection = rejection->next;
  }
  return NULL;
}

static iree_string_view_t loom_low_target_legalize_reference_source_contract(
    void) {
  return IREE_SV("target.reference");
}

static iree_string_view_t loom_low_target_legalize_reference_constraint_key(
    const loom_target_legalizer_result_t* result) {
  switch ((loom_contract_rejection_detail_t)result->source_rejection_detail) {
    case LOOM_CONTRACT_REJECTION_DETAIL_MATRIX_LHS_FRAGMENT_OWNERSHIP:
      return IREE_SV("reference.lhs_fragment_ownership");
    case LOOM_CONTRACT_REJECTION_DETAIL_MATRIX_RHS_FRAGMENT_OWNERSHIP:
      return IREE_SV("reference.rhs_fragment_ownership");
    case LOOM_CONTRACT_REJECTION_DETAIL_MATRIX_INIT_FRAGMENT_OWNERSHIP:
      return IREE_SV("reference.init_fragment_ownership");
    case LOOM_CONTRACT_REJECTION_DETAIL_MATRIX_RESULT_FRAGMENT_PAYLOAD:
      return IREE_SV("reference.result_fragment_payload");
    case LOOM_CONTRACT_REJECTION_DETAIL_NONE:
    default:
      break;
  }
  const uint32_t bits = result->source_rejection_bits;
  if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_INVALID_REQUEST)) {
    return IREE_SV("reference.request_shape");
  }
  if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_ROLE)) {
    return IREE_SV("reference.fragment_roles");
  }
  if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_SHAPE)) {
    return IREE_SV("reference.fragment_shape");
  }
  if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_SCHEMA)) {
    return IREE_SV("reference.fragment_schema");
  }
  if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_AUXILIARY_OPERAND)) {
    return IREE_SV("reference.auxiliary_operands");
  }
  if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_NUMERIC)) {
    return IREE_SV("reference.numeric_type");
  }
  if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_FRAGMENT)) {
    return IREE_SV("reference.fragment_ownership");
  }
  if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_CAPABILITY)) {
    return IREE_SV("reference.capability");
  }
  if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_POLICY)) {
    return IREE_SV("reference.policy");
  }
  if (result->missing_feature_bits != 0) {
    return IREE_SV("reference.target_features");
  }
  if (result->missing_fact_bits != 0) {
    return IREE_SV("reference.value_facts");
  }
  if (result->target_rejection_bits != 0) {
    return IREE_SV("reference.target_constraints");
  }
  return IREE_SV("reference.unsupported");
}

static iree_status_t loom_low_target_legalize_make_final_rejection(
    loom_low_target_legalize_function_state_t* state,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    const loom_low_target_legalize_final_rejection_t* final_rejection,
    const loom_target_contract_rejection_t** out_rejection) {
  *out_rejection = NULL;
  loom_target_contract_rejection_t* rejection = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      environment->arena, sizeof(*rejection), (void**)&rejection));
  loom_diagnostic_param_t* params = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      environment->arena, 7, sizeof(*params), (void**)&params));
  const loom_target_bundle_t* bundle = environment->bundle;
  const loom_target_config_t* config = bundle ? bundle->config : NULL;
  const loom_target_export_plan_t* export_plan =
      bundle ? bundle->export_plan : NULL;
  params[0] = loom_param_string(loom_low_target_legalize_nonempty(
      bundle ? bundle->name : iree_string_view_empty(), IREE_SV("<empty>")));
  params[1] = loom_param_string(loom_low_target_legalize_nonempty(
      export_plan ? export_plan->name : iree_string_view_empty(),
      IREE_SV("<empty>")));
  params[2] = loom_param_string(loom_low_target_legalize_nonempty(
      config ? config->name : iree_string_view_empty(), IREE_SV("<empty>")));
  params[3] = loom_param_string(loom_low_target_legalize_nonempty(
      loom_low_target_legalize_function_name(state), IREE_SV("<module>")));
  params[4] = loom_param_string(loom_op_name(environment->module, source_op));
  params[5] =
      loom_param_string(loom_low_target_legalize_reference_source_contract());
  params[6] =
      loom_param_string(loom_low_target_legalize_reference_constraint_key(
          &final_rejection->result));
  *rejection = (loom_target_contract_rejection_t){
      .error_ref = LOOM_ERR_TARGET_039_REF,
      .params = params,
      .param_count = 7,
  };
  *out_rejection = rejection;
  return iree_ok_status();
}

static iree_status_t
loom_low_target_legalize_apply_final_rejection_to_contract_query(
    loom_low_target_legalize_function_state_t* state,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op, loom_target_contract_query_result_t* result) {
  if (result->outcome != LOOM_TARGET_CONTRACT_QUERY_UNHANDLED) {
    return iree_ok_status();
  }
  const loom_low_target_legalize_final_rejection_t* final_rejection =
      loom_low_target_legalize_find_final_rejection(state, source_op);
  if (final_rejection == NULL) {
    return iree_ok_status();
  }
  const loom_target_contract_rejection_t* rejection = NULL;
  IREE_RETURN_IF_ERROR(loom_low_target_legalize_make_final_rejection(
      state, environment, source_op, final_rejection, &rejection));
  *result = loom_target_contract_query_result_empty();
  result->outcome = final_rejection->result.action ==
                            LOOM_TARGET_LEGALIZER_ACTION_REJECT_INVALID_IR
                        ? LOOM_TARGET_CONTRACT_QUERY_INVALID_IR
                        : LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED;
  result->source_rejection_bits = final_rejection->result.source_rejection_bits;
  result->source_rejection_detail =
      final_rejection->result.source_rejection_detail;
  result->target_rejection_bits = final_rejection->result.target_rejection_bits;
  result->missing_feature_bits = final_rejection->result.missing_feature_bits;
  result->missing_fact_bits = final_rejection->result.missing_fact_bits;
  result->rejection = rejection;
  return iree_ok_status();
}

static void loom_low_target_legalize_destroy_query_scope(
    loom_low_target_legalize_function_state_t* state) {
  if (state->query_scope == NULL) {
    return;
  }
  loom_low_lower_source_query_scope_destroy(state->query_scope);
  state->query_scope = NULL;
  state->query_scope_fact_table = NULL;
}

static iree_status_t loom_low_target_legalize_refresh_query_scope(
    loom_low_target_legalize_function_state_t* state,
    const loom_value_fact_table_t* fact_table) {
  loom_low_target_legalize_destroy_query_scope(state);
  state->lower_options.fact_table = (loom_value_fact_table_t*)fact_table;
  IREE_RETURN_IF_ERROR(loom_low_lower_source_query_scope_create(
      state->module, state->selection->func, &state->lower_options,
      state->query_scope_arena, &state->query_scope));
  state->query_scope_fact_table = fact_table;
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_ensure_query_scope(
    loom_low_target_legalize_function_state_t* state,
    const loom_value_fact_table_t* fact_table) {
  if (state->query_scope != NULL &&
      state->query_scope_fact_table == fact_table) {
    return iree_ok_status();
  }
  return loom_low_target_legalize_refresh_query_scope(state, fact_table);
}

static iree_status_t loom_low_target_legalize_query_contract(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  loom_low_target_legalize_function_state_t* state =
      (loom_low_target_legalize_function_state_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_low_target_legalize_ensure_query_scope(
      state, environment->fact_table));
  const loom_target_contract_query_callback_t callback =
      loom_low_lower_source_query_scope_callback(state->query_scope);
  IREE_RETURN_IF_ERROR(
      callback.fn(callback.user_data, environment, source_op, out_result));
  return loom_low_target_legalize_apply_final_rejection_to_contract_query(
      state, environment, source_op, out_result);
}

static iree_status_t loom_low_target_legalize_rewrite_op(
    void* user_data, loom_greedy_rewrite_driver_t* driver, loom_op_t* op,
    loom_greedy_rewrite_result_t* result, bool* out_changed) {
  *out_changed = false;
  loom_low_target_legalize_function_state_t* state =
      (loom_low_target_legalize_function_state_t*)user_data;
  driver->rewriter.flags = 0;
  bool erased = false;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_erase_if_dead(&driver->rewriter, op, &erased));
  if (erased) {
    loom_greedy_rewrite_result_record_change(
        result, &driver->rewriter,
        LOOM_GREEDY_REWRITE_CHANGE_FLAG_COUNT_MODIFIED_OP);
    *out_changed = true;
    return iree_ok_status();
  }

  const loom_target_legalizer_op_entry_t op_entry =
      loom_target_legalizer_registry_lookup_kind(state->legalizer_registry,
                                                 op->kind);
  if (loom_target_legalizer_op_entry_is_empty(op_entry)) {
    return iree_ok_status();
  }
  const bool report_wants_rows =
      loom_low_target_legalize_report_wants_rows(state);
  const iree_string_view_t source_op_name =
      report_wants_rows ? loom_op_name(state->module, op)
                        : iree_string_view_empty();
  const uint32_t source_op_kind = op->kind;

  state->legalization_context.fact_table = driver->latest_facts;
  state->legalization_context.rewriter = &driver->rewriter;
  state->legalization_context.arena = driver->scratch_arena;
  loom_target_contract_query_result_t query_result =
      loom_target_contract_query_result_empty();
  IREE_RETURN_IF_ERROR(loom_target_legalization_query_contract(
      &state->legalization_context, op, &query_result));
  if (query_result.outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL) {
    loom_low_target_legalize_record_report_row(
        state, source_op_name, source_op_kind,
        LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_LEGAL, &query_result,
        /*legalizer_entry=*/NULL, /*created_op_count=*/0,
        /*erased_op_count=*/0);
    loom_pass_statistic_add(state->pass,
                            LOOM_LOW_TARGET_LEGALIZE_STAT_OPS_LEGAL, 1);
    return iree_ok_status();
  }
  if (query_result.outcome == LOOM_TARGET_CONTRACT_QUERY_INVALID_IR) {
    loom_low_target_legalize_record_report_row(
        state, source_op_name, source_op_kind,
        LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REJECT_INVALID_IR,
        &query_result, /*legalizer_entry=*/NULL, /*created_op_count=*/0,
        /*erased_op_count=*/0);
    loom_pass_statistic_add(state->pass,
                            LOOM_LOW_TARGET_LEGALIZE_STAT_OPS_DEFERRED, 1);
    return iree_ok_status();
  }

  for (uint16_t i = 0; i < op_entry.entry_count; ++i) {
    const loom_target_legalizer_entry_t* entry =
        &state->legalizer_registry->entries[op_entry.entry_start + i];
    driver->rewriter.flags = 0;
    const uint64_t created_op_count_before = driver->rewriter.created_op_count;
    const uint64_t erased_op_count_before = driver->rewriter.erased_op_count;
    loom_target_legalizer_result_t legalizer_result = {
        .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
    };
    state->legalization_context.contract_query_result = &query_result;
    iree_status_t legalizer_status = entry->legalize(
        entry, &state->legalization_context, op, &legalizer_result);
    state->legalization_context.contract_query_result = NULL;
    IREE_RETURN_IF_ERROR(legalizer_status);
    const bool legalizer_mutated =
        iree_any_bit_set(driver->rewriter.flags, LOOM_REWRITER_FLAG_CHANGED);
    if (legalizer_result.action == LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN) {
      if (!legalizer_mutated) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "target legalizer reported REWRITTEN without mutating IR");
      }
    } else if (legalizer_mutated) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "target legalizer action %d mutated IR without reporting REWRITTEN",
          (int)legalizer_result.action);
    }
    switch (legalizer_result.action) {
      case LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT:
        continue;
      case LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN: {
        loom_greedy_rewrite_result_record_change(
            result, &driver->rewriter,
            LOOM_GREEDY_REWRITE_CHANGE_FLAG_COUNT_MODIFIED_OP);
        const loom_target_contract_query_result_t rewritten_report_result =
            loom_low_target_legalize_result_for_legalizer_report(
                &query_result, &legalizer_result);
        loom_low_target_legalize_record_report_row(
            state, source_op_name, source_op_kind,
            LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REWRITTEN,
            &rewritten_report_result, entry,
            driver->rewriter.created_op_count - created_op_count_before,
            driver->rewriter.erased_op_count - erased_op_count_before);
        loom_pass_statistic_add(state->pass,
                                LOOM_LOW_TARGET_LEGALIZE_STAT_OPS_REWRITTEN, 1);
        *out_changed = true;
        return iree_ok_status();
      }
      case LOOM_TARGET_LEGALIZER_ACTION_DEFER: {
        const loom_target_contract_query_result_t deferred_report_result =
            loom_low_target_legalize_result_for_legalizer_report(
                &query_result, &legalizer_result);
        loom_low_target_legalize_record_report_row(
            state, source_op_name, source_op_kind,
            LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_DEFERRED,
            &deferred_report_result, entry,
            /*created_op_count=*/0, /*erased_op_count=*/0);
        loom_pass_statistic_add(state->pass,
                                LOOM_LOW_TARGET_LEGALIZE_STAT_OPS_DEFERRED, 1);
        return iree_ok_status();
      }
      case LOOM_TARGET_LEGALIZER_ACTION_REJECT_INVALID_IR: {
        IREE_RETURN_IF_ERROR(loom_low_target_legalize_record_final_rejection(
            state, op, entry, legalizer_result));
        const loom_target_contract_query_result_t invalid_report_result =
            loom_low_target_legalize_result_for_legalizer_report(
                &query_result, &legalizer_result);
        loom_low_target_legalize_record_report_row(
            state, source_op_name, source_op_kind,
            LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REJECT_INVALID_IR,
            &invalid_report_result, entry,
            /*created_op_count=*/0, /*erased_op_count=*/0);
        loom_pass_statistic_add(state->pass,
                                LOOM_LOW_TARGET_LEGALIZE_STAT_OPS_DEFERRED, 1);
        return iree_ok_status();
      }
      case LOOM_TARGET_LEGALIZER_ACTION_REJECT_UNSUPPORTED_FINAL: {
        IREE_RETURN_IF_ERROR(loom_low_target_legalize_record_final_rejection(
            state, op, entry, legalizer_result));
        const loom_target_contract_query_result_t unsupported_report_result =
            loom_low_target_legalize_result_for_legalizer_report(
                &query_result, &legalizer_result);
        loom_low_target_legalize_record_report_row(
            state, source_op_name, source_op_kind,
            LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REJECT_UNSUPPORTED_FINAL,
            &unsupported_report_result, entry,
            /*created_op_count=*/0, /*erased_op_count=*/0);
        loom_pass_statistic_add(state->pass,
                                LOOM_LOW_TARGET_LEGALIZE_STAT_OPS_DEFERRED, 1);
        return iree_ok_status();
      }
      default:
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "target legalizer returned unknown action %d",
                                (int)legalizer_result.action);
    }
  }

  loom_pass_statistic_add(state->pass,
                          LOOM_LOW_TARGET_LEGALIZE_STAT_OPS_DEFERRED, 1);
  loom_low_target_legalize_record_report_row(
      state, source_op_name, source_op_kind,
      LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_UNHANDLED, &query_result,
      /*legalizer_entry=*/NULL, /*created_op_count=*/0,
      /*erased_op_count=*/0);
  return iree_ok_status();
}

static void loom_low_target_legalize_changed(
    void* user_data, loom_greedy_rewrite_driver_t* driver) {
  (void)driver;
  loom_low_target_legalize_destroy_query_scope(
      (loom_low_target_legalize_function_state_t*)user_data);
}

static iree_status_t loom_low_target_legalize_verify_final(
    loom_module_t* module, loom_low_target_legalize_function_state_t* state,
    const loom_low_target_legalize_pass_state_t* pass_state,
    const loom_value_fact_table_t* fact_table, uint32_t* out_error_count) {
  *out_error_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_low_target_legalize_refresh_query_scope(state, fact_table));
  const loom_target_contract_query_callback_t contract_query = {
      .fn = loom_low_target_legalize_query_contract,
      .user_data = state,
  };
  loom_target_low_legality_result_t result = {0};
  const loom_target_low_legality_options_t legality_options = {
      .bundle = state->selection->target_bundle,
      .target_ref = state->selection->target_ref,
      .descriptor_registry = state->lower_options.descriptor_registry,
      .error_catalog = state->selection->policy->error_catalog,
      .provider_list = state->legality_provider_list,
      .contract_query = contract_query,
      .type_supported = state->selection->policy->source_type_supported,
      .fact_table = (loom_value_fact_table_t*)fact_table,
      .value_domain =
          loom_low_lower_source_query_scope_value_domain(state->query_scope),
      .emitter = state->pass->diagnostic_emitter,
      .max_errors = pass_state->max_errors,
  };
  state->use_final_rejections = true;
  iree_status_t status = loom_target_low_verify_function_legality(
      module, state->selection->func, &legality_options, &result);
  state->use_final_rejections = false;
  *out_error_count = result.error_count;
  return status;
}

static iree_status_t loom_low_target_legalize_run_final_structure_lowering(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function) {
  const loom_pass_info_t* original_info = pass->info;
  int64_t* original_statistics = pass->statistics;
  void* original_state = pass->state;
  pass->info = loom_scf_to_cfg_pass_info();
  pass->statistics = NULL;
  pass->state = NULL;
  iree_status_t status = loom_scf_to_cfg_run(pass, module, function);
  pass->info = original_info;
  pass->statistics = original_statistics;
  pass->state = original_state;
  return status;
}

static iree_status_t loom_low_target_legalize_acquire_final_facts(
    loom_pass_t* pass, loom_module_t* module,
    const loom_low_source_selection_t* selection,
    const loom_greedy_rewrite_driver_t* rewrite_driver,
    const loom_value_fact_table_t** out_fact_table) {
  *out_fact_table = loom_greedy_rewrite_driver_fact_table(rewrite_driver);
  if (pass->value_facts == NULL) {
    return iree_ok_status();
  }
  loom_value_fact_table_t* fact_table = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_value_facts_acquire(
      pass, module,
      loom_pass_value_fact_scope_function_for_target(selection->func,
                                                     selection->target_bundle),
      &fact_table));
  *out_fact_table = fact_table;
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_function(
    loom_pass_t* pass, loom_module_t* module,
    const loom_low_target_legalize_pass_state_t* pass_state,
    const loom_low_source_selection_t* selection,
    const loom_target_legalizer_registry_t* legalizer_registry,
    loom_target_low_legality_provider_list_t legality_provider_list,
    iree_arena_allocator_t* arena, bool* out_emitted_error_diagnostics) {
  *out_emitted_error_diagnostics = false;
  const loom_low_pass_capability_t* low_capability =
      loom_low_pass_capability_from_pass(pass);
  const loom_low_descriptor_registry_t* descriptor_registry =
      loom_low_pass_capability_descriptor_registry(low_capability);

  loom_low_target_legalize_function_state_t state = {
      .pass = pass,
      .module = module,
      .selection = selection,
      .query_scope_arena = arena,
      .legalizer_registry = legalizer_registry,
      .legality_provider_list = legality_provider_list,
      .compile_report = loom_low_pass_capability_compile_report(low_capability),
  };
  IREE_RETURN_IF_ERROR(loom_target_low_descriptor_set_select_for_bundle(
      descriptor_registry, selection->target_bundle, &state.descriptor_set));

  loom_value_fact_table_t* seed_facts = NULL;
  if (pass->value_facts != NULL) {
    IREE_RETURN_IF_ERROR(loom_pass_value_facts_acquire(
        pass, module,
        loom_pass_value_fact_scope_function_for_target(
            selection->func, selection->target_bundle),
        &seed_facts));
  }
  state.lower_options = (loom_low_lower_options_t){
      .target_ref = selection->target_ref,
      .bundle = selection->target_bundle,
      .descriptor_registry = descriptor_registry,
      .legality_provider_list = legality_provider_list,
      .policy = selection->policy,
      .fact_table = seed_facts,
      .emitter = pass->diagnostic_emitter,
      .max_errors = pass_state->max_errors,
  };

  loom_pass_value_fact_owner_t rewrite_value_facts = {0};
  loom_pass_value_fact_owner_initialize(module->arena.block_pool,
                                        &rewrite_value_facts);
  iree_arena_allocator_t rewrite_arena;
  iree_arena_initialize(module->arena.block_pool, &rewrite_arena);
  loom_greedy_rewrite_driver_t rewrite_driver;
  loom_greedy_rewrite_driver_initialize(module, &rewrite_arena,
                                        &rewrite_value_facts, &rewrite_driver);

  state.legalization_context = (loom_target_legalization_context_t){
      .pass = pass,
      .module = module,
      .function = selection->func,
      .bundle = selection->target_bundle,
      .target_ref = selection->target_ref,
      .descriptor_set = state.descriptor_set,
      .mode = pass_state->mode,
      .contract_query =
          {
              .fn = loom_low_target_legalize_query_contract,
              .user_data = &state,
          },
  };
  const loom_greedy_rewrite_options_t rewrite_options = {
      .max_iterations = pass_state->max_iterations,
      .seed_facts = seed_facts,
  };
  const loom_greedy_rewrite_callbacks_t rewrite_callbacks = {
      .user_data = &state,
      .rewrite_op = loom_low_target_legalize_rewrite_op,
      .changed = loom_low_target_legalize_changed,
  };
  loom_greedy_rewrite_result_t rewrite_result = {0};
  iree_status_t status = loom_greedy_rewrite_run_region(
      &rewrite_driver, selection->func, loom_func_like_body(selection->func),
      selection->func.op, &rewrite_options, &rewrite_callbacks,
      &rewrite_result);
  if (iree_status_is_ok(status) && rewrite_result.changed) {
    loom_pass_mark_changed(pass);
    if (pass->value_facts != NULL) {
      loom_pass_value_fact_owner_invalidate(pass->value_facts);
    }
  }

  uint32_t final_error_count = 0;
  if (iree_status_is_ok(status) &&
      pass_state->mode == LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    status = loom_low_target_legalize_run_final_structure_lowering(
        pass, module, selection->func);
  }
  if (iree_status_is_ok(status) &&
      pass_state->mode == LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    const loom_value_fact_table_t* final_facts = NULL;
    status = loom_low_target_legalize_acquire_final_facts(
        pass, module, selection, &rewrite_driver, &final_facts);
    if (!iree_status_is_ok(status)) {
      goto cleanup;
    }
    status = loom_low_target_legalize_verify_final(
        module, &state, pass_state, final_facts, &final_error_count);
  }
cleanup:
  loom_low_target_legalize_destroy_query_scope(&state);
  loom_greedy_rewrite_driver_deinitialize(&rewrite_driver);
  iree_arena_deinitialize(&rewrite_arena);
  loom_pass_value_fact_owner_deinitialize(&rewrite_value_facts);
  IREE_RETURN_IF_ERROR(status);

  loom_pass_statistic_add(pass, LOOM_LOW_TARGET_LEGALIZE_STAT_ERRORS,
                          final_error_count);
  *out_emitted_error_diagnostics = final_error_count != 0;
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_compose_providers(
    const loom_target_legalizer_provider_list_t* target_provider_list,
    iree_arena_allocator_t* arena,
    const loom_target_legalizer_provider_t* const** out_providers,
    iree_host_size_t* out_provider_count) {
  const loom_target_legalizer_provider_t* scalar_provider =
      loom_scalar_target_legalizer_provider();
  const loom_target_legalizer_provider_t* vector_provider =
      loom_vector_target_legalizer_provider();
  const iree_host_size_t target_provider_count =
      target_provider_list ? target_provider_list->count : 0;
  const iree_host_size_t provider_count = target_provider_count + 2;
  const loom_target_legalizer_provider_t** providers = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, provider_count, sizeof(*providers), (void**)&providers));
  for (iree_host_size_t i = 0; i < target_provider_count; ++i) {
    providers[i] = target_provider_list->values[i];
  }
  providers[target_provider_count] = scalar_provider;
  providers[target_provider_count + 1] = vector_provider;
  *out_providers = providers;
  *out_provider_count = provider_count;
  return iree_ok_status();
}

iree_status_t loom_low_target_legalize_run(loom_pass_t* pass,
                                           loom_module_t* module) {
  const loom_low_target_legalize_pass_state_t* state =
      (const loom_low_target_legalize_pass_state_t*)pass->state;
  const loom_low_pass_capability_t* low_capability =
      loom_low_pass_capability_from_pass(pass);
  const loom_low_lower_policy_registry_t* policy_registry =
      loom_low_pass_capability_lower_policy_registry(low_capability);
  const loom_target_low_legality_provider_list_t* legality_provider_list =
      loom_low_pass_capability_legality_provider_list(low_capability);
  const loom_target_legalizer_provider_list_t* target_legalizer_provider_list =
      loom_low_pass_capability_legalizer_provider_list(low_capability);

  iree_arena_allocator_t run_arena;
  iree_arena_initialize(module->arena.block_pool, &run_arena);

  const loom_target_legalizer_provider_t* const* legalizer_providers = NULL;
  iree_host_size_t legalizer_provider_count = 0;
  iree_status_t status = loom_low_target_legalize_compose_providers(
      target_legalizer_provider_list, &run_arena, &legalizer_providers,
      &legalizer_provider_count);

  loom_target_legalizer_registry_t legalizer_registry = {0};
  if (iree_status_is_ok(status)) {
    status = loom_target_legalizer_registry_compose(
        legalizer_providers, legalizer_provider_count, &legalizer_registry,
        &run_arena);
  }

  loom_low_source_selection_list_t selection_list = {0};
  if (iree_status_is_ok(status)) {
    const loom_low_source_selection_options_t selection_options = {
        .policy_registry = policy_registry,
        .diagnostic_emitter = pass->diagnostic_emitter,
        .lowering_kind = IREE_SV("target-legalize"),
    };
    status = loom_low_select_source_funcs(module, &selection_options,
                                          &run_arena, &selection_list);
  }

  bool emitted_error_diagnostics = false;
  uint32_t function_count = 0;
  const loom_target_low_legality_provider_list_t legality_list =
      legality_provider_list ? *legality_provider_list
                             : loom_target_low_legality_provider_list_empty();
  for (iree_host_size_t i = 0;
       i < selection_list.count && iree_status_is_ok(status) &&
       !emitted_error_diagnostics;
       ++i) {
    status = loom_low_target_legalize_function(
        pass, module, state, &selection_list.values[i], &legalizer_registry,
        legality_list, &run_arena, &emitted_error_diagnostics);
    if (iree_status_is_ok(status)) {
      ++function_count;
    }
  }

  iree_arena_deinitialize(&run_arena);
  IREE_RETURN_IF_ERROR(status);

  loom_pass_statistic_add(pass, LOOM_LOW_TARGET_LEGALIZE_STAT_FUNCTIONS,
                          function_count);
  return iree_ok_status();
}
