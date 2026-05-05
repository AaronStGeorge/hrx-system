// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/passes/source_to_low.h"

#include <string.h>

#include "loom/codegen/low/pass_environment.h"
#include "loom/codegen/low/source_selection.h"
#include "loom/pass/pipeline.h"
#include "loom/pass/registry.h"
#include "loom/pass/value_facts.h"

typedef struct loom_low_source_to_low_pass_state_t {
  // Maximum number of lowering diagnostics emitted before stopping.
  uint32_t max_errors;
  // True when max_errors was explicitly provided.
  bool has_max_errors_option;
} loom_low_source_to_low_pass_state_t;

typedef struct loom_low_source_to_low_parse_context_t {
  // Mutable pass state being populated.
  loom_low_source_to_low_pass_state_t* state;
} loom_low_source_to_low_parse_context_t;

static const loom_pass_option_def_t kLowSourceToLowOptions[] = {
    {IREE_SVL("max-errors"),
     IREE_SVL("Maximum number of source-to-low diagnostics to emit; zero "
              "means no limit.")},
};

enum {
  LOOM_LOW_SOURCE_TO_LOW_STAT_ERRORS = 0,
  LOOM_LOW_SOURCE_TO_LOW_STAT_FUNCTIONS = 1,
  LOOM_LOW_SOURCE_TO_LOW_STAT_REMARKS = 2,
};

static const loom_pass_statistic_def_t kLowSourceToLowStatistics[] = {
    {IREE_SVL("errors"), IREE_SVL("Number of lowering errors emitted.")},
    {IREE_SVL("functions"), IREE_SVL("Number of source funcs lowered.")},
    {IREE_SVL("remarks"), IREE_SVL("Number of lowering remarks emitted.")},
};

static const loom_pass_info_t loom_low_source_to_low_pass_info_storage = {
    .name = IREE_SVL("source-to-low"),
    .description = IREE_SVL("Lower targeted source funcs to target-low IR."),
    .kind = LOOM_PASS_MODULE,
    .option_defs = kLowSourceToLowOptions,
    .option_count = IREE_ARRAYSIZE(kLowSourceToLowOptions),
    .statistic_defs = kLowSourceToLowStatistics,
    .statistic_count = IREE_ARRAYSIZE(kLowSourceToLowStatistics),
};

const loom_pass_info_t* loom_low_source_to_low_pass_info(void) {
  return &loom_low_source_to_low_pass_info_storage;
}

static iree_status_t loom_low_source_to_low_parse_max_errors(
    uint32_t max_errors, loom_low_source_to_low_parse_context_t* context) {
  if (context->state->has_max_errors_option) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "duplicate option 'max-errors' for pass 'source-to-low'");
  }
  context->state->max_errors = max_errors;
  context->state->has_max_errors_option = true;
  return iree_ok_status();
}

static iree_status_t loom_low_source_to_low_parse_option(
    void* user_data, iree_string_view_t name, iree_string_view_t value) {
  loom_low_source_to_low_parse_context_t* context =
      (loom_low_source_to_low_parse_context_t*)user_data;
  if (iree_string_view_equal(name, IREE_SV("max-errors"))) {
    uint32_t max_errors = 0;
    IREE_RETURN_IF_ERROR(loom_pass_option_parse_uint32(
        IREE_SV("source-to-low"), name, value, &max_errors));
    return loom_low_source_to_low_parse_max_errors(max_errors, context);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown option '%.*s' for pass 'source-to-low'",
                          (int)name.size, name.data);
}

iree_status_t loom_low_source_to_low_create(loom_pass_t* pass,
                                            iree_string_view_t options) {
  loom_low_source_to_low_pass_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(pass->instance_arena, sizeof(*state),
                                           (void**)&state));
  memset(state, 0, sizeof(*state));
  state->max_errors = 20;

  loom_low_source_to_low_parse_context_t context = {
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
        IREE_RETURN_IF_ERROR(loom_low_source_to_low_parse_max_errors(
            option->uint32_value, &context));
        continue;
      }
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown decoded option '%.*s' for pass "
                              "'source-to-low'",
                              (int)option->schema->name.size,
                              option->schema->name.data);
    }
  } else {
    IREE_RETURN_IF_ERROR(
        loom_pass_options_parse(pass->info->name, options,
                                (loom_pass_option_parse_callback_t){
                                    .fn = loom_low_source_to_low_parse_option,
                                    .user_data = &context,
                                }));
  }

  pass->state = state;
  return iree_ok_status();
}

iree_status_t loom_low_source_to_low_run(loom_pass_t* pass,
                                         loom_module_t* module) {
  const loom_low_source_to_low_pass_state_t* state =
      (const loom_low_source_to_low_pass_state_t*)pass->state;
  const loom_low_pass_capability_t* low_capability =
      loom_low_pass_capability_from_pass(pass);
  const loom_low_descriptor_registry_t* descriptor_registry =
      loom_low_pass_capability_descriptor_registry(low_capability);
  const loom_low_lower_policy_registry_t* policy_registry =
      loom_low_pass_capability_lower_policy_registry(low_capability);
  if (!descriptor_registry || !policy_registry) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "pass 'source-to-low' requires pass environment low descriptor and "
        "lowering policy registries");
  }

  iree_arena_allocator_t selection_arena;
  iree_arena_initialize(module->arena.block_pool, &selection_arena);
  loom_low_source_selection_list_t selection_list = {0};
  const loom_low_source_selection_options_t selection_options = {
      .policy_registry = policy_registry,
      .diagnostic_emitter = pass->diagnostic_emitter,
      .lowering_kind = IREE_SV("source-to-low"),
  };
  iree_status_t status = loom_low_select_source_funcs(
      module, &selection_options, &selection_arena, &selection_list);
  uint32_t function_count = 0;
  for (iree_host_size_t i = 0;
       i < selection_list.count && iree_status_is_ok(status); ++i) {
    const loom_low_source_selection_t* selection = &selection_list.values[i];
    const loom_target_low_legality_provider_list_t* legality_provider_list =
        loom_low_pass_capability_legality_provider_list(low_capability);
    loom_value_fact_table_t* fact_table = NULL;
    status = loom_pass_value_facts_acquire(
        pass, module,
        loom_pass_value_fact_scope_function_for_target(
            selection->func, selection->target_bundle),
        &fact_table);
    if (!iree_status_is_ok(status)) break;
    const loom_low_lower_options_t lower_options = {
        .target_ref = selection->target_ref,
        .bundle = selection->target_bundle,
        .descriptor_registry = descriptor_registry,
        .legality_provider_list =
            legality_provider_list
                ? *legality_provider_list
                : loom_target_low_legality_provider_list_empty(),
        .policy = selection->policy,
        .fact_table = fact_table,
        .emitter = pass->diagnostic_emitter,
        .max_errors = state ? state->max_errors : 20,
    };
    loom_low_lower_result_t lower_result = {0};
    status = loom_low_lower_function(module, selection->func, &lower_options,
                                     &lower_result);
    loom_pass_value_fact_owner_invalidate(pass->value_facts);
    loom_pass_statistic_add(pass, LOOM_LOW_SOURCE_TO_LOW_STAT_ERRORS,
                            lower_result.error_count);
    loom_pass_statistic_add(pass, LOOM_LOW_SOURCE_TO_LOW_STAT_REMARKS,
                            lower_result.remark_count);
    if (iree_status_is_ok(status) && lower_result.error_count > 0) {
      break;
    }
    if (iree_status_is_ok(status)) {
      IREE_ASSERT(lower_result.low_func_op != NULL);
      ++function_count;
    }
  }
  iree_arena_deinitialize(&selection_arena);
  IREE_RETURN_IF_ERROR(status);

  loom_pass_statistic_add(pass, LOOM_LOW_SOURCE_TO_LOW_STAT_FUNCTIONS,
                          function_count);
  if (function_count > 0) {
    loom_pass_mark_changed(pass);
  }
  return iree_ok_status();
}
