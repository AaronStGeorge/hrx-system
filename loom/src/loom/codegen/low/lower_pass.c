// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower_pass.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/requirements.h"
#include "loom/pass/pipeline.h"
#include "loom/pass/registry.h"
#include "loom/target/module_compiler.h"

typedef struct loom_low_source_to_low_pass_state_t {
  // Optional module-local target.bundle symbol selected by the pass.
  iree_string_view_t target_symbol;
  // Maximum number of lowering diagnostics emitted before stopping.
  uint32_t max_errors;
  // True when max_errors was explicitly provided.
  bool has_max_errors_option;
  // True when target_symbol was explicitly provided.
  bool has_target_option;
} loom_low_source_to_low_pass_state_t;

typedef struct loom_low_source_to_low_parse_context_t {
  // Mutable pass state being populated.
  loom_low_source_to_low_pass_state_t* state;
} loom_low_source_to_low_parse_context_t;

static const loom_pass_option_def_t kLowSourceToLowOptions[] = {
    {IREE_SVL("max-errors"),
     IREE_SVL("Maximum number of source-to-low diagnostics to emit; zero "
              "means no limit.")},
    {IREE_SVL("target"),
     IREE_SVL("Optional target.bundle symbol to lower, with or without '@'.")},
};

enum {
  LOOM_LOW_SOURCE_TO_LOW_STAT_ERRORS = 0,
  LOOM_LOW_SOURCE_TO_LOW_STAT_FUNCTIONS = 1,
  LOOM_LOW_SOURCE_TO_LOW_STAT_REMARKS = 2,
};

static const loom_pass_statistic_def_t kLowSourceToLowStatistics[] = {
    {IREE_SVL("errors"), IREE_SVL("Number of lowering errors emitted.")},
    {IREE_SVL("functions"), IREE_SVL("Number of source functions lowered.")},
    {IREE_SVL("remarks"), IREE_SVL("Number of lowering remarks emitted.")},
};

static const loom_pass_info_t loom_low_source_to_low_pass_info_storage = {
    .name = IREE_SVL("source-to-low"),
    .description =
        IREE_SVL("Lower one target-exported source function to target-low IR."),
    .kind = LOOM_PASS_MODULE,
    .option_defs = kLowSourceToLowOptions,
    .option_count = IREE_ARRAYSIZE(kLowSourceToLowOptions),
    .statistic_defs = kLowSourceToLowStatistics,
    .statistic_count = IREE_ARRAYSIZE(kLowSourceToLowStatistics),
};

const loom_pass_info_t* loom_low_source_to_low_pass_info(void) {
  return &loom_low_source_to_low_pass_info_storage;
}

bool loom_low_source_to_low_pass_config_satisfies_requirement(
    const loom_low_source_to_low_pass_config_t* config,
    iree_string_view_t requirement) {
  if (iree_string_view_equal(
          requirement,
          IREE_SV(LOOM_LOW_PASS_REQUIREMENT_TARGET_LOW_DESCRIPTOR_REGISTRY))) {
    return config && config->descriptor_registry;
  }
  if (iree_string_view_equal(
          requirement,
          IREE_SV(
              LOOM_LOW_PASS_REQUIREMENT_TARGET_LOW_LOWER_POLICY_REGISTRY))) {
    return config && config->policy_registry;
  }
  return false;
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

static iree_status_t loom_low_source_to_low_parse_target(
    iree_string_view_t target,
    loom_low_source_to_low_parse_context_t* context) {
  if (context->state->has_target_option) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate option 'target' for pass "
                            "'source-to-low'");
  }
  target = iree_string_view_trim(target);
  if (iree_string_view_is_empty(target)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass 'source-to-low' option 'target' must not be empty");
  }
  context->state->target_symbol = target;
  context->state->has_target_option = true;
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
  if (iree_string_view_equal(name, IREE_SV("target"))) {
    return loom_low_source_to_low_parse_target(value, context);
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
      if (iree_string_view_equal(option->schema->name, IREE_SV("target"))) {
        IREE_RETURN_IF_ERROR(loom_low_source_to_low_parse_target(
            option->string_value, &context));
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

static bool loom_low_source_to_low_policy_registry_has_bundle(
    void* user_data, const loom_target_bundle_t* bundle) {
  return loom_low_lower_policy_registry_has_bundle(
      (const loom_low_lower_policy_registry_t*)user_data, bundle);
}

iree_status_t loom_low_source_to_low_run(loom_pass_t* pass,
                                         loom_module_t* module) {
  const loom_low_source_to_low_pass_state_t* state =
      (const loom_low_source_to_low_pass_state_t*)pass->state;
  const loom_low_source_to_low_pass_config_t* config =
      (const loom_low_source_to_low_pass_config_t*)pass->user_data;
  if (!config || !config->descriptor_registry || !config->policy_registry) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "pass 'source-to-low' requires injected low descriptor and lowering "
        "policy registries");
  }
  IREE_RETURN_IF_ERROR(
      loom_low_lower_policy_registry_verify(config->policy_registry));

  const loom_target_module_compile_options_t compile_options = {
      .target_symbol = state ? state->target_symbol : iree_string_view_empty(),
      .max_errors = state ? state->max_errors : 20,
  };
  loom_target_module_compile_target_t target = {0};
  IREE_RETURN_IF_ERROR(loom_target_module_compile_select_target(
      module, &compile_options,
      loom_low_source_to_low_policy_registry_has_bundle,
      (void*)config->policy_registry, IREE_SV("source-to-low"), &target));

  loom_func_like_t source_function = {0};
  IREE_RETURN_IF_ERROR(loom_target_module_compile_find_source_function(
      module, &target.bundle_storage.bundle, &source_function));

  const loom_low_lower_policy_t* policy = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_policy_registry_lookup_for_bundle(
      config->policy_registry, &target.bundle_storage.bundle, &policy));

  const loom_low_lower_options_t lower_options = {
      .target_ref = target.target_ref,
      .bundle = &target.bundle_storage.bundle,
      .descriptor_registry = config->descriptor_registry,
      .descriptor_requirements =
          LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
      .policy = policy,
      .emitter = pass->diagnostic_emitter,
      .max_errors = state ? state->max_errors : 20,
  };
  loom_low_lower_result_t lower_result = {0};
  IREE_RETURN_IF_ERROR(loom_low_lower_function(module, source_function,
                                               &lower_options, &lower_result));

  loom_pass_statistic_add(pass, LOOM_LOW_SOURCE_TO_LOW_STAT_ERRORS,
                          lower_result.error_count);
  loom_pass_statistic_add(pass, LOOM_LOW_SOURCE_TO_LOW_STAT_REMARKS,
                          lower_result.remark_count);
  if (lower_result.error_count > 0 || lower_result.low_func_op == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source-to-low lowering failed with %" PRIu32 " error%s",
        lower_result.error_count, lower_result.error_count == 1 ? "" : "s");
  }

  loom_pass_statistic_add(pass, LOOM_LOW_SOURCE_TO_LOW_STAT_FUNCTIONS, 1);
  loom_pass_mark_changed(pass);
  return iree_ok_status();
}
