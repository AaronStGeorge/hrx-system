// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/passes/math/legalize.h"

#include <string.h>

#include "loom/pass/pipeline.h"
#include "loom/pass/registry.h"
#include "loom/passes/math/patterns.h"
#include "loom/rewrite/greedy.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

static const loom_pass_option_def_t kMathLegalizeOptions[] = {
    {IREE_SVL("max-iterations"),
     IREE_SVL("Maximum number of worklist iterations.")},
};

enum {
  LOOM_MATH_LEGALIZE_STAT_OPS_REWRITTEN = 0,
};

static const loom_pass_statistic_def_t kMathLegalizeStatistics[] = {
    {IREE_SVL("ops-rewritten"),
     IREE_SVL("Number of math operations rewritten.")},
};

static const loom_pass_info_t loom_math_legalize_pass_info_storage = {
    .name = IREE_SVL("legalize-math"),
    .description = IREE_SVL("Rewrite semantic math ops to target-ready IR."),
    .kind = LOOM_PASS_FUNCTION,
    .option_defs = kMathLegalizeOptions,
    .option_count = IREE_ARRAYSIZE(kMathLegalizeOptions),
    .statistic_defs = kMathLegalizeStatistics,
    .statistic_count = IREE_ARRAYSIZE(kMathLegalizeStatistics),
};

const loom_pass_info_t* loom_math_legalize_pass_info(void) {
  return &loom_math_legalize_pass_info_storage;
}

typedef struct loom_math_legalize_options_t {
  // Maximum number of fixed-point iterations. Zero selects the default.
  uint32_t max_iterations;
} loom_math_legalize_options_t;

static iree_status_t loom_math_legalize_parse_option(void* user_data,
                                                     iree_string_view_t name,
                                                     iree_string_view_t value) {
  loom_math_legalize_options_t* options =
      (loom_math_legalize_options_t*)user_data;
  if (iree_string_view_equal(name, IREE_SV("max-iterations"))) {
    if (options->max_iterations != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "duplicate option 'max-iterations' for pass 'legalize-math'");
    }
    IREE_RETURN_IF_ERROR(loom_pass_option_parse_uint32(
        IREE_SV("legalize-math"), name, value, &options->max_iterations));
    if (options->max_iterations == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pass 'legalize-math' option 'max-iterations' "
                              "must be greater than 0");
    }
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown option '%.*s' for pass 'legalize-math'",
                          (int)name.size, name.data);
}

iree_status_t loom_math_legalize_create(loom_pass_t* pass,
                                        iree_string_view_t options_string) {
  loom_math_legalize_options_t* options = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(pass->instance_arena,
                                           sizeof(*options), (void**)&options));
  memset(options, 0, sizeof(*options));
  if (pass->decoded_options) {
    for (uint16_t i = 0; i < pass->decoded_options->option_count; ++i) {
      const loom_pass_decoded_option_t* option =
          &pass->decoded_options->options[i];
      if (!option->present) {
        continue;
      }
      if (iree_string_view_equal(option->schema->name,
                                 IREE_SV("max-iterations"))) {
        options->max_iterations = option->uint32_value;
        continue;
      }
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unknown decoded option '%.*s' for pass 'legalize-math'",
          (int)option->schema->name.size, option->schema->name.data);
    }
  } else {
    IREE_RETURN_IF_ERROR(
        loom_pass_options_parse(pass->info->name, options_string,
                                (loom_pass_option_parse_callback_t){
                                    .fn = loom_math_legalize_parse_option,
                                    .user_data = options,
                                }));
  }
  pass->state = options;
  return iree_ok_status();
}

iree_status_t loom_math_legalize_run(loom_pass_t* pass, loom_module_t* module,
                                     loom_func_like_t function) {
  const loom_math_legalize_options_t* options =
      (const loom_math_legalize_options_t*)pass->state;

  loom_math_legalize_pattern_table_t pattern_table = {0};
  IREE_RETURN_IF_ERROR(
      loom_math_legalize_collect_patterns(pass->arena, &pattern_table));

  loom_greedy_rewrite_driver_t driver;
  loom_greedy_rewrite_driver_initialize(module, pass->arena, pass->value_facts,
                                        &driver);
  loom_greedy_rewrite_options_t rewrite_options = {
      .max_iterations = options ? options->max_iterations : 0,
  };
  loom_greedy_rewrite_result_t result = {0};
  iree_status_t status = loom_greedy_rewrite_run_patterns(
      &driver, function, pattern_table.patterns, pattern_table.pattern_count,
      &rewrite_options, &result);
  loom_greedy_rewrite_driver_deinitialize(&driver);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  if (result.changed) {
    loom_pass_mark_changed(pass);
  }
  if (pass->statistics) {
    loom_pass_statistic_add(pass, LOOM_MATH_LEGALIZE_STAT_OPS_REWRITTEN,
                            result.ops_modified);
  }
  return iree_ok_status();
}
