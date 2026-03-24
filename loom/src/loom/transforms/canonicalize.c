// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/canonicalize.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/transforms/rewriter.h"

//===----------------------------------------------------------------------===//
// Constant materialization
//===----------------------------------------------------------------------===//

// Materializes a scalar.constant from exact facts.
static iree_status_t loom_canonicalize_materialize_constant(
    loom_builder_t* builder, loom_value_facts_t facts, loom_type_t result_type,
    loom_location_id_t location, loom_value_id_t* out_value_id) {
  loom_attribute_t attr;
  if (loom_value_facts_is_float(facts)) {
    attr = loom_attr_f64(loom_value_facts_as_f64(facts));
  } else {
    attr = loom_attr_i64(facts.range_lo);
  }
  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(builder, attr, result_type,
                                                  location, &constant_op));
  *out_value_id = loom_scalar_constant_result(constant_op);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

static const loom_pass_option_def_t kCanonicalizeOptions[] = {
    {IREE_SVL("max-iterations"),
     IREE_SVL("Maximum number of worklist iterations.")},
};

enum {
  LOOM_CANONICALIZE_STAT_OPS_MODIFIED = 0,
};

static const loom_pass_statistic_def_t kCanonicalizeStatistics[] = {
    {IREE_SVL("ops-modified"),
     IREE_SVL("Number of ops simplified by canonicalization.")},
};

const loom_pass_info_t loom_canonicalize_pass_info = {
    .name = IREE_SVL("canonicalize"),
    .description = IREE_SVL("Apply op-specific canonicalization patterns."),
    .kind = LOOM_PASS_FUNCTION,
    .option_defs = kCanonicalizeOptions,
    .option_count = 1,
    .statistic_defs = kCanonicalizeStatistics,
    .statistic_count = 1,
};

//===----------------------------------------------------------------------===//
// Implementation
//===----------------------------------------------------------------------===//

#define LOOM_CANONICALIZE_DEFAULT_MAX_ITERATIONS 10

iree_status_t loom_canonicalize_run(loom_pass_t* pass, loom_module_t* module,
                                    loom_func_like_t function) {
  if (!loom_func_like_body(function)) return iree_ok_status();

  uint32_t max_iterations = LOOM_CANONICALIZE_DEFAULT_MAX_ITERATIONS;

  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));
  rewriter.materialize_constant = loom_canonicalize_materialize_constant;
  iree_status_t status = loom_rewriter_enable_analysis(&rewriter, function);

  for (uint32_t iteration = 0;
       iree_status_is_ok(status) && iteration < max_iterations; ++iteration) {
    status = loom_rewriter_seed_function(&rewriter, function);
    if (!iree_status_is_ok(status)) break;
    bool any_changed = false;

    loom_op_t* op = NULL;
    while ((op = loom_rewriter_pop(&rewriter)) != NULL) {
      // Mini-DCE: erase trivially dead ops before fold/canonicalize.
      bool erased = false;
      iree_status_t dce_status =
          loom_rewriter_erase_if_dead(&rewriter, op, &erased);
      if (!iree_status_is_ok(dce_status)) {
        loom_rewriter_deinitialize(&rewriter);
        return dce_status;
      }
      if (erased) {
        any_changed = true;
        continue;
      }

      // Try fold: constant fold via facts and replace with constants.
      bool folded = false;
      iree_status_t status = loom_rewriter_try_fold(&rewriter, op, &folded);
      if (!iree_status_is_ok(status)) {
        loom_rewriter_deinitialize(&rewriter);
        return status;
      }
      if (folded) {
        any_changed = true;
        if (pass->statistics) {
          loom_pass_statistic_add(pass, LOOM_CANONICALIZE_STAT_OPS_MODIFIED, 1);
        }
        continue;
      }

      // Structural canonicalization patterns.
      const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
      if (!vtable || !vtable->canonicalize) continue;

      rewriter.flags = 0;
      status = vtable->canonicalize(op, &rewriter);
      if (!iree_status_is_ok(status)) {
        loom_rewriter_deinitialize(&rewriter);
        return status;
      }
      if (rewriter.flags & LOOM_REWRITER_FLAG_CHANGED) {
        any_changed = true;
        if (pass->statistics) {
          loom_pass_statistic_add(pass, LOOM_CANONICALIZE_STAT_OPS_MODIFIED, 1);
        }
      }
    }

    if (!any_changed) break;
  }

  loom_rewriter_deinitialize(&rewriter);
  return status;
}
