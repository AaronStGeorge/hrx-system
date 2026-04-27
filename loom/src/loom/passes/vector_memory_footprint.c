// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/passes/vector_memory_footprint.h"

#include <inttypes.h>

#include "loom/analysis/vector_memory_footprint.h"

enum {
  LOOM_VECTOR_MEMORY_FOOTPRINT_STAT_OPS_CHECKED = 0,
  LOOM_VECTOR_MEMORY_FOOTPRINT_STAT_OPS_SKIPPED = 1,
};

static const loom_pass_statistic_def_t kVectorMemoryFootprintStatistics[] = {
    {IREE_SVL("ops-checked"), IREE_SVL("Number of memory ops checked.")},
    {IREE_SVL("ops-skipped"),
     IREE_SVL("Number of memory ops skipped because no lane accesses memory.")},
};

static const loom_pass_info_t loom_vector_memory_footprint_pass_info_storage = {
    .name = IREE_SVL("vector-memory-footprint"),
    .description = IREE_SVL(
        "Prove vector memory footprints are in bounds before lowering."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_defs = kVectorMemoryFootprintStatistics,
    .statistic_count = IREE_ARRAYSIZE(kVectorMemoryFootprintStatistics),
};

const loom_pass_info_t* loom_vector_memory_footprint_pass_info(void) {
  return &loom_vector_memory_footprint_pass_info_storage;
}

iree_status_t loom_vector_memory_footprint_run(loom_pass_t* pass,
                                               loom_module_t* module,
                                               loom_func_like_t function) {
  loom_vector_memory_footprint_result_t result = {0};
  const loom_vector_memory_footprint_options_t options = {
      .arena = pass->arena,
      .emitter = pass->diagnostic_emitter,
  };
  IREE_RETURN_IF_ERROR(loom_vector_memory_footprint_verify_function(
      module, function, &options, &result));
  if (pass->statistics) {
    loom_pass_statistic_add(pass, LOOM_VECTOR_MEMORY_FOOTPRINT_STAT_OPS_CHECKED,
                            result.checked_op_count);
    loom_pass_statistic_add(pass, LOOM_VECTOR_MEMORY_FOOTPRINT_STAT_OPS_SKIPPED,
                            result.skipped_op_count);
  }
  if (result.error_count != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "vector memory footprint proof failed with %" PRIu32 " error%s",
        result.error_count, result.error_count == 1 ? "" : "s");
  }
  return iree_ok_status();
}
