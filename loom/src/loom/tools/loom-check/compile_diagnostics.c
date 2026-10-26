// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/compile_diagnostics.h"

#include "loom/testing/test_diagnostic_loomc.h"
#include "loom/tooling/cli/loomc_diagnostic_json.h"

iree_status_t loom_check_compile_append_result(
    loom_check_diagnostic_collector_t* collector, const loomc_result_t* result,
    bool* out_succeeded) {
  *out_succeeded = loomc_result_succeeded(result);
  IREE_RETURN_IF_ERROR(
      loom_check_compile_append_result_diagnostics(collector, result));
  // An operation may reject input without a diagnostic. Empty annotation
  // matching must not turn that failed operation into a successful check.
  if (!*out_succeeded &&
      !loom_check_diagnostic_collector_has_error(collector)) {
    return iree_make_status(
        (iree_status_code_t)loomc_result_status_code(result),
        "compiler operation failed without an error diagnostic");
  }
  return iree_ok_status();
}

iree_status_t loom_check_compile_append_result_diagnostics(
    loom_check_diagnostic_collector_t* collector,
    const loomc_result_t* source_result) {
  const loomc_host_size_t diagnostic_count =
      loomc_result_diagnostic_count(source_result);
  for (loomc_host_size_t i = 0; i < diagnostic_count; ++i) {
    const loomc_diagnostic_t* source =
        loomc_result_diagnostic_at(source_result, i);
    loom_check_collected_diagnostic_t target = {0};
    IREE_RETURN_IF_ERROR(loom_test_diagnostic_materialize_loomc(
        source, collector->arena, &target));
    loom_output_stream_t diagnostic_stream;
    IREE_RETURN_IF_ERROR(loom_json_value_list_begin_value(
        &collector->result->diagnostics, &diagnostic_stream));
    IREE_RETURN_IF_ERROR(loom_tooling_cli_write_loomc_diagnostic_json(
        &diagnostic_stream, source));
    IREE_RETURN_IF_ERROR(
        loom_check_diagnostic_collector_append(collector, &target));
  }
  return iree_ok_status();
}
