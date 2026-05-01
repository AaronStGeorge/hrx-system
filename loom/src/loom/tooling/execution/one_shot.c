// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/one_shot.h"

void loom_run_one_shot_options_initialize(
    loom_run_one_shot_options_t* out_options) {
  *out_options = (loom_run_one_shot_options_t){0};
  out_options->hal_workgroup_count[0] = 1;
  out_options->hal_workgroup_count[1] = 1;
  out_options->hal_workgroup_count[2] = 1;
}

void loom_run_one_shot_result_initialize(
    iree_allocator_t allocator, loom_run_one_shot_result_t* out_result) {
  *out_result = (loom_run_one_shot_result_t){0};
  iree_string_builder_initialize(allocator, &out_result->output);
}

void loom_run_one_shot_result_deinitialize(loom_run_one_shot_result_t* result) {
  if (result == NULL) {
    return;
  }
  iree_string_builder_deinitialize(&result->output);
  *result = (loom_run_one_shot_result_t){0};
}
