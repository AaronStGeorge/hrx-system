// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/one_shot.h"

void loom_run_one_shot_options_initialize(
    loom_run_one_shot_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  *out_options = (loom_run_one_shot_options_t){0};
  loom_run_vm_invocation_options_initialize(&out_options->vm_options);
  loom_run_hal_invocation_options_initialize(&out_options->hal_options);
}

void loom_run_one_shot_result_initialize(
    iree_allocator_t allocator, loom_run_one_shot_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(out_result);
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
