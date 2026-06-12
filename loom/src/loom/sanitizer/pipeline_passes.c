// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/sanitizer/pipeline_passes.h"

static const loom_pass_info_t
    loom_sanitizer_insert_assertions_pass_info_storage = {
        .name = IREE_SVL("sanitizer-insert-assertions"),
        .description = IREE_SVL(
            "Insert semantic sanitizer assertions for enabled checks."),
        .kind = LOOM_PASS_FUNCTION,
};

const loom_pass_info_t* loom_sanitizer_insert_assertions_pass_info(void) {
  return &loom_sanitizer_insert_assertions_pass_info_storage;
}

iree_status_t loom_sanitizer_insert_assertions_run(loom_pass_t* pass,
                                                   loom_module_t* module,
                                                   loom_func_like_t function) {
  (void)pass;
  (void)module;
  (void)function;
  return iree_ok_status();
}

static const loom_pass_info_t
    loom_sanitizer_materialize_assertions_pass_info_storage = {
        .name = IREE_SVL("sanitizer-materialize-assertions"),
        .description =
            IREE_SVL("Materialize sanitizer assertions for target reporting."),
        .kind = LOOM_PASS_FUNCTION,
};

const loom_pass_info_t* loom_sanitizer_materialize_assertions_pass_info(void) {
  return &loom_sanitizer_materialize_assertions_pass_info_storage;
}

iree_status_t loom_sanitizer_materialize_assertions_run(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function) {
  (void)pass;
  (void)module;
  (void)function;
  return iree_ok_status();
}
