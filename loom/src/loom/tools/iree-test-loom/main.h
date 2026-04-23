// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared iree-test-loom command-line implementation.

#ifndef LOOM_TOOLS_IREE_TEST_LOOM_MAIN_H_
#define LOOM_TOOLS_IREE_TEST_LOOM_MAIN_H_

#include "iree/base/api.h"
#include "loom/tooling/execution/session.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_test_loom_configuration_t {
  // Null-terminated executable name used in help and diagnostics.
  const char* tool_name;
  // Target-low descriptor registry package linked into this runner.
  loom_run_initialize_low_descriptor_registry_callback_t
      initialize_low_descriptor_registry;
} iree_test_loom_configuration_t;

// Runs the configured iree-test-loom command-line tool.
int iree_test_loom_main(int argc, char** argv,
                        const iree_test_loom_configuration_t* configuration);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_TEST_LOOM_MAIN_H_
