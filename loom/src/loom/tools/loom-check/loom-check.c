// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Default loom-check binary for core .loom-test files.
//
// The stock tool accepts the production dialects, the synthetic test dialect,
// and the core/test target-low descriptor package. Backend-owned tests should
// use a backend runner binary so this target does not link every target table.

#include "iree/base/api.h"
#include "loom/ops/test/registry.h"
#include "loom/target/low_descriptor_registry_core_test.h"
#include "loom/tools/loom-check/main.h"

static iree_status_t loom_check_cli_register_context(void* user_data,
                                                     loom_context_t* context) {
  IREE_RETURN_IF_ERROR(
      loom_check_register_production_context(user_data, context));
  return loom_test_dialect_register(context);
}

static iree_status_t loom_check_cli_initialize_low_descriptor_registry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  (void)user_data;
  IREE_ASSERT_ARGUMENT(out_registry);
  loom_target_core_test_low_descriptor_registry_initialize(out_registry);
  return iree_ok_status();
}

static const loom_check_environment_t kLoomCheckEnvironment = {
    .register_context =
        {
            .fn = loom_check_cli_register_context,
            .user_data = NULL,
        },
    .initialize_low_descriptor_registry =
        {
            .fn = loom_check_cli_initialize_low_descriptor_registry,
            .user_data = NULL,
        },
};

int main(int argc, char** argv) {
  return loom_check_main(argc, argv, &kLoomCheckEnvironment);
}
