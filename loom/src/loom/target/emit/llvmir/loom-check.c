// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-check runner for LLVMIR target-owned .loom-test files.

#include "iree/base/api.h"
#include "loom/target/emit/llvmir/loom_check.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/tools/loom-check/main.h"

static iree_status_t loom_llvmir_loom_check_initialize_low_descriptor_registry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  (void)user_data;
  IREE_ASSERT_ARGUMENT(out_registry);
  loom_target_low_descriptor_registry_initialize_from_tables(out_registry, NULL,
                                                             0, NULL, 0);
  return iree_ok_status();
}

static const loom_check_emit_provider_t* const kLoomCheckLlvmirEmitProviders[] =
    {
        &loom_llvmir_loom_check_emit_provider,
};

static const loom_check_requirement_provider_t* const
    kLoomCheckLlvmirRequirementProviders[] = {
        &loom_llvmir_loom_check_requirement_provider,
};

static const loom_check_environment_t kLoomCheckLlvmirEnvironment = {
    .register_context =
        {
            .fn = loom_check_register_production_context,
            .user_data = NULL,
        },
    .initialize_low_descriptor_registry =
        {
            .fn = loom_llvmir_loom_check_initialize_low_descriptor_registry,
            .user_data = NULL,
        },
    .emit_providers =
        {
            .providers = kLoomCheckLlvmirEmitProviders,
            .provider_count = IREE_ARRAYSIZE(kLoomCheckLlvmirEmitProviders),
        },
    .requirement_providers =
        {
            .providers = kLoomCheckLlvmirRequirementProviders,
            .provider_count =
                IREE_ARRAYSIZE(kLoomCheckLlvmirRequirementProviders),
        },
    .iree_run_loom_path = IREE_SVL(""),
};

int main(int argc, char** argv) {
  return loom_check_main(argc, argv, &kLoomCheckLlvmirEnvironment);
}
