// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-check binary with all build-enabled target providers linked in.

#include "loom/target/arch/wasm/check_provider.h"
#include "loom/target/arch/x86/check_provider.h"
#include "loom/target/emit/ireevm/check_provider.h"
#include "loom/target/emit/llvmir/check_provider.h"
#include "loom/tools/loom-check/main.h"

#ifndef LOOM_CHECK_HAVE_AMDGPU
#define LOOM_CHECK_HAVE_AMDGPU 0
#endif  // LOOM_CHECK_HAVE_AMDGPU

#if LOOM_CHECK_HAVE_AMDGPU
#include "loom/target/arch/amdgpu/check_provider.h"
#endif  // LOOM_CHECK_HAVE_AMDGPU

static const loom_check_provider_t* const kLoomCheckProviders[] = {
    &loom_ireevm_check_provider, &loom_wasm_check_provider,
    &loom_x86_check_provider,
#if LOOM_CHECK_HAVE_AMDGPU
    &loom_amdgpu_check_provider,
#endif  // LOOM_CHECK_HAVE_AMDGPU
    &loom_llvmir_check_provider,
};

static const loom_check_provider_set_t kLoomCheckProviderSet = {
    .providers = kLoomCheckProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomCheckProviders),
};

int main(int argc, char** argv) {
  return loom_check_provider_main(argc, argv, &kLoomCheckProviderSet);
}
