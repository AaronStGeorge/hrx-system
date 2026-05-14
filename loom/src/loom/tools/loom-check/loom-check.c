// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-check binary with all build-enabled target providers linked in.

#include <stddef.h>

#include "loom/tools/loom-check/provider.h"

#ifndef LOOM_CHECK_HAVE_AMDGPU
#define LOOM_CHECK_HAVE_AMDGPU 0
#endif  // LOOM_CHECK_HAVE_AMDGPU
#ifndef LOOM_CHECK_HAVE_IREEVM
#define LOOM_CHECK_HAVE_IREEVM 0
#endif  // LOOM_CHECK_HAVE_IREEVM
#ifndef LOOM_CHECK_HAVE_LLVMIR
#define LOOM_CHECK_HAVE_LLVMIR 0
#endif  // LOOM_CHECK_HAVE_LLVMIR
#ifndef LOOM_CHECK_HAVE_SPIRV
#define LOOM_CHECK_HAVE_SPIRV 0
#endif  // LOOM_CHECK_HAVE_SPIRV
#ifndef LOOM_CHECK_HAVE_WASM
#define LOOM_CHECK_HAVE_WASM 0
#endif  // LOOM_CHECK_HAVE_WASM
#ifndef LOOM_CHECK_HAVE_X86
#define LOOM_CHECK_HAVE_X86 0
#endif  // LOOM_CHECK_HAVE_X86

#define LOOM_CHECK_HAVE_ANY_PROVIDER                                          \
  (LOOM_CHECK_HAVE_AMDGPU || LOOM_CHECK_HAVE_IREEVM ||                        \
   LOOM_CHECK_HAVE_LLVMIR || LOOM_CHECK_HAVE_SPIRV || LOOM_CHECK_HAVE_WASM || \
   LOOM_CHECK_HAVE_X86)

#if LOOM_CHECK_HAVE_AMDGPU
#include "loom/target/arch/amdgpu/check/provider.h"
#endif  // LOOM_CHECK_HAVE_AMDGPU
#if LOOM_CHECK_HAVE_IREEVM
#include "loom/target/arch/ireevm/check/provider.h"
#endif  // LOOM_CHECK_HAVE_IREEVM
#if LOOM_CHECK_HAVE_LLVMIR
#include "loom/target/emit/llvmir/check/provider.h"
#endif  // LOOM_CHECK_HAVE_LLVMIR
#if LOOM_CHECK_HAVE_SPIRV
#include "loom/target/arch/spirv/check/provider.h"
#endif  // LOOM_CHECK_HAVE_SPIRV
#if LOOM_CHECK_HAVE_WASM
#include "loom/target/arch/wasm/check/provider.h"
#endif  // LOOM_CHECK_HAVE_WASM
#if LOOM_CHECK_HAVE_X86
#include "loom/target/arch/x86/check/provider.h"
#endif  // LOOM_CHECK_HAVE_X86

#if LOOM_CHECK_HAVE_ANY_PROVIDER
static const loom_check_provider_t* const kLoomCheckProviders[] = {
#if LOOM_CHECK_HAVE_AMDGPU
    &loom_amdgpu_check_provider,
#endif  // LOOM_CHECK_HAVE_AMDGPU
#if LOOM_CHECK_HAVE_IREEVM
    &loom_ireevm_check_provider,
#endif  // LOOM_CHECK_HAVE_IREEVM
#if LOOM_CHECK_HAVE_LLVMIR
    &loom_llvmir_check_provider,
#endif  // LOOM_CHECK_HAVE_LLVMIR
#if LOOM_CHECK_HAVE_SPIRV
    &loom_spirv_check_provider,
#endif  // LOOM_CHECK_HAVE_SPIRV
#if LOOM_CHECK_HAVE_WASM
    &loom_wasm_check_provider,
#endif  // LOOM_CHECK_HAVE_WASM
#if LOOM_CHECK_HAVE_X86
    &loom_x86_check_provider,
#endif  // LOOM_CHECK_HAVE_X86
};
#endif  // LOOM_CHECK_HAVE_ANY_PROVIDER

static const loom_check_provider_set_t kLoomCheckProviderSet = {
#if LOOM_CHECK_HAVE_ANY_PROVIDER
    .providers = kLoomCheckProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomCheckProviders),
#else
    .providers = NULL,
    .provider_count = 0,
#endif  // LOOM_CHECK_HAVE_ANY_PROVIDER
};

int main(int argc, char** argv) {
  return loom_check_provider_main(argc, argv, &kLoomCheckProviderSet);
}
