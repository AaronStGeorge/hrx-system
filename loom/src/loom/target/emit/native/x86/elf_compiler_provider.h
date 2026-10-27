// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// x86 ELF compiler capability composition.

#ifndef LOOM_TARGET_EMIT_NATIVE_X86_ELF_COMPILER_PROVIDER_H_
#define LOOM_TARGET_EMIT_NATIVE_X86_ELF_COMPILER_PROVIDER_H_

#include "loom/target/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// x86 ELF object, shared-library, and HAL-library emission contribution.
extern const loom_target_provider_t loom_x86_elf_compiler_provider;

// Complete x86 ELF compiler capability containing target identity and ELF
// artifact emission.
extern const loom_target_provider_set_t loom_x86_elf_compiler_provider_set;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_X86_ELF_COMPILER_PROVIDER_H_
