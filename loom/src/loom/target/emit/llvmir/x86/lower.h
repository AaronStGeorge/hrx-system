// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// LLVMIR lowering provider for x86 target-specific vector contracts.

#ifndef LOOM_TARGET_LLVMIR_X86_LOWER_H_
#define LOOM_TARGET_LLVMIR_X86_LOWER_H_

#include "loom/target/emit/llvmir/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the static x86 LLVMIR lowering provider.
const loom_llvmir_lowering_provider_t* loom_llvmir_x86_lowering_provider(void);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_LLVMIR_X86_LOWER_H_
