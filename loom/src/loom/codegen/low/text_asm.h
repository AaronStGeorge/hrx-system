// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Descriptor-backed implementation of the parser-facing low asm environment.

#ifndef LOOM_CODEGEN_LOW_TEXT_ASM_H_
#define LOOM_CODEGEN_LOW_TEXT_ASM_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/format/text/low_asm.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initializes |out_environment| to resolve asm packets through
// |descriptor_registry| and build canonical low dialect operations. The
// descriptor registry is borrowed and must outlive every parse using the
// returned environment.
void loom_low_descriptor_text_asm_environment_initialize(
    const loom_low_descriptor_registry_t* descriptor_registry,
    loom_text_low_asm_environment_t* out_environment);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_TEXT_ASM_H_
