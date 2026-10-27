// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P ELF compiler capability composition.

#ifndef LOOM_TARGET_EMIT_NATIVE_AIE2P_ELF_COMPILER_PROVIDER_H_
#define LOOM_TARGET_EMIT_NATIVE_AIE2P_ELF_COMPILER_PROVIDER_H_

#include "loom/target/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Canonical XDNA ELF emission for AIE2P target environments.
extern const loom_target_provider_t loom_aie2p_xdna_artifact_provider;

// Complete AIE2P compiler capability containing target identity and XDNA ELF
// artifact emission.
extern const loom_target_provider_set_t loom_aie2p_compiler_provider_set;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AIE2P_ELF_COMPILER_PROVIDER_H_
