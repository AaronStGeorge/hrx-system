// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Opt-in registry for built-in LLVM target profiles.
//
// The target implementing this header intentionally pulls in the built-in
// target providers selected for Loom developer tools. JIT and AOT embeddings
// that want a smaller target surface should depend on specific provider
// packages instead.

#ifndef LOOM_TARGET_LLVMIR_TARGET_PRESETS_H_
#define LOOM_TARGET_LLVMIR_TARGET_PRESETS_H_

#include "loom/target/emit/llvmir/target_env.h"

#ifdef __cplusplus
extern "C" {
#endif

// Looks up a built-in target profile by name. Empty names resolve to the
// default host object profile supplied by the x86 provider.
iree_status_t loom_llvmir_target_profile_lookup(
    iree_string_view_t profile_name,
    const loom_llvmir_target_profile_t** out_profile);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_LLVMIR_TARGET_PRESETS_H_
