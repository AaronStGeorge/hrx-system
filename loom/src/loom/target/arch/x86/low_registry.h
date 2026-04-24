// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// x86 target-low registry package.

#ifndef LOOM_TARGET_ARCH_X86_LOW_REGISTRY_H_
#define LOOM_TARGET_ARCH_X86_LOW_REGISTRY_H_

#include "loom/target/low_descriptor_registry.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const loom_target_bundle_t loom_x86_low_target_bundle_avx512_core;
extern const loom_target_bundle_t loom_x86_low_target_bundle_packed_dot_core;
extern const loom_target_bundle_t
    loom_x86_low_target_bundle_avx512_packed_dot_core;

void loom_x86_low_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_LOW_REGISTRY_H_
