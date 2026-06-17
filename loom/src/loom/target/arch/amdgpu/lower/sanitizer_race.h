// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU sanitizer race-observation legality.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_H_

#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

// Verifies sanitizer.race.access legality for AMDGPU target-low selection.
iree_status_t loom_amdgpu_low_legality_verify_sanitizer_race_access(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Verifies sanitizer.race.sync legality for AMDGPU target-low selection.
iree_status_t loom_amdgpu_low_legality_verify_sanitizer_race_sync(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_H_
