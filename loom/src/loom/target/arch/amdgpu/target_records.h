// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU target record rows.

#ifndef LOOM_TARGET_ARCH_AMDGPU_TARGET_RECORDS_H_
#define LOOM_TARGET_ARCH_AMDGPU_TARGET_RECORDS_H_

#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const loom_target_bundle_table_t loom_amdgpu_target_bundles;

// Returns the target bundle selected by an AMDGPU descriptor-set stable ID, or
// NULL when no target record is supported for that descriptor set.
const loom_target_bundle_t* loom_amdgpu_target_bundle_for_descriptor_set(
    uint64_t descriptor_set_stable_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_TARGET_RECORDS_H_
