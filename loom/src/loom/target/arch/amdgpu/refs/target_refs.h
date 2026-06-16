// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU target-low descriptor reference constants and lookup helpers.

#ifndef LOOM_TARGET_ARCH_AMDGPU_REFS_TARGET_REFS_H_
#define LOOM_TARGET_ARCH_AMDGPU_REFS_TARGET_REFS_H_

#include <stdint.h>

#include "loom/codegen/low/descriptors.h"
#include "loom/target/arch/amdgpu/refs/target_refs_tables.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint16_t loom_amdgpu_descriptor_ref_t;

uint32_t loom_amdgpu_descriptor_ref_ordinal(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref);

const loom_low_descriptor_t* loom_amdgpu_descriptor_ref_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_REFS_TARGET_REFS_H_
