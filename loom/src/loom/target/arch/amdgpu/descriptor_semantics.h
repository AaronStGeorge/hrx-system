// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared AMDGPU descriptor semantic predicates.
//
// AMDGPU target-low descriptors carry compact target facts generated from the
// Python descriptor tables. This layer centralizes the backend predicates that
// need to be consistent across wait-state and wait-counter planning without
// making each planner duplicate semantic-tag strings or schedule-resource
// traversal.

#ifndef LOOM_TARGET_ARCH_AMDGPU_DESCRIPTOR_SEMANTICS_H_
#define LOOM_TARGET_ARCH_AMDGPU_DESCRIPTOR_SEMANTICS_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |descriptor| issues on a resource with |kind|.
bool loom_amdgpu_descriptor_uses_resource_kind(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_low_resource_kind_t kind);

// Returns true when |descriptor| issues on the vector ALU.
bool loom_amdgpu_descriptor_uses_vector_alu(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

// Returns true when |descriptor| is a transcendental VALU packet.
bool loom_amdgpu_descriptor_is_transcendental(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_DESCRIPTOR_SEMANTICS_H_
