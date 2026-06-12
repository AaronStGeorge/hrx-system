// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Builder-level lookup for generated AMDGPU target-low descriptor references.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_DESCRIPTOR_REF_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_DESCRIPTOR_REF_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_builder_t loom_builder_t;

// Resolves |descriptor_ref| to a descriptor row and interns its opcode spelling
// in the builder module.
iree_status_t loom_amdgpu_lookup_descriptor_ref(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_low_descriptor_t** out_descriptor,
    loom_string_id_t* out_opcode_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_DESCRIPTOR_REF_H_
