// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU descriptor-set-local register-class helpers.

#ifndef LOOM_TARGET_ARCH_AMDGPU_REGISTER_CLASS_H_
#define LOOM_TARGET_ARCH_AMDGPU_REGISTER_CLASS_H_

#include <stdint.h>

#include "loom/codegen/low/descriptors.h"
#include "loom/target/arch/amdgpu/gfx950_descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |reg_class_id| names the accumulator register file in
// |descriptor_set|. Register-class IDs are descriptor-set-local; GFX11/GFX12
// use the same numeric ID for M0 that GFX950 uses for AGPR.
static inline bool loom_amdgpu_register_class_is_agpr(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t reg_class_id) {
  return descriptor_set != NULL &&
         descriptor_set->stable_id == AMDGPU_GFX950_CORE_DESCRIPTOR_SET_ID &&
         reg_class_id == AMDGPU_GFX950_CORE_REG_CLASS_ID_AGPR;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_REGISTER_CLASS_H_
