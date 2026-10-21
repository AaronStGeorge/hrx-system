// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured AMDGPU profile representation shared with exact-build consumers.
// Profile construction and the family descriptor are declared in profile.h.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PROFILE_TYPES_H_
#define LOOM_TARGET_ARCH_AMDGPU_PROFILE_TYPES_H_

#include "loom/target/arch/amdgpu/target_identity.h"
#include "loom/target/profile.h"

// Structured external AMDGPU target refinement profile.
typedef struct loom_amdgpu_target_profile_t {
  // Target-neutral family identity and bundle projection.
  loom_target_profile_t base;

  // Structured target and AMDHSA feature identity.
  loom_amdgpu_target_identity_t identity;
} loom_amdgpu_target_profile_t;

#endif  // LOOM_TARGET_ARCH_AMDGPU_PROFILE_TYPES_H_
