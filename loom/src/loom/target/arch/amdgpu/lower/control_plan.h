// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Stable target plan identities shared by AMDGPU branch planners. Target data
// remains owned by the component that creates each plan.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_CONTROL_PLAN_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_CONTROL_PLAN_H_

enum {
  LOOM_AMDGPU_BRANCH_PLAN_THEN_MASKED_REGION = 0,
  LOOM_AMDGPU_BRANCH_PLAN_IF_ELSE_DIAMOND = 1,
  LOOM_AMDGPU_BRANCH_PLAN_DIVERGENT_LOOP = 2,
  LOOM_AMDGPU_BRANCH_PLAN_UNIFORM = 3,
  LOOM_AMDGPU_BRANCH_PLAN_EXACT_FALSE = 4,
  LOOM_AMDGPU_BRANCH_PLAN_EXACT_TRUE = 5,
  LOOM_AMDGPU_BRANCH_PLAN_LOOP_EXIT_REGION = 6,
};

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_CONTROL_PLAN_H_
