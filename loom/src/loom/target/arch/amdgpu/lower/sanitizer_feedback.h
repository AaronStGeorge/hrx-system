// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared AMDGPU sanitizer feedback lowering contracts.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_FEEDBACK_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_FEEDBACK_H_

#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_sanitizer_report_source_t {
  // Device-visible dispatch packet pointer captured for host diagnostics.
  loom_value_id_t dispatch_ptr;
  // X dimension workgroup id captured for host diagnostics.
  loom_value_id_t workgroup_id_x;
  // X dimension workitem id captured for host diagnostics.
  loom_value_id_t workitem_id_x;
} loom_amdgpu_sanitizer_report_source_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_FEEDBACK_H_
