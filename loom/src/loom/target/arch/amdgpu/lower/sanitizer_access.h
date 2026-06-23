// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU target-low sanitizer access-check builders.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_ACCESS_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_ACCESS_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/attribute.h"
#include "loom/ir/location.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_builder_t loom_builder_t;

typedef struct loom_amdgpu_sanitizer_access_check_t {
  // Native lane mask identifying lanes that failed the access check.
  loom_value_id_t failure_mask;
  // Shadow address selected for reporting a failed access.
  loom_value_id_t shadow_address;
  // Shadow byte selected for reporting a failed access.
  loom_value_id_t shadow_value;
} loom_amdgpu_sanitizer_access_check_t;

// Emits an AMDGPU ASAN-style shadow check for one static access range.
//
// |fault_address| must be a 64-bit SGPR or VGPR register range. |access_size|
// must be non-zero and is packetized into <=8 byte chunks so every touched
// shadow byte is covered.
// |wavefront_size| must be 32 or 64 and controls how the returned SGPRx2
// |failure_mask| is canonicalized before it is consumed as an EXEC-width lane
// mask by loom_amdgpu_build_sanitizer_access_report_failure_mask_branch.
iree_status_t loom_amdgpu_build_sanitizer_access_check(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t asan_config_symbol, loom_value_id_t fault_address,
    uint32_t access_size, uint32_t wavefront_size, loom_location_id_t location,
    loom_amdgpu_sanitizer_access_check_t* out_check);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_ACCESS_H_
