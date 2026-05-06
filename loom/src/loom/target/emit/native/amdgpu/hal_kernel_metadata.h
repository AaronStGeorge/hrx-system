// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU HAL kernel-library metadata shared by native artifacts and packages.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_HAL_KERNEL_METADATA_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_HAL_KERNEL_METADATA_H_

#include "iree/base/api.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t loom_amdgpu_hal_kernel_binding_flags_t;

enum {
  // Binding is read-only for the duration of a dispatch.
  LOOM_AMDGPU_HAL_KERNEL_BINDING_READ_ONLY = UINT64_C(1) << 0,
  // Binding points at an indirection table consumed by the kernel.
  LOOM_AMDGPU_HAL_KERNEL_BINDING_INDIRECT = UINT64_C(1) << 1,
};

// One exported HAL entry point in an AMDGPU kernel library.
typedef struct loom_amdgpu_hal_kernel_export_t {
  // HSA-visible kernel descriptor symbol name, including the `.kd` suffix.
  iree_string_view_t symbol_name;
  // Fixed workgroup size for the exported kernel.
  loom_target_workgroup_size_t workgroup_size;
  // Number of 32-bit push constants consumed by the kernel ABI.
  uint32_t constant_count;
  // Per-binding behavior flags in HAL binding ordinal order.
  const loom_amdgpu_hal_kernel_binding_flags_t* binding_flags;
  // Number of entries in |binding_flags|.
  iree_host_size_t binding_count;
} loom_amdgpu_hal_kernel_export_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_HAL_KERNEL_METADATA_H_
