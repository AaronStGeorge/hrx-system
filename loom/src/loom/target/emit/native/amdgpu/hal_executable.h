// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU HAL executable container packaging.
//
// This layer wraps a complete AMDGPU HSACO image in the IREE HAL AMDGPU
// ExecutableDef FlatBuffer envelope consumed by the production AMDGPU HAL
// executable loader.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_HAL_EXECUTABLE_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_HAL_EXECUTABLE_H_

#include "iree/base/api.h"
#include "loom/target/emit/native/amdgpu/hal_kernel_metadata.h"

#ifdef __cplusplus
extern "C" {
#endif

// Allocator-owned IREE HAL AMDGPU executable container bytes.
typedef struct loom_amdgpu_hal_executable_t {
  // Allocator-owned executable format string used to select the HAL loader.
  iree_string_view_t executable_format;
  // Allocator-owned header-prefixed AMDGPU ExecutableDef contents.
  uint8_t* data;
  // Number of bytes in |data|.
  iree_host_size_t data_length;
} loom_amdgpu_hal_executable_t;

// Releases storage owned by |executable|. Safe to call on a zero-initialized
// executable object.
void loom_amdgpu_hal_executable_deinitialize(
    loom_amdgpu_hal_executable_t* executable, iree_allocator_t allocator);

// Packages a header-prefixed IREE HAL AMDGPU executable container.
//
// |isa| is the canonical executable format string, such as
// `amdgcn-amd-amdhsa--gfx1100`. |hsaco| must contain a complete AMDGPU HSA
// code-object ELF image. |exports| must describe the entry points in canonical
// executable entry-point order.
iree_status_t loom_amdgpu_package_hal_executable(
    iree_string_view_t isa, iree_const_byte_span_t hsaco,
    const loom_amdgpu_hal_kernel_export_t* exports,
    iree_host_size_t export_count, iree_allocator_t allocator,
    loom_amdgpu_hal_executable_t* out_executable);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_HAL_EXECUTABLE_H_
