// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Native AMDGPU kernel ABI projection for HSACO emission.
//
// This layer describes raw kernarg layouts that are not the normal Loom HAL
// dispatch ABI. It is intended for runtime builtin kernels and future
// custom-direct native kernels whose host launch path already owns a concrete
// kernarg struct.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_NATIVE_KERNEL_ABI_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_NATIVE_KERNEL_ABI_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/target/emit/native/amdgpu/hsaco.h"
#include "loom/target/emit/native/amdgpu/metadata.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_native_kernel_abi_argument_t {
  // Source-level or ABI-level argument name, if one is available.
  iree_string_view_t name;
  // Byte offset in the kernel kernarg segment.
  uint32_t offset;
  // Byte length of the argument storage.
  uint32_t size;
  // Byte alignment when known, or 0 to omit the field.
  uint32_t alignment;
  // AMDGPU metadata `.value_kind` classification.
  loom_amdgpu_metadata_argument_kind_t kind;
  // AMDGPU address-space string such as `global`, or empty to omit.
  iree_string_view_t address_space;
  // Source access qualifier string such as `read_only`, or empty to omit.
  iree_string_view_t access;
  // Effective access qualifier string such as `read_only`, or empty to omit.
  iree_string_view_t actual_access;
} loom_amdgpu_native_kernel_abi_argument_t;

typedef struct loom_amdgpu_native_kernel_abi_t {
  // Kernel kernarg segment size in bytes.
  uint32_t kernarg_segment_size;
  // Kernel kernarg segment alignment in bytes.
  uint32_t kernarg_segment_alignment;
  // Descriptor-only flags requested by the native kernel entry.
  loom_amdgpu_kernel_descriptor_flags_t descriptor_flags;
  // Argument records in kernarg offset order.
  const loom_amdgpu_native_kernel_abi_argument_t* arguments;
  // Number of records in |arguments|.
  iree_host_size_t argument_count;
} loom_amdgpu_native_kernel_abi_t;

// Applies |abi| to |kernel| for native HSACO emission.
//
// The function sets |kernel->metadata| kernarg fields and descriptor options
// from |abi| and clones argument records into |arena|. String views inside
// argument records are borrowed from the caller and must remain live until the
// HSACO write using |kernel| completes.
iree_status_t loom_amdgpu_native_kernel_abi_apply_to_hsaco_kernel(
    const loom_amdgpu_native_kernel_abi_t* abi,
    loom_amdgpu_hsaco_kernel_t* kernel, iree_arena_allocator_t* arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_NATIVE_KERNEL_ABI_H_
