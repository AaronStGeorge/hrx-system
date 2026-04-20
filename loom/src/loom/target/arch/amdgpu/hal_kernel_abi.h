// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU HAL-kernel ABI layout over target-low resources.
//
// This layer derives kernarg storage from symbol-backed low.abi.resource
// records. It intentionally stays below LLVMIR/native artifact emission so the
// same resource ABI can feed the temporary assembly path, direct HSACO writing,
// and future backends.

#ifndef LOOM_TARGET_ARCH_AMDGPU_HAL_KERNEL_ABI_H_
#define LOOM_TARGET_ARCH_AMDGPU_HAL_KERNEL_ABI_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Kernarg storage for one HAL binding pointer.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_SIZE 8u

// Required kernarg alignment for HAL binding pointers.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_ALIGNMENT 8u

// Stable low.live_in source spelling for the AMDGPU kernarg segment pointer.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_KERNARG_SEGMENT_PTR_SOURCE \
  "amdgpu.kernarg_segment_ptr"

typedef struct loom_amdgpu_hal_kernarg_resource_t {
  // Defining low.abi.resource op for diagnostics and cross-checks.
  const loom_op_t* resource_op;
  // Module-local symbol defining the resource record.
  loom_symbol_ref_t symbol;
  // Borrowed resource symbol name without the leading '@'.
  iree_string_view_t name;
  // HAL binding ordinal used by the runtime dispatch path.
  uint32_t binding_index;
  // Byte offset of the pointer entry in the kernarg segment.
  uint32_t kernarg_offset;
  // Byte length of the pointer entry in the kernarg segment.
  uint32_t kernarg_size;
  // Byte alignment of the pointer entry in the kernarg segment.
  uint32_t kernarg_alignment;
  // Source/storage semantic type declared by the resource record.
  loom_type_t semantic_type;
  // Target-low value type produced by low.resource for this binding.
  loom_type_t abi_type;
} loom_amdgpu_hal_kernarg_resource_t;

typedef struct loom_amdgpu_hal_kernel_abi_layout_t {
  // low.func.def operation whose resources are laid out.
  const loom_op_t* function_op;
  // Total kernarg segment size in bytes.
  uint32_t kernarg_segment_size;
  // Required kernarg segment alignment in bytes.
  uint32_t kernarg_segment_alignment;
  // True when the kernel descriptor must request the kernarg segment pointer.
  bool uses_kernarg_segment_ptr;
  // Resource records in HAL binding/kernarg offset order.
  const loom_amdgpu_hal_kernarg_resource_t* resources;
  // Number of resource records in |resources|.
  iree_host_size_t resource_count;
} loom_amdgpu_hal_kernel_abi_layout_t;

// Derives the AMDGPU HAL-kernel ABI layout for |function_op|.
//
// v0 supports only low.abi.resource records with kind hal_buffer_resource,
// dense unique binding indexes starting at zero, semantic type hal.buffer, and
// ABI type reg<amdgpu.sgpr x4>. The kernarg segment stores one 64-bit global
// pointer per binding in binding-index order; later lowering materializes the
// target resource descriptor value consumed by low.resource.
iree_status_t loom_amdgpu_hal_kernel_abi_layout_from_low(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_target_bundle_t* target_bundle,
    loom_amdgpu_hal_kernel_abi_layout_t* out_layout,
    iree_arena_allocator_t* arena);

// Returns true if |value_id| is defined by the kernarg segment pointer live-in.
bool loom_amdgpu_hal_kernel_abi_is_kernarg_segment_ptr_live_in(
    const loom_module_t* module, loom_value_id_t value_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_HAL_KERNEL_ABI_H_
