// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_HSACO_KERNARG_LAYOUT_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_HSACO_KERNARG_LAYOUT_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/util/hsaco_metadata.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Native Kernarg Layout
//===----------------------------------------------------------------------===//

// Runtime-supported maximum native kernarg byte length.
//
// AMDGPU code-object kernarg segment offsets are represented as 16-bit byte
// offsets in the runtime dispatch layout. Code objects requiring larger
// kernarg packets must use a future layout revision.
#define IREE_HAL_AMDGPU_HSACO_KERNARG_LAYOUT_MAX_BYTE_LENGTH UINT16_MAX

// Layout flags for a dispatchable HSACO kernel.
typedef enum iree_hal_amdgpu_hsaco_kernarg_layout_flag_bits_e {
  IREE_HAL_AMDGPU_HSACO_KERNARG_LAYOUT_FLAG_NONE = 0u,
  // The kernel metadata declares a HIP/OpenCL implicit-argument suffix.
  IREE_HAL_AMDGPU_HSACO_KERNARG_LAYOUT_FLAG_IMPLICIT_ARGS = 1u << 0,
} iree_hal_amdgpu_hsaco_kernarg_layout_flag_bits_t;
typedef uint32_t iree_hal_amdgpu_hsaco_kernarg_layout_flags_t;

// HAL input stream consumed by a native kernarg parameter.
typedef uint8_t iree_hal_amdgpu_hsaco_kernarg_parameter_kind_t;
enum iree_hal_amdgpu_hsaco_kernarg_parameter_kind_e {
  IREE_HAL_AMDGPU_HSACO_KERNARG_PARAMETER_KIND_UNKNOWN = 0,
  // The parameter writes one resolved HAL binding device pointer.
  IREE_HAL_AMDGPU_HSACO_KERNARG_PARAMETER_KIND_BINDING = 1,
  // The parameter copies bytes from the HAL dispatch constant stream.
  IREE_HAL_AMDGPU_HSACO_KERNARG_PARAMETER_KIND_CONSTANT = 2,
};

// Mapping from one HAL input stream slice to one native kernarg slice.
typedef struct iree_hal_amdgpu_hsaco_kernarg_parameter_t {
  // Byte offset in the native kernel kernarg segment.
  uint16_t kernarg_offset;
  // Binding ordinal or dispatch-constant byte offset selected by |kind|.
  uint16_t source_offset;
  // Number of bytes written into the native kernarg segment.
  uint16_t byte_length;
  // One of iree_hal_amdgpu_hsaco_kernarg_parameter_kind_e.
  iree_hal_amdgpu_hsaco_kernarg_parameter_kind_t kind;
  // Unused. Must be 0.
  uint8_t reserved;
} iree_hal_amdgpu_hsaco_kernarg_parameter_t;
static_assert(sizeof(iree_hal_amdgpu_hsaco_kernarg_parameter_t) == 8,
              "keep kernarg parameter records compact");

// Native kernarg layout derived from AMDGPU code-object metadata.
typedef struct iree_hal_amdgpu_hsaco_kernarg_layout_t {
  // Layout feature flags.
  iree_hal_amdgpu_hsaco_kernarg_layout_flags_t flags;
  // Number of records required in the parameter layout table.
  uint16_t parameter_count;
  // Number of HAL bindings consumed by the layout.
  uint16_t binding_count;
  // Number of HAL dispatch constant bytes consumed by the layout.
  uint16_t constant_byte_length;
  // Native kernarg byte extent covered by visible HAL parameters.
  uint16_t explicit_kernarg_size;
  // Native kernarg byte offset of the implicit-argument suffix, or UINT16_MAX.
  uint16_t implicit_args_offset;
  // Total native kernarg byte length to reserve and zero before population.
  uint16_t total_kernarg_size;
  // Required native kernarg alignment in bytes.
  uint16_t kernarg_alignment;
  // Unused. Must be 0.
  uint16_t reserved;
} iree_hal_amdgpu_hsaco_kernarg_layout_t;

// Calculates the native dispatch layout required by |kernel|.
//
// Visible `global_buffer` arguments consume dense HAL binding ordinals in
// metadata order. Visible `by_value` arguments consume dense byte slices of the
// HAL dispatch constant stream in metadata order. Native kernarg offsets remain
// exactly as reported by metadata, so bindings and constants may be interleaved
// and padded in the compiled kernel ABI.
//
// Hidden HIP/OpenCL ABI arguments are accepted only as a suffix beginning after
// all visible arguments. The runtime populates the whole implicit-argument
// suffix when any hidden argument is present.
iree_status_t iree_hal_amdgpu_hsaco_kernarg_layout_calculate(
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel,
    iree_hal_amdgpu_hsaco_kernarg_layout_t* out_layout);

// Populates the native kernarg parameter table for |kernel|.
//
// |parameter_capacity| must be at least
// iree_hal_amdgpu_hsaco_kernarg_layout_t::parameter_count returned by
// iree_hal_amdgpu_hsaco_kernarg_layout_calculate. Records are emitted in
// metadata order and can be split into binding/constant loops by the dispatch
// hot path if a later descriptor format wants separate tables.
iree_status_t iree_hal_amdgpu_hsaco_kernarg_layout_populate_parameters(
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel,
    iree_host_size_t parameter_capacity,
    iree_hal_amdgpu_hsaco_kernarg_parameter_t* out_parameters);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_HSACO_KERNARG_LAYOUT_H_
