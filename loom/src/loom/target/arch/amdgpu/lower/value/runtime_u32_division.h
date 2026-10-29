// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_RUNTIME_U32_DIVISION_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_RUNTIME_U32_DIVISION_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Exact runtime-u32 division state prepared once for one divisor.
typedef struct loom_amdgpu_runtime_u32_division_t {
  // Selected SGPR or VGPR carrier for the computation.
  uint16_t register_class_id;
  // One-unit low register type shared by every arithmetic value.
  loom_type_t word_type;
  // Materialized nonzero divisor.
  loom_value_id_t divisor;
  // Materialized integer constant one used by quotient correction.
  loom_value_id_t one;
  // Refined fixed-point reciprocal used by quotient estimation.
  loom_value_id_t reciprocal;
} loom_amdgpu_runtime_u32_division_t;

// Returns whether exact runtime-u32 division can be emitted in
// |register_class_id|.
bool loom_amdgpu_runtime_u32_division_descriptors_supported(
    const loom_low_descriptor_set_t* descriptors, uint16_t register_class_id);

// Prepares one exact runtime-u32 division recipe for |divisor|.
iree_status_t loom_amdgpu_prepare_runtime_u32_division(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint16_t register_class_id, loom_value_id_t divisor, loom_type_t word_type,
    loom_amdgpu_runtime_u32_division_t* out_division);

// Emits exact quotient and remainder using a prepared runtime divisor.
iree_status_t loom_amdgpu_emit_runtime_u32_division(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_runtime_u32_division_t* division,
    loom_value_id_t dividend, loom_value_id_t* out_quotient,
    loom_value_id_t* out_remainder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_RUNTIME_U32_DIVISION_H_
