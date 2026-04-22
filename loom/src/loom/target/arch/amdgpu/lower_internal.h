// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Private AMDGPU source-to-target-low lowering helpers.
//
// This header is target-local glue between lower.c and focused leaf lowerers.
// Keep declarations here narrow: a new declaration should mean two AMDGPU
// lowering invariant clusters genuinely share one contract.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_INTERNAL_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_INTERNAL_H_

#include <stdint.h>

#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/lower.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/low/ops.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES 4u

// Returns a static rank-1 vector lane count for the requested element type, or
// zero when the type is not a supported static rank-1 vector.
uint32_t loom_amdgpu_static_vector_lane_count(loom_type_t type,
                                              loom_scalar_type_t element_type,
                                              uint32_t max_lane_count);

// Returns the i32 lane count for a supported AMDGPU 32-bit vector payload, or
// zero when the source type is not representable as that payload.
uint32_t loom_amdgpu_vector_i32_lane_count(loom_type_t type);

// Builds a one-unit VGPR register type in the current lowering context.
iree_status_t loom_amdgpu_make_vgpr_type(loom_low_lower_context_t* context,
                                         loom_type_t* out_type);

// Maps a source result to the low register type already selected by the active
// lowering policy and verifies that it is a register payload.
iree_status_t loom_amdgpu_low_result_type(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_value_id_t source_result,
                                          loom_type_t* out_low_type);

// Emits one descriptor-backed low.op with source provenance.
iree_status_t loom_amdgpu_emit_low_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t descriptor_id, const loom_value_id_t* operands,
    iree_host_size_t operand_count, loom_named_attr_slice_t attrs,
    const loom_type_t* result_types, iree_host_size_t result_count,
    loom_op_t** out_low_op);

// Emits a low.slice from a register range.
iree_status_t loom_amdgpu_emit_low_slice(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_value_id_t source,
                                         uint32_t lane_offset,
                                         loom_type_t result_type,
                                         loom_value_id_t* out_value);

// Returns true when the target bundle belongs to an AMDGPU contract set.
bool loom_amdgpu_low_legality_bundle_is_amdgpu(
    const loom_target_bundle_t* bundle);

// Returns true for source vector-dot ops handled by the AMDGPU dot lowerer.
bool loom_amdgpu_op_is_vector_dot(loom_op_kind_t kind);

// Returns true when a source vector.dot4i op can lower under the active AMDGPU
// descriptor set.
bool loom_amdgpu_can_lower_vector_dot4i(loom_low_lower_context_t* context,
                                        const loom_op_t* source_op);

// Lowers a source vector.dot4i op to AMDGPU descriptor-backed low packets.
iree_status_t loom_amdgpu_lower_vector_dot4i(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op);

// Verifies source vector-dot legality for AMDGPU target-low selection.
iree_status_t loom_amdgpu_low_legality_verify_vector_dot(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_INTERNAL_H_
