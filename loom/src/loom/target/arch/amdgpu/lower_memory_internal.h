// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Private AMDGPU vector-memory lowering helpers.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_MEMORY_INTERNAL_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_MEMORY_INTERNAL_H_

#include <stdint.h>

#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/lower.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/facts.h"
#include "loom/ir/ir.h"
#include "loom/target/arch/amdgpu/lower_plan.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t loom_amdgpu_memory_access_rejection_flags_t;

#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_TYPE ((uint32_t)1u << 0)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE \
  ((uint32_t)1u << 1)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_SOURCE \
  ((uint32_t)1u << 2)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_STRIDE ((uint32_t)1u << 3)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_B128_DYNAMIC_ALIGNMENT \
  ((uint32_t)1u << 4)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_NEGATIVE_STATIC_OFFSET \
  ((uint32_t)1u << 5)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_B128_STATIC_ALIGNMENT \
  ((uint32_t)1u << 6)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_MISSING \
  ((uint32_t)1u << 7)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE \
  ((uint32_t)1u << 8)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE \
  ((uint32_t)1u << 9)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_MEMORY_SPACE ((uint32_t)1u << 10)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_WORKGROUP_ROOT ((uint32_t)1u << 11)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_SCALAR_DYNAMIC_STRIDE \
  ((uint32_t)1u << 12)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_WORKGROUP_DYNAMIC_INDEX_SOURCE \
  ((uint32_t)1u << 13)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_UNAVAILABLE \
  ((uint32_t)1u << 14)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_ADDRESS \
  ((uint32_t)1u << 15)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_OFFSET_RANGE \
  ((uint32_t)1u << 16)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_PACKED_REGISTER_FOOTPRINT \
  ((uint32_t)1u << 17)

typedef struct loom_amdgpu_memory_access_diagnostic_t {
  // Rejection bits explaining why an access is not legal for this target.
  loom_amdgpu_memory_access_rejection_flags_t rejection_bits;
} loom_amdgpu_memory_access_diagnostic_t;

// Returns the target memory operation represented by a source access plan.
loom_amdgpu_memory_operation_kind_t
loom_amdgpu_memory_operation_kind_from_source(
    const loom_low_source_memory_access_plan_t* source);

// Selects a complete AMDGPU memory access plan from source IR and facts.
bool loom_amdgpu_memory_access_select(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_func_like_t source_function, const loom_op_t* source_op,
    loom_amdgpu_memory_access_plan_t* out_access,
    loom_low_source_memory_access_diagnostic_t* out_source_diagnostic,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic);

// Returns true when an access carries an explicit cache policy.
bool loom_amdgpu_memory_cache_policy_is_present(
    const loom_vector_memory_cache_policy_t* policy);

// Returns true when the selected descriptor set can encode the cache policy on
// the memory access plan.
bool loom_amdgpu_memory_cache_policy_can_lower(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_plan_t* access);

// Returns the GFX12 TH immediate value for a vector memory temporal policy.
bool loom_amdgpu_memory_cache_policy_gfx12_th(uint8_t temporal,
                                              int64_t* out_th);

// Returns a status explaining why the access cache policy cannot be encoded.
iree_status_t loom_amdgpu_memory_cache_policy_rejected_status(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_plan_t* access,
    const loom_vector_memory_cache_policy_t* policy);

// Returns the diagnostic detail for target-specific memory-access rejection
// bits.
iree_string_view_t loom_amdgpu_memory_access_rejection_detail(
    loom_amdgpu_memory_access_rejection_flags_t rejection_bits);

// Records optional memory diagnostics for a selected memory access plan.
iree_status_t loom_amdgpu_record_memory_access_diagnostic(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_plan_t* access,
    loom_amdgpu_memory_operation_kind_t kind);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_MEMORY_INTERNAL_H_
