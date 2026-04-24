// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-low source memory access planning.
//
// This layer owns the target-independent half of vector memory lowering:
// decomposing a typed view/vector access plus value facts into a compact source
// plan with arena-compatible lifetime. Targets wrap this plan with their own
// addressing modes, descriptor choices, register classes, and machine-specific
// fallback decisions.

#ifndef LOOM_CODEGEN_LOW_SOURCE_MEMORY_PLAN_H_
#define LOOM_CODEGEN_LOW_SOURCE_MEMORY_PLAN_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/ir/attribute.h"
#include "loom/ir/ir.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/vector/memory.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE UINT32_MAX

typedef enum loom_low_source_memory_operation_kind_e {
  LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD = 0,
  LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE = 1,
  LOOM_LOW_SOURCE_MEMORY_OPERATION_PREFETCH = 2,
  LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_REDUCE = 3,
  LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_RMW = 4,
} loom_low_source_memory_operation_kind_t;

typedef enum loom_low_source_memory_dynamic_index_source_e {
  // The access has no dynamic index.
  LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_NONE = 0,
  // The dynamic index is an arbitrary SSA value.
  LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE = 1,
  // The dynamic index is produced by kernel.workitem.id.
  LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID = 2,
  // The dynamic index is produced by kernel.workgroup.id.
  LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKGROUP_ID = 3,
} loom_low_source_memory_dynamic_index_source_t;

typedef uint32_t loom_low_source_memory_access_rejection_flags_t;

#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_UNSUPPORTED_OP \
  ((uint32_t)1u << 0)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DESCRIBE_FAILED \
  ((uint32_t)1u << 1)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_LAYOUT ((uint32_t)1u << 2)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_ELEMENT_WIDTH \
  ((uint32_t)1u << 3)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VECTOR_RANK ((uint32_t)1u << 4)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VECTOR_LANE_COUNT \
  ((uint32_t)1u << 5)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE \
  ((uint32_t)1u << 6)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_STATIC_OFFSET \
  ((uint32_t)1u << 7)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_COUNT \
  ((uint32_t)1u << 8)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_AXIS ((uint32_t)1u << 9)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_STRIDE \
  ((uint32_t)1u << 10)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_SOURCE ((uint32_t)1u << 11)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_BASE ((uint32_t)1u << 12)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_BASE_OVERFLOW \
  ((uint32_t)1u << 13)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_CACHE_POLICY \
  ((uint32_t)1u << 14)

typedef struct loom_low_source_memory_access_diagnostic_t {
  // Bitset of source-access rejection reasons observed while planning.
  loom_low_source_memory_access_rejection_flags_t rejection_bits;
} loom_low_source_memory_access_diagnostic_t;

typedef struct loom_low_source_memory_access_plan_t {
  // Source operation category being planned.
  loom_low_source_memory_operation_kind_t operation_kind;
  // Source view SSA value consumed by the memory operation.
  loom_value_id_t view_value_id;
  // Target-independent memory space selected from source view facts.
  loom_value_fact_memory_space_t memory_space;
  // Source SSA value that represents the storage root.
  loom_value_id_t root_value_id;
  // Byte count of one addressed view element.
  uint32_t element_byte_count;
  // Static number of vector lanes addressed by the operation.
  uint32_t vector_lane_count;
  // Byte stride between adjacent vector lanes along the vector axis.
  int64_t vector_lane_byte_stride;
  // Total static byte offset selected from the source view access.
  int64_t static_byte_offset;
  // Dynamic view-axis index used to compute a target address operand, or
  // invalid for a purely static access.
  loom_value_id_t dynamic_index;
  // Source provenance for the dynamic index value.
  loom_low_source_memory_dynamic_index_source_t dynamic_index_source;
  // Workitem/workgroup dimension when |dynamic_index_source| is a coordinate.
  loom_kernel_dimension_t dynamic_index_dimension;
  // View axis represented by |dynamic_index|.
  uint8_t dynamic_axis;
  // Byte stride multiplied by dynamic_index before adding it to the address.
  int64_t dynamic_index_byte_stride;
  // Power-of-two shift used to compute the dynamic byte offset, or
  // LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE when multiplication or
  // target-specific handling is required.
  uint32_t dynamic_index_byte_shift;
  // Optional cache policy copied from the source memory op.
  loom_vector_memory_cache_policy_t cache_policy;
} loom_low_source_memory_access_plan_t;

static inline bool loom_low_source_memory_access_is_dynamic(
    const loom_low_source_memory_access_plan_t* plan) {
  return plan->dynamic_index != LOOM_VALUE_ID_INVALID;
}

// Builds a target-independent source memory plan for vector memory access ops.
//
// Returns false when the source op cannot be decomposed into a complete source
// plan. Targets are responsible for checking their own memory spaces, address
// forms, descriptor availability, immediate ranges, and register footprints.
bool loom_low_source_memory_access_plan_build(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_op_t* source_op, loom_low_source_memory_access_plan_t* out_plan,
    loom_low_source_memory_access_diagnostic_t* out_diagnostic);

// Returns a diagnostic detail string for source memory access rejection flags.
iree_string_view_t loom_low_source_memory_access_rejection_detail(
    loom_low_source_memory_access_rejection_flags_t rejection_bits);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SOURCE_MEMORY_PLAN_H_
