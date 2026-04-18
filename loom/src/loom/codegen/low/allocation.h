// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent allocation sidecar for target-low functions.
//
// Loom low functions remain SSA after allocation. Physical registers, target
// local IDs, and spill slots are sidecar facts over live intervals, while
// copies, split intervals, spills, and reloads are explicit IR when a later
// pass decides to materialize them.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_low_allocation_location_kind_e {
  // Location has not been assigned.
  LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED = 0,
  // Interval is assigned to target-visible physical registers.
  LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER = 1,
  // Interval is assigned to target-local IDs such as VM locals or SPIR-V IDs.
  LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID = 2,
  // Interval must be spilled into a stack, scratch, or private slot.
  LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT = 3,
} loom_low_allocation_location_kind_t;

typedef enum loom_low_allocation_remark_kind_e {
  // Unknown or uninitialized remark kind.
  LOOM_LOW_ALLOCATION_REMARK_UNKNOWN = 0,
  // An interval could not fit in the configured register budget.
  LOOM_LOW_ALLOCATION_REMARK_SPILL = 1,
} loom_low_allocation_remark_kind_t;

typedef enum loom_low_allocation_copy_kind_e {
  // Unknown or uninitialized copy decision kind.
  LOOM_LOW_ALLOCATION_COPY_UNKNOWN = 0,
  // The copy source and result share one assigned location.
  LOOM_LOW_ALLOCATION_COPY_COALESCED = 1,
  // The copy must remain a target move/copy after allocation.
  LOOM_LOW_ALLOCATION_COPY_MATERIALIZED = 2,
} loom_low_allocation_copy_kind_t;

typedef enum loom_low_allocation_diagnostic_bits_e {
  // Emits BACKEND/008 warnings for spill plans predicted by allocation.
  LOOM_LOW_ALLOCATION_DIAGNOSTIC_PREDICTED_SPILLS = 1u << 0,
  // Emits BACKEND/006 remarks for low.copy coalescing decisions.
  LOOM_LOW_ALLOCATION_DIAGNOSTIC_COPY_DECISIONS = 1u << 1,
} loom_low_allocation_diagnostic_bits_t;

// Bitset of loom_low_allocation_diagnostic_bits_t values.
typedef uint32_t loom_low_allocation_diagnostic_flags_t;

// Optional fixed register budget used by tests, tuning loops, and target
// overlays. A missing budget uses the descriptor set's physical register count;
// a descriptor class with physical_count == 0 is treated as unbounded.
typedef struct loom_low_allocation_budget_t {
  // Stable register-class name such as "vm.i32" or "amdgpu.vgpr".
  iree_string_view_t register_class;
  // Maximum allocation units available for |register_class|.
  uint32_t max_units;
} loom_low_allocation_budget_t;

// Assignment for one liveness interval.
typedef struct loom_low_allocation_assignment_t {
  // SSA value represented by this assignment.
  loom_value_id_t value_id;
  // Pressure/allocation class for |value_id|.
  loom_liveness_value_class_t value_class;
  // First live program point covered by this assignment.
  uint32_t start_point;
  // One-past-last live program point covered by this assignment.
  uint32_t end_point;
  // Allocation units required by this interval.
  uint32_t unit_count;
  // Assigned location kind.
  loom_low_allocation_location_kind_t location_kind;
  // Base physical register, target ID, or spill slot ordinal.
  uint32_t location_base;
  // Number of contiguous units assigned at |location_base|.
  uint32_t location_count;
} loom_low_allocation_assignment_t;

// Allocation remark for agent/tool feedback.
typedef struct loom_low_allocation_remark_t {
  // Remark category.
  loom_low_allocation_remark_kind_t kind;
  // Assignment index associated with this remark.
  uint32_t assignment_index;
  // Budget that was exceeded, or UINT32_MAX when no fixed budget was active.
  uint32_t budget_units;
  // Units requested by the assignment.
  uint32_t required_units;
} loom_low_allocation_remark_t;

// Spill materialization plan for one spilled assignment.
typedef struct loom_low_allocation_spill_plan_t {
  // SSA value represented by the spilled assignment.
  loom_value_id_t value_id;
  // Spilled assignment index in |assignments|.
  uint32_t assignment_index;
  // Spill slot ordinal assigned to this interval.
  uint32_t slot_index;
  // Storage space used by the spill slot.
  uint8_t slot_space;
  // Slot size in bytes.
  uint32_t byte_size;
  // Required slot alignment in bytes.
  uint32_t byte_alignment;
  // Predicted stores needed by the current synthetic spill plan.
  uint32_t store_count;
  // Predicted operand-use reloads in the current synthetic spill plan.
  uint32_t reload_count;
} loom_low_allocation_spill_plan_t;

// Copy/coalescing decision for one low.copy op.
typedef struct loom_low_allocation_copy_decision_t {
  // Source SSA value consumed by the low.copy op.
  loom_value_id_t source_value_id;
  // Result SSA value produced by the low.copy op.
  loom_value_id_t result_value_id;
  // Assignment index for |source_value_id|.
  uint32_t source_assignment_index;
  // Assignment index for |result_value_id|.
  uint32_t result_assignment_index;
  // Whether allocation coalesced or must materialize the copy.
  loom_low_allocation_copy_kind_t kind;
} loom_low_allocation_copy_decision_t;

// Options controlling allocation sidecar construction.
typedef struct loom_low_allocation_options_t {
  // Descriptor registry available to the allocator.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Explicit per-class register budgets.
  const loom_low_allocation_budget_t* budgets;
  // Number of entries in |budgets|.
  iree_host_size_t budget_count;
  // Structured diagnostic emitter for allocation failures and feedback.
  iree_diagnostic_emitter_t emitter;
  // Optional structured allocation feedback to emit.
  loom_low_allocation_diagnostic_flags_t diagnostic_flags;
} loom_low_allocation_options_t;

// Allocation sidecar for one low.func.def body. All arrays are arena-owned by
// the caller-provided arena passed to loom_low_allocate_function.
typedef struct loom_low_allocation_sidecar_t {
  // Module containing the allocated low function.
  const loom_module_t* module;
  // low.func.def operation allocated by this sidecar.
  const loom_op_t* function_op;
  // Resolved target context selected by |function_op|.
  loom_low_resolved_target_t target;
  // Liveness analysis that produced the allocated intervals.
  loom_liveness_analysis_t liveness;
  // Allocation mode requested on the low function, or 0 for the default.
  uint8_t allocation_mode;
  // Per-interval assignments in allocation order.
  const loom_low_allocation_assignment_t* assignments;
  // Number of records in |assignments|.
  iree_host_size_t assignment_count;
  // Spill materialization plans in assignment order.
  const loom_low_allocation_spill_plan_t* spill_plans;
  // Number of records in |spill_plans|.
  iree_host_size_t spill_plan_count;
  // Allocation remarks in assignment order.
  const loom_low_allocation_remark_t* remarks;
  // Number of records in |remarks|.
  iree_host_size_t remark_count;
  // Copy/coalescing decisions in source order.
  const loom_low_allocation_copy_decision_t* copy_decisions;
  // Number of records in |copy_decisions|.
  iree_host_size_t copy_decision_count;
  // Number of assignments whose location kind is SPILL_SLOT.
  iree_host_size_t spill_count;
  // Number of low.copy ops coalesced into one location.
  iree_host_size_t coalesced_copy_count;
  // Number of low.copy ops that must remain materialized.
  iree_host_size_t materialized_copy_count;
} loom_low_allocation_sidecar_t;

// Allocates one low.func.def body and writes an arena-owned sidecar. This first
// allocator is deliberately simple and deterministic: it performs per-class
// linear-scan-style first-fit assignment over liveness intervals, uses target
// physical register counts or explicit budgets as hard limits, and reports
// spills as sidecar remarks without mutating IR.
iree_status_t loom_low_allocate_function(
    const loom_module_t* module, const loom_op_t* low_func_op,
    const loom_low_allocation_options_t* options, iree_arena_allocator_t* arena,
    loom_low_allocation_sidecar_t* out_sidecar);

// Verifies that assigned intervals do not overlap on the same physical
// register or target ID range. Spill slots are not treated as registers by this
// verifier because materialized low.spill/low.reload insertion owns their
// eventual storage reuse policy.
iree_status_t loom_low_allocation_verify_sidecar(
    const loom_low_allocation_sidecar_t* sidecar);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_H_
