// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent allocation table for target-low functions.
//
// Loom low functions remain SSA after allocation. Physical registers, target
// local IDs, and spill slots are table facts over live intervals, while
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

// Fixed physical location for one SSA value.
//
// Fixed values model ABI or target live-ins/outs that enter allocation as
// ordinary SSA values but must occupy a specific target-visible location while
// live. The location is reusable outside the value's live interval.
typedef struct loom_low_allocation_fixed_value_t {
  // SSA value forced to the fixed location.
  loom_value_id_t value_id;
  // Target-visible fixed location kind.
  loom_low_allocation_location_kind_t location_kind;
  // Base physical register or target ID.
  uint32_t location_base;
  // Number of contiguous units fixed at |location_base|.
  uint32_t location_count;
} loom_low_allocation_fixed_value_t;

// Whole-function location range owned by target machinery.
//
// Reserved ranges model architectural state that is never allocatable for
// ordinary values in the current low function, such as special registers or
// permanently reserved target IDs. Use fixed values instead for ABI live-ins
// whose registers can be reused after their last use.
typedef struct loom_low_allocation_reserved_range_t {
  // Stable register-class name such as "amdgpu.sgpr" or "x86.gpr".
  iree_string_view_t register_class;
  // Target-visible reserved location kind.
  loom_low_allocation_location_kind_t location_kind;
  // First physical register or target ID in the reserved range.
  uint32_t location_base;
  // Number of contiguous units reserved at |location_base|.
  uint32_t location_count;
} loom_low_allocation_reserved_range_t;

// Assignment for one liveness interval.
typedef struct loom_low_allocation_assignment_t {
  // SSA value represented by this assignment.
  loom_value_id_t value_id;
  // Pressure/allocation class for |value_id|.
  loom_liveness_value_class_t value_class;
  // Descriptor-set-local register class ID for |value_class|.
  uint16_t descriptor_reg_class_id;
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

// Returns true when two assignments name the same concrete descriptor class and
// location kind. Alias-set-aware storage checks require a descriptor set and
// are performed by allocation verification.
static inline bool loom_low_allocation_assignments_share_storage(
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  return lhs->location_kind == rhs->location_kind &&
         lhs->descriptor_reg_class_id == rhs->descriptor_reg_class_id;
}

// Returns true when two same-length assignment subranges name the same units.
static inline bool loom_low_allocation_assignment_subranges_match(
    const loom_low_allocation_assignment_t* lhs, uint32_t lhs_start,
    const loom_low_allocation_assignment_t* rhs, uint32_t rhs_start,
    uint32_t unit_count) {
  return unit_count != 0 &&
         loom_low_allocation_assignments_share_storage(lhs, rhs) &&
         (uint64_t)lhs->location_base + lhs_start ==
             (uint64_t)rhs->location_base + rhs_start;
}

// Returns true when two same-length assignment subranges overlap in storage.
static inline bool loom_low_allocation_assignment_subranges_overlap(
    const loom_low_allocation_assignment_t* lhs, uint32_t lhs_start,
    const loom_low_allocation_assignment_t* rhs, uint32_t rhs_start,
    uint32_t unit_count) {
  if (!loom_low_allocation_assignments_share_storage(lhs, rhs)) {
    return false;
  }
  const uint64_t lhs_begin = (uint64_t)lhs->location_base + lhs_start;
  const uint64_t rhs_begin = (uint64_t)rhs->location_base + rhs_start;
  const uint64_t lhs_end = lhs_begin + unit_count;
  const uint64_t rhs_end = rhs_begin + unit_count;
  return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

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
  loom_low_spill_slot_space_t slot_space;
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

// Copy required to materialize one low.br edge payload into a destination block
// argument location.
typedef struct loom_low_allocation_edge_copy_t {
  // SSA value forwarded by the branch edge.
  loom_value_id_t source_value_id;
  // Destination block argument receiving |source_value_id|.
  loom_value_id_t destination_value_id;
  // Assignment index for |source_value_id|.
  uint32_t source_assignment_index;
  // Assignment index for |destination_value_id|.
  uint32_t destination_assignment_index;
} loom_low_allocation_edge_copy_t;

// Scratch unit reserved for sequencing cyclic edge-copy moves.
typedef struct loom_low_allocation_edge_copy_temporary_t {
  // Storage class of the cyclic move set.
  loom_liveness_value_class_t value_class;
  // Descriptor-set-local register class ID for |value_class|.
  uint16_t descriptor_reg_class_id;
  // Target-visible scratch location kind.
  loom_low_allocation_location_kind_t location_kind;
  // Physical register, target ID, or spill slot ordinal used as scratch.
  uint32_t location;
} loom_low_allocation_edge_copy_temporary_t;

// Contiguous edge-copy group for one low.br terminator.
typedef struct loom_low_allocation_edge_copy_group_t {
  // low.br terminator that owns this outgoing edge.
  const loom_op_t* terminator_op;
  // Source-order ordinal of |terminator_op| in its low function body.
  uint32_t source_ordinal;
  // Program point where the edge copies execute before |terminator_op|.
  uint32_t program_point;
  // First edge-copy record for |terminator_op|.
  uint32_t copy_start;
  // Number of edge-copy records for |terminator_op|.
  uint32_t copy_count;
  // First temporary record reserved for |terminator_op|.
  uint32_t temporary_start;
  // Number of temporary records reserved for |terminator_op|.
  uint32_t temporary_count;
} loom_low_allocation_edge_copy_group_t;

// Options controlling allocation table construction.
typedef struct loom_low_allocation_options_t {
  // Optional operation order used for live intervals.
  loom_liveness_order_t liveness_order;
  // Descriptor registry available to the allocator.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Explicit per-class register budgets.
  const loom_low_allocation_budget_t* budgets;
  // Number of entries in |budgets|.
  iree_host_size_t budget_count;
  // Fixed locations for precolored SSA values.
  const loom_low_allocation_fixed_value_t* fixed_values;
  // Number of entries in |fixed_values|.
  iree_host_size_t fixed_value_count;
  // Whole-function target-owned location ranges.
  const loom_low_allocation_reserved_range_t* reserved_ranges;
  // Number of entries in |reserved_ranges|.
  iree_host_size_t reserved_range_count;
  // Structured diagnostic emitter for allocation failures and feedback.
  iree_diagnostic_emitter_t emitter;
  // Optional structured allocation feedback to emit.
  loom_low_allocation_diagnostic_flags_t diagnostic_flags;
} loom_low_allocation_options_t;

// Allocation table for one target-low function body. All arrays are
// arena-owned by the caller-provided arena passed to
// loom_low_allocate_function.
typedef struct loom_low_allocation_table_t {
  // Module containing the allocated low function.
  const loom_module_t* module;
  // Target-low function operation allocated by this table.
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
  // Edge-copy records grouped by low.br terminator source order.
  const loom_low_allocation_edge_copy_t* edge_copies;
  // Number of records in |edge_copies|.
  iree_host_size_t edge_copy_count;
  // Per-low.br groups indexing |edge_copies|.
  const loom_low_allocation_edge_copy_group_t* edge_copy_groups;
  // Number of records in |edge_copy_groups|.
  iree_host_size_t edge_copy_group_count;
  // Scratch units reserved for cyclic edge-copy groups.
  const loom_low_allocation_edge_copy_temporary_t* edge_copy_temporaries;
  // Number of records in |edge_copy_temporaries|.
  iree_host_size_t edge_copy_temporary_count;
  // Number of assignments whose location kind is SPILL_SLOT.
  iree_host_size_t spill_count;
  // Number of low.copy ops coalesced into one location.
  iree_host_size_t coalesced_copy_count;
  // Number of low.copy ops that must remain materialized.
  iree_host_size_t materialized_copy_count;
} loom_low_allocation_table_t;

// Allocates one target-low function body and writes an arena-owned table.
// This first allocator is deliberately simple and deterministic: it performs
// per-class linear-scan-style first-fit assignment over liveness intervals,
// uses target physical register counts or explicit budgets as hard limits, and
// reports spills as table remarks without mutating IR.
iree_status_t loom_low_allocate_function(
    loom_module_t* module, const loom_op_t* low_func_op,
    const loom_low_allocation_options_t* options, iree_arena_allocator_t* arena,
    loom_low_allocation_table_t* out_table);

// Verifies that assigned intervals do not overlap on the same physical
// register or target ID range. Spill slots are not treated as registers by this
// verifier because materialized low.spill/low.reload insertion owns their
// eventual storage reuse policy.
iree_status_t loom_low_allocation_verify_table(
    const loom_low_allocation_table_t* table);

// Finds the edge-copy group for the source-order node, or NULL when the node
// has no edge-copy payload.
const loom_low_allocation_edge_copy_group_t*
loom_low_allocation_find_edge_copy_group_by_source_ordinal(
    const loom_low_allocation_table_t* table, uint32_t source_ordinal);

// Resolves the descriptor-set register class spelling for |assignment|.
iree_status_t loom_low_allocation_assignment_register_class_name(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_assignment_t* assignment,
    iree_string_view_t* out_register_class_name);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_H_
