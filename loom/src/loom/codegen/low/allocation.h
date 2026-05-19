// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent allocation construction for target-low functions.
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
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/allocation/diagnostics.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/allocation/table.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/storage_lease.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Optional fixed register budget used by tests, tuning loops, and target
// overlays. A missing budget uses the descriptor set's allocatable unit count;
// a descriptor class with allocatable_count == 0 is treated as unbounded.
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

// Options controlling allocation table construction.
typedef struct loom_low_allocation_options_t {
  // Optional operation order used for live intervals.
  loom_liveness_order_t liveness_order;
  // Descriptor registry available to the allocator.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Optional runtime/device target overlay applied when compatible with the
  // function's target record.
  loom_target_selection_t target_selection;
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
  // Optional target storage leases built over the same scheduled low function
  // represented by |liveness_order|.
  loom_low_storage_lease_table_t storage_leases;
  // Structured diagnostic emitter for allocation failures and feedback.
  iree_diagnostic_emitter_t emitter;
  // Optional structured allocation feedback to emit.
  loom_low_allocation_diagnostic_flags_t diagnostic_flags;
} loom_low_allocation_options_t;

// Allocates one target-low function body and writes an arena-owned table.
// This first allocator is deliberately simple and deterministic: it performs
// per-class linear-scan-style first-fit assignment over liveness intervals,
// uses target allocatable unit counts or explicit budgets as hard limits, and
// reports spills as table remarks without mutating IR.
iree_status_t loom_low_allocate_function(
    loom_module_t* module, const loom_op_t* low_func_op,
    const loom_low_allocation_options_t* options, iree_arena_allocator_t* arena,
    loom_low_allocation_table_t* out_table);

// Verifies allocation-table consistency. Register-like assignments must not
// overlap on the same physical register or target ID range. Spill slots are not
// treated as registers by this verifier because materialized
// low.spill/low.reload insertion owns their eventual storage reuse policy.
// Tied-result placement is enforced only when both sides are register-like; a
// spill-slot side defers the tie until spill materialization inserts reloads
// and final allocation runs on the materialized IR.
iree_status_t loom_low_allocation_verify_table(
    const loom_low_allocation_table_t* table);

// Returns true when |op| must materialize its low.concat result as a packet.
// Branch-edge copies can decompose low.concat payloads directly into block
// arguments, so branch-only concats do not require packet-local moves.
bool loom_low_allocation_concat_requires_packet_materialization(
    const loom_low_allocation_table_t* table, const loom_op_t* op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_H_
