// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent emission frame construction for low functions.
//
// The emission frame is the shared producer contract for low emitters: it runs
// the scheduler and allocator over one prepared target-low function, verifies
// that their tables describe the same target function, and returns the
// arena-owned frame that packet emitters consume. It assumes ordinary pass
// pipelines have already prepared the low IR. This layer does not run
// optimization passes, emit bytes, text, JSON, or target artifacts; each target
// emitter owns those artifact decisions.

#ifndef LOOM_CODEGEN_LOW_FRAME_H_
#define LOOM_CODEGEN_LOW_FRAME_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/memory_access.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Options controlling emission frame construction for one low function.
typedef struct loom_low_emission_frame_options_t {
  // Descriptor registry available to scheduling and allocation.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Optional source-derived memory summaries for the scheduled low function.
  loom_low_memory_access_table_t memory_access_table;
  // Optional target-provided register-pressure cliff table.
  loom_low_schedule_pressure_cliff_list_t schedule_pressure_cliffs;
  // Candidate selection strategy used by the scheduler.
  loom_low_schedule_strategy_t schedule_strategy;
  // Optional structured scheduler feedback to emit.
  loom_low_schedule_diagnostic_flags_t schedule_diagnostic_flags;
  // Explicit per-class register budgets passed to allocation.
  const loom_low_allocation_budget_t* allocation_budgets;
  // Number of entries in |allocation_budgets|.
  iree_host_size_t allocation_budget_count;
  // Fixed locations for precolored SSA values passed to allocation.
  const loom_low_allocation_fixed_value_t* allocation_fixed_values;
  // Number of entries in |allocation_fixed_values|.
  iree_host_size_t allocation_fixed_value_count;
  // Whole-function target-owned location ranges passed to allocation.
  const loom_low_allocation_reserved_range_t* allocation_reserved_ranges;
  // Number of entries in |allocation_reserved_ranges|.
  iree_host_size_t allocation_reserved_range_count;
  // Optional structured allocation feedback to emit.
  loom_low_allocation_diagnostic_flags_t allocation_diagnostic_flags;
  // Structured diagnostic emitter shared by scheduling and allocation.
  iree_diagnostic_emitter_t emitter;
} loom_low_emission_frame_options_t;

// Emission-ready production frame for one prepared target-low function. The
// frame and its nested tables borrow from the caller-provided module and arena.
typedef struct loom_low_emission_frame_t {
  // Module containing the prepared low function.
  const loom_module_t* module;
  // Prepared target-low function operation.
  const loom_op_t* function_op;
  // Resolved target context shared by the nested schedule/allocation tables.
  loom_low_resolved_target_t target;
  // Schedule table for the prepared function.
  loom_low_schedule_table_t schedule;
  // Allocation table for the prepared function.
  loom_low_allocation_table_t allocation;
} loom_low_emission_frame_t;

// Schedules, allocates, and validates one target-low function for target
// emitters. |arena| must outlive |out_frame|.
iree_status_t loom_low_emission_frame_build(
    loom_module_t* module, loom_op_t* low_func_op,
    const loom_low_emission_frame_options_t* options,
    iree_arena_allocator_t* arena, loom_low_emission_frame_t* out_frame);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_FRAME_H_
