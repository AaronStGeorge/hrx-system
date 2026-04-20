// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent packetization for low functions.
//
// Packetization is the shared producer contract for low emitters: it runs the
// scheduler and allocator over one low.func.def, verifies that their sidecars
// describe the same target function, and returns the arena-owned sidecars that
// packet emitters consume. This layer does not emit bytes, text, JSON, or
// target artifacts; each target emitter owns those artifact decisions.

#ifndef LOOM_CODEGEN_LOW_PACKETIZATION_H_
#define LOOM_CODEGEN_LOW_PACKETIZATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Options controlling packet sidecar construction for one low function.
typedef struct loom_low_packetization_options_t {
  // Descriptor registry available to scheduling and allocation.
  const loom_low_descriptor_registry_t* descriptor_registry;
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
} loom_low_packetization_options_t;

// Packetization result for one low.func.def. The schedule and allocation
// sidecars borrow from the caller-provided module and arena.
typedef struct loom_low_packetization_t {
  // Schedule sidecar for the packetized function.
  loom_low_schedule_sidecar_t schedule;
  // Allocation sidecar for the packetized function.
  loom_low_allocation_sidecar_t allocation;
} loom_low_packetization_t;

// Schedules, allocates, and validates one low.func.def for target emitters.
// |module| must remain immutable and |arena| must outlive |out_packetization|.
iree_status_t loom_low_packetize_function(
    const loom_module_t* module, const loom_op_t* low_func_op,
    const loom_low_packetization_options_t* options,
    iree_arena_allocator_t* arena, loom_low_packetization_t* out_packetization);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_PACKETIZATION_H_
