// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared reservations for spill plans in one immutable allocation snapshot.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_SPILL_STORAGE_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_SPILL_STORAGE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation/table.h"

#ifdef __cplusplus
extern "C" {
#endif

// One storage lifetime supplied by immutable spill planning. Whole-block
// intervals remain valid when later scheduling reorders inserted traffic.
typedef struct loom_low_spill_storage_interval_t {
  // Borrowed spill plan supplying the storage space, byte size, and alignment.
  const loom_low_allocation_spill_plan_t* plan;
  // Original plan index, preserved through sorting.
  uint32_t plan_index;
  // First block whose execution may access this spilled value.
  uint16_t first_block;
  // Last block whose execution may access this spilled value, inclusive.
  uint16_t last_block;
} loom_low_spill_storage_interval_t;

// Groups equally sized/aligned spill plans in the same storage space when
// their block intervals do not overlap. Sorts |intervals| in place; the caller
// supplies at least two intervals for plans with retained traffic. The producer
// includes sparse holes and predecessor stores in each lifetime, and supplies
// a function-wide interval for cells requiring exclusive ABI storage.
//
// Each interval names a distinct index below |plan_count|. Unselected plans
// map to themselves and cannot occupy a reservation. Each result names the
// earliest selected plan sharing that reservation, so materialization creates
// reservations in plan order without forward references. NULL denotes no shared
// reservations. Results belong to |arena| and are consumed before rewriting the
// snapshot. Reservations from previous repairs retain their placements.
iree_status_t loom_low_allocation_spill_storage_plan(
    loom_low_spill_storage_interval_t* intervals,
    iree_host_size_t interval_count, iree_host_size_t plan_count,
    iree_arena_allocator_t* arena, const uint32_t** out_representatives);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_SPILL_STORAGE_H_
