// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_UTIL_FACT_RANGE_H_
#define LOOM_UTIL_FACT_RANGE_H_

#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Direct auxiliary fact storage covers only the range reached by its owner.
// A function retains module-wide IDs without allocating the preceding values.
// Typed entries and their absent-value representation belong to each family.
typedef struct loom_value_fact_range_t {
  // Value ID addressed by the first entry; ignored while capacity is zero.
  loom_value_id_t first_value_id;
  // Allocated entries, including growth headroom and withdrawn publications.
  iree_host_size_t capacity;
} loom_value_fact_range_t;

static inline bool loom_value_fact_range_contains(loom_value_fact_range_t range,
                                                  loom_value_id_t value_id) {
  return (uint64_t)value_id - range.first_value_id < range.capacity;
}

// Returns an entry offset after lookup or growth established membership.
static inline iree_host_size_t loom_value_fact_range_offset(
    loom_value_fact_range_t range, loom_value_id_t value_id) {
  return (iree_host_size_t)value_id - range.first_value_id;
}

// Extends a typed entry array to cover |value_id|, preserving existing entries.
// New entries copy |empty_entry|, or are zeroed when it is NULL. Range and
// entries change together after allocation succeeds. Growth may move the array;
// payloads referenced by its entries retain their independent arena lifetimes.
iree_status_t loom_value_fact_range_grow(iree_arena_allocator_t* arena,
                                         loom_value_id_t value_id,
                                         iree_host_size_t element_size,
                                         const void* empty_entry,
                                         loom_value_fact_range_t* range,
                                         void** entries);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_FACT_RANGE_H_
