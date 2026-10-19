// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_layout.h"

#include <string.h>

#include "loom/util/fact_table.h"

struct loom_value_fact_layout_origins_t {
  // Immutable axis bindings indexed relative to range.first_value_id; NULL
  // is absent.
  const loom_value_fact_layout_strides_t** entries;
  // Value-ID range covered by the direct entries.
  loom_value_fact_range_t range;
};

static iree_status_t loom_value_fact_layout_origins_ensure_capacity(
    loom_value_fact_table_t* table, loom_value_id_t value_id) {
  if (!table->layout_origins) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(table->transient_arena,
                                             sizeof(*table->layout_origins),
                                             (void**)&table->layout_origins));
    *table->layout_origins = (loom_value_fact_layout_origins_t){0};
  }
  loom_value_fact_layout_origins_t* origins = table->layout_origins;
  return loom_value_fact_range_grow(table->transient_arena, value_id,
                                    sizeof(*origins->entries), NULL,
                                    &origins->range, (void**)&origins->entries);
}

loom_value_fact_layout_strides_t loom_value_fact_table_query_layout_strides(
    const loom_value_fact_table_t* table, loom_value_id_t value_id) {
  const loom_value_fact_layout_origins_t* origins = table->layout_origins;
  if (!origins || !loom_value_fact_range_contains(origins->range, value_id) ||
      !origins
           ->entries[loom_value_fact_range_offset(origins->range, value_id)]) {
    return (loom_value_fact_layout_strides_t){0};
  }
  return *origins
              ->entries[loom_value_fact_range_offset(origins->range, value_id)];
}

void loom_value_fact_table_clear_layout_strides(loom_value_fact_table_t* table,
                                                loom_value_id_t value_id) {
  loom_value_fact_layout_origins_t* origins = table->layout_origins;
  if (origins && loom_value_fact_range_contains(origins->range, value_id)) {
    origins->entries[loom_value_fact_range_offset(origins->range, value_id)] =
        NULL;
  }
}

iree_status_t loom_value_fact_table_define_layout_strides(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_layout_strides_t strides) {
  if (!strides.count) {
    loom_value_fact_table_clear_layout_strides(table, value_id);
    return iree_ok_status();
  }
  const loom_value_fact_layout_strides_t previous =
      loom_value_fact_table_query_layout_strides(table, value_id);
  const iree_host_size_t byte_length = strides.count * sizeof(*strides.values);
  if (previous.count == strides.count &&
      memcmp(previous.values, strides.values, byte_length) == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_value_fact_layout_origins_ensure_capacity(table, value_id));
  loom_value_fact_layout_strides_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(table->transient_arena,
                                           sizeof(*storage) + byte_length,
                                           (void**)&storage));
  loom_value_id_t* values = (loom_value_id_t*)(storage + 1);
  memcpy(values, strides.values, byte_length);
  *storage = (loom_value_fact_layout_strides_t){.values = values,
                                                .count = strides.count};
  table->layout_origins->entries[loom_value_fact_range_offset(
      table->layout_origins->range, value_id)] = storage;
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_forward_layout_strides(
    loom_value_fact_table_t* table, loom_value_id_t source_value_id,
    loom_value_id_t result_value_id) {
  loom_value_fact_layout_origins_t* origins = table->layout_origins;
  if (!origins ||
      !loom_value_fact_range_contains(origins->range, source_value_id) ||
      !origins->entries[loom_value_fact_range_offset(origins->range,
                                                     source_value_id)]) {
    loom_value_fact_table_clear_layout_strides(table, result_value_id);
    return iree_ok_status();
  }
  const loom_value_fact_layout_strides_t* strides =
      origins->entries[loom_value_fact_range_offset(origins->range,
                                                    source_value_id)];
  IREE_RETURN_IF_ERROR(
      loom_value_fact_layout_origins_ensure_capacity(table, result_value_id));
  origins
      ->entries[loom_value_fact_range_offset(origins->range, result_value_id)] =
      strides;
  return iree_ok_status();
}
