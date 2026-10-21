// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_origin.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/util/fact_table.h"

struct loom_value_fact_exact_lane_origin_entry_t {
  // Aggregate value carrying the exact origin.
  loom_value_id_t value_id;
  // Exact lane origin associated with value_id.
  loom_value_fact_exact_lane_origin_t origin;
};

static_assert(sizeof(loom_value_fact_exact_lane_origin_entry_t) == 16,
              "exact lane origin entries must remain compact");

static iree_status_t loom_value_fact_table_ensure_uniform_origin_capacity(
    loom_value_fact_table_t* table, loom_value_id_t value_id) {
  const loom_value_fact_uniform_element_origin_t empty_entry =
      (loom_value_fact_uniform_element_origin_t){LOOM_VALUE_ID_INVALID,
                                                 LOOM_VALUE_ID_INVALID};
  return loom_value_fact_range_grow(
      table->arena, value_id, sizeof(*table->uniform_element_origins.entries),
      &empty_entry, &table->uniform_element_origins.range,
      (void**)&table->uniform_element_origins.entries);
}

static loom_value_fact_static_lane_origin_t
loom_value_fact_static_lane_origin_invalid(void) {
  return (loom_value_fact_static_lane_origin_t){
      .source_value_id = LOOM_VALUE_ID_INVALID,
  };
}

static loom_value_fact_exact_lane_origin_t
loom_value_fact_exact_lane_origin_invalid(void) {
  return (loom_value_fact_exact_lane_origin_t){
      .source_value_id = LOOM_VALUE_ID_INVALID,
  };
}

static loom_value_fact_uniform_scale_origin_t
loom_value_fact_uniform_scale_origin_invalid(void) {
  return (loom_value_fact_uniform_scale_origin_t){
      .source_value_id = LOOM_VALUE_ID_INVALID,
      .scale_value_id = LOOM_VALUE_ID_INVALID,
  };
}

static iree_status_t loom_value_fact_table_ensure_static_lane_origin_capacity(
    loom_value_fact_table_t* table, loom_value_id_t value_id) {
  const loom_value_fact_static_lane_origin_t empty_entry =
      loom_value_fact_static_lane_origin_invalid();
  return loom_value_fact_range_grow(
      table->arena, value_id, sizeof(*table->static_lane_origins.entries),
      &empty_entry, &table->static_lane_origins.range,
      (void**)&table->static_lane_origins.entries);
}

static iree_status_t loom_value_fact_table_ensure_uniform_scale_origin_capacity(
    loom_value_fact_table_t* table, loom_value_id_t value_id) {
  const loom_value_fact_uniform_scale_origin_t empty_entry =
      loom_value_fact_uniform_scale_origin_invalid();
  return loom_value_fact_range_grow(
      table->arena, value_id, sizeof(*table->uniform_scale_origins.entries),
      &empty_entry, &table->uniform_scale_origins.range,
      (void**)&table->uniform_scale_origins.entries);
}

static iree_status_t
loom_value_fact_table_ensure_contextual_query_origin_capacity(
    loom_value_fact_table_t* table, loom_value_id_t value_id) {
  return loom_value_fact_range_grow(
      table->arena, value_id, sizeof(*table->contextual_query_origins.entries),
      NULL, &table->contextual_query_origins.range,
      (void**)&table->contextual_query_origins.entries);
}

static iree_status_t loom_value_fact_table_append_touched_uniform_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id) {
  if (table->uniform_element_origins.touched_count >=
      table->uniform_element_origins.touched_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        table->arena, table->uniform_element_origins.touched_count,
        table->uniform_element_origins.touched_count + 1,
        sizeof(*table->uniform_element_origins.touched_values),
        &table->uniform_element_origins.touched_capacity,
        (void**)&table->uniform_element_origins.touched_values));
  }
  table->uniform_element_origins
      .touched_values[table->uniform_element_origins.touched_count++] =
      value_id;
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_append_touched_static_lane_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id) {
  if (table->static_lane_origins.touched_count >=
      table->static_lane_origins.touched_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        table->arena, table->static_lane_origins.touched_count,
        table->static_lane_origins.touched_count + 1,
        sizeof(*table->static_lane_origins.touched_values),
        &table->static_lane_origins.touched_capacity,
        (void**)&table->static_lane_origins.touched_values));
  }
  table->static_lane_origins
      .touched_values[table->static_lane_origins.touched_count++] = value_id;
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_append_touched_uniform_scale_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id) {
  if (table->uniform_scale_origins.touched_count >=
      table->uniform_scale_origins.touched_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        table->arena, table->uniform_scale_origins.touched_count,
        table->uniform_scale_origins.touched_count + 1,
        sizeof(*table->uniform_scale_origins.touched_values),
        &table->uniform_scale_origins.touched_capacity,
        (void**)&table->uniform_scale_origins.touched_values));
  }
  table->uniform_scale_origins
      .touched_values[table->uniform_scale_origins.touched_count++] = value_id;
  return iree_ok_status();
}

static iree_status_t
loom_value_fact_table_append_touched_contextual_query_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id) {
  if (table->contextual_query_origins.touched_count >=
      table->contextual_query_origins.touched_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        table->arena, table->contextual_query_origins.touched_count,
        table->contextual_query_origins.touched_count + 1,
        sizeof(*table->contextual_query_origins.touched_values),
        &table->contextual_query_origins.touched_capacity,
        (void**)&table->contextual_query_origins.touched_values));
  }
  table->contextual_query_origins
      .touched_values[table->contextual_query_origins.touched_count++] =
      value_id;
  return iree_ok_status();
}

void loom_value_fact_table_clear_origin_scope(loom_value_fact_table_t* table) {
  for (iree_host_size_t i = 0; i < table->uniform_element_origins.touched_count;
       ++i) {
    table->uniform_element_origins.entries[loom_value_fact_range_offset(
        table->uniform_element_origins.range,
        table->uniform_element_origins.touched_values[i])] =
        (loom_value_fact_uniform_element_origin_t){
            .scalar_value_id = LOOM_VALUE_ID_INVALID,
            .exact_scalar_value_id = LOOM_VALUE_ID_INVALID,
        };
  }
  for (iree_host_size_t i = 0; i < table->static_lane_origins.touched_count;
       ++i) {
    table->static_lane_origins.entries[loom_value_fact_range_offset(
        table->static_lane_origins.range,
        table->static_lane_origins.touched_values[i])] =
        loom_value_fact_static_lane_origin_invalid();
  }
  for (iree_host_size_t i = 0; i < table->uniform_scale_origins.touched_count;
       ++i) {
    table->uniform_scale_origins.entries[loom_value_fact_range_offset(
        table->uniform_scale_origins.range,
        table->uniform_scale_origins.touched_values[i])] =
        loom_value_fact_uniform_scale_origin_invalid();
  }
  for (iree_host_size_t i = 0;
       i < table->contextual_query_origins.touched_count; ++i) {
    table->contextual_query_origins.entries[loom_value_fact_range_offset(
        table->contextual_query_origins.range,
        table->contextual_query_origins.touched_values[i])] = 0;
  }
  table->uniform_element_origins.touched_count = 0;
  table->static_lane_origins.touched_count = 0;
  table->exact_lane_origins.count = 0;
  table->uniform_scale_origins.touched_count = 0;
  table->contextual_query_origins.touched_count = 0;
  table->contextual_query_origins.origin_count = 0;
}

static bool loom_value_fact_table_lookup_uniform_element_origin(
    const loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_uniform_element_origin_t* out_origin) {
  if (out_origin) {
    *out_origin = (loom_value_fact_uniform_element_origin_t){
        .scalar_value_id = LOOM_VALUE_ID_INVALID,
        .exact_scalar_value_id = LOOM_VALUE_ID_INVALID,
    };
  }
  if (!loom_value_fact_range_contains(table->uniform_element_origins.range,
                                      value_id) ||
      table->uniform_element_origins.entries == NULL) {
    return false;
  }
  const loom_value_fact_uniform_element_origin_t origin =
      table->uniform_element_origins.entries[loom_value_fact_range_offset(
          table->uniform_element_origins.range, value_id)];
  if (origin.scalar_value_id == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  if (out_origin) {
    *out_origin = origin;
  }
  return true;
}

iree_status_t loom_value_fact_table_define_uniform_element_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_id_t scalar_value_id, loom_value_id_t exact_scalar_value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID ||
      scalar_value_id == LOOM_VALUE_ID_INVALID ||
      exact_scalar_value_id == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_ensure_uniform_origin_capacity(table, value_id));
  if (table->uniform_element_origins
          .entries[loom_value_fact_range_offset(
              table->uniform_element_origins.range, value_id)]
          .scalar_value_id == LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_append_touched_uniform_origin(table, value_id));
  }
  table->uniform_element_origins.entries[loom_value_fact_range_offset(
      table->uniform_element_origins.range, value_id)] =
      (loom_value_fact_uniform_element_origin_t){
          .scalar_value_id = scalar_value_id,
          .exact_scalar_value_id = exact_scalar_value_id,
      };
  return iree_ok_status();
}

bool loom_value_fact_table_query_uniform_element_origin(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id, loom_value_id_t* out_scalar_value_id) {
  if (out_scalar_value_id) {
    *out_scalar_value_id = LOOM_VALUE_ID_INVALID;
  }
  if (table == NULL || module == NULL || value_id >= module->values.count) {
    return false;
  }
  const loom_type_t value_type = loom_module_value_type(module, value_id);
  if (!loom_type_is_shaped(value_type)) {
    return false;
  }

  loom_value_fact_uniform_element_origin_t origin = {0};
  if (!loom_value_fact_table_lookup_uniform_element_origin(table, value_id,
                                                           &origin) ||
      origin.scalar_value_id >= module->values.count) {
    return false;
  }
  const loom_type_t scalar_type =
      loom_module_value_type(module, origin.scalar_value_id);
  if (!loom_type_is_scalar(scalar_type) ||
      loom_type_element_type(scalar_type) !=
          loom_type_element_type(value_type)) {
    return false;
  }
  if (out_scalar_value_id) {
    *out_scalar_value_id = origin.scalar_value_id;
  }
  return true;
}

bool loom_value_fact_table_query_exact_uniform_element_origin(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id, loom_value_id_t* out_scalar_value_id) {
  if (out_scalar_value_id) {
    *out_scalar_value_id = LOOM_VALUE_ID_INVALID;
  }
  if (table == NULL || module == NULL || value_id >= module->values.count) {
    return false;
  }
  const loom_type_t value_type = loom_module_value_type(module, value_id);
  if (!loom_type_is_shaped(value_type)) {
    return false;
  }

  loom_value_fact_uniform_element_origin_t origin = {0};
  if (!loom_value_fact_table_lookup_uniform_element_origin(table, value_id,
                                                           &origin) ||
      origin.exact_scalar_value_id >= module->values.count) {
    return false;
  }
  const loom_type_t scalar_type =
      loom_module_value_type(module, origin.exact_scalar_value_id);
  if (!loom_type_is_scalar(scalar_type)) {
    return false;
  }
  const loom_scalar_type_t value_element = loom_type_element_type(value_type);
  const loom_scalar_type_t scalar_element = loom_type_element_type(scalar_type);
  if (value_element != scalar_element &&
      (!loom_scalar_type_is_float(value_element) ||
       !loom_scalar_type_is_float(scalar_element) ||
       loom_scalar_type_bitwidth(scalar_element) >=
           loom_scalar_type_bitwidth(value_element))) {
    return false;
  }
  if (out_scalar_value_id) {
    *out_scalar_value_id = origin.exact_scalar_value_id;
  }
  return true;
}

static bool loom_value_fact_table_lookup_static_lane_origin(
    const loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_static_lane_origin_t* out_origin) {
  if (out_origin) {
    *out_origin = loom_value_fact_static_lane_origin_invalid();
  }
  if (!loom_value_fact_range_contains(table->static_lane_origins.range,
                                      value_id) ||
      table->static_lane_origins.entries == NULL) {
    return false;
  }
  const loom_value_fact_static_lane_origin_t origin =
      table->static_lane_origins.entries[loom_value_fact_range_offset(
          table->static_lane_origins.range, value_id)];
  if (origin.source_value_id == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  if (out_origin) {
    *out_origin = origin;
  }
  return true;
}

iree_status_t loom_value_fact_table_define_static_lane_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_static_lane_origin_t origin) {
  if (value_id == LOOM_VALUE_ID_INVALID ||
      origin.source_value_id == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  IREE_ASSERT_NE(origin.source_lane_stride, 0u);
  IREE_RETURN_IF_ERROR(loom_value_fact_table_ensure_static_lane_origin_capacity(
      table, value_id));
  if (table->static_lane_origins
          .entries[loom_value_fact_range_offset(
              table->static_lane_origins.range, value_id)]
          .source_value_id == LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_append_touched_static_lane_origin(table,
                                                                value_id));
  }
  table->static_lane_origins.entries[loom_value_fact_range_offset(
      table->static_lane_origins.range, value_id)] = origin;
  return iree_ok_status();
}

static bool loom_value_fact_table_lane_origin_is_valid(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_value_id_t source_value_id, uint32_t source_lane_offset,
    uint32_t source_lane_stride) {
  if (module == NULL || value_id >= module->values.count ||
      source_value_id >= module->values.count || source_lane_stride == 0) {
    return false;
  }
  const loom_type_t value_type = loom_module_value_type(module, value_id);
  const loom_type_t source_type =
      loom_module_value_type(module, source_value_id);
  if (!loom_type_is_vector(value_type) || !loom_type_is_vector(source_type)) {
    return false;
  }

  uint64_t value_lane_count = 0;
  uint64_t source_lane_count = 0;
  if (!loom_type_static_element_count(value_type, &value_lane_count) ||
      !loom_type_static_element_count(source_type, &source_lane_count)) {
    return false;
  }
  uint64_t max_source_lane = source_lane_offset;
  if (value_lane_count > 0) {
    const uint64_t lane_delta_count = value_lane_count - 1u;
    const uint64_t stride = source_lane_stride;
    if (lane_delta_count > (UINT64_MAX - max_source_lane) / stride) {
      return false;
    }
    max_source_lane += lane_delta_count * stride;
  }
  return max_source_lane < source_lane_count;
}

bool loom_value_fact_table_query_static_lane_origin(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id,
    loom_value_fact_static_lane_origin_t* out_origin) {
  if (out_origin) {
    *out_origin = loom_value_fact_static_lane_origin_invalid();
  }
  if (table == NULL) {
    return false;
  }
  loom_value_fact_static_lane_origin_t origin =
      loom_value_fact_static_lane_origin_invalid();
  if (!loom_value_fact_table_lookup_static_lane_origin(table, value_id,
                                                       &origin) ||
      !loom_value_fact_table_lane_origin_is_valid(
          module, value_id, origin.source_value_id, origin.source_lane_offset,
          origin.source_lane_stride)) {
    return false;
  }

  if (out_origin) {
    *out_origin = origin;
  }
  return true;
}

static iree_host_size_t loom_value_fact_table_exact_lane_origin_lower_bound(
    const loom_value_fact_table_t* table, loom_value_id_t value_id) {
  iree_host_size_t lower = 0;
  iree_host_size_t upper = table->exact_lane_origins.count;
  while (lower < upper) {
    const iree_host_size_t middle = lower + (upper - lower) / 2;
    if (table->exact_lane_origins.entries[middle].value_id < value_id) {
      lower = middle + 1;
    } else {
      upper = middle;
    }
  }
  return lower;
}

static bool loom_value_fact_table_lookup_exact_lane_origin(
    const loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_exact_lane_origin_t* out_origin) {
  if (out_origin) {
    *out_origin = loom_value_fact_exact_lane_origin_invalid();
  }
  const iree_host_size_t index =
      loom_value_fact_table_exact_lane_origin_lower_bound(table, value_id);
  if (index == table->exact_lane_origins.count ||
      table->exact_lane_origins.entries[index].value_id != value_id) {
    return false;
  }
  if (out_origin) {
    *out_origin = table->exact_lane_origins.entries[index].origin;
  }
  return true;
}

iree_status_t loom_value_fact_table_define_exact_lane_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_exact_lane_origin_t origin) {
  if (value_id == LOOM_VALUE_ID_INVALID ||
      origin.source_value_id == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  IREE_ASSERT_NE(origin.source_lane_stride, 0u);
  const iree_host_size_t index =
      loom_value_fact_table_exact_lane_origin_lower_bound(table, value_id);
  if (index < table->exact_lane_origins.count &&
      table->exact_lane_origins.entries[index].value_id == value_id) {
    table->exact_lane_origins.entries[index].origin = origin;
    return iree_ok_status();
  }
  if (table->exact_lane_origins.count >= table->exact_lane_origins.capacity) {
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(table->arena, table->exact_lane_origins.count,
                              table->exact_lane_origins.count + 1,
                              sizeof(*table->exact_lane_origins.entries),
                              &table->exact_lane_origins.capacity,
                              (void**)&table->exact_lane_origins.entries));
  }
  if (index < table->exact_lane_origins.count) {
    memmove(&table->exact_lane_origins.entries[index + 1],
            &table->exact_lane_origins.entries[index],
            (table->exact_lane_origins.count - index) *
                sizeof(*table->exact_lane_origins.entries));
  }
  table->exact_lane_origins.entries[index] =
      (loom_value_fact_exact_lane_origin_entry_t){
          .value_id = value_id,
          .origin = origin,
      };
  ++table->exact_lane_origins.count;
  return iree_ok_status();
}

bool loom_value_fact_table_query_exact_lane_origin(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id, loom_value_fact_exact_lane_origin_t* out_origin) {
  if (out_origin) {
    *out_origin = loom_value_fact_exact_lane_origin_invalid();
  }
  if (table == NULL) {
    return false;
  }
  loom_value_fact_exact_lane_origin_t origin =
      loom_value_fact_exact_lane_origin_invalid();
  if (!loom_value_fact_table_lookup_exact_lane_origin(table, value_id,
                                                      &origin) ||
      !loom_value_fact_table_lane_origin_is_valid(
          module, value_id, origin.source_value_id, origin.source_lane_offset,
          origin.source_lane_stride)) {
    return false;
  }
  if (out_origin) {
    *out_origin = origin;
  }
  return true;
}

static bool loom_value_fact_table_lookup_uniform_scale_origin(
    const loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_uniform_scale_origin_t* out_origin) {
  if (out_origin) {
    *out_origin = loom_value_fact_uniform_scale_origin_invalid();
  }
  if (!loom_value_fact_range_contains(table->uniform_scale_origins.range,
                                      value_id) ||
      table->uniform_scale_origins.entries == NULL) {
    return false;
  }
  const loom_value_fact_uniform_scale_origin_t origin =
      table->uniform_scale_origins.entries[loom_value_fact_range_offset(
          table->uniform_scale_origins.range, value_id)];
  if (origin.source_value_id == LOOM_VALUE_ID_INVALID ||
      origin.scale_value_id == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  if (out_origin) {
    *out_origin = origin;
  }
  return true;
}

iree_status_t loom_value_fact_table_define_uniform_scale_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_uniform_scale_origin_t origin) {
  if (value_id == LOOM_VALUE_ID_INVALID ||
      origin.source_value_id == LOOM_VALUE_ID_INVALID ||
      origin.scale_value_id == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_ensure_uniform_scale_origin_capacity(table,
                                                                 value_id));
  if (table->uniform_scale_origins
          .entries[loom_value_fact_range_offset(
              table->uniform_scale_origins.range, value_id)]
          .source_value_id == LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_append_touched_uniform_scale_origin(table,
                                                                  value_id));
  }
  table->uniform_scale_origins.entries[loom_value_fact_range_offset(
      table->uniform_scale_origins.range, value_id)] = origin;
  return iree_ok_status();
}

bool loom_value_fact_table_query_uniform_scale_origin(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id,
    loom_value_fact_uniform_scale_origin_t* out_origin) {
  if (out_origin) {
    *out_origin = loom_value_fact_uniform_scale_origin_invalid();
  }
  if (table == NULL || module == NULL || value_id >= module->values.count) {
    return false;
  }
  loom_value_fact_uniform_scale_origin_t origin =
      loom_value_fact_uniform_scale_origin_invalid();
  if (!loom_value_fact_table_lookup_uniform_scale_origin(table, value_id,
                                                         &origin) ||
      origin.source_value_id >= module->values.count ||
      origin.scale_value_id >= module->values.count) {
    return false;
  }

  const loom_type_t value_type = loom_module_value_type(module, value_id);
  const loom_type_t source_type =
      loom_module_value_type(module, origin.source_value_id);
  const loom_type_t scale_type =
      loom_module_value_type(module, origin.scale_value_id);
  if (!loom_type_is_vector(value_type) || !loom_type_is_vector(source_type) ||
      !loom_type_is_scalar(scale_type) ||
      loom_type_element_type(value_type) !=
          loom_type_element_type(source_type) ||
      loom_type_element_type(value_type) !=
          loom_type_element_type(scale_type)) {
    return false;
  }

  uint64_t value_lane_count = 0;
  uint64_t source_lane_count = 0;
  if (!loom_type_static_element_count(value_type, &value_lane_count) ||
      !loom_type_static_element_count(source_type, &source_lane_count) ||
      value_lane_count != source_lane_count) {
    return false;
  }

  if (out_origin) {
    *out_origin = origin;
  }
  return true;
}

static bool loom_value_fact_contextual_query_origins_equal(
    loom_value_fact_contextual_query_origin_t left,
    loom_value_fact_contextual_query_origin_t right) {
  return left.family_kind == right.family_kind &&
         loom_attribute_equal(&left.key, &right.key);
}

static bool loom_value_fact_table_lookup_contextual_query_origin(
    const loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_contextual_query_origin_t* out_origin) {
  if (out_origin) {
    *out_origin = (loom_value_fact_contextual_query_origin_t){0};
  }
  if (!loom_value_fact_range_contains(table->contextual_query_origins.range,
                                      value_id) ||
      table->contextual_query_origins.entries == NULL) {
    return false;
  }
  const uint32_t origin_id =
      table->contextual_query_origins.entries[loom_value_fact_range_offset(
          table->contextual_query_origins.range, value_id)];
  if (origin_id == 0 ||
      origin_id > table->contextual_query_origins.origin_count) {
    return false;
  }
  if (out_origin) {
    *out_origin = table->contextual_query_origins.origins[origin_id - 1];
  }
  return true;
}

iree_status_t loom_value_fact_table_define_contextual_query_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_contextual_query_origin_t origin) {
  if (value_id == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  IREE_ASSERT_NE(origin.family_kind, LOOM_PARAMETERIZED_ATTR_KIND_ANY);
  IREE_ASSERT_EQ(origin.reserved, 0u);
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_ensure_contextual_query_origin_capacity(table,
                                                                    value_id));

  loom_value_fact_contextual_query_origin_t existing_origin = {0};
  if (loom_value_fact_table_lookup_contextual_query_origin(table, value_id,
                                                           &existing_origin)) {
    IREE_ASSERT(loom_value_fact_contextual_query_origins_equal(existing_origin,
                                                               origin));
    return iree_ok_status();
  }

  if (table->contextual_query_origins.origin_count >= UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "contextual query origin count exceeds uint32_t range");
  }
  if (table->contextual_query_origins.origin_count >=
      table->contextual_query_origins.origin_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        table->arena, table->contextual_query_origins.origin_count,
        table->contextual_query_origins.origin_count + 1,
        sizeof(*table->contextual_query_origins.origins),
        &table->contextual_query_origins.origin_capacity,
        (void**)&table->contextual_query_origins.origins));
  }
  const uint32_t origin_id =
      (uint32_t)++table->contextual_query_origins.origin_count;
  table->contextual_query_origins.origins[origin_id - 1] = origin;
  table->contextual_query_origins.entries[loom_value_fact_range_offset(
      table->contextual_query_origins.range, value_id)] = origin_id;
  return loom_value_fact_table_append_touched_contextual_query_origin(table,
                                                                      value_id);
}

bool loom_value_fact_table_query_contextual_query_origin(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id,
    loom_value_fact_contextual_query_origin_t* out_origin) {
  if (out_origin) {
    *out_origin = (loom_value_fact_contextual_query_origin_t){0};
  }
  if (table == NULL || module == NULL || value_id >= module->values.count ||
      !loom_type_is_scalar(loom_module_value_type(module, value_id))) {
    return false;
  }
  return loom_value_fact_table_lookup_contextual_query_origin(table, value_id,
                                                              out_origin);
}

void loom_value_fact_table_contextual_query_values(
    const loom_value_fact_table_t* table, const loom_value_id_t** out_value_ids,
    iree_host_size_t* out_value_count) {
  if (table == NULL) {
    *out_value_ids = NULL;
    *out_value_count = 0;
    return;
  }
  *out_value_ids = table->contextual_query_origins.touched_values;
  *out_value_count = table->contextual_query_origins.touched_count;
}

iree_status_t loom_value_fact_table_clone_origins(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* source,
    loom_value_id_t value_id) {
  loom_value_fact_uniform_element_origin_t uniform_origin = {0};
  if (loom_value_fact_table_lookup_uniform_element_origin(source, value_id,
                                                          &uniform_origin)) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define_uniform_element_origin(
        target, value_id, uniform_origin.scalar_value_id,
        uniform_origin.exact_scalar_value_id));
  }
  loom_value_fact_static_lane_origin_t lane_origin =
      loom_value_fact_static_lane_origin_invalid();
  if (loom_value_fact_table_lookup_static_lane_origin(source, value_id,
                                                      &lane_origin)) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define_static_lane_origin(
        target, value_id, lane_origin));
  }
  loom_value_fact_exact_lane_origin_t exact_lane_origin =
      loom_value_fact_exact_lane_origin_invalid();
  if (loom_value_fact_table_lookup_exact_lane_origin(source, value_id,
                                                     &exact_lane_origin)) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define_exact_lane_origin(
        target, value_id, exact_lane_origin));
  }
  loom_value_fact_uniform_scale_origin_t scale_origin =
      loom_value_fact_uniform_scale_origin_invalid();
  if (loom_value_fact_table_lookup_uniform_scale_origin(source, value_id,
                                                        &scale_origin)) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define_uniform_scale_origin(
        target, value_id, scale_origin));
  }
  loom_value_fact_contextual_query_origin_t query_origin = {0};
  if (loom_value_fact_table_lookup_contextual_query_origin(source, value_id,
                                                           &query_origin)) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define_contextual_query_origin(
        target, value_id, query_origin));
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_forward_uniform_origin(
    loom_value_fact_table_t* table, loom_value_id_t source_value_id,
    loom_value_id_t result_value_id) {
  loom_value_fact_uniform_element_origin_t existing_origin = {0};
  if (loom_value_fact_table_lookup_uniform_element_origin(
          table, result_value_id, &existing_origin)) {
    return iree_ok_status();
  }
  loom_value_fact_uniform_element_origin_t source_origin = {0};
  if (!loom_value_fact_table_lookup_uniform_element_origin(
          table, source_value_id, &source_origin)) {
    return iree_ok_status();
  }
  return loom_value_fact_table_define_uniform_element_origin(
      table, result_value_id, source_origin.scalar_value_id,
      source_origin.exact_scalar_value_id);
}

static iree_status_t loom_value_fact_table_forward_static_lane_origin(
    loom_value_fact_table_t* table, loom_value_id_t source_value_id,
    loom_value_id_t result_value_id) {
  loom_value_fact_static_lane_origin_t existing_origin =
      loom_value_fact_static_lane_origin_invalid();
  if (loom_value_fact_table_lookup_static_lane_origin(table, result_value_id,
                                                      &existing_origin)) {
    return iree_ok_status();
  }
  loom_value_fact_static_lane_origin_t lane_origin =
      loom_value_fact_static_lane_origin_invalid();
  if (!loom_value_fact_table_lookup_static_lane_origin(table, source_value_id,
                                                       &lane_origin)) {
    return iree_ok_status();
  }
  return loom_value_fact_table_define_static_lane_origin(table, result_value_id,
                                                         lane_origin);
}

static iree_status_t loom_value_fact_table_forward_exact_lane_origin(
    loom_value_fact_table_t* table, loom_value_id_t source_value_id,
    loom_value_id_t result_value_id) {
  loom_value_fact_exact_lane_origin_t existing_origin =
      loom_value_fact_exact_lane_origin_invalid();
  if (loom_value_fact_table_lookup_exact_lane_origin(table, result_value_id,
                                                     &existing_origin)) {
    return iree_ok_status();
  }
  loom_value_fact_exact_lane_origin_t lane_origin =
      loom_value_fact_exact_lane_origin_invalid();
  if (!loom_value_fact_table_lookup_exact_lane_origin(table, source_value_id,
                                                      &lane_origin)) {
    return iree_ok_status();
  }
  return loom_value_fact_table_define_exact_lane_origin(table, result_value_id,
                                                        lane_origin);
}

static iree_status_t loom_value_fact_table_forward_uniform_scale_origin(
    loom_value_fact_table_t* table, loom_value_id_t source_value_id,
    loom_value_id_t result_value_id) {
  loom_value_fact_uniform_scale_origin_t existing_origin =
      loom_value_fact_uniform_scale_origin_invalid();
  if (loom_value_fact_table_lookup_uniform_scale_origin(table, result_value_id,
                                                        &existing_origin)) {
    return iree_ok_status();
  }
  loom_value_fact_uniform_scale_origin_t scale_origin =
      loom_value_fact_uniform_scale_origin_invalid();
  if (!loom_value_fact_table_lookup_uniform_scale_origin(table, source_value_id,
                                                         &scale_origin)) {
    return iree_ok_status();
  }
  return loom_value_fact_table_define_uniform_scale_origin(
      table, result_value_id, scale_origin);
}

static iree_status_t loom_value_fact_table_forward_contextual_query_origin(
    loom_value_fact_table_t* table, loom_value_id_t source_value_id,
    loom_value_id_t result_value_id) {
  if (result_value_id == LOOM_VALUE_ID_INVALID ||
      source_value_id == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  if (loom_value_fact_range_contains(table->contextual_query_origins.range,
                                     result_value_id) &&
      table->contextual_query_origins.entries != NULL &&
      table->contextual_query_origins.entries[loom_value_fact_range_offset(
          table->contextual_query_origins.range, result_value_id)] != 0) {
    return iree_ok_status();
  }
  if (!loom_value_fact_range_contains(table->contextual_query_origins.range,
                                      source_value_id) ||
      table->contextual_query_origins.entries == NULL) {
    return iree_ok_status();
  }
  const uint32_t origin_id =
      table->contextual_query_origins.entries[loom_value_fact_range_offset(
          table->contextual_query_origins.range, source_value_id)];
  if (origin_id == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_ensure_contextual_query_origin_capacity(
          table, result_value_id));
  table->contextual_query_origins.entries[loom_value_fact_range_offset(
      table->contextual_query_origins.range, result_value_id)] = origin_id;
  return loom_value_fact_table_append_touched_contextual_query_origin(
      table, result_value_id);
}

iree_status_t loom_value_fact_table_forward_origins(
    loom_value_fact_table_t* table, loom_value_id_t source_value_id,
    loom_value_id_t result_value_id) {
  IREE_RETURN_IF_ERROR(loom_value_fact_table_forward_uniform_origin(
      table, source_value_id, result_value_id));
  IREE_RETURN_IF_ERROR(loom_value_fact_table_forward_static_lane_origin(
      table, source_value_id, result_value_id));
  IREE_RETURN_IF_ERROR(loom_value_fact_table_forward_exact_lane_origin(
      table, source_value_id, result_value_id));
  IREE_RETURN_IF_ERROR(loom_value_fact_table_forward_uniform_scale_origin(
      table, source_value_id, result_value_id));
  IREE_RETURN_IF_ERROR(loom_value_fact_table_forward_contextual_query_origin(
      table, source_value_id, result_value_id));
  if (table->layout_origins) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_forward_layout_strides(
        table, source_value_id, result_value_id));
  }
  return iree_ok_status();
}
