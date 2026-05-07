// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/passes/linearize_view_accesses.h"

#include <inttypes.h>
#include <stdint.h>
#include <string.h>

#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ir/types.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/fact_table.h"
#include "loom/util/math.h"
#include "loom/util/walk.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

enum {
  LOOM_LINEARIZE_VIEW_ACCESSES_STAT_VIEWS_CREATED = 0,
  LOOM_LINEARIZE_VIEW_ACCESSES_STAT_LOADS_LINEARIZED = 1,
  LOOM_LINEARIZE_VIEW_ACCESSES_STAT_STORES_LINEARIZED = 2,
  LOOM_LINEARIZE_VIEW_ACCESSES_STAT_VECTOR_LOADS_LINEARIZED = 3,
  LOOM_LINEARIZE_VIEW_ACCESSES_STAT_VECTOR_STORES_LINEARIZED = 4,
};

static const loom_pass_statistic_def_t kLinearizeViewAccessesStatistics[] = {
    {IREE_SVL("views-created"),
     IREE_SVL("Number of rank-1 buffer views created.")},
    {IREE_SVL("loads-linearized"),
     IREE_SVL("Number of scalar view.load operations linearized.")},
    {IREE_SVL("stores-linearized"),
     IREE_SVL("Number of scalar view.store operations linearized.")},
    {IREE_SVL("vector-loads-linearized"),
     IREE_SVL("Number of vector.load operations linearized.")},
    {IREE_SVL("vector-stores-linearized"),
     IREE_SVL("Number of vector.store operations linearized.")},
};

static const loom_pass_info_t loom_linearize_view_accesses_pass_info_storage = {
    .name = IREE_SVL("linearize-view-accesses"),
    .description = IREE_SVL("Linearize dense multi-dimensional view accesses."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_defs = kLinearizeViewAccessesStatistics,
    .statistic_count = IREE_ARRAYSIZE(kLinearizeViewAccessesStatistics),
};

const loom_pass_info_t* loom_linearize_view_accesses_pass_info(void) {
  return &loom_linearize_view_accesses_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Worklists
//===----------------------------------------------------------------------===//

typedef struct loom_linearize_view_accesses_op_list_t {
  // Collected memory operations in dominance order.
  loom_op_t** ops;
  // Number of collected operations.
  iree_host_size_t count;
  // Allocated operation pointer capacity.
  iree_host_size_t capacity;
} loom_linearize_view_accesses_op_list_t;

typedef struct loom_linearize_view_accesses_view_map_entry_t {
  // Original multi-dimensional buffer.view operation.
  loom_op_t* original_view_op;
  // Rank-1 buffer.view result built for the same buffer and byte offset.
  loom_value_id_t linear_view;
} loom_linearize_view_accesses_view_map_entry_t;

typedef struct loom_linearize_view_accesses_view_map_t {
  // Lazily materialized linear view entries.
  loom_linearize_view_accesses_view_map_entry_t* entries;
  // Number of entries in use.
  iree_host_size_t count;
  // Allocated entry capacity.
  iree_host_size_t capacity;
} loom_linearize_view_accesses_view_map_t;

typedef struct loom_linearize_view_accesses_collect_context_t {
  // Pass scratch arena used for the collected operation pointer list.
  iree_arena_allocator_t* arena;
  // Collected memory operations.
  loom_linearize_view_accesses_op_list_t* accesses;
} loom_linearize_view_accesses_collect_context_t;

typedef struct loom_linearize_view_accesses_context_t {
  // Owning pass instance.
  loom_pass_t* pass;
  // Module being transformed.
  loom_module_t* module;
  // Rewriter used for IR mutations.
  loom_rewriter_t* rewriter;
  // Function-scoped facts used to resolve SSA layout encodings.
  const loom_value_fact_table_t* fact_table;
  // Lazily materialized rank-1 views.
  loom_linearize_view_accesses_view_map_t* view_map;
} loom_linearize_view_accesses_context_t;

static iree_status_t loom_linearize_view_accesses_op_list_initialize(
    iree_arena_allocator_t* arena,
    loom_linearize_view_accesses_op_list_t* list) {
  list->count = 0;
  list->capacity = 32;
  return iree_arena_allocate_array(arena, list->capacity, sizeof(loom_op_t*),
                                   (void**)&list->ops);
}

static iree_status_t loom_linearize_view_accesses_op_list_push(
    iree_arena_allocator_t* arena, loom_linearize_view_accesses_op_list_t* list,
    loom_op_t* op) {
  if (list->count >= list->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, list->count, list->count + 1, sizeof(loom_op_t*),
        &list->capacity, (void**)&list->ops));
  }
  list->ops[list->count++] = op;
  return iree_ok_status();
}

static iree_status_t loom_linearize_view_accesses_view_map_initialize(
    iree_arena_allocator_t* arena,
    loom_linearize_view_accesses_view_map_t* map) {
  map->count = 0;
  map->capacity = 16;
  return iree_arena_allocate_array(
      arena, map->capacity,
      sizeof(loom_linearize_view_accesses_view_map_entry_t),
      (void**)&map->entries);
}

static iree_status_t loom_linearize_view_accesses_view_map_push(
    iree_arena_allocator_t* arena, loom_linearize_view_accesses_view_map_t* map,
    loom_op_t* original_view_op, loom_value_id_t linear_view) {
  if (map->count >= map->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, map->count, map->count + 1,
        sizeof(loom_linearize_view_accesses_view_map_entry_t), &map->capacity,
        (void**)&map->entries));
  }
  map->entries[map->count++] = (loom_linearize_view_accesses_view_map_entry_t){
      .original_view_op = original_view_op,
      .linear_view = linear_view,
  };
  return iree_ok_status();
}

static iree_status_t loom_linearize_view_accesses_collect_access(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  loom_linearize_view_accesses_collect_context_t* collect_context =
      (loom_linearize_view_accesses_collect_context_t*)user_data;
  *out_result = LOOM_WALK_CONTINUE;
  if (!loom_view_load_isa(op) && !loom_view_store_isa(op) &&
      !loom_vector_load_isa(op) && !loom_vector_store_isa(op)) {
    return iree_ok_status();
  }
  return loom_linearize_view_accesses_op_list_push(
      collect_context->arena, collect_context->accesses, op);
}

//===----------------------------------------------------------------------===//
// Linearization
//===----------------------------------------------------------------------===//

static bool loom_linearize_view_accesses_static_dense_view_type(
    const loom_fact_context_t* fact_context, const loom_module_t* module,
    loom_type_t view_type, int64_t* out_length) {
  *out_length = 0;
  if (!loom_type_is_view(view_type) || loom_type_rank(view_type) <= 1 ||
      !loom_type_is_all_static(view_type)) {
    return false;
  }

  loom_value_facts_t stride_storage[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {
      0};
  loom_value_fact_address_layout_t layout = {0};
  if (!loom_encoding_query_type_address_layout(
          fact_context, module, view_type, stride_storage,
          IREE_ARRAYSIZE(stride_storage), &layout)) {
    return false;
  }
  if (layout.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE) return false;

  int64_t length = 1;
  uint8_t rank = loom_type_rank(view_type);
  for (uint8_t axis = 0; axis < rank; ++axis) {
    int64_t dim = loom_type_dim_static_size_at(view_type, axis);
    if (dim <= 0 || !loom_checked_mul_i64(length, dim, &length)) {
      return false;
    }
  }
  *out_length = length;
  return true;
}

static bool loom_linearize_view_accesses_axis_index(
    loom_module_t* module, loom_attribute_t static_indices,
    loom_value_slice_t dynamic_indices, uint8_t axis,
    uint16_t* dynamic_index_position, int64_t* out_static_index,
    loom_value_id_t* out_dynamic_index) {
  *out_static_index = 0;
  *out_dynamic_index = LOOM_VALUE_ID_INVALID;
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      axis >= static_indices.count) {
    return false;
  }

  int64_t static_index = static_indices.i64_array[axis];
  if (static_index != INT64_MIN) {
    *out_static_index = static_index;
    return true;
  }
  if (*dynamic_index_position >= dynamic_indices.count) return false;
  *out_static_index = INT64_MIN;
  *out_dynamic_index = dynamic_indices.values[(*dynamic_index_position)++];
  return true;
}

static iree_status_t loom_linearize_view_accesses_build_index_constant(
    loom_builder_t* builder, int64_t value, iree_string_view_t name,
    loom_location_id_t location, loom_value_id_t* out_index) {
  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      builder, loom_attr_i64(value), loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      location, &constant_op));
  *out_index = loom_index_constant_result(constant_op);
  if (name.size != 0) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_module_intern_string(builder->module, name, &name_id));
    loom_module_value(builder->module, *out_index)->name_id = name_id;
  }
  return iree_ok_status();
}

static bool loom_linearize_view_accesses_facts_cover_range(
    loom_value_facts_t facts, int64_t lower_bound, int64_t upper_bound) {
  return !loom_value_facts_is_float(facts) && facts.range_lo >= lower_bound &&
         facts.range_hi <= upper_bound;
}

static loom_value_facts_t loom_linearize_view_accesses_apply_index_range(
    loom_value_facts_t facts, loom_value_id_t value, int64_t lower_bound,
    int64_t upper_bound) {
  loom_predicate_t predicate = {
      .kind = LOOM_PREDICATE_RANGE,
      .arg_count = 3,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                   LOOM_PRED_ARG_CONST},
      .args = {value, lower_bound, upper_bound},
  };
  loom_value_facts_apply_predicate(&facts, &predicate);
  return facts;
}

static iree_status_t loom_linearize_view_accesses_name_bounded_index(
    loom_builder_t* builder, loom_value_id_t source_index,
    loom_value_id_t bounded_index, int64_t extent) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t source_name_id =
      loom_module_value(builder->module, source_index)->name_id;
  if (source_name_id != LOOM_STRING_ID_INVALID &&
      source_name_id < builder->module->strings.count) {
    iree_string_view_t source_name =
        builder->module->strings.entries[source_name_id];
    char name[128] = {0};
    static const char kBoundedSuffix[] = "_bounded";
    const iree_host_size_t suffix_length = sizeof(kBoundedSuffix) - 1;
    if (source_name.size + suffix_length < sizeof(name)) {
      memcpy(name, source_name.data, source_name.size);
      memcpy(name + source_name.size, kBoundedSuffix, suffix_length);
      IREE_RETURN_IF_ERROR(loom_module_intern_string(
          builder->module,
          iree_make_string_view(name, source_name.size + suffix_length),
          &name_id));
    }
  }
  if (name_id == LOOM_STRING_ID_INVALID) {
    char name[32] = {0};
    iree_snprintf(name, sizeof(name), "axis%" PRIi64, extent);
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        builder->module, iree_make_cstring_view(name), &name_id));
  }
  loom_module_value(builder->module, bounded_index)->name_id = name_id;
  return iree_ok_status();
}

static iree_status_t loom_linearize_view_accesses_name_linear_index(
    loom_builder_t* builder, loom_value_id_t linear_index, int64_t stride) {
  char linear_name[32] = {0};
  iree_snprintf(linear_name, sizeof(linear_name), "linear%" PRIi64, stride);
  loom_string_id_t linear_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      builder->module, iree_make_cstring_view(linear_name), &linear_name_id));
  loom_module_value(builder->module, linear_index)->name_id = linear_name_id;
  return iree_ok_status();
}

static iree_status_t loom_linearize_view_accesses_name_assumed_linear_index(
    loom_builder_t* builder, loom_value_id_t linear_index, int64_t stride) {
  char linear_name[32] = {0};
  iree_snprintf(linear_name, sizeof(linear_name), "linear%" PRIi64 "_bounded",
                stride);
  loom_string_id_t linear_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      builder->module, iree_make_cstring_view(linear_name), &linear_name_id));
  loom_module_value(builder->module, linear_index)->name_id = linear_name_id;
  return iree_ok_status();
}

static iree_status_t
loom_linearize_view_accesses_materialize_dynamic_axis_index(
    loom_builder_t* builder, const loom_value_fact_table_t* fact_table,
    int64_t extent, loom_value_id_t dynamic_index, loom_location_id_t location,
    loom_value_id_t* out_index, loom_value_facts_t* out_facts) {
  const int64_t upper_bound = extent - 1;
  loom_value_facts_t input_facts =
      fact_table ? loom_value_fact_table_lookup(fact_table, dynamic_index)
                 : loom_value_facts_unknown();
  *out_facts = loom_linearize_view_accesses_apply_index_range(
      input_facts, dynamic_index, 0, upper_bound);
  if (loom_linearize_view_accesses_facts_cover_range(input_facts, 0,
                                                     upper_bound)) {
    *out_index = dynamic_index;
    return iree_ok_status();
  }

  loom_predicate_t predicate = {
      .kind = LOOM_PREDICATE_RANGE,
      .arg_count = 3,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                   LOOM_PRED_ARG_CONST},
      .args = {dynamic_index, 0, upper_bound},
  };
  loom_type_t result_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* assume_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_assume_build(builder, &dynamic_index, 1,
                                               &predicate, 1, &result_type, 1,
                                               location, &assume_op));
  *out_index = loom_index_assume_results(assume_op).values[0];
  return loom_linearize_view_accesses_name_bounded_index(builder, dynamic_index,
                                                         *out_index, extent);
}

static iree_status_t loom_linearize_view_accesses_build_linear_index(
    loom_builder_t* builder, loom_module_t* module, loom_type_t view_type,
    loom_attribute_t static_indices, loom_value_slice_t dynamic_indices,
    const loom_value_fact_table_t* fact_table,
    const int64_t* axis_origin_counts, loom_location_id_t location,
    loom_value_id_t* out_linear_index, int64_t* out_static_linear_index,
    int64_t* out_outer_stride, loom_value_facts_t* out_linear_facts,
    bool* out_generated_linear_index) {
  *out_linear_index = LOOM_VALUE_ID_INVALID;
  *out_static_linear_index = INT64_MIN;
  *out_outer_stride = 1;
  *out_linear_facts = loom_value_facts_unknown();
  *out_generated_linear_index = false;
  uint8_t rank = loom_type_rank(view_type);
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      static_indices.count != rank) {
    return iree_ok_status();
  }

  int64_t static_axis_indices[LOOM_TYPE_MAX_RANK] = {0};
  loom_value_id_t dynamic_axis_indices[LOOM_TYPE_MAX_RANK] = {
      LOOM_VALUE_ID_INVALID};
  uint16_t dynamic_index_position = 0;
  for (uint8_t axis = 0; axis < rank; ++axis) {
    dynamic_axis_indices[axis] = LOOM_VALUE_ID_INVALID;
    if (!loom_linearize_view_accesses_axis_index(
            module, static_indices, dynamic_indices, axis,
            &dynamic_index_position, &static_axis_indices[axis],
            &dynamic_axis_indices[axis])) {
      return iree_ok_status();
    }
  }
  if (dynamic_index_position != dynamic_indices.count) return iree_ok_status();

  int64_t stride = 1;
  int64_t static_offset = 0;
  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  loom_value_facts_t linear_facts = loom_value_facts_exact_i64(0);

  for (uint8_t reverse_axis = 0; reverse_axis < rank; ++reverse_axis) {
    uint8_t axis = (uint8_t)(rank - reverse_axis - 1);
    if (reverse_axis != 0) {
      int64_t next_dim =
          loom_type_dim_static_size_at(view_type, (iree_host_size_t)(axis + 1));
      if (!loom_checked_mul_i64(stride, next_dim, &stride)) {
        return iree_ok_status();
      }
    }

    const int64_t axis_origin_count =
        axis_origin_counts ? axis_origin_counts[axis]
                           : loom_type_dim_static_size_at(view_type, axis);
    if (dynamic_axis_indices[axis] == LOOM_VALUE_ID_INVALID) {
      if (static_axis_indices[axis] == 0) continue;
      int64_t contribution = 0;
      if (!loom_checked_mul_i64(static_axis_indices[axis], stride,
                                &contribution) ||
          !loom_checked_add_i64(static_offset, contribution, &static_offset)) {
        return iree_ok_status();
      }
      loom_value_facts_t contribution_facts =
          loom_value_facts_exact_i64(contribution);
      loom_value_facts_addi(&linear_facts, &contribution_facts, &linear_facts);
      continue;
    }

    loom_value_facts_t dynamic_facts =
        fact_table ? loom_value_fact_table_lookup(fact_table,
                                                  dynamic_axis_indices[axis])
                   : loom_value_facts_unknown();
    int64_t dynamic_exact = 0;
    if (loom_value_facts_as_exact_i64(dynamic_facts, &dynamic_exact) &&
        dynamic_exact >= 0 && dynamic_exact < axis_origin_count) {
      if (dynamic_exact == 0) continue;
      int64_t contribution = 0;
      if (!loom_checked_mul_i64(dynamic_exact, stride, &contribution) ||
          !loom_checked_add_i64(static_offset, contribution, &static_offset)) {
        return iree_ok_status();
      }
      loom_value_facts_t contribution_facts =
          loom_value_facts_exact_i64(contribution);
      loom_value_facts_addi(&linear_facts, &contribution_facts, &linear_facts);
      continue;
    }

    loom_value_id_t stride_index = LOOM_VALUE_ID_INVALID;
    loom_value_id_t axis_index = LOOM_VALUE_ID_INVALID;
    loom_value_facts_t axis_facts = loom_value_facts_unknown();
    IREE_RETURN_IF_ERROR(
        loom_linearize_view_accesses_materialize_dynamic_axis_index(
            builder, fact_table, axis_origin_count, dynamic_axis_indices[axis],
            location, &axis_index, &axis_facts));

    loom_value_facts_t term_facts = axis_facts;
    if (stride != 1) {
      loom_value_facts_t stride_facts = loom_value_facts_exact_i64(stride);
      loom_value_facts_muli(&axis_facts, &stride_facts, &term_facts);
      char stride_name[32] = {0};
      iree_snprintf(stride_name, sizeof(stride_name), "stride%" PRIi64, stride);
      IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_build_index_constant(
          builder, stride, iree_make_cstring_view(stride_name), location,
          &stride_index));
    }
    loom_value_facts_addi(&linear_facts, &term_facts, &linear_facts);

    if (stride == 1) {
      if (accumulator == LOOM_VALUE_ID_INVALID) {
        accumulator = axis_index;
        continue;
      }
      loom_op_t* add_op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_add_build(
          builder, accumulator, axis_index,
          loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), location, &add_op));
      accumulator = loom_index_add_result(add_op);
      *out_generated_linear_index = true;
    } else if (accumulator == LOOM_VALUE_ID_INVALID) {
      loom_op_t* mul_op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_mul_build(
          builder, axis_index, stride_index,
          loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), location, &mul_op));
      accumulator = loom_index_mul_result(mul_op);
      *out_generated_linear_index = true;
    } else {
      loom_op_t* madd_op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_madd_build(
          builder, axis_index, stride_index, accumulator,
          loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), location, &madd_op));
      accumulator = loom_index_madd_result(madd_op);
      *out_generated_linear_index = true;
    }
    IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_name_linear_index(
        builder, accumulator, stride));
  }

  if (static_offset != 0) {
    if (accumulator == LOOM_VALUE_ID_INVALID) {
      *out_static_linear_index = static_offset;
      *out_outer_stride = stride;
      *out_linear_facts = linear_facts;
      return iree_ok_status();
    }
    loom_value_id_t offset_value = LOOM_VALUE_ID_INVALID;
    char offset_name[32] = {0};
    iree_snprintf(offset_name, sizeof(offset_name), "offset%" PRIi64,
                  static_offset);
    IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_build_index_constant(
        builder, static_offset, iree_make_cstring_view(offset_name), location,
        &offset_value));
    loom_op_t* add_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_add_build(
        builder, accumulator, offset_value,
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), location, &add_op));
    accumulator = loom_index_add_result(add_op);
    *out_generated_linear_index = true;
    IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_name_linear_index(
        builder, accumulator, stride));
  } else if (accumulator == LOOM_VALUE_ID_INVALID) {
    *out_static_linear_index = 0;
    *out_linear_facts = linear_facts;
    return iree_ok_status();
  }

  *out_linear_index = accumulator;
  if (accumulator == LOOM_VALUE_ID_INVALID) {
    *out_static_linear_index = static_offset;
  }
  *out_outer_stride = stride;
  *out_linear_facts = linear_facts;
  return iree_ok_status();
}

static iree_status_t loom_linearize_view_accesses_assume_linear_index_bounds(
    loom_builder_t* builder, loom_value_id_t linear_index,
    loom_value_facts_t linear_facts, int64_t origin_count, int64_t outer_stride,
    bool generated_linear_index, loom_location_id_t location,
    loom_value_id_t* out_bounded_index) {
  if (loom_linearize_view_accesses_facts_cover_range(linear_facts, 0,
                                                     origin_count - 1)) {
    if (generated_linear_index) {
      IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_name_linear_index(
          builder, linear_index, outer_stride));
    }
    *out_bounded_index = linear_index;
    return iree_ok_status();
  }

  loom_predicate_t predicate = {
      .kind = LOOM_PREDICATE_RANGE,
      .arg_count = 3,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                   LOOM_PRED_ARG_CONST},
      .args = {linear_index, 0, origin_count - 1},
  };
  loom_type_t result_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* assume_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_assume_build(builder, &linear_index, 1,
                                               &predicate, 1, &result_type, 1,
                                               location, &assume_op));
  *out_bounded_index = loom_index_assume_results(assume_op).values[0];
  return loom_linearize_view_accesses_name_assumed_linear_index(
      builder, *out_bounded_index, outer_stride);
}

static iree_status_t loom_linearize_view_accesses_get_linear_view(
    loom_linearize_view_accesses_context_t* context, loom_op_t* view_op,
    loom_type_t view_type, int64_t linear_length,
    loom_value_id_t* out_linear_view) {
  for (iree_host_size_t i = 0; i < context->view_map->count; ++i) {
    loom_linearize_view_accesses_view_map_entry_t entry =
        context->view_map->entries[i];
    if (entry.original_view_op == view_op) {
      *out_linear_view = entry.linear_view;
      return iree_ok_status();
    }
  }

  loom_type_t linear_type = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, loom_type_element_type(view_type),
      loom_dim_pack_static(linear_length), view_type.encoding_id);
  linear_type.encoding_flags = view_type.encoding_flags;

  loom_builder_t* builder = &context->rewriter->builder;
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_after(builder, view_op);
  loom_op_t* linear_view_op = NULL;
  iree_status_t status =
      loom_buffer_view_build(builder, loom_buffer_view_buffer(view_op),
                             loom_buffer_view_byte_offset(view_op), linear_type,
                             view_op->location, &linear_view_op);
  loom_builder_restore(builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  loom_value_id_t linear_view = loom_buffer_view_result(linear_view_op);
  loom_module_value(context->module, linear_view)->name_id =
      loom_module_value(context->module, loom_buffer_view_result(view_op))
          ->name_id;
  IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_view_map_push(
      context->pass->arena, context->view_map, view_op, linear_view));
  if (context->pass->statistics) {
    loom_pass_statistic_add(context->pass,
                            LOOM_LINEARIZE_VIEW_ACCESSES_STAT_VIEWS_CREATED, 1);
  }
  *out_linear_view = linear_view;
  return iree_ok_status();
}

static bool loom_linearize_view_accesses_get_source_view(
    const loom_fact_context_t* fact_context, loom_module_t* module,
    loom_value_id_t view, loom_op_t** out_view_op, loom_type_t* out_view_type,
    int64_t* out_linear_length) {
  *out_view_op = NULL;
  *out_view_type = loom_type_none();
  *out_linear_length = 0;
  loom_type_t view_type = loom_module_value_type(module, view);
  int64_t linear_length = 0;
  if (!loom_linearize_view_accesses_static_dense_view_type(
          fact_context, module, view_type, &linear_length)) {
    return false;
  }

  const loom_value_t* view_value = loom_module_value(module, view);
  if (loom_value_is_block_arg(view_value)) return false;
  loom_op_t* view_op = loom_value_def_op(view_value);
  if (!loom_buffer_view_isa(view_op)) return false;

  *out_view_op = view_op;
  *out_view_type = view_type;
  *out_linear_length = linear_length;
  return true;
}

static bool loom_linearize_view_accesses_static_contiguous_vector_access(
    const loom_fact_context_t* fact_context, loom_module_t* module,
    loom_type_t view_type, loom_type_t vector_type, int64_t linear_length,
    int64_t* out_lane_count, int64_t* out_origin_count) {
  *out_lane_count = 0;
  *out_origin_count = 0;
  loom_vector_memory_access_t access;
  if (!loom_vector_memory_access_describe(fact_context, module, view_type,
                                          vector_type, &access)) {
    return false;
  }
  if (access.layout_kind != LOOM_VECTOR_MEMORY_LAYOUT_DENSE) return false;
  if (access.vector_rank != 1) return false;
  if (access.first_vector_axis + 1 != access.view_rank) return false;
  if (loom_type_dim_is_dynamic_at(vector_type, 0)) return false;

  int64_t lane_count = loom_type_dim_static_size_at(vector_type, 0);
  if (lane_count <= 0 || lane_count > linear_length) return false;
  const int64_t trailing_axis_dim =
      loom_type_dim_static_size_at(view_type, access.view_rank - 1);
  if (lane_count > trailing_axis_dim) return false;
  *out_lane_count = lane_count;
  *out_origin_count = linear_length - lane_count + 1;
  return true;
}

static void loom_linearize_view_accesses_vector_axis_origin_counts(
    loom_type_t view_type, int64_t lane_count, int64_t* out_axis_counts) {
  const uint8_t rank = loom_type_rank(view_type);
  for (uint8_t axis = 0; axis < rank; ++axis) {
    out_axis_counts[axis] = loom_type_dim_static_size_at(view_type, axis);
  }
  out_axis_counts[rank - 1] -= lane_count - 1;
}

static iree_status_t loom_linearize_view_accesses_rewrite_load(
    loom_linearize_view_accesses_context_t* context, loom_op_t* load_op,
    bool* out_changed) {
  loom_op_t* view_op = NULL;
  loom_type_t view_type = loom_type_none();
  int64_t linear_length = 0;
  const loom_fact_context_t* fact_context =
      context->fact_table ? &context->fact_table->context : NULL;
  if (!loom_linearize_view_accesses_get_source_view(
          fact_context, context->module, loom_view_load_view(load_op), &view_op,
          &view_type, &linear_length)) {
    return iree_ok_status();
  }

  loom_builder_t* builder = &context->rewriter->builder;
  loom_builder_set_before(builder, load_op);
  loom_value_id_t linear_index = LOOM_VALUE_ID_INVALID;
  int64_t static_linear_index = INT64_MIN;
  int64_t outer_stride = 1;
  loom_value_facts_t linear_facts = loom_value_facts_unknown();
  bool generated_linear_index = false;
  IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_build_linear_index(
      builder, context->module, view_type,
      loom_view_load_static_indices(load_op), loom_view_load_indices(load_op),
      context->fact_table, /*axis_origin_counts=*/NULL, load_op->location,
      &linear_index, &static_linear_index, &outer_stride, &linear_facts,
      &generated_linear_index));
  if (linear_index == LOOM_VALUE_ID_INVALID &&
      static_linear_index == INT64_MIN) {
    return iree_ok_status();
  }
  if (linear_index != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_linearize_view_accesses_assume_linear_index_bounds(
            builder, linear_index, linear_facts, linear_length, outer_stride,
            generated_linear_index, load_op->location, &linear_index));
  }

  loom_value_id_t linear_view = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_get_linear_view(
      context, view_op, view_type, linear_length, &linear_view));

  int64_t static_indices[] = {
      linear_index == LOOM_VALUE_ID_INVALID ? static_linear_index : INT64_MIN};
  loom_value_id_t dynamic_indices[] = {linear_index};
  iree_host_size_t dynamic_index_count = linear_index == LOOM_VALUE_ID_INVALID
                                             ? 0
                                             : IREE_ARRAYSIZE(dynamic_indices);
  loom_view_load_build_flags_t build_flags = 0;
  loom_attribute_t cache_scope = loom_memory_access_cache_scope(
      loom_memory_access_cast(context->module, load_op));
  loom_attribute_t cache_temporal = loom_memory_access_cache_temporal(
      loom_memory_access_cast(context->module, load_op));
  if (cache_scope.kind != 0) {
    build_flags |= LOOM_VIEW_LOAD_BUILD_FLAG_HAS_CACHE_SCOPE;
  }
  if (cache_temporal.kind != 0) {
    build_flags |= LOOM_VIEW_LOAD_BUILD_FLAG_HAS_CACHE_TEMPORAL;
  }

  loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(context->rewriter);
  loom_op_t* linear_load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_view_load_build(
      builder, build_flags, linear_view, dynamic_indices, dynamic_index_count,
      static_indices, IREE_ARRAYSIZE(static_indices),
      loom_view_load_cache_scope(load_op),
      loom_view_load_cache_temporal(load_op),
      loom_module_value_type(context->module, loom_view_load_result(load_op)),
      load_op->location, &linear_load_op));
  loom_value_id_t replacement = loom_view_load_result(linear_load_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      context->rewriter, load_op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      context->rewriter, load_op, &replacement, 1));

  if (context->pass->statistics) {
    loom_pass_statistic_add(
        context->pass, LOOM_LINEARIZE_VIEW_ACCESSES_STAT_LOADS_LINEARIZED, 1);
  }
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_linearize_view_accesses_rewrite_store(
    loom_linearize_view_accesses_context_t* context, loom_op_t* store_op,
    bool* out_changed) {
  loom_op_t* view_op = NULL;
  loom_type_t view_type = loom_type_none();
  int64_t linear_length = 0;
  const loom_fact_context_t* fact_context =
      context->fact_table ? &context->fact_table->context : NULL;
  if (!loom_linearize_view_accesses_get_source_view(
          fact_context, context->module, loom_view_store_view(store_op),
          &view_op, &view_type, &linear_length)) {
    return iree_ok_status();
  }

  loom_builder_t* builder = &context->rewriter->builder;
  loom_builder_set_before(builder, store_op);
  loom_value_id_t linear_index = LOOM_VALUE_ID_INVALID;
  int64_t static_linear_index = INT64_MIN;
  int64_t outer_stride = 1;
  loom_value_facts_t linear_facts = loom_value_facts_unknown();
  bool generated_linear_index = false;
  IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_build_linear_index(
      builder, context->module, view_type,
      loom_view_store_static_indices(store_op),
      loom_view_store_indices(store_op), context->fact_table,
      /*axis_origin_counts=*/NULL, store_op->location, &linear_index,
      &static_linear_index, &outer_stride, &linear_facts,
      &generated_linear_index));
  if (linear_index == LOOM_VALUE_ID_INVALID &&
      static_linear_index == INT64_MIN) {
    return iree_ok_status();
  }
  if (linear_index != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_linearize_view_accesses_assume_linear_index_bounds(
            builder, linear_index, linear_facts, linear_length, outer_stride,
            generated_linear_index, store_op->location, &linear_index));
  }

  loom_value_id_t linear_view = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_get_linear_view(
      context, view_op, view_type, linear_length, &linear_view));

  int64_t static_indices[] = {
      linear_index == LOOM_VALUE_ID_INVALID ? static_linear_index : INT64_MIN};
  loom_value_id_t dynamic_indices[] = {linear_index};
  iree_host_size_t dynamic_index_count = linear_index == LOOM_VALUE_ID_INVALID
                                             ? 0
                                             : IREE_ARRAYSIZE(dynamic_indices);
  loom_view_store_build_flags_t build_flags = 0;
  loom_attribute_t cache_scope = loom_memory_access_cache_scope(
      loom_memory_access_cast(context->module, store_op));
  loom_attribute_t cache_temporal = loom_memory_access_cache_temporal(
      loom_memory_access_cast(context->module, store_op));
  if (cache_scope.kind != 0) {
    build_flags |= LOOM_VIEW_STORE_BUILD_FLAG_HAS_CACHE_SCOPE;
  }
  if (cache_temporal.kind != 0) {
    build_flags |= LOOM_VIEW_STORE_BUILD_FLAG_HAS_CACHE_TEMPORAL;
  }

  loom_op_t* linear_store_op = NULL;
  IREE_RETURN_IF_ERROR(loom_view_store_build(
      builder, build_flags, loom_view_store_value(store_op), linear_view,
      dynamic_indices, dynamic_index_count, static_indices,
      IREE_ARRAYSIZE(static_indices), loom_view_store_cache_scope(store_op),
      loom_view_store_cache_temporal(store_op), store_op->location,
      &linear_store_op));
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(context->rewriter, store_op));

  if (context->pass->statistics) {
    loom_pass_statistic_add(
        context->pass, LOOM_LINEARIZE_VIEW_ACCESSES_STAT_STORES_LINEARIZED, 1);
  }
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_linearize_view_accesses_rewrite_vector_load(
    loom_linearize_view_accesses_context_t* context, loom_op_t* load_op,
    bool* out_changed) {
  loom_op_t* view_op = NULL;
  loom_type_t view_type = loom_type_none();
  int64_t linear_length = 0;
  const loom_fact_context_t* fact_context =
      context->fact_table ? &context->fact_table->context : NULL;
  if (!loom_linearize_view_accesses_get_source_view(
          fact_context, context->module, loom_vector_load_view(load_op),
          &view_op, &view_type, &linear_length)) {
    return iree_ok_status();
  }

  loom_type_t result_type =
      loom_module_value_type(context->module, loom_vector_load_result(load_op));
  int64_t lane_count = 0;
  int64_t origin_count = 0;
  if (!loom_linearize_view_accesses_static_contiguous_vector_access(
          fact_context, context->module, view_type, result_type, linear_length,
          &lane_count, &origin_count)) {
    return iree_ok_status();
  }

  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_vector_memory_cache_policy_from_op(context->module, load_op,
                                               &cache_policy)) {
    return iree_ok_status();
  }

  loom_builder_t* builder = &context->rewriter->builder;
  loom_builder_set_before(builder, load_op);
  int64_t axis_origin_counts[LOOM_TYPE_MAX_RANK] = {0};
  loom_linearize_view_accesses_vector_axis_origin_counts(view_type, lane_count,
                                                         axis_origin_counts);
  loom_value_id_t linear_index = LOOM_VALUE_ID_INVALID;
  int64_t static_linear_index = INT64_MIN;
  int64_t outer_stride = 1;
  loom_value_facts_t linear_facts = loom_value_facts_unknown();
  bool generated_linear_index = false;
  IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_build_linear_index(
      builder, context->module, view_type,
      loom_vector_load_static_indices(load_op),
      loom_vector_load_indices(load_op), context->fact_table,
      axis_origin_counts, load_op->location, &linear_index,
      &static_linear_index, &outer_stride, &linear_facts,
      &generated_linear_index));
  if (linear_index == LOOM_VALUE_ID_INVALID &&
      static_linear_index == INT64_MIN) {
    return iree_ok_status();
  }
  if (linear_index != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_linearize_view_accesses_assume_linear_index_bounds(
            builder, linear_index, linear_facts, origin_count, outer_stride,
            generated_linear_index, load_op->location, &linear_index));
  }

  loom_value_id_t linear_view = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_get_linear_view(
      context, view_op, view_type, linear_length, &linear_view));

  int64_t static_indices[] = {
      linear_index == LOOM_VALUE_ID_INVALID ? static_linear_index : INT64_MIN};
  loom_value_id_t dynamic_indices[] = {linear_index};
  iree_host_size_t dynamic_index_count = linear_index == LOOM_VALUE_ID_INVALID
                                             ? 0
                                             : IREE_ARRAYSIZE(dynamic_indices);
  loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(context->rewriter);
  loom_op_t* linear_load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_load_build(
      builder, cache_policy.build_flags, linear_view, dynamic_indices,
      dynamic_index_count, static_indices, IREE_ARRAYSIZE(static_indices),
      cache_policy.cache_scope, cache_policy.cache_temporal, result_type,
      load_op->location, &linear_load_op));
  loom_value_id_t replacement = loom_vector_load_result(linear_load_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      context->rewriter, load_op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      context->rewriter, load_op, &replacement, 1));

  if (context->pass->statistics) {
    loom_pass_statistic_add(
        context->pass,
        LOOM_LINEARIZE_VIEW_ACCESSES_STAT_VECTOR_LOADS_LINEARIZED, 1);
  }
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_linearize_view_accesses_rewrite_vector_store(
    loom_linearize_view_accesses_context_t* context, loom_op_t* store_op,
    bool* out_changed) {
  loom_op_t* view_op = NULL;
  loom_type_t view_type = loom_type_none();
  int64_t linear_length = 0;
  const loom_fact_context_t* fact_context =
      context->fact_table ? &context->fact_table->context : NULL;
  if (!loom_linearize_view_accesses_get_source_view(
          fact_context, context->module, loom_vector_store_view(store_op),
          &view_op, &view_type, &linear_length)) {
    return iree_ok_status();
  }

  loom_type_t value_type = loom_module_value_type(
      context->module, loom_vector_store_value(store_op));
  int64_t lane_count = 0;
  int64_t origin_count = 0;
  if (!loom_linearize_view_accesses_static_contiguous_vector_access(
          fact_context, context->module, view_type, value_type, linear_length,
          &lane_count, &origin_count)) {
    return iree_ok_status();
  }

  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_vector_memory_cache_policy_from_op(context->module, store_op,
                                               &cache_policy)) {
    return iree_ok_status();
  }

  loom_builder_t* builder = &context->rewriter->builder;
  loom_builder_set_before(builder, store_op);
  int64_t axis_origin_counts[LOOM_TYPE_MAX_RANK] = {0};
  loom_linearize_view_accesses_vector_axis_origin_counts(view_type, lane_count,
                                                         axis_origin_counts);
  loom_value_id_t linear_index = LOOM_VALUE_ID_INVALID;
  int64_t static_linear_index = INT64_MIN;
  int64_t outer_stride = 1;
  loom_value_facts_t linear_facts = loom_value_facts_unknown();
  bool generated_linear_index = false;
  IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_build_linear_index(
      builder, context->module, view_type,
      loom_vector_store_static_indices(store_op),
      loom_vector_store_indices(store_op), context->fact_table,
      axis_origin_counts, store_op->location, &linear_index,
      &static_linear_index, &outer_stride, &linear_facts,
      &generated_linear_index));
  if (linear_index == LOOM_VALUE_ID_INVALID &&
      static_linear_index == INT64_MIN) {
    return iree_ok_status();
  }
  if (linear_index != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_linearize_view_accesses_assume_linear_index_bounds(
            builder, linear_index, linear_facts, origin_count, outer_stride,
            generated_linear_index, store_op->location, &linear_index));
  }

  loom_value_id_t linear_view = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_get_linear_view(
      context, view_op, view_type, linear_length, &linear_view));

  int64_t static_indices[] = {
      linear_index == LOOM_VALUE_ID_INVALID ? static_linear_index : INT64_MIN};
  loom_value_id_t dynamic_indices[] = {linear_index};
  iree_host_size_t dynamic_index_count = linear_index == LOOM_VALUE_ID_INVALID
                                             ? 0
                                             : IREE_ARRAYSIZE(dynamic_indices);
  loom_op_t* linear_store_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_store_build(
      builder, cache_policy.build_flags, loom_vector_store_value(store_op),
      linear_view, dynamic_indices, dynamic_index_count, static_indices,
      IREE_ARRAYSIZE(static_indices), cache_policy.cache_scope,
      cache_policy.cache_temporal, store_op->location, &linear_store_op));
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(context->rewriter, store_op));

  if (context->pass->statistics) {
    loom_pass_statistic_add(
        context->pass,
        LOOM_LINEARIZE_VIEW_ACCESSES_STAT_VECTOR_STORES_LINEARIZED, 1);
  }
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_linearize_view_accesses_rewrite_access(
    loom_linearize_view_accesses_context_t* context, loom_op_t* op,
    bool* out_changed) {
  if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
    return iree_ok_status();
  }
  if (loom_view_load_isa(op)) {
    return loom_linearize_view_accesses_rewrite_load(context, op, out_changed);
  }
  if (loom_view_store_isa(op)) {
    return loom_linearize_view_accesses_rewrite_store(context, op, out_changed);
  }
  if (loom_vector_load_isa(op)) {
    return loom_linearize_view_accesses_rewrite_vector_load(context, op,
                                                            out_changed);
  }
  if (loom_vector_store_isa(op)) {
    return loom_linearize_view_accesses_rewrite_vector_store(context, op,
                                                             out_changed);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Pass entry point
//===----------------------------------------------------------------------===//

iree_status_t loom_linearize_view_accesses_run(loom_pass_t* pass,
                                               loom_module_t* module,
                                               loom_func_like_t function) {
  if (!loom_func_like_body(function)) return iree_ok_status();

  loom_value_fact_table_t* fact_table = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_value_facts_acquire(
      pass, module, loom_pass_value_fact_scope_function(function),
      &fact_table));

  loom_linearize_view_accesses_op_list_t accesses = {0};
  IREE_RETURN_IF_ERROR(
      loom_linearize_view_accesses_op_list_initialize(pass->arena, &accesses));
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  loom_linearize_view_accesses_collect_context_t collect_context = {
      .arena = pass->arena,
      .accesses = &accesses,
  };
  IREE_RETURN_IF_ERROR(
      loom_walk_function(module, function, LOOM_WALK_PRE_ORDER,
                         (loom_walk_callback_t){
                             .fn = loom_linearize_view_accesses_collect_access,
                             .user_data = &collect_context,
                         },
                         pass->arena, &walk_result));

  loom_linearize_view_accesses_view_map_t view_map = {0};
  IREE_RETURN_IF_ERROR(
      loom_linearize_view_accesses_view_map_initialize(pass->arena, &view_map));

  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));
  iree_status_t status = iree_ok_status();
  bool changed = false;
  loom_linearize_view_accesses_context_t context = {
      .pass = pass,
      .module = module,
      .rewriter = &rewriter,
      .fact_table = fact_table,
      .view_map = &view_map,
  };
  for (iree_host_size_t i = 0; i < accesses.count && iree_status_is_ok(status);
       ++i) {
    bool access_changed = false;
    status = loom_linearize_view_accesses_rewrite_access(
        &context, accesses.ops[i], &access_changed);
    changed = changed || access_changed;
  }
  if (changed) {
    loom_pass_value_fact_owner_invalidate(pass->value_facts);
    if (iree_status_is_ok(status)) loom_pass_mark_changed(pass);
  }

  loom_rewriter_deinitialize(&rewriter);
  return status;
}
