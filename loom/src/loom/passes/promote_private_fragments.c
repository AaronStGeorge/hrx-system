// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/passes/promote_private_fragments.h"

#include <stdint.h>
#include <string.h>

#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/walk.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

enum {
  LOOM_PROMOTE_PRIVATE_FRAGMENTS_STAT_FRAGMENTS_PROMOTED = 0,
  LOOM_PROMOTE_PRIVATE_FRAGMENTS_STAT_LOADS_PROMOTED = 1,
};

static const loom_pass_statistic_def_t kPromotePrivateFragmentsStatistics[] = {
    {IREE_SVL("fragments-promoted"),
     IREE_SVL("Number of private fragment views promoted to SSA vectors.")},
    {IREE_SVL("loads-promoted"),
     IREE_SVL("Number of private scalar loads replaced with vector.extract.")},
};

static const loom_pass_info_t loom_promote_private_fragments_pass_info_storage =
    {
        .name = IREE_SVL("promote-private-fragments"),
        .description =
            IREE_SVL("Promote simple private fragment buffers to SSA vectors."),
        .kind = LOOM_PASS_FUNCTION,
        .statistic_defs = kPromotePrivateFragmentsStatistics,
        .statistic_count = IREE_ARRAYSIZE(kPromotePrivateFragmentsStatistics),
};

const loom_pass_info_t* loom_promote_private_fragments_pass_info(void) {
  return &loom_promote_private_fragments_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Candidate collection
//===----------------------------------------------------------------------===//

#define LOOM_PROMOTE_PRIVATE_FRAGMENTS_INITIAL_VIEW_CAPACITY 16

typedef struct loom_promote_private_fragments_view_list_t {
  // Collected buffer.view operations in dominance order.
  loom_op_t** ops;
  // Number of collected view operations.
  iree_host_size_t count;
  // Allocated view operation pointer capacity.
  iree_host_size_t capacity;
} loom_promote_private_fragments_view_list_t;

typedef struct loom_promote_private_fragments_collect_context_t {
  // Pass scratch arena used for the collected view pointer list.
  iree_arena_allocator_t* arena;
  // Collected buffer.view operations.
  loom_promote_private_fragments_view_list_t* views;
} loom_promote_private_fragments_collect_context_t;

static iree_status_t loom_promote_private_fragments_view_list_initialize(
    iree_arena_allocator_t* arena,
    loom_promote_private_fragments_view_list_t* list) {
  list->count = 0;
  list->capacity = LOOM_PROMOTE_PRIVATE_FRAGMENTS_INITIAL_VIEW_CAPACITY;
  return iree_arena_allocate_array(arena, list->capacity, sizeof(loom_op_t*),
                                   (void**)&list->ops);
}

static iree_status_t loom_promote_private_fragments_view_list_push(
    iree_arena_allocator_t* arena,
    loom_promote_private_fragments_view_list_t* list, loom_op_t* op) {
  if (list->count >= list->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, list->count, list->count + 1, sizeof(loom_op_t*),
        &list->capacity, (void**)&list->ops));
  }
  list->ops[list->count++] = op;
  return iree_ok_status();
}

static iree_status_t loom_promote_private_fragments_collect_view(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  loom_promote_private_fragments_collect_context_t* collect_context =
      (loom_promote_private_fragments_collect_context_t*)user_data;
  *out_result = LOOM_WALK_CONTINUE;
  if (!loom_buffer_view_isa(op)) return iree_ok_status();
  return loom_promote_private_fragments_view_list_push(
      collect_context->arena, collect_context->views, op);
}

//===----------------------------------------------------------------------===//
// Pattern model
//===----------------------------------------------------------------------===//

typedef struct loom_promote_private_fragments_copy_loop_t {
  // Copy loop to erase after materializing the vector load.
  loom_op_t* loop_op;
  // Scalar source load inside the copy loop.
  loom_op_t* source_load_op;
  // Scalar store to the private fragment inside the copy loop.
  loom_op_t* private_store_op;
  // Source view loaded by the copy loop.
  loom_value_id_t source_view;
  // Loop induction variable.
  loom_value_id_t induction_variable;
  // Static vector.load origin indices.
  int64_t static_indices[LOOM_TYPE_MAX_RANK];
  // Number of static vector.load origin indices.
  uint16_t static_index_count;
  // Dynamic vector.load origin operands.
  loom_value_id_t indices[LOOM_TYPE_MAX_RANK];
  // Number of dynamic vector.load origin operands.
  uint16_t index_count;
  // Result type of the replacement vector.load.
  loom_type_t vector_type;
} loom_promote_private_fragments_copy_loop_t;

typedef struct loom_promote_private_fragments_load_list_t {
  // Private view.load operations to replace with vector.extract.
  loom_op_t** ops;
  // Number of collected private loads.
  iree_host_size_t count;
  // Allocated private load pointer capacity.
  iree_host_size_t capacity;
} loom_promote_private_fragments_load_list_t;

static iree_status_t loom_promote_private_fragments_load_list_push(
    iree_arena_allocator_t* arena,
    loom_promote_private_fragments_load_list_t* list, loom_op_t* op) {
  if (list->count >= list->capacity) {
    iree_host_size_t new_count = list->count + 1;
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(arena, list->count, new_count, sizeof(loom_op_t*),
                              &list->capacity, (void**)&list->ops));
  }
  list->ops[list->count++] = op;
  return iree_ok_status();
}

static bool loom_promote_private_fragments_exact_index_constant(
    const loom_module_t* module, loom_value_id_t value_id, int64_t* out_value) {
  if (value_id == LOOM_VALUE_ID_INVALID) return false;
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) return false;
  const loom_op_t* op = loom_value_def_op(value);
  if (!loom_index_constant_isa(op)) return false;
  loom_attribute_t attr = loom_index_constant_value(op);
  if (attr.kind != LOOM_ATTR_I64) return false;
  *out_value = attr.i64;
  return true;
}

static bool loom_promote_private_fragments_is_private_rank1_view(
    const loom_module_t* module, const loom_op_t* view_op, int64_t* out_length,
    loom_op_t** out_alloca_op) {
  if (!loom_buffer_view_isa(view_op)) return false;
  loom_value_id_t view = loom_buffer_view_result(view_op);
  loom_type_t view_type = loom_module_value_type(module, view);
  if (!loom_type_is_view(view_type) || loom_type_rank(view_type) != 1 ||
      loom_type_dim_is_dynamic_at(view_type, 0)) {
    return false;
  }
  int64_t length = loom_type_dim_static_size_at(view_type, 0);
  if (length <= 0) return false;

  const loom_value_t* buffer_value =
      loom_module_value(module, loom_buffer_view_buffer(view_op));
  if (loom_value_is_block_arg(buffer_value)) return false;
  loom_op_t* alloca_op = loom_value_def_op(buffer_value);
  if (!loom_buffer_alloca_isa(alloca_op)) return false;
  if (loom_buffer_alloca_memory_space(alloca_op) !=
      LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE) {
    return false;
  }

  *out_length = length;
  *out_alloca_op = alloca_op;
  return true;
}

static bool loom_promote_private_fragments_is_rank1_dynamic_index(
    loom_attribute_t static_indices, loom_value_slice_t indices,
    loom_value_id_t index) {
  return static_indices.kind == LOOM_ATTR_I64_ARRAY &&
         static_indices.count == 1 &&
         static_indices.i64_array[0] == INT64_MIN && indices.count == 1 &&
         indices.values[0] == index;
}

static bool loom_promote_private_fragments_source_origin_from_copy_load(
    const loom_module_t* module, const loom_op_t* source_load_op,
    loom_value_id_t induction_variable, int64_t fragment_length,
    loom_promote_private_fragments_copy_loop_t* out_copy_loop) {
  loom_value_id_t source_view = loom_view_load_view(source_load_op);
  loom_type_t source_view_type = loom_module_value_type(module, source_view);
  if (!loom_type_is_view(source_view_type)) return false;
  uint8_t source_rank = loom_type_rank(source_view_type);
  if (source_rank == 0 || source_rank > LOOM_TYPE_MAX_RANK) return false;
  uint8_t vector_axis = (uint8_t)(source_rank - 1);
  if (loom_type_dim_is_dynamic_at(source_view_type, vector_axis) ||
      loom_type_dim_static_size_at(source_view_type, vector_axis) !=
          fragment_length) {
    return false;
  }

  loom_attribute_t source_static_indices =
      loom_view_load_static_indices(source_load_op);
  loom_value_slice_t source_indices = loom_view_load_indices(source_load_op);
  if (source_static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      source_static_indices.count != source_rank) {
    return false;
  }

  uint16_t dynamic_index_ordinal = 0;
  bool found_induction_variable = false;
  for (uint8_t axis = 0; axis < source_rank; ++axis) {
    int64_t static_index = source_static_indices.i64_array[axis];
    if (static_index != INT64_MIN) {
      out_copy_loop->static_indices[axis] = static_index;
      continue;
    }
    if (dynamic_index_ordinal >= source_indices.count) return false;
    loom_value_id_t dynamic_index =
        source_indices.values[dynamic_index_ordinal];
    if (dynamic_index == induction_variable) {
      if (axis != vector_axis || found_induction_variable) return false;
      out_copy_loop->static_indices[axis] = 0;
      found_induction_variable = true;
    } else {
      out_copy_loop->static_indices[axis] = INT64_MIN;
      out_copy_loop->indices[out_copy_loop->index_count++] = dynamic_index;
    }
    ++dynamic_index_ordinal;
  }
  if (!found_induction_variable ||
      dynamic_index_ordinal != source_indices.count) {
    return false;
  }

  loom_type_t source_result_type =
      loom_module_value_type(module, loom_view_load_result(source_load_op));
  if (!loom_type_is_scalar(source_result_type)) return false;
  out_copy_loop->source_view = source_view;
  out_copy_loop->static_index_count = source_rank;
  out_copy_loop->vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, loom_type_element_type(source_result_type),
      loom_dim_pack_static(fragment_length), 0);
  return true;
}

static bool loom_promote_private_fragments_loop_body_matches(
    const loom_op_t* loop_op, const loom_op_t* source_load_op,
    const loom_op_t* private_store_op) {
  loom_region_t* body = loom_scf_for_body(loop_op);
  if (!body || body->block_count != 1) return false;
  loom_block_t* block = loom_region_entry_block(body);
  iree_host_size_t non_terminator_count = 0;
  bool saw_source_load = false;
  bool saw_private_store = false;
  loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (loom_scf_yield_isa(op)) continue;
    ++non_terminator_count;
    if (op == source_load_op) {
      saw_source_load = true;
    } else if (op == private_store_op) {
      saw_private_store = true;
    } else {
      return false;
    }
  }
  return non_terminator_count == 2 && saw_source_load && saw_private_store &&
         source_load_op->next_op == private_store_op;
}

static bool loom_promote_private_fragments_read_copy_loop(
    const loom_module_t* module, loom_value_id_t private_view,
    loom_op_t* private_store_op, int64_t fragment_length,
    loom_promote_private_fragments_copy_loop_t* out_copy_loop) {
  memset(out_copy_loop, 0, sizeof(*out_copy_loop));

  loom_op_t* loop_op = private_store_op->parent_op;
  if (!loom_scf_for_isa(loop_op)) return false;
  loom_region_t* body = loom_scf_for_body(loop_op);
  if (!body || body->block_count != 1 ||
      loom_region_entry_arg_count(body) == 0) {
    return false;
  }
  if (loom_scf_for_iter_args(loop_op).count != 0 ||
      loom_scf_for_results(loop_op).count != 0) {
    return false;
  }

  int64_t lower_bound = 0;
  int64_t upper_bound = 0;
  int64_t step = 0;
  if (!loom_promote_private_fragments_exact_index_constant(
          module, loom_scf_for_lower_bound(loop_op), &lower_bound) ||
      !loom_promote_private_fragments_exact_index_constant(
          module, loom_scf_for_upper_bound(loop_op), &upper_bound) ||
      !loom_promote_private_fragments_exact_index_constant(
          module, loom_scf_for_step(loop_op), &step)) {
    return false;
  }
  if (lower_bound != 0 || upper_bound != fragment_length || step != 1) {
    return false;
  }

  loom_value_id_t induction_variable = loom_region_entry_arg_id(body, 0);
  if (!loom_promote_private_fragments_is_rank1_dynamic_index(
          loom_view_store_static_indices(private_store_op),
          loom_view_store_indices(private_store_op), induction_variable)) {
    return false;
  }

  loom_value_id_t stored_value = loom_view_store_value(private_store_op);
  const loom_value_t* stored = loom_module_value(module, stored_value);
  if (loom_value_is_block_arg(stored)) return false;
  loom_op_t* source_load_op = loom_value_def_op(stored);
  if (!loom_view_load_isa(source_load_op)) return false;
  if (source_load_op->parent_op != loop_op ||
      source_load_op->parent_block != private_store_op->parent_block) {
    return false;
  }
  if (!loom_value_has_single_use(stored)) return false;
  if (!loom_promote_private_fragments_loop_body_matches(loop_op, source_load_op,
                                                        private_store_op)) {
    return false;
  }

  out_copy_loop->loop_op = loop_op;
  out_copy_loop->source_load_op = source_load_op;
  out_copy_loop->private_store_op = private_store_op;
  out_copy_loop->induction_variable = induction_variable;
  if (!loom_promote_private_fragments_source_origin_from_copy_load(
          module, source_load_op, induction_variable, fragment_length,
          out_copy_loop)) {
    return false;
  }
  return loom_view_store_view(private_store_op) == private_view;
}

static iree_status_t loom_promote_private_fragments_collect_uses(
    loom_pass_t* pass, loom_module_t* module, loom_value_id_t private_view,
    int64_t fragment_length,
    loom_promote_private_fragments_copy_loop_t* out_copy_loop,
    loom_promote_private_fragments_load_list_t* out_loads,
    bool* out_promotable) {
  *out_promotable = false;
  out_loads->count = 0;
  out_loads->capacity = 4;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(pass->arena, out_loads->capacity,
                                sizeof(loom_op_t*), (void**)&out_loads->ops));

  const loom_value_t* view_value = loom_module_value(module, private_view);
  bool found_copy_loop = false;
  const loom_use_t* use = NULL;
  loom_value_for_each_use(view_value, use) {
    loom_op_t* user_op = loom_use_user_op(*use);
    if (loom_view_load_isa(user_op) && loom_use_operand_index(*use) == 0) {
      IREE_RETURN_IF_ERROR(loom_promote_private_fragments_load_list_push(
          pass->arena, out_loads, user_op));
      continue;
    }
    if (loom_view_store_isa(user_op) && loom_use_operand_index(*use) == 1) {
      if (found_copy_loop) return iree_ok_status();
      if (!loom_promote_private_fragments_read_copy_loop(
              module, private_view, user_op, fragment_length, out_copy_loop)) {
        return iree_ok_status();
      }
      found_copy_loop = true;
      continue;
    }
    return iree_ok_status();
  }

  *out_promotable = found_copy_loop && out_loads->count > 0;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Rewrite
//===----------------------------------------------------------------------===//

typedef struct loom_promote_private_fragments_context_t {
  // Pass instance owning statistics and scratch allocations.
  loom_pass_t* pass;
  // Module being transformed.
  loom_module_t* module;
  // Shared rewriter for all promotions.
  loom_rewriter_t* rewriter;
} loom_promote_private_fragments_context_t;

static iree_status_t loom_promote_private_fragments_replace_load(
    loom_promote_private_fragments_context_t* context, loom_op_t* load_op,
    loom_value_id_t vector_value) {
  loom_builder_set_before(&context->rewriter->builder, load_op);
  loom_value_id_t checkpoint =
      loom_rewriter_value_checkpoint(context->rewriter);
  loom_value_id_t result = loom_view_load_result(load_op);
  loom_type_t result_type = loom_module_value_type(context->module, result);
  loom_value_slice_t indices = loom_view_load_indices(load_op);
  loom_attribute_t static_indices = loom_view_load_static_indices(load_op);

  loom_op_t* extract_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_extract_build(
      &context->rewriter->builder, vector_value, indices.values, indices.count,
      static_indices.i64_array, static_indices.count, result_type,
      load_op->location, &extract_op));
  loom_value_id_t replacement = loom_vector_extract_result(extract_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      context->rewriter, load_op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      context->rewriter, load_op, &replacement, 1));
  if (context->pass->statistics) {
    loom_pass_statistic_add(
        context->pass, LOOM_PROMOTE_PRIVATE_FRAGMENTS_STAT_LOADS_PROMOTED, 1);
  }
  return iree_ok_status();
}

static iree_status_t loom_promote_private_fragments_try_erase_dead(
    loom_promote_private_fragments_context_t* context, loom_op_t* op) {
  if (!op || iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
    return iree_ok_status();
  }
  bool erased = false;
  return loom_rewriter_erase_if_dead(context->rewriter, op, &erased);
}

static iree_status_t loom_promote_private_fragments_name_vector_value(
    loom_promote_private_fragments_context_t* context,
    loom_value_id_t private_view, loom_value_id_t vector_value) {
  const loom_value_t* private_view_value =
      loom_module_value(context->module, private_view);
  if (private_view_value->name_id == LOOM_STRING_ID_INVALID) {
    return iree_ok_status();
  }
  iree_string_view_t private_view_name =
      context->module->strings.entries[private_view_value->name_id];
  iree_string_view_t suffix = IREE_SV("_vector");
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(context->rewriter->arena,
                                           private_view_name.size + suffix.size,
                                           (void**)&storage));
  memcpy(storage, private_view_name.data, private_view_name.size);
  memcpy(storage + private_view_name.size, suffix.data, suffix.size);

  loom_string_id_t vector_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      context->module,
      iree_make_string_view(storage, private_view_name.size + suffix.size),
      &vector_name_id));
  return loom_module_set_value_name(context->module, vector_value,
                                    vector_name_id);
}

static iree_status_t loom_promote_private_fragments_promote(
    loom_promote_private_fragments_context_t* context, loom_op_t* view_op,
    loom_op_t* alloca_op,
    const loom_promote_private_fragments_copy_loop_t* copy_loop,
    const loom_promote_private_fragments_load_list_t* loads) {
  loom_builder_set_before(&context->rewriter->builder, copy_loop->loop_op);
  int64_t* static_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->rewriter->builder.arena, copy_loop->static_index_count,
      sizeof(int64_t), (void**)&static_indices));
  memcpy(static_indices, copy_loop->static_indices,
         copy_loop->static_index_count * sizeof(int64_t));

  loom_op_t* vector_load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_load_build(
      &context->rewriter->builder, 0, copy_loop->source_view,
      copy_loop->indices, copy_loop->index_count, static_indices,
      copy_loop->static_index_count, 0, 0, copy_loop->vector_type,
      copy_loop->source_load_op->location, &vector_load_op));
  loom_value_id_t vector_value = loom_vector_load_result(vector_load_op);
  IREE_RETURN_IF_ERROR(loom_promote_private_fragments_name_vector_value(
      context, loom_buffer_view_result(view_op), vector_value));

  for (iree_host_size_t i = 0; i < loads->count; ++i) {
    IREE_RETURN_IF_ERROR(loom_promote_private_fragments_replace_load(
        context, loads->ops[i], vector_value));
  }

  IREE_RETURN_IF_ERROR(
      loom_rewriter_erase(context->rewriter, copy_loop->loop_op));
  IREE_RETURN_IF_ERROR(
      loom_promote_private_fragments_try_erase_dead(context, view_op));
  IREE_RETURN_IF_ERROR(
      loom_promote_private_fragments_try_erase_dead(context, alloca_op));

  if (context->pass->statistics) {
    loom_pass_statistic_add(
        context->pass, LOOM_PROMOTE_PRIVATE_FRAGMENTS_STAT_FRAGMENTS_PROMOTED,
        1);
  }
  return iree_ok_status();
}

static iree_status_t loom_promote_private_fragments_process_view(
    loom_promote_private_fragments_context_t* context, loom_op_t* view_op,
    bool* out_changed) {
  if (iree_any_bit_set(view_op->flags, LOOM_OP_FLAG_DEAD)) {
    return iree_ok_status();
  }

  int64_t fragment_length = 0;
  loom_op_t* alloca_op = NULL;
  if (!loom_promote_private_fragments_is_private_rank1_view(
          context->module, view_op, &fragment_length, &alloca_op)) {
    return iree_ok_status();
  }

  loom_value_id_t private_view = loom_buffer_view_result(view_op);
  loom_promote_private_fragments_copy_loop_t copy_loop = {0};
  loom_promote_private_fragments_load_list_t loads = {0};
  bool promotable = false;
  IREE_RETURN_IF_ERROR(loom_promote_private_fragments_collect_uses(
      context->pass, context->module, private_view, fragment_length, &copy_loop,
      &loads, &promotable));
  if (!promotable) return iree_ok_status();

  IREE_RETURN_IF_ERROR(loom_promote_private_fragments_promote(
      context, view_op, alloca_op, &copy_loop, &loads));
  *out_changed = true;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Pass entry point
//===----------------------------------------------------------------------===//

iree_status_t loom_promote_private_fragments_run(loom_pass_t* pass,
                                                 loom_module_t* module,
                                                 loom_func_like_t function) {
  if (!loom_func_like_body(function)) return iree_ok_status();

  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));
  loom_promote_private_fragments_view_list_t views = {0};
  iree_status_t status =
      loom_promote_private_fragments_view_list_initialize(pass->arena, &views);

  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  loom_promote_private_fragments_collect_context_t collect_context = {
      .arena = pass->arena,
      .views = &views,
  };
  if (iree_status_is_ok(status)) {
    status = loom_walk_function(
        module, function, LOOM_WALK_PRE_ORDER,
        (loom_walk_callback_t){
            .fn = loom_promote_private_fragments_collect_view,
            .user_data = &collect_context,
        },
        pass->arena, &walk_result);
  }

  loom_promote_private_fragments_context_t context = {
      .pass = pass,
      .module = module,
      .rewriter = &rewriter,
  };
  bool changed = false;
  for (iree_host_size_t i = 0; i < views.count && iree_status_is_ok(status);
       ++i) {
    status = loom_promote_private_fragments_process_view(&context, views.ops[i],
                                                         &changed);
  }
  if (iree_status_is_ok(status) && changed) {
    loom_pass_mark_changed(pass);
  }

  loom_rewriter_deinitialize(&rewriter);
  return status;
}
