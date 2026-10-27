// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/buffer/materialize.h"

#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/global/rodata_pool.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/walk.h"

static const loom_pass_info_t loom_materialize_buffer_constants_info = {
    .name = IREE_SVL("materialize-buffer-constants"),
    .description =
        IREE_SVL("Pool immutable payloads and expand initialized storage."),
    .kind = LOOM_PASS_MODULE,
};

const loom_pass_info_t* loom_materialize_buffer_constants_pass_info(void) {
  return &loom_materialize_buffer_constants_info;
}

typedef struct loom_buffer_literal_collection_t {
  // Pass-local storage for the source traversal's retained operations.
  iree_arena_allocator_t* arena;
  // Literal operations in source traversal order.
  loom_op_t** operations;
  // Number of retained operations.
  iree_host_size_t count;
  // Allocated operation slots.
  iree_host_size_t capacity;
} loom_buffer_literal_collection_t;

static iree_status_t loom_buffer_literals_collect(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  *out_result = LOOM_WALK_CONTINUE;
  if (!loom_buffer_constant_isa(op) && !loom_buffer_alloca_init_isa(op)) {
    return iree_ok_status();
  }
  loom_buffer_literal_collection_t* literals = user_data;
  if (literals->count == literals->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        literals->arena, literals->count, 8, sizeof(*literals->operations),
        &literals->capacity, (void**)&literals->operations));
  }
  literals->operations[literals->count++] = op;
  return iree_ok_status();
}

static iree_status_t loom_buffer_literal_build_constant(
    loom_builder_t* builder, int64_t value, loom_type_t type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(builder, loom_attr_i64(value),
                                                 type, location, &op));
  *out_value = loom_index_constant_result(op);
  return iree_ok_status();
}

// Divides the logical byte elements into contiguous, nonoverlapping ranges.
// Keeping each range as buffer.copy lets native bulk-copy providers
// participate; source expansion does not choose the target's transfer width or
// scalar loop.
static iree_status_t loom_buffer_literal_workgroup_range(
    loom_builder_t* builder, int64_t byte_count, loom_location_id_t location,
    loom_value_id_t* out_offset, loom_value_id_t* out_length) {
  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const loom_type_t offset_type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  loom_value_id_t linear_id = LOOM_VALUE_ID_INVALID;
  loom_value_id_t workitem_count = LOOM_VALUE_ID_INVALID;
  loom_value_id_t length = LOOM_VALUE_ID_INVALID;
  loom_value_id_t one = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_literal_build_constant(
      builder, 0, index_type, location, &linear_id));
  IREE_RETURN_IF_ERROR(loom_buffer_literal_build_constant(
      builder, 1, index_type, location, &one));
  workitem_count = one;
  IREE_RETURN_IF_ERROR(loom_buffer_literal_build_constant(
      builder, byte_count, index_type, location, &length));
  for (int dimension = LOOM_KERNEL_DIMENSION_Z;
       dimension >= LOOM_KERNEL_DIMENSION_X; --dimension) {
    loom_op_t* coordinate = NULL;
    loom_op_t* size = NULL;
    loom_op_t* product = NULL;
    loom_op_t* sum = NULL;
    loom_op_t* count = NULL;
    IREE_RETURN_IF_ERROR(loom_kernel_workitem_id_build(
        builder, (loom_kernel_dimension_t)dimension, index_type, location,
        &coordinate));
    IREE_RETURN_IF_ERROR(loom_kernel_workgroup_size_build(
        builder, (loom_kernel_dimension_t)dimension, index_type, location,
        &size));
    IREE_RETURN_IF_ERROR(loom_index_mul_build(
        builder, linear_id, loom_kernel_workgroup_size_result(size), location,
        &product));
    IREE_RETURN_IF_ERROR(
        loom_index_add_build(builder, loom_index_mul_result(product),
                             loom_kernel_workitem_id_result(coordinate),
                             index_type, location, &sum));
    IREE_RETURN_IF_ERROR(loom_index_mul_build(
        builder, workitem_count, loom_kernel_workgroup_size_result(size),
        location, &count));
    linear_id = loom_index_add_result(sum);
    workitem_count = loom_index_mul_result(count);
  }
  loom_op_t* excess = NULL;
  loom_op_t* rounded = NULL;
  loom_op_t* chunk = NULL;
  loom_op_t* start = NULL;
  loom_op_t* begin = NULL;
  loom_op_t* remaining = NULL;
  loom_op_t* count = NULL;
  IREE_RETURN_IF_ERROR(loom_index_sub_build(builder, workitem_count, one,
                                            index_type, location, &excess));
  IREE_RETURN_IF_ERROR(loom_index_add_build(builder, length,
                                            loom_index_sub_result(excess),
                                            index_type, location, &rounded));
  IREE_RETURN_IF_ERROR(loom_index_div_build(builder,
                                            loom_index_add_result(rounded),
                                            workitem_count, location, &chunk));
  IREE_RETURN_IF_ERROR(loom_index_mul_build(
      builder, linear_id, loom_index_div_result(chunk), location, &start));
  IREE_RETURN_IF_ERROR(loom_index_min_build(
      builder, loom_index_mul_result(start), length, location, &begin));
  IREE_RETURN_IF_ERROR(loom_index_sub_build(builder, length,
                                            loom_index_min_result(begin),
                                            index_type, location, &remaining));
  IREE_RETURN_IF_ERROR(
      loom_index_min_build(builder, loom_index_sub_result(remaining),
                           loom_index_div_result(chunk), location, &count));
  loom_op_t* offset = NULL;
  loom_op_t* byte_length = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_cast_build(builder, loom_index_min_result(begin), index_type,
                            offset_type, location, &offset));
  IREE_RETURN_IF_ERROR(
      loom_index_cast_build(builder, loom_index_min_result(count), index_type,
                            offset_type, location, &byte_length));
  *out_offset = loom_index_cast_result(offset);
  *out_length = loom_index_cast_result(byte_length);
  return iree_ok_status();
}

static iree_status_t loom_buffer_literal_initialize(
    loom_builder_t* builder, const loom_op_t* op, loom_value_id_t source,
    loom_value_id_t* out_result) {
  const int64_t byte_count =
      (int64_t)loom_buffer_alloca_init_contents(op).data_length;
  const loom_type_t offset_type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  const loom_value_fact_memory_space_t memory_space =
      loom_buffer_alloca_init_memory_space(op);
  const loom_value_id_t pool = loom_buffer_alloca_init_pool(op);
  loom_value_id_t byte_length = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_literal_build_constant(
      builder, byte_count, offset_type, op->location, &byte_length));
  loom_op_t* allocation = NULL;
  IREE_RETURN_IF_ERROR(loom_buffer_alloca_build(
      builder,
      pool != LOOM_VALUE_ID_INVALID ? LOOM_BUFFER_ALLOCA_BUILD_FLAG_HAS_POOL
                                    : 0,
      memory_space, pool, loom_buffer_alloca_init_base_alignment(op),
      byte_length, loom_type_buffer(), op->location, &allocation));
  *out_result = loom_buffer_alloca_result(allocation);
  if (byte_count != 0) {
    loom_value_id_t offset = LOOM_VALUE_ID_INVALID;
    if (memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
      IREE_RETURN_IF_ERROR(loom_buffer_literal_workgroup_range(
          builder, byte_count, op->location, &offset, &byte_length));
    } else {
      IREE_RETURN_IF_ERROR(loom_buffer_literal_build_constant(
          builder, 0, offset_type, op->location, &offset));
    }
    loom_op_t* copy = NULL;
    IREE_RETURN_IF_ERROR(
        loom_buffer_copy_build(builder, source, offset, *out_result, offset,
                               byte_length, op->location, &copy));
  }
  if (memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    loom_op_t* barrier = NULL;
    IREE_RETURN_IF_ERROR(loom_kernel_barrier_build(
        builder, LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP,
        LOOM_ATOMIC_SCOPE_WORKGROUP, LOOM_ATOMIC_ORDERING_ACQ_REL, op->location,
        &barrier));
  }
  return iree_ok_status();
}

static iree_status_t loom_buffer_literal_materialize(loom_rewriter_t* rewriter,
                                                     loom_rodata_pool_t* pool,
                                                     loom_op_t* op) {
  const bool immutable = loom_buffer_constant_isa(op);
  const loom_attribute_t contents =
      immutable ? loom_buffer_constant_contents_attr(op)
                : loom_buffer_alloca_init_contents_attr(op);
  const uint64_t alignment =
      immutable ? (loom_buffer_constant_has_alignment(op)
                       ? (uint64_t)loom_buffer_constant_alignment(op)
                       : 1)
                : (uint64_t)loom_buffer_alloca_init_base_alignment(op);
  loom_rodata_id_t payload = 0;
  IREE_RETURN_IF_ERROR(loom_rodata_pool_intern(pool, contents, alignment,
                                               op->location, &payload));
  loom_symbol_ref_t symbol = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(
      loom_rodata_pool_reference(pool, rewriter->module, payload, &symbol));
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_type_t type = loom_type_buffer();
  loom_op_t* load = NULL;
  IREE_RETURN_IF_ERROR(loom_global_load_build(&rewriter->builder, symbol, &type,
                                              1, op->location, &load));
  loom_value_id_t replacement = loom_global_load_result(load).values[0];
  if (!immutable) {
    IREE_RETURN_IF_ERROR(loom_buffer_literal_initialize(
        &rewriter->builder, op, replacement, &replacement));
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

iree_status_t loom_materialize_buffer_constants_run(loom_pass_t* pass,
                                                    loom_module_t* module) {
  loom_buffer_literal_collection_t literals = {.arena = pass->arena};
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  IREE_RETURN_IF_ERROR(loom_walk_region(
      module, module->body, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_buffer_literals_collect, &literals},
      &walk_result));
  if (!literals.count) {
    return iree_ok_status();
  }
  loom_rodata_pool_t* pool = NULL;
  IREE_RETURN_IF_ERROR(loom_rodata_pool_allocate(pass->arena, &pool));
  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module, pass->arena);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < literals.count && iree_status_is_ok(status);
       ++i) {
    status = loom_buffer_literal_materialize(&rewriter, pool,
                                             literals.operations[i]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_rodata_pool_finalize(pool, module);
  }
  if (rewriter.flags & LOOM_REWRITER_FLAG_CHANGED) {
    loom_pass_mark_changed(pass);
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
