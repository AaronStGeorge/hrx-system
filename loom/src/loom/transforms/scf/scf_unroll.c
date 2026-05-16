// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/scf/scf_unroll.h"

#include <inttypes.h>
#include <stdint.h>

#include "loom/ir/attribute.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"
#include "loom/pass/pipeline.h"
#include "loom/pass/registry.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/fact_table.h"
#include "loom/util/walk.h"

//===----------------------------------------------------------------------===//
// Options and statistics
//===----------------------------------------------------------------------===//

#define LOOM_SCF_UNROLL_DEFAULT_MAX_TRIP_COUNT 8

typedef struct loom_scf_unroll_options_t {
  // Maximum static trip count the pass will materialize inline.
  uint32_t max_trip_count;
} loom_scf_unroll_options_t;

static const loom_pass_option_def_t kScfUnrollOptions[] = {
    {IREE_SVL("max-trip-count"),
     IREE_SVL("Maximum static loop trip count to unroll.")},
};

enum {
  LOOM_SCF_UNROLL_STAT_LOOPS_UNROLLED = 0,
  LOOM_SCF_UNROLL_STAT_ITERATIONS_MATERIALIZED = 1,
};

static const loom_pass_statistic_def_t kScfUnrollStatistics[] = {
    {IREE_SVL("loops-unrolled"), IREE_SVL("Number of scf.for loops unrolled.")},
    {IREE_SVL("iterations-materialized"),
     IREE_SVL("Number of loop body copies materialized.")},
};

static const loom_pass_info_t loom_scf_unroll_pass_info_storage = {
    .name = IREE_SVL("unroll-scf-for"),
    .description = IREE_SVL("Unroll small static scf.for loops."),
    .kind = LOOM_PASS_FUNCTION,
    .option_defs = kScfUnrollOptions,
    .option_count = IREE_ARRAYSIZE(kScfUnrollOptions),
    .statistic_defs = kScfUnrollStatistics,
    .statistic_count = IREE_ARRAYSIZE(kScfUnrollStatistics),
};

const loom_pass_info_t* loom_scf_unroll_pass_info(void) {
  return &loom_scf_unroll_pass_info_storage;
}

static iree_status_t loom_scf_unroll_parse_option(void* user_data,
                                                  iree_string_view_t name,
                                                  iree_string_view_t value) {
  loom_scf_unroll_options_t* options = (loom_scf_unroll_options_t*)user_data;
  if (iree_string_view_equal(name, IREE_SV("max-trip-count"))) {
    if (options->max_trip_count != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "duplicate option 'max-trip-count' for pass 'unroll-scf-for'");
    }
    IREE_RETURN_IF_ERROR(loom_pass_option_parse_uint32(
        IREE_SV("unroll-scf-for"), name, value, &options->max_trip_count));
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown option '%.*s' for pass 'unroll-scf-for'",
                          (int)name.size, name.data);
}

iree_status_t loom_scf_unroll_create(loom_pass_t* pass,
                                     iree_string_view_t options_string) {
  loom_scf_unroll_options_t* options = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(pass->instance_arena,
                                           sizeof(*options), (void**)&options));
  options->max_trip_count = 0;
  if (pass->decoded_options) {
    for (uint16_t i = 0; i < pass->decoded_options->option_count; ++i) {
      const loom_pass_decoded_option_t* option =
          &pass->decoded_options->options[i];
      if (!option->present) {
        continue;
      }
      if (iree_string_view_equal(option->schema->name,
                                 IREE_SV("max-trip-count"))) {
        options->max_trip_count = option->uint32_value;
        continue;
      }
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unknown decoded option '%.*s' for pass 'unroll-scf-for'",
          (int)option->schema->name.size, option->schema->name.data);
    }
  } else {
    IREE_RETURN_IF_ERROR(
        loom_pass_options_parse(pass->info->name, options_string,
                                (loom_pass_option_parse_callback_t){
                                    .fn = loom_scf_unroll_parse_option,
                                    .user_data = options,
                                }));
  }
  if (options->max_trip_count == 0) {
    options->max_trip_count = LOOM_SCF_UNROLL_DEFAULT_MAX_TRIP_COUNT;
  }
  pass->state = options;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Candidate collection
//===----------------------------------------------------------------------===//

#define LOOM_SCF_UNROLL_INITIAL_LOOP_CAPACITY 16

typedef struct loom_scf_unroll_loop_list_t {
  // Collected scf.for operations in traversal order.
  loom_op_t** ops;
  // Number of collected loop operations.
  iree_host_size_t count;
  // Allocated loop pointer capacity.
  iree_host_size_t capacity;
} loom_scf_unroll_loop_list_t;

typedef struct loom_scf_unroll_collect_context_t {
  // Pass scratch arena used for the collected loop pointer list.
  iree_arena_allocator_t* arena;
  // Collected scf.for operations.
  loom_scf_unroll_loop_list_t* loops;
} loom_scf_unroll_collect_context_t;

static iree_status_t loom_scf_unroll_loop_list_initialize(
    iree_arena_allocator_t* arena, loom_scf_unroll_loop_list_t* list) {
  list->count = 0;
  list->capacity = LOOM_SCF_UNROLL_INITIAL_LOOP_CAPACITY;
  return iree_arena_allocate_array(arena, list->capacity, sizeof(loom_op_t*),
                                   (void**)&list->ops);
}

static iree_status_t loom_scf_unroll_loop_list_push(
    iree_arena_allocator_t* arena, loom_scf_unroll_loop_list_t* list,
    loom_op_t* op) {
  if (list->count >= list->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, list->count, list->count + 1, sizeof(loom_op_t*),
        &list->capacity, (void**)&list->ops));
  }
  list->ops[list->count++] = op;
  return iree_ok_status();
}

static iree_status_t loom_scf_unroll_collect_loop(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  loom_scf_unroll_collect_context_t* collect_context =
      (loom_scf_unroll_collect_context_t*)user_data;
  *out_result = LOOM_WALK_CONTINUE;
  if (!loom_scf_for_isa(op)) {
    return iree_ok_status();
  }
  return loom_scf_unroll_loop_list_push(collect_context->arena,
                                        collect_context->loops, op);
}

//===----------------------------------------------------------------------===//
// Unrolling
//===----------------------------------------------------------------------===//

typedef struct loom_scf_unroll_context_t {
  // Current pass invocation.
  loom_pass_t* pass;
  // Module being transformed.
  loom_module_t* module;
  // Rewriter used for cloning and erasure.
  loom_rewriter_t* rewriter;
  // Value facts for the current function snapshot.
  loom_value_fact_table_t* fact_table;
  // Maximum trip count to materialize inline.
  uint32_t max_trip_count;
} loom_scf_unroll_context_t;

static bool loom_scf_unroll_exact_i64(const loom_value_fact_table_t* facts,
                                      loom_value_id_t value,
                                      int64_t* out_integer) {
  return loom_value_facts_as_exact_i64(
      loom_value_fact_table_lookup(facts, value), out_integer);
}

static bool loom_scf_unroll_static_trip_count(
    const loom_scf_unroll_context_t* context, loom_op_t* op,
    uint32_t* out_trip_count, int64_t* out_lower, int64_t* out_step) {
  *out_trip_count = 0;
  *out_lower = 0;
  *out_step = 0;

  int64_t lower = 0;
  int64_t upper = 0;
  int64_t step = 0;
  if (!loom_scf_unroll_exact_i64(context->fact_table,
                                 loom_scf_for_lower_bound(op), &lower) ||
      !loom_scf_unroll_exact_i64(context->fact_table,
                                 loom_scf_for_upper_bound(op), &upper) ||
      !loom_scf_unroll_exact_i64(context->fact_table, loom_scf_for_step(op),
                                 &step)) {
    return false;
  }
  if (step <= 0) {
    return false;
  }
  if (upper <= lower) {
    *out_lower = lower;
    *out_step = step;
    return true;
  }

  int64_t span = 0;
  if ((lower > 0 && upper < INT64_MIN + lower) ||
      (lower < 0 && upper > INT64_MAX + lower)) {
    return false;
  }
  span = upper - lower;
  uint64_t trip_count = ((uint64_t)span + (uint64_t)step - 1) / (uint64_t)step;
  if (trip_count > context->max_trip_count) {
    return false;
  }

  *out_trip_count = (uint32_t)trip_count;
  *out_lower = lower;
  *out_step = step;
  return true;
}

static bool loom_scf_unroll_can_unroll(loom_op_t* op) {
  if (loom_scf_for_results(op).count != 0 ||
      loom_scf_for_iter_args(op).count != 0) {
    return false;
  }
  loom_region_t* body = loom_scf_for_body(op);
  if (!body || body->block_count != 1) {
    return false;
  }
  loom_block_t* body_block = loom_region_entry_block(body);
  return body_block && body_block->arg_count == 1;
}

static iree_status_t loom_scf_unroll_build_iteration_index(
    loom_scf_unroll_context_t* context, loom_op_t* op,
    loom_value_id_t induction_variable, int64_t lower, int64_t step,
    uint32_t ordinal, loom_value_id_t* out_index) {
  int64_t scaled_step = 0;
  int64_t iteration_value = 0;
  if (!iree_checked_mul_i64((int64_t)ordinal, step, &scaled_step) ||
      !iree_checked_add_i64(lower, scaled_step, &iteration_value)) {
    *out_index = LOOM_VALUE_ID_INVALID;
    return iree_ok_status();
  }

  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      &context->rewriter->builder, loom_attr_i64(iteration_value),
      loom_module_value_type(context->module, loom_scf_for_lower_bound(op)),
      op->location, &constant_op));
  *out_index = loom_index_constant_result(constant_op);

  char suffix[32] = {0};
  int suffix_length =
      iree_snprintf(suffix, sizeof(suffix), "%" PRId64, iteration_value);
  if (suffix_length <= 0 || (iree_host_size_t)suffix_length >= sizeof(suffix)) {
    return iree_ok_status();
  }
  return loom_rewriter_try_set_derived_value_name(
      context->rewriter, induction_variable, *out_index,
      iree_make_string_view(suffix, (iree_host_size_t)suffix_length));
}

static iree_status_t loom_scf_unroll_clone_iteration(
    loom_scf_unroll_context_t* context, loom_op_t* op,
    const loom_block_t* body_block, loom_value_id_t iteration_index) {
  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      context->module, context->module, &context->module->arena,
      &(loom_ir_remap_options_t){
          .allow_unmapped_values = true,
          .remap_symbol = loom_ir_remap_symbol_callback_empty(),
      },
      &remap));
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_map_value(&remap, body_block->arg_ids[0], iteration_index));
  return loom_ir_clone_block_ops(
      &context->rewriter->builder, body_block, &remap,
      &(loom_ir_clone_block_options_t){.omit_terminators = true});
}

static iree_status_t loom_scf_unroll_try_unroll(
    loom_scf_unroll_context_t* context, loom_op_t* op, bool* out_changed) {
  *out_changed = false;
  if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD) ||
      !loom_scf_unroll_can_unroll(op)) {
    return iree_ok_status();
  }

  uint32_t trip_count = 0;
  int64_t lower = 0;
  int64_t step = 0;
  if (!loom_scf_unroll_static_trip_count(context, op, &trip_count, &lower,
                                         &step)) {
    return iree_ok_status();
  }

  loom_region_t* body = loom_scf_for_body(op);
  loom_block_t* body_block = loom_region_entry_block(body);
  loom_builder_set_before(&context->rewriter->builder, op);
  loom_value_id_t induction_variable = body_block->arg_ids[0];
  for (uint32_t ordinal = 0; ordinal < trip_count; ++ordinal) {
    loom_value_id_t iteration_index = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scf_unroll_build_iteration_index(
        context, op, induction_variable, lower, step, ordinal,
        &iteration_index));
    if (iteration_index == LOOM_VALUE_ID_INVALID) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_scf_unroll_clone_iteration(
        context, op, body_block, iteration_index));
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(context->rewriter, op));

  if (context->pass->statistics) {
    loom_pass_statistic_add(context->pass, LOOM_SCF_UNROLL_STAT_LOOPS_UNROLLED,
                            1);
    loom_pass_statistic_add(context->pass,
                            LOOM_SCF_UNROLL_STAT_ITERATIONS_MATERIALIZED,
                            trip_count);
  }
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_scf_unroll_process_function_once(
    loom_scf_unroll_context_t* context, loom_func_like_t function,
    bool* out_changed) {
  *out_changed = false;
  loom_scf_unroll_loop_list_t loops = {0};
  IREE_RETURN_IF_ERROR(
      loom_scf_unroll_loop_list_initialize(context->pass->arena, &loops));

  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  loom_scf_unroll_collect_context_t collect_context = {
      .arena = context->pass->arena,
      .loops = &loops,
  };
  IREE_RETURN_IF_ERROR(
      loom_walk_function(context->module, function, LOOM_WALK_PRE_ORDER,
                         (loom_walk_callback_t){
                             .fn = loom_scf_unroll_collect_loop,
                             .user_data = &collect_context,
                         },
                         context->pass->arena, &walk_result));

  for (iree_host_size_t i = 0; i < loops.count; ++i) {
    bool changed = false;
    IREE_RETURN_IF_ERROR(
        loom_scf_unroll_try_unroll(context, loops.ops[i], &changed));
    *out_changed = *out_changed || changed;
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Pass entry point
//===----------------------------------------------------------------------===//

iree_status_t loom_scf_unroll_run(loom_pass_t* pass, loom_module_t* module,
                                  loom_func_like_t function) {
  if (!loom_func_like_body(function)) {
    return iree_ok_status();
  }
  const loom_scf_unroll_options_t* options =
      (const loom_scf_unroll_options_t*)pass->state;

  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));

  loom_scf_unroll_context_t context = {
      .pass = pass,
      .module = module,
      .rewriter = &rewriter,
      .max_trip_count = options ? options->max_trip_count
                                : LOOM_SCF_UNROLL_DEFAULT_MAX_TRIP_COUNT,
  };

  iree_status_t status = iree_ok_status();
  bool changed = true;
  bool any_changed = false;
  while (iree_status_is_ok(status) && changed) {
    changed = false;
    status = loom_pass_value_facts_acquire(
        pass, module, loom_pass_value_fact_scope_function(function),
        &context.fact_table);
    if (!iree_status_is_ok(status)) break;
    status =
        loom_scf_unroll_process_function_once(&context, function, &changed);
    if (changed) {
      any_changed = true;
      loom_pass_value_fact_owner_invalidate(pass->value_facts);
    }
  }

  if (iree_status_is_ok(status) && any_changed) {
    loom_pass_mark_changed(pass);
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
