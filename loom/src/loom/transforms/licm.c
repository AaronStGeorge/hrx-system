// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/licm.h"

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/transforms/rewriter.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

enum {
  LOOM_LICM_STAT_LOOPS_VISITED = 0,
  LOOM_LICM_STAT_OPS_HOISTED = 1,
};

static const loom_pass_statistic_def_t kLICMStatistics[] = {
    {IREE_SVL("loops-visited"),
     IREE_SVL("Number of loop-like operations inspected.")},
    {IREE_SVL("ops-hoisted"),
     IREE_SVL("Number of loop-invariant operations hoisted.")},
};

static const loom_pass_info_t loom_licm_pass_info_storage = {
    .name = IREE_SVL("licm"),
    .description = IREE_SVL("Hoist loop-invariant pure operations."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_defs = kLICMStatistics,
    .statistic_count = IREE_ARRAYSIZE(kLICMStatistics),
};

const loom_pass_info_t* loom_licm_pass_info(void) {
  return &loom_licm_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Scratch stacks
//===----------------------------------------------------------------------===//

#define LOOM_LICM_INITIAL_REGION_STACK_CAPACITY 16

typedef struct loom_licm_region_stack_t {
  // Region pointers waiting to be processed.
  loom_region_t** regions;
  // Number of queued region pointers.
  iree_host_size_t count;
  // Allocated region pointer capacity.
  iree_host_size_t capacity;
} loom_licm_region_stack_t;

static iree_status_t loom_licm_region_stack_initialize(
    iree_arena_allocator_t* arena, loom_licm_region_stack_t* stack) {
  stack->count = 0;
  stack->capacity = LOOM_LICM_INITIAL_REGION_STACK_CAPACITY;
  return iree_arena_allocate_array(
      arena, stack->capacity, sizeof(loom_region_t*), (void**)&stack->regions);
}

static iree_status_t loom_licm_region_stack_push(
    iree_arena_allocator_t* arena, loom_licm_region_stack_t* stack,
    loom_region_t* region) {
  if (!region || region->block_count == 0) return iree_ok_status();
  if (stack->count >= stack->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, stack->count, stack->count + 1, sizeof(loom_region_t*),
        &stack->capacity, (void**)&stack->regions));
  }
  stack->regions[stack->count++] = region;
  return iree_ok_status();
}

static loom_region_t* loom_licm_region_stack_pop(
    loom_licm_region_stack_t* stack) {
  return stack->count > 0 ? stack->regions[--stack->count] : NULL;
}

typedef struct loom_licm_context_t {
  // Pass instance owning statistics and the scratch arena.
  loom_pass_t* pass;
  // Module being transformed.
  loom_module_t* module;
  // Rewriter used for all IR movement.
  loom_rewriter_t* rewriter;
  // Region DFS stack for finding loop-like ops in the function.
  loom_licm_region_stack_t function_stack;
  // Region DFS stack for scanning one loop body for hoistable ops.
  loom_licm_region_stack_t loop_stack;
  // Region DFS stack for candidate dependency checks.
  loom_licm_region_stack_t dependency_stack;
} loom_licm_context_t;

//===----------------------------------------------------------------------===//
// Ancestry and availability
//===----------------------------------------------------------------------===//

static bool loom_licm_op_is_nested_under(const loom_op_t* root,
                                         const loom_op_t* op) {
  if (!root || !op) return false;
  for (const loom_op_t* current = op; current; current = current->parent_op) {
    if (current == root) return true;
  }
  return false;
}

static const loom_op_t* loom_licm_block_owner_op(const loom_block_t* block) {
  if (!block || !block->first_op) return NULL;
  return block->first_op->parent_op;
}

static bool loom_licm_block_is_nested_under(const loom_op_t* root,
                                            const loom_block_t* block) {
  const loom_op_t* owner_op = loom_licm_block_owner_op(block);
  return loom_licm_op_is_nested_under(root, owner_op);
}

static bool loom_licm_value_moves_with_candidate(const loom_module_t* module,
                                                 const loom_op_t* candidate_op,
                                                 loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return loom_licm_block_is_nested_under(candidate_op,
                                           loom_value_def_block(value));
  }
  return loom_licm_op_is_nested_under(candidate_op, loom_value_def_op(value));
}

static bool loom_licm_value_is_defined_inside_loop(const loom_module_t* module,
                                                   const loom_op_t* loop_op,
                                                   loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return true;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    const loom_op_t* owner_op =
        loom_licm_block_owner_op(loom_value_def_block(value));
    if (!owner_op) return true;
    return loom_licm_op_is_nested_under(loop_op, owner_op);
  }
  return loom_licm_op_is_nested_under(loop_op, loom_value_def_op(value));
}

static bool loom_licm_value_is_available_before_loop(
    const loom_module_t* module, const loom_op_t* loop_op,
    const loom_op_t* candidate_op, loom_value_id_t value_id) {
  if (loom_licm_value_moves_with_candidate(module, candidate_op, value_id)) {
    return true;
  }
  return !loom_licm_value_is_defined_inside_loop(module, loop_op, value_id);
}

typedef struct loom_licm_type_ref_query_t {
  // Module containing value definitions and type-use records.
  const loom_module_t* module;
  // Loop currently being hoisted from.
  const loom_op_t* loop_op;
  // Candidate op subtree that would move as a unit.
  const loom_op_t* candidate_op;
  // Cleared when a type references loop-local SSA outside the candidate.
  bool available;
} loom_licm_type_ref_query_t;

static iree_status_t loom_licm_check_type_ref(loom_value_id_t value_id,
                                              void* user_data) {
  loom_licm_type_ref_query_t* query = (loom_licm_type_ref_query_t*)user_data;
  if (!loom_licm_value_is_available_before_loop(
          query->module, query->loop_op, query->candidate_op, value_id)) {
    query->available = false;
  }
  return iree_ok_status();
}

static iree_status_t loom_licm_type_refs_are_available(
    const loom_module_t* module, const loom_op_t* loop_op,
    const loom_op_t* candidate_op, loom_type_t type, bool* out_available) {
  loom_licm_type_ref_query_t query = {
      .module = module,
      .loop_op = loop_op,
      .candidate_op = candidate_op,
      .available = true,
  };
  IREE_RETURN_IF_ERROR(
      loom_type_walk_value_refs(type, loom_licm_check_type_ref, &query));
  *out_available = query.available;
  return iree_ok_status();
}

static iree_status_t loom_licm_value_type_refs_are_available(
    const loom_module_t* module, const loom_op_t* loop_op,
    const loom_op_t* candidate_op, loom_value_id_t value_id,
    bool* out_available) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    *out_available = false;
    return iree_ok_status();
  }
  return loom_licm_type_refs_are_available(
      module, loop_op, candidate_op, loom_module_value_type(module, value_id),
      out_available);
}

static bool loom_licm_traits_are_hoistable(loom_trait_flags_t traits) {
  if (!(traits & LOOM_TRAIT_PURE)) return false;
  if (traits & (LOOM_TRAIT_TERMINATOR | LOOM_TRAIT_HINT)) return false;
  return !loom_traits_may_read(traits) && !loom_traits_may_write(traits);
}

static bool loom_licm_nested_op_can_move_with_candidate(
    const loom_module_t* module, const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable) return false;
  loom_trait_flags_t traits = vtable->effective_traits
                                  ? vtable->effective_traits((loom_op_t*)op)
                                  : vtable->traits;
  if (traits & LOOM_TRAIT_TERMINATOR) {
    return (traits & LOOM_TRAIT_PURE) && !loom_traits_may_read(traits) &&
           !loom_traits_may_write(traits);
  }
  return loom_licm_traits_are_hoistable(traits);
}

static iree_status_t loom_licm_op_dependencies_are_available(
    const loom_module_t* module, const loom_op_t* loop_op,
    const loom_op_t* candidate_op, const loom_op_t* op, bool* out_available) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (!loom_licm_value_is_available_before_loop(module, loop_op, candidate_op,
                                                  operands[i])) {
      *out_available = false;
      return iree_ok_status();
    }
  }

  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_licm_value_type_refs_are_available(
        module, loop_op, candidate_op, results[i], out_available));
    if (!*out_available) return iree_ok_status();
  }

  *out_available = true;
  return iree_ok_status();
}

static iree_status_t loom_licm_block_arg_type_refs_are_available(
    const loom_module_t* module, const loom_op_t* loop_op,
    const loom_op_t* candidate_op, const loom_block_t* block,
    bool* out_available) {
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_licm_value_type_refs_are_available(
        module, loop_op, candidate_op, loom_block_arg_id(block, i),
        out_available));
    if (!*out_available) return iree_ok_status();
  }
  *out_available = true;
  return iree_ok_status();
}

static iree_status_t loom_licm_subtree_dependencies_are_available(
    loom_licm_context_t* context, const loom_op_t* loop_op,
    const loom_op_t* candidate_op, bool* out_available) {
  context->dependency_stack.count = 0;
  IREE_RETURN_IF_ERROR(loom_licm_op_dependencies_are_available(
      context->module, loop_op, candidate_op, candidate_op, out_available));
  if (!*out_available) return iree_ok_status();

  loom_region_t** regions = loom_op_regions(candidate_op);
  for (uint8_t i = 0; i < candidate_op->region_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_licm_region_stack_push(
        context->pass->arena, &context->dependency_stack, regions[i]));
  }

  while (true) {
    loom_region_t* region =
        loom_licm_region_stack_pop(&context->dependency_stack);
    if (!region) break;

    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      IREE_RETURN_IF_ERROR(loom_licm_block_arg_type_refs_are_available(
          context->module, loop_op, candidate_op, block, out_available));
      if (!*out_available) return iree_ok_status();

      loom_op_t* child_op = NULL;
      loom_block_for_each_op(block, child_op) {
        if (!loom_licm_nested_op_can_move_with_candidate(context->module,
                                                         child_op)) {
          *out_available = false;
          return iree_ok_status();
        }
        IREE_RETURN_IF_ERROR(loom_licm_op_dependencies_are_available(
            context->module, loop_op, candidate_op, child_op, out_available));
        if (!*out_available) return iree_ok_status();

        loom_region_t** child_regions = loom_op_regions(child_op);
        for (uint8_t i = 0; i < child_op->region_count; ++i) {
          IREE_RETURN_IF_ERROR(loom_licm_region_stack_push(
              context->pass->arena, &context->dependency_stack,
              child_regions[i]));
        }
      }
    }
  }

  *out_available = true;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Hoisting
//===----------------------------------------------------------------------===//

static iree_status_t loom_licm_op_is_hoistable(loom_licm_context_t* context,
                                               loom_op_t* loop_op,
                                               loom_op_t* candidate_op,
                                               bool* out_hoistable) {
  *out_hoistable = false;
  const loom_op_vtable_t* vtable =
      loom_op_vtable(context->module, candidate_op);
  if (!vtable) return iree_ok_status();

  loom_trait_flags_t traits = vtable->effective_traits
                                  ? vtable->effective_traits(candidate_op)
                                  : vtable->traits;
  if (!loom_licm_traits_are_hoistable(traits)) return iree_ok_status();
  if (loom_op_regions_have_read_effects(candidate_op) ||
      loom_op_regions_have_write_effects(candidate_op)) {
    return iree_ok_status();
  }

  return loom_licm_subtree_dependencies_are_available(
      context, loop_op, candidate_op, out_hoistable);
}

static iree_status_t loom_licm_push_child_regions(
    loom_licm_context_t* context, loom_licm_region_stack_t* stack,
    loom_op_t* op) {
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_licm_region_stack_push(context->pass->arena, stack, regions[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_licm_hoist_from_loop_body(
    loom_licm_context_t* context, loom_loop_like_t loop, bool* out_changed) {
  loom_region_t* body = loom_loop_like_body(loop);
  if (!body) return iree_ok_status();

  context->loop_stack.count = 0;
  IREE_RETURN_IF_ERROR(loom_licm_region_stack_push(context->pass->arena,
                                                   &context->loop_stack, body));

  while (true) {
    loom_region_t* region = loom_licm_region_stack_pop(&context->loop_stack);
    if (!region) break;

    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      loom_op_t* op = block->first_op;
      while (op) {
        loom_op_t* next_op = op->next_op;
        if (op->flags & LOOM_OP_FLAG_DEAD) {
          op = next_op;
          continue;
        }

        bool hoistable = false;
        IREE_RETURN_IF_ERROR(
            loom_licm_op_is_hoistable(context, loop.op, op, &hoistable));
        if (hoistable) {
          IREE_RETURN_IF_ERROR(
              loom_rewriter_move_before(context->rewriter, op, loop.op));
          *out_changed = true;
          if (context->pass->statistics) {
            loom_pass_statistic_add(context->pass, LOOM_LICM_STAT_OPS_HOISTED,
                                    1);
          }
        } else {
          IREE_RETURN_IF_ERROR(
              loom_licm_push_child_regions(context, &context->loop_stack, op));
        }

        op = next_op;
      }
    }
  }

  return iree_ok_status();
}

static iree_status_t loom_licm_process_function_once(
    loom_licm_context_t* context, loom_func_like_t function,
    bool* out_changed) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body) return iree_ok_status();

  context->function_stack.count = 0;
  IREE_RETURN_IF_ERROR(loom_licm_region_stack_push(
      context->pass->arena, &context->function_stack, body));

  while (true) {
    loom_region_t* region =
        loom_licm_region_stack_pop(&context->function_stack);
    if (!region) break;

    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      loom_op_t* op = block->first_op;
      while (op) {
        loom_op_t* next_op = op->next_op;
        if (op->flags & LOOM_OP_FLAG_DEAD) {
          op = next_op;
          continue;
        }

        loom_loop_like_t loop = loom_loop_like_cast(context->module, op);
        if (loom_loop_like_isa(loop)) {
          if (context->pass->statistics) {
            loom_pass_statistic_add(context->pass, LOOM_LICM_STAT_LOOPS_VISITED,
                                    1);
          }
          IREE_RETURN_IF_ERROR(
              loom_licm_hoist_from_loop_body(context, loop, out_changed));
        }

        IREE_RETURN_IF_ERROR(loom_licm_push_child_regions(
            context, &context->function_stack, op));
        op = next_op;
      }
    }
  }

  return iree_ok_status();
}

iree_status_t loom_licm_run(loom_pass_t* pass, loom_module_t* module,
                            loom_func_like_t function) {
  if (!loom_func_like_body(function)) return iree_ok_status();

  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));

  loom_licm_context_t context = {
      .pass = pass,
      .module = module,
      .rewriter = &rewriter,
  };
  iree_status_t status =
      loom_licm_region_stack_initialize(pass->arena, &context.function_stack);
  if (iree_status_is_ok(status)) {
    status =
        loom_licm_region_stack_initialize(pass->arena, &context.loop_stack);
  }
  if (iree_status_is_ok(status)) {
    status = loom_licm_region_stack_initialize(pass->arena,
                                               &context.dependency_stack);
  }

  bool changed = true;
  while (iree_status_is_ok(status) && changed) {
    changed = false;
    status = loom_licm_process_function_once(&context, function, &changed);
  }

  loom_rewriter_deinitialize(&rewriter);
  return status;
}
