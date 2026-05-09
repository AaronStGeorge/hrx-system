// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/control.h"

#include <string.h>

#include "loom/ops/cfg/ops.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/target_refs.h"

enum {
  LOOM_AMDGPU_BRANCH_PLAN_THEN_MASKED_REGION = 0,
  LOOM_AMDGPU_BRANCH_PLAN_IF_ELSE_DIAMOND = 1,
};

typedef struct loom_amdgpu_region_exit_edge_t {
  // Source terminator whose successor edge exits the divergent region.
  const loom_op_t* terminator;
  // Successor ordinal on terminator that exits the divergent region.
  uint8_t successor_index;
} loom_amdgpu_region_exit_edge_t;

typedef struct loom_amdgpu_masked_region_t {
  // Source CFG edges that leave the divergent region for its continuation.
  loom_amdgpu_region_exit_edge_t* exit_edges;
  // Number of entries in exit_edges.
  iree_host_size_t exit_edge_count;
  // Guard false destination reached when no lane enters the masked region.
  loom_block_t* guard_false_dest;
  // Shared continuation reached by region exits.
  loom_block_t* continuation;
  // Optional pass-through false-arm terminator providing merge fallback values.
  const loom_op_t* false_passthrough_terminator;
} loom_amdgpu_masked_region_t;

typedef struct loom_amdgpu_bounded_region_t {
  // Source CFG edges that leave this region for its continuation.
  loom_amdgpu_region_exit_edge_t* exit_edges;
  // Number of entries in exit_edges.
  iree_host_size_t exit_edge_count;
  // Region entry block.
  loom_block_t* entry;
  // Shared continuation block this region exits to.
  loom_block_t* continuation;
  // Per-source-block membership bits.
  uint8_t* contains_block;
} loom_amdgpu_bounded_region_t;

typedef struct loom_amdgpu_branch_plan_t {
  // Low-only block that restores EXEC for the guard false edge.
  loom_block_t* restore_block;
  // Effective guard false destination after EXEC has been restored.
  loom_block_t* restore_dest;
  // False-arm terminator used when restore_block bypasses a pass-through block.
  const loom_op_t* restore_passthrough_terminator;
  // Low-only block that restores EXEC and merges pass-through false values.
  loom_block_t* merge_restore_block;
  // Shared continuation reached after merge_restore_block.
  loom_block_t* merge_restore_dest;
  // False-arm terminator providing pass-through merge fallback values.
  const loom_op_t* false_passthrough_terminator;
  // Low-only block that computes the inactive else mask for if/else diamonds.
  loom_block_t* else_dispatch_block;
  // Low-only block that enters the else body when no true lanes were active.
  loom_block_t* no_true_else_entry_block;
  // Original low destination for the else body.
  loom_block_t* else_body_block;
  // Low-only block that restores EXEC when no false lanes were active.
  loom_block_t* true_only_restore_block;
  // Number of values merged by an if/else diamond.
  uint16_t if_else_merge_arg_count;
} loom_amdgpu_branch_plan_t;

typedef struct loom_amdgpu_zero_placeholder_t {
  // Low value type represented by value.
  loom_type_t type;
  // Reusable zero value for an inactive merge placeholder.
  loom_value_id_t value;
} loom_amdgpu_zero_placeholder_t;

static iree_status_t loom_amdgpu_emit_plain_cond_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_condition, loom_block_t* low_true_dest,
    loom_block_t* low_false_dest) {
  loom_op_t* low_cond_br_op = NULL;
  return loom_low_cond_br_build(loom_low_lower_context_builder(context),
                                low_condition, low_true_dest, low_false_dest,
                                source_op->location, &low_cond_br_op);
}

static iree_status_t loom_amdgpu_condition_is_reg_class(
    loom_low_lower_context_t* context, loom_type_t low_type,
    uint16_t reg_class_id, uint32_t unit_count, bool* out_match) {
  *out_match = false;
  bool is_class = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, reg_class_id, &is_class));
  *out_match =
      is_class && loom_type_register_unit_count(low_type) == unit_count;
  return iree_ok_status();
}

static uint16_t loom_amdgpu_source_region_block_index(
    loom_region_t* source_body, const loom_block_t* block) {
  uint16_t block_index = 0;
  const bool found =
      loom_region_try_block_index(source_body, block, &block_index);
  IREE_ASSERT(found);
  (void)found;
  return block_index;
}

static loom_region_t* loom_amdgpu_source_body(
    loom_low_lower_context_t* context) {
  loom_func_like_t source_function =
      loom_low_lower_context_source_function(context);
  loom_region_t* source_body = loom_func_like_body(source_function);
  IREE_ASSERT(source_body != NULL);
  return source_body;
}

static iree_host_size_t loom_amdgpu_successor_edge_count(
    loom_region_t* source_body) {
  iree_host_size_t edge_count = 0;
  for (uint16_t block_index = 0; block_index < source_body->block_count;
       ++block_index) {
    const loom_op_t* terminator =
        loom_block_const_last_op(loom_region_block(source_body, block_index));
    if (terminator != NULL) {
      edge_count += terminator->successor_count;
    }
  }
  return edge_count;
}

static void loom_amdgpu_record_masked_region_exit_edge(
    const loom_op_t* terminator, uint8_t successor_index,
    loom_amdgpu_masked_region_t* region) {
  region->exit_edges[region->exit_edge_count++] =
      (loom_amdgpu_region_exit_edge_t){
          .terminator = terminator,
          .successor_index = successor_index,
      };
}

static void loom_amdgpu_record_bounded_region_exit_edge(
    const loom_op_t* terminator, uint8_t successor_index,
    loom_amdgpu_bounded_region_t* region) {
  region->exit_edges[region->exit_edge_count++] =
      (loom_amdgpu_region_exit_edge_t){
          .terminator = terminator,
          .successor_index = successor_index,
      };
}

static bool loom_amdgpu_region_has_single_entry(
    loom_region_t* source_body, const uint8_t* in_region,
    const loom_block_t* guard_block, const loom_block_t* region_entry,
    const loom_op_t* ignored_entry_terminator,
    uint8_t ignored_entry_successor_index) {
  const uint16_t region_entry_index =
      loom_amdgpu_source_region_block_index(source_body, region_entry);

  for (uint16_t block_index = 0; block_index < source_body->block_count;
       ++block_index) {
    const loom_block_t* source_block =
        loom_region_block(source_body, block_index);
    const loom_op_t* terminator = loom_block_const_last_op(source_block);
    if (terminator == NULL) {
      continue;
    }
    loom_block_t* const* successors = loom_op_const_successors(terminator);
    for (uint8_t successor_index = 0;
         successor_index < terminator->successor_count; ++successor_index) {
      const uint16_t successor_block_index =
          loom_amdgpu_source_region_block_index(source_body,
                                                successors[successor_index]);
      if (!in_region[successor_block_index]) {
        continue;
      }
      if (in_region[block_index]) {
        continue;
      }
      if (terminator == ignored_entry_terminator &&
          successor_index == ignored_entry_successor_index) {
        continue;
      }
      if (source_block == guard_block &&
          successor_block_index == region_entry_index) {
        continue;
      }
      return false;
    }
  }
  return true;
}

static iree_status_t loom_amdgpu_verify_masked_region_single_entry(
    loom_low_lower_context_t* context, const loom_op_t* diagnostic_op,
    loom_region_t* source_body, const uint8_t* in_region,
    const loom_block_t* region_entry, const loom_op_t* ignored_entry_terminator,
    uint8_t ignored_entry_successor_index) {
  if (!loom_amdgpu_region_has_single_entry(
          source_body, in_region, diagnostic_op->parent_block, region_entry,
          ignored_entry_terminator, ignored_entry_successor_index)) {
    return loom_low_lower_emit_branch_constraint(
        context, diagnostic_op, IREE_SV("masked_region_single_entry"));
  }
  return iree_ok_status();
}

static bool loom_amdgpu_try_analyze_bounded_region(
    loom_region_t* source_body, loom_block_t* entry, loom_block_t* continuation,
    loom_amdgpu_region_exit_edge_t* exit_edges, uint8_t* contains_block,
    uint16_t* stack, loom_amdgpu_bounded_region_t* out_region) {
  *out_region = (loom_amdgpu_bounded_region_t){
      .exit_edges = exit_edges,
      .entry = entry,
      .continuation = continuation,
      .contains_block = contains_block,
  };
  if (entry == continuation) {
    return false;
  }
  memset(contains_block, 0,
         (iree_host_size_t)source_body->block_count * sizeof(*contains_block));

  const uint16_t entry_index =
      loom_amdgpu_source_region_block_index(source_body, entry);
  uint16_t stack_count = 0;
  stack[stack_count++] = entry_index;
  contains_block[entry_index] = 1;

  while (stack_count != 0) {
    const uint16_t block_index = stack[--stack_count];
    loom_block_t* source_block = loom_region_block(source_body, block_index);
    const loom_op_t* terminator = loom_block_const_last_op(source_block);
    if (terminator == NULL || terminator->successor_count == 0) {
      return false;
    }

    loom_block_t* const* successors = loom_op_const_successors(terminator);
    for (uint8_t successor_index = 0;
         successor_index < terminator->successor_count; ++successor_index) {
      loom_block_t* successor = successors[successor_index];
      if (successor == continuation) {
        loom_amdgpu_record_bounded_region_exit_edge(terminator, successor_index,
                                                    out_region);
        continue;
      }

      const uint16_t successor_block_index =
          loom_amdgpu_source_region_block_index(source_body, successor);
      if (contains_block[successor_block_index]) {
        continue;
      }
      contains_block[successor_block_index] = 1;
      stack[stack_count++] = successor_block_index;
    }
  }

  return out_region->exit_edge_count != 0;
}

static bool loom_amdgpu_bounded_regions_are_disjoint(
    loom_region_t* source_body, const loom_amdgpu_bounded_region_t* lhs,
    const loom_amdgpu_bounded_region_t* rhs) {
  for (uint16_t block_index = 0; block_index < source_body->block_count;
       ++block_index) {
    if (lhs->contains_block[block_index] && rhs->contains_block[block_index]) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_try_false_passthrough_continuation(
    loom_block_t* false_dest, loom_block_t** out_continuation,
    const loom_op_t** out_false_terminator) {
  *out_continuation = NULL;
  *out_false_terminator = NULL;
  if (false_dest->arg_count != 0 || false_dest->op_count != 1) {
    return false;
  }
  const loom_op_t* false_terminator = loom_block_const_last_op(false_dest);
  if (!loom_cfg_br_isa(false_terminator)) {
    return false;
  }
  loom_block_t* continuation = loom_cfg_br_dest(false_terminator);
  if (continuation == false_dest) {
    return false;
  }
  *out_continuation = continuation;
  *out_false_terminator = false_terminator;
  return true;
}

static iree_status_t loom_amdgpu_analyze_then_masked_region(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_masked_region_t* out_region) {
  *out_region = (loom_amdgpu_masked_region_t){0};
  loom_block_t* source_entry = loom_cfg_cond_br_true_dest(source_op);
  loom_block_t* guard_false_dest = loom_cfg_cond_br_false_dest(source_op);
  loom_block_t* source_continuation = guard_false_dest;
  const loom_op_t* false_passthrough_terminator = NULL;
  loom_block_t* passthrough_continuation = NULL;
  const loom_op_t* passthrough_terminator = NULL;
  if (loom_amdgpu_try_false_passthrough_continuation(guard_false_dest,
                                                     &passthrough_continuation,
                                                     &passthrough_terminator)) {
    source_continuation = passthrough_continuation;
    false_passthrough_terminator = passthrough_terminator;
  }
  if (source_entry->arg_count != 0 || guard_false_dest->arg_count != 0) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("destination_block_arguments_absent"));
  }
  if (source_entry == source_continuation) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_single_continuation"));
  }

  loom_region_t* source_body = loom_amdgpu_source_body(context);
  const iree_host_size_t edge_capacity =
      loom_amdgpu_successor_edge_count(source_body);
  loom_amdgpu_region_exit_edge_t* exit_edges = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, edge_capacity, sizeof(*exit_edges), (void**)&exit_edges));

  uint8_t* in_region = NULL;
  uint16_t* stack = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, source_body->block_count, sizeof(*in_region),
      (void**)&in_region));
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, source_body->block_count, sizeof(*stack), (void**)&stack));
  memset(in_region, 0,
         (iree_host_size_t)source_body->block_count * sizeof(*in_region));

  const uint16_t entry_index =
      loom_amdgpu_source_region_block_index(source_body, source_entry);
  uint16_t stack_count = 0;
  stack[stack_count++] = entry_index;
  in_region[entry_index] = 1;

  loom_amdgpu_masked_region_t region = {
      .exit_edges = exit_edges,
      .exit_edge_count = 0,
      .guard_false_dest = guard_false_dest,
      .continuation = source_continuation,
      .false_passthrough_terminator = false_passthrough_terminator,
  };
  while (stack_count != 0) {
    const uint16_t block_index = stack[--stack_count];
    loom_block_t* source_block = loom_region_block(source_body, block_index);
    const loom_op_t* terminator = loom_block_const_last_op(source_block);
    if (terminator == NULL || terminator->successor_count == 0) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_exits_by_cfg"));
    }

    loom_block_t* const* successors = loom_op_const_successors(terminator);
    for (uint8_t successor_index = 0;
         successor_index < terminator->successor_count; ++successor_index) {
      loom_block_t* successor = successors[successor_index];
      if (successor == source_continuation) {
        loom_amdgpu_record_masked_region_exit_edge(terminator, successor_index,
                                                   &region);
        continue;
      }

      const uint16_t successor_block_index =
          loom_amdgpu_source_region_block_index(source_body, successor);
      if (in_region[successor_block_index]) {
        continue;
      }
      in_region[successor_block_index] = 1;
      stack[stack_count++] = successor_block_index;
    }
  }

  if (region.exit_edge_count == 0) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_single_continuation"));
  }
  const loom_op_t* ignored_entry_terminator =
      region.false_passthrough_terminator != NULL ? source_op : NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_verify_masked_region_single_entry(
      context, source_op, source_body, in_region, source_entry,
      ignored_entry_terminator, /*ignored_entry_successor_index=*/1));
  *out_region = region;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_restore_block_like_dest(
    loom_low_lower_context_t* context, loom_block_t* low_dest,
    loom_block_t** out_restore_block) {
  *out_restore_block = NULL;
  loom_block_t* restore_block = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_append_low_block(context, &restore_block));
  loom_module_t* module = loom_low_lower_context_module(context);
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  for (uint16_t i = 0; i < low_dest->arg_count; ++i) {
    const loom_value_id_t dest_arg = loom_block_arg_id(low_dest, i);
    loom_value_id_t restore_arg = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
        builder, restore_block, loom_module_value_type(module, dest_arg),
        &restore_arg));
    IREE_RETURN_IF_ERROR(
        loom_module_copy_value_name(module, dest_arg, restore_arg));
  }
  *out_restore_block = restore_block;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_block_args_like_dest(
    loom_low_lower_context_t* context, loom_block_t* dest_block,
    loom_block_t* block, loom_value_id_t* out_arg_ids) {
  loom_module_t* module = loom_low_lower_context_module(context);
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  for (uint16_t i = 0; i < dest_block->arg_count; ++i) {
    const loom_value_id_t dest_arg = loom_block_arg_id(dest_block, i);
    loom_value_id_t block_arg = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
        builder, block, loom_module_value_type(module, dest_arg), &block_arg));
    IREE_RETURN_IF_ERROR(
        loom_module_copy_value_name(module, dest_arg, block_arg));
    if (out_arg_ids != NULL) {
      out_arg_ids[i] = block_arg;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_prepare_then_masked_region(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_amdgpu_masked_region_t region = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_analyze_then_masked_region(context, source_op, &region));

  loom_amdgpu_branch_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*plan), (void**)&plan));
  *plan = (loom_amdgpu_branch_plan_t){0};
  loom_block_t* guard_false_low_dest = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_block(
      context, region.guard_false_dest, &guard_false_low_dest));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_restore_block_like_dest(
      context, guard_false_low_dest, &plan->restore_block));
  IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
      context, source_op, 1, plan->restore_block, &plan->restore_dest));
  if (region.false_passthrough_terminator != NULL) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_block(
        context, region.continuation, &plan->restore_dest));
    plan->restore_passthrough_terminator = region.false_passthrough_terminator;
  }

  loom_block_t* exit_restore_block = plan->restore_block;
  loom_block_t* exit_restore_dest = plan->restore_dest;
  if (region.continuation != region.guard_false_dest) {
    loom_block_t* continuation_low_dest = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_block(
        context, region.continuation, &continuation_low_dest));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_restore_block_like_dest(
        context, continuation_low_dest, &exit_restore_block));
    exit_restore_dest = continuation_low_dest;
    plan->merge_restore_block = exit_restore_block;
    plan->merge_restore_dest = exit_restore_dest;
    plan->false_passthrough_terminator = region.false_passthrough_terminator;
  }

  for (iree_host_size_t i = 0; i < region.exit_edge_count; ++i) {
    loom_block_t* previous_low_dest = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
        context, region.exit_edges[i].terminator,
        region.exit_edges[i].successor_index, exit_restore_block,
        &previous_low_dest));
    if (previous_low_dest != exit_restore_dest) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_exit_edge_interposition"));
    }
  }

  return loom_low_lower_set_branch_plan(
      context, source_op,
      loom_low_lower_plan_make(LOOM_AMDGPU_BRANCH_PLAN_THEN_MASKED_REGION,
                               plan));
}

static bool loom_amdgpu_try_if_else_diamond(
    const loom_op_t* source_op, const loom_op_t** out_true_terminator,
    const loom_op_t** out_false_terminator) {
  *out_true_terminator = NULL;
  *out_false_terminator = NULL;
  loom_block_t* true_dest = loom_cfg_cond_br_true_dest(source_op);
  loom_block_t* false_dest = loom_cfg_cond_br_false_dest(source_op);
  if (true_dest->arg_count != 0 || false_dest->arg_count != 0 ||
      true_dest == false_dest || true_dest->op_count == 0 ||
      false_dest->op_count == 0) {
    return false;
  }
  const loom_op_t* true_terminator = loom_block_const_last_op(true_dest);
  const loom_op_t* false_terminator = loom_block_const_last_op(false_dest);
  if (!loom_cfg_br_isa(true_terminator) || !loom_cfg_br_isa(false_terminator)) {
    return false;
  }
  loom_block_t* merge_block = loom_cfg_br_dest(true_terminator);
  if (merge_block != loom_cfg_br_dest(false_terminator) ||
      merge_block == true_dest || merge_block == false_dest) {
    return false;
  }
  const loom_value_slice_t true_args = loom_cfg_br_args(true_terminator);
  const loom_value_slice_t false_args = loom_cfg_br_args(false_terminator);
  if (true_args.count != merge_block->arg_count ||
      false_args.count != merge_block->arg_count) {
    return false;
  }
  *out_true_terminator = true_terminator;
  *out_false_terminator = false_terminator;
  return true;
}

static iree_status_t loom_amdgpu_verify_if_else_merge_args(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_block_t* merge_low_dest, uint16_t merge_arg_count) {
  loom_module_t* module = loom_low_lower_context_module(context);
  for (uint16_t i = 0; i < merge_arg_count; ++i) {
    const loom_value_id_t merge_arg = loom_block_arg_id(merge_low_dest, i);
    const loom_type_t merge_type = loom_module_value_type(module, merge_arg);
    bool is_vgpr = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
        context, merge_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &is_vgpr));
    const uint32_t lane_count = loom_type_register_unit_count(merge_type);
    if (!is_vgpr || lane_count == 0 ||
        lane_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_merge_vgpr_values"));
    }
  }
  return iree_ok_status();
}

static bool loom_amdgpu_try_analyze_if_else_regions(
    loom_region_t* source_body, const loom_op_t* source_op,
    loom_amdgpu_region_exit_edge_t* true_exit_edges,
    loom_amdgpu_region_exit_edge_t* false_exit_edges,
    uint8_t* true_contains_block, uint8_t* false_contains_block,
    uint16_t* stack, loom_amdgpu_bounded_region_t* out_true_region,
    loom_amdgpu_bounded_region_t* out_false_region) {
  loom_block_t* true_entry = loom_cfg_cond_br_true_dest(source_op);
  loom_block_t* false_entry = loom_cfg_cond_br_false_dest(source_op);
  if (true_entry->arg_count != 0 || false_entry->arg_count != 0 ||
      true_entry == false_entry) {
    return false;
  }

  for (uint16_t block_index = 0; block_index < source_body->block_count;
       ++block_index) {
    loom_block_t* continuation = loom_region_block(source_body, block_index);
    if (continuation == source_op->parent_block || continuation == true_entry ||
        continuation == false_entry) {
      continue;
    }

    loom_amdgpu_bounded_region_t true_region = {0};
    if (!loom_amdgpu_try_analyze_bounded_region(
            source_body, true_entry, continuation, true_exit_edges,
            true_contains_block, stack, &true_region)) {
      continue;
    }
    loom_amdgpu_bounded_region_t false_region = {0};
    if (!loom_amdgpu_try_analyze_bounded_region(
            source_body, false_entry, continuation, false_exit_edges,
            false_contains_block, stack, &false_region)) {
      continue;
    }
    if (!loom_amdgpu_bounded_regions_are_disjoint(source_body, &true_region,
                                                  &false_region)) {
      continue;
    }
    if (!loom_amdgpu_region_has_single_entry(
            source_body, true_region.contains_block, source_op->parent_block,
            true_entry, /*ignored_entry_terminator=*/NULL,
            /*ignored_entry_successor_index=*/0)) {
      continue;
    }
    if (!loom_amdgpu_region_has_single_entry(
            source_body, false_region.contains_block, source_op->parent_block,
            false_entry, /*ignored_entry_terminator=*/NULL,
            /*ignored_entry_successor_index=*/0)) {
      continue;
    }

    *out_true_region = true_region;
    *out_false_region = false_region;
    return true;
  }

  return false;
}

static iree_status_t loom_amdgpu_prepare_if_else_regions(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bounded_region_t* true_region,
    const loom_amdgpu_bounded_region_t* false_region) {
  loom_block_t* merge_low_dest = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_block(
      context, true_region->continuation, &merge_low_dest));

  loom_amdgpu_branch_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*plan), (void**)&plan));
  *plan = (loom_amdgpu_branch_plan_t){0};
  plan->if_else_merge_arg_count = merge_low_dest->arg_count;

  if (plan->if_else_merge_arg_count == 0) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_append_low_block(context, &plan->else_dispatch_block));
    IREE_RETURN_IF_ERROR(
        loom_low_lower_append_low_block(context, &plan->restore_block));

    IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
        context, source_op, 1, plan->else_dispatch_block,
        &plan->else_body_block));

    loom_block_t* common_restore_dest = NULL;
    for (iree_host_size_t i = 0; i < true_region->exit_edge_count; ++i) {
      loom_block_t* previous_low_dest = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
          context, true_region->exit_edges[i].terminator,
          true_region->exit_edges[i].successor_index, plan->else_dispatch_block,
          &previous_low_dest));
      if (common_restore_dest == NULL) {
        common_restore_dest = previous_low_dest;
      } else if (previous_low_dest != common_restore_dest) {
        return loom_low_lower_emit_branch_constraint(
            context, source_op,
            IREE_SV("masked_region_exit_edge_interposition"));
      }
    }

    for (iree_host_size_t i = 0; i < false_region->exit_edge_count; ++i) {
      loom_block_t* previous_low_dest = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
          context, false_region->exit_edges[i].terminator,
          false_region->exit_edges[i].successor_index, plan->restore_block,
          &previous_low_dest));
      if (common_restore_dest == NULL) {
        common_restore_dest = previous_low_dest;
      } else if (previous_low_dest != common_restore_dest) {
        return loom_low_lower_emit_branch_constraint(
            context, source_op,
            IREE_SV("masked_region_exit_edge_interposition"));
      }
    }
    if (common_restore_dest == NULL) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_single_continuation"));
    }
    plan->restore_dest = common_restore_dest;

    return loom_low_lower_set_branch_plan(
        context, source_op,
        loom_low_lower_plan_make(LOOM_AMDGPU_BRANCH_PLAN_IF_ELSE_DIAMOND,
                                 plan));
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_verify_if_else_merge_args(
      context, source_op, merge_low_dest, plan->if_else_merge_arg_count));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_append_low_block(context, &plan->else_dispatch_block));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_block_args_like_dest(
      context, merge_low_dest, plan->else_dispatch_block, NULL));
  IREE_RETURN_IF_ERROR(loom_low_lower_append_low_block(
      context, &plan->no_true_else_entry_block));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_append_low_block(context, &plan->true_only_restore_block));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_append_low_block(context, &plan->merge_restore_block));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_block_args_like_dest(
      context, merge_low_dest, plan->merge_restore_block, NULL));

  IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
      context, source_op, 1, plan->no_true_else_entry_block,
      &plan->else_body_block));

  loom_block_t* common_restore_dest = NULL;
  for (iree_host_size_t i = 0; i < true_region->exit_edge_count; ++i) {
    loom_block_t* previous_low_dest = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
        context, true_region->exit_edges[i].terminator,
        true_region->exit_edges[i].successor_index, plan->else_dispatch_block,
        &previous_low_dest));
    if (common_restore_dest == NULL) {
      common_restore_dest = previous_low_dest;
    } else if (previous_low_dest != common_restore_dest) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_exit_edge_interposition"));
    }
  }
  for (iree_host_size_t i = 0; i < false_region->exit_edge_count; ++i) {
    loom_block_t* previous_low_dest = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
        context, false_region->exit_edges[i].terminator,
        false_region->exit_edges[i].successor_index, plan->merge_restore_block,
        &previous_low_dest));
    if (common_restore_dest == NULL) {
      common_restore_dest = previous_low_dest;
    } else if (previous_low_dest != common_restore_dest) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_exit_edge_interposition"));
    }
  }

  if (common_restore_dest == NULL) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_single_continuation"));
  }
  plan->merge_restore_dest = common_restore_dest;

  return loom_low_lower_set_branch_plan(
      context, source_op,
      loom_low_lower_plan_make(LOOM_AMDGPU_BRANCH_PLAN_IF_ELSE_DIAMOND, plan));
}

static iree_status_t loom_amdgpu_try_emit_if_else_diamond_argument_error(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_emitted) {
  *out_emitted = false;
  loom_block_t* true_dest = loom_cfg_cond_br_true_dest(source_op);
  loom_block_t* false_dest = loom_cfg_cond_br_false_dest(source_op);
  if (true_dest->arg_count != 0 || false_dest->arg_count != 0 ||
      true_dest == false_dest || true_dest->op_count == 0 ||
      false_dest->op_count == 0) {
    return iree_ok_status();
  }
  const loom_op_t* true_terminator = loom_block_const_last_op(true_dest);
  const loom_op_t* false_terminator = loom_block_const_last_op(false_dest);
  if (!loom_cfg_br_isa(true_terminator) || !loom_cfg_br_isa(false_terminator)) {
    return iree_ok_status();
  }
  loom_block_t* passthrough_continuation = NULL;
  const loom_op_t* passthrough_terminator = NULL;
  if (loom_amdgpu_try_false_passthrough_continuation(
          false_dest, &passthrough_continuation, &passthrough_terminator)) {
    return iree_ok_status();
  }
  loom_block_t* merge_block = loom_cfg_br_dest(true_terminator);
  if (merge_block != loom_cfg_br_dest(false_terminator) ||
      merge_block == true_dest || merge_block == false_dest ||
      merge_block->arg_count == 0) {
    return iree_ok_status();
  }
  *out_emitted = true;
  return loom_low_lower_emit_branch_constraint(
      context, source_op, IREE_SV("merge_block_arguments_absent"));
}

static iree_status_t loom_amdgpu_prepare_if_else_diamond(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_op_t* true_terminator, const loom_op_t* false_terminator) {
  loom_block_t* merge_low_dest = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_block(
      context, loom_cfg_br_dest(true_terminator), &merge_low_dest));

  loom_amdgpu_branch_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*plan), (void**)&plan));
  *plan = (loom_amdgpu_branch_plan_t){0};
  plan->if_else_merge_arg_count = merge_low_dest->arg_count;

  if (plan->if_else_merge_arg_count == 0) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_append_low_block(context, &plan->else_dispatch_block));
    IREE_RETURN_IF_ERROR(
        loom_low_lower_append_low_block(context, &plan->restore_block));

    IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
        context, source_op, 1, plan->else_dispatch_block,
        &plan->else_body_block));

    loom_block_t* previous_true_merge = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
        context, true_terminator, 0, plan->else_dispatch_block,
        &previous_true_merge));
    IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
        context, false_terminator, 0, plan->restore_block,
        &plan->restore_dest));
    if (previous_true_merge != plan->restore_dest) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("branch_arms_merge_without_arguments"));
    }

    return loom_low_lower_set_branch_plan(
        context, source_op,
        loom_low_lower_plan_make(LOOM_AMDGPU_BRANCH_PLAN_IF_ELSE_DIAMOND,
                                 plan));
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_verify_if_else_merge_args(
      context, source_op, merge_low_dest, plan->if_else_merge_arg_count));

  IREE_RETURN_IF_ERROR(
      loom_low_lower_append_low_block(context, &plan->else_dispatch_block));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_block_args_like_dest(
      context, merge_low_dest, plan->else_dispatch_block, NULL));
  IREE_RETURN_IF_ERROR(loom_low_lower_append_low_block(
      context, &plan->no_true_else_entry_block));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_append_low_block(context, &plan->true_only_restore_block));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_append_low_block(context, &plan->merge_restore_block));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_block_args_like_dest(
      context, merge_low_dest, plan->merge_restore_block, NULL));

  IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
      context, source_op, 1, plan->no_true_else_entry_block,
      &plan->else_body_block));

  loom_block_t* previous_true_merge = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
      context, true_terminator, 0, plan->else_dispatch_block,
      &previous_true_merge));
  loom_block_t* previous_false_merge = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
      context, false_terminator, 0, plan->merge_restore_block,
      &previous_false_merge));
  if (previous_false_merge != previous_true_merge) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("branch_arms_merge_without_arguments"));
  }
  plan->merge_restore_dest = previous_true_merge;

  return loom_low_lower_set_branch_plan(
      context, source_op,
      loom_low_lower_plan_make(LOOM_AMDGPU_BRANCH_PLAN_IF_ELSE_DIAMOND, plan));
}

static iree_status_t loom_amdgpu_prepare_exec_mask_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_op_t* true_terminator = NULL;
  const loom_op_t* false_terminator = NULL;
  if (loom_amdgpu_try_if_else_diamond(source_op, &true_terminator,
                                      &false_terminator)) {
    return loom_amdgpu_prepare_if_else_diamond(
        context, source_op, true_terminator, false_terminator);
  }
  loom_block_t* passthrough_continuation = NULL;
  const loom_op_t* passthrough_terminator = NULL;
  const bool has_false_passthrough =
      loom_amdgpu_try_false_passthrough_continuation(
          loom_cfg_cond_br_false_dest(source_op), &passthrough_continuation,
          &passthrough_terminator);
  loom_region_t* source_body = loom_amdgpu_source_body(context);
  const iree_host_size_t edge_capacity =
      loom_amdgpu_successor_edge_count(source_body);
  loom_amdgpu_region_exit_edge_t* true_exit_edges = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, edge_capacity, sizeof(*true_exit_edges),
      (void**)&true_exit_edges));
  loom_amdgpu_region_exit_edge_t* false_exit_edges = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, edge_capacity, sizeof(*false_exit_edges),
      (void**)&false_exit_edges));
  uint8_t* true_contains_block = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, source_body->block_count, sizeof(*true_contains_block),
      (void**)&true_contains_block));
  uint8_t* false_contains_block = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, source_body->block_count, sizeof(*false_contains_block),
      (void**)&false_contains_block));
  uint16_t* stack = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, source_body->block_count, sizeof(*stack), (void**)&stack));
  loom_amdgpu_bounded_region_t true_region = {0};
  loom_amdgpu_bounded_region_t false_region = {0};
  if (!has_false_passthrough &&
      loom_amdgpu_try_analyze_if_else_regions(
          source_body, source_op, true_exit_edges, false_exit_edges,
          true_contains_block, false_contains_block, stack, &true_region,
          &false_region)) {
    return loom_amdgpu_prepare_if_else_regions(context, source_op, &true_region,
                                               &false_region);
  }
  bool emitted_diagnostic = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_if_else_diamond_argument_error(
      context, source_op, &emitted_diagnostic));
  if (emitted_diagnostic) {
    return iree_ok_status();
  }
  return loom_amdgpu_prepare_then_masked_region(context, source_op);
}

iree_status_t loom_amdgpu_prepare_branch(void* user_data,
                                         loom_low_lower_context_t* context,
                                         const loom_op_t* source_terminator) {
  (void)user_data;
  if (!loom_cfg_cond_br_isa(source_terminator)) {
    return iree_ok_status();
  }

  loom_type_t condition_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
      context, source_terminator, loom_cfg_cond_br_condition(source_terminator),
      &condition_type));

  bool is_sgpr_mask = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_condition_is_reg_class(
      context, condition_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2,
      &is_sgpr_mask));
  if (!is_sgpr_mask) {
    return iree_ok_status();
  }
  return loom_amdgpu_prepare_exec_mask_branch(context, source_terminator);
}

static iree_status_t loom_amdgpu_emit_exec_restore_block(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_block_t* restore_block,
    loom_block_t* restore_dest) {
  if (restore_block->op_count != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU EXEC restore block was emitted twice");
  }

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, restore_block);
  loom_op_t* restore_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &saved_exec, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
      &restore_op);
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, restore_dest, restore_block->arg_ids,
                               restore_block->arg_count, source_op->location,
                               &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_exec_restore_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_block_t* restore_block,
    loom_block_t* restore_dest, const loom_value_id_t* args,
    uint16_t arg_count) {
  if (restore_block->op_count != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU EXEC restore block was emitted twice");
  }

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, restore_block);
  loom_op_t* restore_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &saved_exec, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
      &restore_op);
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, restore_dest, args, arg_count,
                               source_op->location, &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_exec_restore_passthrough_block(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_block_t* restore_block,
    loom_block_t* restore_dest, const loom_op_t* passthrough_terminator) {
  IREE_ASSERT(loom_cfg_br_isa(passthrough_terminator));
  loom_value_slice_t args = loom_cfg_br_args(passthrough_terminator);
  loom_value_id_t* low_args = NULL;
  if (args.count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
        context, args.count, sizeof(*low_args), (void**)&low_args));
  }
  for (uint16_t i = 0; i < args.count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_lookup_value(context, args.values[i], &low_args[i]));
  }
  return loom_amdgpu_emit_exec_restore_branch(context, source_op, saved_exec,
                                              restore_block, restore_dest,
                                              low_args, args.count);
}

static iree_status_t loom_amdgpu_emit_zero_vgpr_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t value_type, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  bool is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, value_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &is_vgpr));
  if (!is_vgpr) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_merge_vgpr_values"));
  }

  const uint32_t lane_count = loom_type_register_unit_count(value_type);
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_merge_vgpr_values"));
  }
  loom_type_t lane_type =
      loom_type_register(loom_type_register_class_id(value_type), 1);
  if (lane_count == 1) {
    return loom_amdgpu_emit_const_u32(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
                                      lane_type, out_value);
  }

  loom_value_id_t zero_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, lane_type,
      &zero_lane));
  loom_value_id_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    lanes[i] = zero_lane;
  }
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lanes, lane_count, value_type,
      source_op->location, &concat_op));
  *out_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_masked_merge_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_false_value, loom_value_id_t low_true_value,
    loom_value_id_t low_condition, loom_type_t lane_type,
    loom_value_id_t* out_merged_value) {
  const loom_value_id_t select_operands[] = {
      low_false_value,
      low_true_value,
      low_condition,
  };
  loom_op_t* select_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
      select_operands, IREE_ARRAYSIZE(select_operands),
      loom_named_attr_slice_empty(), &lane_type, 1, &select_op));
  *out_merged_value = loom_value_slice_get(loom_low_op_results(select_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_masked_merge_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_false_value, loom_value_id_t low_true_value,
    loom_value_id_t low_condition, loom_type_t value_type,
    loom_value_id_t* out_merged_value) {
  *out_merged_value = LOOM_VALUE_ID_INVALID;
  bool is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, value_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &is_vgpr));
  if (!is_vgpr) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_merge_vgpr_values"));
  }

  const uint32_t lane_count = loom_type_register_unit_count(value_type);
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_merge_vgpr_values"));
  }
  loom_type_t lane_type =
      loom_type_register(loom_type_register_class_id(value_type), 1);
  if (lane_count == 1) {
    return loom_amdgpu_emit_masked_merge_lane(
        context, source_op, low_false_value, low_true_value, low_condition,
        lane_type, out_merged_value);
  }

  loom_value_id_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    loom_value_id_t false_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_false_value, i, lane_type, &false_lane));
    loom_value_id_t true_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_true_value, i, lane_type, &true_lane));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_masked_merge_lane(
        context, source_op, false_lane, true_lane, low_condition, lane_type,
        &lanes[i]));
  }

  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lanes, lane_count, value_type,
      source_op->location, &concat_op));
  *out_merged_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_masked_merge_restore_block(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_value_id_t low_condition,
    const loom_amdgpu_branch_plan_t* plan) {
  IREE_ASSERT(plan->merge_restore_block != NULL);
  IREE_ASSERT(plan->merge_restore_dest != NULL);
  IREE_ASSERT(plan->false_passthrough_terminator != NULL);
  if (plan->merge_restore_block->op_count != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU EXEC merge restore block was emitted "
                            "twice");
  }

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, plan->merge_restore_block);

  loom_op_t* restore_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &saved_exec, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
      &restore_op);

  loom_value_slice_t false_args =
      loom_cfg_br_args(plan->false_passthrough_terminator);
  IREE_ASSERT_EQ(false_args.count, plan->merge_restore_block->arg_count);
  IREE_ASSERT_EQ(false_args.count, plan->merge_restore_dest->arg_count);
  loom_value_id_t* merged_args = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_allocate_scratch_array(
        context, false_args.count, sizeof(*merged_args), (void**)&merged_args);
  }
  for (uint16_t i = 0; i < false_args.count && iree_status_is_ok(status); ++i) {
    loom_value_id_t low_false_value = LOOM_VALUE_ID_INVALID;
    status = loom_low_lower_lookup_value(context, false_args.values[i],
                                         &low_false_value);
    if (!iree_status_is_ok(status)) {
      break;
    }
    const loom_value_id_t low_true_value =
        loom_block_arg_id(plan->merge_restore_block, i);
    loom_module_t* module = loom_low_lower_context_module(context);
    status = loom_amdgpu_emit_masked_merge_value(
        context, source_op, low_false_value, low_true_value, low_condition,
        loom_module_value_type(module, low_true_value), &merged_args[i]);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status =
        loom_low_br_build(builder, plan->merge_restore_dest, merged_args,
                          false_args.count, source_op->location, &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_if_else_merge_restore_block(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_value_id_t low_condition,
    const loom_amdgpu_branch_plan_t* plan) {
  IREE_ASSERT(plan->merge_restore_block != NULL);
  IREE_ASSERT(plan->merge_restore_dest != NULL);
  if (plan->merge_restore_block->op_count != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU EXEC merge restore block was emitted "
                            "twice");
  }

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, plan->merge_restore_block);

  loom_op_t* restore_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &saved_exec, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
      &restore_op);

  const uint16_t arg_count = plan->if_else_merge_arg_count;
  IREE_ASSERT_EQ(arg_count, plan->merge_restore_block->arg_count);
  IREE_ASSERT_EQ(arg_count, plan->merge_restore_dest->arg_count);
  loom_value_id_t* merged_args = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_allocate_scratch_array(
        context, arg_count, sizeof(*merged_args), (void**)&merged_args);
  }
  loom_module_t* module = loom_low_lower_context_module(context);
  for (uint16_t i = 0; i < arg_count && iree_status_is_ok(status); ++i) {
    const loom_value_id_t low_false_value =
        loom_block_arg_id(plan->merge_restore_block, i);
    const loom_value_id_t low_true_value =
        loom_block_arg_id(plan->else_dispatch_block, i);
    status = loom_amdgpu_emit_masked_merge_value(
        context, source_op, low_false_value, low_true_value, low_condition,
        loom_module_value_type(module, low_true_value), &merged_args[i]);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, plan->merge_restore_dest, merged_args,
                               arg_count, source_op->location, &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_no_true_else_entry_block(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, const loom_amdgpu_branch_plan_t* plan) {
  if (plan->no_true_else_entry_block->op_count != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU no-true else entry block was emitted "
                            "twice");
  }

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, plan->no_true_else_entry_block);

  loom_op_t* restore_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &saved_exec, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
      &restore_op);

  const uint16_t arg_count = plan->if_else_merge_arg_count;
  loom_value_id_t* zero_args = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_allocate_scratch_array(
        context, arg_count, sizeof(*zero_args), (void**)&zero_args);
  }
  loom_amdgpu_zero_placeholder_t* placeholders = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_allocate_scratch_array(
        context, arg_count, sizeof(*placeholders), (void**)&placeholders);
  }
  uint16_t placeholder_count = 0;
  loom_module_t* module = loom_low_lower_context_module(context);
  for (uint16_t i = 0; i < arg_count && iree_status_is_ok(status); ++i) {
    zero_args[i] = LOOM_VALUE_ID_INVALID;
    // The no-true path reaches this block with EXEC empty. Restore the original
    // lane mask before defining placeholder VGPRs so the later hardware select
    // never reads an unwritten register operand, even though the placeholders
    // are semantically ignored when every active lane takes the else arm.
    const loom_value_id_t true_arg =
        loom_block_arg_id(plan->else_dispatch_block, i);
    const loom_type_t value_type = loom_module_value_type(module, true_arg);
    for (uint16_t j = 0; j < placeholder_count; ++j) {
      if (loom_type_equal(placeholders[j].type, value_type)) {
        zero_args[i] = placeholders[j].value;
        break;
      }
    }
    if (zero_args[i] != LOOM_VALUE_ID_INVALID) {
      continue;
    }
    status = loom_amdgpu_emit_zero_vgpr_value(context, source_op, value_type,
                                              &zero_args[i]);
    if (iree_status_is_ok(status)) {
      placeholders[placeholder_count++] = (loom_amdgpu_zero_placeholder_t){
          .type = value_type,
          .value = zero_args[i],
      };
    }
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, plan->else_dispatch_block, zero_args,
                               arg_count, source_op->location, &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_else_dispatch_block(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_value_id_t low_condition,
    loom_type_t condition_type, loom_type_t active_type,
    const loom_amdgpu_branch_plan_t* plan) {
  if (plan->else_dispatch_block->op_count != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU else dispatch block was emitted twice");
  }

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, plan->else_dispatch_block);

  loom_op_t* else_active_op = NULL;
  iree_status_t status = iree_ok_status();
  if (plan->if_else_merge_arg_count == 0) {
    // Store-only branches do not introduce merge operands, so no later VALU
    // select can observe values from lanes that skipped a branch body.
    status = loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64_EXEC,
        &saved_exec, 1, loom_named_attr_slice_empty(), &active_type, 1,
        &else_active_op);
  } else {
    // Value-producing branches may reach this block either from the true body
    // or from a no-true placeholder block. Recompute the else mask from the
    // original saved EXEC instead of assuming the predecessor left EXEC set to
    // exactly the true lanes.
    loom_op_t* restore_op = NULL;
    status = loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
        &saved_exec, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
        &restore_op);
    const loom_value_id_t true_mask_operands[] = {
        saved_exec,
        low_condition,
    };
    loom_op_t* true_mask_op = NULL;
    if (iree_status_is_ok(status)) {
      status = loom_amdgpu_emit_low_op(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
          true_mask_operands, IREE_ARRAYSIZE(true_mask_operands),
          loom_named_attr_slice_empty(), &condition_type, 1, &true_mask_op);
    }
    if (iree_status_is_ok(status)) {
      const loom_value_id_t true_mask = loom_op_const_results(true_mask_op)[0];
      status = loom_amdgpu_emit_low_op(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64_EXEC,
          &true_mask, 1, loom_named_attr_slice_empty(), &active_type, 1,
          &else_active_op);
    }
  }
  if (iree_status_is_ok(status)) {
    loom_block_t* const restore_dest = plan->true_only_restore_block != NULL
                                           ? plan->true_only_restore_block
                                           : plan->restore_block;
    status = loom_amdgpu_emit_plain_cond_branch(
        context, source_op, loom_op_const_results(else_active_op)[0],
        plan->else_body_block, restore_dest);
  }

  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_exec_mask_cond_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_condition, loom_block_t* low_true_dest,
    loom_block_t* low_false_dest, loom_type_t condition_type) {
  loom_low_lower_plan_t branch_plan = loom_low_lower_plan_empty();
  if (!loom_low_lower_lookup_branch_plan(context, source_op, &branch_plan)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU divergent branch has no prepared plan");
  }
  const loom_amdgpu_branch_plan_t* plan =
      (const loom_amdgpu_branch_plan_t*)branch_plan.target_data;
  IREE_ASSERT(plan != NULL);
  if (branch_plan.id != LOOM_AMDGPU_BRANCH_PLAN_THEN_MASKED_REGION &&
      branch_plan.id != LOOM_AMDGPU_BRANCH_PLAN_IF_ELSE_DIAMOND) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU divergent branch plan id is invalid");
  }

  loom_type_t active_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &active_type));
  const loom_type_t result_types[] = {condition_type, active_type};

  loom_op_t* saveexec_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64,
      &low_condition, 1, loom_named_attr_slice_empty(), result_types,
      IREE_ARRAYSIZE(result_types), &saveexec_op));
  const loom_value_id_t saved_exec = loom_op_const_results(saveexec_op)[0];
  const loom_value_id_t active = loom_op_const_results(saveexec_op)[1];
  if (plan->restore_block != NULL) {
    if (plan->restore_passthrough_terminator != NULL) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_exec_restore_passthrough_block(
          context, source_op, saved_exec, plan->restore_block,
          plan->restore_dest, plan->restore_passthrough_terminator));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_exec_restore_block(
          context, source_op, saved_exec, plan->restore_block,
          plan->restore_dest));
    }
  }
  if (branch_plan.id == LOOM_AMDGPU_BRANCH_PLAN_THEN_MASKED_REGION &&
      plan->merge_restore_block != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_masked_merge_restore_block(
        context, source_op, saved_exec, low_condition, plan));
  }

  if (branch_plan.id == LOOM_AMDGPU_BRANCH_PLAN_IF_ELSE_DIAMOND &&
      plan->if_else_merge_arg_count == 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_else_dispatch_block(
        context, source_op, saved_exec, low_condition, condition_type,
        active_type, plan));
  }
  if (branch_plan.id == LOOM_AMDGPU_BRANCH_PLAN_IF_ELSE_DIAMOND &&
      plan->if_else_merge_arg_count != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_no_true_else_entry_block(
        context, source_op, saved_exec, plan));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_exec_restore_branch(
        context, source_op, saved_exec, plan->true_only_restore_block,
        plan->merge_restore_dest, plan->else_dispatch_block->arg_ids,
        plan->if_else_merge_arg_count));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_if_else_merge_restore_block(
        context, source_op, saved_exec, low_condition, plan));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_else_dispatch_block(
        context, source_op, saved_exec, low_condition, condition_type,
        active_type, plan));
  }

  return loom_amdgpu_emit_plain_cond_branch(context, source_op, active,
                                            low_true_dest, low_false_dest);
}

iree_status_t loom_amdgpu_emit_cond_branch(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_value_id_t low_condition,
                                           loom_block_t* low_true_dest,
                                           loom_block_t* low_false_dest) {
  (void)user_data;
  loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t condition_type = loom_module_value_type(module, low_condition);
  bool is_scc = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_condition_is_reg_class(
      context, condition_type, LOOM_AMDGPU_REG_CLASS_ID_SCC, 1, &is_scc));
  if (is_scc) {
    return loom_amdgpu_emit_plain_cond_branch(context, source_op, low_condition,
                                              low_true_dest, low_false_dest);
  }

  bool is_sgpr_mask = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_condition_is_reg_class(
      context, condition_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2,
      &is_sgpr_mask));
  if (is_sgpr_mask) {
    return loom_amdgpu_emit_exec_mask_cond_branch(
        context, source_op, low_condition, low_true_dest, low_false_dest,
        condition_type);
  }

  return loom_low_lower_emit_branch_condition_type_unsupported(
      context, source_op, condition_type, IREE_SV("amdgpu.branch_condition"));
}
