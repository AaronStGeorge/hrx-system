// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/loop_execution.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/ir/ir.h"
#include "loom/util/cfg_loop.h"

typedef struct loom_low_loop_execution_binding_t {
  // Next sparse binding in insertion order.
  struct loom_low_loop_execution_binding_t* next;
  // Exact Low control operation emitted for a source loop header.
  const loom_op_t* control_op;
  // Producer-proven execution count for the represented loop.
  loom_low_loop_execution_facts_t facts;
} loom_low_loop_execution_binding_t;

static_assert(sizeof(loom_low_loop_execution_binding_t) <= 32,
              "Low loop execution bindings must remain compact");

struct loom_low_loop_execution_map_t {
  // Owner of map and binding storage.
  iree_arena_allocator_t* arena;
  // First sparse binding, or NULL for an empty map.
  loom_low_loop_execution_binding_t* head;
  // Last sparse binding, or NULL for an empty map.
  loom_low_loop_execution_binding_t* tail;
  // Number of retained bindings.
  uint32_t count;
};

static_assert(sizeof(loom_low_loop_execution_map_t) <= 32,
              "Low loop execution maps must remain compact");

iree_status_t loom_low_loop_execution_map_allocate(
    iree_arena_allocator_t* arena, loom_low_loop_execution_map_t** out_map) {
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(**out_map), (void**)out_map));
  **out_map = (loom_low_loop_execution_map_t){.arena = arena};
  return iree_ok_status();
}

iree_status_t loom_low_loop_execution_map_insert(
    loom_low_loop_execution_map_t* map, const loom_op_t* control_op,
    loom_low_loop_execution_facts_t facts) {
  IREE_ASSERT(control_op != NULL);
  IREE_ASSERT(facts.count_kind == LOOM_LOW_LOOP_EXECUTION_COUNT_BOUNDED ||
              facts.count_kind == LOOM_LOW_LOOP_EXECUTION_COUNT_EXACT);
  IREE_ASSERT_LT(map->count, UINT32_MAX);
  loom_low_loop_execution_binding_t* binding = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(map->arena, sizeof(*binding), (void**)&binding));
  *binding = (loom_low_loop_execution_binding_t){
      .control_op = control_op,
      .facts = facts,
  };
  if (map->tail != NULL) {
    map->tail->next = binding;
  } else {
    map->head = binding;
  }
  map->tail = binding;
  ++map->count;
  return iree_ok_status();
}

iree_status_t loom_low_loop_execution_map_transfer(
    const loom_low_loop_execution_map_t* source,
    loom_low_loop_execution_map_t* target) {
  if (source == target) {
    return iree_ok_status();
  }
  for (const loom_low_loop_execution_binding_t* binding = source->head;
       binding != NULL; binding = binding->next) {
    IREE_RETURN_IF_ERROR(loom_low_loop_execution_map_insert(
        target, binding->control_op, binding->facts));
  }
  return iree_ok_status();
}

iree_status_t loom_low_loop_execution_index_build(
    const loom_low_loop_execution_map_t* map, const loom_region_t* body,
    iree_arena_allocator_t* arena, loom_low_loop_execution_index_t* out_index) {
  *out_index = (loom_low_loop_execution_index_t){0};
  if (map == NULL || map->count == 0 || body == NULL ||
      body->block_count == 0) {
    return iree_ok_status();
  }
  const loom_low_loop_execution_facts_t** facts_by_block = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, body->block_count,
                                                 sizeof(*facts_by_block),
                                                 (void**)&facts_by_block));
  memset(facts_by_block, 0,
         (iree_host_size_t)body->block_count * sizeof(*facts_by_block));
  for (const loom_low_loop_execution_binding_t* binding = map->head;
       binding != NULL; binding = binding->next) {
    const loom_op_t* control_op = binding->control_op;
    if (iree_any_bit_set(control_op->flags, LOOM_OP_FLAG_DEAD) ||
        control_op->parent_block == NULL ||
        control_op->parent_block->parent_region != body ||
        control_op->parent_block->last_op != control_op) {
      continue;
    }
    const uint16_t block_index = control_op->parent_block->region_index;
    IREE_ASSERT_LT(block_index, body->block_count);
    IREE_ASSERT(facts_by_block[block_index] == NULL);
    facts_by_block[block_index] = &binding->facts;
  }
  *out_index = (loom_low_loop_execution_index_t){
      .facts_by_block = facts_by_block,
      .block_count = body->block_count,
  };
  return iree_ok_status();
}

const loom_low_loop_execution_facts_t* loom_low_loop_execution_index_lookup(
    const loom_low_loop_execution_index_t* index, const loom_block_t* header) {
  if (index->facts_by_block == NULL || header == NULL ||
      header->region_index >= index->block_count || header->last_op == NULL) {
    return NULL;
  }
  const loom_low_loop_execution_facts_t* facts =
      index->facts_by_block[header->region_index];
  return facts;
}

struct loom_low_loop_execution_clone_t {
  // Map receiving cloned control bindings.
  loom_low_loop_execution_map_t* target;
  // Source bindings indexed by current source block order.
  const loom_low_loop_execution_binding_t* const* bindings_by_block;
  // Number of entries in |bindings_by_block|.
  iree_host_size_t block_count;
};

iree_status_t loom_low_loop_execution_clone_allocate(
    const loom_low_loop_execution_map_t* source,
    loom_low_loop_execution_map_t* target,
    iree_arena_allocator_t* scratch_arena,
    loom_low_loop_execution_clone_t** out_clone) {
  IREE_RETURN_IF_ERROR(iree_arena_allocate(scratch_arena, sizeof(**out_clone),
                                           (void**)out_clone));
  **out_clone = (loom_low_loop_execution_clone_t){.target = target};
  if (source == NULL || source->count == 0) {
    return iree_ok_status();
  }

  const loom_region_t* body = NULL;
  for (const loom_low_loop_execution_binding_t* binding = source->head;
       binding != NULL && body == NULL; binding = binding->next) {
    if (!iree_any_bit_set(binding->control_op->flags, LOOM_OP_FLAG_DEAD) &&
        binding->control_op->parent_block != NULL) {
      body = binding->control_op->parent_block->parent_region;
    }
  }
  if (body == NULL || body->block_count == 0) {
    return iree_ok_status();
  }

  const loom_low_loop_execution_binding_t** bindings_by_block = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, body->block_count, sizeof(*bindings_by_block),
      (void**)&bindings_by_block));
  memset(bindings_by_block, 0,
         (iree_host_size_t)body->block_count * sizeof(*bindings_by_block));
  for (const loom_low_loop_execution_binding_t* binding = source->head;
       binding != NULL; binding = binding->next) {
    const loom_op_t* control_op = binding->control_op;
    if (iree_any_bit_set(control_op->flags, LOOM_OP_FLAG_DEAD) ||
        control_op->parent_block == NULL ||
        control_op->parent_block->parent_region != body) {
      continue;
    }
    const uint16_t block_index = control_op->parent_block->region_index;
    IREE_ASSERT_LT(block_index, body->block_count);
    IREE_ASSERT(bindings_by_block[block_index] == NULL);
    bindings_by_block[block_index] = binding;
  }
  (*out_clone)->bindings_by_block = bindings_by_block;
  (*out_clone)->block_count = body->block_count;
  return iree_ok_status();
}

iree_status_t loom_low_loop_execution_clone_op(void* context,
                                               const loom_op_t* source_op,
                                               loom_op_t* target_op) {
  loom_low_loop_execution_clone_t* clone = context;
  if (clone->bindings_by_block == NULL || source_op->parent_block == NULL ||
      source_op->parent_block->region_index >= clone->block_count) {
    return iree_ok_status();
  }
  const loom_low_loop_execution_binding_t* binding =
      clone->bindings_by_block[source_op->parent_block->region_index];
  return binding != NULL && binding->control_op == source_op
             ? loom_low_loop_execution_map_insert(clone->target, target_op,
                                                  binding->facts)
             : iree_ok_status();
}

enum loom_low_loop_execution_factor_flag_bits_e {
  // The represented loop path is structurally canonical.
  LOOM_LOW_LOOP_EXECUTION_FACTOR_FLAG_VALID = 1u << 0,
  // |product| is representable and every nonzero factor is known.
  LOOM_LOW_LOOP_EXECUTION_FACTOR_FLAG_KNOWN = 1u << 1,
};
typedef uint8_t loom_low_loop_execution_factor_flags_t;

typedef struct loom_low_loop_execution_factor_t {
  // Path product, meaningful only when KNOWN is set.
  uint64_t product;
  // Structural validity and arithmetic certainty of this path.
  loom_low_loop_execution_factor_flags_t flags;
} loom_low_loop_execution_factor_t;

static loom_low_loop_execution_factor_t loom_low_loop_execution_multiply(
    loom_low_loop_execution_factor_t lhs,
    loom_low_loop_execution_factor_t rhs) {
  loom_low_loop_execution_factor_t result = {0};
  if (!iree_all_bits_set(lhs.flags,
                         LOOM_LOW_LOOP_EXECUTION_FACTOR_FLAG_VALID) ||
      !iree_all_bits_set(rhs.flags,
                         LOOM_LOW_LOOP_EXECUTION_FACTOR_FLAG_VALID)) {
    return result;
  }
  result.flags = LOOM_LOW_LOOP_EXECUTION_FACTOR_FLAG_VALID;
  const bool lhs_known =
      iree_all_bits_set(lhs.flags, LOOM_LOW_LOOP_EXECUTION_FACTOR_FLAG_KNOWN);
  const bool rhs_known =
      iree_all_bits_set(rhs.flags, LOOM_LOW_LOOP_EXECUTION_FACTOR_FLAG_KNOWN);
  if ((lhs_known && lhs.product == 0) || (rhs_known && rhs.product == 0)) {
    result.flags |= LOOM_LOW_LOOP_EXECUTION_FACTOR_FLAG_KNOWN;
    return result;
  }
  if (lhs_known && rhs_known &&
      iree_checked_mul_u64(lhs.product, rhs.product, &result.product)) {
    result.flags |= LOOM_LOW_LOOP_EXECUTION_FACTOR_FLAG_KNOWN;
  }
  return result;
}

static loom_low_loop_execution_factor_t
loom_low_loop_execution_path_index_factor(
    const loom_low_loop_execution_path_index_t* index, uint32_t level,
    uint32_t loop_index) {
  const iree_host_size_t offset =
      (iree_host_size_t)level * index->forest->interval_count + loop_index;
  return (loom_low_loop_execution_factor_t){
      .product = index->products[offset],
      .flags = index->factor_flags[offset],
  };
}

iree_status_t loom_low_loop_execution_path_index_build(
    const loom_cfg_loop_forest_t* forest,
    const loom_low_loop_execution_index_t* facts, iree_arena_allocator_t* arena,
    loom_low_loop_execution_path_index_t* out_index) {
  *out_index = (loom_low_loop_execution_path_index_t){
      .forest = forest,
  };
  if (forest->interval_count == 0 || forest->innermost_loop_indices == NULL) {
    return iree_ok_status();
  }

  uint32_t maximum_depth = 0;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, forest->interval_count,
                                                 sizeof(*out_index->depths),
                                                 (void**)&out_index->depths));
  for (uint32_t i = 0; i < forest->interval_count; ++i) {
    const uint32_t parent = forest->intervals[i].parent_loop_index;
    IREE_ASSERT(parent == LOOM_CFG_LOOP_NONE || parent < i);
    const uint32_t depth =
        parent == LOOM_CFG_LOOP_NONE ? 1 : out_index->depths[parent] + 1;
    out_index->depths[i] = depth;
    maximum_depth = iree_max(maximum_depth, depth);
  }
  const uint32_t level_count =
      32u - (uint32_t)iree_math_count_leading_zeros_u32(maximum_depth);
  out_index->level_count = level_count;
  const iree_host_size_t slot_count =
      forest->interval_count * (iree_host_size_t)level_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, slot_count, sizeof(*out_index->ancestors),
      (void**)&out_index->ancestors));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, slot_count,
                                                 sizeof(*out_index->products),
                                                 (void**)&out_index->products));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, slot_count, sizeof(*out_index->factor_flags),
      (void**)&out_index->factor_flags));

  for (uint32_t i = 0; i < forest->interval_count; ++i) {
    const loom_cfg_loop_interval_t* interval = &forest->intervals[i];
    out_index->ancestors[i] = interval->parent_loop_index;
    out_index->products[i] = 0;
    out_index->factor_flags[i] =
        interval->is_canonical ? LOOM_LOW_LOOP_EXECUTION_FACTOR_FLAG_VALID : 0;
    const loom_low_loop_execution_facts_t* loop_facts =
        facts->facts_by_block != NULL
            ? facts->facts_by_block[interval->header_index]
            : NULL;
    if (interval->is_canonical && loop_facts != NULL) {
      out_index->products[i] = loop_facts->trip_count;
      out_index->factor_flags[i] |= LOOM_LOW_LOOP_EXECUTION_FACTOR_FLAG_KNOWN;
    }
  }
  for (uint32_t level = 1; level < level_count; ++level) {
    const iree_host_size_t row =
        (iree_host_size_t)level * forest->interval_count;
    const iree_host_size_t previous_row = row - forest->interval_count;
    for (uint32_t i = 0; i < forest->interval_count; ++i) {
      const uint32_t middle = out_index->ancestors[previous_row + i];
      if (middle == LOOM_CFG_LOOP_NONE) {
        out_index->ancestors[row + i] = LOOM_CFG_LOOP_NONE;
        out_index->products[row + i] = 0;
        out_index->factor_flags[row + i] = 0;
        continue;
      }
      out_index->ancestors[row + i] =
          out_index->ancestors[previous_row + middle];
      const loom_low_loop_execution_factor_t factor =
          loom_low_loop_execution_multiply(
              loom_low_loop_execution_path_index_factor(out_index, level - 1,
                                                        i),
              loom_low_loop_execution_path_index_factor(out_index, level - 1,
                                                        middle));
      out_index->products[row + i] = factor.product;
      out_index->factor_flags[row + i] = factor.flags;
    }
  }
  return iree_ok_status();
}

// Computes the maximum execution multiplier introduced by crossing from one
// loop nesting level into another. The doubling table makes each query
// logarithmic while retaining zero-trip absorption and path-local absence.
bool loom_low_loop_execution_path_multiplier(
    const loom_low_loop_execution_path_index_t* index, uint32_t source_block,
    uint32_t destination_block, uint64_t* out_execution_multiplier) {
  const loom_cfg_loop_forest_t* forest = index->forest;
  if (index->level_count == 0) {
    return false;
  }
  const uint32_t source_loop = forest->innermost_loop_indices[source_block];
  uint32_t loop_index = forest->innermost_loop_indices[destination_block];
  if (loop_index == LOOM_CFG_LOOP_NONE || loop_index == source_loop) {
    return false;
  }
  const uint32_t source_depth =
      source_loop == LOOM_CFG_LOOP_NONE ? 0 : index->depths[source_loop];
  const uint32_t destination_depth = index->depths[loop_index];
  if (source_depth >= destination_depth) {
    return false;
  }
  uint32_t remaining_depth = destination_depth - source_depth;
  loom_low_loop_execution_factor_t factor = {
      .product = 1,
      .flags = LOOM_LOW_LOOP_EXECUTION_FACTOR_FLAG_VALID |
               LOOM_LOW_LOOP_EXECUTION_FACTOR_FLAG_KNOWN,
  };

  const loom_cfg_loop_interval_t* destination_interval =
      &forest->intervals[loop_index];
  if (destination_interval->header_index == destination_block) {
    loom_low_loop_execution_factor_t header_factor =
        loom_low_loop_execution_path_index_factor(index, 0, loop_index);
    if (iree_all_bits_set(header_factor.flags,
                          LOOM_LOW_LOOP_EXECUTION_FACTOR_FLAG_KNOWN)) {
      if (header_factor.product == UINT64_MAX) {
        header_factor.flags &= ~LOOM_LOW_LOOP_EXECUTION_FACTOR_FLAG_KNOWN;
      } else {
        ++header_factor.product;
      }
    }
    factor = loom_low_loop_execution_multiply(factor, header_factor);
    loop_index = destination_interval->parent_loop_index;
    --remaining_depth;
  }
  for (uint32_t level = 0; remaining_depth != 0; ++level) {
    if (remaining_depth & 1u) {
      if (loop_index == LOOM_CFG_LOOP_NONE || level >= index->level_count) {
        return false;
      }
      factor = loom_low_loop_execution_multiply(
          factor,
          loom_low_loop_execution_path_index_factor(index, level, loop_index));
      const iree_host_size_t offset =
          (iree_host_size_t)level * forest->interval_count + loop_index;
      loop_index = index->ancestors[offset];
    }
    remaining_depth >>= 1;
  }
  if (loop_index != source_loop ||
      !iree_all_bits_set(factor.flags,
                         LOOM_LOW_LOOP_EXECUTION_FACTOR_FLAG_VALID |
                             LOOM_LOW_LOOP_EXECUTION_FACTOR_FLAG_KNOWN)) {
    return false;
  }
  *out_execution_multiplier = factor.product;
  return true;
}
