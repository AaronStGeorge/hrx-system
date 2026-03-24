// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_table.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

//===----------------------------------------------------------------------===//
// Capacity management
//===----------------------------------------------------------------------===//

static iree_status_t loom_value_fact_table_ensure_capacity(
    loom_value_fact_table_t* table, iree_host_size_t minimum_count) {
  if (minimum_count <= table->capacity) return iree_ok_status();

  iree_host_size_t new_capacity =
      table->capacity > 0 ? table->capacity * 2 : 256;
  while (new_capacity < minimum_count) {
    new_capacity *= 2;
  }

  loom_value_facts_t* new_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(table->arena, new_capacity,
                                                 sizeof(loom_value_facts_t),
                                                 (void**)&new_entries));

  // Zero-fill: known_divisor == 0 is the "undefined" sentinel.
  memset(new_entries, 0, new_capacity * sizeof(loom_value_facts_t));
  if (table->count > 0) {
    memcpy(new_entries, table->entries,
           table->count * sizeof(loom_value_facts_t));
  }

  table->entries = new_entries;
  table->capacity = new_capacity;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Scratch buffers
//===----------------------------------------------------------------------===//

iree_status_t loom_value_fact_table_facts_scratch(
    loom_value_fact_table_t* table, iree_host_size_t count,
    loom_value_facts_t** out) {
  if (count <= table->scratch.facts.capacity) {
    *out = table->scratch.facts.values;
    return iree_ok_status();
  }
  loom_value_facts_t* new_scratch = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      table->arena, count, sizeof(loom_value_facts_t), (void**)&new_scratch));
  table->scratch.facts.values = new_scratch;
  table->scratch.facts.capacity = count;
  *out = new_scratch;
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_value_id_scratch(
    loom_value_fact_table_t* table, iree_host_size_t count,
    loom_value_id_t** out) {
  if (count <= table->scratch.value_ids.capacity) {
    *out = table->scratch.value_ids.values;
    return iree_ok_status();
  }
  loom_value_id_t* new_scratch = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      table->arena, count, sizeof(loom_value_id_t), (void**)&new_scratch));
  table->scratch.value_ids.values = new_scratch;
  table->scratch.value_ids.capacity = count;
  *out = new_scratch;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t loom_value_fact_table_initialize(
    loom_value_fact_table_t* table, iree_arena_allocator_t* arena,
    iree_host_size_t initial_capacity) {
  memset(table, 0, sizeof(*table));
  table->arena = arena;
  if (initial_capacity == 0) return iree_ok_status();
  return loom_value_fact_table_ensure_capacity(table, initial_capacity);
}

iree_status_t loom_value_fact_table_define(loom_value_fact_table_t* table,
                                           loom_value_id_t value_id,
                                           loom_value_facts_t facts) {
  IREE_RETURN_IF_ERROR(loom_value_fact_table_ensure_capacity(
      table, (iree_host_size_t)value_id + 1));
  table->entries[value_id] = facts;
  if ((iree_host_size_t)value_id + 1 > table->count) {
    table->count = (iree_host_size_t)value_id + 1;
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Forward pass: compute facts for ops
//===----------------------------------------------------------------------===//

iree_status_t loom_value_fact_table_compute_op(loom_value_fact_table_t* table,
                                               const loom_module_t* module,
                                               const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable || !vtable->fold) return iree_ok_status();

  // Get scratch for operand + result facts.
  iree_host_size_t total =
      (iree_host_size_t)op->operand_count + op->result_count;
  loom_value_facts_t* scratch = NULL;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_facts_scratch(table, total, &scratch));
  loom_value_facts_t* operand_facts = scratch;
  loom_value_facts_t* result_facts = scratch + op->operand_count;

  // Gather operand facts from the table.
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    operand_facts[i] = loom_value_fact_table_lookup(table, operands[i]);
  }

  // Call the fold function.
  vtable->fold(module, op, operand_facts, result_facts);

  // Store result facts.
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] != LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(
          loom_value_fact_table_define(table, results[i], result_facts[i]));
    }
  }

  return iree_ok_status();
}

#define LOOM_FACT_TABLE_INITIAL_REGION_STACK 8

iree_status_t loom_value_fact_table_compute(loom_value_fact_table_t* table,
                                            loom_module_t* module,
                                            loom_func_like_t function) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body) return iree_ok_status();

  // Iterative DFS over all regions in the function (same pattern as
  // loom_rewriter_seed_function). Visits ops in dominance order so
  // operand facts are computed before their users.
  iree_host_size_t stack_capacity = LOOM_FACT_TABLE_INITIAL_REGION_STACK;
  loom_region_t** region_stack = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(table->arena, stack_capacity,
                                                 sizeof(loom_region_t*),
                                                 (void**)&region_stack));
  iree_host_size_t stack_count = 0;
  region_stack[stack_count++] = body;

  while (stack_count > 0) {
    loom_region_t* region = region_stack[--stack_count];
    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) {
        IREE_RETURN_IF_ERROR(
            loom_value_fact_table_compute_op(table, module, op));
        if (op->region_count == 0) continue;
        // Ensure space for nested regions.
        iree_host_size_t needed = stack_count + op->region_count;
        if (needed > stack_capacity) {
          IREE_RETURN_IF_ERROR(iree_arena_grow_array(
              table->arena, stack_count, needed, sizeof(loom_region_t*),
              &stack_capacity, (void**)&region_stack));
        }
        loom_region_t** regions = loom_op_regions(op);
        for (uint8_t r = 0; r < op->region_count; ++r) {
          if (regions[r]) {
            region_stack[stack_count++] = regions[r];
          }
        }
      }
    }
  }

  return iree_ok_status();
}
