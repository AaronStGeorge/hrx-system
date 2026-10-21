// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <string.h>

#include "loom/analysis/liveness.h"

static uintptr_t loom_liveness_operation_pointer(
    const loom_liveness_analysis_t* analysis, uint32_t operation_index) {
  return (uintptr_t)loom_liveness_operation_at(analysis, operation_index)->op;
}

// Sorts row indices by operation pointer with one stable pass for each pointer
// byte that varies. Operations live in a small number of arena blocks in the
// common case, so this usually needs two or three linear passes. The retained
// result stays four bytes per operation regardless of host pointer width.
static iree_status_t loom_liveness_sort_operation_indices(
    const loom_liveness_analysis_t* analysis, uint32_t* indices,
    iree_arena_allocator_t* block_arena) {
  const uint32_t count = (uint32_t)analysis->operation_count;
  IREE_ASSERT_EQ((iree_host_size_t)count, analysis->operation_count);
  if (count < 2) {
    return iree_ok_status();
  }

  uint32_t* scratch_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      block_arena, count, sizeof(*scratch_indices), (void**)&scratch_indices));

  const uintptr_t first_pointer =
      loom_liveness_operation_pointer(analysis, indices[0]);
  uintptr_t varying_bits = 0;
  for (uint32_t i = 1; i < count; ++i) {
    varying_bits |=
        first_pointer ^ loom_liveness_operation_pointer(analysis, indices[i]);
  }

  uint32_t* source = indices;
  uint32_t* target = scratch_indices;
  for (uint32_t shift = 0; shift < sizeof(uintptr_t) * 8u; shift += 8u) {
    if (((varying_bits >> shift) & UINT8_MAX) == 0) {
      continue;
    }
    uint32_t offsets[UINT8_MAX + 1u] = {0};
    for (uint32_t i = 0; i < count; ++i) {
      const uint8_t byte =
          (uint8_t)(loom_liveness_operation_pointer(analysis, source[i]) >>
                    shift);
      ++offsets[byte];
    }
    uint32_t start = 0;
    for (uint32_t i = 0; i <= UINT8_MAX; ++i) {
      const uint32_t bucket_count = offsets[i];
      offsets[i] = start;
      start += bucket_count;
    }
    for (uint32_t i = 0; i < count; ++i) {
      const uint8_t byte =
          (uint8_t)(loom_liveness_operation_pointer(analysis, source[i]) >>
                    shift);
      target[offsets[byte]++] = source[i];
    }
    uint32_t* temporary = source;
    source = target;
    target = temporary;
  }
  if (source != indices) {
    memcpy(indices, source, (iree_host_size_t)count * sizeof(*indices));
  }
  return iree_ok_status();
}

iree_status_t loom_liveness_retain_operation_index(
    loom_liveness_analysis_t* analysis, iree_arena_allocator_t* arena) {
  if (analysis->operation_indices_by_pointer != NULL ||
      analysis->operation_count == 0) {
    return iree_ok_status();
  }
  uint32_t* indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, analysis->operation_count, sizeof(*indices), (void**)&indices));
  for (iree_host_size_t i = 0; i < analysis->operation_count; ++i) {
    indices[i] = (uint32_t)i;
  }
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  const iree_status_t status =
      loom_liveness_sort_operation_indices(analysis, indices, &scratch_arena);
  iree_arena_deinitialize(&scratch_arena);
  IREE_RETURN_IF_ERROR(status);
  analysis->operation_indices_by_pointer = indices;
  return iree_ok_status();
}

uint32_t loom_liveness_operation_index_for_op(
    const loom_liveness_analysis_t* analysis, const loom_op_t* op) {
  if (analysis->operation_count == 0) {
    return UINT32_MAX;
  }
  IREE_ASSERT_ARGUMENT(analysis->operation_indices_by_pointer);
  const uintptr_t pointer = (uintptr_t)op;
  uint32_t first = 0;
  uint32_t count = (uint32_t)analysis->operation_count;
  while (count != 0) {
    const uint32_t half = count / 2;
    const uint32_t middle = first + half;
    const uint32_t operation_index =
        analysis->operation_indices_by_pointer[middle];
    const loom_liveness_operation_point_t* operation =
        loom_liveness_operation_at(analysis, operation_index);
    const uintptr_t candidate = (uintptr_t)operation->op;
    if (candidate < pointer) {
      first = middle + 1;
      count -= half + 1;
    } else {
      count = half;
    }
  }
  if (first == analysis->operation_count) {
    return UINT32_MAX;
  }
  const uint32_t operation_index =
      analysis->operation_indices_by_pointer[first];
  const loom_liveness_operation_point_t* operation =
      loom_liveness_operation_at(analysis, operation_index);
  return operation->op == op ? operation_index : UINT32_MAX;
}
