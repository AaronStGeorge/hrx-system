// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/edge_affinity.h"

#include <stdlib.h>

#include "iree/base/internal/math.h"

typedef struct loom_low_edge_affinity_candidate_t {
  // Packed ordered source storage-owner ordinals.
  uint64_t key;
  // Logical edge contributing this preference.
  uint32_t edge_group;
  // Stable input position selecting deterministic provenance.
  uint32_t sequence;
  // Operation forwarding this pair.
  const loom_op_t* op;
} loom_low_edge_affinity_candidate_t;

static uint64_t loom_low_edge_affinity_pair_key(loom_value_ordinal_t lhs,
                                                loom_value_ordinal_t rhs) {
  const loom_value_ordinal_t first = iree_min(lhs, rhs);
  const loom_value_ordinal_t second = iree_max(lhs, rhs);
  return ((uint64_t)first << 32) | second;
}

static int loom_low_edge_affinity_compare_constraints(const void* lhs,
                                                      const void* rhs) {
  const loom_low_edge_affinity_constraint_t* lhs_constraint =
      (const loom_low_edge_affinity_constraint_t*)lhs;
  const loom_low_edge_affinity_constraint_t* rhs_constraint =
      (const loom_low_edge_affinity_constraint_t*)rhs;
  const uint64_t lhs_key = loom_low_edge_affinity_pair_key(
      lhs_constraint->first, lhs_constraint->second);
  const uint64_t rhs_key = loom_low_edge_affinity_pair_key(
      rhs_constraint->first, rhs_constraint->second);
  return lhs_key < rhs_key ? -1 : lhs_key > rhs_key ? 1 : 0;
}

static int loom_low_edge_affinity_compare_inputs(const void* lhs,
                                                 const void* rhs) {
  const loom_low_edge_affinity_input_t* lhs_input =
      (const loom_low_edge_affinity_input_t*)lhs;
  const loom_low_edge_affinity_input_t* rhs_input =
      (const loom_low_edge_affinity_input_t*)rhs;
  if (lhs_input->destination_origin != rhs_input->destination_origin) {
    return lhs_input->destination_origin < rhs_input->destination_origin ? -1
                                                                         : 1;
  }
  if (lhs_input->edge_group != rhs_input->edge_group) {
    return lhs_input->edge_group < rhs_input->edge_group ? -1 : 1;
  }
  if (lhs_input->source_origin != rhs_input->source_origin) {
    return lhs_input->source_origin < rhs_input->source_origin ? -1 : 1;
  }
  return lhs_input->sequence < rhs_input->sequence   ? -1
         : lhs_input->sequence > rhs_input->sequence ? 1
                                                     : 0;
}

static int loom_low_edge_affinity_compare_candidates(const void* lhs,
                                                     const void* rhs) {
  const loom_low_edge_affinity_candidate_t* lhs_candidate =
      (const loom_low_edge_affinity_candidate_t*)lhs;
  const loom_low_edge_affinity_candidate_t* rhs_candidate =
      (const loom_low_edge_affinity_candidate_t*)rhs;
  if (lhs_candidate->key != rhs_candidate->key) {
    return lhs_candidate->key < rhs_candidate->key ? -1 : 1;
  }
  if (lhs_candidate->edge_group != rhs_candidate->edge_group) {
    return lhs_candidate->edge_group < rhs_candidate->edge_group ? -1 : 1;
  }
  return lhs_candidate->sequence < rhs_candidate->sequence   ? -1
         : lhs_candidate->sequence > rhs_candidate->sequence ? 1
                                                             : 0;
}

static bool loom_low_edge_affinity_sorted_pair_contains(
    const loom_low_edge_affinity_constraint_t* pairs,
    iree_host_size_t pair_count, uint64_t key) {
  iree_host_size_t first = 0;
  iree_host_size_t count = pair_count;
  while (count != 0) {
    const iree_host_size_t half = count / 2;
    const iree_host_size_t middle = first + half;
    const uint64_t middle_key = loom_low_edge_affinity_pair_key(
        pairs[middle].first, pairs[middle].second);
    if (middle_key < key) {
      first = middle + 1;
      count -= half + 1;
    } else {
      count = half;
    }
  }
  return first < pair_count &&
         loom_low_edge_affinity_pair_key(pairs[first].first,
                                         pairs[first].second) == key;
}

iree_status_t loom_low_edge_affinity_build(
    loom_low_edge_affinity_input_t* inputs, iree_host_size_t input_count,
    loom_low_edge_affinity_constraint_t* constraints,
    iree_host_size_t constraint_count, loom_value_ordinal_t value_count,
    iree_arena_allocator_t* arena,
    loom_low_edge_affinity_list_t* out_affinities) {
  *out_affinities = (loom_low_edge_affinity_list_t){0};
  if (input_count < 2 || constraint_count == 0) {
    return iree_ok_status();
  }

  iree_host_size_t hard_pair_count = 0;
  for (iree_host_size_t i = 0; i < constraint_count; ++i) {
    IREE_ASSERT_LT(constraints[i].first, value_count);
    IREE_ASSERT_LT(constraints[i].second, value_count);
    if (constraints[i].first != constraints[i].second) {
      constraints[hard_pair_count++] = (loom_low_edge_affinity_constraint_t){
          .first = iree_min(constraints[i].first, constraints[i].second),
          .second = iree_max(constraints[i].first, constraints[i].second),
      };
    }
  }
  if (hard_pair_count == 0) {
    return iree_ok_status();
  }
  qsort(constraints, hard_pair_count, sizeof(*constraints),
        loom_low_edge_affinity_compare_constraints);
  iree_host_size_t unique_hard_pair_count = 0;
  for (iree_host_size_t i = 0; i < hard_pair_count; ++i) {
    if (unique_hard_pair_count == 0 ||
        constraints[i].first != constraints[unique_hard_pair_count - 1].first ||
        constraints[i].second !=
            constraints[unique_hard_pair_count - 1].second) {
      constraints[unique_hard_pair_count++] = constraints[i];
    }
  }

  qsort(inputs, input_count, sizeof(*inputs),
        loom_low_edge_affinity_compare_inputs);
  iree_host_size_t unique_input_count = 0;
  for (iree_host_size_t i = 0; i < input_count; ++i) {
    const loom_low_edge_affinity_input_t* input = &inputs[i];
    if (unique_input_count == 0 ||
        input->destination_origin !=
            inputs[unique_input_count - 1].destination_origin ||
        input->edge_group != inputs[unique_input_count - 1].edge_group ||
        input->source_origin != inputs[unique_input_count - 1].source_origin) {
      inputs[unique_input_count++] = *input;
    }
  }
  uint32_t* input_offsets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, (iree_host_size_t)value_count + 1, sizeof(*input_offsets),
      (void**)&input_offsets));
  memset(input_offsets, 0,
         ((iree_host_size_t)value_count + 1) * sizeof(*input_offsets));
  for (iree_host_size_t i = 0; i < unique_input_count; ++i) {
    IREE_ASSERT_LT(inputs[i].destination_origin, value_count);
    IREE_ASSERT_LT(inputs[i].source_origin, value_count);
    ++input_offsets[inputs[i].destination_origin + 1];
  }
  for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
    input_offsets[i + 1] += input_offsets[i];
  }

  loom_low_edge_affinity_candidate_t* candidates = NULL;
  iree_host_size_t candidate_count = 0;
  iree_host_size_t candidate_capacity = 0;
  for (iree_host_size_t h = 0; h < unique_hard_pair_count; ++h) {
    const loom_value_ordinal_t first_destination = constraints[h].first;
    const loom_value_ordinal_t second_destination = constraints[h].second;
    uint32_t first_index = input_offsets[first_destination];
    const uint32_t first_end = input_offsets[first_destination + 1];
    uint32_t second_index = input_offsets[second_destination];
    const uint32_t second_end = input_offsets[second_destination + 1];
    while (first_index < first_end && second_index < second_end) {
      const uint32_t first_group = inputs[first_index].edge_group;
      const uint32_t second_group = inputs[second_index].edge_group;
      uint32_t first_group_end = first_index + 1;
      while (first_group_end < first_end &&
             inputs[first_group_end].edge_group == first_group) {
        ++first_group_end;
      }
      uint32_t second_group_end = second_index + 1;
      while (second_group_end < second_end &&
             inputs[second_group_end].edge_group == second_group) {
        ++second_group_end;
      }
      if (first_group < second_group) {
        first_index = first_group_end;
        continue;
      }
      if (second_group < first_group) {
        second_index = second_group_end;
        continue;
      }
      for (uint32_t i = first_index; i < first_group_end; ++i) {
        for (uint32_t j = second_index; j < second_group_end; ++j) {
          const loom_value_ordinal_t first_source = inputs[i].source_origin;
          const loom_value_ordinal_t second_source = inputs[j].source_origin;
          if (first_source == second_source) {
            continue;
          }
          const uint64_t source_key =
              loom_low_edge_affinity_pair_key(first_source, second_source);
          if (loom_low_edge_affinity_sorted_pair_contains(
                  constraints, unique_hard_pair_count, source_key)) {
            continue;
          }
          if (candidate_count == candidate_capacity) {
            IREE_RETURN_IF_ERROR(iree_arena_grow_array(
                arena, candidate_count, candidate_count + 1,
                sizeof(*candidates), &candidate_capacity, (void**)&candidates));
          }
          const loom_low_edge_affinity_input_t* provenance =
              inputs[i].sequence <= inputs[j].sequence ? &inputs[i]
                                                       : &inputs[j];
          candidates[candidate_count++] = (loom_low_edge_affinity_candidate_t){
              .key = source_key,
              .edge_group = first_group,
              .sequence = provenance->sequence,
              .op = provenance->op,
          };
        }
      }
      first_index = first_group_end;
      second_index = second_group_end;
    }
  }
  if (candidate_count == 0) {
    return iree_ok_status();
  }

  qsort(candidates, candidate_count, sizeof(*candidates),
        loom_low_edge_affinity_compare_candidates);
  iree_host_size_t affinity_count = 1;
  for (iree_host_size_t i = 1; i < candidate_count; ++i) {
    affinity_count += candidates[i].key != candidates[i - 1].key;
  }
  loom_low_edge_affinity_t* affinities = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, affinity_count, sizeof(*affinities), (void**)&affinities));
  iree_host_size_t affinity_index = 0;
  for (iree_host_size_t i = 0; i < candidate_count;) {
    const loom_low_edge_affinity_candidate_t* first = &candidates[i];
    const loom_low_edge_affinity_candidate_t* provenance = first;
    uint32_t priority = 0;
    uint32_t previous_group = UINT32_MAX;
    iree_host_size_t end = i;
    while (end < candidate_count && candidates[end].key == first->key) {
      if (candidates[end].edge_group != previous_group) {
        priority = iree_min(priority + 1, (uint32_t)UINT16_MAX);
        previous_group = candidates[end].edge_group;
      }
      if (candidates[end].sequence < provenance->sequence) {
        provenance = &candidates[end];
      }
      ++end;
    }
    affinities[affinity_index++] = (loom_low_edge_affinity_t){
        .first = (loom_value_ordinal_t)(first->key >> 32),
        .second = (loom_value_ordinal_t)first->key,
        .priority = (uint16_t)priority,
        .op = provenance->op,
    };
    i = end;
  }
  IREE_ASSERT_EQ(affinity_index, affinity_count);
  *out_affinities = (loom_low_edge_affinity_list_t){
      .values = affinities,
      .count = affinity_count,
  };
  return iree_ok_status();
}
