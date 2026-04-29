// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/liveness.h"

#include "iree/base/internal/math.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/util/cfg_graph.h"

typedef struct loom_liveness_bitset_t {
  uint64_t* words;
  iree_host_size_t word_count;
} loom_liveness_bitset_t;

typedef struct loom_liveness_block_state_t {
  loom_liveness_bitset_t use;
  loom_liveness_bitset_t def;
  loom_liveness_bitset_t live_in;
  loom_liveness_bitset_t live_out;
} loom_liveness_block_state_t;

typedef struct loom_liveness_mutable_interval_t {
  // Interval fields being assembled.
  loom_liveness_interval_t interval;
  // True once either a definition or live point has established bounds.
  bool has_bounds;
} loom_liveness_mutable_interval_t;

typedef struct loom_liveness_pressure_state_t {
  // Mutable pressure summaries being built.
  loom_liveness_pressure_summary_t* summaries;
  // Number of initialized pressure summaries.
  iree_host_size_t count;
  // Number of summary records allocated.
  iree_host_size_t capacity;
} loom_liveness_pressure_state_t;

typedef struct loom_liveness_build_state_t {
  // Module containing the analyzed region.
  loom_module_t* module;
  // Region being analyzed.
  const loom_region_t* region;
  // Optional operation order for each block.
  loom_liveness_order_t order;
  // Arena owning all analysis result storage.
  iree_arena_allocator_t* arena;
  // Value IDs indexed by region-local value ordinal.
  loom_value_id_t* value_ids;
  // Number of initialized local value IDs.
  iree_host_size_t value_count;
  // Number of local value ID records allocated.
  iree_host_size_t value_id_capacity;
  // Number of 64-bit words in local value bitsets.
  iree_host_size_t word_count;
  // Mutable per-block liveness state.
  loom_liveness_block_state_t* block_states;
  // Mutable intervals indexed by region-local value ordinal.
  loom_liveness_mutable_interval_t* interval_states;
  // Region-local value ordinal to interval-index table.
  uint32_t* value_interval_indices;
  // Mutable pressure summary state.
  loom_liveness_pressure_state_t pressure_state;
  // True while |module|'s value ordinal scratch is owned by this analysis.
  bool value_ordinal_scratch_acquired;
} loom_liveness_build_state_t;

typedef iree_status_t (*loom_liveness_value_fn_t)(void* user_data,
                                                  loom_value_id_t value_id);

typedef struct loom_liveness_value_callback_t {
  // Function invoked for each visited value.
  loom_liveness_value_fn_t fn;
  // Opaque callback payload passed to |fn|.
  void* user_data;
} loom_liveness_value_callback_t;

static inline loom_liveness_value_callback_t loom_liveness_value_callback_make(
    loom_liveness_value_fn_t fn, void* user_data) {
  return (loom_liveness_value_callback_t){
      .fn = fn,
      .user_data = user_data,
  };
}

//===----------------------------------------------------------------------===//
// Bitsets
//===----------------------------------------------------------------------===//

static iree_host_size_t loom_liveness_word_count(iree_host_size_t bit_count) {
  return (bit_count + 63u) / 64u;
}

static iree_status_t loom_liveness_bitset_allocate(
    iree_arena_allocator_t* arena, iree_host_size_t word_count,
    loom_liveness_bitset_t* out_bitset) {
  out_bitset->word_count = word_count;
  if (word_count == 0) {
    out_bitset->words = NULL;
    return iree_ok_status();
  }
  return iree_arena_allocate_array(arena, word_count,
                                   sizeof(*out_bitset->words),
                                   (void**)&out_bitset->words);
}

static void loom_liveness_bitset_clear_all(loom_liveness_bitset_t bitset) {
  if (bitset.word_count == 0) return;
  memset(bitset.words, 0, bitset.word_count * sizeof(*bitset.words));
}

static void loom_liveness_bitset_copy(loom_liveness_bitset_t target,
                                      loom_liveness_bitset_t source) {
  IREE_ASSERT(target.word_count == source.word_count);
  if (target.word_count == 0) return;
  memcpy(target.words, source.words, target.word_count * sizeof(*target.words));
}

static bool loom_liveness_bitset_equals(loom_liveness_bitset_t lhs,
                                        loom_liveness_bitset_t rhs) {
  IREE_ASSERT(lhs.word_count == rhs.word_count);
  for (iree_host_size_t i = 0; i < lhs.word_count; ++i) {
    if (lhs.words[i] != rhs.words[i]) return false;
  }
  return true;
}

static bool loom_liveness_bitset_set(loom_liveness_bitset_t bitset,
                                     loom_value_ordinal_t value_ordinal) {
  iree_host_size_t word_index = value_ordinal / 64u;
  IREE_ASSERT(word_index < bitset.word_count);
  uint64_t mask = UINT64_C(1) << (value_ordinal % 64u);
  uint64_t old_word = bitset.words[word_index];
  bitset.words[word_index] = old_word | mask;
  return old_word != bitset.words[word_index];
}

static bool loom_liveness_bitset_reset(loom_liveness_bitset_t bitset,
                                       loom_value_ordinal_t value_ordinal) {
  iree_host_size_t word_index = value_ordinal / 64u;
  IREE_ASSERT(word_index < bitset.word_count);
  uint64_t mask = UINT64_C(1) << (value_ordinal % 64u);
  uint64_t old_word = bitset.words[word_index];
  bitset.words[word_index] = old_word & ~mask;
  return old_word != bitset.words[word_index];
}

static bool loom_liveness_bitset_test(loom_liveness_bitset_t bitset,
                                      loom_value_ordinal_t value_ordinal) {
  iree_host_size_t word_index = value_ordinal / 64u;
  IREE_ASSERT(word_index < bitset.word_count);
  uint64_t mask = UINT64_C(1) << (value_ordinal % 64u);
  return (bitset.words[word_index] & mask) != 0;
}

static bool loom_liveness_bitset_union(loom_liveness_bitset_t target,
                                       loom_liveness_bitset_t source) {
  IREE_ASSERT(target.word_count == source.word_count);
  bool changed = false;
  for (iree_host_size_t i = 0; i < target.word_count; ++i) {
    uint64_t old_word = target.words[i];
    target.words[i] = old_word | source.words[i];
    changed |= old_word != target.words[i];
  }
  return changed;
}

static void loom_liveness_bitset_union_minus(loom_liveness_bitset_t target,
                                             loom_liveness_bitset_t lhs,
                                             loom_liveness_bitset_t rhs) {
  IREE_ASSERT(target.word_count == lhs.word_count);
  IREE_ASSERT(target.word_count == rhs.word_count);
  for (iree_host_size_t i = 0; i < target.word_count; ++i) {
    target.words[i] = lhs.words[i] & ~rhs.words[i];
  }
}

static iree_host_size_t loom_liveness_bitset_count(
    loom_liveness_bitset_t bitset) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < bitset.word_count; ++i) {
    uint64_t bits = bitset.words[i];
    while (bits != 0) {
      bits &= bits - 1u;
      ++count;
    }
  }
  return count;
}

static iree_status_t loom_liveness_bitset_values(
    loom_liveness_build_state_t* state, loom_liveness_bitset_t bitset,
    const loom_value_id_t** out_values, iree_host_size_t* out_count) {
  iree_host_size_t count = loom_liveness_bitset_count(bitset);
  *out_count = count;
  if (count == 0) {
    *out_values = NULL;
    return iree_ok_status();
  }
  loom_value_id_t* values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, count, sizeof(*values), (void**)&values));
  iree_host_size_t value_index = 0;
  for (iree_host_size_t word_index = 0; word_index < bitset.word_count;
       ++word_index) {
    uint64_t bits = bitset.words[word_index];
    while (bits != 0) {
      uint32_t bit_index = iree_math_count_trailing_zeros_u64(bits);
      const loom_value_ordinal_t value_ordinal =
          (loom_value_ordinal_t)(word_index * 64u + bit_index);
      IREE_ASSERT_LT(value_ordinal, state->value_count);
      values[value_index++] = state->value_ids[value_ordinal];
      bits &= bits - 1u;
    }
  }
  *out_values = values;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Value classification and intervals
//===----------------------------------------------------------------------===//

static loom_liveness_value_class_t loom_liveness_classify_value(
    const loom_module_t* module, loom_value_id_t value_id) {
  loom_type_t type = loom_module_value_type(module, value_id);
  loom_liveness_value_class_t value_class = {
      .type_kind = loom_type_kind(type),
      .element_type = loom_type_element_type(type),
      .register_class_id = LOOM_STRING_ID_INVALID,
  };
  if (loom_type_is_register(type)) {
    value_class.register_class_id = loom_type_register_class_id(type);
  }
  return value_class;
}

static uint32_t loom_liveness_value_unit_count(const loom_module_t* module,
                                               loom_value_id_t value_id) {
  loom_type_t type = loom_module_value_type(module, value_id);
  if (loom_type_is_register(type)) return loom_type_register_unit_count(type);
  return 1;
}

bool loom_liveness_value_class_equal(loom_liveness_value_class_t lhs,
                                     loom_liveness_value_class_t rhs) {
  return lhs.type_kind == rhs.type_kind &&
         lhs.element_type == rhs.element_type &&
         lhs.register_class_id == rhs.register_class_id;
}

static iree_status_t loom_liveness_value_ordinal(
    loom_liveness_build_state_t* state, loom_value_id_t value_id,
    loom_value_ordinal_t* out_ordinal) {
  IREE_ASSERT_ARGUMENT(out_ordinal);
  if (value_id >= state->module->values.count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "liveness saw out-of-range value id %u",
                            (unsigned)value_id);
  }
  const loom_value_ordinal_t ordinal =
      loom_module_value_ordinal_scratch_lookup(state->module, value_id);
  if (ordinal == LOOM_VALUE_ORDINAL_INVALID || ordinal >= state->value_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "liveness saw value id %u outside the analyzed "
                            "region-local value set",
                            (unsigned)value_id);
  }
  *out_ordinal = ordinal;
  return iree_ok_status();
}

static iree_status_t loom_liveness_ensure_interval(
    loom_liveness_build_state_t* state, loom_value_id_t value_id,
    loom_liveness_mutable_interval_t** out_interval) {
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_liveness_value_ordinal(state, value_id, &value_ordinal));
  uint32_t interval_index = state->value_interval_indices[value_ordinal];
  if (interval_index == UINT32_MAX) {
    interval_index = value_ordinal;
    state->value_interval_indices[value_ordinal] = interval_index;
    loom_liveness_mutable_interval_t* interval_state =
        &state->interval_states[interval_index];
    *interval_state = (loom_liveness_mutable_interval_t){
        .interval =
            {
                .value_id = value_id,
                .start_point = 0,
                .end_point = 0,
                .value_class =
                    loom_liveness_classify_value(state->module, value_id),
                .unit_count =
                    loom_liveness_value_unit_count(state->module, value_id),
            },
        .has_bounds = false,
    };
  }
  *out_interval = &state->interval_states[interval_index];
  return iree_ok_status();
}

static iree_status_t loom_liveness_note_definition(
    loom_liveness_build_state_t* state, loom_value_id_t value_id,
    uint32_t point) {
  loom_liveness_mutable_interval_t* interval_state = NULL;
  IREE_RETURN_IF_ERROR(
      loom_liveness_ensure_interval(state, value_id, &interval_state));
  if (!interval_state->has_bounds) {
    interval_state->interval.start_point = point;
    interval_state->interval.end_point = point;
    interval_state->has_bounds = true;
  } else if (point < interval_state->interval.start_point) {
    interval_state->interval.start_point = point;
  }
  return iree_ok_status();
}

static iree_status_t loom_liveness_note_live_point(
    loom_liveness_build_state_t* state, loom_value_id_t value_id,
    uint32_t point) {
  loom_liveness_mutable_interval_t* interval_state = NULL;
  IREE_RETURN_IF_ERROR(
      loom_liveness_ensure_interval(state, value_id, &interval_state));
  if (!interval_state->has_bounds) {
    interval_state->interval.start_point = point;
    interval_state->interval.end_point = point + 1u;
    interval_state->has_bounds = true;
    return iree_ok_status();
  }
  if (point < interval_state->interval.start_point) {
    interval_state->interval.start_point = point;
  }
  if (point + 1u > interval_state->interval.end_point) {
    interval_state->interval.end_point = point + 1u;
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Use collection
//===----------------------------------------------------------------------===//

typedef struct loom_liveness_type_ref_callback_state_t {
  loom_liveness_value_callback_t visitor;
} loom_liveness_type_ref_callback_state_t;

static iree_status_t loom_liveness_type_ref_callback(loom_value_id_t value_id,
                                                     void* user_data) {
  loom_liveness_type_ref_callback_state_t* state =
      (loom_liveness_type_ref_callback_state_t*)user_data;
  return state->visitor.fn(state->visitor.user_data, value_id);
}

static iree_status_t loom_liveness_for_each_type_ref(
    loom_type_t type, loom_liveness_value_callback_t visitor) {
  loom_liveness_type_ref_callback_state_t state = {
      .visitor = visitor,
  };
  return loom_type_walk_value_refs(type, loom_liveness_type_ref_callback,
                                   &state);
}

static bool loom_liveness_op_defines_value(const loom_op_t* op,
                                           loom_value_id_t value_id) {
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == value_id) return true;
  }
  return false;
}

static bool loom_liveness_region_is_nested_in_op(const loom_op_t* owner_op,
                                                 const loom_region_t* region);

static bool loom_liveness_block_is_nested_in_op(const loom_op_t* owner_op,
                                                const loom_block_t* block) {
  return block &&
         loom_liveness_region_is_nested_in_op(owner_op, block->parent_region);
}

static bool loom_liveness_value_is_defined_inside_op(
    const loom_op_t* owner_op, const loom_module_t* module,
    loom_value_id_t value_id) {
  if (value_id >= module->values.count) return false;
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return loom_liveness_block_is_nested_in_op(owner_op,
                                               loom_value_def_block(value));
  }
  const loom_op_t* def_op = loom_value_def_op(value);
  while (def_op) {
    if (def_op == owner_op) return true;
    def_op = def_op->parent_op;
  }
  return false;
}

static bool loom_liveness_region_is_nested_in_op(const loom_op_t* owner_op,
                                                 const loom_region_t* region) {
  if (!owner_op || !region) return false;
  loom_region_t* const* regions = loom_op_regions(owner_op);
  for (uint8_t i = 0; i < owner_op->region_count; ++i) {
    if (regions[i] == region) return true;
    const loom_block_t* block = NULL;
    loom_region_for_each_block(regions[i], block) {
      const loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) {
        if (loom_liveness_region_is_nested_in_op(op, region)) return true;
      }
    }
  }
  return false;
}

typedef struct loom_liveness_external_use_state_t {
  const loom_module_t* module;
  const loom_op_t* owner_op;
  loom_liveness_value_callback_t visitor;
} loom_liveness_external_use_state_t;

static iree_status_t loom_liveness_external_value_callback(
    void* user_data, loom_value_id_t value_id) {
  loom_liveness_external_use_state_t* state =
      (loom_liveness_external_use_state_t*)user_data;
  if (loom_liveness_value_is_defined_inside_op(state->owner_op, state->module,
                                               value_id)) {
    return iree_ok_status();
  }
  return state->visitor.fn(state->visitor.user_data, value_id);
}

static iree_status_t loom_liveness_for_each_region_external_use(
    loom_liveness_external_use_state_t* state, const loom_region_t* region) {
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    for (uint16_t i = 0; i < block->arg_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_liveness_for_each_type_ref(
          loom_block_arg_type(state->module, block, i),
          loom_liveness_value_callback_make(
              loom_liveness_external_value_callback, state)));
    }
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      const loom_value_id_t* operands = loom_op_const_operands(op);
      for (uint16_t i = 0; i < op->operand_count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_liveness_external_value_callback(state, operands[i]));
        IREE_RETURN_IF_ERROR(loom_liveness_for_each_type_ref(
            loom_module_value_type(state->module, operands[i]),
            loom_liveness_value_callback_make(
                loom_liveness_external_value_callback, state)));
      }
      const loom_value_id_t* results = loom_op_const_results(op);
      for (uint16_t i = 0; i < op->result_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_liveness_for_each_type_ref(
            loom_module_value_type(state->module, results[i]),
            loom_liveness_value_callback_make(
                loom_liveness_external_value_callback, state)));
      }
      loom_region_t* const* regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_liveness_for_each_region_external_use(state, regions[i]));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_liveness_for_each_nested_external_use(
    const loom_module_t* module, const loom_op_t* owner_op,
    loom_liveness_value_callback_t visitor) {
  loom_liveness_external_use_state_t state = {
      .module = module,
      .owner_op = owner_op,
      .visitor = visitor,
  };
  loom_region_t* const* regions = loom_op_regions(owner_op);
  for (uint8_t i = 0; i < owner_op->region_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_liveness_for_each_region_external_use(&state, regions[i]));
  }
  return iree_ok_status();
}

typedef struct loom_liveness_result_type_ref_state_t {
  const loom_op_t* op;
  loom_liveness_value_callback_t visitor;
} loom_liveness_result_type_ref_state_t;

static iree_status_t loom_liveness_result_type_ref_callback(
    loom_value_id_t value_id, void* user_data) {
  loom_liveness_result_type_ref_state_t* state =
      (loom_liveness_result_type_ref_state_t*)user_data;
  if (loom_liveness_op_defines_value(state->op, value_id)) {
    return iree_ok_status();
  }
  return state->visitor.fn(state->visitor.user_data, value_id);
}

static iree_status_t loom_liveness_for_each_op_use(
    const loom_module_t* module, const loom_op_t* op,
    loom_liveness_value_callback_t visitor) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(visitor.fn(visitor.user_data, operands[i]));
    IREE_RETURN_IF_ERROR(loom_liveness_for_each_type_ref(
        loom_module_value_type(module, operands[i]), visitor));
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_type_t result_type = loom_module_value_type(module, results[i]);
    loom_liveness_result_type_ref_state_t state = {
        .op = op,
        .visitor = visitor,
    };
    IREE_RETURN_IF_ERROR(loom_type_walk_value_refs(
        result_type, loom_liveness_result_type_ref_callback, &state));
  }
  return loom_liveness_for_each_nested_external_use(module, op, visitor);
}

static iree_status_t loom_liveness_append_value_id(
    loom_liveness_build_state_t* state, loom_value_id_t value_id) {
  if (state->value_count >= LOOM_VALUE_ORDINAL_INVALID) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "liveness local value count exceeds uint32_t");
  }
  if (state->value_count >= state->value_id_capacity) {
    iree_host_size_t old_capacity = state->value_id_capacity;
    iree_host_size_t minimum_capacity =
        old_capacity == 0 ? 64 : old_capacity * 2;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->arena, old_capacity, minimum_capacity, sizeof(*state->value_ids),
        &state->value_id_capacity, (void**)&state->value_ids));
  }
  const loom_value_ordinal_t value_ordinal =
      (loom_value_ordinal_t)state->value_count;
  state->value_ids[state->value_count++] = value_id;
  loom_module_value_ordinal_scratch_set(state->module, value_id, value_ordinal);
  return iree_ok_status();
}

static iree_status_t loom_liveness_register_value(
    loom_liveness_build_state_t* state, loom_value_id_t value_id) {
  if (value_id >= state->module->values.count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "liveness saw out-of-range value id %u",
                            (unsigned)value_id);
  }
  if (loom_module_value_ordinal_scratch_lookup(state->module, value_id) !=
      LOOM_VALUE_ORDINAL_INVALID) {
    return iree_ok_status();
  }
  return loom_liveness_append_value_id(state, value_id);
}

static iree_status_t loom_liveness_register_value_callback(
    void* user_data, loom_value_id_t value_id) {
  return loom_liveness_register_value((loom_liveness_build_state_t*)user_data,
                                      value_id);
}

static iree_status_t loom_liveness_register_region_values(
    loom_liveness_build_state_t* state) {
  loom_liveness_value_callback_t visitor = loom_liveness_value_callback_make(
      loom_liveness_register_value_callback, state);
  const loom_block_t* block = NULL;
  loom_region_for_each_block(state->region, block) {
    for (uint16_t i = 0; i < block->arg_count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_liveness_register_value(state, loom_block_arg_id(block, i)));
      IREE_RETURN_IF_ERROR(loom_liveness_for_each_type_ref(
          loom_block_arg_type(state->module, block, i), visitor));
    }
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      const loom_value_id_t* results = loom_op_const_results(op);
      for (uint16_t i = 0; i < op->result_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_liveness_register_value(state, results[i]));
      }
      IREE_RETURN_IF_ERROR(
          loom_liveness_for_each_op_use(state->module, op, visitor));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_liveness_build_state_initialize_values(
    loom_liveness_build_state_t* state) {
  loom_module_value_ordinal_scratch_acquire(state->module);
  state->value_ordinal_scratch_acquired = true;
  return loom_liveness_register_region_values(state);
}

static void loom_liveness_build_state_deinitialize(
    loom_liveness_build_state_t* state) {
  if (!state->value_ordinal_scratch_acquired) return;
  for (iree_host_size_t i = 0; i < state->value_count; ++i) {
    loom_module_value_ordinal_scratch_clear(state->module, state->value_ids[i]);
  }
  loom_module_value_ordinal_scratch_release(state->module);
  state->value_ordinal_scratch_acquired = false;
}

//===----------------------------------------------------------------------===//
// Local use/def construction
//===----------------------------------------------------------------------===//

typedef struct loom_liveness_use_def_state_t {
  loom_liveness_build_state_t* build_state;
  loom_liveness_block_state_t* block_state;
} loom_liveness_use_def_state_t;

static iree_status_t loom_liveness_add_block_use(void* user_data,
                                                 loom_value_id_t value_id) {
  loom_liveness_use_def_state_t* state =
      (loom_liveness_use_def_state_t*)user_data;
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  IREE_RETURN_IF_ERROR(loom_liveness_value_ordinal(state->build_state, value_id,
                                                   &value_ordinal));
  if (!loom_liveness_bitset_test(state->block_state->def, value_ordinal)) {
    loom_liveness_bitset_set(state->block_state->use, value_ordinal);
  }
  return iree_ok_status();
}

static iree_status_t loom_liveness_collect_block_use_def(
    loom_liveness_build_state_t* state, const loom_block_t* block,
    loom_liveness_block_state_t* block_state) {
  loom_liveness_use_def_state_t use_def_state = {
      .build_state = state,
      .block_state = block_state,
  };
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
    IREE_RETURN_IF_ERROR(loom_liveness_value_ordinal(
        state, loom_block_arg_id(block, i), &value_ordinal));
    loom_liveness_bitset_set(block_state->def, value_ordinal);
  }
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_liveness_for_each_type_ref(
        loom_block_arg_type(state->module, block, i),
        loom_liveness_value_callback_make(loom_liveness_add_block_use,
                                          &use_def_state)));
  }
  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    const loom_value_id_t* results = loom_op_const_results(op);
    for (uint16_t i = 0; i < op->result_count; ++i) {
      loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_liveness_value_ordinal(state, results[i], &value_ordinal));
      loom_liveness_bitset_set(block_state->def, value_ordinal);
    }
    IREE_RETURN_IF_ERROR(loom_liveness_for_each_op_use(
        state->module, op,
        loom_liveness_value_callback_make(loom_liveness_add_block_use,
                                          &use_def_state)));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Dataflow
//===----------------------------------------------------------------------===//

static iree_status_t loom_liveness_allocate_block_states(
    loom_liveness_build_state_t* state, iree_host_size_t block_count) {
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->arena, block_count,
                                                 sizeof(*state->block_states),
                                                 (void**)&state->block_states));
  for (iree_host_size_t i = 0; i < block_count; ++i) {
    loom_liveness_block_state_t* block_state = &state->block_states[i];
    IREE_RETURN_IF_ERROR(loom_liveness_bitset_allocate(
        state->arena, state->word_count, &block_state->use));
    IREE_RETURN_IF_ERROR(loom_liveness_bitset_allocate(
        state->arena, state->word_count, &block_state->def));
    IREE_RETURN_IF_ERROR(loom_liveness_bitset_allocate(
        state->arena, state->word_count, &block_state->live_in));
    IREE_RETURN_IF_ERROR(loom_liveness_bitset_allocate(
        state->arena, state->word_count, &block_state->live_out));
    loom_liveness_bitset_clear_all(block_state->use);
    loom_liveness_bitset_clear_all(block_state->def);
    loom_liveness_bitset_clear_all(block_state->live_in);
    loom_liveness_bitset_clear_all(block_state->live_out);
  }
  return iree_ok_status();
}

static iree_status_t loom_liveness_run_dataflow(
    loom_liveness_build_state_t* state, const loom_cfg_graph_t* graph) {
  loom_liveness_bitset_t next_live_in = {0};
  loom_liveness_bitset_t next_live_out = {0};
  IREE_RETURN_IF_ERROR(loom_liveness_bitset_allocate(
      state->arena, state->word_count, &next_live_in));
  IREE_RETURN_IF_ERROR(loom_liveness_bitset_allocate(
      state->arena, state->word_count, &next_live_out));
  bool changed = true;
  while (changed) {
    changed = false;
    for (iree_host_size_t reverse_index = graph->block_count; reverse_index > 0;
         --reverse_index) {
      uint16_t block_index = (uint16_t)(reverse_index - 1u);
      loom_liveness_block_state_t* block_state =
          &state->block_states[block_index];

      loom_liveness_bitset_clear_all(next_live_out);
      loom_cfg_block_index_span_t successors =
          loom_cfg_graph_successors(graph, block_index);
      for (iree_host_size_t i = 0; i < successors.count; ++i) {
        loom_liveness_bitset_union(
            next_live_out, state->block_states[successors.values[i]].live_in);
      }
      loom_liveness_bitset_union_minus(next_live_in, next_live_out,
                                       block_state->def);
      loom_liveness_bitset_union(next_live_in, block_state->use);

      bool block_changed =
          !loom_liveness_bitset_equals(next_live_in, block_state->live_in) ||
          !loom_liveness_bitset_equals(next_live_out, block_state->live_out);
      if (block_changed) {
        loom_liveness_bitset_copy(block_state->live_in, next_live_in);
        loom_liveness_bitset_copy(block_state->live_out, next_live_out);
        changed = true;
      }
    }
  }
  return iree_ok_status();
}

static void loom_liveness_initialize_local_liveness(
    loom_liveness_build_state_t* state, iree_host_size_t block_count) {
  for (iree_host_size_t i = 0; i < block_count; ++i) {
    loom_liveness_block_state_t* block_state = &state->block_states[i];
    loom_liveness_bitset_copy(block_state->live_in, block_state->use);
  }
}

//===----------------------------------------------------------------------===//
// Program points and intervals
//===----------------------------------------------------------------------===//

static uint32_t loom_liveness_block_point_span(const loom_block_t* block) {
  return block->op_count;
}

typedef struct loom_liveness_point_use_state_t {
  loom_liveness_build_state_t* build_state;
  uint32_t point;
} loom_liveness_point_use_state_t;

static iree_status_t loom_liveness_note_use_at_point(void* user_data,
                                                     loom_value_id_t value_id) {
  loom_liveness_point_use_state_t* state =
      (loom_liveness_point_use_state_t*)user_data;
  return loom_liveness_note_live_point(state->build_state, value_id,
                                       state->point);
}

static iree_status_t loom_liveness_note_bitset_live_point(
    loom_liveness_build_state_t* state, loom_liveness_bitset_t bitset,
    uint32_t point) {
  for (iree_host_size_t word_index = 0; word_index < bitset.word_count;
       ++word_index) {
    uint64_t bits = bitset.words[word_index];
    while (bits != 0) {
      uint32_t bit_index = iree_math_count_trailing_zeros_u64(bits);
      const loom_value_ordinal_t value_ordinal =
          (loom_value_ordinal_t)(word_index * 64u + bit_index);
      IREE_ASSERT_LT(value_ordinal, state->value_count);
      IREE_RETURN_IF_ERROR(loom_liveness_note_live_point(
          state, state->value_ids[value_ordinal], point));
      bits &= bits - 1u;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_liveness_finalize_intervals(
    loom_liveness_build_state_t* state,
    const loom_liveness_block_info_t* block_infos) {
  for (iree_host_size_t block_index = 0;
       block_index < state->region->block_count; ++block_index) {
    const loom_block_t* block =
        loom_region_const_block(state->region, (uint16_t)block_index);
    const loom_liveness_block_info_t* block_info = &block_infos[block_index];
    const loom_liveness_block_state_t* block_state =
        &state->block_states[block_index];

    for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
      loom_value_id_t arg_id = loom_block_arg_id(block, arg_index);
      IREE_RETURN_IF_ERROR(loom_liveness_note_definition(
          state, arg_id, block_info->start_point));
      loom_liveness_point_use_state_t type_use_state = {
          .build_state = state,
          .point = block_info->start_point,
      };
      IREE_RETURN_IF_ERROR(loom_liveness_for_each_type_ref(
          loom_block_arg_type(state->module, block, arg_index),
          loom_liveness_value_callback_make(loom_liveness_note_use_at_point,
                                            &type_use_state)));
    }

    IREE_RETURN_IF_ERROR(loom_liveness_note_bitset_live_point(
        state, block_state->live_in, block_info->start_point));
    IREE_RETURN_IF_ERROR(loom_liveness_note_bitset_live_point(
        state, block_state->live_out, block_info->end_point));

    uint32_t point = block_info->start_point;
    if (!loom_liveness_order_is_empty(state->order)) {
      const loom_liveness_block_order_t* block_order =
          &state->order.blocks[block_index];
      for (iree_host_size_t ordered_index = 0;
           ordered_index < block_order->op_count; ++ordered_index) {
        const loom_op_t* op = block_order->ops[ordered_index];
        loom_liveness_point_use_state_t use_state = {
            .build_state = state,
            .point = point,
        };
        IREE_RETURN_IF_ERROR(loom_liveness_for_each_op_use(
            state->module, op,
            loom_liveness_value_callback_make(loom_liveness_note_use_at_point,
                                              &use_state)));
        const loom_value_id_t* results = loom_op_const_results(op);
        for (uint16_t result_index = 0; result_index < op->result_count;
             ++result_index) {
          IREE_RETURN_IF_ERROR(loom_liveness_note_definition(
              state, results[result_index], point + 1u));
        }
        ++point;
      }
      continue;
    }
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      loom_liveness_point_use_state_t use_state = {
          .build_state = state,
          .point = point,
      };
      IREE_RETURN_IF_ERROR(loom_liveness_for_each_op_use(
          state->module, op,
          loom_liveness_value_callback_make(loom_liveness_note_use_at_point,
                                            &use_state)));
      const loom_value_id_t* results = loom_op_const_results(op);
      for (uint16_t result_index = 0; result_index < op->result_count;
           ++result_index) {
        IREE_RETURN_IF_ERROR(loom_liveness_note_definition(
            state, results[result_index], point + 1u));
      }
      ++point;
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Pressure summaries
//===----------------------------------------------------------------------===//

static iree_status_t loom_liveness_pressure_find_or_add(
    loom_liveness_build_state_t* state, loom_liveness_value_class_t value_class,
    iree_host_size_t* out_index) {
  loom_liveness_pressure_state_t* pressure = &state->pressure_state;
  for (iree_host_size_t i = 0; i < pressure->count; ++i) {
    if (loom_liveness_value_class_equal(pressure->summaries[i].value_class,
                                        value_class)) {
      *out_index = i;
      return iree_ok_status();
    }
  }
  if (pressure->count >= pressure->capacity) {
    iree_host_size_t old_capacity = pressure->capacity;
    iree_host_size_t new_capacity = old_capacity == 0 ? 8 : old_capacity * 2;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->arena, old_capacity, new_capacity, sizeof(*pressure->summaries),
        &new_capacity, (void**)&pressure->summaries));
    memset(pressure->summaries + old_capacity, 0,
           (new_capacity - old_capacity) * sizeof(*pressure->summaries));
    pressure->capacity = new_capacity;
  }
  *out_index = pressure->count++;
  pressure->summaries[*out_index].value_class = value_class;
  return iree_ok_status();
}

typedef struct loom_liveness_pressure_bucket_t {
  loom_liveness_value_class_t value_class;
  uint32_t live_units;
  uint32_t live_values;
} loom_liveness_pressure_bucket_t;

typedef struct loom_liveness_pressure_snapshot_t {
  loom_liveness_build_state_t* build_state;
  loom_liveness_pressure_bucket_t* buckets;
  iree_host_size_t count;
  iree_host_size_t capacity;
} loom_liveness_pressure_snapshot_t;

static iree_status_t loom_liveness_pressure_snapshot_bucket(
    loom_liveness_pressure_snapshot_t* snapshot,
    loom_liveness_value_class_t value_class,
    loom_liveness_pressure_bucket_t** out_bucket) {
  for (iree_host_size_t i = 0; i < snapshot->count; ++i) {
    if (loom_liveness_value_class_equal(snapshot->buckets[i].value_class,
                                        value_class)) {
      *out_bucket = &snapshot->buckets[i];
      return iree_ok_status();
    }
  }
  if (snapshot->count >= snapshot->capacity) {
    iree_host_size_t old_capacity = snapshot->capacity;
    iree_host_size_t new_capacity = old_capacity == 0 ? 8 : old_capacity * 2;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        snapshot->build_state->arena, old_capacity, new_capacity,
        sizeof(*snapshot->buckets), &new_capacity, (void**)&snapshot->buckets));
    memset(snapshot->buckets + old_capacity, 0,
           (new_capacity - old_capacity) * sizeof(*snapshot->buckets));
    snapshot->capacity = new_capacity;
  }
  *out_bucket = &snapshot->buckets[snapshot->count++];
  **out_bucket = (loom_liveness_pressure_bucket_t){
      .value_class = value_class,
  };
  return iree_ok_status();
}

static iree_status_t loom_liveness_pressure_snapshot_add(
    void* user_data, loom_value_id_t value_id) {
  loom_liveness_pressure_snapshot_t* snapshot =
      (loom_liveness_pressure_snapshot_t*)user_data;
  loom_liveness_mutable_interval_t* interval_state = NULL;
  IREE_RETURN_IF_ERROR(loom_liveness_ensure_interval(
      snapshot->build_state, value_id, &interval_state));
  loom_liveness_pressure_bucket_t* bucket = NULL;
  IREE_RETURN_IF_ERROR(loom_liveness_pressure_snapshot_bucket(
      snapshot, interval_state->interval.value_class, &bucket));
  bucket->live_units += interval_state->interval.unit_count;
  bucket->live_values += 1;
  return iree_ok_status();
}

static iree_status_t loom_liveness_record_pressure_snapshot(
    loom_liveness_pressure_snapshot_t* snapshot,
    loom_liveness_bitset_t live_values, const loom_block_t* block,
    const loom_op_t* op, uint32_t point) {
  loom_liveness_build_state_t* state = snapshot->build_state;
  snapshot->count = 0;
  for (iree_host_size_t word_index = 0; word_index < live_values.word_count;
       ++word_index) {
    uint64_t bits = live_values.words[word_index];
    while (bits != 0) {
      uint32_t bit_index = iree_math_count_trailing_zeros_u64(bits);
      const loom_value_ordinal_t value_ordinal =
          (loom_value_ordinal_t)(word_index * 64u + bit_index);
      IREE_ASSERT_LT(value_ordinal, state->value_count);
      IREE_RETURN_IF_ERROR(loom_liveness_pressure_snapshot_add(
          snapshot, state->value_ids[value_ordinal]));
      bits &= bits - 1u;
    }
  }
  for (iree_host_size_t i = 0; i < snapshot->count; ++i) {
    iree_host_size_t summary_index = 0;
    IREE_RETURN_IF_ERROR(loom_liveness_pressure_find_or_add(
        state, snapshot->buckets[i].value_class, &summary_index));
    loom_liveness_pressure_summary_t* summary =
        &state->pressure_state.summaries[summary_index];
    if (snapshot->buckets[i].live_units > summary->peak_live_units ||
        (snapshot->buckets[i].live_units == summary->peak_live_units &&
         snapshot->buckets[i].live_values > summary->peak_live_values)) {
      summary->peak_live_units = snapshot->buckets[i].live_units;
      summary->peak_live_values = snapshot->buckets[i].live_values;
      summary->peak_block = block;
      summary->peak_op = op;
      summary->peak_point = point;
    }
  }
  return iree_ok_status();
}

typedef struct loom_liveness_add_to_bitset_state_t {
  loom_liveness_build_state_t* build_state;
  loom_liveness_bitset_t* bitset;
} loom_liveness_add_to_bitset_state_t;

static iree_status_t loom_liveness_add_use_to_bitset(void* user_data,
                                                     loom_value_id_t value_id) {
  loom_liveness_add_to_bitset_state_t* state =
      (loom_liveness_add_to_bitset_state_t*)user_data;
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  IREE_RETURN_IF_ERROR(loom_liveness_value_ordinal(state->build_state, value_id,
                                                   &value_ordinal));
  loom_liveness_bitset_set(*state->bitset, value_ordinal);
  return iree_ok_status();
}

static iree_status_t loom_liveness_compute_pressure(
    loom_liveness_build_state_t* state,
    const loom_liveness_block_info_t* block_infos) {
  loom_liveness_bitset_t live;
  IREE_RETURN_IF_ERROR(
      loom_liveness_bitset_allocate(state->arena, state->word_count, &live));
  loom_liveness_pressure_snapshot_t snapshot = {
      .build_state = state,
      .buckets = NULL,
      .count = 0,
      .capacity = 0,
  };
  for (iree_host_size_t block_index = 0;
       block_index < state->region->block_count; ++block_index) {
    const loom_block_t* block =
        loom_region_const_block(state->region, (uint16_t)block_index);
    const loom_liveness_block_info_t* block_info = &block_infos[block_index];
    const loom_liveness_block_state_t* block_state =
        &state->block_states[block_index];
    loom_liveness_bitset_copy(live, block_state->live_out);
    IREE_RETURN_IF_ERROR(loom_liveness_record_pressure_snapshot(
        &snapshot, live, block, NULL, block_info->end_point));

    uint32_t point = block_info->end_point;
    if (!loom_liveness_order_is_empty(state->order)) {
      const loom_liveness_block_order_t* block_order =
          &state->order.blocks[block_index];
      for (iree_host_size_t reverse_index = block_order->op_count;
           reverse_index > 0; --reverse_index) {
        const loom_op_t* op = block_order->ops[reverse_index - 1u];
        const loom_value_id_t* results = loom_op_const_results(op);
        for (uint16_t result_index = 0; result_index < op->result_count;
             ++result_index) {
          loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
          IREE_RETURN_IF_ERROR(loom_liveness_value_ordinal(
              state, results[result_index], &value_ordinal));
          loom_liveness_bitset_reset(live, value_ordinal);
        }
        loom_liveness_add_to_bitset_state_t add_state = {
            .build_state = state,
            .bitset = &live,
        };
        IREE_RETURN_IF_ERROR(loom_liveness_for_each_op_use(
            state->module, op,
            loom_liveness_value_callback_make(loom_liveness_add_use_to_bitset,
                                              &add_state)));
        --point;
        IREE_RETURN_IF_ERROR(loom_liveness_record_pressure_snapshot(
            &snapshot, live, block, op, point));
      }
      IREE_RETURN_IF_ERROR(loom_liveness_record_pressure_snapshot(
          &snapshot, live, block, NULL, block_info->start_point));
      continue;
    }
    const loom_op_t* op = block->last_op;
    while (op) {
      const loom_value_id_t* results = loom_op_const_results(op);
      for (uint16_t result_index = 0; result_index < op->result_count;
           ++result_index) {
        loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
        IREE_RETURN_IF_ERROR(loom_liveness_value_ordinal(
            state, results[result_index], &value_ordinal));
        loom_liveness_bitset_reset(live, value_ordinal);
      }
      loom_liveness_add_to_bitset_state_t add_state = {
          .build_state = state,
          .bitset = &live,
      };
      IREE_RETURN_IF_ERROR(loom_liveness_for_each_op_use(
          state->module, op,
          loom_liveness_value_callback_make(loom_liveness_add_use_to_bitset,
                                            &add_state)));
      --point;
      IREE_RETURN_IF_ERROR(loom_liveness_record_pressure_snapshot(
          &snapshot, live, block, op, point));
      op = op->prev_op;
    }
    IREE_RETURN_IF_ERROR(loom_liveness_record_pressure_snapshot(
        &snapshot, live, block, NULL, block_info->start_point));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Finalization
//===----------------------------------------------------------------------===//

static iree_status_t loom_liveness_finalize_block_infos(
    loom_liveness_build_state_t* state,
    loom_liveness_block_info_t** out_block_infos) {
  loom_liveness_block_info_t* block_infos = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->arena, state->region->block_count,
                                sizeof(*block_infos), (void**)&block_infos));
  uint32_t point = 0;
  for (uint16_t block_index = 0; block_index < state->region->block_count;
       ++block_index) {
    const loom_block_t* block =
        loom_region_const_block(state->region, block_index);
    loom_liveness_block_state_t* block_state =
        &state->block_states[block_index];
    loom_liveness_block_info_t* block_info = &block_infos[block_index];
    block_info->block = block;
    block_info->start_point = point;
    block_info->end_point = point + loom_liveness_block_point_span(block);
    IREE_RETURN_IF_ERROR(loom_liveness_bitset_values(
        state, block_state->live_in, &block_info->live_in_values,
        &block_info->live_in_count));
    IREE_RETURN_IF_ERROR(loom_liveness_bitset_values(
        state, block_state->live_out, &block_info->live_out_values,
        &block_info->live_out_count));
    point = block_info->end_point + 1u;
  }
  *out_block_infos = block_infos;
  return iree_ok_status();
}

static iree_status_t loom_liveness_finalize_interval_array(
    loom_liveness_build_state_t* state,
    loom_liveness_interval_t** out_intervals, iree_host_size_t* out_count) {
  iree_host_size_t count = 0;
  for (iree_host_size_t value_ordinal = 0; value_ordinal < state->value_count;
       ++value_ordinal) {
    if (state->value_interval_indices[value_ordinal] != UINT32_MAX) ++count;
  }
  loom_liveness_interval_t* intervals = NULL;
  if (count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, count, sizeof(*intervals), (void**)&intervals));
  }
  iree_host_size_t interval_index = 0;
  for (iree_host_size_t value_ordinal = 0; value_ordinal < state->value_count;
       ++value_ordinal) {
    if (state->value_interval_indices[value_ordinal] == UINT32_MAX) continue;
    state->value_interval_indices[value_ordinal] = (uint32_t)interval_index;
    intervals[interval_index++] =
        state->interval_states[value_ordinal].interval;
  }
  *out_intervals = intervals;
  *out_count = count;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

static iree_status_t loom_liveness_validate_order(const loom_region_t* region,
                                                  loom_liveness_order_t order) {
  if (loom_liveness_order_is_empty(order)) {
    return iree_ok_status();
  }
  if (order.block_count != region->block_count || order.blocks == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "liveness order must provide one entry per region block");
  }
  for (iree_host_size_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block =
        loom_region_const_block(region, (uint16_t)block_index);
    const loom_liveness_block_order_t* block_order = &order.blocks[block_index];
    if (block_order->block != block) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "liveness order block %zu does not match region block order",
          block_index);
    }
    if (block_order->op_count != block->op_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "liveness order block %zu has %zu op(s) for a block with %u op(s)",
          block_index, block_order->op_count, block->op_count);
    }
    if (block_order->op_count != 0 && block_order->ops == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "liveness order block %zu has no operation order",
                              block_index);
    }
    for (iree_host_size_t op_index = 0; op_index < block_order->op_count;
         ++op_index) {
      const loom_op_t* op = block_order->ops[op_index];
      if (op == NULL || op->parent_block != block) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "liveness order block %zu entry %zu is not owned by the block",
            block_index, op_index);
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_liveness_analyze_region(
    loom_module_t* module, const loom_region_t* region,
    iree_arena_allocator_t* arena, loom_liveness_analysis_t* out_analysis) {
  return loom_liveness_analyze_region_with_order(
      module, region, loom_liveness_order_empty(), arena, out_analysis);
}

iree_status_t loom_liveness_analyze_region_with_order(
    loom_module_t* module, const loom_region_t* region,
    loom_liveness_order_t order, iree_arena_allocator_t* arena,
    loom_liveness_analysis_t* out_analysis) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(region);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_analysis);
  memset(out_analysis, 0, sizeof(*out_analysis));

  loom_liveness_build_state_t state = {
      .module = module,
      .region = region,
      .order = order,
      .arena = arena,
  };

  iree_status_t status = loom_liveness_validate_order(region, order);
  if (iree_status_is_ok(status)) {
    status = loom_liveness_build_state_initialize_values(&state);
  }
  if (iree_status_is_ok(status)) {
    state.word_count = loom_liveness_word_count(state.value_count);
    status = loom_liveness_allocate_block_states(&state, region->block_count);
  }
  if (iree_status_is_ok(status) && state.value_count > 0) {
    status = iree_arena_allocate_array(arena, state.value_count,
                                       sizeof(*state.interval_states),
                                       (void**)&state.interval_states);
  }
  if (iree_status_is_ok(status) && state.value_count > 0) {
    memset(state.interval_states, 0,
           state.value_count * sizeof(*state.interval_states));
    status = iree_arena_allocate_array(arena, state.value_count,
                                       sizeof(*state.value_interval_indices),
                                       (void**)&state.value_interval_indices);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < state.value_count; ++i) {
      state.value_interval_indices[i] = UINT32_MAX;
    }
  }

  for (uint16_t block_index = 0;
       iree_status_is_ok(status) && block_index < region->block_count;
       ++block_index) {
    status = loom_liveness_collect_block_use_def(
        &state, loom_region_const_block(region, block_index),
        &state.block_states[block_index]);
  }

  bool is_cfg = iree_any_bit_set(region->flags, LOOM_REGION_INSTANCE_FLAG_CFG);
  loom_cfg_graph_t graph = {0};
  if (iree_status_is_ok(status) && is_cfg) {
    status = loom_cfg_graph_build(region, arena, &graph);
  }
  if (iree_status_is_ok(status) && is_cfg) {
    if (graph.malformed) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "CFG graph is malformed; run Loom verification "
                                "before liveness analysis");
    }
  }
  if (iree_status_is_ok(status) && is_cfg) {
    status = loom_liveness_run_dataflow(&state, &graph);
  } else if (iree_status_is_ok(status)) {
    loom_liveness_initialize_local_liveness(&state, region->block_count);
  }

  loom_liveness_block_info_t* block_infos = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_liveness_finalize_block_infos(&state, &block_infos);
  }
  if (iree_status_is_ok(status)) {
    status = loom_liveness_finalize_intervals(&state, block_infos);
  }
  if (iree_status_is_ok(status)) {
    status = loom_liveness_compute_pressure(&state, block_infos);
  }

  loom_liveness_interval_t* intervals = NULL;
  iree_host_size_t interval_count = 0;
  if (iree_status_is_ok(status)) {
    status = loom_liveness_finalize_interval_array(&state, &intervals,
                                                   &interval_count);
  }

  if (iree_status_is_ok(status)) {
    *out_analysis = (loom_liveness_analysis_t){
        .module = module,
        .region = region,
        .is_cfg = is_cfg,
        .blocks = block_infos,
        .block_count = region->block_count,
        .intervals = intervals,
        .interval_count = interval_count,
        .value_ids = state.value_ids,
        .value_count = state.value_count,
        .value_interval_indices = state.value_interval_indices,
        .pressure_summaries = state.pressure_state.summaries,
        .pressure_summary_count = state.pressure_state.count,
    };
  }

  loom_liveness_build_state_deinitialize(&state);
  return status;
}

const loom_liveness_interval_t* loom_liveness_interval_for_value(
    const loom_liveness_analysis_t* analysis, loom_value_id_t value_id) {
  if (!analysis) return NULL;
  for (iree_host_size_t i = 0; i < analysis->value_count; ++i) {
    if (analysis->value_ids[i] != value_id) continue;
    if (!analysis->value_interval_indices) return NULL;
    uint32_t interval_index = analysis->value_interval_indices[i];
    if (interval_index == UINT32_MAX ||
        interval_index >= analysis->interval_count) {
      return NULL;
    }
    return &analysis->intervals[interval_index];
  }
  return NULL;
}

const loom_liveness_block_info_t* loom_liveness_block_info_for_block(
    const loom_liveness_analysis_t* analysis, const loom_block_t* block) {
  if (!analysis) return NULL;
  if (!block) return NULL;
  for (iree_host_size_t i = 0; i < analysis->block_count; ++i) {
    if (analysis->blocks[i].block == block) return &analysis->blocks[i];
  }
  return NULL;
}

static const loom_liveness_pressure_summary_t*
loom_liveness_pressure_summary_for_class(
    const loom_liveness_analysis_t* analysis,
    loom_liveness_value_class_t value_class) {
  for (iree_host_size_t i = 0; i < analysis->pressure_summary_count; ++i) {
    const loom_liveness_pressure_summary_t* summary =
        &analysis->pressure_summaries[i];
    if (loom_liveness_value_class_equal(summary->value_class, value_class)) {
      return summary;
    }
  }
  return NULL;
}

static loom_liveness_pressure_budget_violation_flags_t
loom_liveness_pressure_budget_violation_bits(
    const loom_liveness_pressure_summary_t* summary,
    const loom_liveness_pressure_budget_t* budget) {
  loom_liveness_pressure_budget_violation_flags_t violation_bits = 0;
  if (budget->max_live_units != UINT32_MAX &&
      summary->peak_live_units > budget->max_live_units) {
    violation_bits |= LOOM_LIVENESS_PRESSURE_BUDGET_VIOLATION_LIVE_UNITS;
  }
  if (budget->max_live_values != UINT32_MAX &&
      summary->peak_live_values > budget->max_live_values) {
    violation_bits |= LOOM_LIVENESS_PRESSURE_BUDGET_VIOLATION_LIVE_VALUES;
  }
  return violation_bits;
}

iree_status_t loom_liveness_collect_pressure_budget_violations(
    const loom_liveness_analysis_t* analysis,
    const loom_liveness_pressure_budget_t* budgets,
    iree_host_size_t budget_count, iree_arena_allocator_t* arena,
    const loom_liveness_pressure_budget_violation_t** out_violations,
    iree_host_size_t* out_violation_count) {
  IREE_ASSERT_ARGUMENT(analysis);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_violations);
  IREE_ASSERT_ARGUMENT(out_violation_count);
  *out_violations = NULL;
  *out_violation_count = 0;
  if (budget_count == 0) return iree_ok_status();
  if (!budgets) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pressure budgets are required when budget count "
                            "is non-zero");
  }

  iree_host_size_t violation_count = 0;
  for (iree_host_size_t budget_index = 0; budget_index < budget_count;
       ++budget_index) {
    const loom_liveness_pressure_summary_t* summary =
        loom_liveness_pressure_summary_for_class(
            analysis, budgets[budget_index].value_class);
    if (!summary) continue;
    if (loom_liveness_pressure_budget_violation_bits(
            summary, &budgets[budget_index]) != 0) {
      ++violation_count;
    }
  }
  if (violation_count == 0) return iree_ok_status();

  loom_liveness_pressure_budget_violation_t* violations = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, violation_count, sizeof(*violations), (void**)&violations));
  iree_host_size_t violation_index = 0;
  for (iree_host_size_t budget_index = 0; budget_index < budget_count;
       ++budget_index) {
    const loom_liveness_pressure_summary_t* summary =
        loom_liveness_pressure_summary_for_class(
            analysis, budgets[budget_index].value_class);
    if (!summary) continue;
    loom_liveness_pressure_budget_violation_flags_t violation_bits =
        loom_liveness_pressure_budget_violation_bits(summary,
                                                     &budgets[budget_index]);
    if (violation_bits == 0) continue;
    violations[violation_index++] = (loom_liveness_pressure_budget_violation_t){
        .budget_index = budget_index,
        .budget = budgets[budget_index],
        .summary = summary,
        .violation_bits = violation_bits,
    };
  }
  *out_violations = violations;
  *out_violation_count = violation_count;
  return iree_ok_status();
}
