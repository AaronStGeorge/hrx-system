// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/ownership_lifetime.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/analysis/ownership.h"
#include "loom/analysis/scc.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/util/cfg_graph.h"
#include "loom/util/walk.h"

enum {
  LOOM_OWNERSHIP_LIFETIME_ARG_INDEX_NONE = UINT16_MAX,
};

typedef enum loom_ownership_lifetime_value_state_e {
  LOOM_OWNERSHIP_LIFETIME_VALUE_UNKNOWN = 0,
  LOOM_OWNERSHIP_LIFETIME_VALUE_BORROWED = 1,
  LOOM_OWNERSHIP_LIFETIME_VALUE_OWNED = 2,
} loom_ownership_lifetime_value_state_t;

typedef struct loom_ownership_lifetime_bitset_t {
  // Dense 64-bit words keyed by function-local value ordinal.
  uint64_t* words;
  // Number of initialized entries in |words|.
  iree_host_size_t word_count;
} loom_ownership_lifetime_bitset_t;

typedef struct loom_ownership_lifetime_state_bits_t {
  // Values that may be observed but do not carry a release obligation.
  loom_ownership_lifetime_bitset_t borrowed;
  // Values that carry an owned-resource release obligation.
  loom_ownership_lifetime_bitset_t owned;
} loom_ownership_lifetime_state_bits_t;

typedef struct loom_ownership_lifetime_block_state_t {
  // Ownership state at block entry after predecessor joins.
  loom_ownership_lifetime_state_bits_t in;
  // Ownership state after interpreting all operations in the block.
  loom_ownership_lifetime_state_bits_t out;
  // True once at least one predecessor initialized |in|.
  bool initialized;
  // True once this block has been transferred at least once.
  bool processed;
  // True while the block index is present in the work queue.
  bool queued;
} loom_ownership_lifetime_block_state_t;

typedef struct loom_ownership_lifetime_function_summary_t {
  // Function-like op this summary describes.
  loom_func_like_t function;
  // Body region analyzed for this summary, or NULL for conservative callees.
  loom_region_t* body;
  // Number of operands in the callable signature.
  uint16_t arg_count;
  // Number of results in the callable signature.
  uint16_t result_count;
  // Per-argument flag set when the body requires the corresponding operand to
  // arrive owned.
  bool* arg_consumed;
  // Per-result state inferred from returned values.
  loom_ownership_lifetime_value_state_t* result_states;
  // True when suppressed analysis saw a possible diagnostic on this summary.
  bool needs_diagnostic_rerun;
} loom_ownership_lifetime_function_summary_t;

typedef struct loom_ownership_lifetime_module_state_t {
  // Module being checked.
  loom_module_t* module;
  // Caller-owned analysis options.
  const loom_ownership_lifetime_options_t* options;
  // Result object receiving final diagnostic and traversal counters.
  loom_ownership_lifetime_result_t* result;
  // Dense summaries indexed by module symbol ID.
  loom_ownership_lifetime_function_summary_t* summaries;
  // Number of entries in |summaries|.
  iree_host_size_t summary_count;
} loom_ownership_lifetime_module_state_t;

typedef struct loom_ownership_lifetime_graph_node_t {
  // Summary owned by the module state for this bodyful function.
  loom_ownership_lifetime_function_summary_t* summary;
} loom_ownership_lifetime_graph_node_t;

typedef struct loom_ownership_lifetime_graph_t {
  // Module-level analysis state.
  loom_ownership_lifetime_module_state_t* module_state;
  // Reset before each graph successor walk.
  iree_arena_allocator_t* walk_arena;
  // Dense bodyful function nodes.
  loom_ownership_lifetime_graph_node_t* nodes;
  // Number of entries in |nodes|.
  iree_host_size_t node_count;
  // Symbol ID to graph node index map, or IREE_HOST_SIZE_MAX.
  iree_host_size_t* symbol_to_node;
  // Number of entries in |symbol_to_node|.
  iree_host_size_t symbol_to_node_count;
} loom_ownership_lifetime_graph_t;

typedef struct loom_ownership_lifetime_state_t {
  // Module whose function body is being checked.
  loom_module_t* module;
  // Module-level summary table and diagnostics sink.
  loom_ownership_lifetime_module_state_t* module_state;
  // Function-like symbol whose body is being checked.
  loom_func_like_t function;
  // Inferred summary for |function|.
  loom_ownership_lifetime_function_summary_t* summary;
  // Caller-owned analysis options.
  const loom_ownership_lifetime_options_t* options;
  // Result object receiving diagnostic and traversal counters.
  loom_ownership_lifetime_result_t* result;
  // Function body region being checked.
  const loom_region_t* body;
  // Caller-owned function-local value domain.
  const loom_local_value_domain_t* value_domain;
  // Value IDs indexed by region-local value ordinal.
  const loom_value_id_t* value_ids;
  // Number of initialized local value IDs.
  iree_host_size_t value_count;
  // Function argument index that a local value aliases, or NONE.
  uint16_t* value_origin_args;
  // Number of 64-bit words in local value bitsets.
  iree_host_size_t word_count;
  // Values produced by ownership-aware results or consumed from owned state.
  loom_ownership_lifetime_bitset_t ever_seen;
  // Mutable per-block ownership states in region block order.
  loom_ownership_lifetime_block_state_t* blocks;
  // Work queue of block indices whose entry state changed.
  uint16_t* queue;
  // Index of the next queued block to process.
  iree_host_size_t queue_read;
  // One-past-last queued block index.
  iree_host_size_t queue_write;
  // Allocated capacity of |queue|.
  iree_host_size_t queue_capacity;
  // True while computing monotone summaries without user diagnostics.
  bool inference;
  // True when user-facing diagnostics should be emitted.
  bool emit_diagnostics;
  // True when traversal counters should be recorded.
  bool count_statistics;
  // True once this run promoted an argument/result summary fact.
  bool summary_changed;
  // True once a consumed argument promotion requires the function to restart.
  bool needs_restart;
  // True once a user-facing ownership diagnostic has been emitted.
  bool failed;
} loom_ownership_lifetime_state_t;

//===----------------------------------------------------------------------===//
// Bitsets
//===----------------------------------------------------------------------===//

static iree_host_size_t loom_ownership_lifetime_word_count(
    iree_host_size_t bit_count) {
  return (bit_count + 63u) / 64u;
}

static iree_status_t loom_ownership_lifetime_bitset_allocate(
    iree_arena_allocator_t* arena, iree_host_size_t word_count,
    loom_ownership_lifetime_bitset_t* out_bitset) {
  out_bitset->word_count = word_count;
  if (word_count == 0) {
    out_bitset->words = NULL;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, word_count,
                                                 sizeof(*out_bitset->words),
                                                 (void**)&out_bitset->words));
  memset(out_bitset->words, 0, word_count * sizeof(*out_bitset->words));
  return iree_ok_status();
}

static iree_status_t loom_ownership_lifetime_state_bits_allocate(
    iree_arena_allocator_t* arena, iree_host_size_t word_count,
    loom_ownership_lifetime_state_bits_t* out_state) {
  IREE_RETURN_IF_ERROR(loom_ownership_lifetime_bitset_allocate(
      arena, word_count, &out_state->borrowed));
  return loom_ownership_lifetime_bitset_allocate(arena, word_count,
                                                 &out_state->owned);
}

static void loom_ownership_lifetime_bitset_copy(
    loom_ownership_lifetime_bitset_t target,
    loom_ownership_lifetime_bitset_t source) {
  IREE_ASSERT(target.word_count == source.word_count);
  if (target.word_count == 0) {
    return;
  }
  memcpy(target.words, source.words, target.word_count * sizeof(*target.words));
}

static bool loom_ownership_lifetime_bitset_test(
    loom_ownership_lifetime_bitset_t bitset,
    loom_value_ordinal_t value_ordinal) {
  iree_host_size_t word_index = value_ordinal / 64u;
  IREE_ASSERT(word_index < bitset.word_count);
  uint64_t mask = UINT64_C(1) << (value_ordinal % 64u);
  return (bitset.words[word_index] & mask) != 0;
}

static bool loom_ownership_lifetime_bitset_set(
    loom_ownership_lifetime_bitset_t bitset,
    loom_value_ordinal_t value_ordinal) {
  iree_host_size_t word_index = value_ordinal / 64u;
  IREE_ASSERT(word_index < bitset.word_count);
  uint64_t mask = UINT64_C(1) << (value_ordinal % 64u);
  uint64_t old_word = bitset.words[word_index];
  bitset.words[word_index] = old_word | mask;
  return old_word != bitset.words[word_index];
}

static bool loom_ownership_lifetime_bitset_reset(
    loom_ownership_lifetime_bitset_t bitset,
    loom_value_ordinal_t value_ordinal) {
  iree_host_size_t word_index = value_ordinal / 64u;
  IREE_ASSERT(word_index < bitset.word_count);
  uint64_t mask = UINT64_C(1) << (value_ordinal % 64u);
  uint64_t old_word = bitset.words[word_index];
  bitset.words[word_index] = old_word & ~mask;
  return old_word != bitset.words[word_index];
}

static bool loom_ownership_lifetime_state_bits_equal(
    loom_ownership_lifetime_state_bits_t lhs,
    loom_ownership_lifetime_state_bits_t rhs) {
  IREE_ASSERT(lhs.borrowed.word_count == rhs.borrowed.word_count);
  IREE_ASSERT(lhs.owned.word_count == rhs.owned.word_count);
  for (iree_host_size_t i = 0; i < lhs.owned.word_count; ++i) {
    if (lhs.owned.words[i] != rhs.owned.words[i]) {
      return false;
    }
    if (lhs.borrowed.words[i] != rhs.borrowed.words[i]) {
      return false;
    }
  }
  return true;
}

static void loom_ownership_lifetime_state_bits_copy(
    loom_ownership_lifetime_state_bits_t target,
    loom_ownership_lifetime_state_bits_t source) {
  loom_ownership_lifetime_bitset_copy(target.borrowed, source.borrowed);
  loom_ownership_lifetime_bitset_copy(target.owned, source.owned);
}

static loom_ownership_lifetime_value_state_t
loom_ownership_lifetime_state_bits_get(
    loom_ownership_lifetime_state_bits_t state,
    loom_value_ordinal_t value_ordinal) {
  if (loom_ownership_lifetime_bitset_test(state.owned, value_ordinal)) {
    return LOOM_OWNERSHIP_LIFETIME_VALUE_OWNED;
  }
  if (loom_ownership_lifetime_bitset_test(state.borrowed, value_ordinal)) {
    return LOOM_OWNERSHIP_LIFETIME_VALUE_BORROWED;
  }
  return LOOM_OWNERSHIP_LIFETIME_VALUE_UNKNOWN;
}

static void loom_ownership_lifetime_state_bits_set(
    loom_ownership_lifetime_state_bits_t state,
    loom_value_ordinal_t value_ordinal,
    loom_ownership_lifetime_value_state_t value_state) {
  switch (value_state) {
    case LOOM_OWNERSHIP_LIFETIME_VALUE_UNKNOWN:
      loom_ownership_lifetime_bitset_reset(state.owned, value_ordinal);
      loom_ownership_lifetime_bitset_reset(state.borrowed, value_ordinal);
      return;
    case LOOM_OWNERSHIP_LIFETIME_VALUE_BORROWED:
      loom_ownership_lifetime_bitset_reset(state.owned, value_ordinal);
      loom_ownership_lifetime_bitset_set(state.borrowed, value_ordinal);
      return;
    case LOOM_OWNERSHIP_LIFETIME_VALUE_OWNED:
      loom_ownership_lifetime_bitset_reset(state.borrowed, value_ordinal);
      loom_ownership_lifetime_bitset_set(state.owned, value_ordinal);
      return;
  }
}

static bool loom_ownership_lifetime_bitset_find_first(
    loom_ownership_lifetime_bitset_t bitset,
    loom_value_ordinal_t* out_value_ordinal) {
  for (iree_host_size_t word_index = 0; word_index < bitset.word_count;
       ++word_index) {
    uint64_t bits = bitset.words[word_index];
    if (bits == 0) {
      continue;
    }
    uint32_t bit_index = iree_math_count_trailing_zeros_u64(bits);
    *out_value_ordinal = (loom_value_ordinal_t)(word_index * 64u + bit_index);
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

static iree_string_view_t loom_ownership_lifetime_phase_name(
    const loom_ownership_lifetime_state_t* state) {
  if (!iree_string_view_is_empty(state->options->phase_name)) {
    return state->options->phase_name;
  }
  return IREE_SV("ownership-lifetime");
}

static iree_string_view_t loom_ownership_lifetime_op_name(
    const loom_ownership_lifetime_state_t* state, const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(state->module, op);
  return vtable ? loom_op_vtable_name(vtable) : IREE_SV("<unknown>");
}

static iree_string_view_t loom_ownership_lifetime_value_name(
    const loom_ownership_lifetime_state_t* state, loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= state->module->values.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_value_t* value = loom_module_value(state->module, value_id);
  if (value->name_id != LOOM_STRING_ID_INVALID &&
      value->name_id < state->module->strings.count) {
    return state->module->strings.entries[value->name_id];
  }
  return IREE_SV("<unnamed>");
}

static iree_string_view_t loom_ownership_lifetime_value_state_name(
    loom_ownership_lifetime_value_state_t value_state, bool ever_seen) {
  switch (value_state) {
    case LOOM_OWNERSHIP_LIFETIME_VALUE_UNKNOWN:
      return ever_seen ? IREE_SV("not-owned") : IREE_SV("untracked");
    case LOOM_OWNERSHIP_LIFETIME_VALUE_BORROWED:
      return IREE_SV("borrowed");
    case LOOM_OWNERSHIP_LIFETIME_VALUE_OWNED:
      return IREE_SV("owned");
  }
  return IREE_SV("unknown");
}

static iree_string_view_t loom_ownership_lifetime_operand_effect_name(
    loom_operand_ownership_effect_t effect) {
  switch (effect) {
    case LOOM_OPERAND_OWNERSHIP_BORROW:
      return IREE_SV("borrow");
    case LOOM_OPERAND_OWNERSHIP_CONSUME:
      return IREE_SV("consume");
    case LOOM_OPERAND_OWNERSHIP_RETAIN:
      return IREE_SV("retain");
    case LOOM_OPERAND_OWNERSHIP_RELEASE:
      return IREE_SV("release");
    case LOOM_OPERAND_OWNERSHIP_DISCARD:
      return IREE_SV("discard");
    case LOOM_OPERAND_OWNERSHIP_ESCAPE:
      return IREE_SV("escape");
    case LOOM_OPERAND_OWNERSHIP_NONE:
      return IREE_SV("none");
  }
  return IREE_SV("unknown");
}

static iree_status_t loom_ownership_lifetime_emit(
    loom_ownership_lifetime_state_t* state, const loom_op_t* op,
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count) {
  if (!state->emit_diagnostics) {
    if (state->summary) {
      state->summary->needs_diagnostic_rerun = true;
    }
    return iree_ok_status();
  }
  state->failed = true;
  ++state->result->error_count;
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(state->options->emitter, &emission);
}

static iree_status_t loom_ownership_lifetime_emit_use_after_consume(
    loom_ownership_lifetime_state_t* state, const loom_op_t* op,
    uint16_t operand_index, loom_value_id_t value_id) {
  loom_diagnostic_field_ref_t operand_ref =
      loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, operand_index);
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_ownership_lifetime_phase_name(state)),
      loom_param_with_field_ref(
          loom_param_string(
              loom_ownership_lifetime_value_name(state, value_id)),
          operand_ref),
      loom_param_string(loom_ownership_lifetime_op_name(state, op)),
  };
  return loom_ownership_lifetime_emit(state, op, LOOM_ERR_DOMINANCE_012, params,
                                      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_ownership_lifetime_emit_requires_owned(
    loom_ownership_lifetime_state_t* state, const loom_op_t* op,
    uint16_t operand_index, loom_value_id_t value_id,
    loom_operand_ownership_effect_t effect,
    loom_ownership_lifetime_value_state_t value_state, bool ever_seen) {
  loom_diagnostic_field_ref_t operand_ref =
      loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, operand_index);
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_ownership_lifetime_phase_name(state)),
      loom_param_string(loom_ownership_lifetime_op_name(state, op)),
      loom_param_with_field_ref(
          loom_param_string(
              loom_ownership_lifetime_value_name(state, value_id)),
          operand_ref),
      loom_param_string(loom_ownership_lifetime_operand_effect_name(effect)),
      loom_param_string(
          loom_ownership_lifetime_value_state_name(value_state, ever_seen)),
  };
  return loom_ownership_lifetime_emit(state, op, LOOM_ERR_DOMINANCE_013, params,
                                      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_ownership_lifetime_emit_leak(
    loom_ownership_lifetime_state_t* state, const loom_op_t* op,
    loom_value_id_t value_id) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_ownership_lifetime_phase_name(state)),
      loom_param_string(loom_ownership_lifetime_value_name(state, value_id)),
  };
  return loom_ownership_lifetime_emit(state, op, LOOM_ERR_DOMINANCE_014, params,
                                      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_ownership_lifetime_emit_join_mismatch(
    loom_ownership_lifetime_state_t* state, const loom_op_t* terminator,
    loom_value_id_t value_id,
    loom_ownership_lifetime_value_state_t current_state,
    loom_ownership_lifetime_value_state_t incoming_state) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_ownership_lifetime_phase_name(state)),
      loom_param_string(loom_ownership_lifetime_value_name(state, value_id)),
      loom_param_string(
          loom_ownership_lifetime_value_state_name(current_state, true)),
      loom_param_string(
          loom_ownership_lifetime_value_state_name(incoming_state, true)),
  };
  return loom_ownership_lifetime_emit(state, terminator, LOOM_ERR_DOMINANCE_015,
                                      params, IREE_ARRAYSIZE(params));
}

//===----------------------------------------------------------------------===//
// State mutation
//===----------------------------------------------------------------------===//

static loom_value_ordinal_t loom_ownership_lifetime_try_ordinal(
    const loom_ownership_lifetime_state_t* state, loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= state->module->values.count) {
    return LOOM_VALUE_ORDINAL_INVALID;
  }
  return loom_local_value_domain_try_ordinal(state->value_domain, value_id);
}

static bool loom_ownership_lifetime_value_ever_seen(
    const loom_ownership_lifetime_state_t* state,
    loom_value_ordinal_t value_ordinal) {
  return loom_ownership_lifetime_bitset_test(state->ever_seen, value_ordinal);
}

static void loom_ownership_lifetime_mark_ever_seen(
    loom_ownership_lifetime_state_t* state,
    loom_value_ordinal_t value_ordinal) {
  loom_ownership_lifetime_bitset_set(state->ever_seen, value_ordinal);
}

static uint16_t loom_ownership_lifetime_origin_arg(
    const loom_ownership_lifetime_state_t* state,
    loom_value_ordinal_t value_ordinal) {
  if (!state->value_origin_args || value_ordinal >= state->value_count) {
    return LOOM_OWNERSHIP_LIFETIME_ARG_INDEX_NONE;
  }
  return state->value_origin_args[value_ordinal];
}

static void loom_ownership_lifetime_set_origin_arg(
    loom_ownership_lifetime_state_t* state, loom_value_ordinal_t value_ordinal,
    uint16_t arg_index) {
  if (state->value_origin_args && value_ordinal < state->value_count) {
    state->value_origin_args[value_ordinal] = arg_index;
  }
}

static bool loom_ownership_lifetime_promote_arg_consumed(
    loom_ownership_lifetime_state_t* state, uint16_t arg_index) {
  if (!state->inference || !state->summary ||
      arg_index >= state->summary->arg_count ||
      state->summary->arg_consumed[arg_index]) {
    return false;
  }
  state->summary->arg_consumed[arg_index] = true;
  state->summary_changed = true;
  state->needs_restart = true;
  return true;
}

static iree_status_t loom_ownership_lifetime_merge_summary_result(
    loom_ownership_lifetime_state_t* state, const loom_op_t* op,
    uint16_t result_index, loom_value_id_t value_id,
    loom_ownership_lifetime_value_state_t incoming_state) {
  if (!state->summary || result_index >= state->summary->result_count ||
      incoming_state == LOOM_OWNERSHIP_LIFETIME_VALUE_UNKNOWN) {
    return iree_ok_status();
  }

  loom_ownership_lifetime_value_state_t current_state =
      state->summary->result_states[result_index];
  if (current_state == incoming_state) {
    return iree_ok_status();
  }
  if (current_state == LOOM_OWNERSHIP_LIFETIME_VALUE_UNKNOWN ||
      (state->inference &&
       current_state == LOOM_OWNERSHIP_LIFETIME_VALUE_BORROWED &&
       incoming_state == LOOM_OWNERSHIP_LIFETIME_VALUE_OWNED)) {
    state->summary->result_states[result_index] = incoming_state;
    state->summary_changed = true;
    return iree_ok_status();
  }
  if (state->inference &&
      current_state == LOOM_OWNERSHIP_LIFETIME_VALUE_OWNED &&
      incoming_state == LOOM_OWNERSHIP_LIFETIME_VALUE_BORROWED) {
    return iree_ok_status();
  }
  return loom_ownership_lifetime_emit_join_mismatch(
      state, op, value_id, current_state, incoming_state);
}

static iree_status_t loom_ownership_lifetime_apply_borrow(
    loom_ownership_lifetime_state_t* state,
    loom_ownership_lifetime_state_bits_t bits, const loom_op_t* op,
    const loom_ownership_operand_effect_t* effect) {
  loom_value_ordinal_t value_ordinal =
      loom_ownership_lifetime_try_ordinal(state, effect->value_id);
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID) {
    return iree_ok_status();
  }
  loom_ownership_lifetime_value_state_t value_state =
      loom_ownership_lifetime_state_bits_get(bits, value_ordinal);
  bool ever_seen =
      loom_ownership_lifetime_value_ever_seen(state, value_ordinal);
  if (value_state == LOOM_OWNERSHIP_LIFETIME_VALUE_UNKNOWN && ever_seen) {
    return loom_ownership_lifetime_emit_use_after_consume(
        state, op, effect->operand_index, effect->value_id);
  }
  if (value_state == LOOM_OWNERSHIP_LIFETIME_VALUE_UNKNOWN &&
      state->inference &&
      loom_ownership_lifetime_origin_arg(state, value_ordinal) ==
          LOOM_OWNERSHIP_LIFETIME_ARG_INDEX_NONE) {
    return iree_ok_status();
  }
  if (value_state == LOOM_OWNERSHIP_LIFETIME_VALUE_UNKNOWN) {
    loom_ownership_lifetime_state_bits_set(
        bits, value_ordinal, LOOM_OWNERSHIP_LIFETIME_VALUE_BORROWED);
  }
  return iree_ok_status();
}

static iree_status_t loom_ownership_lifetime_apply_owned_consume(
    loom_ownership_lifetime_state_t* state,
    loom_ownership_lifetime_state_bits_t bits, const loom_op_t* op,
    const loom_ownership_operand_effect_t* effect) {
  loom_value_ordinal_t value_ordinal =
      loom_ownership_lifetime_try_ordinal(state, effect->value_id);
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID) {
    return iree_ok_status();
  }
  loom_ownership_lifetime_value_state_t value_state =
      loom_ownership_lifetime_state_bits_get(bits, value_ordinal);
  bool ever_seen =
      loom_ownership_lifetime_value_ever_seen(state, value_ordinal);
  if (value_state != LOOM_OWNERSHIP_LIFETIME_VALUE_OWNED) {
    uint16_t arg_index =
        loom_ownership_lifetime_origin_arg(state, value_ordinal);
    if (loom_ownership_lifetime_promote_arg_consumed(state, arg_index)) {
      return iree_ok_status();
    }
    return loom_ownership_lifetime_emit_requires_owned(
        state, op, effect->operand_index, effect->value_id, effect->effect,
        value_state, ever_seen);
  }
  loom_ownership_lifetime_state_bits_set(bits, value_ordinal,
                                         LOOM_OWNERSHIP_LIFETIME_VALUE_UNKNOWN);
  loom_ownership_lifetime_mark_ever_seen(state, value_ordinal);
  return iree_ok_status();
}

static iree_status_t loom_ownership_lifetime_apply_operand_effect(
    loom_ownership_lifetime_state_t* state,
    loom_ownership_lifetime_state_bits_t bits, const loom_op_t* op,
    const loom_ownership_operand_effect_t* effect) {
  if (state->count_statistics) {
    ++state->result->effects_checked;
  }
  switch (effect->effect) {
    case LOOM_OPERAND_OWNERSHIP_BORROW:
    case LOOM_OPERAND_OWNERSHIP_RETAIN:
      return loom_ownership_lifetime_apply_borrow(state, bits, op, effect);
    case LOOM_OPERAND_OWNERSHIP_CONSUME:
    case LOOM_OPERAND_OWNERSHIP_RELEASE:
    case LOOM_OPERAND_OWNERSHIP_DISCARD:
    case LOOM_OPERAND_OWNERSHIP_ESCAPE:
      return loom_ownership_lifetime_apply_owned_consume(state, bits, op,
                                                         effect);
    case LOOM_OPERAND_OWNERSHIP_NONE:
      return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_ownership_lifetime_apply_tied_result(
    loom_ownership_lifetime_state_t* state,
    loom_ownership_lifetime_state_bits_t bits, const loom_op_t* op,
    const loom_ownership_result_effect_t* effect) {
  if (effect->source_operand_index == LOOM_OWNERSHIP_SOURCE_OPERAND_NONE ||
      effect->source_operand_index >= op->operand_count) {
    return iree_ok_status();
  }
  const loom_value_id_t source_value_id =
      loom_op_const_operands(op)[effect->source_operand_index];
  loom_value_ordinal_t source_ordinal =
      loom_ownership_lifetime_try_ordinal(state, source_value_id);
  loom_value_ordinal_t result_ordinal =
      loom_ownership_lifetime_try_ordinal(state, effect->value_id);
  if (source_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      result_ordinal == LOOM_VALUE_ORDINAL_INVALID) {
    return iree_ok_status();
  }
  loom_ownership_lifetime_value_state_t source_state =
      loom_ownership_lifetime_state_bits_get(bits, source_ordinal);
  bool source_seen =
      loom_ownership_lifetime_value_ever_seen(state, source_ordinal);
  if (source_state == LOOM_OWNERSHIP_LIFETIME_VALUE_UNKNOWN && source_seen) {
    return loom_ownership_lifetime_emit_use_after_consume(
        state, op, effect->source_operand_index, source_value_id);
  }
  loom_ownership_lifetime_state_bits_set(bits, source_ordinal,
                                         LOOM_OWNERSHIP_LIFETIME_VALUE_UNKNOWN);
  loom_ownership_lifetime_state_bits_set(bits, result_ordinal, source_state);
  loom_ownership_lifetime_set_origin_arg(
      state, result_ordinal,
      loom_ownership_lifetime_origin_arg(state, source_ordinal));
  if (source_state != LOOM_OWNERSHIP_LIFETIME_VALUE_UNKNOWN) {
    loom_ownership_lifetime_mark_ever_seen(state, source_ordinal);
    loom_ownership_lifetime_mark_ever_seen(state, result_ordinal);
  }
  return iree_ok_status();
}

static iree_status_t loom_ownership_lifetime_apply_result_effect(
    loom_ownership_lifetime_state_t* state,
    loom_ownership_lifetime_state_bits_t bits, const loom_op_t* op,
    const loom_ownership_result_effect_t* effect) {
  if (state->count_statistics) {
    ++state->result->effects_checked;
  }
  loom_value_ordinal_t value_ordinal =
      loom_ownership_lifetime_try_ordinal(state, effect->value_id);
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID) {
    return iree_ok_status();
  }
  switch (effect->effect) {
    case LOOM_RESULT_OWNERSHIP_FRESH:
    case LOOM_RESULT_OWNERSHIP_RETAINED:
      loom_ownership_lifetime_state_bits_set(
          bits, value_ordinal, LOOM_OWNERSHIP_LIFETIME_VALUE_OWNED);
      loom_ownership_lifetime_mark_ever_seen(state, value_ordinal);
      loom_ownership_lifetime_set_origin_arg(
          state, value_ordinal, LOOM_OWNERSHIP_LIFETIME_ARG_INDEX_NONE);
      return iree_ok_status();
    case LOOM_RESULT_OWNERSHIP_BORROWED:
    case LOOM_RESULT_OWNERSHIP_ALIAS:
      loom_ownership_lifetime_state_bits_set(
          bits, value_ordinal, LOOM_OWNERSHIP_LIFETIME_VALUE_BORROWED);
      loom_ownership_lifetime_mark_ever_seen(state, value_ordinal);
      loom_ownership_lifetime_set_origin_arg(
          state, value_ordinal, LOOM_OWNERSHIP_LIFETIME_ARG_INDEX_NONE);
      return iree_ok_status();
    case LOOM_RESULT_OWNERSHIP_TIED:
      return loom_ownership_lifetime_apply_tied_result(state, bits, op, effect);
    case LOOM_RESULT_OWNERSHIP_NONE:
      return iree_ok_status();
  }
  return iree_ok_status();
}

static bool loom_ownership_lifetime_call_kind_is_runtime(
    loom_call_like_kind_t kind) {
  return kind == LOOM_CALL_LIKE_KIND_SEMANTIC ||
         kind == LOOM_CALL_LIKE_KIND_LOW_INTERNAL ||
         kind == LOOM_CALL_LIKE_KIND_LOW_INVOKE;
}

static loom_ownership_lifetime_function_summary_t*
loom_ownership_lifetime_callee_summary(loom_ownership_lifetime_state_t* state,
                                       loom_call_like_t call) {
  loom_symbol_ref_t callee_ref = loom_call_like_callee(call);
  if (!loom_symbol_ref_is_valid(callee_ref) || callee_ref.module_id != 0 ||
      callee_ref.symbol_id >= state->module_state->summary_count) {
    return NULL;
  }
  return &state->module_state->summaries[callee_ref.symbol_id];
}

static loom_ownership_lifetime_value_state_t
loom_ownership_lifetime_call_result_state(
    const loom_ownership_lifetime_state_t* state,
    const loom_ownership_lifetime_function_summary_t* summary,
    uint16_t result_index) {
  if (!summary || !summary->body || result_index >= summary->result_count) {
    return LOOM_OWNERSHIP_LIFETIME_VALUE_BORROWED;
  }
  loom_ownership_lifetime_value_state_t result_state =
      summary->result_states[result_index];
  if (result_state == LOOM_OWNERSHIP_LIFETIME_VALUE_UNKNOWN &&
      !state->inference) {
    return LOOM_OWNERSHIP_LIFETIME_VALUE_BORROWED;
  }
  return result_state;
}

static iree_status_t loom_ownership_lifetime_transfer_call(
    loom_ownership_lifetime_state_t* state,
    loom_ownership_lifetime_state_bits_t bits, const loom_op_t* op,
    loom_call_like_t call) {
  loom_ownership_lifetime_function_summary_t* summary =
      loom_ownership_lifetime_callee_summary(state, call);
  loom_value_slice_t operands = loom_call_like_operands(call);
  uint16_t operand_offset = loom_call_like_operand_offset(call);
  for (uint16_t i = 0;
       i < operands.count && !state->failed && !state->needs_restart; ++i) {
    loom_operand_ownership_effect_t effect_kind =
        summary && summary->body && i < summary->arg_count &&
                summary->arg_consumed[i]
            ? LOOM_OPERAND_OWNERSHIP_CONSUME
            : LOOM_OPERAND_OWNERSHIP_BORROW;
    loom_ownership_operand_effect_t effect = {
        .operand_index = (uint16_t)(operand_offset + i),
        .value_id = operands.values[i],
        .effect = effect_kind,
    };
    IREE_RETURN_IF_ERROR(
        loom_ownership_lifetime_apply_operand_effect(state, bits, op, &effect));
  }

  loom_value_slice_t results = loom_call_like_results(call);
  for (uint16_t i = 0; i < results.count && !state->failed; ++i) {
    if (state->count_statistics) {
      ++state->result->effects_checked;
    }
    loom_value_ordinal_t value_ordinal =
        loom_ownership_lifetime_try_ordinal(state, results.values[i]);
    if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID) {
      continue;
    }
    loom_ownership_lifetime_value_state_t result_state =
        loom_ownership_lifetime_call_result_state(state, summary, i);
    if (result_state == LOOM_OWNERSHIP_LIFETIME_VALUE_UNKNOWN) {
      continue;
    }
    loom_ownership_lifetime_state_bits_set(bits, value_ordinal, result_state);
    loom_ownership_lifetime_mark_ever_seen(state, value_ordinal);
    loom_ownership_lifetime_set_origin_arg(
        state, value_ordinal, LOOM_OWNERSHIP_LIFETIME_ARG_INDEX_NONE);
  }
  return iree_ok_status();
}

static iree_status_t loom_ownership_lifetime_transfer_return(
    loom_ownership_lifetime_state_t* state,
    loom_ownership_lifetime_state_bits_t bits, const loom_op_t* op) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  iree_host_size_t count = 0;
  if (state->summary) {
    count = op->operand_count < state->summary->result_count
                ? op->operand_count
                : state->summary->result_count;
  }
  for (iree_host_size_t i = 0; i < count && !state->failed; ++i) {
    if (state->count_statistics) {
      ++state->result->effects_checked;
    }
    loom_value_ordinal_t value_ordinal =
        loom_ownership_lifetime_try_ordinal(state, operands[i]);
    if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID) {
      continue;
    }
    loom_ownership_lifetime_value_state_t value_state =
        loom_ownership_lifetime_state_bits_get(bits, value_ordinal);
    bool ever_seen =
        loom_ownership_lifetime_value_ever_seen(state, value_ordinal);
    if (value_state == LOOM_OWNERSHIP_LIFETIME_VALUE_UNKNOWN && ever_seen) {
      return loom_ownership_lifetime_emit_use_after_consume(
          state, op, (uint16_t)i, operands[i]);
    }
    if (value_state == LOOM_OWNERSHIP_LIFETIME_VALUE_UNKNOWN &&
        state->inference &&
        loom_ownership_lifetime_origin_arg(state, value_ordinal) ==
            LOOM_OWNERSHIP_LIFETIME_ARG_INDEX_NONE) {
      continue;
    }

    loom_ownership_lifetime_value_state_t result_state =
        value_state == LOOM_OWNERSHIP_LIFETIME_VALUE_OWNED
            ? LOOM_OWNERSHIP_LIFETIME_VALUE_OWNED
            : LOOM_OWNERSHIP_LIFETIME_VALUE_BORROWED;
    IREE_RETURN_IF_ERROR(loom_ownership_lifetime_merge_summary_result(
        state, op, (uint16_t)i, operands[i], result_state));
    if (value_state == LOOM_OWNERSHIP_LIFETIME_VALUE_OWNED) {
      loom_ownership_lifetime_state_bits_set(
          bits, value_ordinal, LOOM_OWNERSHIP_LIFETIME_VALUE_UNKNOWN);
      loom_ownership_lifetime_mark_ever_seen(state, value_ordinal);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_ownership_lifetime_transfer_op(
    loom_ownership_lifetime_state_t* state,
    loom_ownership_lifetime_state_bits_t bits, const loom_op_t* op) {
  if (loom_func_return_isa(op)) {
    return loom_ownership_lifetime_transfer_return(state, bits, op);
  }
  loom_call_like_t call = loom_call_like_cast(state->module, (loom_op_t*)op);
  if (loom_call_like_isa(call) &&
      loom_ownership_lifetime_call_kind_is_runtime(loom_call_like_kind(call))) {
    return loom_ownership_lifetime_transfer_call(state, bits, op, call);
  }
  for (uint16_t i = 0; i < op->operand_count && !state->failed; ++i) {
    loom_ownership_operand_effect_t effect = {0};
    if (!loom_ownership_operand_effect_at(state->module, op, i, &effect)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_ownership_lifetime_apply_operand_effect(state, bits, op, &effect));
  }
  for (uint16_t i = 0; i < op->result_count && !state->failed; ++i) {
    loom_ownership_result_effect_t effect = {0};
    if (!loom_ownership_result_effect_at(state->module, op, i, &effect)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_ownership_lifetime_apply_result_effect(state, bits, op, &effect));
  }
  return iree_ok_status();
}

static iree_status_t loom_ownership_lifetime_transfer_block(
    loom_ownership_lifetime_state_t* state, const loom_block_t* block,
    loom_ownership_lifetime_state_bits_t in,
    loom_ownership_lifetime_state_bits_t out) {
  loom_ownership_lifetime_state_bits_copy(out, in);
  if (state->count_statistics) {
    ++state->result->blocks_checked;
  }
  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (state->count_statistics) {
      ++state->result->ops_checked;
    }
    IREE_RETURN_IF_ERROR(loom_ownership_lifetime_transfer_op(state, out, op));
    if (state->failed || state->needs_restart) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// CFG propagation
//===----------------------------------------------------------------------===//

static iree_status_t loom_ownership_lifetime_allocate_block_states(
    loom_ownership_lifetime_state_t* state, iree_host_size_t block_count) {
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->options->arena, block_count, sizeof(*state->blocks),
      (void**)&state->blocks));
  memset(state->blocks, 0, block_count * sizeof(*state->blocks));
  for (iree_host_size_t i = 0; i < block_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ownership_lifetime_state_bits_allocate(
        state->options->arena, state->word_count, &state->blocks[i].in));
    IREE_RETURN_IF_ERROR(loom_ownership_lifetime_state_bits_allocate(
        state->options->arena, state->word_count, &state->blocks[i].out));
  }
  state->queue_capacity = block_count > 0 ? block_count : 1;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->options->arena, state->queue_capacity,
                                sizeof(*state->queue), (void**)&state->queue));
  return iree_ok_status();
}

static iree_status_t loom_ownership_lifetime_enqueue_block(
    loom_ownership_lifetime_state_t* state, uint16_t block_index) {
  loom_ownership_lifetime_block_state_t* block_state =
      &state->blocks[block_index];
  if (block_state->queued) {
    return iree_ok_status();
  }
  if (state->queue_write >= state->queue_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->options->arena, state->queue_write, state->queue_write + 1,
        sizeof(*state->queue), &state->queue_capacity, (void**)&state->queue));
  }
  state->queue[state->queue_write++] = block_index;
  block_state->queued = true;
  return iree_ok_status();
}

static void loom_ownership_lifetime_initialize_arguments(
    loom_ownership_lifetime_state_t* state,
    loom_ownership_lifetime_state_bits_t entry_state) {
  uint16_t arg_count = 0;
  const loom_value_id_t* arg_ids =
      loom_func_like_arg_ids(state->function, &arg_count);
  if (!arg_ids || !state->summary) {
    return;
  }
  uint16_t count = arg_count < state->summary->arg_count
                       ? arg_count
                       : state->summary->arg_count;
  for (uint16_t i = 0; i < count; ++i) {
    loom_value_ordinal_t value_ordinal =
        loom_ownership_lifetime_try_ordinal(state, arg_ids[i]);
    if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID) {
      continue;
    }
    loom_ownership_lifetime_set_origin_arg(state, value_ordinal, i);
    loom_ownership_lifetime_state_bits_set(
        entry_state, value_ordinal,
        state->summary->arg_consumed[i]
            ? LOOM_OWNERSHIP_LIFETIME_VALUE_OWNED
            : LOOM_OWNERSHIP_LIFETIME_VALUE_BORROWED);
    loom_ownership_lifetime_mark_ever_seen(state, value_ordinal);
  }
}

static bool loom_ownership_lifetime_dequeue_block(
    loom_ownership_lifetime_state_t* state, uint16_t* out_block_index) {
  if (state->queue_read >= state->queue_write) {
    return false;
  }
  uint16_t block_index = state->queue[state->queue_read++];
  state->blocks[block_index].queued = false;
  *out_block_index = block_index;
  return true;
}

static iree_status_t loom_ownership_lifetime_apply_cfg_br_payload(
    loom_ownership_lifetime_state_t* state,
    loom_ownership_lifetime_state_bits_t edge_state,
    const loom_cfg_edge_info_t* edge) {
  if (!loom_cfg_br_isa(edge->terminator)) {
    return iree_ok_status();
  }
  const loom_block_t* target =
      loom_region_const_block(state->body, edge->target_block_index);
  const loom_value_id_t* operands = loom_op_const_operands(edge->terminator);
  iree_host_size_t count = edge->terminator->operand_count < target->arg_count
                               ? edge->terminator->operand_count
                               : target->arg_count;
  for (iree_host_size_t i = 0; i < count; ++i) {
    loom_value_ordinal_t source_ordinal =
        loom_ownership_lifetime_try_ordinal(state, operands[i]);
    loom_value_ordinal_t target_ordinal = loom_ownership_lifetime_try_ordinal(
        state, loom_block_arg_id(target, (uint16_t)i));
    if (source_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
        target_ordinal == LOOM_VALUE_ORDINAL_INVALID) {
      continue;
    }
    loom_ownership_lifetime_value_state_t source_state =
        loom_ownership_lifetime_state_bits_get(edge_state, source_ordinal);
    bool source_seen =
        loom_ownership_lifetime_value_ever_seen(state, source_ordinal);
    if (source_state == LOOM_OWNERSHIP_LIFETIME_VALUE_UNKNOWN && source_seen) {
      return loom_ownership_lifetime_emit_use_after_consume(
          state, edge->terminator, (uint16_t)i, operands[i]);
    }
    loom_ownership_lifetime_state_bits_set(
        edge_state, source_ordinal, LOOM_OWNERSHIP_LIFETIME_VALUE_UNKNOWN);
    loom_ownership_lifetime_state_bits_set(edge_state, target_ordinal,
                                           source_state);
    loom_ownership_lifetime_set_origin_arg(
        state, target_ordinal,
        loom_ownership_lifetime_origin_arg(state, source_ordinal));
    if (source_state != LOOM_OWNERSHIP_LIFETIME_VALUE_UNKNOWN) {
      loom_ownership_lifetime_mark_ever_seen(state, target_ordinal);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_ownership_lifetime_merge_value(
    loom_ownership_lifetime_state_t* state, const loom_op_t* terminator,
    loom_ownership_lifetime_state_bits_t target,
    loom_ownership_lifetime_state_bits_t incoming,
    loom_value_ordinal_t value_ordinal, bool* inout_changed) {
  loom_ownership_lifetime_value_state_t current_state =
      loom_ownership_lifetime_state_bits_get(target, value_ordinal);
  loom_ownership_lifetime_value_state_t incoming_state =
      loom_ownership_lifetime_state_bits_get(incoming, value_ordinal);
  if (current_state == incoming_state) {
    return iree_ok_status();
  }
  if (current_state == LOOM_OWNERSHIP_LIFETIME_VALUE_UNKNOWN &&
      incoming_state == LOOM_OWNERSHIP_LIFETIME_VALUE_BORROWED) {
    loom_ownership_lifetime_state_bits_set(target, value_ordinal,
                                           incoming_state);
    *inout_changed = true;
    return iree_ok_status();
  }
  if (current_state == LOOM_OWNERSHIP_LIFETIME_VALUE_BORROWED &&
      incoming_state == LOOM_OWNERSHIP_LIFETIME_VALUE_UNKNOWN) {
    return iree_ok_status();
  }
  loom_value_id_t value_id = state->value_ids[value_ordinal];
  return loom_ownership_lifetime_emit_join_mismatch(
      state, terminator, value_id, current_state, incoming_state);
}

static iree_status_t loom_ownership_lifetime_merge_state(
    loom_ownership_lifetime_state_t* state, const loom_op_t* terminator,
    loom_ownership_lifetime_state_bits_t target,
    loom_ownership_lifetime_state_bits_t incoming, bool* out_changed) {
  *out_changed = false;
  for (iree_host_size_t word_index = 0;
       word_index < state->word_count && !state->failed; ++word_index) {
    uint64_t touched =
        target.owned.words[word_index] | incoming.owned.words[word_index] |
        target.borrowed.words[word_index] | incoming.borrowed.words[word_index];
    while (touched != 0 && !state->failed) {
      uint32_t bit_index = iree_math_count_trailing_zeros_u64(touched);
      loom_value_ordinal_t value_ordinal =
          (loom_value_ordinal_t)(word_index * 64u + bit_index);
      if (value_ordinal < state->value_count) {
        IREE_RETURN_IF_ERROR(loom_ownership_lifetime_merge_value(
            state, terminator, target, incoming, value_ordinal, out_changed));
      }
      touched &= touched - 1u;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_ownership_lifetime_propagate_edge(
    loom_ownership_lifetime_state_t* state,
    loom_ownership_lifetime_state_bits_t scratch,
    const loom_cfg_edge_info_t* edge) {
  loom_ownership_lifetime_block_state_t* source_state =
      &state->blocks[edge->source_block_index];
  loom_ownership_lifetime_block_state_t* target_state =
      &state->blocks[edge->target_block_index];
  loom_ownership_lifetime_state_bits_copy(scratch, source_state->out);
  IREE_RETURN_IF_ERROR(
      loom_ownership_lifetime_apply_cfg_br_payload(state, scratch, edge));
  if (state->failed) {
    return iree_ok_status();
  }
  if (!target_state->initialized) {
    loom_ownership_lifetime_state_bits_copy(target_state->in, scratch);
    target_state->initialized = true;
    IREE_RETURN_IF_ERROR(
        loom_ownership_lifetime_enqueue_block(state, edge->target_block_index));
    return iree_ok_status();
  }
  bool changed = false;
  IREE_RETURN_IF_ERROR(loom_ownership_lifetime_merge_state(
      state, edge->terminator, target_state->in, scratch, &changed));
  if (changed && !state->failed) {
    IREE_RETURN_IF_ERROR(
        loom_ownership_lifetime_enqueue_block(state, edge->target_block_index));
  }
  return iree_ok_status();
}

static iree_status_t loom_ownership_lifetime_check_exit(
    loom_ownership_lifetime_state_t* state, const loom_block_t* block,
    loom_ownership_lifetime_state_bits_t bits) {
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_ownership_lifetime_bitset_find_first(bits.owned, &value_ordinal)) {
    return iree_ok_status();
  }
  if (value_ordinal >= state->value_count) {
    return iree_ok_status();
  }
  const loom_op_t* op = block->last_op ? block->last_op : state->function.op;
  return loom_ownership_lifetime_emit_leak(state, op,
                                           state->value_ids[value_ordinal]);
}

static iree_status_t loom_ownership_lifetime_run_cfg(
    loom_ownership_lifetime_state_t* state, const loom_cfg_graph_t* graph) {
  if (graph->block_count == 0) {
    return iree_ok_status();
  }
  loom_ownership_lifetime_state_bits_t scratch = {0};
  IREE_RETURN_IF_ERROR(loom_ownership_lifetime_state_bits_allocate(
      state->options->arena, state->word_count, &scratch));
  loom_ownership_lifetime_state_bits_t old_out = {0};
  IREE_RETURN_IF_ERROR(loom_ownership_lifetime_state_bits_allocate(
      state->options->arena, state->word_count, &old_out));

  state->blocks[0].initialized = true;
  loom_ownership_lifetime_initialize_arguments(state, state->blocks[0].in);
  IREE_RETURN_IF_ERROR(loom_ownership_lifetime_enqueue_block(state, 0));
  uint16_t block_index = 0;
  while (!state->failed && !state->needs_restart &&
         loom_ownership_lifetime_dequeue_block(state, &block_index)) {
    if (!loom_cfg_graph_block_is_reachable(graph, block_index)) {
      continue;
    }
    const loom_block_t* block = graph->blocks[block_index].block;
    loom_ownership_lifetime_block_state_t* block_state =
        &state->blocks[block_index];
    bool first_process = !block_state->processed;
    block_state->processed = true;
    loom_ownership_lifetime_state_bits_copy(old_out, block_state->out);
    IREE_RETURN_IF_ERROR(loom_ownership_lifetime_transfer_block(
        state, block, block_state->in, block_state->out));
    if (state->failed || state->needs_restart) {
      return iree_ok_status();
    }
    if (!first_process &&
        loom_ownership_lifetime_state_bits_equal(old_out, block_state->out)) {
      continue;
    }
    loom_cfg_edge_index_span_t successor_edges =
        loom_cfg_graph_successor_edges(graph, block_index);
    for (iree_host_size_t i = 0;
         i < successor_edges.count && !state->failed && !state->needs_restart;
         ++i) {
      const loom_cfg_edge_info_t* edge =
          loom_cfg_graph_edge(graph, successor_edges.values[i]);
      IREE_RETURN_IF_ERROR(
          loom_ownership_lifetime_propagate_edge(state, scratch, edge));
    }
  }

  for (uint16_t i = 0;
       i < graph->block_count && !state->failed && !state->needs_restart; ++i) {
    if (!loom_cfg_graph_block_is_reachable(graph, i)) {
      continue;
    }
    loom_cfg_block_index_span_t successors =
        loom_cfg_graph_successors(graph, i);
    if (successors.count != 0) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_ownership_lifetime_check_exit(
        state, graph->blocks[i].block, state->blocks[i].out));
  }
  return iree_ok_status();
}

static iree_status_t loom_ownership_lifetime_run_linear_regions(
    loom_ownership_lifetime_state_t* state) {
  loom_ownership_lifetime_state_bits_t empty = {0};
  IREE_RETURN_IF_ERROR(loom_ownership_lifetime_state_bits_allocate(
      state->options->arena, state->word_count, &empty));
  loom_ownership_lifetime_initialize_arguments(state, empty);
  for (uint16_t block_index = 0; block_index < state->body->block_count &&
                                 !state->failed && !state->needs_restart;
       ++block_index) {
    const loom_block_t* block =
        loom_region_const_block(state->body, block_index);
    IREE_RETURN_IF_ERROR(loom_ownership_lifetime_transfer_block(
        state, block, empty, state->blocks[block_index].out));
    if (!state->failed && !state->needs_restart) {
      IREE_RETURN_IF_ERROR(loom_ownership_lifetime_check_exit(
          state, block, state->blocks[block_index].out));
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Module summaries
//===----------------------------------------------------------------------===//

static iree_status_t loom_ownership_lifetime_initialize_summary(
    loom_ownership_lifetime_module_state_t* module_state,
    const loom_symbol_t* symbol,
    loom_ownership_lifetime_function_summary_t* summary) {
  if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE) ||
      !symbol->defining_op) {
    return iree_ok_status();
  }
  loom_func_like_t function =
      loom_func_like_cast(module_state->module, symbol->defining_op);
  if (!loom_func_like_isa(function)) {
    return iree_ok_status();
  }
  uint16_t arg_count = 0;
  (void)loom_func_like_arg_ids(function, &arg_count);
  *summary = (loom_ownership_lifetime_function_summary_t){
      .function = function,
      .body = loom_func_like_body(function),
      .arg_count = arg_count,
      .result_count = function.op->result_count,
  };
  if (summary->arg_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        module_state->options->arena, summary->arg_count,
        sizeof(*summary->arg_consumed), (void**)&summary->arg_consumed));
    memset(summary->arg_consumed, 0,
           summary->arg_count * sizeof(*summary->arg_consumed));
  }
  if (summary->result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        module_state->options->arena, summary->result_count,
        sizeof(*summary->result_states), (void**)&summary->result_states));
    memset(summary->result_states, 0,
           summary->result_count * sizeof(*summary->result_states));
  }
  return iree_ok_status();
}

static iree_status_t loom_ownership_lifetime_initialize_module_summaries(
    loom_ownership_lifetime_module_state_t* module_state) {
  module_state->summary_count = module_state->module->symbols.count;
  if (module_state->summary_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      module_state->options->arena, module_state->summary_count,
      sizeof(*module_state->summaries), (void**)&module_state->summaries));
  memset(module_state->summaries, 0,
         module_state->summary_count * sizeof(*module_state->summaries));
  for (iree_host_size_t i = 0; i < module_state->summary_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ownership_lifetime_initialize_summary(
        module_state, &module_state->module->symbols.entries[i],
        &module_state->summaries[i]));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Function analysis
//===----------------------------------------------------------------------===//

static iree_status_t loom_ownership_lifetime_analyze_function(
    loom_ownership_lifetime_module_state_t* module_state,
    loom_ownership_lifetime_function_summary_t* summary,
    iree_arena_allocator_t* arena, bool inference, bool emit_diagnostics,
    bool count_statistics, bool* out_changed) {
  *out_changed = false;
  if (!summary->body) {
    return iree_ok_status();
  }
  if (!emit_diagnostics) {
    summary->needs_diagnostic_rerun = false;
  }

  loom_local_value_domain_t value_domain = {0};
  iree_status_t status = loom_local_value_domain_acquire_for_region(
      module_state->module, summary->body, arena, &value_domain);
  loom_ownership_lifetime_options_t function_options = *module_state->options;
  function_options.arena = arena;
  loom_ownership_lifetime_state_t state = {
      .module = module_state->module,
      .module_state = module_state,
      .function = summary->function,
      .summary = summary,
      .options = &function_options,
      .result = module_state->result,
      .body = summary->body,
      .value_domain = &value_domain,
      .value_ids = value_domain.value_ids,
      .value_count = value_domain.value_count,
      .word_count =
          loom_ownership_lifetime_word_count(value_domain.value_count),
      .inference = inference,
      .emit_diagnostics = emit_diagnostics,
      .count_statistics = count_statistics,
  };

  if (iree_status_is_ok(status)) {
    status = loom_ownership_lifetime_bitset_allocate(arena, state.word_count,
                                                     &state.ever_seen);
  }
  if (iree_status_is_ok(status) && state.value_count > 0) {
    status = iree_arena_allocate_array(arena, state.value_count,
                                       sizeof(*state.value_origin_args),
                                       (void**)&state.value_origin_args);
    if (iree_status_is_ok(status)) {
      memset(state.value_origin_args, 0xFF,
             state.value_count * sizeof(*state.value_origin_args));
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_ownership_lifetime_allocate_block_states(
        &state, summary->body->block_count);
  }

  if (iree_status_is_ok(status)) {
    if (!iree_any_bit_set(summary->body->flags,
                          LOOM_REGION_INSTANCE_FLAG_CFG)) {
      status = loom_ownership_lifetime_run_linear_regions(&state);
    } else {
      loom_cfg_graph_t graph = {0};
      status = loom_cfg_graph_build(module_state->module, summary->body, arena,
                                    &graph);
      if (iree_status_is_ok(status) && graph.malformed) {
        status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                  "CFG graph is malformed; run Loom "
                                  "verification before ownership lifetime "
                                  "analysis");
      }
      if (iree_status_is_ok(status)) {
        status = loom_ownership_lifetime_run_cfg(&state, &graph);
      }
    }
  }

  if (loom_local_value_domain_is_acquired(&value_domain)) {
    loom_local_value_domain_release(&value_domain);
  }
  if (iree_status_is_ok(status)) {
    *out_changed = state.summary_changed;
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Function graph
//===----------------------------------------------------------------------===//

static bool loom_ownership_lifetime_graph_callee_node(
    const loom_ownership_lifetime_graph_t* graph, loom_symbol_ref_t callee,
    iree_host_size_t* out_node) {
  if (!loom_symbol_ref_is_valid(callee) || callee.module_id != 0 ||
      callee.symbol_id >= graph->symbol_to_node_count) {
    return false;
  }
  iree_host_size_t node = graph->symbol_to_node[callee.symbol_id];
  if (node == IREE_HOST_SIZE_MAX) {
    return false;
  }
  *out_node = node;
  return true;
}

typedef struct loom_ownership_lifetime_successor_walk_t {
  // Function graph adapter.
  const loom_ownership_lifetime_graph_t* graph;
  // SCC successor visitor.
  loom_scc_successor_callback_t visitor;
} loom_ownership_lifetime_successor_walk_t;

static iree_status_t loom_ownership_lifetime_visit_successor_call(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_ownership_lifetime_successor_walk_t* walk =
      (loom_ownership_lifetime_successor_walk_t*)user_data;
  loom_call_like_t call =
      loom_call_like_cast(walk->graph->module_state->module, op);
  if (!loom_call_like_isa(call) ||
      !loom_ownership_lifetime_call_kind_is_runtime(
          loom_call_like_kind(call))) {
    return iree_ok_status();
  }
  iree_host_size_t callee_node = IREE_HOST_SIZE_MAX;
  if (!loom_ownership_lifetime_graph_callee_node(
          walk->graph, loom_call_like_callee(call), &callee_node)) {
    return iree_ok_status();
  }
  return walk->visitor.fn(walk->visitor.user_data, callee_node);
}

static iree_status_t loom_ownership_lifetime_visit_successors(
    void* user_data, iree_host_size_t node,
    loom_scc_successor_callback_t visitor) {
  const loom_ownership_lifetime_graph_t* graph =
      (const loom_ownership_lifetime_graph_t*)user_data;
  loom_ownership_lifetime_successor_walk_t walk = {
      .graph = graph,
      .visitor = visitor,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  iree_arena_reset(graph->walk_arena);
  return loom_walk_function(
      graph->module_state->module, graph->nodes[node].summary->function,
      LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_ownership_lifetime_visit_successor_call,
                             &walk},
      graph->walk_arena, &walk_result);
}

static iree_status_t loom_ownership_lifetime_build_graph(
    loom_ownership_lifetime_module_state_t* module_state,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* walk_arena,
    loom_ownership_lifetime_graph_t* out_graph, loom_scc_list_t* out_sccs) {
  memset(out_graph, 0, sizeof(*out_graph));
  out_graph->module_state = module_state;
  out_graph->walk_arena = walk_arena;
  out_graph->symbol_to_node_count = module_state->summary_count;
  if (module_state->summary_count == 0) {
    *out_sccs = (loom_scc_list_t){0};
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module_state->summary_count, sizeof(*out_graph->symbol_to_node),
      (void**)&out_graph->symbol_to_node));
  for (iree_host_size_t i = 0; i < module_state->summary_count; ++i) {
    out_graph->symbol_to_node[i] = IREE_HOST_SIZE_MAX;
  }

  iree_host_size_t function_count = 0;
  for (iree_host_size_t i = 0; i < module_state->summary_count; ++i) {
    if (module_state->summaries[i].body) {
      ++function_count;
    }
  }
  if (function_count == 0) {
    *out_sccs = (loom_scc_list_t){0};
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, function_count,
                                                 sizeof(*out_graph->nodes),
                                                 (void**)&out_graph->nodes));
  out_graph->node_count = function_count;
  iree_host_size_t node = 0;
  for (iree_host_size_t i = 0; i < module_state->summary_count; ++i) {
    if (!module_state->summaries[i].body) {
      continue;
    }
    out_graph->nodes[node] = (loom_ownership_lifetime_graph_node_t){
        .summary = &module_state->summaries[i],
    };
    out_graph->symbol_to_node[i] = node;
    ++node;
  }

  loom_scc_graph_t scc_graph = {
      .node_count = function_count,
      .visit_successors = loom_scc_visit_successors_callback_make(
          loom_ownership_lifetime_visit_successors, out_graph),
  };
  return loom_scc_compute(&scc_graph, NULL, arena, out_sccs);
}

//===----------------------------------------------------------------------===//
// Module analysis
//===----------------------------------------------------------------------===//

static iree_host_size_t loom_ownership_lifetime_scc_max_iterations(
    const loom_scc_t* scc, const loom_ownership_lifetime_graph_t* graph) {
  iree_host_size_t max_iterations = 1;
  for (iree_host_size_t i = 0; i < scc->node_count; ++i) {
    const loom_ownership_lifetime_function_summary_t* summary =
        graph->nodes[scc->nodes[i]].summary;
    max_iterations += summary->arg_count + summary->result_count;
  }
  return max_iterations;
}

static iree_status_t loom_ownership_lifetime_analyze_scc(
    loom_ownership_lifetime_graph_t* graph, const loom_scc_t* scc,
    iree_arena_allocator_t* arena) {
  const iree_host_size_t max_iterations =
      loom_ownership_lifetime_scc_max_iterations(scc, graph);
  bool converged = false;
  for (iree_host_size_t iteration = 0; iteration < max_iterations;
       ++iteration) {
    bool changed = false;
    for (iree_host_size_t i = 0; i < scc->node_count; ++i) {
      loom_ownership_lifetime_function_summary_t* summary =
          graph->nodes[scc->nodes[i]].summary;
      iree_arena_reset(arena);
      bool function_changed = false;
      IREE_RETURN_IF_ERROR(loom_ownership_lifetime_analyze_function(
          graph->module_state, summary, arena, true, false, true,
          &function_changed));
      changed = changed || function_changed;
    }
    if (!changed) {
      converged = true;
      break;
    }
  }
  if (!converged) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "ownership lifetime function summaries did not converge");
  }
  for (iree_host_size_t i = 0; i < scc->node_count; ++i) {
    loom_ownership_lifetime_function_summary_t* summary =
        graph->nodes[scc->nodes[i]].summary;
    if (!summary->needs_diagnostic_rerun) {
      continue;
    }
    iree_arena_reset(arena);
    bool ignored_changed = false;
    IREE_RETURN_IF_ERROR(loom_ownership_lifetime_analyze_function(
        graph->module_state, summary, arena, false, true, false,
        &ignored_changed));
  }
  return iree_ok_status();
}

static iree_status_t loom_ownership_lifetime_analyze_sccs(
    loom_ownership_lifetime_graph_t* graph, const loom_scc_list_t* sccs,
    iree_arena_allocator_t* arena) {
  for (iree_host_size_t i = 0; i < sccs->count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_ownership_lifetime_analyze_scc(graph, &sccs->values[i], arena));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

iree_status_t loom_ownership_lifetime_analyze_module(
    loom_module_t* module, const loom_ownership_lifetime_options_t* options,
    loom_ownership_lifetime_result_t* out_result) {
  if (!module || !options || !options->arena || !out_result) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "ownership lifetime analysis requires module, options, arena, and "
        "output result");
  }
  *out_result = (loom_ownership_lifetime_result_t){0};
  loom_ownership_lifetime_module_state_t module_state = {
      .module = module,
      .options = options,
      .result = out_result,
  };

  iree_arena_allocator_t analysis_arena = {0};
  iree_arena_allocator_t walk_arena = {0};
  iree_arena_initialize(options->arena->block_pool, &analysis_arena);
  iree_arena_initialize(options->arena->block_pool, &walk_arena);
  iree_status_t status =
      loom_ownership_lifetime_initialize_module_summaries(&module_state);
  if (iree_status_is_ok(status)) {
    loom_ownership_lifetime_graph_t graph = {0};
    loom_scc_list_t sccs = {0};
    status = loom_ownership_lifetime_build_graph(&module_state, options->arena,
                                                 &walk_arena, &graph, &sccs);
    if (iree_status_is_ok(status)) {
      status =
          loom_ownership_lifetime_analyze_sccs(&graph, &sccs, &analysis_arena);
    }
  }
  iree_arena_deinitialize(&walk_arena);
  iree_arena_deinitialize(&analysis_arena);
  return status;
}
