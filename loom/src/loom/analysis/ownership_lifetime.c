// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/ownership_lifetime.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/analysis/ownership.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/util/cfg_graph.h"

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

typedef struct loom_ownership_lifetime_state_t {
  // Module whose function body is being checked.
  const loom_module_t* module;
  // Function-like symbol whose body is being checked.
  loom_func_like_t function;
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
  ++state->result->effects_checked;
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
  ++state->result->effects_checked;
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
      return iree_ok_status();
    case LOOM_RESULT_OWNERSHIP_BORROWED:
    case LOOM_RESULT_OWNERSHIP_ALIAS:
      loom_ownership_lifetime_state_bits_set(
          bits, value_ordinal, LOOM_OWNERSHIP_LIFETIME_VALUE_BORROWED);
      loom_ownership_lifetime_mark_ever_seen(state, value_ordinal);
      return iree_ok_status();
    case LOOM_RESULT_OWNERSHIP_TIED:
      return loom_ownership_lifetime_apply_tied_result(state, bits, op, effect);
    case LOOM_RESULT_OWNERSHIP_NONE:
      return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_ownership_lifetime_transfer_op(
    loom_ownership_lifetime_state_t* state,
    loom_ownership_lifetime_state_bits_t bits, const loom_op_t* op) {
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
  ++state->result->blocks_checked;
  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    ++state->result->ops_checked;
    IREE_RETURN_IF_ERROR(loom_ownership_lifetime_transfer_op(state, out, op));
    if (state->failed) {
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
  IREE_RETURN_IF_ERROR(loom_ownership_lifetime_enqueue_block(state, 0));
  uint16_t block_index = 0;
  while (!state->failed &&
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
    if (state->failed) {
      return iree_ok_status();
    }
    if (!first_process &&
        loom_ownership_lifetime_state_bits_equal(old_out, block_state->out)) {
      continue;
    }
    loom_cfg_edge_index_span_t successor_edges =
        loom_cfg_graph_successor_edges(graph, block_index);
    for (iree_host_size_t i = 0; i < successor_edges.count && !state->failed;
         ++i) {
      const loom_cfg_edge_info_t* edge =
          loom_cfg_graph_edge(graph, successor_edges.values[i]);
      IREE_RETURN_IF_ERROR(
          loom_ownership_lifetime_propagate_edge(state, scratch, edge));
    }
  }

  for (uint16_t i = 0; i < graph->block_count && !state->failed; ++i) {
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
  for (uint16_t block_index = 0;
       block_index < state->body->block_count && !state->failed;
       ++block_index) {
    const loom_block_t* block =
        loom_region_const_block(state->body, block_index);
    IREE_RETURN_IF_ERROR(loom_ownership_lifetime_transfer_block(
        state, block, empty, state->blocks[block_index].out));
    if (!state->failed) {
      IREE_RETURN_IF_ERROR(loom_ownership_lifetime_check_exit(
          state, block, state->blocks[block_index].out));
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

iree_status_t loom_ownership_lifetime_verify_function(
    const loom_module_t* module, loom_func_like_t function,
    const loom_ownership_lifetime_options_t* options,
    loom_ownership_lifetime_result_t* out_result) {
  if (!module || !options || !options->arena || !options->value_domain ||
      !out_result) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "ownership lifetime analysis requires module, options, arena, local "
        "value domain, and output result");
  }
  *out_result = (loom_ownership_lifetime_result_t){0};
  loom_region_t* body = loom_func_like_body(function);
  if (!body) {
    return iree_ok_status();
  }

  loom_ownership_lifetime_state_t state = {
      .module = module,
      .function = function,
      .options = options,
      .result = out_result,
      .body = body,
      .value_domain = options->value_domain,
      .value_ids = options->value_domain->value_ids,
      .value_count = options->value_domain->value_count,
      .word_count = loom_ownership_lifetime_word_count(
          options->value_domain->value_count),
  };
  IREE_RETURN_IF_ERROR(loom_ownership_lifetime_bitset_allocate(
      options->arena, state.word_count, &state.ever_seen));
  IREE_RETURN_IF_ERROR(
      loom_ownership_lifetime_allocate_block_states(&state, body->block_count));

  if (!iree_any_bit_set(body->flags, LOOM_REGION_INSTANCE_FLAG_CFG)) {
    return loom_ownership_lifetime_run_linear_regions(&state);
  }

  loom_cfg_graph_t graph = {0};
  IREE_RETURN_IF_ERROR(
      loom_cfg_graph_build(module, body, options->arena, &graph));
  if (graph.malformed) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "CFG graph is malformed; run Loom verification "
                            "before ownership lifetime analysis");
  }
  return loom_ownership_lifetime_run_cfg(&state, &graph);
}
