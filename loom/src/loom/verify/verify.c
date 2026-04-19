// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/verify/verify.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/internal/unicode.h"
#include "loom/error/renderer.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/special_values.h"

//===----------------------------------------------------------------------===//
// Internal verification state
//===----------------------------------------------------------------------===//

// Maximum region nesting depth. Each nested region pushes a watermark
// onto the scope stack. 32 levels covers any realistic IR (function →
// loop → branch → ...).
#define LOOM_VERIFY_MAX_SCOPE_DEPTH 32

// Reusable scratch tables for validating one op's tied-result metadata.
//
// `result_index_bits` and `operand_index_bits` track whether a result or
// operand index has already appeared in the current op's tied-result list.
// The occurrence arrays track which repeated source spelling of a result or
// operand field corresponds to each tied-result entry so duplicate-entry
// diagnostics can highlight both the current claim and the first claim.
// `operand_value_ids` is a reusable sorted scratch copy of the current op's
// valid operand value IDs, used to detect repeated operand values that would
// make name-based tied-result syntax ambiguous.
typedef struct loom_verify_tied_table_t {
  uint64_t* result_index_bits;
  iree_host_size_t result_index_word_capacity;
  uint16_t* result_field_occurrences;
  uint16_t* first_result_field_occurrences;
  iree_host_size_t result_field_occurrence_capacity;

  uint64_t* operand_index_bits;
  iree_host_size_t operand_index_word_capacity;
  uint16_t* operand_field_occurrences;
  uint16_t* first_operand_field_occurrences;
  iree_host_size_t operand_field_occurrence_capacity;

  loom_value_id_t* operand_value_ids;
  iree_host_size_t operand_value_count;
  iree_host_size_t operand_value_capacity;
} loom_verify_tied_table_t;

typedef struct loom_verify_state_t {
  const loom_module_t* module;
  loom_diagnostic_sink_t sink;
  loom_source_resolver_t source_resolver;
  loom_verify_result_t* result;
  uint32_t max_errors;

  // First non-OK status returned by the diagnostic sink. Stored so
  // verification can stop cleanly without leaking status objects.
  iree_status_t diagnostic_status;

  // Scratch arena for all verification-time allocations. Initialized
  // from the module's block pool, deinitialized (bulk O(1) free) when
  // the verification call completes.
  iree_arena_allocator_t arena;

  // Bitset indexed by value_id: one bit per value in the module.
  // A set bit means the value is defined and visible at the current
  // point in the walk.
  uint64_t* defined_bits;
  iree_host_size_t defined_bits_length;  // Number of uint64_t words.

  // Bitset indexed by value_id: marks values that have been consumed
  // by a tied result. Any subsequent use of a consumed value is an
  // error.
  uint64_t* consumed_bits;

  // Op that first consumed each value_id via a tied result. This is NULL for
  // values that have not been consumed or when the consuming op is unknown.
  const loom_op_t** consuming_ops;

  // Reusable per-op scratch for tied-result uniqueness checks.
  loom_verify_tied_table_t tied_table;

  // Stack of value_ids that have been marked as defined during the
  // walk. Used for scope cleanup: when exiting a region, we clear
  // bits for all values defined since the watermark. Grows
  // dynamically via iree_arena_grow_array on overflow.
  uint32_t* defined_stack;
  iree_host_size_t defined_stack_count;
  iree_host_size_t defined_stack_capacity;

  // Scope watermarks: defined_stack index at each region entry.
  iree_host_size_t scope_watermarks[LOOM_VERIFY_MAX_SCOPE_DEPTH];
  uint32_t scope_depth;

} loom_verify_state_t;

// Records the first non-OK diagnostic sink status and consumes any
// later ones so verification never leaks status objects.
static void loom_verify_record_diagnostic_status(loom_verify_state_t* state,
                                                 iree_status_t status) {
  if (iree_status_is_ok(status)) return;
  if (iree_status_is_ok(state->diagnostic_status)) {
    state->diagnostic_status = status;
  } else {
    iree_status_ignore(status);
  }
}

static iree_status_t loom_verify_take_diagnostic_status(
    loom_verify_state_t* state) {
  iree_status_t status = state->diagnostic_status;
  state->diagnostic_status = iree_ok_status();
  return status;
}

static iree_status_t loom_verify_pending_diagnostic_status(
    loom_verify_state_t* state) {
  if (iree_status_is_ok(state->diagnostic_status)) {
    return iree_ok_status();
  }
  return loom_verify_take_diagnostic_status(state);
}

//===----------------------------------------------------------------------===//
// Bitset helpers
//===----------------------------------------------------------------------===//

static inline void loom_bitset_set(uint64_t* bits, uint32_t index) {
  bits[index / 64] |= ((uint64_t)1 << (index % 64));
}

static inline void loom_bitset_clear(uint64_t* bits, uint32_t index) {
  bits[index / 64] &= ~((uint64_t)1 << (index % 64));
}

static inline bool loom_bitset_test(const uint64_t* bits, uint32_t index) {
  return (bits[index / 64] & ((uint64_t)1 << (index % 64))) != 0;
}

static inline iree_host_size_t loom_bitset_word_count(
    iree_host_size_t bit_count) {
  return bit_count > 0 ? (bit_count + 63) / 64 : 0;
}

static int loom_compare_value_ids(const void* lhs, const void* rhs) {
  loom_value_id_t lhs_id = *(const loom_value_id_t*)lhs;
  loom_value_id_t rhs_id = *(const loom_value_id_t*)rhs;
  return (lhs_id > rhs_id) - (lhs_id < rhs_id);
}

// Prepares the reusable tied-result scratch tables for an op with
// |result_count| results and |operand_count| operands. Grows the
// scratch storage from the verifier arena on demand, then clears only the
// active index-bitset words needed by the current op. Operand occurrences are
// counted per logical operand field, not across all operands. For a body op:
//   test.invoke @callee(%arg) ... -> (%arg as f32, %arg as f32)
// operand 0 has occurrence 0 in the ordinary operand list and occurrences 1/2
// in the tied-result clauses, so duplicate-tie tracking starts at 1. Signature
// ties use the function signature arg domain and the tied-result clause is the
// first operand-field spelling we target here, so they start at 0. If a future
// format emits tied-result clauses before ordinary operand spellings, or emits
// extra same-field spellings first, teach parser field spans about that role
// instead of layering more source-order heuristics into the verifier.
static iree_status_t loom_verify_tied_table_reset(
    loom_verify_tied_table_t* tied_table, iree_arena_allocator_t* arena,
    iree_host_size_t result_count, iree_host_size_t operand_count,
    uint16_t operand_occurrence_base) {
  iree_host_size_t result_word_count = loom_bitset_word_count(result_count);
  if (result_word_count > tied_table->result_index_word_capacity) {
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(arena, 0, result_word_count, sizeof(uint64_t),
                              &tied_table->result_index_word_capacity,
                              (void**)&tied_table->result_index_bits));
  }
  if (result_word_count > 0) {
    memset(tied_table->result_index_bits, 0,
           result_word_count * sizeof(tied_table->result_index_bits[0]));
  }
  if (result_count > tied_table->result_field_occurrence_capacity) {
    iree_host_size_t result_occurrence_capacity =
        tied_table->result_field_occurrence_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, 0, result_count, sizeof(uint16_t), &result_occurrence_capacity,
        (void**)&tied_table->result_field_occurrences));
    iree_host_size_t first_occurrence_capacity =
        tied_table->result_field_occurrence_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, 0, result_count, sizeof(uint16_t), &first_occurrence_capacity,
        (void**)&tied_table->first_result_field_occurrences));
    IREE_ASSERT(first_occurrence_capacity == result_occurrence_capacity);
    tied_table->result_field_occurrence_capacity = result_occurrence_capacity;
  }
  if (result_count > 0) {
    memset(tied_table->result_field_occurrences, 0,
           result_count * sizeof(tied_table->result_field_occurrences[0]));
  }

  iree_host_size_t operand_word_count = loom_bitset_word_count(operand_count);
  if (operand_word_count > tied_table->operand_index_word_capacity) {
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(arena, 0, operand_word_count, sizeof(uint64_t),
                              &tied_table->operand_index_word_capacity,
                              (void**)&tied_table->operand_index_bits));
  }
  if (operand_word_count > 0) {
    memset(tied_table->operand_index_bits, 0,
           operand_word_count * sizeof(tied_table->operand_index_bits[0]));
  }
  if (operand_count > tied_table->operand_field_occurrence_capacity) {
    iree_host_size_t operand_occurrence_capacity =
        tied_table->operand_field_occurrence_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, 0, operand_count, sizeof(uint16_t), &operand_occurrence_capacity,
        (void**)&tied_table->operand_field_occurrences));
    iree_host_size_t first_occurrence_capacity =
        tied_table->operand_field_occurrence_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, 0, operand_count, sizeof(uint16_t), &first_occurrence_capacity,
        (void**)&tied_table->first_operand_field_occurrences));
    IREE_ASSERT(first_occurrence_capacity == operand_occurrence_capacity);
    tied_table->operand_field_occurrence_capacity = operand_occurrence_capacity;
  }
  for (iree_host_size_t i = 0; i < operand_count; ++i) {
    tied_table->operand_field_occurrences[i] = operand_occurrence_base;
  }

  if (operand_count > tied_table->operand_value_capacity) {
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(arena, 0, operand_count, sizeof(loom_value_id_t),
                              &tied_table->operand_value_capacity,
                              (void**)&tied_table->operand_value_ids));
  }
  tied_table->operand_value_count = 0;

  return iree_ok_status();
}

// Records one tied-result claim for |result_index| and returns the field
// occurrence sidecar for the current claim. If this result index was claimed
// previously, |out_first_occurrence| receives the first claimant occurrence and
// the function returns true.
static bool loom_verify_tied_table_claim_result(
    loom_verify_tied_table_t* tied_table, uint16_t result_index,
    uint16_t* out_current_occurrence, uint16_t* out_first_occurrence) {
  *out_current_occurrence =
      tied_table->result_field_occurrences[result_index]++;
  if (loom_bitset_test(tied_table->result_index_bits, result_index)) {
    *out_first_occurrence =
        tied_table->first_result_field_occurrences[result_index];
    return true;
  }
  loom_bitset_set(tied_table->result_index_bits, result_index);
  tied_table->first_result_field_occurrences[result_index] =
      *out_current_occurrence;
  return false;
}

// Records one tied-result claim for |operand_index| and returns the field
// occurrence sidecar for the current claim. If this operand index was claimed
// previously, |out_first_occurrence| receives the first claimant occurrence and
// the function returns true.
static bool loom_verify_tied_table_claim_operand(
    loom_verify_tied_table_t* tied_table, uint16_t operand_index,
    uint16_t* out_current_occurrence, uint16_t* out_first_occurrence) {
  *out_current_occurrence =
      tied_table->operand_field_occurrences[operand_index]++;
  if (loom_bitset_test(tied_table->operand_index_bits, operand_index)) {
    *out_first_occurrence =
        tied_table->first_operand_field_occurrences[operand_index];
    return true;
  }
  loom_bitset_set(tied_table->operand_index_bits, operand_index);
  tied_table->first_operand_field_occurrences[operand_index] =
      *out_current_occurrence;
  return false;
}

// Returns true if |value_id| appears in more than one operand slot in the
// current op's sorted operand-value scratch array.
static bool loom_verify_tied_table_has_duplicate_operand_value(
    const loom_verify_tied_table_t* tied_table, loom_value_id_t value_id) {
  iree_host_size_t low = 0;
  iree_host_size_t high = tied_table->operand_value_count;
  while (low < high) {
    iree_host_size_t mid = low + (high - low) / 2;
    if (tied_table->operand_value_ids[mid] < value_id) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low < tied_table->operand_value_count &&
         tied_table->operand_value_ids[low] == value_id &&
         (low + 1 < tied_table->operand_value_count &&
          tied_table->operand_value_ids[low + 1] == value_id);
}

// Returns true if this op's FuncArgs are stored as op operands and therefore
// represent signature definitions instead of ordinary SSA uses.
static bool loom_verify_func_args_are_operands(const loom_op_vtable_t* vtable) {
  return vtable->func_like != NULL && vtable->func_like->args_as_operands;
}

// Returns true if this op has a function signature scope whose arguments are
// defined by FuncArgs and may be referenced by result types/predicates/ties.
static bool loom_verify_has_func_signature_scope(
    const loom_op_vtable_t* vtable) {
  return vtable->func_like != NULL &&
         iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE);
}

// Returns the argument IDs that form a func-like symbol's signature operand
// domain. Bodyful functions read entry block args; bodyless declarations read
// op operands.
static const loom_value_id_t* loom_verify_func_signature_arg_ids(
    const loom_op_t* op, const loom_op_vtable_t* vtable,
    uint16_t* out_arg_count) {
  *out_arg_count = 0;
  if (!vtable->func_like) return NULL;
  if (vtable->func_like->args_as_operands) {
    *out_arg_count = op->operand_count;
    return loom_op_const_operands(op);
  }
  uint8_t body_index = vtable->func_like->body_region_index;
  if (body_index == LOOM_REGION_INDEX_NONE || body_index >= op->region_count) {
    return NULL;
  }
  loom_region_t* body = loom_op_regions(op)[body_index];
  if (!body || body->block_count == 0) return NULL;
  const loom_block_t* entry = loom_region_const_entry_block(body);
  *out_arg_count = entry->arg_count;
  return entry->arg_ids;
}

//===----------------------------------------------------------------------===//
// Scope management
//===----------------------------------------------------------------------===//

static iree_status_t loom_verify_push_scope(loom_verify_state_t* state) {
  if (state->scope_depth >= LOOM_VERIFY_MAX_SCOPE_DEPTH) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "region nesting depth exceeds verifier limit (%d); the input IR "
        "has more nesting than the verifier can track",
        LOOM_VERIFY_MAX_SCOPE_DEPTH);
  }
  state->scope_watermarks[state->scope_depth] = state->defined_stack_count;
  ++state->scope_depth;
  return iree_ok_status();
}

static void loom_verify_pop_scope(loom_verify_state_t* state) {
  if (state->scope_depth == 0) return;
  --state->scope_depth;
  iree_host_size_t watermark = state->scope_watermarks[state->scope_depth];
  // Clear all defined bits for values defined in the scope we're leaving.
  for (iree_host_size_t i = watermark; i < state->defined_stack_count; ++i) {
    loom_bitset_clear(state->defined_bits, state->defined_stack[i]);
  }
  state->defined_stack_count = watermark;
}

static iree_status_t loom_verify_define_value(loom_verify_state_t* state,
                                              loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID) return iree_ok_status();
  loom_bitset_set(state->defined_bits, value_id);
  // Push onto defined stack for scope cleanup. Grow dynamically if
  // the initial capacity heuristic was too small.
  if (state->defined_stack_count >= state->defined_stack_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        &state->arena, state->defined_stack_count,
        state->defined_stack_count + 1, sizeof(uint32_t),
        &state->defined_stack_capacity, (void**)&state->defined_stack));
  }
  state->defined_stack[state->defined_stack_count++] = value_id;
  return iree_ok_status();
}

static void loom_verify_consume_value(loom_verify_state_t* state,
                                      loom_value_id_t value_id,
                                      const loom_op_t* consuming_op) {
  if (value_id == LOOM_VALUE_ID_INVALID) return;
  if (value_id >= state->module->values.count) return;
  if (!loom_bitset_test(state->consumed_bits, value_id)) {
    state->consuming_ops[value_id] = consuming_op;
  }
  loom_bitset_set(state->consumed_bits, value_id);
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Returns the value name for a value_id, or "<unnamed>" if unavailable.
// The name does NOT include the % prefix (consistent with how names are
// stored in the string table). For unnamed values, returns "<unnamed>".
static iree_string_view_t loom_verify_value_name(
    const loom_verify_state_t* state, loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= state->module->values.count) {
    return iree_make_cstring_view("<unnamed>");
  }
  const loom_value_t* value = loom_module_value(state->module, value_id);
  if (value->name_id != LOOM_STRING_ID_INVALID &&
      value->name_id < state->module->strings.count) {
    return state->module->strings.entries[value->name_id];
  }
  return iree_make_cstring_view("<unnamed>");
}

// Returns the symbol name for a symbol ref, or "<unnamed>" if unavailable.
static iree_string_view_t loom_verify_symbol_name(
    const loom_verify_state_t* state, loom_symbol_ref_t ref) {
  if (!loom_symbol_ref_is_valid(ref) ||
      ref.symbol_id >= state->module->symbols.count) {
    return iree_make_cstring_view("<unnamed>");
  }
  const loom_symbol_t* symbol = &state->module->symbols.entries[ref.symbol_id];
  if (symbol->name_id != LOOM_STRING_ID_INVALID &&
      symbol->name_id < state->module->strings.count) {
    return state->module->strings.entries[symbol->name_id];
  }
  return iree_make_cstring_view("<unnamed>");
}

static iree_string_view_t loom_verify_symbol_definition_name(
    const loom_symbol_t* symbol) {
  if (!symbol || !symbol->definition) return IREE_SV("unresolved");
  return loom_symbol_definition_descriptor_name(symbol->definition);
}

//===----------------------------------------------------------------------===//
// Diagnostic emission
//===----------------------------------------------------------------------===//

static bool loom_verify_at_error_limit(const loom_verify_state_t* state) {
  return state->max_errors > 0 &&
         state->result->error_count >= state->max_errors;
}

// Computes the byte offset into |source| for a 1-based (line, column)
// pair. Scans for newlines to find the target line, then walks UTF-8
// codepoints to reach the target column. Columns are counted as
// codepoints (matching the tokenizer's convention). Returns the byte
// offset, clamped to source.size if the position is past end.
static iree_host_size_t loom_source_byte_offset(iree_string_view_t source,
                                                uint32_t line,
                                                uint32_t column) {
  if (line == 0) return 0;
  // Scan newlines to find the byte offset of the start of |line|.
  uint32_t current_line = 1;
  iree_host_size_t offset = 0;
  while (current_line < line && offset < source.size) {
    if (source.data[offset] == '\n') {
      ++current_line;
    }
    ++offset;
  }
  if (current_line < line) return source.size;
  // Walk UTF-8 codepoints to reach the target column (1-based).
  // Column 1 means "start of line" = offset stays where it is.
  iree_host_size_t line_start = offset;
  uint32_t current_column = 1;
  while (current_column < column && offset < source.size &&
         source.data[offset] != '\n') {
    iree_unicode_utf8_decode(source, &offset);
    ++current_column;
  }
  (void)line_start;
  return offset > source.size ? source.size : offset;
}

// Resolves a location ID to a source range via the configured resolver.
static bool loom_verify_resolve_location_id(const loom_verify_state_t* state,
                                            loom_location_id_t location,
                                            loom_source_range_t* out_range) {
  if (!loom_source_resolve(state->source_resolver, state->module, location,
                           out_range)) {
    return false;
  }
  if (out_range->provenance == LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE &&
      out_range->source.size > 0) {
    out_range->provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE;
  }
  return true;
}

// Resolves an op's location to a source range via the configured resolver.
static bool loom_verify_resolve_location(const loom_verify_state_t* state,
                                         const loom_op_t* op,
                                         loom_source_range_t* out_range) {
  if (!op) return false;
  return loom_verify_resolve_location_id(state, op->location, out_range);
}

// Maximum per-token highlight ranges per diagnostic.
#define LOOM_VERIFY_MAX_HIGHLIGHTS 8

// A field highlight target derived from structured diagnostic param metadata.
// |printer_field_ref| is the callback ref emitted by the text printer;
// |diagnostic_field_ref| is the sink-facing ref stored on the output highlight;
// |param_index| links the highlight back to the originating diagnostic param.
typedef struct loom_verify_highlight_target_t {
  loom_print_field_ref_t printer_field_ref;
  loom_diagnostic_field_ref_t diagnostic_field_ref;
  iree_host_size_t param_index;
} loom_verify_highlight_target_t;

// Callback state for collecting field highlights during printing. The verifier
// populates |wanted_targets| from structured param field refs, then passes this
// as user_data to the printer's field callback. The callback records byte
// ranges and preserves the field/param sidecars for matches.
typedef struct loom_highlight_collector_t {
  const loom_verify_highlight_target_t* wanted_targets;
  iree_host_size_t wanted_count;
  uint16_t target_occurrences[LOOM_VERIFY_MAX_HIGHLIGHTS];
  loom_highlight_range_t* highlights;
  iree_host_size_t highlight_count;
} loom_highlight_collector_t;

// Finds the first target whose field kind/index match |field_ref| and whose
// requested occurrence matches the number of equal field refs seen so far for
// that target. |target_occurrences| is updated for every equal field ref so
// repeated spans can be selected deterministically.
static const loom_verify_highlight_target_t* loom_verify_match_highlight_target(
    const loom_verify_highlight_target_t* wanted_targets,
    iree_host_size_t wanted_count, loom_print_field_ref_t field_ref,
    uint16_t* target_occurrences) {
  const loom_verify_highlight_target_t* matching_target = NULL;
  for (iree_host_size_t i = 0; i < wanted_count; ++i) {
    if (!loom_print_field_ref_equal(wanted_targets[i].printer_field_ref,
                                    field_ref)) {
      continue;
    }
    uint16_t occurrence = target_occurrences[i]++;
    if (!matching_target &&
        wanted_targets[i].diagnostic_field_ref.occurrence == occurrence) {
      matching_target = &wanted_targets[i];
    }
  }
  return matching_target;
}

static void loom_highlight_field_callback(void* user_data,
                                          loom_print_field_ref_t field_ref,
                                          iree_host_size_t start,
                                          iree_host_size_t end) {
  loom_highlight_collector_t* collector =
      (loom_highlight_collector_t*)user_data;
  if (collector->highlight_count >= LOOM_VERIFY_MAX_HIGHLIGHTS) return;
  const loom_verify_highlight_target_t* target =
      loom_verify_match_highlight_target(collector->wanted_targets,
                                         collector->wanted_count, field_ref,
                                         collector->target_occurrences);
  if (!target) return;
  collector->highlights[collector->highlight_count].start = start;
  collector->highlights[collector->highlight_count].end = end;
  collector->highlights[collector->highlight_count].field_ref =
      target->diagnostic_field_ref;
  collector->highlights[collector->highlight_count].param_index =
      target->param_index;
  ++collector->highlight_count;
}

// Converts a diagnostic field ref sidecar to the printer callback ref used by
// loom_text_print_operation_with_field_callback. Returns false when
// the diagnostic ref does not identify an op field that the printer can label.
static bool loom_verify_print_field_ref(
    loom_diagnostic_field_ref_t diagnostic_field_ref,
    loom_print_field_ref_t* out_field_ref) {
  loom_print_field_kind_t kind = LOOM_PRINT_FIELD_OPERAND;
  switch (diagnostic_field_ref.kind) {
    case LOOM_DIAGNOSTIC_FIELD_OPERAND:
      kind = LOOM_PRINT_FIELD_OPERAND;
      break;
    case LOOM_DIAGNOSTIC_FIELD_RESULT:
      kind = LOOM_PRINT_FIELD_RESULT;
      break;
    case LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE:
      kind = LOOM_PRINT_FIELD_ATTR;
      break;
    case LOOM_DIAGNOSTIC_FIELD_REGION:
      kind = LOOM_PRINT_FIELD_REGION;
      break;
    case LOOM_DIAGNOSTIC_FIELD_SUCCESSOR:
      kind = LOOM_PRINT_FIELD_SUCCESSOR;
      break;
    case LOOM_DIAGNOSTIC_FIELD_NONE:
    default:
      return false;
  }
  *out_field_ref = loom_print_field_ref(kind, diagnostic_field_ref.index);
  return true;
}

// Collects field highlight targets directly from structured diagnostic params.
static iree_host_size_t loom_collect_highlight_targets(
    const loom_diagnostic_param_t* params, iree_host_size_t param_count,
    loom_verify_highlight_target_t* out_targets) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0;
       i < param_count && count < LOOM_VERIFY_MAX_HIGHLIGHTS; ++i) {
    if (!loom_diagnostic_field_ref_is_set(params[i].field_ref)) continue;
    loom_print_field_ref_t printer_field_ref = {0};
    if (!loom_verify_print_field_ref(params[i].field_ref, &printer_field_ref)) {
      continue;
    }
    out_targets[count++] = (loom_verify_highlight_target_t){
        .printer_field_ref = printer_field_ref,
        .diagnostic_field_ref = params[i].field_ref,
        .param_index = i,
    };
  }
  return count;
}

static bool loom_verify_print_field_ref_from_location_field(
    const loom_location_field_span_t* field_span,
    loom_print_field_ref_t* out_field_ref) {
  loom_print_field_kind_t kind = LOOM_PRINT_FIELD_OPERAND;
  switch (field_span->kind) {
    case LOOM_LOCATION_FIELD_OPERAND:
      kind = LOOM_PRINT_FIELD_OPERAND;
      break;
    case LOOM_LOCATION_FIELD_RESULT:
      kind = LOOM_PRINT_FIELD_RESULT;
      break;
    case LOOM_LOCATION_FIELD_ATTRIBUTE:
      kind = LOOM_PRINT_FIELD_ATTR;
      break;
    case LOOM_LOCATION_FIELD_REGION:
      kind = LOOM_PRINT_FIELD_REGION;
      break;
    case LOOM_LOCATION_FIELD_SUCCESSOR:
      kind = LOOM_PRINT_FIELD_SUCCESSOR;
      break;
    default:
      return false;
  }
  *out_field_ref = loom_print_field_ref(kind, field_span->index);
  return true;
}

// Collects parser-sidecar highlights from the op's file location after
// resolving the whole-op source range to original source text.
static iree_host_size_t loom_collect_source_backed_highlights(
    const loom_module_t* module, const loom_op_t* op,
    const loom_source_range_t* source_location,
    const loom_verify_highlight_target_t* wanted_targets,
    iree_host_size_t wanted_count, loom_highlight_range_t* out_highlights,
    iree_host_size_t max_highlight_count) {
  if (!op || wanted_count == 0 || source_location->source.size == 0 ||
      op->location == LOOM_LOCATION_UNKNOWN ||
      (iree_host_size_t)op->location >= module->locations.count) {
    return 0;
  }

  const loom_location_entry_t* location =
      &module->locations.entries[op->location];
  if (location->kind != LOOM_LOCATION_FILE ||
      location->file.field_span_count == 0 || !location->file.field_spans) {
    return 0;
  }

  uint16_t target_occurrences[LOOM_VERIFY_MAX_HIGHLIGHTS] = {0};
  iree_host_size_t highlight_count = 0;
  for (uint16_t span_index = 0; span_index < location->file.field_span_count &&
                                highlight_count < max_highlight_count;
       ++span_index) {
    loom_print_field_ref_t span_field_ref = {0};
    if (!loom_verify_print_field_ref_from_location_field(
            &location->file.field_spans[span_index], &span_field_ref)) {
      continue;
    }

    const loom_verify_highlight_target_t* target =
        loom_verify_match_highlight_target(wanted_targets, wanted_count,
                                           span_field_ref, target_occurrences);
    if (!target) continue;

    const loom_location_field_span_t* field_span =
        &location->file.field_spans[span_index];
    iree_host_size_t start_offset = loom_source_byte_offset(
        source_location->source, field_span->start_line, field_span->start_col);
    iree_host_size_t end_offset = loom_source_byte_offset(
        source_location->source, field_span->end_line, field_span->end_col);
    if (start_offset >= end_offset || start_offset < source_location->start ||
        end_offset > source_location->end) {
      continue;
    }

    out_highlights[highlight_count++] = (loom_highlight_range_t){
        .start = start_offset,
        .end = end_offset,
        .field_ref = target->diagnostic_field_ref,
        .param_index = target->param_index,
    };
  }

  return highlight_count;
}

// Converts a packed verifier constraint field ref into the diagnostic-local
// sidecar representation. |element_offset| lets callers point at a specific
// element inside a variadic field while preserving the human-facing name shape.
static loom_diagnostic_field_ref_t loom_verify_diagnostic_field_ref(
    uint8_t field_ref, uint16_t element_offset) {
  uint8_t index = LOOM_FIELD_REF_INDEX(field_ref);
  switch (LOOM_FIELD_REF_CATEGORY(field_ref)) {
    case LOOM_FIELD_OPERAND:
      return loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND,
                                       (uint16_t)(index + element_offset));
    case LOOM_FIELD_RESULT:
      return loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT,
                                       (uint16_t)(index + element_offset));
    case LOOM_FIELD_ATTR:
      return loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                       (uint16_t)(index + element_offset));
    case LOOM_FIELD_REGION:
      return loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_REGION,
                                       (uint16_t)(index + element_offset));
    default:
      return loom_diagnostic_field_ref_none();
  }
}

// Returns a string param annotated with the corresponding structured field ref.
static loom_diagnostic_param_t loom_verify_param_string_for_diagnostic_field(
    iree_string_view_t value, loom_diagnostic_field_kind_t field_kind,
    uint16_t field_index) {
  return loom_param_with_field_ref(
      loom_param_string(value),
      loom_diagnostic_field_ref(field_kind, field_index));
}

// Returns a string param annotated with a packed verifier field ref.
static loom_diagnostic_param_t loom_verify_param_string_for_field(
    iree_string_view_t value, uint8_t field_ref) {
  return loom_param_with_field_ref(
      loom_param_string(value), loom_verify_diagnostic_field_ref(field_ref, 0));
}

// Returns an indexed field-name string param annotated with the concrete
// element ref inside a variadic field.
static loom_diagnostic_param_t loom_verify_param_string_for_indexed_field(
    iree_string_view_t value, uint8_t field_ref, uint16_t element_offset) {
  return loom_param_with_field_ref(
      loom_param_string(value),
      loom_verify_diagnostic_field_ref(field_ref, element_offset));
}

// Resolves labeled related ops through the source resolver.
// Returns the number of note entries written to |out_related_locations|.
static iree_host_size_t loom_verify_collect_related_locations(
    const loom_verify_state_t* state,
    const loom_diagnostic_related_op_t* related_ops,
    iree_host_size_t related_op_count,
    loom_diagnostic_related_location_t* out_related_locations,
    loom_highlight_range_t* out_related_highlights) {
  if (!related_ops || related_op_count == 0) {
    return 0;
  }

  iree_host_size_t related_location_count = 0;
  for (iree_host_size_t i = 0;
       i < related_op_count &&
       related_location_count < LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS;
       ++i) {
    if (!related_ops[i].op) continue;
    loom_source_range_t source_location = {
        .provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE,
    };
    if (!loom_verify_resolve_location(state, related_ops[i].op,
                                      &source_location)) {
      continue;
    }
    loom_diagnostic_related_location_t* related_location =
        &out_related_locations[related_location_count];
    *related_location = (loom_diagnostic_related_location_t){
        .label = related_ops[i].label,
        .source_location = source_location,
    };
    if (loom_diagnostic_field_ref_is_set(related_ops[i].field_ref)) {
      loom_print_field_ref_t printer_field_ref = {0};
      if (loom_verify_print_field_ref(related_ops[i].field_ref,
                                      &printer_field_ref)) {
        const loom_verify_highlight_target_t highlight_target = {
            .printer_field_ref = printer_field_ref,
            .diagnostic_field_ref = related_ops[i].field_ref,
            .param_index = 0,
        };
        related_location->highlight_count =
            loom_collect_source_backed_highlights(
                state->module, related_ops[i].op,
                &related_location->source_location, &highlight_target,
                /*wanted_count=*/1,
                &out_related_highlights[related_location_count],
                /*max_highlight_count=*/1);
        if (related_location->highlight_count > 0) {
          related_location->highlights =
              &out_related_highlights[related_location_count];
        }
      }
    }
    ++related_location_count;
  }
  return related_location_count;
}

// Emits one structured diagnostic request through the configured sink.
//
// Source resolution strategy:
//   1. Try the configured source resolver (original source text).
//   2. If that fails, print the op to text and use the printed
//      representation as the diagnostic source.
//
// Per-token highlighting is derived automatically from structured field refs
// attached to diagnostic params. Those refs are passed to the printer's field
// callback, which records byte ranges and preserves the originating param index
// for caret output and machine JSON.
static void loom_verify_emit_diagnostic(
    loom_verify_state_t* state, const loom_diagnostic_emission_t* emission) {
  if (!iree_status_is_ok(state->diagnostic_status)) return;

  if (emission->error->severity == LOOM_DIAGNOSTIC_ERROR) {
    ++state->result->error_count;
  } else if (emission->error->severity == LOOM_DIAGNOSTIC_WARNING) {
    ++state->result->warning_count;
  }

  if (!state->sink.fn) return;

  loom_diagnostic_t diagnostic = {0};
  diagnostic.severity = emission->error->severity;
  diagnostic.error = emission->error;
  diagnostic.params = emission->params;
  diagnostic.param_count = emission->param_count;
  diagnostic.emitter = LOOM_EMITTER_VERIFIER;
  diagnostic.origin.provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE;
  diagnostic.source_location.provenance =
      LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE;

  loom_diagnostic_related_location_t
      related_locations[LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS];
  loom_highlight_range_t
      related_highlights[LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS];
  diagnostic.related_location_count = loom_verify_collect_related_locations(
      state, emission->related_ops, emission->related_op_count,
      related_locations, related_highlights);
  if (diagnostic.related_location_count > 0) {
    diagnostic.related_locations = related_locations;
  }

  // Collect structured field refs from params for per-token highlighting.
  loom_verify_highlight_target_t highlight_targets[LOOM_VERIFY_MAX_HIGHLIGHTS];
  iree_host_size_t highlight_target_count = loom_collect_highlight_targets(
      emission->params, emission->param_count, highlight_targets);
  loom_highlight_range_t highlights[LOOM_VERIFY_MAX_HIGHLIGHTS];

  // Try the source resolver first (original source text).
  bool resolved = loom_verify_resolve_location(state, emission->op,
                                               &diagnostic.source_location);
  if (resolved) {
    diagnostic.origin = diagnostic.source_location;
    diagnostic.highlight_count = loom_collect_source_backed_highlights(
        state->module, emission->op, &diagnostic.source_location,
        highlight_targets, highlight_target_count, highlights,
        IREE_ARRAYSIZE(highlights));
    if (diagnostic.highlight_count > 0) {
      diagnostic.highlights = highlights;
    }
  }

  // Fallback: print the op and use the printed text as the source.
  // The printer's field callback records byte ranges for the derived
  // field refs, giving per-token caret underlines.
  iree_string_builder_t op_text_builder;
  loom_highlight_collector_t collector = {0};
  bool printed_op = false;
  if (!resolved && emission->op) {
    iree_string_builder_initialize(state->module->context->allocator,
                                   &op_text_builder);

    collector.wanted_targets = highlight_targets;
    collector.wanted_count = highlight_target_count;
    collector.highlights = highlights;
    collector.highlight_count = 0;

    loom_print_field_callback_t field_callback = {
        .fn = highlight_target_count > 0 ? loom_highlight_field_callback : NULL,
        .user_data = &collector,
    };
    iree_status_t print_status = loom_text_print_operation_with_field_callback(
        state->module, emission->op, &op_text_builder,
        LOOM_TEXT_PRINT_USE_ALIASES, field_callback);
    if (!iree_status_is_ok(print_status)) {
      // Printing failed (OOM, etc.). Use a static fallback so the
      // diagnostic still has something to display with carets.
      iree_status_ignore(print_status);
      iree_string_builder_deinitialize(&op_text_builder);
      static const char kFallback[] = "<failed to print op>";
      loom_source_range_t fallback_range = {
          .provenance = LOOM_SOURCE_PROVENANCE_PRINTED_IR_FALLBACK,
          .filename = IREE_SV("<verifier>"),
          .source = iree_make_string_view(kFallback, sizeof(kFallback) - 1),
          .start = 0,
          .end = sizeof(kFallback) - 1,
          .start_line = 1,
          .start_column = 1,
          .end_line = 1,
          .end_column = (uint32_t)sizeof(kFallback),
      };
      diagnostic.origin = fallback_range;
      diagnostic.source_location = fallback_range;
    } else if (iree_string_builder_size(&op_text_builder) > 0) {
      iree_host_size_t text_length = iree_string_builder_size(&op_text_builder);
      loom_source_range_t op_range = {0};
      op_range.provenance = LOOM_SOURCE_PROVENANCE_PRINTED_IR_FALLBACK;
      op_range.filename = IREE_SV("<verifier>");
      op_range.source = iree_make_string_view(
          iree_string_builder_buffer(&op_text_builder), text_length);
      op_range.start = 0;
      op_range.end = text_length;
      op_range.start_line = 1;
      op_range.start_column = 1;
      op_range.end_line = 1;
      op_range.end_column = (uint32_t)text_length + 1;
      diagnostic.origin = op_range;
      diagnostic.source_location = op_range;
      printed_op = true;

      if (collector.highlight_count > 0) {
        diagnostic.highlights = highlights;
        diagnostic.highlight_count = collector.highlight_count;
      }
    } else {
      iree_string_builder_deinitialize(&op_text_builder);
    }
  }

  iree_status_t emit_status = loom_diagnostic_emit(&state->sink, &diagnostic);

  if (printed_op) {
    iree_string_builder_deinitialize(&op_text_builder);
  }

  loom_verify_record_diagnostic_status(state, emit_status);
}

static void loom_verify_emit_structured(loom_verify_state_t* state,
                                        const loom_op_t* op,
                                        const loom_error_def_t* error,
                                        const loom_diagnostic_param_t* params,
                                        iree_host_size_t param_count) {
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  loom_verify_emit_diagnostic(state, &emission);
}

static iree_status_t loom_verify_diagnostic_emitter_fn(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  loom_verify_state_t* state = (loom_verify_state_t*)user_data;
  loom_verify_emit_diagnostic(state, emission);
  return loom_verify_pending_diagnostic_status(state);
}

static void loom_verify_emit_symbol_definition_diagnostic(
    loom_verify_state_t* state, const loom_op_t* op, loom_symbol_ref_t ref,
    uint8_t symbol_attr_index, const loom_symbol_t* symbol) {
  loom_diagnostic_param_t params[] = {
      loom_verify_param_string_for_diagnostic_field(
          loom_verify_symbol_name(state, ref), LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
          symbol_attr_index),
  };
  loom_diagnostic_related_op_t related_ops[] = {{
      .label = IREE_SV("first definition here"),
      .op = symbol->defining_op,
  }};
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 5),
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .related_ops = related_ops,
      .related_op_count = IREE_ARRAYSIZE(related_ops),
  };
  loom_verify_emit_diagnostic(state, &emission);
}

static iree_status_t loom_verify_symbol_definition(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable) {
  if (!vtable->symbol_def || !vtable->attr_descriptors) {
    return iree_ok_status();
  }
  uint8_t symbol_attr_index = vtable->symbol_def->name_attr_index;
  if (symbol_attr_index >= vtable->attribute_count ||
      symbol_attr_index >= op->attribute_count) {
    return iree_ok_status();
  }
  const loom_attr_descriptor_t* descriptor =
      &vtable->attr_descriptors[symbol_attr_index];
  if (descriptor->attr_kind != LOOM_ATTR_SYMBOL) return iree_ok_status();
  loom_symbol_ref_t ref =
      loom_attr_as_symbol(loom_op_const_attrs(op)[symbol_attr_index]);
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= state->module->symbols.count) {
    return iree_ok_status();
  }
  const loom_symbol_t* symbol = &state->module->symbols.entries[ref.symbol_id];
  if (symbol->defining_op && symbol->defining_op != op) {
    loom_verify_emit_symbol_definition_diagnostic(state, op, ref,
                                                  symbol_attr_index, symbol);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Vtable lookup
//===----------------------------------------------------------------------===//

static const loom_op_vtable_t* loom_verify_lookup_vtable(
    const loom_verify_state_t* state, loom_op_kind_t kind) {
  return loom_context_resolve_op(state->module->context, kind);
}

static bool loom_verify_trait_conflict(loom_trait_flags_t traits,
                                       iree_string_view_t* out_trait_a,
                                       iree_string_view_t* out_trait_b) {
  if (iree_all_bits_set(traits, LOOM_TRAIT_HINT | LOOM_TRAIT_PURE)) {
    *out_trait_a = IREE_SV("HINT");
    *out_trait_b = IREE_SV("PURE");
    return true;
  }
  if (iree_all_bits_set(traits, LOOM_TRAIT_HINT | LOOM_TRAIT_UNKNOWN_EFFECTS)) {
    *out_trait_a = IREE_SV("HINT");
    *out_trait_b = IREE_SV("UNKNOWN_EFFECTS");
    return true;
  }
  if (iree_all_bits_set(traits,
                        LOOM_TRAIT_HINT | LOOM_TRAIT_NON_DETERMINISTIC)) {
    *out_trait_a = IREE_SV("HINT");
    *out_trait_b = IREE_SV("NON_DETERMINISTIC");
    return true;
  }
  if (iree_all_bits_set(traits, LOOM_TRAIT_HINT | LOOM_TRAIT_READS_MEMORY)) {
    *out_trait_a = IREE_SV("HINT");
    *out_trait_b = IREE_SV("READS_MEMORY");
    return true;
  }
  if (iree_all_bits_set(traits, LOOM_TRAIT_HINT | LOOM_TRAIT_WRITES_MEMORY)) {
    *out_trait_a = IREE_SV("HINT");
    *out_trait_b = IREE_SV("WRITES_MEMORY");
    return true;
  }
  if (iree_all_bits_set(traits,
                        LOOM_TRAIT_PURE | LOOM_TRAIT_NON_DETERMINISTIC)) {
    *out_trait_a = IREE_SV("PURE");
    *out_trait_b = IREE_SV("NON_DETERMINISTIC");
    return true;
  }
  if (iree_all_bits_set(traits, LOOM_TRAIT_PURE | LOOM_TRAIT_UNKNOWN_EFFECTS)) {
    *out_trait_a = IREE_SV("PURE");
    *out_trait_b = IREE_SV("UNKNOWN_EFFECTS");
    return true;
  }
  if (iree_all_bits_set(traits, LOOM_TRAIT_PURE | LOOM_TRAIT_UNIQUE_IDENTITY)) {
    *out_trait_a = IREE_SV("PURE");
    *out_trait_b = IREE_SV("UNIQUE_IDENTITY");
    return true;
  }
  if (iree_all_bits_set(traits, LOOM_TRAIT_PURE | LOOM_TRAIT_READS_MEMORY)) {
    *out_trait_a = IREE_SV("PURE");
    *out_trait_b = IREE_SV("READS_MEMORY");
    return true;
  }
  if (iree_all_bits_set(traits, LOOM_TRAIT_PURE | LOOM_TRAIT_WRITES_MEMORY)) {
    *out_trait_a = IREE_SV("PURE");
    *out_trait_b = IREE_SV("WRITES_MEMORY");
    return true;
  }
  if (iree_all_bits_set(traits,
                        LOOM_TRAIT_UNKNOWN_EFFECTS | LOOM_TRAIT_READS_MEMORY)) {
    *out_trait_a = IREE_SV("UNKNOWN_EFFECTS");
    *out_trait_b = IREE_SV("READS_MEMORY");
    return true;
  }
  if (iree_all_bits_set(
          traits, LOOM_TRAIT_UNKNOWN_EFFECTS | LOOM_TRAIT_WRITES_MEMORY)) {
    *out_trait_a = IREE_SV("UNKNOWN_EFFECTS");
    *out_trait_b = IREE_SV("WRITES_MEMORY");
    return true;
  }
  if (iree_all_bits_set(
          traits, LOOM_TRAIT_SAFE_TO_SPECULATE | LOOM_TRAIT_UNKNOWN_EFFECTS)) {
    *out_trait_a = IREE_SV("SAFE_TO_SPECULATE");
    *out_trait_b = IREE_SV("UNKNOWN_EFFECTS");
    return true;
  }
  if (iree_all_bits_set(traits, LOOM_TRAIT_SAFE_TO_SPECULATE |
                                    LOOM_TRAIT_NON_DETERMINISTIC)) {
    *out_trait_a = IREE_SV("SAFE_TO_SPECULATE");
    *out_trait_b = IREE_SV("NON_DETERMINISTIC");
    return true;
  }
  if (iree_all_bits_set(
          traits, LOOM_TRAIT_SAFE_TO_SPECULATE | LOOM_TRAIT_UNIQUE_IDENTITY)) {
    *out_trait_a = IREE_SV("SAFE_TO_SPECULATE");
    *out_trait_b = IREE_SV("UNIQUE_IDENTITY");
    return true;
  }
  if (iree_all_bits_set(traits,
                        LOOM_TRAIT_SAFE_TO_SPECULATE | LOOM_TRAIT_HINT)) {
    *out_trait_a = IREE_SV("SAFE_TO_SPECULATE");
    *out_trait_b = IREE_SV("HINT");
    return true;
  }
  if (iree_all_bits_set(
          traits, LOOM_TRAIT_SAFE_TO_SPECULATE | LOOM_TRAIT_READS_MEMORY)) {
    *out_trait_a = IREE_SV("SAFE_TO_SPECULATE");
    *out_trait_b = IREE_SV("READS_MEMORY");
    return true;
  }
  if (iree_all_bits_set(
          traits, LOOM_TRAIT_SAFE_TO_SPECULATE | LOOM_TRAIT_WRITES_MEMORY)) {
    *out_trait_a = IREE_SV("SAFE_TO_SPECULATE");
    *out_trait_b = IREE_SV("WRITES_MEMORY");
    return true;
  }
  return false;
}

static void loom_verify_op_trait_flags_consistency(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, loom_trait_flags_t traits) {
  iree_string_view_t trait_a = iree_string_view_empty();
  iree_string_view_t trait_b = iree_string_view_empty();
  if (!loom_verify_trait_conflict(traits, &trait_a, &trait_b)) return;
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_vtable_name(vtable)),
      loom_param_string(trait_a),
      loom_param_string(trait_b),
  };
  loom_verify_emit_structured(
      state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 16), params,
      IREE_ARRAYSIZE(params));
}

static void loom_verify_op_declared_trait_consistency(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable) {
  loom_verify_op_trait_flags_consistency(state, op, vtable, vtable->traits);
}

static void loom_verify_op_effective_trait_consistency(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable) {
  loom_trait_flags_t effective_traits =
      loom_op_effective_traits(state->module, op);
  if (effective_traits == vtable->traits) return;
  loom_verify_op_trait_flags_consistency(state, op, vtable, effective_traits);
}

static void loom_verify_func_purity_body_effects(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable) {
  const loom_func_like_vtable_t* func_vtable = vtable->func_like;
  if (!func_vtable) return;
  if (func_vtable->purity_attr_index == LOOM_ATTR_INDEX_NONE) return;
  if (func_vtable->body_region_index == LOOM_REGION_INDEX_NONE) return;
  if (func_vtable->body_region_index >= op->region_count) return;
  const loom_attribute_t* attrs = loom_op_const_attrs(op);
  if (loom_attr_as_enum(attrs[func_vtable->purity_attr_index]) == 0) return;
  loom_region_t* body = loom_op_regions(op)[func_vtable->body_region_index];
  if (!loom_region_has_read_effects(body) &&
      !loom_region_has_write_effects(body)) {
    return;
  }
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_vtable_name(vtable)),
      loom_param_u32(body ? body->read_effect_count : 0),
      loom_param_u32(body ? body->write_effect_count : 0),
  };
  loom_verify_emit_structured(
      state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 17), params,
      IREE_ARRAYSIZE(params));
}

//===----------------------------------------------------------------------===//
// Value type resolution
//===----------------------------------------------------------------------===//

// Resolves a value_id to its type.
static loom_type_t loom_verify_value_type(const loom_verify_state_t* state,
                                          loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID) {
    loom_type_t none = {0};
    return none;
  }
  if (value_id >= state->module->values.count) {
    loom_type_t none = {0};
    return none;
  }
  return loom_module_value_type(state->module, value_id);
}

// Resolves a field reference on an op to a value_id. For operand and
// result categories, returns the value_id at the given index. For
// attr and region categories, returns LOOM_VALUE_ID_INVALID (these
// fields are not values).
static loom_value_id_t loom_verify_resolve_value_field(const loom_op_t* op,
                                                       uint8_t field_ref) {
  uint8_t category = LOOM_FIELD_REF_CATEGORY(field_ref);
  uint8_t index = LOOM_FIELD_REF_INDEX(field_ref);
  switch (category) {
    case LOOM_FIELD_OPERAND:
      if (index < op->operand_count) return loom_op_const_operands(op)[index];
      break;
    case LOOM_FIELD_RESULT:
      if (index < op->result_count) return loom_op_const_results(op)[index];
      break;
    default:
      break;
  }
  return LOOM_VALUE_ID_INVALID;
}

// Returns true if a field reference points to a variadic field.
// A variadic operand has index == fixed_operand_count when the vtable
// has LOOM_OP_VTABLE_VARIADIC_OPERANDS; similarly for results.
static bool loom_verify_is_variadic_field(const loom_op_vtable_t* vtable,
                                          uint8_t field_ref) {
  uint8_t category = LOOM_FIELD_REF_CATEGORY(field_ref);
  uint8_t index = LOOM_FIELD_REF_INDEX(field_ref);
  switch (category) {
    case LOOM_FIELD_OPERAND:
      return (vtable->vtable_flags & LOOM_OP_VTABLE_VARIADIC_OPERANDS) &&
             index == vtable->fixed_operand_count;
    case LOOM_FIELD_RESULT:
      return (vtable->vtable_flags & LOOM_OP_VTABLE_VARIADIC_RESULTS) &&
             index == vtable->fixed_result_count;
    default:
      return false;
  }
}

// Returns the count of a variadic field. For operands, this is
// op->operand_count - vtable->fixed_operand_count. For results, similar.
static uint16_t loom_verify_variadic_count(const loom_op_t* op,
                                           const loom_op_vtable_t* vtable,
                                           uint8_t field_ref) {
  uint8_t category = LOOM_FIELD_REF_CATEGORY(field_ref);
  uint8_t index = LOOM_FIELD_REF_INDEX(field_ref);
  switch (category) {
    case LOOM_FIELD_OPERAND:
      if (index <= vtable->fixed_operand_count) {
        return (uint16_t)(op->operand_count - index);
      }
      break;
    case LOOM_FIELD_RESULT:
      if (index <= vtable->fixed_result_count) {
        return (uint16_t)(op->result_count - index);
      }
      break;
    default:
      break;
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// Structural checks
//===----------------------------------------------------------------------===//

static void loom_verify_op_structure(loom_verify_state_t* state,
                                     const loom_op_t* op,
                                     const loom_op_vtable_t* vtable) {
  iree_string_view_t op_name = loom_op_vtable_name(vtable);

  // Check operand count.
  bool has_variadic_operands =
      (vtable->vtable_flags & LOOM_OP_VTABLE_VARIADIC_OPERANDS) != 0;
  if (has_variadic_operands) {
    if (op->operand_count < vtable->fixed_operand_count) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(op_name),
          loom_param_u32(op->operand_count),
          loom_param_u32(vtable->fixed_operand_count),
      };
      loom_verify_emit_structured(
          state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 1),
          params, IREE_ARRAYSIZE(params));
    }
  } else {
    if (op->operand_count != vtable->fixed_operand_count) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(op_name),
          loom_param_u32(op->operand_count),
          loom_param_u32(vtable->fixed_operand_count),
      };
      loom_verify_emit_structured(
          state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 1),
          params, IREE_ARRAYSIZE(params));
    }
  }

  // Check result count.
  bool has_variadic_results =
      (vtable->vtable_flags & LOOM_OP_VTABLE_VARIADIC_RESULTS) != 0;
  if (has_variadic_results) {
    if (op->result_count < vtable->fixed_result_count) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(op_name),
          loom_param_u32(op->result_count),
          loom_param_u32(vtable->fixed_result_count),
      };
      loom_verify_emit_structured(
          state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 2),
          params, IREE_ARRAYSIZE(params));
    }
  } else {
    if (op->result_count != vtable->fixed_result_count) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(op_name),
          loom_param_u32(op->result_count),
          loom_param_u32(vtable->fixed_result_count),
      };
      loom_verify_emit_structured(
          state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 2),
          params, IREE_ARRAYSIZE(params));
    }
  }

  // Check attribute count.
  if (op->attribute_count != vtable->attribute_count) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(op_name),
        loom_param_u32(op->attribute_count),
        loom_param_u32(vtable->attribute_count),
    };
    loom_verify_emit_structured(
        state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 3),
        params, IREE_ARRAYSIZE(params));
  }

  // Check region count.
  bool has_variadic_regions =
      iree_any_bit_set(vtable->vtable_flags, LOOM_OP_VTABLE_VARIADIC_REGIONS);
  uint8_t minimum_region_count =
      has_variadic_regions && vtable->region_count > 0
          ? (uint8_t)(vtable->region_count - 1)
          : vtable->region_count;
  bool region_count_matches = has_variadic_regions
                                  ? op->region_count >= minimum_region_count
                                  : op->region_count == vtable->region_count;
  if (!region_count_matches) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(op_name),
        loom_param_u32(op->region_count),
        loom_param_u32(minimum_region_count),
    };
    loom_verify_emit_structured(
        state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 4),
        params, IREE_ARRAYSIZE(params));
  }
}

static bool loom_verify_region_directly_contains_block(
    const loom_region_t* region, const loom_block_t* target) {
  if (!region || !target) return false;
  for (uint16_t i = 0; i < region->block_count; ++i) {
    if (loom_region_const_block(region, i) == target) return true;
  }
  return false;
}

static void loom_verify_successor_targets(loom_verify_state_t* state,
                                          const loom_op_t* op,
                                          const loom_op_vtable_t* vtable) {
  if (op->successor_count == 0) return;
  const loom_region_t* parent_region =
      op->parent_block ? op->parent_block->parent_region : NULL;
  iree_string_view_t op_name = loom_op_vtable_name(vtable);
  loom_block_t* const* successors = loom_op_const_successors(op);
  for (uint8_t i = 0; i < op->successor_count; ++i) {
    loom_diagnostic_field_ref_t successor_ref =
        loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_SUCCESSOR, i);
    if (!successors[i]) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(op_name),
          loom_param_with_field_ref(loom_param_u32(i), successor_ref),
      };
      loom_verify_emit_structured(
          state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 23),
          params, IREE_ARRAYSIZE(params));
      continue;
    }
    if (successors[i]->parent_region != parent_region ||
        !loom_verify_region_directly_contains_block(parent_region,
                                                    successors[i])) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(op_name),
          loom_param_with_field_ref(loom_param_u32(i), successor_ref),
      };
      loom_verify_emit_structured(
          state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 24),
          params, IREE_ARRAYSIZE(params));
    }
  }
}

// Resolves a field reference to its author-facing DSL field name when available
// and otherwise falls back to a positional name such as "operand 0".
static iree_string_view_t loom_verify_field_name(const loom_op_vtable_t* vtable,
                                                 uint8_t field_ref,
                                                 char* buffer,
                                                 iree_host_size_t buffer_size) {
  uint8_t category = LOOM_FIELD_REF_CATEGORY(field_ref);
  uint8_t index = LOOM_FIELD_REF_INDEX(field_ref);
  if (category == LOOM_FIELD_OPERAND && vtable->operand_descriptors) {
    bool has_variadic =
        (vtable->vtable_flags & LOOM_OP_VTABLE_VARIADIC_OPERANDS) != 0;
    uint8_t descriptor_count =
        (uint8_t)(vtable->fixed_operand_count + (has_variadic ? 1 : 0));
    if (index < descriptor_count) {
      return loom_bstring_view(vtable->operand_descriptors[index].name);
    }
  } else if (category == LOOM_FIELD_RESULT && vtable->result_descriptors) {
    bool has_variadic =
        (vtable->vtable_flags & LOOM_OP_VTABLE_VARIADIC_RESULTS) != 0;
    uint8_t descriptor_count =
        (uint8_t)(vtable->fixed_result_count + (has_variadic ? 1 : 0));
    if (index < descriptor_count) {
      return loom_bstring_view(vtable->result_descriptors[index].name);
    }
  } else if (category == LOOM_FIELD_ATTR && vtable->attr_descriptors &&
             index < vtable->attribute_count) {
    return loom_bstring_view(vtable->attr_descriptors[index].name);
  }
  const char* prefix = "field";
  if (category == LOOM_FIELD_OPERAND) {
    prefix = "operand";
  } else if (category == LOOM_FIELD_RESULT) {
    prefix = "result";
  } else if (category == LOOM_FIELD_ATTR) {
    prefix = "attribute";
  } else if (category == LOOM_FIELD_REGION) {
    prefix = "region";
  }
  iree_snprintf(buffer, buffer_size, "%s %u", prefix, index);
  return iree_make_cstring_view(buffer);
}

// Like loom_verify_field_name but for an indexed element of a variadic field.
// Produces names like "inputs[2]" or "results[0]".
static iree_string_view_t loom_verify_indexed_field_name(
    const loom_op_vtable_t* vtable, uint8_t field_ref, uint16_t element_index,
    char* buffer, iree_host_size_t buffer_size) {
  char field_name_buffer[64];
  iree_string_view_t field_name = loom_verify_field_name(
      vtable, field_ref, field_name_buffer, sizeof(field_name_buffer));
  iree_snprintf(buffer, buffer_size, "%.*s[%u]", (int)field_name.size,
                field_name.data, element_index);
  return iree_make_cstring_view(buffer);
}

// Resolves an ordinary operand/result value index to the declared field name.
// Variadic values use their declaring field name plus an element index, such as
// "inputs[2]".
static iree_string_view_t loom_verify_value_field_name(
    const loom_op_vtable_t* vtable, uint8_t category, uint16_t value_index,
    char* buffer, iree_host_size_t buffer_size) {
  if (category == LOOM_FIELD_OPERAND && vtable->operand_descriptors) {
    if (value_index < vtable->fixed_operand_count) {
      return loom_bstring_view(vtable->operand_descriptors[value_index].name);
    }
    if (iree_any_bit_set(vtable->vtable_flags,
                         LOOM_OP_VTABLE_VARIADIC_OPERANDS)) {
      uint16_t element_index = value_index - vtable->fixed_operand_count;
      iree_string_view_t field_name = loom_bstring_view(
          vtable->operand_descriptors[vtable->fixed_operand_count].name);
      iree_snprintf(buffer, buffer_size, "%.*s[%u]", (int)field_name.size,
                    field_name.data, element_index);
      return iree_make_cstring_view(buffer);
    }
  } else if (category == LOOM_FIELD_RESULT && vtable->result_descriptors) {
    if (value_index < vtable->fixed_result_count) {
      return loom_bstring_view(vtable->result_descriptors[value_index].name);
    }
    if (iree_any_bit_set(vtable->vtable_flags,
                         LOOM_OP_VTABLE_VARIADIC_RESULTS)) {
      uint16_t element_index = value_index - vtable->fixed_result_count;
      iree_string_view_t field_name = loom_bstring_view(
          vtable->result_descriptors[vtable->fixed_result_count].name);
      iree_snprintf(buffer, buffer_size, "%.*s[%u]", (int)field_name.size,
                    field_name.data, element_index);
      return iree_make_cstring_view(buffer);
    }
  }
  const char* prefix = category == LOOM_FIELD_RESULT ? "result" : "operand";
  iree_snprintf(buffer, buffer_size, "%s %u", prefix, value_index);
  return iree_make_cstring_view(buffer);
}

//===----------------------------------------------------------------------===//
// Type constraint checks
//===----------------------------------------------------------------------===//

static bool loom_verify_attr_kind_matches_descriptor(
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor) {
  if (descriptor->attr_kind == LOOM_ATTR_ANY) {
    return attr.kind > LOOM_ATTR_ABSENT && attr.kind < LOOM_ATTR_ANY;
  }
  return attr.kind == descriptor->attr_kind;
}

static void loom_verify_predicate_list_attr(loom_verify_state_t* state,
                                            const loom_op_t* op,
                                            iree_string_view_t name,
                                            uint8_t attr_index,
                                            loom_attribute_t attr) {
  if (attr.kind != LOOM_ATTR_PREDICATE_LIST) return;
  loom_diagnostic_param_t attr_name_param =
      loom_verify_param_string_for_diagnostic_field(
          name, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, attr_index);
  if (attr.count > 0 && !attr.predicate_list) {
    loom_diagnostic_param_t params[] = {
        attr_name_param,
        loom_param_u32(attr.count),
    };
    loom_verify_emit_structured(
        state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 19),
        params, IREE_ARRAYSIZE(params));
    return;
  }
  for (uint16_t predicate_index = 0; predicate_index < attr.count;
       ++predicate_index) {
    const loom_predicate_t* predicate = &attr.predicate_list[predicate_index];
    const char* predicate_name = loom_predicate_kind_name(predicate->kind);
    if (!predicate_name) {
      loom_diagnostic_param_t params[] = {
          attr_name_param,
          loom_param_u32(predicate_index),
          loom_param_u32(predicate->kind),
          loom_param_u32(LOOM_PREDICATE_COUNT_),
      };
      loom_verify_emit_structured(
          state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 20),
          params, IREE_ARRAYSIZE(params));
      continue;
    }

    uint8_t expected_argument_count =
        loom_predicate_kind_argument_count(predicate->kind);
    if (predicate->arg_count != expected_argument_count) {
      loom_diagnostic_param_t params[] = {
          attr_name_param,
          loom_param_u32(predicate_index),
          loom_param_string(iree_make_cstring_view(predicate_name)),
          loom_param_u32(expected_argument_count),
          loom_param_u32(predicate->arg_count),
      };
      loom_verify_emit_structured(
          state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 21),
          params, IREE_ARRAYSIZE(params));
    }

    uint8_t argument_count = predicate->arg_count;
    if (argument_count > IREE_ARRAYSIZE(predicate->arg_tags)) {
      argument_count = (uint8_t)IREE_ARRAYSIZE(predicate->arg_tags);
    }
    for (uint8_t argument_index = 0; argument_index < argument_count;
         ++argument_index) {
      uint8_t tag = predicate->arg_tags[argument_index];
      if (tag > LOOM_PRED_ARG_NONE && tag < LOOM_PRED_ARG_COUNT_) continue;
      loom_diagnostic_param_t params[] = {
          attr_name_param,
          loom_param_u32(predicate_index),
          loom_param_u32(argument_index),
          loom_param_u32(tag),
          loom_param_u32(LOOM_PRED_ARG_COUNT_),
      };
      loom_verify_emit_structured(
          state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 22),
          params, IREE_ARRAYSIZE(params));
    }
  }
}

static void loom_verify_type_constraints(loom_verify_state_t* state,
                                         const loom_op_t* op,
                                         const loom_op_vtable_t* vtable) {
  // Check operand type constraints.
  if (vtable->operand_descriptors) {
    bool has_variadic =
        (vtable->vtable_flags & LOOM_OP_VTABLE_VARIADIC_OPERANDS) != 0;
    uint8_t descriptor_count =
        vtable->fixed_operand_count + (has_variadic ? 1 : 0);
    for (uint8_t i = 0; i < descriptor_count && i < op->operand_count; ++i) {
      loom_type_constraint_t constraint =
          (loom_type_constraint_t)vtable->operand_descriptors[i]
              .type_constraint;
      if (constraint == LOOM_TYPE_CONSTRAINT_ANY) continue;

      // For variadic operands (the last descriptor when has_variadic),
      // check all remaining operands.
      uint16_t start = i;
      uint16_t end = (has_variadic && i == vtable->fixed_operand_count)
                         ? op->operand_count
                         : (uint16_t)(i + 1);
      for (uint16_t j = start; j < end; ++j) {
        loom_value_id_t value_id = loom_op_const_operands(op)[j];
        loom_type_t type = loom_verify_value_type(state, value_id);
        if (!loom_type_satisfies_constraint(type, constraint)) {
          uint8_t operand_ref = LOOM_FIELD_REF(LOOM_FIELD_OPERAND, i);
          uint16_t element_offset = (uint16_t)(j - i);
          bool operand_is_variadic =
              has_variadic && i == vtable->fixed_operand_count;
          char operand_name_buffer[64];
          iree_string_view_t operand_name =
              operand_is_variadic
                  ? loom_verify_indexed_field_name(
                        vtable, operand_ref, element_offset,
                        operand_name_buffer, sizeof(operand_name_buffer))
                  : loom_verify_field_name(vtable, operand_ref,
                                           operand_name_buffer,
                                           sizeof(operand_name_buffer));
          loom_diagnostic_param_t operand_param =
              operand_is_variadic
                  ? loom_verify_param_string_for_indexed_field(
                        operand_name, operand_ref, element_offset)
                  : loom_verify_param_string_for_field(operand_name,
                                                       operand_ref);
          loom_diagnostic_param_t params[] = {
              operand_param,
              loom_param_type(type),
              loom_param_string(iree_make_cstring_view(
                  loom_type_constraint_name(constraint))),
          };
          loom_verify_emit_structured(
              state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 3),
              params, IREE_ARRAYSIZE(params));
        }
      }
    }
  }

  // Check result type constraints.
  if (vtable->result_descriptors) {
    bool has_variadic =
        (vtable->vtable_flags & LOOM_OP_VTABLE_VARIADIC_RESULTS) != 0;
    uint8_t descriptor_count =
        vtable->fixed_result_count + (has_variadic ? 1 : 0);
    for (uint8_t i = 0; i < descriptor_count && i < op->result_count; ++i) {
      loom_type_constraint_t constraint =
          (loom_type_constraint_t)vtable->result_descriptors[i].type_constraint;
      if (constraint == LOOM_TYPE_CONSTRAINT_ANY) continue;

      uint16_t start = i;
      uint16_t end = (has_variadic && i == vtable->fixed_result_count)
                         ? op->result_count
                         : (uint16_t)(i + 1);
      for (uint16_t j = start; j < end; ++j) {
        loom_value_id_t value_id = loom_op_const_results(op)[j];
        loom_type_t type = loom_verify_value_type(state, value_id);
        if (!loom_type_satisfies_constraint(type, constraint)) {
          uint8_t result_ref = LOOM_FIELD_REF(LOOM_FIELD_RESULT, i);
          uint16_t element_offset = (uint16_t)(j - i);
          bool result_is_variadic =
              has_variadic && i == vtable->fixed_result_count;
          char result_name_buffer[64];
          iree_string_view_t result_name =
              result_is_variadic
                  ? loom_verify_indexed_field_name(
                        vtable, result_ref, element_offset, result_name_buffer,
                        sizeof(result_name_buffer))
                  : loom_verify_field_name(vtable, result_ref,
                                           result_name_buffer,
                                           sizeof(result_name_buffer));
          loom_diagnostic_param_t result_param =
              result_is_variadic
                  ? loom_verify_param_string_for_indexed_field(
                        result_name, result_ref, element_offset)
                  : loom_verify_param_string_for_field(result_name, result_ref);
          loom_diagnostic_param_t params[] = {
              result_param,
              loom_param_type(type),
              loom_param_string(iree_make_cstring_view(
                  loom_type_constraint_name(constraint))),
          };
          loom_verify_emit_structured(
              state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 4),
              params, IREE_ARRAYSIZE(params));
        }
      }
    }
  }

  // Check attribute kinds and enum value ranges.
  if (vtable->attr_descriptors) {
    const loom_attribute_t* attrs = loom_op_attrs(op);
    for (uint8_t i = 0; i < vtable->attribute_count && i < op->attribute_count;
         ++i) {
      const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
      bool optional = (descriptor->flags & LOOM_ATTR_OPTIONAL) != 0;
      if (optional && loom_attr_is_absent(attrs[i])) continue;
      iree_string_view_t attr_name = loom_bstring_view(descriptor->name);
      if (!loom_verify_attr_kind_matches_descriptor(attrs[i], descriptor)) {
        loom_diagnostic_param_t params[] = {
            loom_verify_param_string_for_diagnostic_field(
                attr_name, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, i),
            loom_param_u32(attrs[i].kind),
            loom_param_u32(descriptor->attr_kind),
        };
        loom_verify_emit_structured(
            state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 5), params,
            IREE_ARRAYSIZE(params));
      }
      if (attrs[i].kind == LOOM_ATTR_ENUM && descriptor->enum_case_count > 0 &&
          (descriptor->flags & LOOM_ATTR_OPEN_ENUM) == 0) {
        uint8_t case_index = (uint8_t)attrs[i].raw;
        if (case_index >= descriptor->enum_case_count) {
          loom_diagnostic_param_t params[] = {
              loom_verify_param_string_for_diagnostic_field(
                  attr_name, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, i),
              loom_param_u32(case_index),
              loom_param_u32(descriptor->enum_case_count),
          };
          loom_verify_emit_structured(
              state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 10),
              params, IREE_ARRAYSIZE(params));
        }
      }
      loom_verify_predicate_list_attr(state, op, attr_name, i, attrs[i]);
    }
  }
}

//===----------------------------------------------------------------------===//
// Operand dictionary representation checks
//===----------------------------------------------------------------------===//

static iree_string_view_t loom_verify_attr_descriptor_name(
    const loom_op_vtable_t* vtable, uint16_t attr_index) {
  if (!vtable->attr_descriptors || attr_index >= vtable->attribute_count) {
    return IREE_SV("operand dictionary names");
  }
  return loom_bstring_view(vtable->attr_descriptors[attr_index].name);
}

static void loom_verify_emit_operand_dict_count_mismatch(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, uint16_t attr_index, uint16_t names_count,
    uint16_t operand_count) {
  iree_string_view_t attr_name =
      loom_verify_attr_descriptor_name(vtable, attr_index);
  loom_diagnostic_param_t params[] = {
      loom_verify_param_string_for_diagnostic_field(
          attr_name, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, attr_index),
      loom_param_u32(names_count),
      loom_param_string(IREE_SV("operand dictionary operands")),
      loom_param_u32(operand_count),
  };
  loom_verify_emit_structured(
      state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 13), params,
      IREE_ARRAYSIZE(params));
}

static void loom_verify_emit_operand_dict_attr_violation(
    loom_verify_state_t* state, const loom_op_t* op,
    iree_string_view_t attr_name, uint16_t attr_index, int64_t actual_value,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_verify_param_string_for_diagnostic_field(
          attr_name, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, attr_index),
      loom_param_i64(actual_value),
      loom_param_string(expected_constraint),
  };
  loom_verify_emit_structured(
      state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 14), params,
      IREE_ARRAYSIZE(params));
}

static void loom_verify_operand_dicts(loom_verify_state_t* state,
                                      const loom_op_t* op,
                                      const loom_op_vtable_t* vtable) {
  if (!vtable->format_elements) return;
  for (uint16_t element_index = 0; element_index < vtable->format_element_count;
       ++element_index) {
    const loom_format_element_t* element =
        &vtable->format_elements[element_index];
    if (element->kind != LOOM_FORMAT_KIND_OPERAND_DICT) continue;

    uint16_t operand_start = element->field_index;
    uint16_t attr_index = element->data;
    if (operand_start > op->operand_count ||
        attr_index >= op->attribute_count) {
      continue;
    }
    uint16_t operand_count = (uint16_t)(op->operand_count - operand_start);
    loom_attribute_t names_attr = loom_op_attrs(op)[attr_index];
    if (loom_attr_is_absent(names_attr)) {
      if (operand_count != 0) {
        loom_verify_emit_operand_dict_count_mismatch(
            state, op, vtable, attr_index, 0, operand_count);
      }
      continue;
    }
    if (names_attr.kind != LOOM_ATTR_DICT) continue;
    if (names_attr.count != operand_count) {
      loom_verify_emit_operand_dict_count_mismatch(
          state, op, vtable, attr_index, names_attr.count, operand_count);
      continue;
    }
    if (names_attr.count == 0) continue;

    iree_string_view_t attr_name =
        loom_verify_attr_descriptor_name(vtable, attr_index);
    if (!names_attr.dict_entries) {
      loom_verify_emit_operand_dict_attr_violation(
          state, op, attr_name, attr_index, names_attr.count,
          IREE_SV("non-null dictionary entries"));
      continue;
    }

    for (uint16_t i = 0; i < names_attr.count; ++i) {
      const loom_named_attr_t* entry = &names_attr.dict_entries[i];
      if (entry->name_id == LOOM_STRING_ID_INVALID ||
          entry->name_id >= state->module->strings.count) {
        loom_verify_emit_operand_dict_attr_violation(
            state, op, attr_name, attr_index, entry->name_id,
            IREE_SV("interned key string id"));
        continue;
      }
      iree_string_view_t key_name =
          state->module->strings.entries[entry->name_id];
      if (i > 0) {
        const loom_named_attr_t* previous_entry =
            &names_attr.dict_entries[i - 1];
        if (previous_entry->name_id < state->module->strings.count) {
          iree_string_view_t previous_key_name =
              state->module->strings.entries[previous_entry->name_id];
          if (iree_string_view_compare(previous_key_name, key_name) >= 0) {
            loom_verify_emit_operand_dict_attr_violation(
                state, op, attr_name, attr_index, i,
                IREE_SV("canonical sorted unique keys"));
          }
        }
      }
      if (entry->value.kind != LOOM_ATTR_I64) {
        loom_diagnostic_param_t params[] = {
            loom_verify_param_string_for_diagnostic_field(
                key_name, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, attr_index),
            loom_param_u32(entry->value.kind),
            loom_param_u32(LOOM_ATTR_I64),
        };
        loom_verify_emit_structured(
            state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 5), params,
            IREE_ARRAYSIZE(params));
        continue;
      }
      int64_t ordinal = entry->value.i64;
      if (ordinal < 0 || ordinal >= operand_count) {
        loom_verify_emit_operand_dict_attr_violation(
            state, op, key_name, attr_index, ordinal,
            IREE_SV("operand ordinal in range"));
        continue;
      }
      for (uint16_t j = 0; j < i; ++j) {
        const loom_named_attr_t* previous_entry = &names_attr.dict_entries[j];
        if (previous_entry->value.kind == LOOM_ATTR_I64 &&
            previous_entry->value.i64 == ordinal) {
          loom_verify_emit_operand_dict_attr_violation(
              state, op, key_name, attr_index, ordinal,
              IREE_SV("unique operand ordinal"));
          break;
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Semantic constraint interpreter
//===----------------------------------------------------------------------===//

// Returns true if the specified property of two types is equal.
static bool loom_constraint_property_equals(
    loom_type_t a, loom_type_t b, loom_constraint_property_t property) {
  switch ((enum loom_constraint_property_e)property) {
    case LOOM_PROPERTY_TYPE:
      return memcmp(&a, &b, sizeof(loom_type_t)) == 0;
    case LOOM_PROPERTY_KIND:
      return loom_type_kind(a) == loom_type_kind(b);
    case LOOM_PROPERTY_ELEMENT_TYPE:
      return loom_type_element_type(a) == loom_type_element_type(b);
    case LOOM_PROPERTY_ENCODING:
      return loom_type_encoding_equals(a, b);
    case LOOM_PROPERTY_SHAPE:
      return loom_type_shape_equals(a, b);
    case LOOM_PROPERTY_RANK:
      return loom_type_rank(a) == loom_type_rank(b);
    default:
      return false;
  }
}

// Default error defs for PAIRWISE_EQ, indexed by property.
static const loom_error_def_t* loom_pairwise_eq_default_error(
    loom_constraint_property_t property) {
  switch ((enum loom_constraint_property_e)property) {
    case LOOM_PROPERTY_TYPE:
      return loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 1);
    case LOOM_PROPERTY_KIND:
      return loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 1);
    case LOOM_PROPERTY_ELEMENT_TYPE:
      return loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 2);
    case LOOM_PROPERTY_ENCODING:
      return loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 1);
    case LOOM_PROPERTY_SHAPE:
      return loom_error_def_lookup(LOOM_ERROR_DOMAIN_SHAPE, 2);
    case LOOM_PROPERTY_RANK:
      return loom_error_def_lookup(LOOM_ERROR_DOMAIN_SHAPE, 1);
    default:
      return loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 1);
  }
}

static const loom_error_def_t* loom_verify_constraint_error_or(
    const loom_constraint_t* constraint,
    const loom_error_def_t* default_error) {
  const loom_error_def_t* error =
      loom_error_def_lookup_ref((loom_error_ref_t)constraint->error_ref);
  return error ? error : default_error;
}

// Builds diagnostic params for a pairwise property mismatch.
// Different properties produce different param schemas:
//   TYPE/KIND: (name_a, type_a, name_b, type_b)
//   ELEMENT_TYPE: (name_a, element_type_a, name_b, element_type_b)
//   ENCODING: (name_a, name_b)
//   SHAPE: (name_a, name_b)
//   RANK: (name_a, rank_a, name_b, rank_b)
static void loom_verify_emit_pairwise_mismatch(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint,
    loom_type_t type_a, loom_type_t type_b, uint8_t ref_a, uint8_t ref_b) {
  char name_a_buffer[32];
  char name_b_buffer[32];
  iree_string_view_t name_a = loom_verify_field_name(
      vtable, ref_a, name_a_buffer, sizeof(name_a_buffer));
  iree_string_view_t name_b = loom_verify_field_name(
      vtable, ref_b, name_b_buffer, sizeof(name_b_buffer));
  const loom_error_def_t* error = loom_verify_constraint_error_or(
      constraint, loom_pairwise_eq_default_error(constraint->property));

  switch ((enum loom_constraint_property_e)constraint->property) {
    case LOOM_PROPERTY_TYPE: {
      loom_diagnostic_param_t params[] = {
          loom_verify_param_string_for_field(name_a, ref_a),
          loom_param_type(type_a),
          loom_verify_param_string_for_field(name_b, ref_b),
          loom_param_type(type_b),
      };
      loom_verify_emit_structured(
          state, op, error, params,
          error->param_count < 4 ? error->param_count : 4);
      break;
    }
    case LOOM_PROPERTY_KIND: {
      loom_diagnostic_param_t params[] = {
          loom_verify_param_string_for_field(name_a, ref_a),
          loom_param_type(type_a),
          loom_verify_param_string_for_field(name_b, ref_b),
          loom_param_type(type_b),
      };
      loom_verify_emit_structured(
          state, op, error, params,
          error->param_count < 4 ? error->param_count : 4);
      break;
    }
    case LOOM_PROPERTY_ELEMENT_TYPE: {
      loom_type_t element_a = loom_type_scalar(loom_type_element_type(type_a));
      loom_type_t element_b = loom_type_scalar(loom_type_element_type(type_b));
      loom_diagnostic_param_t params[] = {
          loom_verify_param_string_for_field(name_a, ref_a),
          loom_param_type(element_a),
          loom_verify_param_string_for_field(name_b, ref_b),
          loom_param_type(element_b),
      };
      loom_verify_emit_structured(
          state, op, error, params,
          error->param_count < 4 ? error->param_count : 4);
      break;
    }
    case LOOM_PROPERTY_ENCODING:
    case LOOM_PROPERTY_SHAPE: {
      loom_diagnostic_param_t params[] = {
          loom_verify_param_string_for_field(name_a, ref_a),
          loom_verify_param_string_for_field(name_b, ref_b),
      };
      loom_verify_emit_structured(
          state, op, error, params,
          error->param_count < 2 ? error->param_count : 2);
      break;
    }
    case LOOM_PROPERTY_RANK: {
      loom_diagnostic_param_t params[] = {
          loom_verify_param_string_for_field(name_a, ref_a),
          loom_param_i64(loom_type_rank(type_a)),
          loom_verify_param_string_for_field(name_b, ref_b),
          loom_param_i64(loom_type_rank(type_b)),
      };
      loom_verify_emit_structured(
          state, op, error, params,
          error->param_count < 4 ? error->param_count : 4);
      break;
    }
    default:
      break;
  }
}

static bool loom_verify_region_entry_yield(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, uint8_t region_index,
    uint16_t* out_yield_count, const loom_value_id_t** out_yield_operands);

// Resolves a variadic value field reference to its value array and
// element count. Returns NULL with |*out_count| = 0 if the field is
// not a value-bearing variadic (e.g., it points at an attr or region,
// or its start index is past the op's operand/result range).
static const loom_value_id_t* loom_verify_resolve_variadic_field(
    const loom_op_t* op, uint8_t field_ref, uint16_t* out_count) {
  uint8_t category = LOOM_FIELD_REF_CATEGORY(field_ref);
  uint8_t start_index = LOOM_FIELD_REF_INDEX(field_ref);
  if (category == LOOM_FIELD_OPERAND && start_index <= op->operand_count) {
    *out_count = (uint16_t)(op->operand_count - start_index);
    return loom_op_const_operands(op) + start_index;
  }
  if (category == LOOM_FIELD_RESULT && start_index <= op->result_count) {
    *out_count = (uint16_t)(op->result_count - start_index);
    return loom_op_const_results(op) + start_index;
  }
  *out_count = 0;
  return NULL;
}

// Emits a pairwise property mismatch where one side is either a scalar
// field reference or an indexed element of a variadic field, and the
// other side is always an indexed element of a variadic field.
// Centralizes the four-param diagnostic format used by PAIRWISE_EQ
// and VARIADIC_MATCH for type/element-type/encoding/shape/rank
// comparisons. The caller passes the error explicitly because some
// callers (e.g., VARIADIC_MATCH) store a different default error on
// the constraint than the type-mismatch path needs.
static void loom_verify_emit_indexed_pairwise_mismatch(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_error_def_t* error,
    uint8_t ref_a, bool a_is_indexed, uint16_t a_element_index,
    loom_type_t type_a, uint8_t ref_b, uint16_t b_element_index,
    loom_type_t type_b) {
  char name_a_buffer[32];
  char name_b_buffer[32];
  iree_string_view_t name_a =
      a_is_indexed
          ? loom_verify_indexed_field_name(vtable, ref_a, a_element_index,
                                           name_a_buffer, sizeof(name_a_buffer))
          : loom_verify_field_name(vtable, ref_a, name_a_buffer,
                                   sizeof(name_a_buffer));
  iree_string_view_t name_b = loom_verify_indexed_field_name(
      vtable, ref_b, b_element_index, name_b_buffer, sizeof(name_b_buffer));
  loom_diagnostic_param_t params[] = {
      a_is_indexed ? loom_verify_param_string_for_indexed_field(name_a, ref_a,
                                                                a_element_index)
                   : loom_verify_param_string_for_field(name_a, ref_a),
      loom_param_type(type_a),
      loom_verify_param_string_for_indexed_field(name_b, ref_b,
                                                 b_element_index),
      loom_param_type(type_b),
  };
  loom_verify_emit_structured(state, op, error, params,
                              error->param_count < 4 ? error->param_count : 4);
}

//===----------------------------------------------------------------------===//
// Constraint relation handlers
//===----------------------------------------------------------------------===//
//
// One handler per loom_constraint_relation_e value. All handlers share
// the same signature so they can be dispatched through the table at
// the bottom of this section. Handlers must validate their own
// constraint args (arg_count, field categories) and silently return on
// malformed inputs — structural verification runs first, so any
// malformed op has already been diagnosed by the time semantic
// constraints are evaluated.
//
// To add a new relation:
//   1. Add the LOOM_RELATION_* enum value in op_defs.h with a doc
//      comment describing the check.
//   2. Add the name string in loom_constraint_relation_name (op_defs.c).
//   3. Add a handler here following the same pattern.
//   4. Add the handler to kVerifyRelationFns below.
//   5. Add the corresponding Constraint constructor in dsl.py and the
//      mapping in c_tables.py CONSTRAINT_MAP.

// PAIRWISE_EQ: every element of every listed field has the same
// property as the first element of the first field. Variadic fields
// are walked elementwise. Args: 1+ value fields. Stops on the first
// mismatch to avoid cascading errors.
static void loom_verify_relation_pairwise_eq(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return;
  loom_value_id_t first_id =
      loom_verify_resolve_value_field(op, constraint->args[0]);
  if (first_id == LOOM_VALUE_ID_INVALID) return;
  loom_type_t first_type = loom_verify_value_type(state, first_id);
  uint8_t first_ref = constraint->args[0];
  bool first_is_variadic =
      loom_verify_is_variadic_field(vtable, constraint->args[0]);

  // PAIRWISE_EQ stores the type-mismatch error as a compact ref on the
  // constraint (Python's SameType, SameElementType, etc.), so honor it when
  // present and fall back to the property's default otherwise.
  const loom_error_def_t* type_error = loom_verify_constraint_error_or(
      constraint, loom_pairwise_eq_default_error(constraint->property));

  // Check remaining elements within the first arg's variadic range.
  if (first_is_variadic) {
    uint16_t count = 0;
    const loom_value_id_t* values =
        loom_verify_resolve_variadic_field(op, first_ref, &count);
    for (uint16_t i = 1; i < count; ++i) {
      loom_type_t other_type = loom_verify_value_type(state, values[i]);
      if (loom_constraint_property_equals(first_type, other_type,
                                          constraint->property)) {
        continue;
      }
      loom_verify_emit_indexed_pairwise_mismatch(
          state, op, vtable, type_error, first_ref, /*a_is_indexed=*/true, 0,
          first_type, first_ref, i, other_type);
      return;
    }
  }

  // Check subsequent constraint args against the reference type.
  for (uint8_t i = 1; i < constraint->arg_count; ++i) {
    uint8_t arg_ref = constraint->args[i];
    if (!loom_verify_is_variadic_field(vtable, arg_ref)) {
      loom_value_id_t other_id = loom_verify_resolve_value_field(op, arg_ref);
      loom_type_t other_type = loom_verify_value_type(state, other_id);
      if (loom_constraint_property_equals(first_type, other_type,
                                          constraint->property)) {
        continue;
      }
      loom_verify_emit_pairwise_mismatch(state, op, vtable, constraint,
                                         first_type, other_type, first_ref,
                                         arg_ref);
      return;
    }
    uint16_t count = 0;
    const loom_value_id_t* values =
        loom_verify_resolve_variadic_field(op, arg_ref, &count);
    for (uint16_t j = 0; j < count; ++j) {
      loom_type_t other_type = loom_verify_value_type(state, values[j]);
      if (loom_constraint_property_equals(first_type, other_type,
                                          constraint->property)) {
        continue;
      }
      loom_verify_emit_indexed_pairwise_mismatch(
          state, op, vtable, type_error, first_ref,
          /*a_is_indexed=*/first_is_variadic, 0, first_type, arg_ref, j,
          other_type);
      return;
    }
  }
}

// ALL_SAME: every element of a single variadic value field shares the
// same property as the first element. Args: 1 variadic value field.
// Stops on the first mismatch.
static void loom_verify_relation_all_same(loom_verify_state_t* state,
                                          const loom_op_t* op,
                                          const loom_op_vtable_t* vtable,
                                          const loom_constraint_t* constraint) {
  if (constraint->arg_count < 1) return;
  uint16_t count = 0;
  const loom_value_id_t* values =
      loom_verify_resolve_variadic_field(op, constraint->args[0], &count);
  if (count <= 1) return;
  loom_type_t first_type = loom_verify_value_type(state, values[0]);
  for (uint16_t i = 1; i < count; ++i) {
    loom_type_t other_type = loom_verify_value_type(state, values[i]);
    if (loom_constraint_property_equals(first_type, other_type,
                                        constraint->property)) {
      continue;
    }
    const loom_error_def_t* error = loom_verify_constraint_error_or(
        constraint, loom_error_def_lookup(LOOM_ERROR_DOMAIN_SHAPE, 3));
    loom_diagnostic_param_t params[] = {
        loom_param_type(first_type),
        loom_param_u32(i),
        loom_param_type(other_type),
    };
    loom_verify_emit_structured(
        state, op, error, params,
        error->param_count < 3 ? error->param_count : 3);
    return;
  }
}

// FIELD_SATISFIES: every element of every listed value field satisfies
// the type constraint stored in the constraint property slot. Args:
// 1+ value fields.
static void loom_verify_relation_field_satisfies(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 1) return;
  loom_type_constraint_t expected =
      (loom_type_constraint_t)constraint->property;
  if (expected >= LOOM_TYPE_CONSTRAINT_COUNT_) return;

  for (uint8_t i = 0; i < constraint->arg_count; ++i) {
    uint8_t field_ref = constraint->args[i];
    if (!loom_verify_is_variadic_field(vtable, field_ref)) {
      loom_value_id_t value_id = loom_verify_resolve_value_field(op, field_ref);
      if (value_id == LOOM_VALUE_ID_INVALID) continue;
      loom_type_t value_type = loom_verify_value_type(state, value_id);
      if (loom_type_satisfies_constraint(value_type, expected)) continue;

      char name_buffer[32];
      iree_string_view_t field_name = loom_verify_field_name(
          vtable, field_ref, name_buffer, sizeof(name_buffer));
      const loom_error_def_t* error =
          LOOM_FIELD_REF_CATEGORY(field_ref) == LOOM_FIELD_RESULT
              ? loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 4)
              : loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 3);
      loom_diagnostic_param_t params[] = {
          loom_verify_param_string_for_field(field_name, field_ref),
          loom_param_type(value_type),
          loom_param_string(
              iree_make_cstring_view(loom_type_constraint_name(expected))),
      };
      loom_verify_emit_structured(
          state, op, error, params,
          error->param_count < 3 ? error->param_count : 3);
      return;
    }

    uint16_t count = 0;
    const loom_value_id_t* values =
        loom_verify_resolve_variadic_field(op, field_ref, &count);
    for (uint16_t j = 0; j < count; ++j) {
      loom_type_t value_type = loom_verify_value_type(state, values[j]);
      if (loom_type_satisfies_constraint(value_type, expected)) continue;

      char name_buffer[32];
      iree_string_view_t field_name = loom_verify_indexed_field_name(
          vtable, field_ref, j, name_buffer, sizeof(name_buffer));
      const loom_error_def_t* error =
          LOOM_FIELD_REF_CATEGORY(field_ref) == LOOM_FIELD_RESULT
              ? loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 4)
              : loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 3);
      loom_diagnostic_param_t params[] = {
          loom_verify_param_string_for_indexed_field(field_name, field_ref, j),
          loom_param_type(value_type),
          loom_param_string(
              iree_make_cstring_view(loom_type_constraint_name(expected))),
      };
      loom_verify_emit_structured(
          state, op, error, params,
          error->param_count < 3 ? error->param_count : 3);
      return;
    }
  }
}

// REGION_ARGS_SATISFY: every entry block argument of a region satisfies the
// type constraint stored in the constraint property slot. Args: (region field).
static void loom_verify_relation_region_args_satisfy(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 1) return;
  loom_type_constraint_t expected =
      (loom_type_constraint_t)constraint->property;
  if (expected >= LOOM_TYPE_CONSTRAINT_COUNT_) return;
  uint8_t region_ref = constraint->args[0];
  if (LOOM_FIELD_REF_CATEGORY(region_ref) != LOOM_FIELD_REGION) return;
  uint8_t region_index = LOOM_FIELD_REF_INDEX(region_ref);
  if (region_index >= op->region_count) return;
  loom_region_t* region = loom_op_regions(op)[region_index];
  if (!region || region->block_count == 0) return;
  loom_block_t* entry = loom_region_entry_block(region);

  char region_name_buffer[32];
  iree_string_view_t region_name = loom_verify_field_name(
      vtable, region_ref, region_name_buffer, sizeof(region_name_buffer));
  for (uint16_t i = 0; i < entry->arg_count; ++i) {
    loom_type_t argument_type =
        loom_verify_value_type(state, loom_block_arg_id(entry, i));
    if (loom_type_satisfies_constraint(argument_type, expected)) continue;

    char argument_name_buffer[64];
    iree_snprintf(argument_name_buffer, sizeof(argument_name_buffer),
                  "%.*s.args[%u]", (int)region_name.size, region_name.data, i);
    const loom_error_def_t* error = loom_verify_constraint_error_or(
        constraint, loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 14));
    loom_diagnostic_param_t params[] = {
        loom_verify_param_string_for_field(
            iree_make_cstring_view(argument_name_buffer), region_ref),
        loom_param_type(argument_type),
        loom_param_string(
            iree_make_cstring_view(loom_type_constraint_name(expected))),
    };
    loom_verify_emit_structured(
        state, op, error, params,
        error->param_count < 3 ? error->param_count : 3);
    return;
  }
}

static bool loom_verify_query_element_bit_width(loom_type_t type,
                                                int32_t* out_bit_width) {
  if (!loom_type_is_scalar(type) && !loom_type_is_shaped(type)) return false;
  int32_t bit_width = loom_scalar_type_bitwidth(loom_type_element_type(type));
  if (bit_width <= 0) return false;
  *out_bit_width = bit_width;
  return true;
}

static bool loom_verify_resolve_i64_attr_field(const loom_op_t* op,
                                               uint8_t attr_ref,
                                               int64_t* out_value) {
  if (LOOM_FIELD_REF_CATEGORY(attr_ref) != LOOM_FIELD_ATTR) return false;
  uint8_t attr_index = LOOM_FIELD_REF_INDEX(attr_ref);
  if (attr_index >= op->attribute_count) return false;
  loom_attribute_t attr = loom_op_attrs(op)[attr_index];
  if (attr.kind != LOOM_ATTR_I64) return false;
  *out_value = loom_attr_as_i64(attr);
  return true;
}

static bool loom_verify_resolve_attr_field(const loom_op_t* op,
                                           uint8_t attr_ref,
                                           loom_attribute_t* out_attr) {
  if (LOOM_FIELD_REF_CATEGORY(attr_ref) != LOOM_FIELD_ATTR) return false;
  uint8_t attr_index = LOOM_FIELD_REF_INDEX(attr_ref);
  if (attr_index >= op->attribute_count) return false;
  *out_attr = loom_op_attrs(op)[attr_index];
  return true;
}

static void loom_verify_emit_i64_attr_constraint(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, uint8_t attr_ref, int64_t actual_value,
    iree_string_view_t expected_constraint) {
  char attr_name_buffer[32];
  iree_string_view_t attr_name = loom_verify_field_name(
      vtable, attr_ref, attr_name_buffer, sizeof(attr_name_buffer));
  loom_diagnostic_param_t params[] = {
      loom_verify_param_string_for_field(attr_name, attr_ref),
      loom_param_i64(actual_value),
      loom_param_string(expected_constraint),
  };
  loom_verify_emit_structured(
      state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 14), params,
      IREE_ARRAYSIZE(params));
}

static void loom_verify_emit_attr_kind_mismatch(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, uint8_t attr_ref,
    const loom_error_def_t* error, loom_attr_kind_t actual_kind,
    loom_attr_kind_t expected_kind) {
  char attr_name_buffer[32];
  iree_string_view_t attr_name = loom_verify_field_name(
      vtable, attr_ref, attr_name_buffer, sizeof(attr_name_buffer));
  if (!error) error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 5);
  loom_diagnostic_param_t params[] = {
      loom_verify_param_string_for_field(attr_name, attr_ref),
      loom_param_u32(actual_kind),
      loom_param_u32(expected_kind),
  };
  loom_verify_emit_structured(state, op, error, params,
                              error->param_count < 3 ? error->param_count : 3);
}

static void loom_verify_emit_value_field_constraint(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, uint8_t field_ref, loom_type_t actual_type,
    iree_string_view_t expected_constraint) {
  char field_name_buffer[32];
  iree_string_view_t field_name = loom_verify_field_name(
      vtable, field_ref, field_name_buffer, sizeof(field_name_buffer));
  const loom_error_def_t* error =
      LOOM_FIELD_REF_CATEGORY(field_ref) == LOOM_FIELD_RESULT
          ? loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 4)
          : loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 3);
  loom_diagnostic_param_t params[] = {
      loom_verify_param_string_for_field(field_name, field_ref),
      loom_param_type(actual_type),
      loom_param_string(expected_constraint),
  };
  loom_verify_emit_structured(state, op, error, params,
                              error->param_count < 3 ? error->param_count : 3);
}

// ATTR_MATCHES_ELEMENT_TYPE: an attribute literal payload kind matches the
// scalar element type of a value field. Args: (attr field, value field).
static void loom_verify_relation_attr_matches_element_type(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return;
  if (constraint->property != LOOM_PROPERTY_ELEMENT_TYPE) return;
  uint8_t attr_ref = constraint->args[0];
  uint8_t field_ref = constraint->args[1];
  if (loom_verify_is_variadic_field(vtable, field_ref)) return;

  loom_attribute_t attr = {0};
  loom_value_id_t value_id = loom_verify_resolve_value_field(op, field_ref);
  if (!loom_verify_resolve_attr_field(op, attr_ref, &attr) ||
      value_id == LOOM_VALUE_ID_INVALID || loom_attr_is_absent(attr)) {
    return;
  }

  loom_type_t value_type = loom_verify_value_type(state, value_id);
  if (!loom_type_is_scalar(value_type) && !loom_type_is_shaped(value_type)) {
    return;
  }

  loom_attr_kind_t expected_kind = LOOM_ATTR_ANY;
  if (loom_attr_matches_scalar_type(attr, loom_type_element_type(value_type),
                                    &expected_kind)) {
    return;
  }
  loom_verify_emit_attr_kind_mismatch(
      state, op, vtable, attr_ref,
      loom_verify_constraint_error_or(constraint, NULL), attr.kind,
      expected_kind);
}

// ATTR_I64_PREDICATE: an i64 attribute satisfies a predicate stored in the
// property slot. Args: (i64 attr field).
static void loom_verify_relation_attr_i64_predicate(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 1) return;
  uint8_t attr_ref = constraint->args[0];
  int64_t value = 0;
  if (!loom_verify_resolve_i64_attr_field(op, attr_ref, &value)) return;

  switch ((enum loom_constraint_property_e)constraint->property) {
    case LOOM_PROPERTY_BIT_WIDTH_POSITIVE:
      if (value > 0) return;
      loom_verify_emit_i64_attr_constraint(state, op, vtable, attr_ref, value,
                                           IREE_SV("positive bit width"));
      return;
    default:
      return;
  }
}

// ELEMENT_WIDTH_ORDER: the scalar or shaped element bit width of the first
// value field is strictly greater or less than the second value field. Args:
// (checked value field, reference value field).
static void loom_verify_relation_element_width_order(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return;
  uint8_t field_ref = constraint->args[0];
  uint8_t reference_ref = constraint->args[1];
  if (loom_verify_is_variadic_field(vtable, field_ref) ||
      loom_verify_is_variadic_field(vtable, reference_ref)) {
    return;
  }

  loom_value_id_t value_id = loom_verify_resolve_value_field(op, field_ref);
  loom_value_id_t reference_id =
      loom_verify_resolve_value_field(op, reference_ref);
  if (value_id == LOOM_VALUE_ID_INVALID ||
      reference_id == LOOM_VALUE_ID_INVALID) {
    return;
  }

  loom_type_t value_type = loom_verify_value_type(state, value_id);
  loom_type_t reference_type = loom_verify_value_type(state, reference_id);
  int32_t value_bit_width = 0;
  int32_t reference_bit_width = 0;
  if (!loom_verify_query_element_bit_width(value_type, &value_bit_width) ||
      !loom_verify_query_element_bit_width(reference_type,
                                           &reference_bit_width)) {
    return;
  }

  const char* relation_text = NULL;
  bool relation_matches = false;
  switch ((enum loom_constraint_property_e)constraint->property) {
    case LOOM_PROPERTY_ELEMENT_WIDTH_GREATER_THAN:
      relation_text = "greater than";
      relation_matches = value_bit_width > reference_bit_width;
      break;
    case LOOM_PROPERTY_ELEMENT_WIDTH_LESS_THAN:
      relation_text = "less than";
      relation_matches = value_bit_width < reference_bit_width;
      break;
    default:
      return;
  }
  if (relation_matches) return;

  char field_name_buffer[32];
  char reference_name_buffer[32];
  iree_string_view_t field_name = loom_verify_field_name(
      vtable, field_ref, field_name_buffer, sizeof(field_name_buffer));
  iree_string_view_t reference_name =
      loom_verify_field_name(vtable, reference_ref, reference_name_buffer,
                             sizeof(reference_name_buffer));

  char expected_buffer[96];
  iree_snprintf(expected_buffer, sizeof(expected_buffer),
                "element bit width %s %.*s", relation_text,
                (int)reference_name.size, reference_name.data);
  const loom_error_def_t* error =
      LOOM_FIELD_REF_CATEGORY(field_ref) == LOOM_FIELD_RESULT
          ? loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 4)
          : loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 3);
  loom_diagnostic_param_t params[] = {
      loom_verify_param_string_for_field(field_name, field_ref),
      loom_param_type(value_type),
      loom_param_string(iree_make_cstring_view(expected_buffer)),
  };
  loom_verify_emit_structured(state, op, error, params,
                              error->param_count < 3 ? error->param_count : 3);
}

// ELEMENT_WIDTH_AT_LEAST_ATTR: the scalar or shaped element bit width of the
// first value field is at least the i64 attribute value. Args: (checked value
// field, i64 attr field).
static void loom_verify_relation_element_width_at_least_attr(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return;
  uint8_t field_ref = constraint->args[0];
  uint8_t attr_ref = constraint->args[1];
  if (loom_verify_is_variadic_field(vtable, field_ref)) return;

  loom_value_id_t value_id = loom_verify_resolve_value_field(op, field_ref);
  int64_t required_bit_width = 0;
  if (value_id == LOOM_VALUE_ID_INVALID ||
      !loom_verify_resolve_i64_attr_field(op, attr_ref, &required_bit_width) ||
      required_bit_width <= 0) {
    return;
  }

  loom_type_t value_type = loom_verify_value_type(state, value_id);
  int32_t element_bit_width = 0;
  if (!loom_verify_query_element_bit_width(value_type, &element_bit_width)) {
    return;
  }
  if ((int64_t)element_bit_width >= required_bit_width) return;

  char field_name_buffer[32];
  char attr_name_buffer[32];
  char expected_buffer[96];
  iree_string_view_t field_name = loom_verify_field_name(
      vtable, field_ref, field_name_buffer, sizeof(field_name_buffer));
  iree_string_view_t attr_name = loom_verify_field_name(
      vtable, attr_ref, attr_name_buffer, sizeof(attr_name_buffer));
  iree_snprintf(expected_buffer, sizeof(expected_buffer),
                "element bit width at least %.*s", (int)attr_name.size,
                attr_name.data);
  const loom_error_def_t* error =
      LOOM_FIELD_REF_CATEGORY(field_ref) == LOOM_FIELD_RESULT
          ? loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 4)
          : loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 3);
  loom_diagnostic_param_t params[] = {
      loom_verify_param_string_for_field(field_name, field_ref),
      loom_param_type(value_type),
      loom_param_string(iree_make_cstring_view(expected_buffer)),
  };
  loom_verify_emit_structured(state, op, error, params,
                              error->param_count < 3 ? error->param_count : 3);
}

// BIT_RANGE_WITHIN_ELEMENT_WIDTH: a bit range described by offset and width
// attributes fits within a scalar or shaped element width. Args: (checked value
// field, offset i64 attr field, width i64 attr field).
static void loom_verify_relation_bit_range_within_element_width(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 3) return;
  uint8_t field_ref = constraint->args[0];
  uint8_t offset_ref = constraint->args[1];
  uint8_t width_ref = constraint->args[2];
  if (loom_verify_is_variadic_field(vtable, field_ref)) return;

  loom_value_id_t value_id = loom_verify_resolve_value_field(op, field_ref);
  int64_t offset = 0;
  int64_t width = 0;
  if (value_id == LOOM_VALUE_ID_INVALID ||
      !loom_verify_resolve_i64_attr_field(op, offset_ref, &offset) ||
      !loom_verify_resolve_i64_attr_field(op, width_ref, &width)) {
    return;
  }

  if (offset < 0) {
    loom_verify_emit_i64_attr_constraint(state, op, vtable, offset_ref, offset,
                                         IREE_SV("non-negative bit offset"));
    return;
  }
  if (width <= 0) {
    loom_verify_emit_i64_attr_constraint(state, op, vtable, width_ref, width,
                                         IREE_SV("positive bitfield width"));
    return;
  }

  loom_type_t value_type = loom_verify_value_type(state, value_id);
  int32_t element_bit_width = 0;
  if (!loom_verify_query_element_bit_width(value_type, &element_bit_width)) {
    return;
  }
  if (offset <= element_bit_width && width <= element_bit_width - offset) {
    return;
  }
  loom_verify_emit_i64_attr_constraint(
      state, op, vtable, width_ref, width,
      IREE_SV("bitfield range within storage element width"));
}

typedef struct loom_verify_total_bit_count_expr_t {
  // Product of all static dimensions and the element bit width.
  uint64_t static_factor;
  // Sorted dynamic dimension value IDs participating in the product.
  loom_value_id_t dynamic_dims[LOOM_TYPE_MAX_RANK];
  // Number of entries used in dynamic_dims.
  uint8_t dynamic_dim_count;
} loom_verify_total_bit_count_expr_t;

static void loom_verify_sort_dynamic_dims(loom_value_id_t* dims,
                                          uint8_t dim_count) {
  for (uint8_t i = 1; i < dim_count; ++i) {
    loom_value_id_t value_id = dims[i];
    uint8_t j = i;
    while (j > 0 && dims[j - 1] > value_id) {
      dims[j] = dims[j - 1];
      --j;
    }
    dims[j] = value_id;
  }
}

static bool loom_verify_total_bit_count_expr(
    loom_type_t type, int32_t element_width,
    loom_verify_total_bit_count_expr_t* out_expr) {
  *out_expr = (loom_verify_total_bit_count_expr_t){
      .static_factor = (uint64_t)element_width,
  };
  if (element_width <= 0) return false;
  if (loom_type_is_scalar(type)) return true;
  if (!loom_type_is_shaped(type)) return false;

  uint8_t rank = loom_type_rank(type);
  for (uint8_t i = 0; i < rank; ++i) {
    if (loom_type_dim_is_dynamic_at(type, i)) {
      if (out_expr->dynamic_dim_count >=
          IREE_ARRAYSIZE(out_expr->dynamic_dims)) {
        return false;
      }
      loom_value_id_t dimension_value_id = loom_type_dim_value_id_at(type, i);
      if (dimension_value_id == LOOM_VALUE_ID_INVALID) return false;
      out_expr->dynamic_dims[out_expr->dynamic_dim_count++] =
          dimension_value_id;
      continue;
    }

    int64_t dimension_size = loom_type_dim_static_size_at(type, i);
    if (dimension_size < 0) return false;
    if (dimension_size == 0) {
      out_expr->static_factor = 0;
      out_expr->dynamic_dim_count = 0;
      return true;
    }
    uint64_t dimension_size_u64 = (uint64_t)dimension_size;
    if (out_expr->static_factor > UINT64_MAX / dimension_size_u64) {
      return false;
    }
    out_expr->static_factor *= dimension_size_u64;
  }

  loom_verify_sort_dynamic_dims(out_expr->dynamic_dims,
                                out_expr->dynamic_dim_count);
  return true;
}

static bool loom_verify_total_bit_count_expr_equal(
    const loom_verify_total_bit_count_expr_t* lhs,
    const loom_verify_total_bit_count_expr_t* rhs) {
  if (lhs->static_factor != rhs->static_factor ||
      lhs->dynamic_dim_count != rhs->dynamic_dim_count) {
    return false;
  }
  for (uint8_t i = 0; i < lhs->dynamic_dim_count; ++i) {
    if (lhs->dynamic_dims[i] != rhs->dynamic_dims[i]) return false;
  }
  return true;
}

// TOTAL_BIT_COUNT_EQUAL: two value fields must have the same total bit count
// when expressed as element-bit-width times element count. Dynamic dimensions
// are compared by SSA identity after sorting, so vector<[%m]x2xi8> and
// vector<[%m]xi16> prove equal without needing arithmetic facts. Args:
// (lhs value field, rhs value field).
static void loom_verify_relation_total_bit_count_equal(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return;
  uint8_t lhs_ref = constraint->args[0];
  uint8_t rhs_ref = constraint->args[1];
  if (loom_verify_is_variadic_field(vtable, lhs_ref) ||
      loom_verify_is_variadic_field(vtable, rhs_ref)) {
    return;
  }

  loom_value_id_t lhs_value_id = loom_verify_resolve_value_field(op, lhs_ref);
  loom_value_id_t rhs_value_id = loom_verify_resolve_value_field(op, rhs_ref);
  if (lhs_value_id == LOOM_VALUE_ID_INVALID ||
      rhs_value_id == LOOM_VALUE_ID_INVALID) {
    return;
  }

  loom_type_t lhs_type = loom_verify_value_type(state, lhs_value_id);
  loom_type_t rhs_type = loom_verify_value_type(state, rhs_value_id);
  int32_t lhs_width = 0;
  int32_t rhs_width = 0;
  if (!loom_verify_query_element_bit_width(lhs_type, &lhs_width) ||
      !loom_verify_query_element_bit_width(rhs_type, &rhs_width)) {
    return;
  }

  if (lhs_width == rhs_width && loom_type_shape_equals(lhs_type, rhs_type)) {
    return;
  }

  loom_verify_total_bit_count_expr_t lhs_bit_count = {0};
  loom_verify_total_bit_count_expr_t rhs_bit_count = {0};
  bool counts_match =
      loom_verify_total_bit_count_expr(lhs_type, lhs_width, &lhs_bit_count) &&
      loom_verify_total_bit_count_expr(rhs_type, rhs_width, &rhs_bit_count) &&
      loom_verify_total_bit_count_expr_equal(&lhs_bit_count, &rhs_bit_count);
  if (counts_match) return;

  char lhs_name_buffer[32];
  char expected_buffer[96];
  iree_string_view_t lhs_name = loom_verify_field_name(
      vtable, lhs_ref, lhs_name_buffer, sizeof(lhs_name_buffer));
  iree_snprintf(expected_buffer, sizeof(expected_buffer),
                "provably same total bit count as %.*s", (int)lhs_name.size,
                lhs_name.data);
  loom_verify_emit_value_field_constraint(
      state, op, vtable, rhs_ref, rhs_type,
      iree_make_cstring_view(expected_buffer));
}

static bool loom_verify_static_bit_count(loom_type_t type,
                                         int64_t bit_width_per_element,
                                         bool* out_is_static,
                                         uint64_t* out_bit_count) {
  *out_is_static = false;
  *out_bit_count = 0;
  if (bit_width_per_element < 0) return false;

  uint64_t element_count = 0;
  if (loom_type_is_scalar(type)) {
    element_count = 1;
  } else if (!loom_type_static_element_count(type, &element_count)) {
    return true;
  }

  *out_is_static = true;
  if (bit_width_per_element == 0) return true;
  uint64_t bit_width = (uint64_t)bit_width_per_element;
  if (element_count > UINT64_MAX / bit_width) return false;
  *out_bit_count = element_count * bit_width;
  return true;
}

static iree_string_view_t loom_verify_payload_bit_count_mismatch_constraint(
    loom_constraint_property_t property) {
  switch ((enum loom_constraint_property_e)property) {
    case LOOM_PROPERTY_PACKED_PAYLOAD_BIT_COUNT_MATCHES_STORAGE:
      return IREE_SV(
          "packed payload bit count equal to result storage bit count");
    case LOOM_PROPERTY_UNPACKED_PAYLOAD_BIT_COUNT_MATCHES_STORAGE:
      return IREE_SV(
          "unpacked payload bit count equal to source storage bit count");
    default:
      return IREE_SV("payload bit count equal to storage bit count");
  }
}

// PAYLOAD_BIT_COUNT_MATCHES_STORAGE: a payload field with a fixed element bit
// width stored in an i64 attr must have the same static total bit count as a
// storage value field. Dynamic shapes are accepted here because the relation is
// only intended to catch concrete bitstream-size mistakes without requiring a
// symbolic arithmetic solver. Args: (payload value field, width i64 attr field,
// storage value field, diagnostic value field).
static void loom_verify_relation_payload_bit_count_matches_storage(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 4) return;
  uint8_t payload_ref = constraint->args[0];
  uint8_t width_ref = constraint->args[1];
  uint8_t storage_ref = constraint->args[2];
  uint8_t diagnostic_ref = constraint->args[3];
  if (loom_verify_is_variadic_field(vtable, payload_ref) ||
      loom_verify_is_variadic_field(vtable, storage_ref) ||
      loom_verify_is_variadic_field(vtable, diagnostic_ref)) {
    return;
  }

  loom_value_id_t payload_value_id =
      loom_verify_resolve_value_field(op, payload_ref);
  loom_value_id_t storage_value_id =
      loom_verify_resolve_value_field(op, storage_ref);
  loom_value_id_t diagnostic_value_id =
      loom_verify_resolve_value_field(op, diagnostic_ref);
  int64_t payload_width = 0;
  if (payload_value_id == LOOM_VALUE_ID_INVALID ||
      storage_value_id == LOOM_VALUE_ID_INVALID ||
      diagnostic_value_id == LOOM_VALUE_ID_INVALID ||
      !loom_verify_resolve_i64_attr_field(op, width_ref, &payload_width) ||
      payload_width <= 0) {
    return;
  }

  loom_type_t payload_type = loom_verify_value_type(state, payload_value_id);
  loom_type_t storage_type = loom_verify_value_type(state, storage_value_id);
  loom_type_t diagnostic_type =
      loom_verify_value_type(state, diagnostic_value_id);
  int32_t storage_width = 0;
  if (!loom_verify_query_element_bit_width(storage_type, &storage_width)) {
    return;
  }

  bool payload_bit_count_is_static = false;
  uint64_t payload_bit_count = 0;
  if (!loom_verify_static_bit_count(payload_type, payload_width,
                                    &payload_bit_count_is_static,
                                    &payload_bit_count)) {
    loom_verify_emit_value_field_constraint(
        state, op, vtable, payload_ref, payload_type,
        IREE_SV("representable static payload bit count"));
    return;
  }

  bool storage_bit_count_is_static = false;
  uint64_t storage_bit_count = 0;
  if (!loom_verify_static_bit_count(storage_type, storage_width,
                                    &storage_bit_count_is_static,
                                    &storage_bit_count)) {
    loom_verify_emit_value_field_constraint(
        state, op, vtable, storage_ref, storage_type,
        IREE_SV("representable static storage bit count"));
    return;
  }

  if (!payload_bit_count_is_static || !storage_bit_count_is_static) return;
  if (payload_bit_count == storage_bit_count) return;

  loom_verify_emit_value_field_constraint(
      state, op, vtable, diagnostic_ref, diagnostic_type,
      loom_verify_payload_bit_count_mismatch_constraint(constraint->property));
}

static iree_string_view_t loom_verify_grouped_last_axis_divisibility_constraint(
    int64_t group_size) {
  switch (group_size) {
    case 2:
      return IREE_SV("last axis extent divisible by 2");
    case 4:
      return IREE_SV("last axis extent divisible by 4");
    case 8:
      return IREE_SV("last axis extent divisible by 8");
    default:
      return IREE_SV("last axis extent divisible by group size");
  }
}

static iree_string_view_t loom_verify_grouped_last_axis_result_constraint(
    int64_t group_size) {
  switch (group_size) {
    case 2:
      return IREE_SV(
          "last axis extent equal to source last axis extent divided by 2");
    case 4:
      return IREE_SV(
          "last axis extent equal to source last axis extent divided by 4");
    case 8:
      return IREE_SV(
          "last axis extent equal to source last axis extent divided by 8");
    default:
      return IREE_SV(
          "last axis extent equal to source last axis extent divided by group "
          "size");
  }
}

// LAST_AXIS_GROUPED_BY: result vector shape equals source vector shape with
// the last axis divided by a small static group size stored in the property
// slot. Args: (source vector field, result vector field).
static void loom_verify_relation_last_axis_grouped_by(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2 || constraint->property == 0) return;
  uint8_t source_ref = constraint->args[0];
  uint8_t result_ref = constraint->args[1];
  loom_value_id_t source_id = loom_verify_resolve_value_field(op, source_ref);
  loom_value_id_t result_id = loom_verify_resolve_value_field(op, result_ref);
  if (source_id == LOOM_VALUE_ID_INVALID ||
      result_id == LOOM_VALUE_ID_INVALID) {
    return;
  }

  loom_type_t source_type = loom_verify_value_type(state, source_id);
  loom_type_t result_type = loom_verify_value_type(state, result_id);
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type)) {
    return;
  }

  char source_name_buffer[32];
  char result_name_buffer[32];
  iree_string_view_t source_name = loom_verify_field_name(
      vtable, source_ref, source_name_buffer, sizeof(source_name_buffer));
  iree_string_view_t result_name = loom_verify_field_name(
      vtable, result_ref, result_name_buffer, sizeof(result_name_buffer));

  uint8_t source_rank = loom_type_rank(source_type);
  uint8_t result_rank = loom_type_rank(result_type);
  if (source_rank == 0 || result_rank == 0) return;
  if (result_rank != source_rank) {
    loom_diagnostic_param_t params[] = {
        loom_verify_param_string_for_field(result_name, result_ref),
        loom_param_i64(result_rank),
        loom_verify_param_string_for_field(source_name, source_ref),
        loom_param_i64(source_rank),
    };
    loom_verify_emit_structured(
        state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_SHAPE, 1), params,
        IREE_ARRAYSIZE(params));
    return;
  }

  uint8_t grouped_axis = source_rank - 1;
  for (uint8_t axis = 0; axis < grouped_axis; ++axis) {
    if (loom_type_dim(source_type, axis) == loom_type_dim(result_type, axis)) {
      continue;
    }
    loom_diagnostic_param_t params[] = {
        loom_verify_param_string_for_field(result_name, result_ref),
        loom_verify_param_string_for_field(source_name, source_ref),
    };
    loom_verify_emit_structured(
        state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_SHAPE, 2), params,
        IREE_ARRAYSIZE(params));
    return;
  }

  if (loom_type_dim_is_dynamic_at(source_type, grouped_axis)) return;

  int64_t source_axis_size =
      loom_type_dim_static_size_at(source_type, grouped_axis);
  int64_t group_size = constraint->property;
  if ((source_axis_size % group_size) != 0) {
    loom_diagnostic_param_t params[] = {
        loom_verify_param_string_for_field(source_name, source_ref),
        loom_param_type(source_type),
        loom_param_string(
            loom_verify_grouped_last_axis_divisibility_constraint(group_size)),
    };
    loom_verify_emit_structured(
        state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 3), params,
        IREE_ARRAYSIZE(params));
    return;
  }

  if (loom_type_dim_is_dynamic_at(result_type, grouped_axis)) return;

  int64_t result_axis_size =
      loom_type_dim_static_size_at(result_type, grouped_axis);
  if (result_axis_size == source_axis_size / group_size) return;

  loom_diagnostic_param_t params[] = {
      loom_verify_param_string_for_field(result_name, result_ref),
      loom_param_type(result_type),
      loom_param_string(
          loom_verify_grouped_last_axis_result_constraint(group_size)),
  };
  loom_verify_emit_structured(state, op,
                              loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 4),
                              params, IREE_ARRAYSIZE(params));
}

// COUNT_MATCHES_RANK: a variadic value field's element count equals
// the rank of a shaped value field. Args: (shaped value field,
// variadic value field).
static void loom_verify_relation_count_matches_rank(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return;
  loom_type_t shaped_type = loom_verify_value_type(
      state, loom_verify_resolve_value_field(op, constraint->args[0]));
  uint16_t variadic_count =
      loom_verify_variadic_count(op, vtable, constraint->args[1]);
  uint8_t rank = loom_type_rank(shaped_type);
  if (variadic_count == rank) return;
  char name_buffer[32];
  iree_string_view_t operand_name = loom_verify_field_name(
      vtable, constraint->args[0], name_buffer, sizeof(name_buffer));
  const loom_error_def_t* error = loom_verify_constraint_error_or(
      constraint, loom_error_def_lookup(LOOM_ERROR_DOMAIN_SUBRANGE, 1));
  loom_diagnostic_param_t params[] = {
      loom_verify_param_string_for_field(operand_name, constraint->args[0]),
      loom_param_u32(variadic_count),
      loom_param_i64(rank),
  };
  loom_verify_emit_structured(state, op, error, params,
                              error->param_count < 3 ? error->param_count : 3);
}

// COUNT_MATCHES_STATIC_ELEMENT_COUNT: a variadic value field's element count
// equals the static element count of a shaped value field. Args: (shaped value
// field, variadic value field).
static void loom_verify_relation_count_matches_static_element_count(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return;
  uint8_t shaped_ref = constraint->args[0];
  uint8_t values_ref = constraint->args[1];
  loom_type_t shaped_type = loom_verify_value_type(
      state, loom_verify_resolve_value_field(op, shaped_ref));
  if (!loom_type_is_shaped(shaped_type) ||
      !loom_type_is_all_static(shaped_type)) {
    return;
  }

  uint64_t expected_count = 0;
  if (!loom_type_static_element_count(shaped_type, &expected_count)) {
    char shaped_name_buffer[32];
    iree_string_view_t shaped_name = loom_verify_field_name(
        vtable, shaped_ref, shaped_name_buffer, sizeof(shaped_name_buffer));
    const loom_error_def_t* error =
        LOOM_FIELD_REF_CATEGORY(shaped_ref) == LOOM_FIELD_RESULT
            ? loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 4)
            : loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 3);
    loom_diagnostic_param_t params[] = {
        loom_verify_param_string_for_field(shaped_name, shaped_ref),
        loom_param_type(shaped_type),
        loom_param_string(IREE_SV("representable static element count")),
    };
    loom_verify_emit_structured(
        state, op, error, params,
        error->param_count < 3 ? error->param_count : 3);
    return;
  }

  uint16_t value_count = loom_verify_variadic_count(op, vtable, values_ref);
  if (value_count == expected_count) return;

  char values_name_buffer[32];
  char shaped_name_buffer[32];
  char expected_name_buffer[64];
  iree_string_view_t values_name = loom_verify_field_name(
      vtable, values_ref, values_name_buffer, sizeof(values_name_buffer));
  iree_string_view_t shaped_name = loom_verify_field_name(
      vtable, shaped_ref, shaped_name_buffer, sizeof(shaped_name_buffer));
  iree_snprintf(expected_name_buffer, sizeof(expected_name_buffer),
                "%.*s static element count", (int)shaped_name.size,
                shaped_name.data);
  uint32_t expected_count_param =
      expected_count > UINT32_MAX ? UINT32_MAX : (uint32_t)expected_count;
  const loom_error_def_t* error = loom_verify_constraint_error_or(
      constraint, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 13));
  loom_diagnostic_param_t params[] = {
      loom_verify_param_string_for_field(values_name, values_ref),
      loom_param_u32(value_count),
      loom_verify_param_string_for_field(
          iree_make_cstring_view(expected_name_buffer), shaped_ref),
      loom_param_u32(expected_count_param),
  };
  loom_verify_emit_structured(state, op, error, params,
                              error->param_count < 4 ? error->param_count : 4);
}

// ATTR_IN_RANGE_RANK: an i64 attribute's value falls within
// [0, rank) of a shaped value field. Args: (shaped value field,
// i64 attr field).
static void loom_verify_relation_attr_in_range_rank(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return;
  loom_type_t shaped_type = loom_verify_value_type(
      state, loom_verify_resolve_value_field(op, constraint->args[0]));
  uint8_t attr_index = LOOM_FIELD_REF_INDEX(constraint->args[1]);
  if (attr_index >= op->attribute_count) return;
  int64_t dim_index = loom_attr_as_i64(loom_op_attrs(op)[attr_index]);
  uint8_t rank = loom_type_rank(shaped_type);
  if (dim_index >= 0 && dim_index < rank) return;
  const loom_error_def_t* error = loom_verify_constraint_error_or(
      constraint, loom_error_def_lookup(LOOM_ERROR_DOMAIN_SUBRANGE, 2));
  loom_diagnostic_param_t params[] = {
      loom_param_i64(dim_index),
      loom_param_i64(rank),
  };
  loom_verify_emit_structured(state, op, error, params,
                              error->param_count < 2 ? error->param_count : 2);
}

// REGION_ARG_COUNT: a region's entry block argument count matches
// the element count of a variadic value field. Args: (region field,
// variadic value field).
static void loom_verify_relation_region_arg_count(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return;
  uint8_t region_index = LOOM_FIELD_REF_INDEX(constraint->args[0]);
  if (region_index >= op->region_count) return;
  loom_region_t* region = loom_op_regions(op)[region_index];
  if (!region || region->block_count == 0) return;
  uint16_t block_arg_count = loom_region_entry_arg_count(region);
  uint16_t input_count =
      loom_verify_variadic_count(op, vtable, constraint->args[1]);
  if (block_arg_count == input_count) return;
  const loom_error_def_t* error = loom_verify_constraint_error_or(
      constraint, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 7));
  loom_diagnostic_param_t params[] = {
      loom_param_u32(block_arg_count),
      loom_param_u32(input_count),
  };
  loom_verify_emit_structured(state, op, error, params,
                              error->param_count < 2 ? error->param_count : 2);
}

// REGION_ARG_MATCH: each region entry block argument's property
// matches the corresponding element of a variadic value field at the
// same position. Args: (region field, variadic operand field).
static void loom_verify_relation_region_arg_match(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return;
  uint8_t region_index = LOOM_FIELD_REF_INDEX(constraint->args[0]);
  if (region_index >= op->region_count) return;
  loom_region_t* region = loom_op_regions(op)[region_index];
  if (!region || region->block_count == 0) return;
  loom_block_t* entry = loom_region_entry_block(region);
  // Block args are sourced from operand-side fields only — region
  // entry args mirror an op's input values, never its result values.
  if (LOOM_FIELD_REF_CATEGORY(constraint->args[1]) != LOOM_FIELD_OPERAND) {
    return;
  }
  uint16_t input_count = 0;
  const loom_value_id_t* input_values =
      loom_verify_resolve_variadic_field(op, constraint->args[1], &input_count);
  if (!input_values) return;
  uint16_t check_count =
      entry->arg_count < input_count ? entry->arg_count : input_count;
  for (uint16_t i = 0; i < check_count; ++i) {
    loom_type_t block_arg_type =
        loom_verify_value_type(state, loom_block_arg_id(entry, i));
    loom_type_t input_type = loom_verify_value_type(state, input_values[i]);
    if (loom_constraint_property_equals(block_arg_type, input_type,
                                        constraint->property)) {
      continue;
    }
    const loom_error_def_t* error = loom_verify_constraint_error_or(
        constraint, loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 8));
    loom_type_t expected_type =
        constraint->property == LOOM_PROPERTY_ELEMENT_TYPE
            ? loom_type_scalar(loom_type_element_type(input_type))
            : input_type;
    loom_diagnostic_param_t params[] = {
        loom_param_u32(i),
        loom_param_type(block_arg_type),
        loom_param_type(expected_type),
    };
    loom_verify_emit_structured(
        state, op, error, params,
        error->param_count < 3 ? error->param_count : 3);
  }
}

// YIELD_COUNT: a region's terminator (yield) operand count matches
// the element count of a variadic value field. Args: (region field,
// variadic value field).
static void loom_verify_relation_yield_count(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return;
  uint16_t yield_count = 0;
  if (!loom_verify_region_entry_yield(state, op, vtable,
                                      LOOM_FIELD_REF_INDEX(constraint->args[0]),
                                      &yield_count, NULL)) {
    return;
  }
  uint16_t result_count =
      loom_verify_variadic_count(op, vtable, constraint->args[1]);
  // A non-variadic result counts as a single element for the purposes
  // of yield-count comparisons (the field still names a value).
  if (LOOM_FIELD_REF_CATEGORY(constraint->args[1]) == LOOM_FIELD_RESULT &&
      LOOM_FIELD_REF_INDEX(constraint->args[1]) < vtable->fixed_result_count) {
    result_count = 1;
  }
  if (yield_count == result_count) return;
  const loom_error_def_t* error = loom_verify_constraint_error_or(
      constraint, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 8));
  loom_diagnostic_param_t params[] = {
      loom_param_u32(yield_count),
      loom_param_u32(result_count),
  };
  loom_verify_emit_structured(state, op, error, params,
                              error->param_count < 2 ? error->param_count : 2);
}

// YIELD_MATCH: each region terminator (yield) operand's property
// matches the corresponding element of a variadic result field at the
// same position. Args: (region field, variadic result field).
static void loom_verify_relation_yield_match(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return;
  // Yields forward into result values — only result-side fields are
  // valid as the second arg.
  if (LOOM_FIELD_REF_CATEGORY(constraint->args[1]) != LOOM_FIELD_RESULT) {
    return;
  }
  uint16_t yield_count = 0;
  const loom_value_id_t* yield_operands = NULL;
  if (!loom_verify_region_entry_yield(state, op, vtable,
                                      LOOM_FIELD_REF_INDEX(constraint->args[0]),
                                      &yield_count, &yield_operands)) {
    return;
  }
  uint16_t result_count = 0;
  const loom_value_id_t* result_values = loom_verify_resolve_variadic_field(
      op, constraint->args[1], &result_count);
  if (!result_values) return;
  uint16_t check_count =
      yield_count < result_count ? yield_count : result_count;
  loom_type_value_remap_t yield_remap = {
      .source_values = result_values,
      .target_values = yield_operands,
      .count = check_count,
  };
  for (uint16_t i = 0; i < check_count; ++i) {
    loom_type_t yield_type = loom_verify_value_type(state, yield_operands[i]);
    loom_type_t result_type = loom_verify_value_type(state, result_values[i]);
    bool matched = constraint->property == LOOM_PROPERTY_TYPE
                       ? loom_type_equal_after_value_remap(
                             result_type, yield_type, &yield_remap)
                       : loom_constraint_property_equals(
                             yield_type, result_type, constraint->property);
    if (matched) {
      continue;
    }
    const loom_error_def_t* error = loom_verify_constraint_error_or(
        constraint, loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 9));
    loom_type_t expected_type =
        loom_type_scalar(loom_type_element_type(result_type));
    loom_diagnostic_param_t params[] = {
        loom_param_type(yield_type),
        loom_param_type(expected_type),
    };
    loom_verify_emit_structured(
        state, op, error, params,
        error->param_count < 2 ? error->param_count : 2);
  }
}

// VARIADIC_MATCH: two variadic value fields agree position-by-position
// on count and per-element property. Args: (variadic value field,
// variadic value field). Stops at the count check on a mismatch to
// avoid cascading per-position errors that obscure the root cause.
static void loom_verify_relation_variadic_match(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return;
  uint8_t ref_a = constraint->args[0];
  uint8_t ref_b = constraint->args[1];
  uint16_t count_a = 0;
  uint16_t count_b = 0;
  const loom_value_id_t* values_a =
      loom_verify_resolve_variadic_field(op, ref_a, &count_a);
  const loom_value_id_t* values_b =
      loom_verify_resolve_variadic_field(op, ref_b, &count_b);
  if (!values_a || !values_b) return;

  if (count_a != count_b) {
    char name_a_buffer[32];
    char name_b_buffer[32];
    iree_string_view_t name_a = loom_verify_field_name(
        vtable, ref_a, name_a_buffer, sizeof(name_a_buffer));
    iree_string_view_t name_b = loom_verify_field_name(
        vtable, ref_b, name_b_buffer, sizeof(name_b_buffer));
    const loom_error_def_t* error = loom_verify_constraint_error_or(
        constraint, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 13));
    loom_diagnostic_param_t params[] = {
        loom_verify_param_string_for_field(name_a, ref_a),
        loom_param_u32(count_a),
        loom_verify_param_string_for_field(name_b, ref_b),
        loom_param_u32(count_b),
    };
    loom_verify_emit_structured(
        state, op, error, params,
        error->param_count < 4 ? error->param_count : 4);
    return;
  }

  // Per-position type mismatches use the property's default type
  // error, NOT the constraint's error_ref: VARIADIC_MATCH stores the count
  // error (ERR_STRUCTURE_013) on the constraint, which is the wrong
  // shape (STRING/U32/STRING/U32) for type-mismatch params.
  const loom_error_def_t* type_error =
      loom_pairwise_eq_default_error(constraint->property);
  loom_type_value_remap_t value_remap = {
      .source_values = values_b,
      .target_values = values_a,
      .count = count_a,
  };
  for (uint16_t i = 0; i < count_a; ++i) {
    loom_type_t type_a = loom_verify_value_type(state, values_a[i]);
    loom_type_t type_b = loom_verify_value_type(state, values_b[i]);
    bool matched =
        constraint->property == LOOM_PROPERTY_TYPE
            ? loom_type_equal_after_value_remap(type_b, type_a, &value_remap)
            : loom_constraint_property_equals(type_a, type_b,
                                              constraint->property);
    if (matched) {
      continue;
    }
    loom_verify_emit_indexed_pairwise_mismatch(state, op, vtable, type_error,
                                               ref_a, /*a_is_indexed=*/true, i,
                                               type_a, ref_b, i, type_b);
  }
}

// Dispatch table for semantic constraint relations. Indexed by the
// loom_constraint_relation_t enum value. Adding a relation requires
// (a) updating the enum in op_defs.h, (b) adding a handler above, and
// (c) appending the handler here. The static_assert below ensures
// every enum value has a handler.
typedef void (*loom_verify_relation_fn_t)(loom_verify_state_t* state,
                                          const loom_op_t* op,
                                          const loom_op_vtable_t* vtable,
                                          const loom_constraint_t* constraint);

static const loom_verify_relation_fn_t kVerifyRelationFns[] = {
    [LOOM_RELATION_PAIRWISE_EQ] = loom_verify_relation_pairwise_eq,
    [LOOM_RELATION_ALL_SAME] = loom_verify_relation_all_same,
    [LOOM_RELATION_FIELD_SATISFIES] = loom_verify_relation_field_satisfies,
    [LOOM_RELATION_REGION_ARGS_SATISFY] =
        loom_verify_relation_region_args_satisfy,
    [LOOM_RELATION_ATTR_I64_PREDICATE] =
        loom_verify_relation_attr_i64_predicate,
    [LOOM_RELATION_ATTR_MATCHES_ELEMENT_TYPE] =
        loom_verify_relation_attr_matches_element_type,
    [LOOM_RELATION_ELEMENT_WIDTH_ORDER] =
        loom_verify_relation_element_width_order,
    [LOOM_RELATION_ELEMENT_WIDTH_AT_LEAST_ATTR] =
        loom_verify_relation_element_width_at_least_attr,
    [LOOM_RELATION_BIT_RANGE_WITHIN_ELEMENT_WIDTH] =
        loom_verify_relation_bit_range_within_element_width,
    [LOOM_RELATION_TOTAL_BIT_COUNT_EQUAL] =
        loom_verify_relation_total_bit_count_equal,
    [LOOM_RELATION_PAYLOAD_BIT_COUNT_MATCHES_STORAGE] =
        loom_verify_relation_payload_bit_count_matches_storage,
    [LOOM_RELATION_COUNT_MATCHES_RANK] =
        loom_verify_relation_count_matches_rank,
    [LOOM_RELATION_COUNT_MATCHES_STATIC_ELEMENT_COUNT] =
        loom_verify_relation_count_matches_static_element_count,
    [LOOM_RELATION_ATTR_IN_RANGE_RANK] =
        loom_verify_relation_attr_in_range_rank,
    [LOOM_RELATION_REGION_ARG_COUNT] = loom_verify_relation_region_arg_count,
    [LOOM_RELATION_REGION_ARG_MATCH] = loom_verify_relation_region_arg_match,
    [LOOM_RELATION_YIELD_COUNT] = loom_verify_relation_yield_count,
    [LOOM_RELATION_YIELD_MATCH] = loom_verify_relation_yield_match,
    [LOOM_RELATION_VARIADIC_MATCH] = loom_verify_relation_variadic_match,
    [LOOM_RELATION_LAST_AXIS_GROUPED_BY] =
        loom_verify_relation_last_axis_grouped_by,
};
static_assert(IREE_ARRAYSIZE(kVerifyRelationFns) == LOOM_RELATION_COUNT_,
              "verify relation dispatch table out of sync with enum");

static void loom_verify_semantic_constraint(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->relation >= LOOM_RELATION_COUNT_) return;
  kVerifyRelationFns[constraint->relation](state, op, vtable, constraint);
}

static void loom_verify_semantic_constraints(loom_verify_state_t* state,
                                             const loom_op_t* op,
                                             const loom_op_vtable_t* vtable) {
  if (!vtable->constraints || vtable->constraint_count == 0) return;
  for (uint8_t i = 0; i < vtable->constraint_count; ++i) {
    if (loom_verify_at_error_limit(state)) return;
    loom_verify_semantic_constraint(state, op, vtable, &vtable->constraints[i]);
  }
}

//===----------------------------------------------------------------------===//
// SSA dominance and linear ownership checks
//===----------------------------------------------------------------------===//

static void loom_verify_operand_dominance(loom_verify_state_t* state,
                                          const loom_op_t* op,
                                          const loom_op_vtable_t* vtable) {
  if (loom_verify_func_args_are_operands(vtable) &&
      iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE)) {
    return;
  }
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    loom_value_id_t value_id = operands[i];
    if (value_id == LOOM_VALUE_ID_INVALID) continue;
    if (value_id >= state->module->values.count) {
      loom_diagnostic_param_t params[] = {
          loom_param_u32(value_id),
          loom_param_u32((uint32_t)state->module->values.count),
      };
      loom_verify_emit_structured(
          state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_DOMINANCE, 3),
          params, IREE_ARRAYSIZE(params));
      continue;
    }
    if (!loom_bitset_test(state->defined_bits, value_id)) {
      iree_string_view_t value_name = loom_verify_value_name(state, value_id);
      loom_diagnostic_field_ref_t operand_ref =
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, i);
      loom_diagnostic_param_t params[] = {
          loom_param_with_field_ref(loom_param_string(value_name), operand_ref),
      };
      loom_verify_emit_structured(
          state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_DOMINANCE, 1),
          params, IREE_ARRAYSIZE(params));
    }
    if (loom_bitset_test(state->consumed_bits, value_id)) {
      const loom_op_t* consuming_op = state->consuming_ops[value_id];
      const loom_op_vtable_t* consuming_vtable =
          consuming_op ? loom_verify_lookup_vtable(state, consuming_op->kind)
                       : NULL;
      iree_string_view_t consuming_op_name =
          consuming_vtable ? loom_op_vtable_name(consuming_vtable)
                           : IREE_SV("<unknown op>");
      iree_string_view_t value_name = loom_verify_value_name(state, value_id);
      loom_diagnostic_field_ref_t operand_ref =
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, i);
      loom_diagnostic_param_t params[] = {
          loom_param_with_field_ref(loom_param_string(value_name), operand_ref),
          loom_param_string(consuming_op_name),
      };
      loom_diagnostic_related_op_t related_ops[] = {{
          .label = IREE_SV("consumed here"),
          .op = consuming_op,
      }};
      loom_diagnostic_emission_t emission = {
          .op = op,
          .error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_DOMINANCE, 2),
          .params = params,
          .param_count = IREE_ARRAYSIZE(params),
          .related_ops = related_ops,
          .related_op_count = IREE_ARRAYSIZE(related_ops),
      };
      loom_verify_emit_diagnostic(state, &emission);
    }
  }
}

static bool loom_verify_op_observes_poison(
    const loom_module_t* module, const loom_op_t* op,
    iree_string_view_t* out_boundary_kind) {
  if (loom_func_return_isa(op)) {
    *out_boundary_kind = IREE_SV("function return");
    return true;
  }
  loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  const loom_trait_flags_t boundary_traits =
      LOOM_TRAIT_READS_MEMORY | LOOM_TRAIT_WRITES_MEMORY |
      LOOM_TRAIT_NON_DETERMINISTIC | LOOM_TRAIT_UNKNOWN_EFFECTS;
  if (iree_any_bit_set(traits, boundary_traits)) {
    *out_boundary_kind = IREE_SV("effectful operation");
    return true;
  }
  *out_boundary_kind = IREE_SV("");
  return false;
}

static void loom_verify_poison_boundaries(loom_verify_state_t* state,
                                          const loom_op_t* op,
                                          const loom_op_vtable_t* vtable) {
  iree_string_view_t boundary_kind = IREE_SV("");
  if (!loom_verify_op_observes_poison(state->module, op, &boundary_kind)) {
    return;
  }

  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (loom_verify_at_error_limit(state)) return;
    loom_value_id_t value_id = operands[i];
    if (value_id == LOOM_VALUE_ID_INVALID) continue;
    if (value_id >= state->module->values.count) continue;
    if (!loom_bitset_test(state->defined_bits, value_id)) continue;
    if (!loom_value_is_poison(state->module, value_id)) continue;

    const loom_value_t* value = loom_module_value(state->module, value_id);
    const loom_op_t* poison_op = loom_value_def_op(value);
    iree_string_view_t value_name = loom_verify_value_name(state, value_id);
    iree_string_view_t op_name = loom_op_vtable_name(vtable);
    loom_diagnostic_field_ref_t operand_ref =
        loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, i);
    loom_diagnostic_param_t params[] = {
        loom_param_with_field_ref(loom_param_string(value_name), operand_ref),
        loom_param_string(boundary_kind),
        loom_param_string(op_name),
    };
    loom_diagnostic_related_op_t related_ops[] = {{
        .label = IREE_SV("poison produced here"),
        .op = poison_op,
    }};
    loom_diagnostic_emission_t emission = {
        .op = op,
        .error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 12),
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
        .related_ops = related_ops,
        .related_op_count = IREE_ARRAYSIZE(related_ops),
    };
    loom_verify_emit_diagnostic(state, &emission);
  }
}

//===----------------------------------------------------------------------===//
// Type payload validation
//===----------------------------------------------------------------------===//

static iree_string_view_t loom_verify_type_well_formed_detail(
    loom_type_t type) {
  loom_type_kind_t kind = loom_type_kind(type);
  if (!loom_type_kind_is_valid(kind)) {
    return IREE_SV("type kind is out of range");
  }
  switch (kind) {
    case LOOM_TYPE_ENCODING:
      if (!loom_encoding_role_is_valid(loom_type_encoding_role(type))) {
        return IREE_SV("encoding role is out of range");
      }
      break;
    case LOOM_TYPE_VECTOR:
      if (loom_type_rank(type) == 0) {
        return IREE_SV("vector types must have rank >= 1");
      }
      if (type.encoding_id != 0 || type.encoding_flags != 0) {
        return IREE_SV(
            "vector types must not carry encoding or layout attachments");
      }
      break;
    default:
      break;
  }
  return iree_string_view_empty();
}

static void loom_verify_type_well_formed(
    loom_verify_state_t* state, const loom_op_t* op, loom_type_t type,
    iree_string_view_t field_name, loom_diagnostic_field_ref_t field_ref) {
  iree_string_view_t detail = loom_verify_type_well_formed_detail(type);
  if (iree_string_view_is_empty(detail)) return;
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(loom_param_string(field_name), field_ref),
      loom_param_type(type),
      loom_param_string(detail),
  };
  loom_verify_emit_structured(state, op,
                              loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 10),
                              params, IREE_ARRAYSIZE(params));
}

static void loom_verify_value_type_well_formed(
    loom_verify_state_t* state, const loom_op_t* op, loom_value_id_t value_id,
    iree_string_view_t field_name, loom_diagnostic_field_ref_t field_ref) {
  if (value_id == LOOM_VALUE_ID_INVALID) return;
  if (value_id >= state->module->values.count) return;
  loom_verify_type_well_formed(state, op,
                               loom_module_value_type(state->module, value_id),
                               field_name, field_ref);
}

static void loom_verify_op_type_well_formedness(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable) {
  char name_buffer[64];
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    iree_string_view_t name = loom_verify_value_field_name(
        vtable, LOOM_FIELD_OPERAND, i, name_buffer, sizeof(name_buffer));
    loom_verify_value_type_well_formed(
        state, op, operands[i], name,
        loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, i));
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    iree_string_view_t name = loom_verify_value_field_name(
        vtable, LOOM_FIELD_RESULT, i, name_buffer, sizeof(name_buffer));
    loom_verify_value_type_well_formed(
        state, op, results[i], name,
        loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT, i));
  }
}

static void loom_verify_block_arg_type_well_formedness(
    loom_verify_state_t* state, const loom_block_t* block) {
  char name_buffer[32];
  for (uint16_t a = 0; a < block->arg_count; ++a) {
    iree_snprintf(name_buffer, sizeof(name_buffer), "block arg %u", a);
    loom_verify_value_type_well_formed(state, NULL, loom_block_arg_id(block, a),
                                       iree_make_cstring_view(name_buffer),
                                       loom_diagnostic_field_ref_none());
  }
}

//===----------------------------------------------------------------------===//
// SSA encoding reference validation
//===----------------------------------------------------------------------===//

static bool loom_verify_op_result_contains_value(const loom_op_t* op,
                                                 loom_value_id_t value_id) {
  if (!op) return false;
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == value_id) return true;
  }
  return false;
}

static bool loom_verify_value_is_named_placeholder(
    const loom_verify_state_t* state, loom_value_id_t value_id) {
  if (value_id >= state->module->values.count) return false;
  const loom_value_t* value = &state->module->values.entries[value_id];
  if (loom_value_is_block_arg(value)) return false;
  return value->name_id != LOOM_STRING_ID_INVALID &&
         loom_def_op(value->def) == NULL;
}

static bool loom_verify_op_allows_declaration_local_encoding_refs(
    const loom_op_vtable_t* vtable) {
  return vtable && vtable->symbol_def &&
         loom_symbol_definition_implements(vtable->symbol_def,
                                           LOOM_SYMBOL_INTERFACE_GLOBAL) &&
         iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE);
}

// Validates a single SSA encoding reference embedded in a value's type.
// If the type carries LOOM_ENCODING_FLAG_SSA, the encoding_id is a
// value_id that must be in range and have type LOOM_TYPE_ENCODING. It must
// also be defined in scope unless the reference is to a sibling result in the
// current op type annotation or to a declaration-local global placeholder.
static void loom_verify_encoding_ref(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, loom_type_t type,
    iree_string_view_t field_name, loom_diagnostic_field_ref_t field_ref,
    bool allow_current_op_results, bool allow_declaration_placeholders) {
  if (!loom_type_has_ssa_encoding(type)) return;
  uint16_t encoding_value_id = loom_type_encoding_value_id(type);
  if (encoding_value_id >= state->module->values.count) {
    loom_diagnostic_param_t params[] = {
        loom_param_with_field_ref(loom_param_string(field_name), field_ref),
        loom_param_u32(encoding_value_id),
        loom_param_u32((uint32_t)state->module->values.count),
    };
    loom_verify_emit_structured(
        state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 3), params,
        IREE_ARRAYSIZE(params));
    return;
  }
  if (!loom_bitset_test(state->defined_bits, encoding_value_id)) {
    bool allowed_current_result =
        allow_current_op_results &&
        loom_verify_op_result_contains_value(op, encoding_value_id);
    bool allowed_declaration_placeholder =
        allow_declaration_placeholders &&
        loom_verify_op_allows_declaration_local_encoding_refs(vtable) &&
        loom_verify_value_is_named_placeholder(state, encoding_value_id);
    if (!allowed_current_result && !allowed_declaration_placeholder) {
      iree_string_view_t value_name =
          loom_verify_value_name(state, encoding_value_id);
      loom_diagnostic_param_t params[] = {
          loom_param_with_field_ref(loom_param_string(field_name), field_ref),
          loom_param_string(value_name),
      };
      loom_verify_emit_structured(
          state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 4),
          params, IREE_ARRAYSIZE(params));
      return;
    }
  }
  loom_type_t encoding_type =
      loom_module_value_type(state->module, encoding_value_id);
  if (!loom_type_is_encoding(encoding_type)) {
    iree_string_view_t value_name =
        loom_verify_value_name(state, encoding_value_id);
    loom_diagnostic_param_t params[] = {
        loom_param_with_field_ref(loom_param_string(field_name), field_ref),
        loom_param_string(value_name),
        loom_param_type(encoding_type),
    };
    loom_verify_emit_structured(
        state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 5), params,
        IREE_ARRAYSIZE(params));
  }
}

// Checks SSA encoding references in all operand and result types of an op.
static void loom_verify_encoding_refs(loom_verify_state_t* state,
                                      const loom_op_t* op,
                                      const loom_op_vtable_t* vtable) {
  char name_buffer[64];
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (operands[i] == LOOM_VALUE_ID_INVALID) continue;
    if (operands[i] >= state->module->values.count) continue;
    loom_type_t type = loom_module_value_type(state->module, operands[i]);
    iree_string_view_t name = loom_verify_value_field_name(
        vtable, LOOM_FIELD_OPERAND, i, name_buffer, sizeof(name_buffer));
    loom_verify_encoding_ref(
        state, op, vtable, type, name,
        loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, i),
        /*allow_current_op_results=*/false,
        /*allow_declaration_placeholders=*/false);
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID) continue;
    if (results[i] >= state->module->values.count) continue;
    loom_type_t type = loom_module_value_type(state->module, results[i]);
    iree_string_view_t name = loom_verify_value_field_name(
        vtable, LOOM_FIELD_RESULT, i, name_buffer, sizeof(name_buffer));
    loom_verify_encoding_ref(
        state, op, vtable, type, name,
        loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT, i),
        /*allow_current_op_results=*/true,
        /*allow_declaration_placeholders=*/true);
  }
}

// Checks SSA encoding references in block argument types. Called when
// entering a block, after block args are defined but before ops are
// verified. The encoding value must already be visible (from an
// enclosing scope or earlier in this scope).
static void loom_verify_block_arg_encoding_refs(loom_verify_state_t* state,
                                                const loom_block_t* block) {
  char name_buffer[32];
  for (uint16_t a = 0; a < block->arg_count; ++a) {
    loom_value_id_t arg_id = loom_block_arg_id(block, a);
    if (arg_id == LOOM_VALUE_ID_INVALID) continue;
    if (arg_id >= state->module->values.count) continue;
    loom_type_t type = loom_module_value_type(state->module, arg_id);
    iree_snprintf(name_buffer, sizeof(name_buffer), "block arg %u", a);
    loom_verify_encoding_ref(state, NULL, NULL, type,
                             iree_make_cstring_view(name_buffer),
                             loom_diagnostic_field_ref_none(),
                             /*allow_current_op_results=*/false,
                             /*allow_declaration_placeholders=*/false);
  }
}

//===----------------------------------------------------------------------===//
// Tied result validation
//===----------------------------------------------------------------------===//

static void loom_verify_emit_duplicate_tied_result(loom_verify_state_t* state,
                                                   const loom_op_t* op,
                                                   iree_string_view_t op_name,
                                                   uint16_t result_index,
                                                   uint16_t current_occurrence,
                                                   uint16_t first_occurrence) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_u32(result_index),
          loom_diagnostic_field_ref_with_occurrence(
              LOOM_DIAGNOSTIC_FIELD_RESULT, result_index, current_occurrence)),
      loom_param_string(op_name),
  };
  loom_diagnostic_related_op_t related_ops[] = {{
      .label = IREE_SV("previously tied here"),
      .op = op,
      .field_ref = loom_diagnostic_field_ref_with_occurrence(
          LOOM_DIAGNOSTIC_FIELD_RESULT, result_index, first_occurrence),
  }};
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_DOMINANCE, 6),
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .related_ops = related_ops,
      .related_op_count = IREE_ARRAYSIZE(related_ops),
  };
  loom_verify_emit_diagnostic(state, &emission);
}

static void loom_verify_emit_duplicate_tied_operand(loom_verify_state_t* state,
                                                    const loom_op_t* op,
                                                    iree_string_view_t op_name,
                                                    uint16_t operand_index,
                                                    uint16_t current_occurrence,
                                                    uint16_t first_occurrence) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(loom_param_u32(operand_index),
                                loom_diagnostic_field_ref_with_occurrence(
                                    LOOM_DIAGNOSTIC_FIELD_OPERAND,
                                    operand_index, current_occurrence)),
      loom_param_string(op_name),
  };
  loom_diagnostic_related_op_t related_ops[] = {{
      .label = IREE_SV("previously tied here"),
      .op = op,
      .field_ref = loom_diagnostic_field_ref_with_occurrence(
          LOOM_DIAGNOSTIC_FIELD_OPERAND, operand_index, first_occurrence),
  }};
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_DOMINANCE, 7),
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .related_ops = related_ops,
      .related_op_count = IREE_ARRAYSIZE(related_ops),
  };
  loom_verify_emit_diagnostic(state, &emission);
}

static iree_status_t loom_verify_tied_results(loom_verify_state_t* state,
                                              const loom_op_t* op,
                                              const loom_op_vtable_t* vtable) {
  if (op->tied_result_count == 0) return iree_ok_status();

  const bool has_signature_ties = loom_verify_has_func_signature_scope(vtable);
  uint16_t tied_operand_count = 0;
  const loom_value_id_t* tied_operands = NULL;
  if (has_signature_ties) {
    tied_operands =
        loom_verify_func_signature_arg_ids(op, vtable, &tied_operand_count);
  } else {
    tied_operand_count = op->operand_count;
    tied_operands = loom_op_const_operands(op);
  }

  IREE_RETURN_IF_ERROR(loom_verify_tied_table_reset(
      &state->tied_table, &state->arena, op->result_count, tied_operand_count,
      has_signature_ties ? 0 : 1));

  iree_string_view_t op_name = loom_op_vtable_name(vtable);
  const loom_tied_result_t* tied = loom_op_tied_results(op);

  // Copy and sort this op's valid operand values so ties to repeated operand
  // values can be diagnosed as ambiguous.
  for (uint16_t i = 0; i < tied_operand_count; ++i) {
    loom_value_id_t value_id = tied_operands[i];
    if (value_id == LOOM_VALUE_ID_INVALID ||
        value_id >= state->module->values.count) {
      continue;
    }
    state->tied_table
        .operand_value_ids[state->tied_table.operand_value_count++] = value_id;
  }
  if (state->tied_table.operand_value_count > 1) {
    qsort(state->tied_table.operand_value_ids,
          state->tied_table.operand_value_count, sizeof(loom_value_id_t),
          loom_compare_value_ids);
  }

  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    if (tied[i].result_index >= op->result_count) {
      loom_diagnostic_param_t params[] = {
          loom_param_with_field_ref(
              loom_param_u32(tied[i].result_index),
              loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT,
                                        tied[i].result_index)),
          loom_param_string(op_name),
          loom_param_u32(op->result_count),
      };
      loom_verify_emit_structured(
          state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_DOMINANCE, 4),
          params, IREE_ARRAYSIZE(params));
    } else {
      uint16_t current_occurrence = 0;
      uint16_t first_occurrence = 0;
      if (loom_verify_tied_table_claim_result(
              &state->tied_table, tied[i].result_index, &current_occurrence,
              &first_occurrence)) {
        loom_verify_emit_duplicate_tied_result(
            state, op, op_name, tied[i].result_index, current_occurrence,
            first_occurrence);
      }
    }
    if (tied[i].operand_index >= tied_operand_count || !tied_operands) {
      loom_diagnostic_param_t params[] = {
          loom_param_with_field_ref(
              loom_param_u32(tied[i].operand_index),
              loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND,
                                        tied[i].operand_index)),
          loom_param_string(op_name),
          loom_param_u32(tied_operand_count),
      };
      loom_verify_emit_structured(
          state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_DOMINANCE, 5),
          params, IREE_ARRAYSIZE(params));
      continue;
    }
    uint16_t current_operand_occurrence = 0;
    uint16_t first_operand_occurrence = 0;
    if (loom_verify_tied_table_claim_operand(
            &state->tied_table, tied[i].operand_index,
            &current_operand_occurrence, &first_operand_occurrence)) {
      loom_verify_emit_duplicate_tied_operand(
          state, op, op_name, tied[i].operand_index, current_operand_occurrence,
          first_operand_occurrence);
    }

    loom_value_id_t consumed_id = tied_operands[tied[i].operand_index];
    // DOMINANCE/008 stays single-site for now because the duplicate-value check
    // runs over a sorted operand-value multiset, which intentionally discards
    // the source slot of the earlier equal-value operand.
    if (consumed_id != LOOM_VALUE_ID_INVALID &&
        consumed_id < state->module->values.count &&
        loom_verify_tied_table_has_duplicate_operand_value(&state->tied_table,
                                                           consumed_id)) {
      loom_diagnostic_field_ref_t operand_ref = loom_diagnostic_field_ref(
          LOOM_DIAGNOSTIC_FIELD_OPERAND, tied[i].operand_index);
      loom_diagnostic_param_t params[] = {
          loom_param_with_field_ref(loom_param_u32(tied[i].operand_index),
                                    operand_ref),
          loom_param_string(op_name),
          loom_param_with_field_ref(
              loom_param_string(loom_verify_value_name(state, consumed_id)),
              operand_ref),
      };
      loom_verify_emit_structured(
          state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_DOMINANCE, 8),
          params, IREE_ARRAYSIZE(params));
    }

    // Ties on regular body ops consume the operand's storage. Func-like symbol
    // signatures are caller-side ownership contracts, so their entry args are
    // validated as tie targets but not marked consumed at function entry.
    if (!has_signature_ties) {
      loom_verify_consume_value(state, consumed_id, op);
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Symbol reference checks
//===----------------------------------------------------------------------===//

static void loom_verify_symbol_references(loom_verify_state_t* state,
                                          const loom_op_t* op,
                                          const loom_op_vtable_t* vtable) {
  if (!vtable->attr_descriptors) return;
  const loom_attribute_t* attrs = loom_op_attrs(op);
  for (uint8_t i = 0; i < vtable->attribute_count && i < op->attribute_count;
       ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    if (descriptor->attr_kind != LOOM_ATTR_SYMBOL) continue;
    if ((descriptor->flags & LOOM_ATTR_OPTIONAL) &&
        loom_attr_is_absent(attrs[i])) {
      continue;
    }
    loom_symbol_ref_t ref = loom_attr_as_symbol(attrs[i]);
    if (!loom_symbol_ref_is_valid(ref)) continue;

    if (ref.module_id != 0) {
      loom_diagnostic_param_t params[] = {
          loom_param_with_field_ref(
              loom_param_u32(ref.module_id),
              loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, i)),
      };
      loom_verify_emit_structured(
          state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 4), params,
          IREE_ARRAYSIZE(params));
      continue;
    }

    if (ref.symbol_id >= state->module->symbols.count) {
      loom_diagnostic_param_t params[] = {
          loom_param_with_field_ref(
              loom_param_u32(ref.symbol_id),
              loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, i)),
          loom_param_u32((uint32_t)state->module->symbols.count),
      };
      loom_verify_emit_structured(
          state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 1), params,
          IREE_ARRAYSIZE(params));
      continue;
    }

    const loom_symbol_t* symbol =
        &state->module->symbols.entries[ref.symbol_id];
    if (symbol->definition == NULL || symbol->defining_op == NULL) {
      loom_diagnostic_param_t params[] = {
          loom_verify_param_string_for_diagnostic_field(
              loom_verify_symbol_name(state, ref),
              LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, i),
      };
      loom_verify_emit_structured(
          state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 2), params,
          IREE_ARRAYSIZE(params));
      continue;
    }

    if (descriptor->symbol_ref &&
        !loom_symbol_implements(symbol, descriptor->symbol_ref->interfaces)) {
      loom_diagnostic_param_t params[] = {
          loom_verify_param_string_for_diagnostic_field(
              loom_verify_symbol_name(state, ref),
              LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, i),
          loom_param_string(loom_verify_symbol_definition_name(symbol)),
          loom_param_string(
              loom_symbol_reference_descriptor_name(descriptor->symbol_ref)),
      };
      loom_diagnostic_related_op_t related_ops[] = {{
          .label = IREE_SV("defined here"),
          .op = symbol->defining_op,
      }};
      loom_diagnostic_emission_t emission = {
          .op = op,
          .error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 3),
          .params = params,
          .param_count = IREE_ARRAYSIZE(params),
          .related_ops = related_ops,
          .related_op_count = IREE_ARRAYSIZE(related_ops),
      };
      loom_verify_emit_diagnostic(state, &emission);
    }
  }
}

//===----------------------------------------------------------------------===//
// Region structure checks
//===----------------------------------------------------------------------===//

static const loom_op_t* loom_verify_block_last_live_op(
    const loom_block_t* block) {
  return block->last_op;
}

static bool loom_verify_op_is_terminator(loom_verify_state_t* state,
                                         const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_verify_lookup_vtable(state, op->kind);
  return vtable && iree_any_bit_set(vtable->traits, LOOM_TRAIT_TERMINATOR);
}

static bool loom_verify_region_terminator_matches(
    const loom_region_descriptor_t* region_descriptor,
    const loom_op_t* terminator) {
  if (!terminator || region_descriptor->terminator == LOOM_OP_KIND_UNKNOWN) {
    return true;
  }
  return terminator->kind == region_descriptor->terminator ||
         terminator->kind == region_descriptor->implicit_terminator;
}

static bool loom_verify_region_has_cfg_successors(const loom_region_t* region) {
  if (!region) return false;
  if (iree_any_bit_set(region->flags, LOOM_REGION_INSTANCE_FLAG_CFG)) {
    return true;
  }
  for (uint16_t b = 0; b < region->block_count; ++b) {
    const loom_block_t* block = loom_region_const_block(region, b);
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (op->successor_count > 0) return true;
    }
  }
  return false;
}

static bool loom_verify_region_entry_yield(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, uint8_t region_index,
    uint16_t* out_yield_count, const loom_value_id_t** out_yield_operands) {
  *out_yield_count = 0;
  if (out_yield_operands) {
    *out_yield_operands = NULL;
  }
  if (region_index >= op->region_count) {
    return false;
  }
  const loom_region_descriptor_t* region_descriptor =
      loom_op_vtable_region_descriptor(vtable, region_index);
  if (!region_descriptor) return false;
  loom_region_t* region = loom_op_regions(op)[region_index];
  if (!region || region->block_count == 0) return false;

  const loom_block_t* entry = loom_region_const_entry_block(region);
  const loom_op_t* terminator = loom_verify_block_last_live_op(entry);
  if (terminator && loom_verify_op_is_terminator(state, terminator) &&
      loom_verify_region_terminator_matches(region_descriptor, terminator)) {
    *out_yield_count = terminator->operand_count;
    if (out_yield_operands) {
      *out_yield_operands = loom_op_const_operands(terminator);
    }
    return true;
  }
  return region_descriptor->implicit_terminator != LOOM_OP_KIND_UNKNOWN;
}

static iree_string_view_t loom_verify_op_kind_name(loom_verify_state_t* state,
                                                   loom_op_kind_t kind) {
  const loom_op_vtable_t* vtable = loom_verify_lookup_vtable(state, kind);
  if (!vtable) return IREE_SV("<unknown op>");
  return loom_op_vtable_name(vtable);
}

static void loom_verify_region_terminator_kind(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, uint8_t region_index,
    bool region_uses_cfg_successors,
    const loom_region_descriptor_t* region_descriptor,
    const loom_op_t* terminator_op) {
  if (!terminator_op || region_uses_cfg_successors ||
      loom_verify_region_terminator_matches(region_descriptor, terminator_op)) {
    return;
  }
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_vtable_name(vtable)),
      loom_param_u32(region_index),
      loom_param_string(
          loom_verify_op_kind_name(state, region_descriptor->terminator)),
      loom_param_string(loom_verify_op_kind_name(state, terminator_op->kind)),
  };
  loom_verify_emit_structured(
      state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 18), params,
      IREE_ARRAYSIZE(params));
}

static void loom_verify_region_structure(loom_verify_state_t* state,
                                         const loom_op_t* op,
                                         const loom_op_vtable_t* vtable) {
  if (!vtable->region_descriptors) return;
  iree_string_view_t op_name = loom_op_vtable_name(vtable);
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    const loom_region_descriptor_t* region_descriptor =
        loom_op_vtable_region_descriptor(vtable, i);
    if (!region_descriptor) return;
    loom_region_t* region = regions[i];
    if (!region) {
      // NULL region is only valid for optional regions.
      continue;
    }
    if ((region_descriptor->flags & LOOM_REGION_SINGLE_BLOCK) &&
        region->block_count != 1) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(op_name),
          loom_param_u32(i),
          loom_param_u32(region->block_count),
      };
      loom_verify_emit_structured(
          state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 6),
          params, IREE_ARRAYSIZE(params));
    }
    bool region_uses_cfg_successors =
        loom_verify_region_has_cfg_successors(region);
    for (uint16_t b = 0; b < region->block_count; ++b) {
      const loom_block_t* block = loom_region_const_block(region, b);
      const loom_op_t* terminator_op = NULL;
      const loom_op_t* current_op = NULL;
      loom_block_for_each_op(block, current_op) {
        if (terminator_op) {
          const loom_op_vtable_t* current_vtable =
              loom_verify_lookup_vtable(state, current_op->kind);
          iree_string_view_t current_name =
              current_vtable ? loom_op_vtable_name(current_vtable)
                             : IREE_SV("<unknown op>");
          loom_diagnostic_param_t params[] = {
              loom_param_string(current_name),
          };
          loom_diagnostic_related_op_t related_ops[] = {{
              .label = IREE_SV("terminator here"),
              .op = terminator_op,
          }};
          loom_diagnostic_emission_t emission = {
              .op = current_op,
              .error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 12),
              .params = params,
              .param_count = IREE_ARRAYSIZE(params),
              .related_ops = related_ops,
              .related_op_count = IREE_ARRAYSIZE(related_ops),
          };
          loom_verify_emit_diagnostic(state, &emission);
          continue;
        }
        if (loom_verify_op_is_terminator(state, current_op)) {
          terminator_op = current_op;
        }
      }
      if (!terminator_op &&
          (region_descriptor->implicit_terminator == LOOM_OP_KIND_UNKNOWN ||
           region_uses_cfg_successors)) {
        loom_diagnostic_param_t params[] = {
            loom_param_string(op_name),
            loom_param_u32(i),
        };
        loom_verify_emit_structured(
            state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 5),
            params, IREE_ARRAYSIZE(params));
      }
      loom_verify_region_terminator_kind(state, op, vtable, i,
                                         region_uses_cfg_successors,
                                         region_descriptor, terminator_op);
    }
  }
}

//===----------------------------------------------------------------------===//
// Per-op verification (main dispatch)
//===----------------------------------------------------------------------===//

static iree_status_t loom_verify_op(loom_verify_state_t* state,
                                    const loom_op_t* op);

static iree_status_t loom_verify_region(loom_verify_state_t* state,
                                        loom_region_t* region) {
  if (!region) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_verify_push_scope(state));
  for (uint16_t b = 0; b < region->block_count; ++b) {
    loom_block_t* block = loom_region_block(region, b);
    // Define block arguments, then validate any SSA encoding
    // references in their types (encoding values must be visible
    // from the enclosing scope).
    for (uint16_t a = 0; a < block->arg_count; ++a) {
      IREE_RETURN_IF_ERROR(
          loom_verify_define_value(state, loom_block_arg_id(block, a)));
    }
    loom_verify_block_arg_type_well_formedness(state, block);
    iree_status_t diagnostic_status =
        loom_verify_pending_diagnostic_status(state);
    if (!iree_status_is_ok(diagnostic_status)) {
      loom_verify_pop_scope(state);
      return diagnostic_status;
    }
    loom_verify_block_arg_encoding_refs(state, block);
    diagnostic_status = loom_verify_pending_diagnostic_status(state);
    if (!iree_status_is_ok(diagnostic_status)) {
      loom_verify_pop_scope(state);
      return diagnostic_status;
    }
    loom_op_t* current = NULL;
    loom_block_for_each_op(block, current) {
      if (loom_verify_at_error_limit(state)) break;
      IREE_RETURN_IF_ERROR(loom_verify_op(state, current));
    }
  }
  loom_verify_pop_scope(state);
  return iree_ok_status();
}

static iree_status_t loom_verify_op(loom_verify_state_t* state,
                                    const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_verify_lookup_vtable(state, op->kind);
  if (!vtable) {
    // Unknown op kind — emit structured diagnostic.
    char kind_buffer[16];
    iree_snprintf(kind_buffer, sizeof(kind_buffer), "0x%04x", op->kind);
    loom_diagnostic_param_t params[] = {
        loom_param_u32(op->kind),
        loom_param_string(iree_make_cstring_view(kind_buffer)),
    };
    loom_verify_emit_structured(
        state, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 9),
        params, IREE_ARRAYSIZE(params));
    IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));
    // Still define results so downstream dominance checks don't cascade.
    for (uint16_t i = 0; i < op->result_count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_verify_define_value(state, loom_op_const_results(op)[i]));
    }
    return iree_ok_status();
  }

  loom_verify_op_declared_trait_consistency(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  const bool has_signature_scope = loom_verify_has_func_signature_scope(vtable);
  if (has_signature_scope) {
    IREE_RETURN_IF_ERROR(loom_verify_push_scope(state));
    uint16_t arg_count = 0;
    const loom_value_id_t* arg_ids =
        loom_verify_func_signature_arg_ids(op, vtable, &arg_count);
    for (uint16_t i = 0; i < arg_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_verify_define_value(state, arg_ids[i]));
    }
  }

  // Operand references must be valid and in scope. This must run
  // before any type-level checks since those read operand types.
  loom_verify_operand_dominance(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Structural checks: operand/result/attr/region counts match the
  // vtable declaration. Must pass before per-field checks below.
  loom_verify_op_structure(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Successor targets are direct block pointers. Reject malformed edges before
  // region-structure and dominance-sensitive checks observe CFG shape.
  loom_verify_successor_targets(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Per-instance trait callbacks may refine the static vtable declaration but
  // must preserve the same effect and optimization invariants as declared
  // traits. Run this after structural checks so callbacks can safely inspect
  // attributes.
  loom_verify_op_effective_trait_consistency(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Operand and result type payloads must satisfy core representation
  // invariants before table-driven constraints interpret them.
  loom_verify_op_type_well_formedness(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Per-field type constraint checks (operand/result types satisfy
  // the type category declared in the vtable descriptors).
  loom_verify_type_constraints(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // OperandDict format elements are stored as ordinary variadic operands plus
  // a key -> operand-ordinal DICT attribute. Verify that sidecar metadata is
  // canonical and exactly describes the operand segment before later passes
  // depend on keyed lookup.
  loom_verify_operand_dicts(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // SSA encoding references embedded in operand/result types must point to
  // valid LOOM_TYPE_ENCODING values. Ordinary references must be in scope;
  // result co-references and global declaration placeholders have explicit
  // rules in loom_verify_encoding_ref.
  loom_verify_encoding_refs(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Poison may flow through pure SSA computation, but it must not be consumed
  // by an operation that observes values outside ordinary use-def propagation.
  loom_verify_poison_boundaries(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Semantic constraints: table-driven (relation, property) interpreter.
  // Requires structural and type checks to have passed so field refs
  // resolve to valid values with the expected type categories.
  loom_verify_semantic_constraints(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Tied result validation and linear ownership consume marking.
  IREE_RETURN_IF_ERROR(loom_verify_tied_results(state, op, vtable));
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Symbol reference attributes must point to valid symbol table entries.
  IREE_RETURN_IF_ERROR(loom_verify_symbol_definition(state, op, vtable));
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));
  loom_verify_symbol_references(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Region structure: single-block invariants, terminator presence.
  loom_verify_region_structure(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Define result values. Must happen after all checks on this op
  // so that a result cannot appear to dominate its own defining op.
  if (has_signature_scope) {
    loom_verify_pop_scope(state);
  } else {
    for (uint16_t i = 0; i < op->result_count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_verify_define_value(state, loom_op_const_results(op)[i]));
    }
  }

  // Recurse into regions. Region contents see this op's results
  // (defined above) and all enclosing scope values.
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (loom_verify_at_error_limit(state)) break;
    IREE_RETURN_IF_ERROR(loom_verify_region(state, regions[i]));
  }
  loom_verify_func_purity_body_effects(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Op-specific verification callback. Runs last — all table-driven checks
  // have passed, so the op is structurally sound and type-correct.
  if (vtable->verify) {
    iree_diagnostic_emitter_t emitter = {
        .fn = loom_verify_diagnostic_emitter_fn,
        .user_data = state,
    };
    IREE_RETURN_IF_ERROR(vtable->verify(state->module, op, emitter));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Source resolver: table-based implementation
//===----------------------------------------------------------------------===//

bool loom_source_table_resolve(void* user_data, const loom_module_t* module,
                               loom_location_id_t location,
                               loom_source_range_t* out_range) {
  loom_source_table_resolver_t* table =
      (loom_source_table_resolver_t*)user_data;
  if (!table || table->count == 0) return false;
  if (location == LOOM_LOCATION_UNKNOWN) return false;

  // Look up the location entry from the module's location table.
  if ((iree_host_size_t)location >= module->locations.count) return false;
  const loom_location_entry_t* entry = &module->locations.entries[location];
  if (entry->kind != LOOM_LOCATION_FILE) return false;

  // Find the matching source buffer by source_id.
  const loom_source_entry_t* source_entry = NULL;
  for (iree_host_size_t i = 0; i < table->count; ++i) {
    if (table->entries[i].source_id == entry->file.source_id) {
      source_entry = &table->entries[i];
      break;
    }
  }
  if (!source_entry) return false;

  // Compute byte offsets from line/column into the source buffer.
  iree_host_size_t start_offset = loom_source_byte_offset(
      source_entry->source, entry->file.start_line, entry->file.start_col);
  iree_host_size_t end_offset = loom_source_byte_offset(
      source_entry->source, entry->file.end_line, entry->file.end_col);

  *out_range = (loom_source_range_t){
      .provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE,
      .filename = source_entry->filename,
      .source = source_entry->source,
      .start = start_offset,
      .end = end_offset,
      .start_line = entry->file.start_line,
      .start_column = entry->file.start_col,
      .end_line = entry->file.end_line,
      .end_column = entry->file.end_col,
  };
  return true;
}

//===----------------------------------------------------------------------===//
// Module and function verification entry points
//===----------------------------------------------------------------------===//

static void loom_verify_state_deinitialize(loom_verify_state_t* state);

static iree_status_t loom_verify_state_initialize(
    loom_verify_state_t* state, const loom_module_t* module,
    const loom_verify_options_t* options, loom_verify_result_t* out_result) {
  memset(state, 0, sizeof(*state));
  state->module = module;
  state->result = out_result;
  out_result->error_count = 0;
  out_result->warning_count = 0;

  if (options) {
    state->sink = options->sink;
    state->max_errors = options->max_errors;
    state->source_resolver = options->source_resolver;
  }

  iree_host_size_t value_count = module->values.count;
  // Ensure at least 1 word so we always have valid pointers.
  state->defined_bits_length = value_count > 0 ? (value_count + 63) / 64 : 1;

  // Initialize scratch arena from the module's block pool. All
  // verification-time allocations go here; bulk O(1) free on exit.
  iree_arena_initialize(module->arena.block_pool, &state->arena);

  // Allocate bitsets and defined stack. iree_arena_allocate_array
  // uses checked multiplication (overflow → RESOURCE_EXHAUSTED).
  // On any failure, deinitialize returns all arena blocks to the pool.
  iree_status_t status =
      iree_arena_allocate_array(&state->arena, state->defined_bits_length,
                                sizeof(uint64_t), (void**)&state->defined_bits);
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(
        &state->arena, state->defined_bits_length, sizeof(uint64_t),
        (void**)&state->consumed_bits);
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t consuming_op_count = value_count > 0 ? value_count : 1;
    status = iree_arena_allocate_array(&state->arena, consuming_op_count,
                                       sizeof(state->consuming_ops[0]),
                                       (void**)&state->consuming_ops);
    if (iree_status_is_ok(status)) {
      memset(state->consuming_ops, 0,
             consuming_op_count * sizeof(state->consuming_ops[0]));
    }
  }
  if (iree_status_is_ok(status)) {
    memset(state->defined_bits, 0,
           state->defined_bits_length * sizeof(uint64_t));
    memset(state->consumed_bits, 0,
           state->defined_bits_length * sizeof(uint64_t));
  }
  if (iree_status_is_ok(status)) {
    // Allocate defined stack. Start with value_count/4 capacity (most
    // values are results that get defined). Grows dynamically via
    // iree_arena_grow_array if the heuristic undersizes.
    state->defined_stack_capacity = value_count > 16 ? value_count / 4 : 16;
    status = iree_arena_allocate_array(
        &state->arena, state->defined_stack_capacity, sizeof(uint32_t),
        (void**)&state->defined_stack);
  }
  if (!iree_status_is_ok(status)) {
    loom_verify_state_deinitialize(state);
    return status;
  }

  return iree_ok_status();
}

static void loom_verify_state_deinitialize(loom_verify_state_t* state) {
  IREE_ASSERT(iree_status_is_ok(state->diagnostic_status));
  iree_status_ignore(state->diagnostic_status);
  iree_arena_deinitialize(&state->arena);
}

iree_status_t loom_verify_module(const loom_module_t* module,
                                 const loom_verify_options_t* options,
                                 loom_verify_result_t* out_result) {
  if (!module || !out_result) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module and out_result must be non-NULL");
  }
  if (!module->context) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "module has no context (vtables unavailable)");
  }

  loom_verify_state_t state;
  IREE_RETURN_IF_ERROR(
      loom_verify_state_initialize(&state, module, options, out_result));

  // Walk the module body.
  if (module->body) {
    // Module-level invariant: only symbol-defining ops (func.def,
    // func.decl, etc.) are allowed at the top level. Ops without
    // LOOM_TRAIT_SYMBOL_DEFINE belong inside function bodies.
    if (module->body->block_count > 0) {
      loom_block_t* entry = loom_region_entry_block(module->body);
      const loom_op_t* op = NULL;
      loom_block_for_each_op(entry, op) {
        const loom_op_vtable_t* vtable =
            loom_verify_lookup_vtable(&state, op->kind);
        if (vtable &&
            !iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE)) {
          iree_string_view_t op_name = loom_op_vtable_name(vtable);
          loom_diagnostic_param_t params[] = {
              loom_param_string(op_name),
          };
          loom_verify_emit_structured(
              &state, op,
              loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 11), params,
              IREE_ARRAYSIZE(params));
          iree_status_t diagnostic_status =
              loom_verify_pending_diagnostic_status(&state);
          if (!iree_status_is_ok(diagnostic_status)) {
            loom_verify_state_deinitialize(&state);
            return diagnostic_status;
          }
        }
      }
    }

    iree_status_t walk_status = loom_verify_region(&state, module->body);
    if (!iree_status_is_ok(walk_status)) {
      loom_verify_state_deinitialize(&state);
      return walk_status;
    }
  }

  loom_verify_state_deinitialize(&state);
  return iree_ok_status();
}

iree_status_t loom_verify_function(const loom_module_t* module,
                                   loom_func_like_t function,
                                   const loom_verify_options_t* options,
                                   loom_verify_result_t* out_result) {
  if (!module || !loom_func_like_isa(function) || !out_result) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "module, function, and out_result must be non-NULL");
  }
  if (!module->context) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "module has no context (vtables unavailable)");
  }

  loom_verify_state_t state;
  IREE_RETURN_IF_ERROR(
      loom_verify_state_initialize(&state, module, options, out_result));

  iree_status_t verify_status = loom_verify_op(&state, function.op);
  if (!iree_status_is_ok(verify_status)) {
    loom_verify_state_deinitialize(&state);
    return verify_status;
  }

  loom_verify_state_deinitialize(&state);
  return iree_ok_status();
}
