// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/verify.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/internal/unicode.h"
#include "loom/error/renderer.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

//===----------------------------------------------------------------------===//
// Internal verification state
//===----------------------------------------------------------------------===//

// Maximum region nesting depth. Each nested region pushes a watermark
// onto the scope stack. 32 levels covers any realistic IR (function →
// loop → branch → ...).
#define LOOM_VERIFY_MAX_SCOPE_DEPTH 32

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
                                      loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID) return;
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

//===----------------------------------------------------------------------===//
// Diagnostic emission
//===----------------------------------------------------------------------===//

static bool loom_verify_at_error_limit(const loom_verify_state_t* state) {
  return state->max_errors > 0 &&
         state->result->error_count >= state->max_errors;
}

// Resolves an op's location to a source range via the configured resolver.
static bool loom_verify_resolve_location(const loom_verify_state_t* state,
                                         const loom_op_t* op,
                                         loom_source_range_t* out_range) {
  if (!op) return false;
  return loom_source_resolve(state->source_resolver, state->module,
                             op->location, out_range);
}

// Maximum per-token highlight ranges per diagnostic.
#define LOOM_VERIFY_MAX_HIGHLIGHTS 8

// Callback state for collecting field highlights during printing.
// The verifier populates |wanted_refs| with the field refs it wants
// highlighted, then passes this as user_data to the printer's field
// callback. The callback records byte ranges for matches.
typedef struct loom_highlight_collector_t {
  const loom_field_ref_t* wanted_refs;
  iree_host_size_t wanted_count;
  loom_highlight_range_t* highlights;
  iree_host_size_t highlight_count;
} loom_highlight_collector_t;

static void loom_highlight_field_callback(void* user_data,
                                          loom_field_ref_t field_ref,
                                          iree_host_size_t start,
                                          iree_host_size_t end) {
  loom_highlight_collector_t* collector =
      (loom_highlight_collector_t*)user_data;
  if (collector->highlight_count >= LOOM_VERIFY_MAX_HIGHLIGHTS) return;
  for (iree_host_size_t i = 0; i < collector->wanted_count; ++i) {
    if (collector->wanted_refs[i] == field_ref) {
      collector->highlights[collector->highlight_count].start = start;
      collector->highlights[collector->highlight_count].end = end;
      ++collector->highlight_count;
      return;
    }
  }
}

// Derives field refs from STRING params whose values are "operand N",
// "result N", or "attribute N". These are field references that the
// verifier built from known field indices — safe to parse because we
// constructed them ourselves in a known format.
static iree_host_size_t loom_derive_field_refs(
    const loom_diagnostic_param_t* params, iree_host_size_t param_count,
    loom_field_ref_t* out_refs) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0;
       i < param_count && count < LOOM_VERIFY_MAX_HIGHLIGHTS; ++i) {
    if (params[i].kind != LOOM_PARAM_STRING) continue;
    iree_string_view_t value = params[i].string;
    uint8_t category = 0;
    const char* number_start = NULL;
    if (iree_string_view_starts_with(value, IREE_SV("operand "))) {
      category = LOOM_FIELD_OPERAND;
      number_start = value.data + 8;
    } else if (iree_string_view_starts_with(value, IREE_SV("result "))) {
      category = LOOM_FIELD_RESULT;
      number_start = value.data + 7;
    } else if (iree_string_view_starts_with(value, IREE_SV("attribute "))) {
      category = LOOM_FIELD_ATTR;
      number_start = value.data + 10;
    } else {
      continue;
    }
    uint16_t index = 0;
    while (number_start < value.data + value.size && *number_start >= '0' &&
           *number_start <= '9') {
      index = index * 10 + (uint16_t)(*number_start - '0');
      ++number_start;
    }
    out_refs[count++] = LOOM_FIELD_REF(category, index);
  }
  return count;
}

// Emits a structured diagnostic with typed error def and params.
//
// Source resolution strategy:
//   1. Try the configured source resolver (original source text).
//   2. If that fails, print the op to text and use the printed
//      representation as the diagnostic source.
//
// Per-token highlighting is derived automatically from the params:
// STRING params whose values match "operand N" / "result N" /
// "attribute N" are mapped to field refs and passed to the printer's
// field callback, which records their byte ranges for caret output.
static void loom_verify_emit_structured(loom_verify_state_t* state,
                                        const loom_op_t* op,
                                        const loom_error_def_t* error,
                                        const loom_diagnostic_param_t* params,
                                        iree_host_size_t param_count) {
  if (!iree_status_is_ok(state->diagnostic_status)) return;

  if (error->severity == LOOM_DIAGNOSTIC_ERROR) {
    ++state->result->error_count;
  } else if (error->severity == LOOM_DIAGNOSTIC_WARNING) {
    ++state->result->warning_count;
  }

  if (!state->sink.fn) return;

  loom_diagnostic_t diagnostic = {0};
  diagnostic.severity = error->severity;
  diagnostic.error = error;
  diagnostic.params = params;
  diagnostic.param_count = param_count;
  diagnostic.emitter = LOOM_EMITTER_VERIFIER;

  // Derive field refs from the params for per-token highlighting.
  loom_field_ref_t highlight_refs[LOOM_VERIFY_MAX_HIGHLIGHTS];
  iree_host_size_t highlight_ref_count =
      loom_derive_field_refs(params, param_count, highlight_refs);

  // Try the source resolver first (original source text).
  bool resolved =
      loom_verify_resolve_location(state, op, &diagnostic.source_location);
  if (resolved) {
    diagnostic.origin = diagnostic.source_location;
  }

  // Fallback: print the op and use the printed text as the source.
  // The printer's field callback records byte ranges for the derived
  // field refs, giving per-token caret underlines.
  iree_string_builder_t op_text_builder;
  loom_highlight_range_t highlights[LOOM_VERIFY_MAX_HIGHLIGHTS];
  loom_highlight_collector_t collector = {0};
  bool printed_op = false;
  if (!resolved && op) {
    iree_string_builder_initialize(state->module->context->allocator,
                                   &op_text_builder);

    collector.wanted_refs = highlight_refs;
    collector.wanted_count = highlight_ref_count;
    collector.highlights = highlights;
    collector.highlight_count = 0;

    loom_print_field_callback_t field_callback = {
        .fn = highlight_ref_count > 0 ? loom_highlight_field_callback : NULL,
        .user_data = &collector,
    };
    iree_status_t print_status = loom_text_print_operation_with_field_callback(
        state->module, op, &op_text_builder, LOOM_TEXT_PRINT_USE_ALIASES,
        field_callback);
    if (!iree_status_is_ok(print_status)) {
      // Printing failed (OOM, etc.). Use a static fallback so the
      // diagnostic still has something to display with carets.
      iree_status_ignore(print_status);
      iree_string_builder_deinitialize(&op_text_builder);
      static const char kFallback[] = "<failed to print op>";
      loom_source_range_t fallback_range = {
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

static iree_status_t loom_verify_diagnostic_emitter_fn(
    void* user_data, const loom_op_t* op, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count) {
  loom_verify_state_t* state = (loom_verify_state_t*)user_data;
  loom_verify_emit_structured(state, op, error, params, param_count);
  return loom_verify_pending_diagnostic_status(state);
}

//===----------------------------------------------------------------------===//
// Vtable lookup
//===----------------------------------------------------------------------===//

static const loom_op_vtable_t* loom_verify_lookup_vtable(
    const loom_verify_state_t* state, loom_op_kind_t kind) {
  return loom_context_resolve_op(state->module->context, kind);
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
      loom_verify_emit_structured(state, op, &loom_err_structure_001, params,
                                  3);
    }
  } else {
    if (op->operand_count != vtable->fixed_operand_count) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(op_name),
          loom_param_u32(op->operand_count),
          loom_param_u32(vtable->fixed_operand_count),
      };
      loom_verify_emit_structured(state, op, &loom_err_structure_001, params,
                                  3);
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
      loom_verify_emit_structured(state, op, &loom_err_structure_002, params,
                                  3);
    }
  } else {
    if (op->result_count != vtable->fixed_result_count) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(op_name),
          loom_param_u32(op->result_count),
          loom_param_u32(vtable->fixed_result_count),
      };
      loom_verify_emit_structured(state, op, &loom_err_structure_002, params,
                                  3);
    }
  }

  // Check attribute count.
  if (op->attribute_count != vtable->attribute_count) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(op_name),
        loom_param_u32(op->attribute_count),
        loom_param_u32(vtable->attribute_count),
    };
    loom_verify_emit_structured(state, op, &loom_err_structure_003, params, 3);
  }

  // Check region count.
  if (op->region_count != vtable->region_count) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(op_name),
        loom_param_u32(op->region_count),
        loom_param_u32(vtable->region_count),
    };
    loom_verify_emit_structured(state, op, &loom_err_structure_004, params, 3);
  }
}

//===----------------------------------------------------------------------===//
// Type constraint checks
//===----------------------------------------------------------------------===//

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
          // Build operand name like "operand 0".
          char operand_name_buffer[32];
          snprintf(operand_name_buffer, sizeof(operand_name_buffer),
                   "operand %u", j);
          loom_diagnostic_param_t params[] = {
              loom_param_string(iree_make_cstring_view(operand_name_buffer)),
              loom_param_type(type),
              loom_param_string(iree_make_cstring_view(
                  loom_type_constraint_name(constraint))),
          };
          loom_verify_emit_structured(state, op, &loom_err_type_003, params, 3);
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
          char result_name_buffer[32];
          snprintf(result_name_buffer, sizeof(result_name_buffer), "result %u",
                   j);
          loom_diagnostic_param_t params[] = {
              loom_param_string(iree_make_cstring_view(result_name_buffer)),
              loom_param_type(type),
              loom_param_string(iree_make_cstring_view(
                  loom_type_constraint_name(constraint))),
          };
          loom_verify_emit_structured(state, op, &loom_err_type_004, params, 3);
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
      if (optional && attrs[i].raw == 0) continue;
      iree_string_view_t attr_name = loom_bstring_view(descriptor->name);
      if (attrs[i].kind != descriptor->attr_kind) {
        loom_diagnostic_param_t params[] = {
            loom_param_string(attr_name),
            loom_param_u32(attrs[i].kind),
            loom_param_u32(descriptor->attr_kind),
        };
        loom_verify_emit_structured(state, op, &loom_err_type_005, params, 3);
      }
      if (attrs[i].kind == LOOM_ATTR_ENUM && descriptor->enum_case_count > 0) {
        uint8_t case_index = (uint8_t)attrs[i].raw;
        if (case_index >= descriptor->enum_case_count) {
          loom_diagnostic_param_t params[] = {
              loom_param_string(attr_name),
              loom_param_u32(case_index),
              loom_param_u32(descriptor->enum_case_count),
          };
          loom_verify_emit_structured(state, op, &loom_err_structure_010,
                                      params, 3);
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Semantic constraint interpreter
//===----------------------------------------------------------------------===//

// Resolves a field reference to a name for diagnostic params.
// Returns the field name from the vtable format elements if available,
// otherwise a positional name like "operand 0" or "result 1".
static iree_string_view_t loom_verify_field_name(const loom_op_t* op,
                                                 const loom_op_vtable_t* vtable,
                                                 uint8_t field_ref,
                                                 char* buffer,
                                                 iree_host_size_t buffer_size) {
  uint8_t category = LOOM_FIELD_REF_CATEGORY(field_ref);
  uint8_t index = LOOM_FIELD_REF_INDEX(field_ref);
  const char* prefix =
      (category == LOOM_FIELD_OPERAND)
          ? "operand"
          : ((category == LOOM_FIELD_RESULT) ? "result" : "field");
  snprintf(buffer, buffer_size, "%s %u", prefix, index);
  return iree_make_cstring_view(buffer);
}

// Like loom_verify_field_name but for an indexed element of a variadic
// field. Produces names like "operand 0[2]" or "result 1[0]".
static iree_string_view_t loom_verify_indexed_field_name(
    const loom_op_t* op, const loom_op_vtable_t* vtable, uint8_t field_ref,
    uint16_t element_index, char* buffer, iree_host_size_t buffer_size) {
  uint8_t category = LOOM_FIELD_REF_CATEGORY(field_ref);
  uint8_t index = LOOM_FIELD_REF_INDEX(field_ref);
  const char* prefix =
      (category == LOOM_FIELD_OPERAND)
          ? "operand"
          : ((category == LOOM_FIELD_RESULT) ? "result" : "field");
  snprintf(buffer, buffer_size, "%s %u[%u]", prefix, index, element_index);
  return iree_make_cstring_view(buffer);
}

// Returns true if the specified property of two types is equal.
static bool loom_constraint_property_equals(
    loom_type_t a, loom_type_t b, loom_constraint_property_t property) {
  switch ((enum loom_constraint_property_e)property) {
    case LOOM_PROPERTY_TYPE:
      return memcmp(&a, &b, sizeof(loom_type_t)) == 0;
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
      return &loom_err_type_001;
    case LOOM_PROPERTY_ELEMENT_TYPE:
      return &loom_err_type_002;
    case LOOM_PROPERTY_ENCODING:
      return &loom_err_encoding_001;
    case LOOM_PROPERTY_SHAPE:
      return &loom_err_shape_002;
    case LOOM_PROPERTY_RANK:
      return &loom_err_shape_001;
    default:
      return &loom_err_type_001;
  }
}

// Builds diagnostic params for a pairwise property mismatch.
// Different properties produce different param schemas:
//   TYPE: (name_a, type_a, name_b, type_b)
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
      op, vtable, ref_a, name_a_buffer, sizeof(name_a_buffer));
  iree_string_view_t name_b = loom_verify_field_name(
      op, vtable, ref_b, name_b_buffer, sizeof(name_b_buffer));
  const loom_error_def_t* error =
      constraint->error ? constraint->error
                        : loom_pairwise_eq_default_error(constraint->property);

  switch ((enum loom_constraint_property_e)constraint->property) {
    case LOOM_PROPERTY_TYPE: {
      loom_diagnostic_param_t params[] = {
          loom_param_string(name_a),
          loom_param_type(type_a),
          loom_param_string(name_b),
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
          loom_param_string(name_a),
          loom_param_type(element_a),
          loom_param_string(name_b),
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
          loom_param_string(name_a),
          loom_param_string(name_b),
      };
      loom_verify_emit_structured(
          state, op, error, params,
          error->param_count < 2 ? error->param_count : 2);
      break;
    }
    case LOOM_PROPERTY_RANK: {
      loom_diagnostic_param_t params[] = {
          loom_param_string(name_a),
          loom_param_i64(loom_type_rank(type_a)),
          loom_param_string(name_b),
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

static void loom_verify_semantic_constraint(
    loom_verify_state_t* state, const loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  switch ((enum loom_constraint_relation_e)constraint->relation) {
    case LOOM_RELATION_PAIRWISE_EQ: {
      if (constraint->arg_count < 2) break;
      // Collect the reference type from the first element of the first
      // arg. For variadic args, the first element is at the field's
      // start index.
      loom_value_id_t first_id =
          loom_verify_resolve_value_field(op, constraint->args[0]);
      if (first_id == LOOM_VALUE_ID_INVALID) break;
      loom_type_t first_type = loom_verify_value_type(state, first_id);
      uint8_t first_ref = constraint->args[0];
      // Track whether the first value itself was from a variadic. If
      // so, we also need to check the remaining elements of arg 0.
      bool first_is_variadic =
          loom_verify_is_variadic_field(vtable, constraint->args[0]);
      bool found_mismatch = false;
      // Check remaining elements of the first arg's variadic range.
      if (first_is_variadic) {
        uint8_t category = LOOM_FIELD_REF_CATEGORY(first_ref);
        uint8_t start_index = LOOM_FIELD_REF_INDEX(first_ref);
        uint16_t count = (category == LOOM_FIELD_OPERAND) ? op->operand_count
                                                          : op->result_count;
        const loom_value_id_t* values = (category == LOOM_FIELD_OPERAND)
                                            ? loom_op_const_operands(op)
                                            : loom_op_const_results(op);
        for (uint16_t vi = start_index + 1; vi < count; ++vi) {
          loom_type_t other_type = loom_verify_value_type(state, values[vi]);
          if (!loom_constraint_property_equals(first_type, other_type,
                                               constraint->property)) {
            // Emit with indexed field names for both sides.
            char name_a_buffer[32];
            char name_b_buffer[32];
            iree_string_view_t name_a = loom_verify_indexed_field_name(
                op, vtable, first_ref, 0, name_a_buffer, sizeof(name_a_buffer));
            iree_string_view_t name_b = loom_verify_indexed_field_name(
                op, vtable, first_ref, (uint16_t)(vi - start_index),
                name_b_buffer, sizeof(name_b_buffer));
            const loom_error_def_t* error =
                constraint->error
                    ? constraint->error
                    : loom_pairwise_eq_default_error(constraint->property);
            loom_diagnostic_param_t params[] = {
                loom_param_string(name_a),
                loom_param_type(first_type),
                loom_param_string(name_b),
                loom_param_type(other_type),
            };
            loom_verify_emit_structured(
                state, op, error, params,
                error->param_count < 4 ? error->param_count : 4);
            found_mismatch = true;
            break;
          }
        }
      }
      if (found_mismatch) break;
      // Check subsequent constraint args against the first type.
      for (uint8_t i = 1; i < constraint->arg_count; ++i) {
        uint8_t arg_ref = constraint->args[i];
        bool is_variadic = loom_verify_is_variadic_field(vtable, arg_ref);
        if (!is_variadic) {
          // Scalar field: single comparison.
          loom_value_id_t other_id =
              loom_verify_resolve_value_field(op, arg_ref);
          loom_type_t other_type = loom_verify_value_type(state, other_id);
          if (!loom_constraint_property_equals(first_type, other_type,
                                               constraint->property)) {
            loom_verify_emit_pairwise_mismatch(state, op, vtable, constraint,
                                               first_type, other_type,
                                               first_ref, arg_ref);
            found_mismatch = true;
            break;
          }
        } else {
          // Variadic field: check each element.
          uint8_t category = LOOM_FIELD_REF_CATEGORY(arg_ref);
          uint8_t start_index = LOOM_FIELD_REF_INDEX(arg_ref);
          uint16_t count = (category == LOOM_FIELD_OPERAND) ? op->operand_count
                                                            : op->result_count;
          const loom_value_id_t* values = (category == LOOM_FIELD_OPERAND)
                                              ? loom_op_const_operands(op)
                                              : loom_op_const_results(op);
          for (uint16_t vi = start_index; vi < count; ++vi) {
            loom_type_t other_type = loom_verify_value_type(state, values[vi]);
            if (!loom_constraint_property_equals(first_type, other_type,
                                                 constraint->property)) {
              // Emit with the first arg's name and the indexed variadic
              // element's name.
              char name_a_buffer[32];
              char name_b_buffer[32];
              iree_string_view_t name_a;
              if (first_is_variadic) {
                name_a = loom_verify_indexed_field_name(op, vtable, first_ref,
                                                        0, name_a_buffer,
                                                        sizeof(name_a_buffer));
              } else {
                name_a =
                    loom_verify_field_name(op, vtable, first_ref, name_a_buffer,
                                           sizeof(name_a_buffer));
              }
              iree_string_view_t name_b = loom_verify_indexed_field_name(
                  op, vtable, arg_ref, (uint16_t)(vi - start_index),
                  name_b_buffer, sizeof(name_b_buffer));
              const loom_error_def_t* error =
                  constraint->error
                      ? constraint->error
                      : loom_pairwise_eq_default_error(constraint->property);
              loom_diagnostic_param_t params[] = {
                  loom_param_string(name_a),
                  loom_param_type(first_type),
                  loom_param_string(name_b),
                  loom_param_type(other_type),
              };
              loom_verify_emit_structured(
                  state, op, error, params,
                  error->param_count < 4 ? error->param_count : 4);
              found_mismatch = true;
              break;
            }
          }
          if (found_mismatch) break;
        }
      }
      break;
    }

    case LOOM_RELATION_ALL_SAME: {
      if (constraint->arg_count < 1) break;
      uint8_t field_ref = constraint->args[0];
      uint8_t category = LOOM_FIELD_REF_CATEGORY(field_ref);
      uint8_t start_index = LOOM_FIELD_REF_INDEX(field_ref);
      uint16_t count = 0;
      const loom_value_id_t* values = NULL;
      if (category == LOOM_FIELD_OPERAND && start_index <= op->operand_count) {
        values = loom_op_const_operands(op) + start_index;
        count = (uint16_t)(op->operand_count - start_index);
      } else if (category == LOOM_FIELD_RESULT &&
                 start_index <= op->result_count) {
        values = loom_op_const_results(op) + start_index;
        count = (uint16_t)(op->result_count - start_index);
      }
      if (count > 1) {
        loom_type_t first_type = loom_verify_value_type(state, values[0]);
        for (uint16_t i = 1; i < count; ++i) {
          loom_type_t other_type = loom_verify_value_type(state, values[i]);
          if (!loom_constraint_property_equals(first_type, other_type,
                                               constraint->property)) {
            const loom_error_def_t* error =
                constraint->error ? constraint->error : &loom_err_shape_003;
            loom_diagnostic_param_t params[] = {
                loom_param_type(first_type),
                loom_param_u32(i),
                loom_param_type(other_type),
            };
            loom_verify_emit_structured(
                state, op, error, params,
                error->param_count < 3 ? error->param_count : 3);
            break;
          }
        }
      }
      break;
    }

    case LOOM_RELATION_COUNT_MATCHES_RANK: {
      // args[0] = shaped value, args[1] = variadic value field.
      if (constraint->arg_count < 2) break;
      loom_type_t shaped_type = loom_verify_value_type(
          state, loom_verify_resolve_value_field(op, constraint->args[0]));
      uint16_t variadic_count =
          loom_verify_variadic_count(op, vtable, constraint->args[1]);
      uint8_t rank = loom_type_rank(shaped_type);
      if (variadic_count != rank) {
        char name_buffer[32];
        iree_string_view_t operand_name = loom_verify_field_name(
            op, vtable, constraint->args[0], name_buffer, sizeof(name_buffer));
        const loom_error_def_t* error =
            constraint->error ? constraint->error : &loom_err_subrange_001;
        loom_diagnostic_param_t params[] = {
            loom_param_string(operand_name),
            loom_param_u32(variadic_count),
            loom_param_i64(rank),
        };
        loom_verify_emit_structured(
            state, op, error, params,
            error->param_count < 3 ? error->param_count : 3);
      }
      break;
    }

    case LOOM_RELATION_ATTR_IN_RANGE_RANK: {
      // args[0] = shaped value field, args[1] = attr field (i64 index).
      if (constraint->arg_count < 2) break;
      loom_type_t shaped_type = loom_verify_value_type(
          state, loom_verify_resolve_value_field(op, constraint->args[0]));
      uint8_t attr_index = LOOM_FIELD_REF_INDEX(constraint->args[1]);
      if (attr_index < op->attribute_count) {
        int64_t dim_index = loom_attr_as_i64(loom_op_attrs(op)[attr_index]);
        uint8_t rank = loom_type_rank(shaped_type);
        if (dim_index < 0 || dim_index >= rank) {
          const loom_error_def_t* error =
              constraint->error ? constraint->error : &loom_err_subrange_002;
          loom_diagnostic_param_t params[] = {
              loom_param_i64(dim_index),
              loom_param_i64(rank),
          };
          loom_verify_emit_structured(
              state, op, error, params,
              error->param_count < 2 ? error->param_count : 2);
        }
      }
      break;
    }

    case LOOM_RELATION_REGION_ARG_COUNT: {
      // args[0] = region, args[1] = value field (count source).
      if (constraint->arg_count < 2) break;
      uint8_t region_index = LOOM_FIELD_REF_INDEX(constraint->args[0]);
      if (region_index >= op->region_count) break;
      loom_region_t* region = loom_op_regions(op)[region_index];
      if (!region || region->block_count == 0) break;
      uint16_t block_arg_count = region->blocks[0].arg_count;
      uint16_t input_count =
          loom_verify_variadic_count(op, vtable, constraint->args[1]);
      if (block_arg_count != input_count) {
        const loom_error_def_t* error =
            constraint->error ? constraint->error : &loom_err_structure_007;
        loom_diagnostic_param_t params[] = {
            loom_param_u32(block_arg_count),
            loom_param_u32(input_count),
        };
        loom_verify_emit_structured(
            state, op, error, params,
            error->param_count < 2 ? error->param_count : 2);
      }
      break;
    }

    case LOOM_RELATION_REGION_ARG_MATCH: {
      // args[0] = region, args[1] = value field (type source).
      if (constraint->arg_count < 2) break;
      uint8_t region_index = LOOM_FIELD_REF_INDEX(constraint->args[0]);
      if (region_index >= op->region_count) break;
      loom_region_t* region = loom_op_regions(op)[region_index];
      if (!region || region->block_count == 0) break;
      loom_block_t* entry = &region->blocks[0];

      uint8_t input_ref = constraint->args[1];
      uint8_t input_category = LOOM_FIELD_REF_CATEGORY(input_ref);
      uint8_t input_start = LOOM_FIELD_REF_INDEX(input_ref);
      const loom_value_id_t* input_values = NULL;
      uint16_t input_count = 0;
      if (input_category == LOOM_FIELD_OPERAND &&
          input_start <= op->operand_count) {
        input_values = loom_op_const_operands(op) + input_start;
        input_count = (uint16_t)(op->operand_count - input_start);
      }

      uint16_t check_count =
          entry->arg_count < input_count ? entry->arg_count : input_count;
      for (uint16_t i = 0; i < check_count; ++i) {
        loom_type_t block_arg_type =
            loom_verify_value_type(state, entry->arg_ids[i]);
        loom_type_t input_type = loom_verify_value_type(state, input_values[i]);
        if (!loom_constraint_property_equals(block_arg_type, input_type,
                                             constraint->property)) {
          const loom_error_def_t* error =
              constraint->error ? constraint->error : &loom_err_type_008;
          loom_type_t expected_type =
              loom_type_scalar(loom_type_element_type(input_type));
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
      break;
    }

    case LOOM_RELATION_YIELD_COUNT: {
      // args[0] = region, args[1] = result field.
      if (constraint->arg_count < 2) break;
      uint8_t region_index = LOOM_FIELD_REF_INDEX(constraint->args[0]);
      if (region_index >= op->region_count) break;
      loom_region_t* region = loom_op_regions(op)[region_index];
      if (!region || region->block_count == 0) break;
      loom_block_t* entry = &region->blocks[0];
      if (entry->op_count == 0) break;

      const loom_op_t* terminator = entry->ops[entry->op_count - 1];
      uint16_t yield_count = terminator->operand_count;

      uint16_t result_count =
          loom_verify_variadic_count(op, vtable, constraint->args[1]);
      if (LOOM_FIELD_REF_CATEGORY(constraint->args[1]) == LOOM_FIELD_RESULT &&
          LOOM_FIELD_REF_INDEX(constraint->args[1]) <
              vtable->fixed_result_count) {
        result_count = 1;
      }

      if (yield_count != result_count) {
        const loom_error_def_t* error =
            constraint->error ? constraint->error : &loom_err_structure_008;
        loom_diagnostic_param_t params[] = {
            loom_param_u32(yield_count),
            loom_param_u32(result_count),
        };
        loom_verify_emit_structured(
            state, op, error, params,
            error->param_count < 2 ? error->param_count : 2);
      }
      break;
    }

    case LOOM_RELATION_YIELD_MATCH: {
      // args[0] = region, args[1] = result field.
      if (constraint->arg_count < 2) break;
      uint8_t region_index = LOOM_FIELD_REF_INDEX(constraint->args[0]);
      if (region_index >= op->region_count) break;
      loom_region_t* region = loom_op_regions(op)[region_index];
      if (!region || region->block_count == 0) break;
      loom_block_t* entry = &region->blocks[0];
      if (entry->op_count == 0) break;

      const loom_op_t* terminator = entry->ops[entry->op_count - 1];
      const loom_value_id_t* yield_operands =
          loom_op_const_operands(terminator);

      uint8_t result_ref = constraint->args[1];
      uint8_t result_start = LOOM_FIELD_REF_INDEX(result_ref);
      const loom_value_id_t* result_values =
          loom_op_const_results(op) + result_start;
      uint16_t result_count = (result_start < op->result_count)
                                  ? (uint16_t)(op->result_count - result_start)
                                  : 0;

      uint16_t check_count = terminator->operand_count < result_count
                                 ? terminator->operand_count
                                 : result_count;
      for (uint16_t i = 0; i < check_count; ++i) {
        loom_type_t yield_type =
            loom_verify_value_type(state, yield_operands[i]);
        loom_type_t result_type =
            loom_verify_value_type(state, result_values[i]);
        if (!loom_constraint_property_equals(yield_type, result_type,
                                             constraint->property)) {
          const loom_error_def_t* error =
              constraint->error ? constraint->error : &loom_err_type_009;
          loom_diagnostic_param_t params[] = {
              loom_param_type(yield_type),
              loom_param_type(result_type),
          };
          loom_verify_emit_structured(
              state, op, error, params,
              error->param_count < 2 ? error->param_count : 2);
        }
      }
      break;
    }
    default:
      break;
  }
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
  iree_string_view_t op_name = loom_op_vtable_name(vtable);
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    loom_value_id_t value_id = operands[i];
    if (value_id == LOOM_VALUE_ID_INVALID) continue;
    if (value_id >= state->module->values.count) {
      loom_diagnostic_param_t params[] = {
          loom_param_u32(value_id),
          loom_param_u32((uint32_t)state->module->values.count),
      };
      loom_verify_emit_structured(state, op, &loom_err_dominance_003, params,
                                  2);
      continue;
    }
    if (!loom_bitset_test(state->defined_bits, value_id)) {
      iree_string_view_t value_name = loom_verify_value_name(state, value_id);
      loom_diagnostic_param_t params[] = {
          loom_param_string(value_name),
      };
      loom_verify_emit_structured(state, op, &loom_err_dominance_001, params,
                                  1);
    }
    if (loom_bitset_test(state->consumed_bits, value_id)) {
      iree_string_view_t value_name = loom_verify_value_name(state, value_id);
      loom_diagnostic_param_t params[] = {
          loom_param_string(value_name),
          loom_param_string(op_name),
      };
      loom_verify_emit_structured(state, op, &loom_err_dominance_002, params,
                                  2);
    }
  }
}

//===----------------------------------------------------------------------===//
// SSA encoding reference validation
//===----------------------------------------------------------------------===//

// Validates a single SSA encoding reference embedded in a value's type.
// If the type carries LOOM_ENCODING_FLAG_SSA, the encoding_id is a
// value_id that must be in range, defined in scope, and have type
// LOOM_TYPE_ENCODING. This catches corrupted modules, builder bugs,
// and pass bugs that create types with dangling encoding references.
static void loom_verify_encoding_ref(loom_verify_state_t* state,
                                     const loom_op_t* op, loom_type_t type,
                                     const char* field_name) {
  if (!loom_type_has_ssa_encoding(type)) return;
  uint16_t encoding_value_id = loom_type_encoding_value_id(type);
  if (encoding_value_id >= state->module->values.count) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(iree_make_cstring_view(field_name)),
        loom_param_u32(encoding_value_id),
        loom_param_u32((uint32_t)state->module->values.count),
    };
    loom_verify_emit_structured(state, op, &loom_err_encoding_003, params, 3);
    return;
  }
  if (!loom_bitset_test(state->defined_bits, encoding_value_id)) {
    iree_string_view_t value_name =
        loom_verify_value_name(state, encoding_value_id);
    loom_diagnostic_param_t params[] = {
        loom_param_string(iree_make_cstring_view(field_name)),
        loom_param_string(value_name),
    };
    loom_verify_emit_structured(state, op, &loom_err_encoding_004, params, 2);
    return;
  }
  loom_type_t encoding_type =
      loom_module_value_type(state->module, encoding_value_id);
  if (!loom_type_is_encoding(encoding_type)) {
    iree_string_view_t value_name =
        loom_verify_value_name(state, encoding_value_id);
    loom_diagnostic_param_t params[] = {
        loom_param_string(iree_make_cstring_view(field_name)),
        loom_param_string(value_name),
        loom_param_type(encoding_type),
    };
    loom_verify_emit_structured(state, op, &loom_err_encoding_005, params, 3);
  }
}

// Checks SSA encoding references in all operand and result types of an op.
static void loom_verify_encoding_refs(loom_verify_state_t* state,
                                      const loom_op_t* op) {
  char name_buffer[32];
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (operands[i] == LOOM_VALUE_ID_INVALID) continue;
    if (operands[i] >= state->module->values.count) continue;
    loom_type_t type = loom_module_value_type(state->module, operands[i]);
    snprintf(name_buffer, sizeof(name_buffer), "operand %u", i);
    loom_verify_encoding_ref(state, op, type, name_buffer);
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID) continue;
    if (results[i] >= state->module->values.count) continue;
    loom_type_t type = loom_module_value_type(state->module, results[i]);
    snprintf(name_buffer, sizeof(name_buffer), "result %u", i);
    loom_verify_encoding_ref(state, op, type, name_buffer);
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
    loom_value_id_t arg_id = block->arg_ids[a];
    if (arg_id == LOOM_VALUE_ID_INVALID) continue;
    if (arg_id >= state->module->values.count) continue;
    loom_type_t type = loom_module_value_type(state->module, arg_id);
    snprintf(name_buffer, sizeof(name_buffer), "block arg %u", a);
    loom_verify_encoding_ref(state, NULL, type, name_buffer);
  }
}

//===----------------------------------------------------------------------===//
// Tied result validation
//===----------------------------------------------------------------------===//

static void loom_verify_tied_results(loom_verify_state_t* state,
                                     const loom_op_t* op,
                                     const loom_op_vtable_t* vtable) {
  iree_string_view_t op_name = loom_op_vtable_name(vtable);
  const loom_tied_result_t* tied = loom_op_tied_results(op);
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    if (tied[i].result_index >= op->result_count) {
      loom_diagnostic_param_t params[] = {
          loom_param_u32(tied[i].result_index),
          loom_param_string(op_name),
          loom_param_u32(op->result_count),
      };
      loom_verify_emit_structured(state, op, &loom_err_dominance_004, params,
                                  3);
    }
    if (tied[i].operand_index >= op->operand_count) {
      loom_diagnostic_param_t params[] = {
          loom_param_u32(tied[i].operand_index),
          loom_param_string(op_name),
          loom_param_u32(op->operand_count),
      };
      loom_verify_emit_structured(state, op, &loom_err_dominance_005, params,
                                  3);
    }
    // Mark the tied operand as consumed.
    if (tied[i].operand_index < op->operand_count) {
      loom_value_id_t consumed_id =
          loom_op_const_operands(op)[tied[i].operand_index];
      loom_verify_consume_value(state, consumed_id);
    }
  }
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
    if (vtable->attr_descriptors[i].attr_kind != LOOM_ATTR_SYMBOL) continue;
    if ((vtable->attr_descriptors[i].flags & LOOM_ATTR_OPTIONAL) &&
        attrs[i].raw == 0) {
      continue;
    }
    loom_symbol_ref_t ref = loom_attr_as_symbol(attrs[i]);
    if (!loom_symbol_ref_is_valid(ref)) continue;

    if (ref.symbol_id >= state->module->symbols.count) {
      loom_diagnostic_param_t params[] = {
          loom_param_u32(ref.symbol_id),
          loom_param_u32((uint32_t)state->module->symbols.count),
      };
      loom_verify_emit_structured(state, op, &loom_err_symbol_001, params, 2);
      continue;
    }

    const loom_symbol_t* symbol =
        &state->module->symbols.entries[ref.symbol_id];
    if (symbol->kind == LOOM_SYMBOL_NONE || symbol->defining_op == NULL) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(loom_verify_symbol_name(state, ref)),
      };
      loom_verify_emit_structured(state, op, &loom_err_symbol_002, params, 1);
    }
  }
}

//===----------------------------------------------------------------------===//
// Region structure checks
//===----------------------------------------------------------------------===//

static void loom_verify_region_structure(loom_verify_state_t* state,
                                         const loom_op_t* op,
                                         const loom_op_vtable_t* vtable) {
  if (!vtable->region_descriptors) return;
  iree_string_view_t op_name = loom_op_vtable_name(vtable);
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < vtable->region_count && i < op->region_count; ++i) {
    loom_region_t* region = regions[i];
    if (!region) {
      // NULL region is only valid for optional regions.
      continue;
    }
    if ((vtable->region_descriptors[i].flags & LOOM_REGION_SINGLE_BLOCK) &&
        region->block_count != 1) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(op_name),
          loom_param_u32(i),
          loom_param_u32(region->block_count),
      };
      loom_verify_emit_structured(state, op, &loom_err_structure_006, params,
                                  3);
    }
    // Check that each non-empty block has a terminator as its last op.
    for (uint16_t b = 0; b < region->block_count; ++b) {
      loom_block_t* block = &region->blocks[b];
      if (block->op_count == 0) continue;
      const loom_op_t* last_op = block->ops[block->op_count - 1];
      const loom_op_vtable_t* last_vtable =
          loom_verify_lookup_vtable(state, last_op->kind);
      if (last_vtable && !(last_vtable->traits & LOOM_TRAIT_TERMINATOR)) {
        loom_diagnostic_param_t params[] = {
            loom_param_string(op_name),
            loom_param_u32(i),
        };
        loom_verify_emit_structured(state, op, &loom_err_structure_005, params,
                                    2);
      }
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
    loom_block_t* block = &region->blocks[b];
    // Define block arguments, then validate any SSA encoding
    // references in their types (encoding values must be visible
    // from the enclosing scope).
    for (uint16_t a = 0; a < block->arg_count; ++a) {
      IREE_RETURN_IF_ERROR(loom_verify_define_value(state, block->arg_ids[a]));
    }
    loom_verify_block_arg_encoding_refs(state, block);
    iree_status_t diagnostic_status =
        loom_verify_pending_diagnostic_status(state);
    if (!iree_status_is_ok(diagnostic_status)) {
      loom_verify_pop_scope(state);
      return diagnostic_status;
    }
    for (uint16_t i = 0; i < block->op_count; ++i) {
      if (loom_verify_at_error_limit(state)) break;
      loom_op_t* current = block->ops[i];
      if (current->flags & LOOM_OP_FLAG_DEAD) continue;
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
    snprintf(kind_buffer, sizeof(kind_buffer), "0x%04x", op->kind);
    loom_diagnostic_param_t params[] = {
        loom_param_u32(op->kind),
        loom_param_string(iree_make_cstring_view(kind_buffer)),
    };
    loom_verify_emit_structured(state, op, &loom_err_structure_009, params, 2);
    IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));
    // Still define results so downstream dominance checks don't cascade.
    for (uint16_t i = 0; i < op->result_count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_verify_define_value(state, loom_op_const_results(op)[i]));
    }
    return iree_ok_status();
  }

  // Operand references must be valid and in scope. This must run
  // before any type-level checks since those read operand types.
  loom_verify_operand_dominance(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Structural checks: operand/result/attr/region counts match the
  // vtable declaration. Must pass before per-field checks below.
  loom_verify_op_structure(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Per-field type constraint checks (operand/result types satisfy
  // the type category declared in the vtable descriptors).
  loom_verify_type_constraints(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // SSA encoding references embedded in operand/result types must
  // point to valid, in-scope values of type LOOM_TYPE_ENCODING.
  // Requires operand dominance to have run (uses the defined_bits).
  loom_verify_encoding_refs(state, op);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Semantic constraints: table-driven (relation, property) interpreter.
  // Requires structural and type checks to have passed so field refs
  // resolve to valid values with the expected type categories.
  loom_verify_semantic_constraints(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Tied result validation and linear ownership consume marking.
  loom_verify_tied_results(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Symbol reference attributes must point to valid symbol table entries.
  loom_verify_symbol_references(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Region structure: single-block invariants, terminator presence.
  loom_verify_region_structure(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Define result values. Must happen after all checks on this op
  // so that a result cannot appear to dominate its own defining op.
  for (uint16_t i = 0; i < op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_verify_define_value(state, loom_op_const_results(op)[i]));
  }

  // Recurse into regions. Region contents see this op's results
  // (defined above) and all enclosing scope values.
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (loom_verify_at_error_limit(state)) break;
    IREE_RETURN_IF_ERROR(loom_verify_region(state, regions[i]));
  }

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
    memset(state->defined_bits, 0,
           state->defined_bits_length * sizeof(uint64_t));
    status = iree_arena_allocate_array(
        &state->arena, state->defined_bits_length, sizeof(uint64_t),
        (void**)&state->consumed_bits);
  }
  if (iree_status_is_ok(status)) {
    memset(state->consumed_bits, 0,
           state->defined_bits_length * sizeof(uint64_t));
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
      loom_block_t* entry = &module->body->blocks[0];
      for (uint16_t i = 0; i < entry->op_count; ++i) {
        const loom_op_t* op = entry->ops[i];
        if (op->flags & LOOM_OP_FLAG_DEAD) continue;
        const loom_op_vtable_t* vtable =
            loom_verify_lookup_vtable(&state, op->kind);
        if (vtable &&
            !iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE)) {
          iree_string_view_t op_name = loom_op_vtable_name(vtable);
          loom_diagnostic_param_t params[] = {
              loom_param_string(op_name),
          };
          loom_verify_emit_structured(&state, op, &loom_err_structure_011,
                                      params, IREE_ARRAYSIZE(params));
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

  // Define function arguments.
  uint16_t arg_count = 0;
  const loom_value_id_t* arg_ids = loom_func_like_arg_ids(function, &arg_count);
  for (uint16_t i = 0; i < arg_count; ++i) {
    iree_status_t define_status = loom_verify_define_value(&state, arg_ids[i]);
    if (!iree_status_is_ok(define_status)) {
      loom_verify_state_deinitialize(&state);
      return define_status;
    }
  }

  // Walk the function body.
  loom_region_t* body = loom_func_like_body(function);
  if (body) {
    iree_status_t walk_status = loom_verify_region(&state, body);
    if (!iree_status_is_ok(walk_status)) {
      loom_verify_state_deinitialize(&state);
      return walk_status;
    }
  }

  loom_verify_state_deinitialize(&state);
  return iree_ok_status();
}
