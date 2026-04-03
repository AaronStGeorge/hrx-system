// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/parser.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include "loom/format/text/parser_internal.h"
#include "loom/ir/context.h"

//===----------------------------------------------------------------------===//
// Name scope
//===----------------------------------------------------------------------===//

// Fibonacci hashing for interned string IDs. Multiplying by the golden ratio
// constant mixes sequential IDs well for power-of-two table capacities.
static uint32_t loom_parser_scope_hash(loom_string_id_t name_id) {
  return (uint32_t)name_id * 2654435769u;
}

static void loom_parser_scope_initialize_entries(
    loom_parser_scope_entry_t* entries, iree_host_size_t capacity) {
  for (iree_host_size_t i = 0; i < capacity; ++i) {
    entries[i].name_id = LOOM_STRING_ID_INVALID;
    entries[i].value_id = LOOM_VALUE_ID_INVALID;
  }
}

// Ensures the scope's hash table is allocated and has room for at
// least one more entry. Called lazily before the first define().
static iree_status_t loom_parser_scope_ensure_capacity(
    loom_parser_scope_t* scope, iree_arena_allocator_t* arena) {
  // Grow if load factor exceeds ~70%.
  iree_host_size_t needed = scope->count + 1;
  if (scope->capacity > 0 && needed * 10 < scope->capacity * 7) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity =
      scope->capacity == 0 ? 16 : scope->capacity * 2;
  loom_parser_scope_entry_t* new_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, new_capacity, sizeof(loom_parser_scope_entry_t),
      (void**)&new_entries));
  loom_parser_scope_initialize_entries(new_entries, new_capacity);
  // Rehash existing entries.
  for (iree_host_size_t i = 0; i < scope->capacity; ++i) {
    if (scope->entries[i].name_id == LOOM_STRING_ID_INVALID) continue;
    uint32_t hash = loom_parser_scope_hash(scope->entries[i].name_id);
    iree_host_size_t slot = hash & (new_capacity - 1);
    while (new_entries[slot].name_id != LOOM_STRING_ID_INVALID) {
      slot = (slot + 1) & (new_capacity - 1);
    }
    new_entries[slot] = scope->entries[i];
  }
  // Old entries are arena-allocated — no need to free.
  scope->entries = new_entries;
  scope->capacity = new_capacity;
  return iree_ok_status();
}

iree_status_t loom_parser_scope_define(loom_parser_scope_t* scope,
                                       iree_arena_allocator_t* arena,
                                       loom_string_id_t name_id,
                                       loom_value_id_t value_id,
                                       bool* out_duplicate) {
  if (out_duplicate) *out_duplicate = false;
  IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
  IREE_RETURN_IF_ERROR(loom_parser_scope_ensure_capacity(scope, arena));
  uint32_t hash = loom_parser_scope_hash(name_id);
  iree_host_size_t slot = hash & (scope->capacity - 1);
  while (scope->entries[slot].name_id != LOOM_STRING_ID_INVALID) {
    if (scope->entries[slot].name_id == name_id) {
      if (out_duplicate) *out_duplicate = true;
      return iree_ok_status();
    }
    slot = (slot + 1) & (scope->capacity - 1);
  }
  scope->entries[slot].name_id = name_id;
  scope->entries[slot].value_id = value_id;
  ++scope->count;
  return iree_ok_status();
}

loom_value_id_t loom_parser_scope_lookup(const loom_parser_scope_t* scope,
                                         loom_string_id_t name_id) {
  if (name_id == LOOM_STRING_ID_INVALID) return LOOM_VALUE_ID_INVALID;
  const loom_parser_scope_t* current = scope;
  while (current) {
    if (current->capacity > 0) {
      uint32_t hash = loom_parser_scope_hash(name_id);
      iree_host_size_t slot = hash & (current->capacity - 1);
      while (current->entries[slot].name_id != LOOM_STRING_ID_INVALID) {
        if (current->entries[slot].name_id == name_id) {
          return current->entries[slot].value_id;
        }
        slot = (slot + 1) & (current->capacity - 1);
      }
    }
    current = current->parent;
  }
  return LOOM_VALUE_ID_INVALID;
}

iree_status_t loom_parser_scope_push(iree_arena_allocator_t* arena,
                                     loom_parser_scope_t* parent,
                                     loom_parser_scope_t** out_scope) {
  loom_parser_scope_t* scope = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(loom_parser_scope_t), (void**)&scope));
  memset(scope, 0, sizeof(*scope));
  scope->parent = parent;
  *out_scope = scope;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Result scope
//===----------------------------------------------------------------------===//

static iree_host_size_t loom_parser_result_scope_hash_capacity_for_count(
    iree_host_size_t entry_count) {
  iree_host_size_t hash_capacity = 32;
  while (entry_count * 10 >= hash_capacity * 7) {
    hash_capacity *= 2;
  }
  return hash_capacity;
}

iree_status_t loom_parser_result_scope_prepare(
    loom_parser_result_scope_t* scope, iree_host_size_t entry_count,
    iree_arena_allocator_t* arena) {
  scope->count = 0;
  scope->hash_capacity = 0;
  if (entry_count <= LOOM_PARSER_RESULT_SCOPE_INLINE_ENTRIES) {
    return iree_ok_status();
  }

  iree_host_size_t hash_capacity =
      loom_parser_result_scope_hash_capacity_for_count(entry_count);
  if (hash_capacity > scope->hash_storage_capacity) {
    loom_parser_scope_entry_t* hash_entries = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, hash_capacity, sizeof(*hash_entries), (void**)&hash_entries));
    scope->hash_entries = hash_entries;
    scope->hash_storage_capacity = hash_capacity;
  }
  loom_parser_scope_initialize_entries(scope->hash_entries, hash_capacity);
  scope->hash_capacity = hash_capacity;
  return iree_ok_status();
}

void loom_parser_result_scope_reset(loom_parser_result_scope_t* scope) {
  scope->count = 0;
  scope->hash_capacity = 0;
}

iree_status_t loom_parser_result_scope_define(loom_parser_result_scope_t* scope,
                                              loom_string_id_t name_id,
                                              loom_value_id_t value_id,
                                              bool* out_duplicate) {
  if (out_duplicate) *out_duplicate = false;
  IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
  if (scope->hash_capacity == 0) {
    for (uint16_t entry_index = 0; entry_index < scope->count; ++entry_index) {
      if (scope->inline_entries[entry_index].name_id == name_id) {
        if (out_duplicate) *out_duplicate = true;
        return iree_ok_status();
      }
    }
    IREE_ASSERT(scope->count < LOOM_PARSER_RESULT_SCOPE_INLINE_ENTRIES);
    scope->inline_entries[scope->count].name_id = name_id;
    scope->inline_entries[scope->count].value_id = value_id;
    ++scope->count;
    return iree_ok_status();
  }

  iree_host_size_t slot =
      loom_parser_scope_hash(name_id) & (scope->hash_capacity - 1);
  while (scope->hash_entries[slot].name_id != LOOM_STRING_ID_INVALID) {
    if (scope->hash_entries[slot].name_id == name_id) {
      if (out_duplicate) *out_duplicate = true;
      return iree_ok_status();
    }
    slot = (slot + 1) & (scope->hash_capacity - 1);
  }
  scope->hash_entries[slot].name_id = name_id;
  scope->hash_entries[slot].value_id = value_id;
  ++scope->count;
  return iree_ok_status();
}

static loom_value_id_t loom_parser_result_scope_lookup(
    const loom_parser_result_scope_t* scope, loom_string_id_t name_id) {
  if (scope->count == 0) return LOOM_VALUE_ID_INVALID;
  if (scope->hash_capacity == 0) {
    for (uint16_t entry_index = 0; entry_index < scope->count; ++entry_index) {
      if (scope->inline_entries[entry_index].name_id == name_id) {
        return scope->inline_entries[entry_index].value_id;
      }
    }
    return LOOM_VALUE_ID_INVALID;
  }

  iree_host_size_t slot =
      loom_parser_scope_hash(name_id) & (scope->hash_capacity - 1);
  while (scope->hash_entries[slot].name_id != LOOM_STRING_ID_INVALID) {
    if (scope->hash_entries[slot].name_id == name_id) {
      return scope->hash_entries[slot].value_id;
    }
    slot = (slot + 1) & (scope->hash_capacity - 1);
  }
  return LOOM_VALUE_ID_INVALID;
}

loom_value_id_t loom_parser_lookup_value(const loom_parser_t* parser,
                                         loom_string_id_t name_id) {
  if (name_id == LOOM_STRING_ID_INVALID) return LOOM_VALUE_ID_INVALID;
  loom_value_id_t value_id =
      loom_parser_result_scope_lookup(&parser->result_scope, name_id);
  if (value_id != LOOM_VALUE_ID_INVALID) return value_id;
  return loom_parser_scope_lookup(parser->scope, name_id);
}

static iree_status_t loom_parser_emit_duplicate_value_name(
    loom_parser_t* parser, loom_token_t name_token) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(name_token.text),
  };
  return loom_parser_emit(parser, &loom_err_parse_002, params,
                          IREE_ARRAYSIZE(params), name_token);
}

iree_status_t loom_parser_define_value_name(loom_parser_t* parser,
                                            loom_token_t name_token,
                                            loom_value_id_t value_id) {
  if (iree_string_view_is_empty(name_token.text)) return iree_ok_status();

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(parser->module, name_token.text, &name_id));
  parser->module->values.entries[value_id].name_id = name_id;
  bool duplicate = false;
  IREE_RETURN_IF_ERROR(loom_parser_scope_define(
      parser->scope, &parser->parser_arena, name_id, value_id, &duplicate));
  if (!duplicate) return iree_ok_status();
  return loom_parser_emit_duplicate_value_name(parser, name_token);
}

iree_status_t loom_parser_resolve_value(loom_parser_t* parser,
                                        loom_token_t name_token,
                                        loom_value_id_t* out_value_id) {
  loom_string_id_t name_id =
      loom_module_lookup_string(parser->module, name_token.text);
  *out_value_id = loom_parser_lookup_value(parser, name_id);
  if (*out_value_id != LOOM_VALUE_ID_INVALID) return iree_ok_status();

  loom_diagnostic_param_t params[] = {
      loom_param_string(name_token.text),
  };
  return loom_parser_emit(parser, &loom_err_parse_001, params,
                          IREE_ARRAYSIZE(params), name_token);
}

iree_status_t loom_parser_emit_result_count_mismatch(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    loom_token_t op_name_token, uint16_t expected_count,
    uint16_t actual_count) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_vtable_name(vtable)),
      loom_param_u32(expected_count),
      loom_param_u32(actual_count),
  };
  return loom_parser_emit(parser, &loom_err_parse_009, params,
                          IREE_ARRAYSIZE(params), op_name_token);
}

//===----------------------------------------------------------------------===//
// Alias table
//===----------------------------------------------------------------------===//

iree_status_t loom_alias_table_add(loom_alias_table_t* table,
                                   iree_arena_allocator_t* arena,
                                   iree_string_view_t name,
                                   uint16_t encoding_id) {
  if (table->count >= table->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, table->count, 8, sizeof(loom_alias_entry_t), &table->capacity,
        (void**)&table->entries));
  }
  table->entries[table->count].name = name;
  table->entries[table->count].encoding_id = encoding_id;
  ++table->count;
  return iree_ok_status();
}

uint16_t loom_alias_table_lookup(const loom_alias_table_t* table,
                                 iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < table->count; ++i) {
    if (iree_string_view_equal(table->entries[i].name, name)) {
      return table->entries[i].encoding_id;
    }
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// Diagnostic emission
//===----------------------------------------------------------------------===//

static loom_source_range_t loom_parser_token_origin(iree_string_view_t filename,
                                                    iree_string_view_t source,
                                                    loom_token_t token) {
  if (token.kind == LOOM_TOKEN_NONE || token.kind == LOOM_TOKEN_EOF) {
    return (loom_source_range_t){
        .filename = filename,
        .source = source,
        .start = source.size,
        .end = source.size,
        .start_line = token.line,
        .start_column = token.column,
        .end_line = token.line,
        .end_column = token.end_column,
    };
  }

  return loom_source_range_from_token(filename, source,
                                      loom_token_lexeme(token), token.line,
                                      token.column, token.end_column);
}

iree_status_t loom_parser_emit(loom_parser_t* parser,
                               const loom_error_def_t* error,
                               const loom_diagnostic_param_t* params,
                               iree_host_size_t param_count,
                               loom_token_t token) {
  if (error->severity == LOOM_DIAGNOSTIC_ERROR) {
    ++parser->error_count;
  }
  loom_source_range_t origin =
      loom_parser_token_origin(parser->filename, parser->source, token);
  loom_diagnostic_t diagnostic = {
      .severity = error->severity,
      .error = error,
      .params = params,
      .param_count = param_count,
      .emitter = LOOM_EMITTER_PARSER,
      .origin = origin,
      // The text parser currently reports the parse token itself as the
      // user-facing location. Parsed loc() metadata can diverge from origin in
      // later verifier/reader diagnostics.
      .source_location = origin,
  };
  IREE_RETURN_IF_ERROR(
      loom_diagnostic_emit(&parser->diagnostic_sink, &diagnostic));

  // When the error limit is reached, emit ERR_PARSE_012 so the sink
  // sees an explicit "too many errors" diagnostic before we stop.
  if (parser->max_errors > 0 && parser->error_count == parser->max_errors) {
    loom_diagnostic_param_t limit_params[] = {
        loom_param_u32(parser->error_count),
    };
    loom_diagnostic_t limit_diagnostic = {
        .severity = LOOM_DIAGNOSTIC_ERROR,
        .error = &loom_err_parse_012,
        .params = limit_params,
        .param_count = IREE_ARRAYSIZE(limit_params),
        .emitter = LOOM_EMITTER_PARSER,
        .origin = diagnostic.origin,
        .source_location = diagnostic.source_location,
    };
    IREE_RETURN_IF_ERROR(
        loom_diagnostic_emit(&parser->diagnostic_sink, &limit_diagnostic));
  }

  return iree_ok_status();
}

iree_status_t loom_parser_expect(loom_parser_t* parser, loom_token_kind_t kind,
                                 loom_token_t* out_token) {
  // Propagate pending scan errors — these are real infrastructure failures.
  if (!iree_status_is_ok(parser->tokenizer.status)) {
    return loom_tokenizer_consume_status(&parser->tokenizer);
  }
  loom_token_t token = loom_tokenizer_next(&parser->tokenizer);
  // A scan inside next() may have produced an error.
  if (!iree_status_is_ok(parser->tokenizer.status)) {
    return loom_tokenizer_consume_status(&parser->tokenizer);
  }
  if (token.kind != kind) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(loom_token_lexeme(token)),
        loom_param_string(loom_token_kind_name(kind)),
    };
    return loom_parser_emit(parser, &loom_err_parse_003, params,
                            IREE_ARRAYSIZE(params), token);
  }
  if (out_token) *out_token = token;
  return iree_ok_status();
}

bool loom_parser_at_error_limit(const loom_parser_t* parser) {
  return parser->max_errors > 0 && parser->error_count >= parser->max_errors;
}

//===----------------------------------------------------------------------===//
// Error recovery
//===----------------------------------------------------------------------===//

void loom_parser_sync_to_newline(loom_parser_t* parser) {
  // The tokenizer doesn't expose "at start of line" directly, but
  // we can advance until we find an op-starting token or EOF.
  // A simple heuristic: advance until we see a token at column 1 or
  // whose kind is SSA_VALUE (result name), OP_NAME, BLOCK_LABEL,
  // RBRACE (end of region), or EOF.
  for (;;) {
    loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
    if (token.kind == LOOM_TOKEN_EOF) break;
    if (token.kind == LOOM_TOKEN_RBRACE) break;
    // If token is at column 1 (or 3+ for indented ops), it's likely
    // the start of a new op. We use the heuristic that SSA_VALUE at
    // position <= the next line start means we've synced.
    if (token.kind == LOOM_TOKEN_SSA_VALUE && token.column <= 2) break;
    if (token.kind == LOOM_TOKEN_OP_NAME && token.column <= 2) break;
    if (token.kind == LOOM_TOKEN_BLOCK_LABEL) break;
    loom_tokenizer_next(&parser->tokenizer);
  }
}

void loom_parser_sync_to_brace(loom_parser_t* parser) {
  int depth = 1;
  for (;;) {
    loom_token_t token = loom_tokenizer_next(&parser->tokenizer);
    if (token.kind == LOOM_TOKEN_EOF) break;
    if (token.kind == LOOM_TOKEN_LBRACE) ++depth;
    if (token.kind == LOOM_TOKEN_RBRACE) {
      --depth;
      if (depth <= 0) break;
    }
  }
}

//===----------------------------------------------------------------------===//
// Op accumulator
//===----------------------------------------------------------------------===//

void loom_parsed_op_initialize(loom_parsed_op_t* parsed) {
  memset(parsed, 0, sizeof(*parsed));
  parsed->operand_ids = parsed->inline_operand_ids;
  parsed->operand_capacity = LOOM_PARSED_OP_INLINE_OPERANDS;
  parsed->result_ids = parsed->inline_result_ids;
  parsed->result_name_tokens = parsed->inline_result_name_tokens;
  parsed->result_capacity = LOOM_PARSED_OP_INLINE_RESULTS;
  parsed->attributes = parsed->inline_attributes;
  parsed->attribute_capacity = LOOM_PARSED_OP_INLINE_ATTRS;
  parsed->regions = parsed->inline_regions;
  parsed->region_capacity = LOOM_PARSED_OP_INLINE_REGIONS;
  parsed->tied_results = parsed->inline_tied_results;
  parsed->tied_result_capacity = LOOM_PARSED_OP_INLINE_TIED;
}

iree_status_t loom_parsed_op_add_operand(loom_parsed_op_t* parsed,
                                         iree_arena_allocator_t* arena,
                                         loom_value_id_t value_id) {
  if (parsed->operand_count >= parsed->operand_capacity) {
    iree_host_size_t capacity = parsed->operand_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, parsed->operand_count, 0, sizeof(loom_value_id_t), &capacity,
        (void**)&parsed->operand_ids));
    parsed->operand_capacity = (uint16_t)capacity;
  }
  parsed->operand_ids[parsed->operand_count++] = value_id;
  return iree_ok_status();
}

iree_status_t loom_parsed_op_add_result(loom_parsed_op_t* parsed,
                                        iree_arena_allocator_t* arena,
                                        loom_value_id_t value_id,
                                        loom_token_t name_token) {
  if (parsed->result_count >= parsed->result_capacity) {
    iree_host_size_t capacity = parsed->result_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, parsed->result_count, 0, sizeof(loom_value_id_t), &capacity,
        (void**)&parsed->result_ids));
    iree_host_size_t name_token_capacity = parsed->result_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, parsed->result_count, capacity, sizeof(loom_token_t),
        &name_token_capacity, (void**)&parsed->result_name_tokens));
    IREE_ASSERT(name_token_capacity == capacity);
    parsed->result_capacity = (uint16_t)capacity;
  }
  parsed->result_ids[parsed->result_count] = value_id;
  parsed->result_name_tokens[parsed->result_count] = name_token;
  ++parsed->result_count;
  return iree_ok_status();
}

iree_status_t loom_parsed_op_set_attribute(loom_parsed_op_t* parsed,
                                           iree_arena_allocator_t* arena,
                                           uint8_t field_index,
                                           loom_attribute_t attr) {
  if (field_index >= parsed->attribute_capacity) {
    iree_host_size_t capacity = parsed->attribute_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, parsed->attribute_count, (iree_host_size_t)field_index + 1,
        sizeof(loom_attribute_t), &capacity, (void**)&parsed->attributes));
    // Zero newly exposed slots so unset attributes read as zero.
    memset(&parsed->attributes[parsed->attribute_count], 0,
           (capacity - parsed->attribute_count) * sizeof(loom_attribute_t));
    parsed->attribute_capacity = (uint8_t)capacity;
  }
  parsed->attributes[field_index] = attr;
  if (field_index >= parsed->attribute_count) {
    parsed->attribute_count = field_index + 1;
  }
  return iree_ok_status();
}

iree_status_t loom_parsed_op_add_region(loom_parsed_op_t* parsed,
                                        iree_arena_allocator_t* arena,
                                        loom_region_t* region) {
  if (parsed->region_count >= parsed->region_capacity) {
    iree_host_size_t capacity = parsed->region_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, parsed->region_count, 0, sizeof(loom_region_t*), &capacity,
        (void**)&parsed->regions));
    parsed->region_capacity = (uint8_t)capacity;
  }
  parsed->regions[parsed->region_count++] = region;
  return iree_ok_status();
}

iree_status_t loom_parsed_op_add_tied_result(loom_parsed_op_t* parsed,
                                             iree_arena_allocator_t* arena,
                                             loom_tied_result_t tied) {
  if (parsed->tied_result_count >= parsed->tied_result_capacity) {
    iree_host_size_t capacity = parsed->tied_result_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, parsed->tied_result_count, 0, sizeof(loom_tied_result_t),
        &capacity, (void**)&parsed->tied_results));
    parsed->tied_result_capacity = (uint16_t)capacity;
  }
  parsed->tied_results[parsed->tied_result_count++] = tied;
  return iree_ok_status();
}

iree_status_t loom_parser_add_pending_block_arg(loom_parser_t* parser,
                                                loom_value_id_t value_id,
                                                loom_token_t name_token) {
  loom_parser_pending_block_args_t* pending_block_args =
      &parser->pending_block_args;
  if (pending_block_args->count >= pending_block_args->capacity) {
    iree_host_size_t capacity = pending_block_args->capacity;
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(&parser->parser_arena, pending_block_args->count,
                              8, sizeof(loom_parser_pending_block_arg_t),
                              &capacity, (void**)&pending_block_args->entries));
    pending_block_args->capacity = (uint16_t)capacity;
  }
  pending_block_args->entries[pending_block_args->count++] =
      (loom_parser_pending_block_arg_t){
          .value_id = value_id,
          .name_token = name_token,
      };
  return iree_ok_status();
}

// Clears the active pending-arg list while retaining arena-backed storage for
// reuse by later REGION elements.
static void loom_parser_pending_block_args_reset(
    loom_parser_pending_block_args_t* pending_block_args) {
  pending_block_args->count = 0;
}

//===----------------------------------------------------------------------===//
// Attribute parsing
//===----------------------------------------------------------------------===//

iree_status_t loom_parse_attr_value(loom_parser_t* parser,
                                    const loom_attr_descriptor_t* descriptor,
                                    loom_attribute_t* out_attr) {
  switch (descriptor->attr_kind) {
    case LOOM_ATTR_I64: {
      loom_token_t token = loom_token_none();
      LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_INTEGER, &token);
      int64_t value = 0;
      if (!iree_string_view_atoi_int64(token.text, &value)) {
        loom_diagnostic_param_t params[] = {
            loom_param_string(token.text),
        };
        return loom_parser_emit(parser, &loom_err_parse_015, params,
                                IREE_ARRAYSIZE(params), token);
      }
      *out_attr = loom_attr_i64(value);
      return iree_ok_status();
    }
    case LOOM_ATTR_F64: {
      loom_token_t token = loom_token_none();
      LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_FLOAT, &token);
      double value = 0.0;
      if (!iree_string_view_atod(token.text, &value)) {
        loom_diagnostic_param_t params[] = {
            loom_param_string(token.text),
        };
        return loom_parser_emit(parser, &loom_err_parse_016, params,
                                IREE_ARRAYSIZE(params), token);
      }
      *out_attr = loom_attr_f64(value);
      return iree_ok_status();
    }
    case LOOM_ATTR_STRING: {
      loom_token_t token = loom_token_none();
      LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_STRING, &token);
      // Token text is the content without surrounding quotes (the
      // tokenizer strips them, like %, @, #, ^ on other token types).
      loom_string_id_t string_id = 0;
      IREE_RETURN_IF_ERROR(
          loom_module_intern_string(parser->module, token.text, &string_id));
      *out_attr = loom_attr_string(string_id);
      return iree_ok_status();
    }
    case LOOM_ATTR_BOOL: {
      if (loom_tokenizer_try_consume_keyword(&parser->tokenizer,
                                             IREE_SV("true"))) {
        *out_attr = loom_attr_bool(true);
      } else if (loom_tokenizer_try_consume_keyword(&parser->tokenizer,
                                                    IREE_SV("false"))) {
        *out_attr = loom_attr_bool(false);
      } else {
        loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
        loom_diagnostic_param_t params[] = {
            loom_param_string(loom_token_lexeme(peek)),
            loom_param_string(IREE_SV("'true' or 'false'")),
        };
        return loom_parser_emit(parser, &loom_err_parse_003, params,
                                IREE_ARRAYSIZE(params), peek);
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_ENUM: {
      loom_token_t token = loom_token_none();
      LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, &token);
      // Look up the enum case name.
      if (!descriptor->enum_case_names) {
        // Internal bug: enum attribute declared without case names.
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "enum attribute has no case name table");
      }
      // Linear scan through case names. Enum case lists are short.
      for (uint8_t i = 0; i < descriptor->enum_case_count; ++i) {
        if (descriptor->enum_case_names[i] &&
            loom_bstring_equal(descriptor->enum_case_names[i], token.text)) {
          *out_attr = loom_attr_enum(i);
          return iree_ok_status();
        }
      }
      {
        iree_string_view_t enum_name =
            descriptor->name ? loom_attr_descriptor_name(descriptor)
                             : IREE_SV("enum");
        loom_diagnostic_param_t params[] = {
            loom_param_string(enum_name),
            loom_param_string(token.text),
        };
        return loom_parser_emit(parser, &loom_err_parse_017, params,
                                IREE_ARRAYSIZE(params), token);
      }
    }
    case LOOM_ATTR_I64_ARRAY: {
      // [1, 2, 3] — static array of integers.
      if (!loom_tokenizer_try_consume(&parser->tokenizer,
                                      LOOM_TOKEN_LBRACKET)) {
        loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
        loom_diagnostic_param_t params[] = {
            loom_param_string(loom_token_lexeme(peek)),
            loom_param_string(IREE_SV("'['")),
        };
        return loom_parser_emit(parser, &loom_err_parse_003, params,
                                IREE_ARRAYSIZE(params), peek);
      }
      // Parse into a stack buffer, then arena-copy.
      int64_t stack_values[32];
      uint16_t count = 0;
      while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACKET) &&
             !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
        if (count > 0) {
          if (!loom_tokenizer_try_consume(&parser->tokenizer,
                                          LOOM_TOKEN_COMMA)) {
            break;
          }
        }
        loom_token_t token = loom_token_none();
        LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_INTEGER, &token);
        if (count >= 32) {
          loom_diagnostic_param_t params[] = {
              loom_param_string(loom_token_lexeme(token)),
          };
          return loom_parser_emit(parser, &loom_err_parse_004, params,
                                  IREE_ARRAYSIZE(params), token);
        }
        int64_t value = 0;
        if (!iree_string_view_atoi_int64(token.text, &value)) {
          loom_diagnostic_param_t params[] = {
              loom_param_string(token.text),
          };
          return loom_parser_emit(parser, &loom_err_parse_015, params,
                                  IREE_ARRAYSIZE(params), token);
        }
        stack_values[count++] = value;
      }
      if (!loom_tokenizer_try_consume(&parser->tokenizer,
                                      LOOM_TOKEN_RBRACKET)) {
        loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
        loom_diagnostic_param_t params[] = {
            loom_param_string(loom_token_lexeme(peek)),
            loom_param_string(IREE_SV("']'")),
        };
        return loom_parser_emit(parser, &loom_err_parse_003, params,
                                IREE_ARRAYSIZE(params), peek);
      }
      int64_t* arena_values = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&parser->module->arena,
                                                     count, sizeof(int64_t),
                                                     (void**)&arena_values));
      memcpy(arena_values, stack_values, count * sizeof(int64_t));
      *out_attr = loom_attr_i64_array(arena_values, count);
      return iree_ok_status();
    }
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported attribute kind %d",
                              (int)descriptor->attr_kind);
  }
}

//===----------------------------------------------------------------------===//
// Predicate parsing
//===----------------------------------------------------------------------===//

// Predicate kind names for lookup.
static const struct {
  loom_bstring_t name;
  uint8_t kind;
} loom_predicate_names[] = {
    {(const uint8_t*)"\x02"
                     "eq",
     LOOM_PREDICATE_EQ},
    {(const uint8_t*)"\x02"
                     "lt",
     LOOM_PREDICATE_LT},
    {(const uint8_t*)"\x02"
                     "le",
     LOOM_PREDICATE_LE},
    {(const uint8_t*)"\x02"
                     "gt",
     LOOM_PREDICATE_GT},
    {(const uint8_t*)"\x02"
                     "ge",
     LOOM_PREDICATE_GE},
    {(const uint8_t*)"\x03"
                     "mul",
     LOOM_PREDICATE_MUL},
    {(const uint8_t*)"\x03"
                     "min",
     LOOM_PREDICATE_MIN},
    {(const uint8_t*)"\x03"
                     "max",
     LOOM_PREDICATE_MAX},
    {(const uint8_t*)"\x04"
                     "pow2",
     LOOM_PREDICATE_POW2},
    {(const uint8_t*)"\x05"
                     "range",
     LOOM_PREDICATE_RANGE},
};

iree_status_t loom_parse_predicate_list(loom_parser_t* parser,
                                        loom_attribute_t* out_attr) {
  // [pred(args), ...]
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LBRACKET)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    loom_diagnostic_param_t params[] = {
        loom_param_string(loom_token_lexeme(peek)),
        loom_param_string(IREE_SV("'['")),
    };
    return loom_parser_emit(parser, &loom_err_parse_003, params,
                            IREE_ARRAYSIZE(params), peek);
  }

  loom_predicate_t stack_predicates[16];
  uint16_t count = 0;

  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACKET) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (count > 0) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }
    if (count >= 16) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      loom_diagnostic_param_t params[] = {
          loom_param_string(loom_token_lexeme(peek)),
      };
      return loom_parser_emit(parser, &loom_err_parse_004, params,
                              IREE_ARRAYSIZE(params), peek);
    }

    // Parse predicate kind name.
    loom_token_t name_token = loom_token_none();
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, &name_token);

    uint8_t pred_kind = UINT8_MAX;
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(loom_predicate_names);
         ++i) {
      if (loom_bstring_equal(loom_predicate_names[i].name, name_token.text)) {
        pred_kind = loom_predicate_names[i].kind;
        break;
      }
    }
    if (pred_kind == UINT8_MAX) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(name_token.text),
      };
      return loom_parser_emit(parser, &loom_err_parse_013, params,
                              IREE_ARRAYSIZE(params), name_token);
    }

    // Expect '('.
    if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LPAREN)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      loom_diagnostic_param_t params[] = {
          loom_param_string(loom_token_lexeme(peek)),
          loom_param_string(IREE_SV("'('")),
      };
      return loom_parser_emit(parser, &loom_err_parse_003, params,
                              IREE_ARRAYSIZE(params), peek);
    }

    loom_predicate_t predicate = {
        .kind = pred_kind,
    };

    // Parse arguments.
    while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RPAREN) &&
           !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
      if (predicate.arg_count > 0) {
        if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
          break;
        }
      }
      if (predicate.arg_count >= 3) {
        loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
        loom_diagnostic_param_t params[] = {
            loom_param_string(loom_token_lexeme(peek)),
        };
        return loom_parser_emit(parser, &loom_err_parse_004, params,
                                IREE_ARRAYSIZE(params), peek);
      }

      loom_token_t arg_token = loom_tokenizer_peek(&parser->tokenizer);
      if (arg_token.kind == LOOM_TOKEN_SSA_VALUE) {
        // SSA value reference.
        loom_tokenizer_next(&parser->tokenizer);
        loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
        LOOM_PARSE_RESOLVE_VALUE(parser, arg_token, &value_id);
        predicate.arg_tags[predicate.arg_count] = LOOM_PRED_ARG_VALUE;
        predicate.args[predicate.arg_count] = (int64_t)value_id;
      } else if (arg_token.kind == LOOM_TOKEN_RESULT_ORDINAL) {
        // Predicate ordinals (#N) are no longer supported.
        loom_diagnostic_param_t params[] = {
            loom_param_string(loom_token_lexeme(arg_token)),
            loom_param_string(
                IREE_SV("a predicate argument (%name or integer)")),
        };
        return loom_parser_emit(parser, &loom_err_parse_003, params,
                                IREE_ARRAYSIZE(params), arg_token);
      } else if (arg_token.kind == LOOM_TOKEN_INTEGER) {
        // Constant.
        loom_tokenizer_next(&parser->tokenizer);
        int64_t value = 0;
        if (!iree_string_view_atoi_int64(arg_token.text, &value)) {
          loom_diagnostic_param_t params[] = {
              loom_param_string(arg_token.text),
          };
          return loom_parser_emit(parser, &loom_err_parse_015, params,
                                  IREE_ARRAYSIZE(params), arg_token);
        }
        predicate.arg_tags[predicate.arg_count] = LOOM_PRED_ARG_CONST;
        predicate.args[predicate.arg_count] = value;
      } else {
        loom_diagnostic_param_t params[] = {
            loom_param_string(loom_token_lexeme(arg_token)),
            loom_param_string(IREE_SV("a predicate argument")),
        };
        return loom_parser_emit(parser, &loom_err_parse_003, params,
                                IREE_ARRAYSIZE(params), arg_token);
      }
      ++predicate.arg_count;
    }

    if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RPAREN)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      loom_diagnostic_param_t params[] = {
          loom_param_string(loom_token_lexeme(peek)),
          loom_param_string(IREE_SV("')'")),
      };
      return loom_parser_emit(parser, &loom_err_parse_003, params,
                              IREE_ARRAYSIZE(params), peek);
    }

    stack_predicates[count++] = predicate;
  }

  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RBRACKET)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    loom_diagnostic_param_t params[] = {
        loom_param_string(loom_token_lexeme(peek)),
        loom_param_string(IREE_SV("']'")),
    };
    return loom_parser_emit(parser, &loom_err_parse_003, params,
                            IREE_ARRAYSIZE(params), peek);
  }

  loom_predicate_t* arena_predicates = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&parser->module->arena, count,
                                                 sizeof(loom_predicate_t),
                                                 (void**)&arena_predicates));
  memcpy(arena_predicates, stack_predicates, count * sizeof(loom_predicate_t));
  *out_attr = loom_attr_predicate_list(arena_predicates, count);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Dict attribute parsing
//===----------------------------------------------------------------------===//

iree_status_t loom_parse_attr_dict(loom_parser_t* parser,
                                   loom_attribute_t* out_attr) {
  // {key = value, key = value, ...}
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LBRACE)) {
    // Empty dict — no brace means absent.
    memset(out_attr, 0, sizeof(*out_attr));
    return iree_ok_status();
  }

  loom_named_attr_t stack_entries[16];
  uint16_t count = 0;

  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACE) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (count > 0) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }
    if (count >= 16) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      loom_diagnostic_param_t params[] = {
          loom_param_string(loom_token_lexeme(peek)),
      };
      return loom_parser_emit(parser, &loom_err_parse_004, params,
                              IREE_ARRAYSIZE(params), peek);
    }

    // Key.
    loom_token_t key_token = loom_token_none();
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, &key_token);
    loom_string_id_t key_id = 0;
    IREE_RETURN_IF_ERROR(
        loom_module_intern_string(parser->module, key_token.text, &key_id));

    // '='.
    if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_EQUALS)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      loom_diagnostic_param_t params[] = {
          loom_param_string(loom_token_lexeme(peek)),
          loom_param_string(IREE_SV("'='")),
      };
      return loom_parser_emit(parser, &loom_err_parse_003, params,
                              IREE_ARRAYSIZE(params), peek);
    }

    // Value: integer, float, string, or bool.
    loom_token_t val_token = loom_tokenizer_peek(&parser->tokenizer);
    loom_attribute_t value = {0};
    if (val_token.kind == LOOM_TOKEN_INTEGER) {
      loom_tokenizer_next(&parser->tokenizer);
      int64_t int_value = 0;
      if (!iree_string_view_atoi_int64(val_token.text, &int_value)) {
        loom_diagnostic_param_t params[] = {
            loom_param_string(val_token.text),
        };
        return loom_parser_emit(parser, &loom_err_parse_015, params,
                                IREE_ARRAYSIZE(params), val_token);
      }
      value = loom_attr_i64(int_value);
    } else if (val_token.kind == LOOM_TOKEN_FLOAT) {
      loom_tokenizer_next(&parser->tokenizer);
      double float_value = 0.0;
      if (!iree_string_view_atod(val_token.text, &float_value)) {
        loom_diagnostic_param_t params[] = {
            loom_param_string(val_token.text),
        };
        return loom_parser_emit(parser, &loom_err_parse_016, params,
                                IREE_ARRAYSIZE(params), val_token);
      }
      value = loom_attr_f64(float_value);
    } else if (val_token.kind == LOOM_TOKEN_STRING) {
      loom_tokenizer_next(&parser->tokenizer);
      loom_string_id_t string_id = 0;
      IREE_RETURN_IF_ERROR(loom_module_intern_string(
          parser->module, val_token.text, &string_id));
      value = loom_attr_string(string_id);
    } else if (val_token.kind == LOOM_TOKEN_BARE_IDENT &&
               iree_string_view_equal(val_token.text, IREE_SV("true"))) {
      loom_tokenizer_next(&parser->tokenizer);
      value = loom_attr_bool(true);
    } else if (val_token.kind == LOOM_TOKEN_BARE_IDENT &&
               iree_string_view_equal(val_token.text, IREE_SV("false"))) {
      loom_tokenizer_next(&parser->tokenizer);
      value = loom_attr_bool(false);
    } else {
      loom_diagnostic_param_t params[] = {
          loom_param_string(loom_token_lexeme(val_token)),
          loom_param_string(IREE_SV("an attribute value")),
      };
      return loom_parser_emit(parser, &loom_err_parse_003, params,
                              IREE_ARRAYSIZE(params), val_token);
    }

    stack_entries[count].name_id = key_id;
    stack_entries[count].reserved = 0;
    stack_entries[count].value = value;
    ++count;
  }

  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RBRACE)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    loom_diagnostic_param_t params[] = {
        loom_param_string(loom_token_lexeme(peek)),
        loom_param_string(IREE_SV("'}'")),
    };
    return loom_parser_emit(parser, &loom_err_parse_003, params,
                            IREE_ARRAYSIZE(params), peek);
  }

  if (count == 0) {
    memset(out_attr, 0, sizeof(*out_attr));
    return iree_ok_status();
  }

  loom_named_attr_t* arena_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&parser->module->arena, count,
                                                 sizeof(loom_named_attr_t),
                                                 (void**)&arena_entries));
  memcpy(arena_entries, stack_entries, count * sizeof(loom_named_attr_t));
  *out_attr = loom_make_attr_dict(arena_entries, count);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Keyword matching
//===----------------------------------------------------------------------===//

// Returns the token kind that a keyword would produce as a standalone
// token, or LOOM_TOKEN_BARE_IDENT for text keywords (import, where, etc.).
// Used by the optional group presence check to peek for punctuation
// keywords like ',' or '->' without assuming BARE_IDENT.
loom_token_kind_t loom_keyword_token_kind(uint16_t keyword_id) {
  switch (keyword_id) {
    case LOOM_KW_COMMA:
      return LOOM_TOKEN_COMMA;
    case LOOM_KW_COLON:
      return LOOM_TOKEN_COLON;
    case LOOM_KW_ARROW:
      return LOOM_TOKEN_ARROW;
    case LOOM_KW_EQUALS:
      return LOOM_TOKEN_EQUALS;
    case LOOM_KW_LPAREN:
      return LOOM_TOKEN_LPAREN;
    case LOOM_KW_RPAREN:
      return LOOM_TOKEN_RPAREN;
    case LOOM_KW_LBRACKET:
      return LOOM_TOKEN_LBRACKET;
    case LOOM_KW_RBRACKET:
      return LOOM_TOKEN_RBRACKET;
    case LOOM_KW_LBRACE:
      return LOOM_TOKEN_LBRACE;
    case LOOM_KW_RBRACE:
      return LOOM_TOKEN_RBRACE;
    default:
      return LOOM_TOKEN_BARE_IDENT;
  }
}

iree_status_t loom_parse_keyword(loom_parser_t* parser, uint16_t keyword_id) {
  switch (keyword_id) {
    case LOOM_KW_COMMA:
      return loom_parser_expect(parser, LOOM_TOKEN_COMMA, NULL);
    case LOOM_KW_COLON:
      return loom_parser_expect(parser, LOOM_TOKEN_COLON, NULL);
    case LOOM_KW_ARROW:
      return loom_parser_expect(parser, LOOM_TOKEN_ARROW, NULL);
    case LOOM_KW_EQUALS:
      return loom_parser_expect(parser, LOOM_TOKEN_EQUALS, NULL);
    case LOOM_KW_LPAREN:
      return loom_parser_expect(parser, LOOM_TOKEN_LPAREN, NULL);
    case LOOM_KW_RPAREN:
      return loom_parser_expect(parser, LOOM_TOKEN_RPAREN, NULL);
    case LOOM_KW_LBRACKET:
      return loom_parser_expect(parser, LOOM_TOKEN_LBRACKET, NULL);
    case LOOM_KW_RBRACKET:
      return loom_parser_expect(parser, LOOM_TOKEN_RBRACKET, NULL);
    case LOOM_KW_LBRACE:
      return loom_parser_expect(parser, LOOM_TOKEN_LBRACE, NULL);
    case LOOM_KW_RBRACE:
      return loom_parser_expect(parser, LOOM_TOKEN_RBRACE, NULL);
    default: {
      // Text keyword — match as BARE_IDENT.
      iree_string_view_t expected =
          loom_bstring_view(loom_keyword_bstrings[keyword_id]);
      if (!loom_tokenizer_try_consume_keyword(&parser->tokenizer, expected)) {
        loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
        loom_diagnostic_param_t params[] = {
            loom_param_string(loom_token_lexeme(peek)),
            loom_param_string(expected),
        };
        return loom_parser_emit(parser, &loom_err_parse_003, params,
                                IREE_ARRAYSIZE(params), peek);
      }
      return iree_ok_status();
    }
  }
}

//===----------------------------------------------------------------------===//
// Op finalization — accumulator -> loom_op_t
//===----------------------------------------------------------------------===//

static iree_status_t loom_finalize_op(
    loom_parser_t* parser, loom_op_kind_t kind, const loom_op_vtable_t* vtable,
    loom_parsed_op_t* parsed, loom_location_id_t location, loom_op_t** out_op) {
  // Allocate and insert the op. The attribute storage must use the
  // vtable's declared count, not the parser's high-water-mark. The
  // vtable defines the storage layout: the printer and typed accessors
  // index by field_index up to vtable->attribute_count. If the parser
  // only set index 0 of a 5-attribute op, the op still needs 5 slots
  // allocated. The allocation is zero-filled, so unset optional
  // attributes at higher indices read as zero (LOOM_ATTR_NONE).
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
      &parser->builder, kind, parsed->operand_count, parsed->result_count,
      parsed->region_count, parsed->tied_result_count, vtable->attribute_count,
      location, &op));

  // Copy operands.
  if (parsed->operand_count > 0) {
    memcpy(loom_op_operands(op), parsed->operand_ids,
           parsed->operand_count * sizeof(loom_value_id_t));
  }

  // Fill in implicit result types from result descriptors. When the
  // format doesn't include a RESULT_TYPE element for a result, the
  // value's type is NONE. Fixed-type constraints (e.g., I1 for
  // comparison results) provide the concrete type.
  if (vtable->result_descriptors) {
    for (uint16_t i = 0;
         i < parsed->result_count && i < vtable->fixed_result_count; ++i) {
      loom_value_t* value =
          &parser->module->values.entries[parsed->result_ids[i]];
      if (value->type.header != 0) {
        continue;
      }
      if (vtable->result_descriptors[i].type_constraint ==
          LOOM_TYPE_CONSTRAINT_I1) {
        value->type = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
      }
    }
  }

  // Copy result value_ids into the op and define names in scope.
  // Values already have their types and names set during LHS parsing
  // and the format walk.
  loom_value_id_t* result_slots = loom_op_results(op);
  for (uint16_t i = 0; i < parsed->result_count; ++i) {
    loom_value_id_t value_id = parsed->result_ids[i];
    result_slots[i] = value_id;
    loom_string_id_t name_id = parser->module->values.entries[value_id].name_id;
    if (name_id != LOOM_STRING_ID_INVALID &&
        name_id < parser->module->strings.count) {
      bool duplicate = false;
      IREE_RETURN_IF_ERROR(loom_parser_scope_define(
          parser->scope, &parser->parser_arena, name_id, value_id, &duplicate));
      if (duplicate) {
        IREE_RETURN_IF_ERROR(loom_parser_emit_duplicate_value_name(
            parser, parsed->result_name_tokens[i]));
        return iree_ok_status();
      }
    }
  }

  // Copy regions.
  if (parsed->region_count > 0) {
    memcpy(loom_op_regions(op), parsed->regions,
           parsed->region_count * sizeof(loom_region_t*));
  }

  // Copy tied results.
  if (parsed->tied_result_count > 0) {
    memcpy(loom_op_tied_results(op), parsed->tied_results,
           parsed->tied_result_count * sizeof(loom_tied_result_t));
  }

  // Copy attributes.
  if (parsed->attribute_count > 0) {
    memcpy(loom_op_attrs(op), parsed->attributes,
           parsed->attribute_count * sizeof(loom_attribute_t));
  }

  // Set instance flags.
  op->instance_flags = parsed->instance_flags;

  // Link symbol-defining ops incrementally so the symbol table has
  // valid defining_op pointers throughout parsing. Use-def chains
  // are built in batch by loom_module_compute_uses after parsing.
  if (iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE)) {
    loom_module_link_symbol_defining_op(parser->module, op, vtable);
  }

  *out_op = op;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Op parsing
//===----------------------------------------------------------------------===//

static iree_status_t loom_parse_op(loom_parser_t* parser) {
  uint32_t errors_before = parser->error_count;
  loom_parsed_op_t parsed;
  loom_parsed_op_initialize(&parsed);

  // Capture the start position of the op for source location tracking.
  // This is the first token — either a result name or the op name.
  loom_token_t start_token = loom_tokenizer_peek(&parser->tokenizer);

  // Parse LHS result names: %a, %b = op.name ...
  // Or just: op.name ... (for ops with no results).
  if (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE)) {
    // Collect result names until we see '='.
    for (;;) {
      if (parsed.result_count > 0) {
        if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
          break;
        }
      }
      if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE)) {
        break;
      }
      loom_token_t name_token = loom_tokenizer_next(&parser->tokenizer);
      loom_type_t none_type = {0};
      loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_module_define_value(parser->module, none_type, &value_id));
      if (name_token.text.size > 0) {
        loom_string_id_t name_id = 0;
        IREE_RETURN_IF_ERROR(loom_module_intern_string(
            parser->module, name_token.text, &name_id));
        parser->module->values.entries[value_id].name_id = name_id;
      }
      IREE_RETURN_IF_ERROR(loom_parsed_op_add_result(
          &parsed, &parser->parser_arena, value_id, name_token));
    }
    // Expect '='.
    if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_EQUALS)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      loom_diagnostic_param_t params[] = {
          loom_param_string(loom_token_lexeme(peek)),
          loom_param_string(IREE_SV("'='")),
      };
      return loom_parser_emit(parser, &loom_err_parse_003, params,
                              IREE_ARRAYSIZE(params), peek);
    }
  }

  // Parse op name.
  loom_token_t op_name_token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_OP_NAME, &op_name_token);

  // Look up the op vtable.
  loom_op_kind_t kind;
  const loom_op_vtable_t* vtable = loom_context_lookup_op_by_name(
      parser->context, op_name_token.text, &kind);
  if (!vtable) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(op_name_token.text),
    };
    return loom_parser_emit(parser, &loom_err_parse_006, params,
                            IREE_ARRAYSIZE(params), op_name_token);
  }

  bool is_symbol_definition =
      iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE);
  if (is_symbol_definition && parsed.result_count > 0) {
    return loom_parser_emit_result_count_mismatch(parser, vtable, op_name_token,
                                                  /*expected_count=*/0,
                                                  parsed.result_count);
  }

  // Walk the format elements.
  IREE_RETURN_IF_ERROR(
      loom_parser_walk_format(parser, vtable, op_name_token, &parsed));

  // If FUNC_ARGS pushed a scope but no REGION consumed the pending
  // block args (e.g., func.decl, func.ukernel), the args represent
  // the function signature rather than body block args. Store them as
  // the op's operands so the printer can recover the arg names and types.
  if (parser->pre_func_arg_scope) {
    if (parser->pending_block_args.count > 0) {
      for (uint16_t i = 0; i < parser->pending_block_args.count; ++i) {
        IREE_RETURN_IF_ERROR(loom_parsed_op_add_operand(
            &parsed, &parser->parser_arena,
            parser->pending_block_args.entries[i].value_id));
      }
    }
    parser->scope = parser->pre_func_arg_scope;
    parser->pre_func_arg_scope = NULL;
    loom_parser_pending_block_args_reset(&parser->pending_block_args);
  }
  if (parser->error_count > 0) {
    loom_parser_pending_block_args_reset(&parser->pending_block_args);
  }

  bool has_variadic_results =
      (vtable->vtable_flags & LOOM_OP_VTABLE_VARIADIC_RESULTS) != 0;
  if (!is_symbol_definition && !has_variadic_results &&
      parser->error_count == errors_before &&
      parsed.result_count != vtable->fixed_result_count) {
    IREE_RETURN_IF_ERROR(loom_parser_emit_result_count_mismatch(
        parser, vtable, op_name_token, vtable->fixed_result_count,
        parsed.result_count));
  }

  // Create a source location spanning the full op text.
  loom_location_id_t location = LOOM_LOCATION_UNKNOWN;
  if (parser->source_id != LOOM_SOURCE_ID_INVALID) {
    loom_location_entry_t entry = loom_location_file_range(
        parser->source_id, (uint16_t)start_token.line,
        (uint16_t)start_token.column,
        (uint16_t)parser->tokenizer.consumed_end_line,
        (uint16_t)parser->tokenizer.consumed_end_column);
    IREE_RETURN_IF_ERROR(
        loom_module_add_location(parser->module, entry, &location));
  }

  // Finalize the op.
  loom_op_t* op = NULL;
  return loom_finalize_op(parser, kind, vtable, &parsed, location, &op);
}

//===----------------------------------------------------------------------===//
// Block and region parsing
//===----------------------------------------------------------------------===//

static iree_status_t loom_parse_block_body(loom_parser_t* parser,
                                           loom_block_t* block) {
  // Set the builder's insertion point to this block.
  loom_builder_set_block(&parser->builder, block);

  // Parse ops until we hit '}' (end of region) or EOF.
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACE) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_BLOCK_LABEL)) {
    if (loom_parser_at_error_limit(parser)) break;
    uint32_t errors_before = parser->error_count;
    IREE_RETURN_IF_ERROR(loom_parse_op(parser));
    if (parser->error_count > errors_before) {
      loom_parser_sync_to_newline(parser);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_parse_region_body(loom_parser_t* parser,
                                            loom_region_t* region,
                                            bool* out_region_end_consumed) {
  *out_region_end_consumed = false;

  // Seed the entry block with pending block args from FUNC_ARGS, BINDING_LIST,
  // or implicit region operands. Values are already defined in the module;
  // region-local names are defined in the child scope here.
  if (parser->pending_block_args.count > 0) {
    loom_block_t* entry_block = &region->blocks[0];
    for (uint16_t i = 0; i < parser->pending_block_args.count; ++i) {
      const loom_parser_pending_block_arg_t pending_arg =
          parser->pending_block_args.entries[i];
      LOOM_PARSE_DEFINE_VALUE_NAME(parser, pending_arg.name_token,
                                   pending_arg.value_id);
      IREE_RETURN_IF_ERROR(loom_block_add_arg(parser->module, entry_block,
                                              pending_arg.value_id));
    }
    loom_parser_pending_block_args_reset(&parser->pending_block_args);
  }

  // Parse blocks. The entry block may not have an explicit label.
  bool first_block = true;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACE) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (loom_parser_at_error_limit(parser)) break;

    loom_block_t* block = NULL;
    if (first_block) {
      block = &region->blocks[0];
      first_block = false;
    } else {
      // Additional blocks need to be allocated.
      IREE_RETURN_IF_ERROR(loom_module_allocate_block(parser->module, &block));
      // Attach to region (this needs region growth support).
      // For now, we only handle single-block regions fully.
    }

    // Parse optional block label: ^label(args):
    if (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_BLOCK_LABEL)) {
      loom_token_t label_token = loom_tokenizer_next(&parser->tokenizer);
      loom_string_id_t label_id = 0;
      IREE_RETURN_IF_ERROR(loom_module_intern_string(
          parser->module, label_token.text, &label_id));
      block->label_id = label_id;

      // Parse optional block args: (arg: type, ...).
      if (loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LPAREN)) {
        while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RPAREN) &&
               !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
          if (block->arg_count > 0) {
            if (!loom_tokenizer_try_consume(&parser->tokenizer,
                                            LOOM_TOKEN_COMMA)) {
              break;
            }
          }
          loom_token_t arg_name_token = loom_token_none();
          LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_SSA_VALUE, &arg_name_token);
          LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COLON, NULL);
          loom_type_t arg_type = {0};
          IREE_RETURN_IF_ERROR(
              loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, &arg_type));

          loom_value_id_t arg_value_id = 0;
          IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
              &parser->builder, block, arg_type, &arg_value_id));
          LOOM_PARSE_DEFINE_VALUE_NAME(parser, arg_name_token, arg_value_id);
        }
        LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RPAREN, NULL);
      }

      LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COLON, NULL);
    }

    IREE_RETURN_IF_ERROR(loom_parse_block_body(parser, block));
  }

  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RBRACE, NULL);
  *out_region_end_consumed = true;
  return iree_ok_status();
}

iree_status_t loom_parse_region(loom_parser_t* parser,
                                loom_region_t** out_region) {
  uint32_t errors_before = parser->error_count;

  // Expect '{'.
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LBRACE)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    loom_diagnostic_param_t params[] = {
        loom_param_string(loom_token_lexeme(peek)),
        loom_param_string(IREE_SV("'{'")),
    };
    return loom_parser_emit(parser, &loom_err_parse_003, params,
                            IREE_ARRAYSIZE(params), peek);
  }

  // Push a new scope for the region — unless FUNC_ARGS already pushed
  // one. In that case, the func-arg scope IS the region scope (args and
  // dim values defined during FUNC_ARGS are visible in the body).
  loom_parser_scope_t* outer_scope = NULL;
  if (parser->pre_func_arg_scope) {
    // FUNC_ARGS already pushed a scope. The current scope is the func-arg
    // scope, which becomes the region scope. Restore to pre_func_arg_scope
    // when the region ends.
    outer_scope = parser->pre_func_arg_scope;
    parser->pre_func_arg_scope = NULL;
  } else {
    outer_scope = parser->scope;
    IREE_RETURN_IF_ERROR(loom_parser_scope_push(&parser->parser_arena,
                                                outer_scope, &parser->scope));
  }

  // Allocate the region with a single entry block initially.
  loom_region_t* region = NULL;
  IREE_RETURN_IF_ERROR(loom_module_allocate_region(parser->module, 1, &region));

  // Save and set builder insertion point.
  loom_builder_ip_t saved_ip = loom_builder_save(&parser->builder);

  bool region_end_consumed = false;
  iree_status_t status =
      loom_parse_region_body(parser, region, &region_end_consumed);
  if (parser->error_count > errors_before && !region_end_consumed &&
      !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    loom_parser_sync_to_brace(parser);
  }
  loom_parser_pending_block_args_reset(&parser->pending_block_args);

  // Restore scope and insertion point.
  parser->scope = outer_scope;
  loom_builder_restore(&parser->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);
  if (parser->error_count > errors_before) return iree_ok_status();

  *out_region = region;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Module parsing
//===----------------------------------------------------------------------===//

static iree_status_t loom_parse_module_body(loom_parser_t* parser) {
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (loom_parser_at_error_limit(parser)) break;

    // Check for attribute aliases: #alias = #encoding<params>.
    if (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_HASH_ATTR)) {
      loom_token_t alias_token = loom_tokenizer_next(&parser->tokenizer);
      if (loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_EQUALS)) {
        // Parse the encoding on the RHS.
        loom_token_t enc_token = loom_tokenizer_peek(&parser->tokenizer);
        if (enc_token.kind == LOOM_TOKEN_HASH_ATTR) {
          loom_tokenizer_next(&parser->tokenizer);
          // #name or #name<params>. Build encoding.
          iree_string_view_t enc_name = enc_token.text;

          loom_encoding_t encoding = {0};
          IREE_RETURN_IF_ERROR(loom_module_intern_string(
              parser->module, enc_name, &encoding.name_id));

          // Intern the alias name too.
          loom_string_id_t alias_name_id = 0;
          // The alias name includes '#', e.g. "#q6_k".
          iree_string_view_t full_alias = iree_make_string_view(
              alias_token.text.data - 1, alias_token.text.size + 1);
          IREE_RETURN_IF_ERROR(loom_module_intern_string(
              parser->module, full_alias, &alias_name_id));
          encoding.alias_id = alias_name_id;

          // Parse optional params.
          if (loom_tokenizer_try_consume(&parser->tokenizer,
                                         LOOM_TOKEN_LANGLE)) {
            loom_named_attr_t* attrs = NULL;
            uint8_t attr_count = 0;
            IREE_RETURN_IF_ERROR(
                loom_parse_encoding_params(parser, &attrs, &attr_count));
            encoding.attribute_count = attr_count;
            encoding.attributes = attrs;
          }

          uint16_t encoding_id = 0;
          IREE_RETURN_IF_ERROR(loom_module_add_encoding(
              parser->module, &encoding, &encoding_id));

          // Register in alias table.
          IREE_RETURN_IF_ERROR(loom_alias_table_add(&parser->aliases,
                                                    &parser->parser_arena,
                                                    full_alias, encoding_id));
        }
        continue;
      }
      // Not an alias — this is an error.
      loom_diagnostic_param_t params[] = {
          loom_param_string(loom_token_lexeme(alias_token)),
      };
      IREE_RETURN_IF_ERROR(loom_parser_emit(parser, &loom_err_parse_014, params,
                                            IREE_ARRAYSIZE(params),
                                            alias_token));
      loom_parser_sync_to_newline(parser);
      continue;
    }

    // Parse a top-level op (function definition, etc.).
    uint32_t errors_before = parser->error_count;
    IREE_RETURN_IF_ERROR(loom_parse_op(parser));
    if (parser->error_count > errors_before) {
      loom_parser_sync_to_newline(parser);
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t loom_text_parse(iree_string_view_t source,
                              iree_string_view_t filename,
                              loom_context_t* context,
                              iree_arena_block_pool_t* block_pool,
                              const loom_text_parse_options_t* options,
                              loom_module_t** out_module) {
  *out_module = NULL;

  // Register the source filename for location tracking. This interns
  // the string into the context so locations can reference it by ID.
  loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
  if (!iree_string_view_is_empty(filename)) {
    IREE_RETURN_IF_ERROR(
        loom_context_register_source(context, filename, &source_id));
  }

  // Allocate the module using the context's host allocator.
  loom_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(loom_module_allocate(context, IREE_SV(""), block_pool,
                                            NULL, context->allocator, &module));

  // Initialize the parser.
  loom_parser_scope_t root_scope = {0};
  loom_parser_t parser = {
      .context = context,
      .module = module,
      .filename = filename,
      .source = source,
      .source_id = source_id,
      .diagnostic_sink =
          options ? options->diagnostic_sink : (loom_diagnostic_sink_t){0},
      .max_errors = options ? options->max_errors : 0,
      .scope = &root_scope,
  };
  if (parser.max_errors == 0) parser.max_errors = 20;
  loom_tokenizer_initialize(source, filename, &parser.tokenizer);
  iree_arena_initialize(block_pool, &parser.parser_arena);
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &parser.builder);

  // Parse the module body.
  iree_status_t status = loom_parse_module_body(&parser);

  // Check for tokenizer scan errors.
  if (iree_status_is_ok(status)) {
    status = loom_tokenizer_consume_status(&parser.tokenizer);
  }

  // Build use-def chains in one pass.
  if (iree_status_is_ok(status) && parser.error_count == 0) {
    status = loom_module_compute_uses(module);
  }

  // Cleanup.
  iree_arena_deinitialize(&parser.parser_arena);
  loom_tokenizer_deinitialize(&parser.tokenizer);

  // Infrastructure failure — propagate directly.
  if (!iree_status_is_ok(status)) {
    loom_module_free(module);
    return status;
  }

  // Parse errors were emitted as diagnostics but no infrastructure failure.
  // The caller checks *out_module — NULL means parse errors occurred.
  if (parser.error_count > 0) {
    loom_module_free(module);
    return iree_ok_status();
  }

  *out_module = module;
  return iree_ok_status();
}
