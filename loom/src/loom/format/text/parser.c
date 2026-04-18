// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/parser.h"

#include <inttypes.h>
#include <math.h>
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

// Resets a parser-owned scope frame for reuse under a new lexical parent while
// preserving the retained hash-table allocation.
static void loom_parser_scope_reset(loom_parser_scope_t* scope,
                                    loom_parser_scope_t* parent) {
  bool had_entries = scope->count > 0;
  scope->next_free = NULL;
  scope->parent = parent;
  scope->count = 0;
  if (had_entries && scope->capacity > 0) {
    loom_parser_scope_initialize_entries(scope->entries, scope->capacity);
  }
}

static bool loom_parse_special_f64_spelling(iree_string_view_t text,
                                            double* out_value) {
  if (iree_string_view_equal(text, IREE_SV("nan")) ||
      iree_string_view_equal(text, IREE_SV("-nan"))) {
    *out_value = NAN;
    return true;
  }
  if (iree_string_view_equal(text, IREE_SV("inf"))) {
    *out_value = INFINITY;
    return true;
  }
  if (iree_string_view_equal(text, IREE_SV("-inf"))) {
    *out_value = -INFINITY;
    return true;
  }
  return false;
}

static iree_status_t loom_parse_f64_token(loom_parser_t* parser,
                                          loom_token_t token,
                                          double* out_value) {
  if (loom_parse_special_f64_spelling(token.text, out_value)) {
    return iree_ok_status();
  }
  if (!iree_string_view_atod(token.text, out_value)) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(token.text),
    };
    return loom_parser_emit(parser,
                            loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 16),
                            params, IREE_ARRAYSIZE(params), token);
  }
  return iree_ok_status();
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

loom_value_id_t loom_parser_scope_lookup_local(const loom_parser_scope_t* scope,
                                               loom_string_id_t name_id) {
  if (name_id == LOOM_STRING_ID_INVALID) return LOOM_VALUE_ID_INVALID;
  if (!scope || scope->capacity == 0) return LOOM_VALUE_ID_INVALID;
  uint32_t hash = loom_parser_scope_hash(name_id);
  iree_host_size_t slot = hash & (scope->capacity - 1);
  while (scope->entries[slot].name_id != LOOM_STRING_ID_INVALID) {
    if (scope->entries[slot].name_id == name_id) {
      return scope->entries[slot].value_id;
    }
    slot = (slot + 1) & (scope->capacity - 1);
  }
  return LOOM_VALUE_ID_INVALID;
}

loom_value_id_t loom_parser_scope_lookup(const loom_parser_scope_t* scope,
                                         loom_string_id_t name_id) {
  if (name_id == LOOM_STRING_ID_INVALID) return LOOM_VALUE_ID_INVALID;
  const loom_parser_scope_t* current = scope;
  while (current) {
    loom_value_id_t value_id = loom_parser_scope_lookup_local(current, name_id);
    if (value_id != LOOM_VALUE_ID_INVALID) return value_id;
    current = current->parent;
  }
  return LOOM_VALUE_ID_INVALID;
}

iree_status_t loom_parser_scope_push(loom_parser_t* parser,
                                     loom_parser_scope_t* parent,
                                     loom_parser_scope_t** out_scope) {
  loom_parser_scope_t* scope = parser->scope_free_list;
  if (scope) {
    parser->scope_free_list = scope->next_free;
    loom_parser_scope_reset(scope, parent);
    *out_scope = scope;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      &parser->parser_arena, sizeof(loom_parser_scope_t), (void**)&scope));
  memset(scope, 0, sizeof(*scope));
  scope->parent = parent;
  *out_scope = scope;
  return iree_ok_status();
}

void loom_parser_scope_pop(loom_parser_t* parser) {
  loom_parser_scope_t* scope = parser->scope;
  IREE_ASSERT_ARGUMENT(scope);
  parser->scope = scope->parent;
  scope->parent = NULL;
  scope->next_free = parser->scope_free_list;
  parser->scope_free_list = scope;
}

//===----------------------------------------------------------------------===//
// Definition scope
//===----------------------------------------------------------------------===//

iree_status_t loom_parser_add_unresolved_placeholder(loom_parser_t* parser,
                                                     loom_value_id_t value_id,
                                                     loom_token_t name_token) {
  loom_parser_unresolved_placeholders_t* placeholders =
      &parser->unresolved_placeholders;
  if (placeholders->count >= placeholders->capacity) {
    iree_host_size_t capacity = placeholders->capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        &parser->parser_arena, placeholders->count, placeholders->count + 1,
        sizeof(loom_parser_unresolved_placeholder_t), &capacity,
        (void**)&placeholders->entries));
    placeholders->capacity = capacity;
  }
  placeholders->entries[placeholders->count++] =
      (loom_parser_unresolved_placeholder_t){
          .value_id = value_id,
          .name_token = name_token,
          .resolved = false,
      };
  return iree_ok_status();
}

static loom_parser_unresolved_placeholder_t*
loom_parser_lookup_unresolved_placeholder(loom_parser_t* parser,
                                          loom_value_id_t value_id) {
  if (!loom_parser_in_definition_scope(parser)) return NULL;
  for (iree_host_size_t i = parser->definition_scope.placeholder_start;
       i < parser->unresolved_placeholders.count; ++i) {
    loom_parser_unresolved_placeholder_t* placeholder =
        &parser->unresolved_placeholders.entries[i];
    if (placeholder->value_id == value_id) return placeholder;
  }
  return NULL;
}

iree_status_t loom_parser_definition_scope_push(
    loom_parser_t* parser, uint16_t pop_at,
    loom_parser_definition_scope_flags_t flags) {
  if (loom_parser_in_definition_scope(parser)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "nested Scope(...) format elements are not supported");
  }
  IREE_RETURN_IF_ERROR(
      loom_parser_scope_push(parser, parser->scope, &parser->scope));
  parser->definition_scope.placeholder_start =
      parser->unresolved_placeholders.count;
  parser->definition_scope.pop_at = pop_at;
  parser->definition_scope.flags = flags;
  return iree_ok_status();
}

void loom_parser_definition_scope_discard(loom_parser_t* parser) {
  if (!loom_parser_in_definition_scope(parser)) return;
  loom_parser_scope_pop(parser);
  parser->unresolved_placeholders.count =
      parser->definition_scope.placeholder_start;
  parser->definition_scope = (loom_parser_definition_scope_t){
      .placeholder_start = 0,
      .pop_at = UINT16_MAX,
      .flags = 0,
  };
}

iree_status_t loom_parser_definition_scope_pop_if_needed(
    loom_parser_t* parser, uint16_t format_index) {
  if (parser->definition_scope.pop_at != format_index) {
    return iree_ok_status();
  }
  const iree_host_size_t placeholder_start =
      parser->definition_scope.placeholder_start;
  for (iree_host_size_t i = placeholder_start;
       i < parser->unresolved_placeholders.count; ++i) {
    const loom_parser_unresolved_placeholder_t placeholder =
        parser->unresolved_placeholders.entries[i];
    if (placeholder.value_id == LOOM_VALUE_ID_INVALID ||
        placeholder.value_id >= parser->module->values.count) {
      continue;
    }
    if (placeholder.resolved) {
      continue;
    }
    loom_diagnostic_param_t params[] = {
        loom_param_string(placeholder.name_token.text),
    };
    iree_status_t status = loom_parser_emit(
        parser, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 22), params,
        IREE_ARRAYSIZE(params), placeholder.name_token);
    loom_parser_definition_scope_discard(parser);
    return status;
  }
  loom_parser_definition_scope_discard(parser);
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

iree_status_t loom_parser_emit_duplicate_value_name(loom_parser_t* parser,
                                                    loom_token_t name_token) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(name_token.text),
  };
  return loom_parser_emit(parser,
                          loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 2),
                          params, IREE_ARRAYSIZE(params), name_token);
}

iree_status_t loom_parser_define_value_name(loom_parser_t* parser,
                                            loom_token_t name_token,
                                            loom_value_id_t value_id) {
  if (iree_string_view_is_empty(name_token.text)) return iree_ok_status();

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(parser->module, name_token.text, &name_id));
  IREE_RETURN_IF_ERROR(
      loom_module_set_value_name(parser->module, value_id, name_id));
  bool duplicate = false;
  IREE_RETURN_IF_ERROR(loom_parser_scope_define(
      parser->scope, &parser->parser_arena, name_id, value_id, &duplicate));
  if (!duplicate) return iree_ok_status();
  return loom_parser_emit_duplicate_value_name(parser, name_token);
}

iree_status_t loom_parser_define_value(loom_parser_t* parser,
                                       loom_token_t name_token,
                                       loom_type_t type,
                                       loom_value_id_t* out_value_id) {
  if (iree_string_view_is_empty(name_token.text)) {
    return loom_module_define_value(parser->module, type, out_value_id);
  }

  loom_string_id_t name_id =
      loom_module_lookup_string(parser->module, name_token.text);
  if (name_id != LOOM_STRING_ID_INVALID) {
    loom_value_id_t value_id =
        loom_parser_scope_lookup_local(parser->scope, name_id);
    if (value_id != LOOM_VALUE_ID_INVALID) {
      loom_parser_unresolved_placeholder_t* placeholder =
          loom_parser_lookup_unresolved_placeholder(parser, value_id);
      if (placeholder && !placeholder->resolved) {
        loom_type_t placeholder_type =
            parser->module->values.entries[value_id].type;
        if (loom_type_kind(placeholder_type) != LOOM_TYPE_NONE &&
            !loom_type_equal(placeholder_type, type)) {
          return loom_parser_emit_duplicate_value_name(parser, name_token);
        }
        IREE_RETURN_IF_ERROR(
            loom_module_set_value_type(parser->module, value_id, type));
        placeholder->resolved = true;
        *out_value_id = value_id;
        return iree_ok_status();
      }
      return loom_parser_emit_duplicate_value_name(parser, name_token);
    }
  }

  IREE_RETURN_IF_ERROR(
      loom_module_define_value(parser->module, type, out_value_id));
  return loom_parser_define_value_name(parser, name_token, *out_value_id);
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
  return loom_parser_emit(parser,
                          loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 1),
                          params, IREE_ARRAYSIZE(params), name_token);
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
  return loom_parser_emit(parser,
                          loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 9),
                          params, IREE_ARRAYSIZE(params), op_name_token);
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
        .provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE,
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

  return loom_source_range_from_token(filename, source, token.source_text,
                                      token.line, token.column,
                                      token.end_column);
}

static iree_status_t loom_parser_format_token_text(
    loom_parser_t* parser, loom_token_t token,
    iree_string_view_t* out_display_text) {
  if (token.kind == LOOM_TOKEN_NONE || token.kind == LOOM_TOKEN_EOF) {
    *out_display_text = loom_token_kind_name(token.kind);
    return iree_ok_status();
  }

  if (token.kind == LOOM_TOKEN_STRING) {
    iree_host_size_t display_size = 2;
    for (iree_host_size_t i = 0; i < token.text.size; ++i) {
      switch (token.text.data[i]) {
        case '"':
        case '\\':
        case '\n':
        case '\t':
          display_size += 2;
          break;
        default:
          ++display_size;
          break;
      }
    }
    char* display_text = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        &parser->parser_arena, display_size, (void**)&display_text));
    iree_host_size_t display_offset = 0;
    display_text[display_offset++] = '"';
    for (iree_host_size_t i = 0; i < token.text.size; ++i) {
      switch (token.text.data[i]) {
        case '"':
          display_text[display_offset++] = '\\';
          display_text[display_offset++] = '"';
          break;
        case '\\':
          display_text[display_offset++] = '\\';
          display_text[display_offset++] = '\\';
          break;
        case '\n':
          display_text[display_offset++] = '\\';
          display_text[display_offset++] = 'n';
          break;
        case '\t':
          display_text[display_offset++] = '\\';
          display_text[display_offset++] = 't';
          break;
        default:
          display_text[display_offset++] = token.text.data[i];
          break;
      }
    }
    display_text[display_offset++] = '"';
    *out_display_text = iree_make_string_view(display_text, display_offset);
    return iree_ok_status();
  }

  char prefix = '\0';
  switch (token.kind) {
    case LOOM_TOKEN_SSA_VALUE:
      prefix = '%';
      break;
    case LOOM_TOKEN_SYMBOL:
      prefix = '@';
      break;
    case LOOM_TOKEN_HASH_ATTR:
      prefix = '#';
      break;
    case LOOM_TOKEN_BLOCK_LABEL:
      prefix = '^';
      break;
    default:
      *out_display_text = token.text;
      return iree_ok_status();
  }

  iree_host_size_t display_size = token.text.size + 1;
  char* display_text = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&parser->parser_arena, display_size,
                                           (void**)&display_text));
  display_text[0] = prefix;
  if (!iree_string_view_is_empty(token.text)) {
    memcpy(display_text + 1, token.text.data, token.text.size);
  }
  *out_display_text = iree_make_string_view(display_text, display_size);
  return iree_ok_status();
}

static iree_status_t loom_parser_emit_diagnostic(
    loom_parser_t* parser, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count,
    loom_token_t token,
    const loom_diagnostic_related_location_t* related_locations,
    iree_host_size_t related_location_count) {
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
      .related_locations = related_locations,
      .related_location_count = related_location_count,
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
        .error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 12),
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

iree_status_t loom_parser_emit(loom_parser_t* parser,
                               const loom_error_def_t* error,
                               const loom_diagnostic_param_t* params,
                               iree_host_size_t param_count,
                               loom_token_t token) {
  return loom_parser_emit_diagnostic(parser, error, params, param_count, token,
                                     NULL, 0);
}

iree_status_t loom_parser_emit_related(loom_parser_t* parser,
                                       const loom_error_def_t* error,
                                       const loom_diagnostic_param_t* params,
                                       iree_host_size_t param_count,
                                       loom_token_t token,
                                       iree_string_view_t related_label,
                                       loom_token_t related_token) {
  loom_diagnostic_related_location_t related_location = {
      .label = related_label,
      .source_location = loom_parser_token_origin(
          parser->filename, parser->source, related_token),
  };
  return loom_parser_emit_diagnostic(parser, error, params, param_count, token,
                                     &related_location, 1);
}

static iree_status_t loom_parser_emit_tokenizer_error(loom_parser_t* parser,
                                                      loom_token_t token) {
  IREE_ASSERT_ARGUMENT(parser->tokenizer.error.error);
  return loom_parser_emit(parser, parser->tokenizer.error.error,
                          parser->tokenizer.error.params,
                          parser->tokenizer.error.param_count, token);
}

iree_status_t loom_parser_emit_unexpected_token(loom_parser_t* parser,
                                                loom_token_t token,
                                                iree_string_view_t expected) {
  if (token.kind == LOOM_TOKEN_ERROR) {
    return loom_parser_emit_tokenizer_error(parser, token);
  }
  iree_string_view_t actual_text = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loom_parser_format_token_text(parser, token, &actual_text));
  loom_diagnostic_param_t params[] = {
      loom_param_string(actual_text),
      loom_param_string(expected),
  };
  return loom_parser_emit(parser,
                          loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 3),
                          params, IREE_ARRAYSIZE(params), token);
}

iree_status_t loom_parser_emit_token_text_error(loom_parser_t* parser,
                                                const loom_error_def_t* error,
                                                loom_token_t token) {
  if (token.kind == LOOM_TOKEN_ERROR) {
    return loom_parser_emit_tokenizer_error(parser, token);
  }
  iree_string_view_t token_text = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loom_parser_format_token_text(parser, token, &token_text));
  loom_diagnostic_param_t params[] = {
      loom_param_string(token_text),
  };
  return loom_parser_emit(parser, error, params, IREE_ARRAYSIZE(params), token);
}

iree_status_t loom_parser_expect(loom_parser_t* parser, loom_token_kind_t kind,
                                 loom_token_t* out_token) {
  // Propagate pending tokenizer infrastructure failures. Malformed user input
  // arrives as LOOM_TOKEN_ERROR and is emitted through the parser sink below.
  if (!iree_status_is_ok(parser->tokenizer.status)) {
    return loom_tokenizer_consume_status(&parser->tokenizer);
  }
  loom_token_t token = loom_tokenizer_next(&parser->tokenizer);
  // A scan inside next() may have produced an infrastructure failure.
  if (!iree_status_is_ok(parser->tokenizer.status)) {
    return loom_tokenizer_consume_status(&parser->tokenizer);
  }
  if (token.kind == LOOM_TOKEN_ERROR) {
    return loom_parser_emit_tokenizer_error(parser, token);
  }
  if (token.kind != kind) {
    return loom_parser_emit_unexpected_token(parser, token,
                                             loom_token_kind_name(kind));
  }
  if (out_token) *out_token = token;
  return iree_ok_status();
}

bool loom_parser_at_error_limit(const loom_parser_t* parser) {
  return parser->max_errors > 0 && parser->error_count >= parser->max_errors;
}

//===----------------------------------------------------------------------===//
// Location parsing
//===----------------------------------------------------------------------===//

enum {
  LOOM_PARSER_LOCATION_MAX_NESTING_DEPTH = 16,
};

static iree_status_t loom_parser_emit_location_error(loom_parser_t* parser,
                                                     iree_string_view_t detail,
                                                     loom_token_t token) {
  if (token.kind == LOOM_TOKEN_ERROR) {
    return loom_parser_emit_tokenizer_error(parser, token);
  }
  loom_diagnostic_param_t params[] = {
      loom_param_string(detail),
  };
  return loom_parser_emit(parser,
                          loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 11),
                          params, IREE_ARRAYSIZE(params), token);
}

static iree_status_t loom_parser_resolve_location_source(
    loom_parser_t* parser, iree_string_view_t source_name,
    loom_source_id_t* out_source_id) {
  if (parser->cached_location.source_id != LOOM_SOURCE_ID_INVALID &&
      iree_string_view_equal(parser->cached_location.source_name,
                             source_name)) {
    *out_source_id = parser->cached_location.source_id;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_context_register_source(
      parser->context, source_name, out_source_id));
  parser->cached_location.source_name = source_name;
  parser->cached_location.source_id = *out_source_id;
  return iree_ok_status();
}

static iree_status_t loom_parse_location_u16(loom_parser_t* parser,
                                             uint16_t* out_value) {
  loom_token_t token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_INTEGER, &token);

  int64_t value = 0;
  if (!iree_string_view_atoi_int64(token.text, &value) || value < 0 ||
      value > UINT16_MAX) {
    return loom_parser_emit_location_error(
        parser, IREE_SV("line/column must be an integer in [0, 65535]"), token);
  }

  *out_value = (uint16_t)value;
  return iree_ok_status();
}

static iree_status_t loom_parse_location_coordinate_pair(loom_parser_t* parser,
                                                         uint16_t* out_line,
                                                         uint16_t* out_column) {
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COLON, NULL);
  IREE_RETURN_IF_ERROR(loom_parse_location_u16(parser, out_line));
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COLON, NULL);
  return loom_parse_location_u16(parser, out_column);
}

static iree_status_t loom_parse_location_body(loom_parser_t* parser,
                                              uint16_t nesting_depth,
                                              loom_location_id_t* out_location);

static iree_status_t loom_parse_file_location_body(
    loom_parser_t* parser, loom_location_id_t* out_location) {
  loom_token_t source_token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_STRING, &source_token);

  loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_parser_resolve_location_source(
      parser, source_token.text, &source_id));

  uint16_t start_line = 0;
  uint16_t start_column = 0;
  IREE_RETURN_IF_ERROR(
      loom_parse_location_coordinate_pair(parser, &start_line, &start_column));

  uint16_t end_line = start_line;
  uint16_t end_column = start_column;
  if (loom_tokenizer_try_consume_keyword(&parser->tokenizer, IREE_SV("to"))) {
    IREE_RETURN_IF_ERROR(loom_parse_location_u16(parser, &end_line));
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COLON, NULL);
    IREE_RETURN_IF_ERROR(loom_parse_location_u16(parser, &end_column));
  }

  loom_location_entry_t entry = loom_location_file_range(
      source_id, start_line, start_column, end_line, end_column);
  return loom_module_add_location(parser->module, entry, out_location);
}

static iree_status_t loom_parse_fused_location_body(
    loom_parser_t* parser, uint16_t nesting_depth,
    loom_location_id_t* out_location) {
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, NULL);
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LANGLE, NULL);

  uint32_t errors_before = parser->error_count;
  loom_location_id_t* child_locations = NULL;
  iree_host_size_t child_count = 0;
  iree_host_size_t child_capacity = 0;
  if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RANGLE)) {
    for (;;) {
      if (child_count >= child_capacity) {
        IREE_RETURN_IF_ERROR(iree_arena_grow_array(
            &parser->parser_arena, child_count, /*minimum_capacity=*/8,
            sizeof(*child_locations), &child_capacity,
            (void**)&child_locations));
      }
      IREE_RETURN_IF_ERROR(
          loom_parse_location_body(parser, (uint16_t)(nesting_depth + 1),
                                   &child_locations[child_count]));
      if (parser->error_count > errors_before) return iree_ok_status();
      ++child_count;
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }
  }
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RANGLE, NULL);

  if (child_count > UINT32_MAX) {
    loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_location_error(
        parser, IREE_SV("fused location has too many children"), token);
  }

  loom_location_id_t* module_children = NULL;
  if (child_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &parser->module->arena, child_count, sizeof(*module_children),
        (void**)&module_children));
    memcpy(module_children, child_locations,
           child_count * sizeof(*module_children));
  }

  loom_location_entry_t entry = {
      .kind = LOOM_LOCATION_FUSED,
  };
  entry.fused.count = (uint32_t)child_count;
  entry.fused.children = module_children;
  return loom_module_add_location(parser->module, entry, out_location);
}

static iree_status_t loom_parse_opaque_location_body(
    loom_parser_t* parser, loom_location_id_t* out_location) {
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, NULL);
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LANGLE, NULL);

  loom_token_t source_token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_STRING, &source_token);
  loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_parser_resolve_location_source(
      parser, source_token.text, &source_id));

  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_COMMA, NULL);

  loom_token_t data_token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_STRING, &data_token);
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RANGLE, NULL);

  if (data_token.text.size > UINT32_MAX) {
    return loom_parser_emit_location_error(
        parser, IREE_SV("opaque location data is too large"), data_token);
  }

  uint8_t* module_data = NULL;
  if (!iree_string_view_is_empty(data_token.text)) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        &parser->module->arena, data_token.text.size, (void**)&module_data));
    memcpy(module_data, data_token.text.data, data_token.text.size);
  }

  loom_location_entry_t entry = {
      .kind = LOOM_LOCATION_OPAQUE,
  };
  entry.opaque.source_id = source_id;
  entry.opaque.data_length = (uint32_t)data_token.text.size;
  entry.opaque.data = module_data;
  return loom_module_add_location(parser->module, entry, out_location);
}

static iree_status_t loom_parse_location_body(
    loom_parser_t* parser, uint16_t nesting_depth,
    loom_location_id_t* out_location) {
  if (nesting_depth >= LOOM_PARSER_LOCATION_MAX_NESTING_DEPTH) {
    loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_location_error(
        parser, IREE_SV("location nesting exceeds the maximum supported depth"),
        token);
  }

  loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
  if (token.kind == LOOM_TOKEN_STRING) {
    return loom_parse_file_location_body(parser, out_location);
  }
  if (token.kind == LOOM_TOKEN_BARE_IDENT &&
      iree_string_view_equal(token.text, IREE_SV("fused"))) {
    return loom_parse_fused_location_body(parser, nesting_depth, out_location);
  }
  if (token.kind == LOOM_TOKEN_BARE_IDENT &&
      iree_string_view_equal(token.text, IREE_SV("opaque"))) {
    return loom_parse_opaque_location_body(parser, out_location);
  }
  return loom_parser_emit_location_error(
      parser, IREE_SV("expected a file location string, 'fused', or 'opaque'"),
      token);
}

static iree_status_t loom_parse_optional_op_location(
    loom_parser_t* parser, loom_location_id_t fallback_location,
    loom_location_id_t* out_location) {
  *out_location = fallback_location;
  if (!loom_tokenizer_at_keyword(&parser->tokenizer, IREE_SV("loc"))) {
    return iree_ok_status();
  }

  uint32_t errors_before = parser->error_count;
  if (!loom_tokenizer_try_consume_keyword(&parser->tokenizer, IREE_SV("loc"))) {
    loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, token, IREE_SV("'loc'"));
  }
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LPAREN, NULL);
  IREE_RETURN_IF_ERROR(
      loom_parse_location_body(parser, /*nesting_depth=*/0, out_location));
  if (parser->error_count > errors_before) return iree_ok_status();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RPAREN, NULL);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Error recovery
//===----------------------------------------------------------------------===//

void loom_parser_sync_to_newline(loom_parser_t* parser) {
  // The tokenizer doesn't expose "at start of line" directly, but
  // we can advance until we find an op-starting token or EOF.
  // A simple heuristic: advance until we see a token at column 1 or
  // whose kind is SSA_VALUE (result name), OP_NAME, LOOM_TOKEN_ERROR
  // (lexical error at the next sibling op), BLOCK_LABEL, RBRACE (end of
  // region), or EOF.
  for (;;) {
    loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
    if (token.kind == LOOM_TOKEN_EOF) break;
    if (token.kind == LOOM_TOKEN_RBRACE) break;
    // If token is at column 1 (or 3+ for indented ops), it's likely
    // the start of a new op. We use the heuristic that SSA_VALUE at
    // position <= the next line start means we've synced.
    if (token.kind == LOOM_TOKEN_SSA_VALUE && token.column <= 2) break;
    if (token.kind == LOOM_TOKEN_OP_NAME && token.column <= 2) break;
    if (token.kind == LOOM_TOKEN_ERROR && token.column <= 2) break;
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
  parsed->successors = parsed->inline_successors;
  parsed->successor_label_tokens = parsed->inline_successor_label_tokens;
  parsed->successor_capacity = LOOM_PARSED_OP_INLINE_SUCCESSORS;
  parsed->result_ids = parsed->inline_result_ids;
  parsed->result_name_tokens = parsed->inline_result_name_tokens;
  parsed->result_capacity = LOOM_PARSED_OP_INLINE_RESULTS;
  parsed->attributes = parsed->inline_attributes;
  parsed->attribute_capacity = LOOM_PARSED_OP_INLINE_ATTRS;
  parsed->regions = parsed->inline_regions;
  parsed->region_capacity = LOOM_PARSED_OP_INLINE_REGIONS;
  parsed->tied_results = parsed->inline_tied_results;
  parsed->tied_result_capacity = LOOM_PARSED_OP_INLINE_TIED;
  parsed->field_spans = parsed->inline_field_spans;
  parsed->field_span_capacity = LOOM_PARSED_OP_INLINE_FIELD_SPANS;
}

void loom_parsed_op_reset(loom_parsed_op_t* parsed) {
  parsed->operand_count = 0;
  parsed->successor_count = 0;
  parsed->result_count = 0;
  parsed->tied_result_count = 0;
  parsed->field_span_count = 0;
  parsed->attribute_count = 0;
  parsed->region_count = 0;
  parsed->instance_flags = 0;
}

iree_status_t loom_parser_acquire_parsed_op(loom_parser_t* parser,
                                            loom_parsed_op_t** out_parsed) {
  loom_parsed_op_t* parsed = parser->parsed_op_free_list;
  if (parsed) {
    parser->parsed_op_free_list = parsed->next_free;
    parsed->next_free = NULL;
    loom_parsed_op_reset(parsed);
    *out_parsed = parsed;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate(&parser->parser_arena,
                                           sizeof(*parsed), (void**)&parsed));
  loom_parsed_op_initialize(parsed);
  *out_parsed = parsed;
  return iree_ok_status();
}

void loom_parser_release_parsed_op(loom_parser_t* parser,
                                   loom_parsed_op_t* parsed) {
  if (!parsed) return;
  parsed->next_free = parser->parsed_op_free_list;
  parser->parsed_op_free_list = parsed;
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

iree_status_t loom_parsed_op_set_operand(loom_parsed_op_t* parsed,
                                         iree_arena_allocator_t* arena,
                                         uint16_t index,
                                         loom_value_id_t value_id) {
  iree_host_size_t required_capacity = (iree_host_size_t)index + 1;
  if (required_capacity > parsed->operand_capacity) {
    iree_host_size_t capacity = parsed->operand_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, parsed->operand_count, required_capacity,
        sizeof(loom_value_id_t), &capacity, (void**)&parsed->operand_ids));
    parsed->operand_capacity = (uint16_t)capacity;
  }
  while (parsed->operand_count <= index) {
    parsed->operand_ids[parsed->operand_count++] = LOOM_VALUE_ID_INVALID;
  }
  parsed->operand_ids[index] = value_id;
  return iree_ok_status();
}

iree_status_t loom_parsed_op_set_successor(loom_parsed_op_t* parsed,
                                           iree_arena_allocator_t* arena,
                                           uint8_t index, loom_block_t* block,
                                           loom_token_t label_token) {
  if (index == UINT8_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "successor field index exceeds storage limit");
  }
  iree_host_size_t required_capacity = (iree_host_size_t)index + 1;
  if (required_capacity > parsed->successor_capacity) {
    iree_host_size_t capacity = parsed->successor_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, parsed->successor_count, required_capacity,
        sizeof(loom_block_t*), &capacity, (void**)&parsed->successors));
    iree_host_size_t token_capacity = parsed->successor_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, parsed->successor_count, capacity, sizeof(loom_token_t),
        &token_capacity, (void**)&parsed->successor_label_tokens));
    IREE_ASSERT(token_capacity == capacity);
    parsed->successor_capacity = (uint8_t)capacity;
  }
  while (parsed->successor_count <= index) {
    parsed->successors[parsed->successor_count] = NULL;
    parsed->successor_label_tokens[parsed->successor_count] = loom_token_none();
    ++parsed->successor_count;
  }
  parsed->successors[index] = block;
  parsed->successor_label_tokens[index] = label_token;
  return iree_ok_status();
}

iree_status_t loom_parsed_op_add_result(loom_parsed_op_t* parsed,
                                        iree_arena_allocator_t* arena,
                                        loom_value_id_t value_id,
                                        loom_token_t name_token) {
  IREE_RETURN_IF_ERROR(loom_parsed_op_add_field_span(
      parsed, arena, LOOM_LOCATION_FIELD_RESULT, parsed->result_count,
      name_token, name_token.line, name_token.end_column));
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
  if (field_index > parsed->attribute_count) {
    memset(&parsed->attributes[parsed->attribute_count], 0,
           (field_index - parsed->attribute_count) * sizeof(loom_attribute_t));
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

iree_status_t loom_parsed_op_set_region(loom_parsed_op_t* parsed,
                                        iree_arena_allocator_t* arena,
                                        uint8_t index, loom_region_t* region) {
  iree_host_size_t required_capacity = (iree_host_size_t)index + 1;
  if (required_capacity > parsed->region_capacity) {
    iree_host_size_t capacity = parsed->region_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, parsed->region_count, required_capacity, sizeof(loom_region_t*),
        &capacity, (void**)&parsed->regions));
    parsed->region_capacity = (uint8_t)capacity;
  }
  while (parsed->region_count <= index) {
    parsed->regions[parsed->region_count++] = NULL;
  }
  parsed->regions[index] = region;
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

iree_status_t loom_parsed_op_add_field_span(
    loom_parsed_op_t* parsed, iree_arena_allocator_t* arena,
    loom_location_field_kind_t kind, uint16_t index, loom_token_t start_token,
    uint32_t end_line, uint32_t end_column) {
  if (start_token.kind == LOOM_TOKEN_NONE ||
      start_token.kind == LOOM_TOKEN_EOF) {
    return iree_ok_status();
  }
  if (parsed->field_span_count >= parsed->field_span_capacity) {
    iree_host_size_t capacity = parsed->field_span_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, parsed->field_span_count, 0, sizeof(loom_location_field_span_t),
        &capacity, (void**)&parsed->field_spans));
    parsed->field_span_capacity = (uint16_t)capacity;
  }
  parsed->field_spans[parsed->field_span_count++] =
      (loom_location_field_span_t){
          .kind = kind,
          .index = index,
          .start_line = (uint16_t)start_token.line,
          .start_col = (uint16_t)start_token.column,
          .end_line = (uint16_t)end_line,
          .end_col = (uint16_t)end_column,
      };
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
static void loom_parser_pending_block_args_clear(
    loom_parser_pending_block_args_t* pending_block_args) {
  pending_block_args->count = 0;
}

//===----------------------------------------------------------------------===//
// Attribute parsing
//===----------------------------------------------------------------------===//

static iree_status_t loom_parse_i64_array_attr(loom_parser_t* parser,
                                               loom_attribute_t* out_attr) {
  // [1, 2, 3]
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LBRACKET)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'['"));
  }

  int64_t stack_values[32];
  uint16_t count = 0;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACKET) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (count > 0) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }
    loom_token_t token = loom_token_none();
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_INTEGER, &token);
    if (count >= IREE_ARRAYSIZE(stack_values)) {
      return loom_parser_emit_token_text_error(
          parser, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 4), token);
    }
    int64_t value = 0;
    if (!iree_string_view_atoi_int64(token.text, &value)) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(token.text),
      };
      return loom_parser_emit(
          parser, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 15), params,
          IREE_ARRAYSIZE(params), token);
    }
    stack_values[count++] = value;
  }
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RBRACKET)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("']'"));
  }

  int64_t* arena_values = NULL;
  if (count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &parser->module->arena, count, sizeof(int64_t), (void**)&arena_values));
    memcpy(arena_values, stack_values, count * sizeof(int64_t));
  }
  *out_attr = loom_attr_i64_array(arena_values, count);
  return iree_ok_status();
}

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
        return loom_parser_emit(
            parser, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 15), params,
            IREE_ARRAYSIZE(params), token);
      }
      *out_attr = loom_attr_i64(value);
      return iree_ok_status();
    }
    case LOOM_ATTR_F64: {
      loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
      if (token.kind == LOOM_TOKEN_BARE_IDENT) {
        double special_value = 0.0;
        if (!loom_parse_special_f64_spelling(token.text, &special_value)) {
          return loom_parser_emit_unexpected_token(parser, token,
                                                   IREE_SV("FLOAT"));
        }
      } else if (token.kind != LOOM_TOKEN_FLOAT) {
        return loom_parser_emit_unexpected_token(parser, token,
                                                 IREE_SV("FLOAT"));
      }
      token = loom_tokenizer_next(&parser->tokenizer);
      double value = 0.0;
      IREE_RETURN_IF_ERROR(loom_parse_f64_token(parser, token, &value));
      *out_attr = loom_attr_f64(value);
      return iree_ok_status();
    }
    case LOOM_ATTR_STRING: {
      loom_token_t token = loom_token_none();
      LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_STRING, &token);
      // Token text is the decoded content without surrounding quotes, so
      // intern it directly.
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
        return loom_parser_emit_unexpected_token(parser, peek,
                                                 IREE_SV("'true' or 'false'"));
      }
      return iree_ok_status();
    }
    case LOOM_ATTR_SYMBOL: {
      return loom_parse_symbol_ref_attr(parser, out_attr);
    }
    case LOOM_ATTR_ENUM: {
      loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
      if (token.kind != LOOM_TOKEN_BARE_IDENT &&
          token.kind != LOOM_TOKEN_OP_NAME) {
        return loom_parser_emit_unexpected_token(parser, token,
                                                 IREE_SV("identifier"));
      }
      (void)loom_tokenizer_next(&parser->tokenizer);
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
        return loom_parser_emit(
            parser, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 17), params,
            IREE_ARRAYSIZE(params), token);
      }
    }
    case LOOM_ATTR_I64_ARRAY: {
      return loom_parse_i64_array_attr(parser, out_attr);
    }
    case LOOM_ATTR_TYPE: {
      loom_type_t type = {0};
      IREE_RETURN_IF_ERROR(
          loom_parse_type(parser, LOOM_TYPE_PARSE_BODY, &type));
      loom_type_id_t type_id = LOOM_TYPE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_module_intern_type_id(parser->module, type, &type_id));
      *out_attr = loom_attr_type(type_id);
      return iree_ok_status();
    }
    case LOOM_ATTR_ENCODING: {
      uint16_t encoding_id = 0;
      IREE_RETURN_IF_ERROR(loom_parse_static_encoding(
          parser, LOOM_STRING_ID_INVALID, &encoding_id));
      *out_attr = loom_attr_encoding(encoding_id);
      return iree_ok_status();
    }
    case LOOM_ATTR_ANY:
      return loom_parse_generic_attr_value(parser, /*nesting_depth=*/0,
                                           out_attr);
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported attribute kind %d",
                              (int)descriptor->attr_kind);
  }
}

iree_status_t loom_parse_symbol_ref_attr(loom_parser_t* parser,
                                         loom_attribute_t* out_attr) {
  loom_token_t token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_SYMBOL, &token);
  loom_string_id_t name_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(parser->module, token.text, &name_id));
  loom_symbol_ref_t ref = {.module_id = 0};
  ref.symbol_id = loom_symbol_map_find(&parser->symbol_lookup, name_id);
  if (ref.symbol_id == LOOM_SYMBOL_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_module_add_symbol(parser->module, name_id, &ref.symbol_id));
    IREE_RETURN_IF_ERROR(loom_symbol_map_insert(
        &parser->symbol_lookup, &parser->parser_arena, name_id, ref.symbol_id));
  }
  *out_attr = loom_attr_symbol(ref);
  return iree_ok_status();
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
                     "ne",
     LOOM_PREDICATE_NE},
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
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'['"));
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
      return loom_parser_emit_token_text_error(
          parser, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 4), peek);
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
      return loom_parser_emit(
          parser, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 13), params,
          IREE_ARRAYSIZE(params), name_token);
    }

    // Expect '('.
    if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LPAREN)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'('"));
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
        return loom_parser_emit_token_text_error(
            parser, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 4), peek);
      }

      loom_token_t arg_token = loom_tokenizer_peek(&parser->tokenizer);
      if (arg_token.kind == LOOM_TOKEN_SSA_VALUE) {
        // SSA value reference.
        loom_tokenizer_next(&parser->tokenizer);
        loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
        LOOM_PARSE_RESOLVE_VALUE(parser, arg_token, &value_id);
        predicate.arg_tags[predicate.arg_count] = LOOM_PRED_ARG_VALUE;
        predicate.args[predicate.arg_count] = (int64_t)value_id;
      } else if (arg_token.kind == LOOM_TOKEN_INTEGER) {
        // Constant.
        loom_tokenizer_next(&parser->tokenizer);
        int64_t value = 0;
        if (!iree_string_view_atoi_int64(arg_token.text, &value)) {
          loom_diagnostic_param_t params[] = {
              loom_param_string(arg_token.text),
          };
          return loom_parser_emit(
              parser, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 15),
              params, IREE_ARRAYSIZE(params), arg_token);
        }
        predicate.arg_tags[predicate.arg_count] = LOOM_PRED_ARG_CONST;
        predicate.args[predicate.arg_count] = value;
      } else {
        return loom_parser_emit_unexpected_token(
            parser, arg_token, IREE_SV("a predicate argument"));
      }
      ++predicate.arg_count;
    }

    if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RPAREN)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("')'"));
    }
    uint8_t expected_argument_count =
        loom_predicate_kind_argument_count(pred_kind);
    if (predicate.arg_count != expected_argument_count) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(name_token.text),
          loom_param_u32(expected_argument_count),
          loom_param_u32(predicate.arg_count),
      };
      return loom_parser_emit(
          parser, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 31), params,
          IREE_ARRAYSIZE(params), name_token);
    }

    stack_predicates[count++] = predicate;
  }

  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_RBRACKET)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("']'"));
  }

  loom_predicate_t* arena_predicates = NULL;
  if (count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &parser->module->arena, count, sizeof(loom_predicate_t),
        (void**)&arena_predicates));
    memcpy(arena_predicates, stack_predicates,
           count * sizeof(loom_predicate_t));
  }
  *out_attr = loom_attr_predicate_list(arena_predicates, count);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Dict attribute parsing
//===----------------------------------------------------------------------===//

typedef struct loom_parsed_attr_dict_entry_t {
  loom_named_attr_t attr;
  loom_token_t key_token;
} loom_parsed_attr_dict_entry_t;

static iree_status_t loom_parser_emit_duplicate_attr_dict_key(
    loom_parser_t* parser, loom_token_t key_token,
    loom_token_t previous_key_token) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(key_token.text),
  };
  return loom_parser_emit_related(
      parser, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 20), params,
      IREE_ARRAYSIZE(params), key_token, IREE_SV("previously defined here"),
      previous_key_token);
}

static iree_string_view_t loom_parsed_attr_dict_entry_name(
    const loom_module_t* module, const loom_parsed_attr_dict_entry_t* entry) {
  return module->strings.entries[entry->attr.name_id];
}

static iree_status_t loom_parse_present_attr_dict(loom_parser_t* parser,
                                                  uint16_t nesting_depth,
                                                  loom_attribute_t* out_attr);

iree_status_t loom_parse_generic_attr_value(loom_parser_t* parser,
                                            uint16_t nesting_depth,
                                            loom_attribute_t* out_attr) {
  loom_token_t value_token = loom_tokenizer_peek(&parser->tokenizer);
  switch (value_token.kind) {
    case LOOM_TOKEN_INTEGER: {
      loom_tokenizer_next(&parser->tokenizer);
      int64_t int_value = 0;
      if (!iree_string_view_atoi_int64(value_token.text, &int_value)) {
        loom_diagnostic_param_t params[] = {
            loom_param_string(value_token.text),
        };
        return loom_parser_emit(
            parser, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 15), params,
            IREE_ARRAYSIZE(params), value_token);
      }
      *out_attr = loom_attr_i64(int_value);
      return iree_ok_status();
    }
    case LOOM_TOKEN_FLOAT: {
      loom_tokenizer_next(&parser->tokenizer);
      double float_value = 0.0;
      IREE_RETURN_IF_ERROR(
          loom_parse_f64_token(parser, value_token, &float_value));
      *out_attr = loom_attr_f64(float_value);
      return iree_ok_status();
    }
    case LOOM_TOKEN_STRING: {
      loom_tokenizer_next(&parser->tokenizer);
      loom_string_id_t string_id = LOOM_STRING_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_module_intern_string(
          parser->module, value_token.text, &string_id));
      *out_attr = loom_attr_string(string_id);
      return iree_ok_status();
    }
    case LOOM_TOKEN_SYMBOL:
      return loom_parse_symbol_ref_attr(parser, out_attr);
    case LOOM_TOKEN_BARE_IDENT: {
      double special_value = 0.0;
      if (loom_parse_special_f64_spelling(value_token.text, &special_value)) {
        loom_tokenizer_next(&parser->tokenizer);
        *out_attr = loom_attr_f64(special_value);
        return iree_ok_status();
      }
      if (iree_string_view_equal(value_token.text, IREE_SV("true"))) {
        loom_tokenizer_next(&parser->tokenizer);
        *out_attr = loom_attr_bool(true);
        return iree_ok_status();
      }
      if (iree_string_view_equal(value_token.text, IREE_SV("false"))) {
        loom_tokenizer_next(&parser->tokenizer);
        *out_attr = loom_attr_bool(false);
        return iree_ok_status();
      }
      loom_tokenizer_next(&parser->tokenizer);
      loom_string_id_t ident_id = LOOM_STRING_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_module_intern_string(
          parser->module, value_token.text, &ident_id));
      *out_attr = loom_attr_string(ident_id);
      return iree_ok_status();
    }
    case LOOM_TOKEN_LBRACKET:
      return loom_parse_i64_array_attr(parser, out_attr);
    case LOOM_TOKEN_HASH_ATTR: {
      uint16_t encoding_id = 0;
      IREE_RETURN_IF_ERROR(loom_parse_static_encoding(
          parser, LOOM_STRING_ID_INVALID, &encoding_id));
      *out_attr = loom_attr_encoding(encoding_id);
      return iree_ok_status();
    }
    case LOOM_TOKEN_LBRACE:
      return loom_parse_present_attr_dict(parser, (uint16_t)(nesting_depth + 1),
                                          out_attr);
    default:
      break;
  }
  return loom_parser_emit_unexpected_token(parser, value_token,
                                           IREE_SV("an attribute value"));
}

static void loom_parser_sort_attr_dict_entries(
    const loom_module_t* module, loom_parsed_attr_dict_entry_t* entries,
    uint16_t count) {
  for (uint16_t i = 1; i < count; ++i) {
    loom_parsed_attr_dict_entry_t entry = entries[i];
    iree_string_view_t key_name =
        loom_parsed_attr_dict_entry_name(module, &entry);
    uint16_t slot = i;
    while (slot > 0) {
      iree_string_view_t previous_key_name =
          loom_parsed_attr_dict_entry_name(module, &entries[slot - 1]);
      if (iree_string_view_compare(previous_key_name, key_name) <= 0) {
        break;
      }
      entries[slot] = entries[slot - 1];
      --slot;
    }
    entries[slot] = entry;
  }
}

static iree_status_t loom_parse_present_attr_dict(loom_parser_t* parser,
                                                  uint16_t nesting_depth,
                                                  loom_attribute_t* out_attr) {
  loom_token_t open_brace_token = loom_token_none();
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_LBRACE, &open_brace_token);
  if (nesting_depth >= LOOM_ATTR_DICT_MAX_NESTING_DEPTH) {
    loom_diagnostic_param_t params[] = {
        loom_param_u32(LOOM_ATTR_DICT_MAX_NESTING_DEPTH),
    };
    return loom_parser_emit(parser,
                            loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 21),
                            params, IREE_ARRAYSIZE(params), open_brace_token);
  }

  uint32_t errors_before = parser->error_count;
  loom_parsed_attr_dict_entry_t stack_entries[16];
  uint16_t count = 0;

  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACE) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (count > 0) {
      if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_COMMA)) {
        break;
      }
    }
    if (count >= IREE_ARRAYSIZE(stack_entries)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_token_text_error(
          parser, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 4), peek);
    }

    loom_token_t key_token = loom_token_none();
    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_BARE_IDENT, &key_token);
    loom_string_id_t key_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_module_intern_string(parser->module, key_token.text, &key_id));
    for (uint16_t i = 0; i < count; ++i) {
      if (stack_entries[i].attr.name_id == key_id) {
        return loom_parser_emit_duplicate_attr_dict_key(
            parser, key_token, stack_entries[i].key_token);
      }
    }

    LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_EQUALS, NULL);

    loom_attribute_t value = {0};
    IREE_RETURN_IF_ERROR(
        loom_parse_generic_attr_value(parser, nesting_depth, &value));
    if (parser->error_count > errors_before) return iree_ok_status();

    stack_entries[count].attr.name_id = key_id;
    stack_entries[count].attr.reserved = 0;
    stack_entries[count].attr.value = value;
    stack_entries[count].key_token = key_token;
    ++count;
  }

  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RBRACE, NULL);

  if (count == 0) {
    *out_attr = loom_make_canonical_attr_dict(NULL, 0);
    return iree_ok_status();
  }

  loom_parser_sort_attr_dict_entries(parser->module, stack_entries, count);

  loom_named_attr_t* arena_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&parser->module->arena, count,
                                                 sizeof(*arena_entries),
                                                 (void**)&arena_entries));
  for (uint16_t i = 0; i < count; ++i) {
    arena_entries[i] = stack_entries[i].attr;
  }
  *out_attr = loom_make_canonical_attr_dict(arena_entries, count);
  return iree_ok_status();
}

iree_status_t loom_parse_attr_dict(loom_parser_t* parser,
                                   loom_attribute_t* out_attr) {
  // {key = value, key = value, ...}
  if (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_LBRACE)) {
    // Empty dict — no brace means absent.
    memset(out_attr, 0, sizeof(*out_attr));
    return iree_ok_status();
  }
  return loom_parse_present_attr_dict(parser, /*nesting_depth=*/0, out_attr);
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
      loom_bstring_t keyword_bstring =
          loom_keyword_bstring((loom_keyword_id_t)keyword_id);
      if (!keyword_bstring) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "keyword id %u is out of range", keyword_id);
      }
      iree_string_view_t expected = loom_bstring_view(keyword_bstring);
      if (!loom_tokenizer_try_consume_keyword(&parser->tokenizer, expected)) {
        loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
        return loom_parser_emit_unexpected_token(parser, peek, expected);
      }
      return iree_ok_status();
    }
  }
}

//===----------------------------------------------------------------------===//
// Op finalization — accumulator -> loom_op_t
//===----------------------------------------------------------------------===//

static void loom_parser_set_region_parent_op(loom_region_t* region,
                                             loom_op_t* parent_op) {
  if (!region) return;
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(region, block_index);
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) { op->parent_op = parent_op; }
  }
}

static loom_block_t* loom_parser_find_block_by_label(
    const loom_parser_t* parser, loom_region_t* region,
    iree_string_view_t label) {
  if (!region) return NULL;
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(region, block_index);
    if (block->label_id == LOOM_STRING_ID_INVALID ||
        block->label_id >= parser->module->strings.count) {
      continue;
    }
    if (iree_string_view_equal(parser->module->strings.entries[block->label_id],
                               label)) {
      return block;
    }
  }
  return NULL;
}

static iree_status_t loom_parser_add_pending_successor_ref(
    loom_parser_t* parser, loom_region_t* region, loom_op_t* op,
    uint8_t successor_index, loom_token_t label_token) {
  loom_parser_pending_successor_refs_t* pending =
      &parser->pending_successor_refs;
  if (pending->count >= pending->capacity) {
    iree_host_size_t capacity = pending->capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        &parser->parser_arena, pending->count, pending->count + 1,
        sizeof(loom_parser_pending_successor_ref_t), &capacity,
        (void**)&pending->entries));
    pending->capacity = capacity;
  }
  pending->entries[pending->count++] = (loom_parser_pending_successor_ref_t){
      .region = region,
      .op = op,
      .label_token = label_token,
      .successor_index = successor_index,
  };
  return iree_ok_status();
}

static iree_status_t loom_parser_resolve_pending_successor_refs(
    loom_parser_t* parser, loom_region_t* region,
    iree_host_size_t pending_start) {
  loom_parser_pending_successor_refs_t* pending =
      &parser->pending_successor_refs;
  for (iree_host_size_t i = pending_start; i < pending->count; ++i) {
    loom_parser_pending_successor_ref_t ref = pending->entries[i];
    if (ref.region != region) continue;
    loom_block_t* target =
        loom_parser_find_block_by_label(parser, region, ref.label_token.text);
    if (!target) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(ref.label_token.text),
      };
      return loom_parser_emit(
          parser, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 32), params,
          IREE_ARRAYSIZE(params), ref.label_token);
    }
    loom_op_successors(ref.op)[ref.successor_index] = target;
  }
  pending->count = pending_start;
  return iree_ok_status();
}

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
  if (parsed->successor_count > 0) {
    IREE_RETURN_IF_ERROR(loom_builder_allocate_op_with_successors(
        &parser->builder, kind, parsed->operand_count, parsed->result_count,
        parsed->successor_count, parsed->region_count,
        parsed->tied_result_count, vtable->attribute_count, location, &op));
  } else {
    IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
        &parser->builder, kind, parsed->operand_count, parsed->result_count,
        parsed->region_count, parsed->tied_result_count,
        vtable->attribute_count, location, &op));
  }

  // Copy operands.
  if (parsed->operand_count > 0) {
    memcpy(loom_op_operands(op), parsed->operand_ids,
           parsed->operand_count * sizeof(loom_value_id_t));
  }

  // Copy successors and record labels that need region-end resolution. Forward
  // references intentionally remain NULL until the enclosing region has parsed
  // every block label.
  if (parsed->successor_count > 0) {
    memcpy(loom_op_successors(op), parsed->successors,
           parsed->successor_count * sizeof(loom_block_t*));
    loom_region_t* successor_region =
        op->parent_block ? op->parent_block->parent_region : NULL;
    if (successor_region) {
      successor_region->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;
    }
    for (uint8_t i = 0; i < parsed->successor_count; ++i) {
      if (parsed->successor_label_tokens[i].kind == LOOM_TOKEN_BLOCK_LABEL) {
        IREE_RETURN_IF_ERROR(loom_parser_add_pending_successor_ref(
            parser, successor_region, op, i,
            parsed->successor_label_tokens[i]));
      }
    }
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
        IREE_RETURN_IF_ERROR(
            loom_module_set_value_type(parser->module, parsed->result_ids[i],
                                       loom_type_scalar(LOOM_SCALAR_TYPE_I1)));
      }
    }
  }

  // Copy result value IDs into the op. Body-op results become visible in the
  // current lexical scope here; symbol-definition results are signature values
  // only and must not leak into the module scope or their function body.
  // Values already have their types and names set during LHS parsing and the
  // format walk.
  loom_value_id_t* result_slots = loom_op_results(op);
  for (uint16_t i = 0; i < parsed->result_count; ++i) {
    loom_value_id_t value_id = parsed->result_ids[i];
    result_slots[i] = value_id;
    if (iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE)) continue;
    loom_string_id_t name_id = parser->module->values.entries[value_id].name_id;
    if (name_id != LOOM_STRING_ID_INVALID &&
        name_id < parser->module->strings.count) {
      bool duplicate = false;
      IREE_RETURN_IF_ERROR(loom_parser_scope_define(
          parser->scope, &parser->parser_arena, name_id, value_id, &duplicate));
      if (duplicate) {
        return loom_parser_emit_duplicate_value_name(
            parser, parsed->result_name_tokens[i]);
      }
    }
  }

  // Copy regions.
  if (parsed->region_count > 0) {
    memcpy(loom_op_regions(op), parsed->regions,
           parsed->region_count * sizeof(loom_region_t*));
    for (uint8_t i = 0; i < parsed->region_count; ++i) {
      loom_parser_set_region_parent_op(parsed->regions[i], op);
    }
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

static iree_status_t loom_parse_op_into(loom_parser_t* parser,
                                        loom_parsed_op_t* parsed) {
  uint32_t errors_before = parser->error_count;

  // Capture the start position of the op for source location tracking.
  // This is the first token — either a result name or the op name.
  loom_token_t start_token = loom_tokenizer_peek(&parser->tokenizer);
  const iree_string_view_t* comments = NULL;
  iree_host_size_t comment_count = 0;
  loom_tokenizer_take_pending_comments(&parser->tokenizer, &comments,
                                       &comment_count);

  // Parse LHS result names: %a, %b = op.name ...
  // Or just: op.name ... (for ops with no results).
  if (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_SSA_VALUE)) {
    // Collect result names until we see '='.
    for (;;) {
      if (parsed->result_count > 0) {
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
        IREE_RETURN_IF_ERROR(
            loom_module_set_value_name(parser->module, value_id, name_id));
      }
      IREE_RETURN_IF_ERROR(loom_parsed_op_add_result(
          parsed, &parser->parser_arena, value_id, name_token));
    }
    // Expect '='.
    if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_EQUALS)) {
      loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
      return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'='"));
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
    return loom_parser_emit(parser,
                            loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 6),
                            params, IREE_ARRAYSIZE(params), op_name_token);
  }

  bool is_symbol_definition =
      iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE);
  if (is_symbol_definition && parsed->result_count > 0) {
    return loom_parser_emit_result_count_mismatch(parser, vtable, op_name_token,
                                                  /*expected_count=*/0,
                                                  parsed->result_count);
  }

  // Walk the format elements.
  IREE_RETURN_IF_ERROR(
      loom_parser_walk_format(parser, vtable, op_name_token, parsed));

  if (parser->error_count > errors_before) {
    loom_parser_definition_scope_discard(parser);
  }

  // If FUNC_ARGS produced pending args but no REGION consumed them (e.g.,
  // func.decl, func.ukernel), these are declaration signature args. Store
  // their value IDs as op operands so FuncArgs printing and func-like
  // verification can recover the signature.
  if (parser->error_count == errors_before &&
      parser->pending_block_args.count > 0) {
    for (uint16_t i = 0; i < parser->pending_block_args.count; ++i) {
      uint16_t operand_index = parsed->operand_count;
      IREE_RETURN_IF_ERROR(loom_parsed_op_add_operand(
          parsed, &parser->parser_arena,
          parser->pending_block_args.entries[i].value_id));
      loom_token_t name_token =
          parser->pending_block_args.entries[i].name_token;
      IREE_RETURN_IF_ERROR(loom_parsed_op_add_field_span(
          parsed, &parser->parser_arena, LOOM_LOCATION_FIELD_OPERAND,
          operand_index, name_token, name_token.line, name_token.end_column));
    }
    loom_parser_pending_block_args_clear(&parser->pending_block_args);
  } else if (parser->error_count > 0) {
    loom_parser_pending_block_args_clear(&parser->pending_block_args);
  }

  bool has_variadic_results =
      (vtable->vtable_flags & LOOM_OP_VTABLE_VARIADIC_RESULTS) != 0;
  if (!is_symbol_definition && !has_variadic_results &&
      parser->error_count == errors_before &&
      parsed->result_count != vtable->fixed_result_count) {
    IREE_RETURN_IF_ERROR(loom_parser_emit_result_count_mismatch(
        parser, vtable, op_name_token, vtable->fixed_result_count,
        parsed->result_count));
  }

  // Build the implicit parser-source fallback first, then let an explicit
  // trailing loc(...) annotation override it when present.
  loom_location_id_t fallback_location = LOOM_LOCATION_UNKNOWN;
  if (parser->source_id != LOOM_SOURCE_ID_INVALID) {
    loom_location_entry_t entry = loom_location_file_range(
        parser->source_id, (uint16_t)start_token.line,
        (uint16_t)start_token.column,
        (uint16_t)parser->tokenizer.consumed_end_line,
        (uint16_t)parser->tokenizer.consumed_end_column);
    IREE_RETURN_IF_ERROR(
        loom_module_add_location(parser->module, entry, &fallback_location));
  }
  loom_location_id_t location = fallback_location;
  IREE_RETURN_IF_ERROR(
      loom_parse_optional_op_location(parser, location, &location));
  if (parser->error_count > errors_before) {
    return iree_ok_status();
  }
  if (location == fallback_location &&
      fallback_location != LOOM_LOCATION_UNKNOWN &&
      parsed->field_span_count > 0) {
    IREE_RETURN_IF_ERROR(loom_module_attach_location_field_spans(
        parser->module, fallback_location, parsed->field_spans,
        parsed->field_span_count));
  }

  // Finalize the op.
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_finalize_op(parser, kind, vtable, parsed, location, &op));
  return loom_module_attach_op_comments(parser->module, op, comments,
                                        comment_count);
}

static iree_status_t loom_parse_op(loom_parser_t* parser) {
  loom_parsed_op_t* parsed = NULL;
  IREE_RETURN_IF_ERROR(loom_parser_acquire_parsed_op(parser, &parsed));
  iree_status_t status = loom_parse_op_into(parser, parsed);
  loom_parser_result_scope_reset(&parser->result_scope);
  loom_parser_release_parsed_op(parser, parsed);
  return status;
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

static bool loom_parser_block_has_explicit_terminator(
    loom_parser_t* parser, const loom_block_t* block) {
  if (block->op_count == 0) return false;
  const loom_op_t* last_op = loom_block_const_last_op(block);
  const loom_op_vtable_t* last_vtable =
      loom_context_resolve_op(parser->context, last_op->kind);
  return last_vtable &&
         iree_any_bit_set(last_vtable->traits, LOOM_TRAIT_TERMINATOR);
}

static iree_status_t loom_parser_append_implicit_terminator(
    loom_parser_t* parser, const loom_region_descriptor_t* region_descriptor,
    loom_block_t* block) {
  IREE_ASSERT_ARGUMENT(region_descriptor);
  if (region_descriptor->implicit_terminator == LOOM_OP_KIND_UNKNOWN ||
      loom_parser_block_has_explicit_terminator(parser, block)) {
    return iree_ok_status();
  }

  loom_builder_set_block(&parser->builder, block);
  loom_location_id_t location = LOOM_LOCATION_UNKNOWN;
  if (parser->source_id != LOOM_SOURCE_ID_INVALID) {
    loom_token_t token = loom_tokenizer_peek(&parser->tokenizer);
    if (token.kind != LOOM_TOKEN_EOF && token.kind != LOOM_TOKEN_NONE) {
      loom_location_entry_t entry = loom_location_file_range(
          parser->source_id, (uint16_t)token.line, (uint16_t)token.column,
          (uint16_t)token.line, (uint16_t)token.end_column);
      IREE_RETURN_IF_ERROR(
          loom_module_add_location(parser->module, entry, &location));
    }
  }
  loom_op_t* terminator = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
      &parser->builder, region_descriptor->implicit_terminator,
      /*operand_count=*/0, /*result_count=*/0, /*region_count=*/0,
      /*tied_result_count=*/0, /*attribute_count=*/0, location, &terminator));
  return loom_builder_finalize_op(&parser->builder, terminator);
}

static iree_status_t loom_parse_region_body(
    loom_parser_t* parser, const loom_region_descriptor_t* region_descriptor,
    loom_region_t* region, bool* out_region_end_consumed) {
  *out_region_end_consumed = false;

  // Seed the entry block with pending block args from FUNC_ARGS, BINDING_LIST,
  // or implicit region operands. Values are already defined in the module;
  // region-local names are defined in the child scope here.
  if (parser->pending_block_args.count > 0) {
    loom_block_t* entry_block = loom_region_entry_block(region);
    for (uint16_t i = 0; i < parser->pending_block_args.count; ++i) {
      const loom_parser_pending_block_arg_t pending_arg =
          parser->pending_block_args.entries[i];
      LOOM_PARSE_DEFINE_VALUE_NAME(parser, pending_arg.name_token,
                                   pending_arg.value_id);
      IREE_RETURN_IF_ERROR(loom_block_add_arg(parser->module, entry_block,
                                              pending_arg.value_id));
    }
    loom_parser_pending_block_args_clear(&parser->pending_block_args);
  }

  // Parse blocks. The entry block may not have an explicit label.
  bool first_block = true;
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_RBRACE) &&
         !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (loom_parser_at_error_limit(parser)) break;

    loom_block_t* block = NULL;
    if (first_block) {
      block = loom_region_entry_block(region);
      first_block = false;
    } else {
      IREE_RETURN_IF_ERROR(
          loom_region_append_block(parser->module, region, &block));
    }

    // Parse optional block label: ^label(args):
    if (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_BLOCK_LABEL)) {
      const iree_string_view_t* comments = NULL;
      iree_host_size_t comment_count = 0;
      loom_tokenizer_take_pending_comments(&parser->tokenizer, &comments,
                                           &comment_count);
      loom_token_t label_token = loom_tokenizer_next(&parser->tokenizer);
      loom_string_id_t label_id = 0;
      IREE_RETURN_IF_ERROR(loom_module_intern_string(
          parser->module, label_token.text, &label_id));
      if (loom_parser_find_block_by_label(parser, region, label_token.text)) {
        loom_diagnostic_param_t params[] = {
            loom_param_string(label_token.text),
        };
        return loom_parser_emit(
            parser, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 33), params,
            IREE_ARRAYSIZE(params), label_token);
      }
      block->label_id = label_id;
      IREE_RETURN_IF_ERROR(loom_module_attach_block_comments(
          parser->module, block, comments, comment_count));

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

    const uint32_t block_errors_before = parser->error_count;
    IREE_RETURN_IF_ERROR(loom_parse_block_body(parser, block));
    if (parser->error_count == block_errors_before) {
      IREE_RETURN_IF_ERROR(loom_parser_append_implicit_terminator(
          parser, region_descriptor, block));
    }
  }

  if (first_block && !loom_parser_at_error_limit(parser)) {
    IREE_RETURN_IF_ERROR(loom_parser_append_implicit_terminator(
        parser, region_descriptor, loom_region_entry_block(region)));
  }

  loom_tokenizer_discard_pending_comments(&parser->tokenizer);
  LOOM_PARSE_EXPECT(parser, LOOM_TOKEN_RBRACE, NULL);
  *out_region_end_consumed = true;
  return iree_ok_status();
}

static iree_status_t loom_parse_braced_region(
    loom_parser_t* parser, const loom_region_descriptor_t* region_descriptor,
    loom_region_t** out_region) {
  IREE_ASSERT_ARGUMENT(region_descriptor);
  uint32_t errors_before = parser->error_count;

  // Expect '{'.
  if (!loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_LBRACE)) {
    loom_token_t peek = loom_tokenizer_peek(&parser->tokenizer);
    return loom_parser_emit_unexpected_token(parser, peek, IREE_SV("'{'"));
  }

  // Push a new lexical scope for the region. Pending FUNC_ARGS and
  // BINDING_LIST names are defined in this child scope when seeding the entry
  // block, so signature/result names from the parent op's Scope(...) do not
  // leak into the body.
  loom_parser_scope_t* outer_scope = parser->scope;
  IREE_RETURN_IF_ERROR(
      loom_parser_scope_push(parser, outer_scope, &parser->scope));

  // Allocate the region with a single entry block initially.
  loom_region_t* region = NULL;
  IREE_RETURN_IF_ERROR(loom_module_allocate_region(parser->module, 1, &region));

  // Save and set builder insertion point.
  loom_builder_ip_t saved_ip = loom_builder_save(&parser->builder);

  iree_host_size_t pending_successor_start =
      parser->pending_successor_refs.count;
  bool region_end_consumed = false;
  iree_status_t status = loom_parse_region_body(parser, region_descriptor,
                                                region, &region_end_consumed);
  if (iree_status_is_ok(status) && parser->error_count == errors_before) {
    status = loom_parser_resolve_pending_successor_refs(
        parser, region, pending_successor_start);
  }
  if (parser->error_count > errors_before && !region_end_consumed &&
      !loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    loom_parser_sync_to_brace(parser);
  }
  loom_parser_pending_block_args_clear(&parser->pending_block_args);
  if (parser->error_count > errors_before || !iree_status_is_ok(status)) {
    parser->pending_successor_refs.count = pending_successor_start;
  }

  // Restore scope and insertion point.
  loom_parser_scope_pop(parser);
  loom_builder_restore(&parser->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);
  if (parser->error_count > errors_before) return iree_ok_status();

  *out_region = region;
  return iree_ok_status();
}

iree_status_t loom_parse_region(
    loom_parser_t* parser, const loom_region_descriptor_t* region_descriptor,
    loom_region_t** out_region) {
  return loom_parse_braced_region(parser, region_descriptor, out_region);
}

iree_status_t loom_parse_region_with_syntax(
    loom_parser_t* parser, const loom_region_descriptor_t* region_descriptor,
    loom_region_syntax_t syntax, loom_region_t** out_region) {
  switch (syntax) {
    case LOOM_REGION_SYNTAX_DEFAULT: {
      return loom_parse_braced_region(parser, region_descriptor, out_region);
    }
    case LOOM_REGION_SYNTAX_TEST_DO: {
      IREE_RETURN_IF_ERROR(loom_parse_keyword(parser, LOOM_KW_DO));
      return loom_parse_braced_region(parser, region_descriptor, out_region);
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported region syntax %u", (uint32_t)syntax);
  }
}

//===----------------------------------------------------------------------===//
// Module parsing
//===----------------------------------------------------------------------===//

static iree_status_t loom_parse_module_body(loom_parser_t* parser) {
  while (!loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_EOF)) {
    if (loom_parser_at_error_limit(parser)) break;

    // Check for attribute aliases: #alias = #encoding<params>.
    if (loom_tokenizer_at(&parser->tokenizer, LOOM_TOKEN_HASH_ATTR)) {
      loom_tokenizer_discard_pending_comments(&parser->tokenizer);
      loom_token_t alias_token = loom_tokenizer_next(&parser->tokenizer);
      if (loom_tokenizer_try_consume(&parser->tokenizer, LOOM_TOKEN_EQUALS)) {
        uint32_t errors_before = parser->error_count;

        if (loom_context_lookup_encoding_vtable(parser->context,
                                                alias_token.text)) {
          loom_diagnostic_param_t params[] = {
              loom_param_string(
                  IREE_SV("alias name shadows a registered encoding family")),
          };
          IREE_RETURN_IF_ERROR(loom_parser_emit(
              parser, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 14),
              params, IREE_ARRAYSIZE(params), alias_token));
          loom_parser_sync_to_newline(parser);
          continue;
        }
        if (loom_alias_table_lookup(&parser->aliases, alias_token.text) != 0) {
          loom_diagnostic_param_t params[] = {
              loom_param_string(IREE_SV("duplicate encoding alias name")),
          };
          IREE_RETURN_IF_ERROR(loom_parser_emit(
              parser, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 14),
              params, IREE_ARRAYSIZE(params), alias_token));
          loom_parser_sync_to_newline(parser);
          continue;
        }

        loom_string_id_t alias_name_id = 0;
        IREE_RETURN_IF_ERROR(loom_module_intern_string(
            parser->module, alias_token.text, &alias_name_id));

        uint16_t encoding_id = 0;
        IREE_RETURN_IF_ERROR(
            loom_parse_static_encoding(parser, alias_name_id, &encoding_id));
        if (parser->error_count > errors_before) {
          loom_parser_sync_to_newline(parser);
          continue;
        }

        if (encoding_id != 0) {
          IREE_RETURN_IF_ERROR(
              loom_alias_table_add(&parser->aliases, &parser->parser_arena,
                                   alias_token.text, encoding_id));
        }
        continue;
      }
      // Not an alias — this is an error.
      IREE_RETURN_IF_ERROR(loom_parser_emit_token_text_error(
          parser, loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 14),
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
      .cached_location =
          {
              .source_name = filename,
              .source_id = source_id,
          },
      .diagnostic_sink =
          options ? options->diagnostic_sink : (loom_diagnostic_sink_t){0},
      .max_errors = options ? options->max_errors : 0,
      .scope = &root_scope,
      .definition_scope =
          {
              .pop_at = UINT16_MAX,
          },
  };
  if (parser.max_errors == 0) parser.max_errors = 20;
  iree_arena_initialize(block_pool, &parser.parser_arena);
  loom_tokenizer_initialize(source, filename, &parser.parser_arena,
                            &parser.tokenizer);
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &parser.builder);

  // Parse the module body.
  iree_status_t status = loom_parse_module_body(&parser);

  // Check for tokenizer scan errors.
  if (iree_status_is_ok(status)) {
    status = loom_tokenizer_consume_status(&parser.tokenizer);
  }

  // Resolve any module-level successor labels after the top-level block has
  // been fully parsed. Nested region references resolve at each region exit.
  if (iree_status_is_ok(status) && parser.error_count == 0) {
    status = loom_parser_resolve_pending_successor_refs(&parser, module->body,
                                                        /*pending_start=*/0);
  }

  // Build use-def chains in one pass.
  if (iree_status_is_ok(status) && parser.error_count == 0) {
    status = loom_module_compute_uses(module);
  }

  // Cleanup.
  loom_tokenizer_deinitialize(&parser.tokenizer);
  iree_arena_deinitialize(&parser.parser_arena);

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
