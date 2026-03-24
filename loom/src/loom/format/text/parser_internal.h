// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Internal types and declarations shared across the parser's
// implementation files (parser.c, parser_types.c, parser_format.c).
// Not a public API — do not include from outside this library.

#ifndef LOOM_FORMAT_TEXT_PARSER_INTERNAL_H_
#define LOOM_FORMAT_TEXT_PARSER_INTERNAL_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/diagnostic.h"
#include "loom/format/text/tokenizer.h"
#include "loom/ir/ir.h"
#include "loom/ir/module.h"
#include "loom/ir/symbol_map.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Name scope
//===----------------------------------------------------------------------===//

// Open-addressed hash table entry. Keys are string views into the
// source buffer (zero-copy). Empty slots have name.data == NULL.
typedef struct loom_scope_entry_t {
  iree_string_view_t name;
  loom_value_id_t value_id;
} loom_scope_entry_t;

// A name scope for SSA value resolution. Inner scopes see outer
// names via parent chain; outer scopes cannot see inner names.
typedef struct loom_parser_scope_t {
  struct loom_parser_scope_t* parent;
  loom_scope_entry_t* entries;
  iree_host_size_t capacity;  // Power of 2. 0 = not yet allocated.
  iree_host_size_t count;
} loom_parser_scope_t;

// Defines a name in this scope. Sets |*out_duplicate| to true if the
// name is already defined in this scope (not a parent). The caller
// decides whether duplication is an error.
iree_status_t loom_scope_define(loom_parser_scope_t* scope,
                                iree_arena_allocator_t* arena,
                                iree_string_view_t name,
                                loom_value_id_t value_id, bool* out_duplicate);

// Looks up a name in this scope and parent chain. Returns
// LOOM_VALUE_ID_INVALID if not found.
loom_value_id_t loom_scope_lookup(const loom_parser_scope_t* scope,
                                  iree_string_view_t name);

// Allocates a new child scope from the parser arena.
iree_status_t loom_scope_push(iree_arena_allocator_t* arena,
                              loom_parser_scope_t* parent,
                              loom_parser_scope_t** out_scope);

//===----------------------------------------------------------------------===//
// Alias table
//===----------------------------------------------------------------------===//

typedef struct loom_alias_entry_t {
  iree_string_view_t name;  // e.g., "#q6_k"
  uint16_t encoding_id;     // 1-based index into module encoding table.
} loom_alias_entry_t;

typedef struct loom_alias_table_t {
  loom_alias_entry_t* entries;
  iree_host_size_t capacity;
  iree_host_size_t count;
} loom_alias_table_t;

// Registers an encoding alias. Grows the table via the arena.
iree_status_t loom_alias_table_add(loom_alias_table_t* table,
                                   iree_arena_allocator_t* arena,
                                   iree_string_view_t name,
                                   uint16_t encoding_id);

// Returns the encoding_id for |name|, or 0 if not found.
uint16_t loom_alias_table_lookup(const loom_alias_table_t* table,
                                 iree_string_view_t name);

//===----------------------------------------------------------------------===//
// Type parse mode
//===----------------------------------------------------------------------===//

// Controls how dynamic dim names are resolved during type parsing.
typedef enum loom_type_parse_mode_e {
  // Function arg context: [%M] creates a new index value if not
  // already defined in scope. Used for function signatures.
  LOOM_TYPE_PARSE_ARG = 0,
  // Op body context: [%M] must already be defined in scope.
  LOOM_TYPE_PARSE_BODY = 1,
  // Function return context: [%M] looks up, [#N] is ordinal.
  LOOM_TYPE_PARSE_RETURN = 2,
} loom_type_parse_mode_t;

//===----------------------------------------------------------------------===//
// Parser context
//===----------------------------------------------------------------------===//

typedef struct loom_parser_t {
  loom_tokenizer_t tokenizer;
  loom_module_t* module;
  loom_context_t* context;
  iree_arena_allocator_t parser_arena;
  loom_builder_t builder;
  loom_parser_scope_t* scope;
  loom_alias_table_t aliases;
  loom_symbol_map_t symbol_lookup;
  loom_diagnostic_sink_t diagnostic_sink;
  uint32_t error_count;
  uint32_t max_errors;
  iree_string_view_t filename;
  iree_string_view_t source;
  loom_source_id_t source_id;

  // Pending block arguments from FUNC_ARGS or BINDING_LIST, consumed
  // by the next REGION format element to seed the entry block.
  // For ops with FUNC_ARGS but no REGION (func.decl), cleaned up
  // after the format walk completes.
  loom_value_id_t* pending_block_arg_ids;
  uint16_t pending_block_arg_count;
  uint16_t pending_block_arg_capacity;

  // Scope saved before FUNC_ARGS pushed its scope. Non-NULL only
  // when FUNC_ARGS has pushed a scope that hasn't been consumed by
  // a REGION yet. BINDING_LIST does not push a scope.
  loom_parser_scope_t* pre_func_arg_scope;

  // Definition scope for global definitions. When > 0, type parsing
  // uses ARG mode (creates new index values for unknown [%name] dims)
  // rather than BODY mode (requires existing names). Incremented by
  // SCOPE format elements, decremented after the last child element.
  uint8_t definition_scope_depth;

  // Format element index at which to pop the current definition scope.
  // UINT16_MAX when no scope pop is pending. Zero-init safe: the pop
  // check requires definition_scope_depth > 0, which is only true
  // after a SCOPE element has been processed.
  uint16_t scope_pop_at;
} loom_parser_t;

//===----------------------------------------------------------------------===//
// Diagnostics and error recovery
//===----------------------------------------------------------------------===//

// Emits a structured diagnostic through the parser's sink. Increments
// error_count for error-severity diagnostics.
iree_status_t loom_parser_emit(loom_parser_t* parser,
                               const loom_error_def_t* error,
                               const loom_diagnostic_param_t* params,
                               iree_host_size_t param_count,
                               loom_token_t token);

// Consumes the next token and verifies it matches |kind|. On match,
// sets |*out_token| (if non-NULL) and returns iree_ok_status(). On
// mismatch, emits ERR_PARSE_003 ("unexpected token") through the
// diagnostic sink and returns iree_ok_status() — the diagnostic IS
// the error. Callers detect the mismatch by checking error_count.
// Returns non-ok status only on infrastructure failures (pending scan
// error in the tokenizer, OOM in the diagnostic sink).
iree_status_t loom_parser_expect(loom_parser_t* parser, loom_token_kind_t kind,
                                 loom_token_t* out_token);

// Expects a token and bails out of the enclosing function on mismatch.
// On match: consumes the token, sets |out_token| (if non-NULL), continues.
// On mismatch: emits ERR_PARSE_003, returns iree_ok_status() from the
// enclosing function. On infrastructure error: propagates the error.
#define LOOM_PARSE_EXPECT(parser, kind, out_token)                           \
  do {                                                                       \
    uint32_t _expect_errors = (parser)->error_count;                         \
    IREE_RETURN_IF_ERROR(loom_parser_expect((parser), (kind), (out_token))); \
    if ((parser)->error_count > _expect_errors) return iree_ok_status();     \
  } while (0)

// Creates a synthetic token for diagnostic emission in character-level
// parsing contexts (e.g., shaped type interiors). The text view must
// point into the source buffer so byte offsets resolve correctly.
loom_token_t loom_make_synthetic_token(iree_string_view_t text);

// Returns true if the parser has exceeded its error limit.
bool loom_parser_at_error_limit(const loom_parser_t* parser);

// Advances past tokens until a newline-starting token (the start of
// the next op line) is reached. Used for op-level error recovery.
void loom_parser_sync_to_newline(loom_parser_t* parser);

// Advances past tokens until the matching '}' for a region is found.
// Handles nested brace depth.
void loom_parser_sync_to_brace(loom_parser_t* parser);

//===----------------------------------------------------------------------===//
// Op accumulator
//===----------------------------------------------------------------------===//

#define LOOM_PARSED_OP_INLINE_OPERANDS 16
#define LOOM_PARSED_OP_INLINE_RESULTS 8
#define LOOM_PARSED_OP_INLINE_ATTRS 8
#define LOOM_PARSED_OP_INLINE_REGIONS 4
#define LOOM_PARSED_OP_INLINE_TIED 4

// Accumulates parsed fields during format walk. Pointers start aimed
// at the inline arrays and redirect to arena overflow on growth.
// Fields are ordered by alignment: pointers, then counts, then inline
// storage, to eliminate padding.
typedef struct loom_parsed_op_t {
  loom_value_id_t* operand_ids;
  iree_string_view_t* result_names;
  loom_type_t* result_types;
  loom_attribute_t* attributes;
  loom_region_t** regions;
  loom_tied_result_t* tied_results;

  uint16_t operand_count;
  uint16_t operand_capacity;
  uint16_t result_count;
  uint16_t result_capacity;
  uint16_t tied_result_count;
  uint16_t tied_result_capacity;
  uint8_t attribute_count;
  uint8_t attribute_capacity;
  uint8_t region_count;
  uint8_t region_capacity;
  uint8_t instance_flags;
  uint8_t reserved_;

  loom_value_id_t inline_operand_ids[LOOM_PARSED_OP_INLINE_OPERANDS];
  iree_string_view_t inline_result_names[LOOM_PARSED_OP_INLINE_RESULTS];
  loom_type_t inline_result_types[LOOM_PARSED_OP_INLINE_RESULTS];
  loom_attribute_t inline_attributes[LOOM_PARSED_OP_INLINE_ATTRS];
  loom_region_t* inline_regions[LOOM_PARSED_OP_INLINE_REGIONS];
  loom_tied_result_t inline_tied_results[LOOM_PARSED_OP_INLINE_TIED];
} loom_parsed_op_t;

// Resets a parsed op accumulator, pointing all arrays at inline storage.
void loom_parsed_op_initialize(loom_parsed_op_t* parsed);

// Appends an operand value ID. Spills to arena on overflow.
iree_status_t loom_parsed_op_add_operand(loom_parsed_op_t* parsed,
                                         iree_arena_allocator_t* arena,
                                         loom_value_id_t value_id);

// Appends a result name and placeholder type. Spills to arena on overflow.
iree_status_t loom_parsed_op_add_result(loom_parsed_op_t* parsed,
                                        iree_arena_allocator_t* arena,
                                        iree_string_view_t name,
                                        loom_type_t type);

// Sets the attribute at |index|. Grows the attribute array if needed,
// zero-filling any gaps between attribute_count and |index|.
iree_status_t loom_parsed_op_set_attribute(loom_parsed_op_t* parsed,
                                           iree_arena_allocator_t* arena,
                                           uint8_t index,
                                           loom_attribute_t attr);

// Appends a region pointer. Spills to arena on overflow.
iree_status_t loom_parsed_op_add_region(loom_parsed_op_t* parsed,
                                        iree_arena_allocator_t* arena,
                                        loom_region_t* region);

// Appends a tied result entry. Spills to arena on overflow.
iree_status_t loom_parsed_op_add_tied_result(loom_parsed_op_t* parsed,
                                             iree_arena_allocator_t* arena,
                                             loom_tied_result_t tied);

// Appends a value ID to the parser's pending block arg list. These
// are consumed by the next REGION format element to seed the entry
// block. Arena-allocated with growth.
iree_status_t loom_parser_add_pending_block_arg(loom_parser_t* parser,
                                                loom_value_id_t value_id);

//===----------------------------------------------------------------------===//
// Type parsing (parser_types.c)
//===----------------------------------------------------------------------===//

// Parses a type from the token stream according to |mode|.
iree_status_t loom_parse_type(loom_parser_t* parser,
                              loom_type_parse_mode_t mode,
                              loom_type_t* out_type);

// Parses encoding parameters from a string slice containing
// comma-separated key=value pairs (e.g., "bits=8, type=q6_k").
// Arena-allocates the output array in the module arena. Used by
// both inline encoding in types (#name<params>) and top-level
// encoding alias definitions (#alias = #name<params>).
iree_status_t loom_parse_encoding_params(loom_parser_t* parser,
                                         iree_string_view_t params_text,
                                         loom_named_attr_t** out_attrs,
                                         uint8_t* out_count);

//===----------------------------------------------------------------------===//
// Value parsing helpers (parser.c)
//===----------------------------------------------------------------------===//

// Parses an attribute value according to the descriptor's kind
// (integer, string, enum, symbol ref, type ref, etc.).
iree_status_t loom_parse_attr_value(loom_parser_t* parser,
                                    const loom_attr_descriptor_t* descriptor,
                                    loom_attribute_t* out_attr);

// Parses a bracket-enclosed predicate list: [pred(args), ...].
iree_status_t loom_parse_predicate_list(loom_parser_t* parser,
                                        loom_attribute_t* out_attr);

// Parses a brace-enclosed attribute dictionary: {key = value, ...}.
iree_status_t loom_parse_attr_dict(loom_parser_t* parser,
                                   loom_attribute_t* out_attr);

// Returns the token kind a keyword would produce as a standalone token.
// LOOM_TOKEN_BARE_IDENT for text keywords (import, where, etc.),
// LOOM_TOKEN_COMMA for ',', LOOM_TOKEN_ARROW for '->', etc.
loom_token_kind_t loom_keyword_token_kind(uint16_t keyword_id);

// Consumes the keyword identified by |keyword_id|. Uses the appropriate
// token kind for punctuation keywords, or BARE_IDENT text match for
// word keywords.
iree_status_t loom_parse_keyword(loom_parser_t* parser, uint16_t keyword_id);

// Parses a region: '{' block* '}'.
iree_status_t loom_parse_region(loom_parser_t* parser,
                                loom_region_t** out_region);

//===----------------------------------------------------------------------===//
// Format walker (parser_format.c)
//===----------------------------------------------------------------------===//

// Walks a vtable's format elements, parsing each according to its kind.
// Populates |parsed| with operands, results, attributes, and regions.
iree_status_t loom_parser_walk_format(loom_parser_t* parser,
                                      const loom_op_vtable_t* vtable,
                                      loom_parsed_op_t* parsed);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_FORMAT_TEXT_PARSER_INTERNAL_H_
