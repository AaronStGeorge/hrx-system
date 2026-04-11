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

typedef struct loom_parser_t loom_parser_t;
typedef struct loom_parsed_op_t loom_parsed_op_t;
typedef struct loom_parser_encoding_params_t loom_parser_encoding_params_t;
typedef struct loom_parser_scope_t loom_parser_scope_t;
typedef struct loom_parser_type_list_t loom_parser_type_list_t;

// Pending block argument prepared by FUNC_ARGS, BINDING_LIST, or an implicit
// region operand such as a loop IV.
typedef struct loom_parser_pending_block_arg_t {
  loom_value_id_t value_id;
  // Name to define in the child region scope when REGION consumes this arg.
  // Already-scoped function arguments use loom_token_none().
  loom_token_t name_token;
} loom_parser_pending_block_arg_t;

// Growable scratch list of values that should become entry-block arguments for
// the next parsed REGION element.
typedef struct loom_parser_pending_block_args_t {
  loom_parser_pending_block_arg_t* entries;
  uint16_t count;
  uint16_t capacity;
} loom_parser_pending_block_args_t;

// Placeholder SSA value created by ARG-mode type parsing inside the current
// Scope(...). |resolved| tracks whether a later declaration in the same scope
// bound that placeholder name, and |name_token| is retained so scope-exit
// diagnostics can point at the original forward reference without storing
// source locations on every lexical scope entry.
typedef struct loom_parser_unresolved_placeholder_t {
  loom_value_id_t value_id;
  loom_token_t name_token;
  bool resolved;
} loom_parser_unresolved_placeholder_t;

// Growable scratch list of signature/global placeholders created by ARG-mode
// type parsing. Entries are appended as placeholders are created and truncated
// when the one active Scope(...) exits.
typedef struct loom_parser_unresolved_placeholders_t {
  loom_parser_unresolved_placeholder_t* entries;
  iree_host_size_t count;
  iree_host_size_t capacity;
} loom_parser_unresolved_placeholders_t;

enum loom_parser_definition_scope_flag_bits_e {
  // ARG-mode placeholders in this Scope(...) are declarations as soon as type
  // parsing infers an index/encoding binding type from first use. Global
  // definitions need this because `%dim`/`%enc` names appear only inside the
  // result type and optional predicates, not as later `%name: type` binders.
  LOOM_PARSER_DEFINITION_SCOPE_FLAG_RESOLVE_PLACEHOLDERS_FROM_USE = 1u << 0,
};
typedef uint8_t loom_parser_definition_scope_flags_t;

// Tracks the one active Scope(...) declaration scope in an op format. Nested
// Scope(...) is intentionally unsupported by the DSL, so the C parser only
// needs one placeholder pop index and one format-element pop point.
//
// This metadata intentionally sits next to, not inside, the lexical scope
// chain: `parser->scope` owns name visibility, while `definition_scope` owns
// the lifetime and resolution policy for ARG-mode placeholders created by type
// parsing in the current declaration wrapper.
typedef struct loom_parser_definition_scope_t {
  iree_host_size_t placeholder_start;
  uint16_t pop_at;  // UINT16_MAX when no Scope(...) is active.
  loom_parser_definition_scope_flags_t flags;
} loom_parser_definition_scope_t;

//===----------------------------------------------------------------------===//
// Name scope
//===----------------------------------------------------------------------===//

// Open-addressed hash table entry keyed by interned string ID.
// Empty slots have name_id == LOOM_STRING_ID_INVALID.
typedef struct loom_parser_scope_entry_t {
  loom_string_id_t name_id;
  loom_value_id_t value_id;
} loom_parser_scope_entry_t;

// A name scope for SSA value resolution. Inner scopes see outer
// names via parent chain; outer scopes cannot see inner names.
typedef struct loom_parser_scope_t {
  // Parser-owned scopes returned to parser->scope_free_list after pop.
  struct loom_parser_scope_t* next_free;
  struct loom_parser_scope_t* parent;
  loom_parser_scope_entry_t* entries;
  iree_host_size_t capacity;  // Power of 2. 0 = not yet allocated.
  iree_host_size_t count;
} loom_parser_scope_t;

// Defines a name in this scope. Sets |*out_duplicate| to true if the
// name is already defined in this scope (not a parent). The caller
// decides whether duplication is an error.
iree_status_t loom_parser_scope_define(loom_parser_scope_t* scope,
                                       iree_arena_allocator_t* arena,
                                       loom_string_id_t name_id,
                                       loom_value_id_t value_id,
                                       bool* out_duplicate);

// Looks up a name in this scope and parent chain. Returns
// LOOM_VALUE_ID_INVALID if not found.
loom_value_id_t loom_parser_scope_lookup(const loom_parser_scope_t* scope,
                                         loom_string_id_t name_id);

// Looks up a name in only this scope (not parents). Returns
// LOOM_VALUE_ID_INVALID if not found.
loom_value_id_t loom_parser_scope_lookup_local(const loom_parser_scope_t* scope,
                                               loom_string_id_t name_id);

// Pushes a parser-owned child scope with |parent| as its lexical parent. Reused
// child scopes retain their hash-table allocation and are reset to empty
// sentinels on acquire. The root module scope is stack-owned and is not managed
// by this helper.
iree_status_t loom_parser_scope_push(loom_parser_t* parser,
                                     loom_parser_scope_t* parent,
                                     loom_parser_scope_t** out_scope);

// Pops |parser->scope| to its parent and returns the released child frame to
// parser->scope_free_list with its retained hash-table storage intact.
void loom_parser_scope_pop(loom_parser_t* parser);

//===----------------------------------------------------------------------===//
// Result scope
//===----------------------------------------------------------------------===//

#define LOOM_PARSER_RESULT_SCOPE_INLINE_ENTRIES 16

// Parser-owned scratch overlay for op result names while parsing result type
// annotations. Small result sets use inline linear scan; larger sets spill to
// a retained hash table with a per-op active capacity.
typedef struct loom_parser_result_scope_t {
  loom_parser_scope_entry_t
      inline_entries[LOOM_PARSER_RESULT_SCOPE_INLINE_ENTRIES];
  loom_parser_scope_entry_t* hash_entries;
  iree_host_size_t hash_capacity;          // Active hash slots for this op.
  iree_host_size_t hash_storage_capacity;  // Retained backing allocation.
  uint16_t count;
} loom_parser_result_scope_t;

// Reserves result-scope storage for |entry_count| named results and clears only
// the active spill hash slots needed by this op.
iree_status_t loom_parser_result_scope_prepare(
    loom_parser_result_scope_t* scope, iree_host_size_t entry_count,
    iree_arena_allocator_t* arena);

// Drops the active result overlay. Retained spill storage is left untouched.
void loom_parser_result_scope_reset(loom_parser_result_scope_t* scope);

// Defines a result name in the active result overlay. Sets |*out_duplicate| if
// the name is already present in this overlay.
iree_status_t loom_parser_result_scope_define(loom_parser_result_scope_t* scope,
                                              loom_string_id_t name_id,
                                              loom_value_id_t value_id,
                                              bool* out_duplicate);

// Resolves a value ID from the active result overlay, then the lexical scope
// chain. Returns LOOM_VALUE_ID_INVALID if not found.
loom_value_id_t loom_parser_lookup_value(const loom_parser_t* parser,
                                         loom_string_id_t name_id);

// Interns |name_token.text|, stores the resulting name_id on |value_id|, and
// defines the value in the current scope. Emits ERR_PARSE_002 from
// |name_token| if that name is already present in the current scope. No-op for
// empty token text.
iree_status_t loom_parser_define_value_name(loom_parser_t* parser,
                                            loom_token_t name_token,
                                            loom_value_id_t value_id);

// Defines a typed SSA value in the current scope and returns its value ID.
// If |name_token| names an unresolved placeholder in the one active
// declaration Scope(...), that existing placeholder value is rebound to |type|
// and returned instead of allocating a second SSA value. If the placeholder
// already carries an incompatible inferred type from an earlier ARG-mode use,
// emits a parser diagnostic from |name_token|.
iree_status_t loom_parser_define_value(loom_parser_t* parser,
                                       loom_token_t name_token,
                                       loom_type_t type,
                                       loom_value_id_t* out_value_id);

// Emits ERR_PARSE_002 for a duplicate SSA value or block argument name.
iree_status_t loom_parser_emit_duplicate_value_name(loom_parser_t* parser,
                                                    loom_token_t name_token);

// Resolves an SSA name token against the current scope chain. On success,
// sets |*out_value_id| to a valid value ID. When the name is not in scope,
// emits ERR_PARSE_001 from |name_token|'s original source location and
// returns iree_ok_status(); callers should detect that via error_count.
iree_status_t loom_parser_resolve_value(loom_parser_t* parser,
                                        loom_token_t name_token,
                                        loom_value_id_t* out_value_id);

// Enters a one-level declaration scope that should pop after format element
// `pop_at`. `flags` controls how ARG-mode placeholders created in this
// declaration wrapper become resolved. Returns a non-OK status if the op format
// attempts nested Scope(...), which is rejected by the Python DSL and
// unsupported in C.
iree_status_t loom_parser_definition_scope_push(
    loom_parser_t* parser, uint16_t pop_at,
    loom_parser_definition_scope_flags_t flags);

// Pops the active declaration scope when |format_index| reaches pop_at and
// emits an unresolved-placeholder diagnostic for any Scope-local placeholder
// values that were never bound by a concrete declaration.
iree_status_t loom_parser_definition_scope_pop_if_needed(loom_parser_t* parser,
                                                         uint16_t format_index);

// Discards the active declaration scope without diagnostics. Used when a parse
// error inside Scope(...) triggers op-level recovery before normal scope exit.
void loom_parser_definition_scope_discard(loom_parser_t* parser);

// Tracks a newly created ARG-mode placeholder so Scope(...) exit can diagnose
// unresolved forward references precisely.
iree_status_t loom_parser_add_unresolved_placeholder(loom_parser_t* parser,
                                                     loom_value_id_t value_id,
                                                     loom_token_t name_token);

//===----------------------------------------------------------------------===//
// Alias table
//===----------------------------------------------------------------------===//

typedef struct loom_alias_entry_t {
  // Bare alias name without '#', e.g. "q6_k".
  iree_string_view_t name;
  // 1-based index into the module encoding table.
  uint16_t encoding_id;
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

  // Parser-owned reusable child lexical scope frames. The root module scope is
  // stack-owned in loom_text_parse and is not part of this free list.
  loom_parser_scope_t* scope_free_list;

  // Reusable result-name overlay while parsing result type annotations. This is
  // intentionally separate from the lexical scope representation: the hot path
  // for 0-16 result names is a tiny inline linear scan, while larger outliers
  // use a retained spill hash table sized to the current op.
  loom_parser_result_scope_t result_scope;

  // Parser-owned reusable op scratch frames. One active frame is leased per
  // recursive loom_parse_op call, so child-region parsing cannot clobber the
  // parent op's accumulator. Released frames retain their spill buffers.
  loom_parsed_op_t* parsed_op_free_list;

  // Parser-owned reusable encoding-parameter scratch frames. Inline encoding
  // specs can nest recursively in attribute values, so each active
  // loom_parse_static_encoding call leases its own frame.
  loom_parser_encoding_params_t* encoding_params_free_list;

  // Parser-owned reusable type-list scratch frames for first-class function
  // signatures and dialect type parameters. Nested composite type parses lease
  // independent frames, and released frames retain their FAM capacity.
  loom_parser_type_list_t* type_list_free_list;

  loom_alias_table_t aliases;
  loom_symbol_map_t symbol_lookup;
  loom_diagnostic_sink_t diagnostic_sink;
  uint32_t error_count;
  uint32_t max_errors;
  iree_string_view_t filename;
  iree_string_view_t source;
  loom_source_id_t source_id;

  // One-entry lookaside for repeated loc("same_file":...) annotations. Most
  // explicit locations in a generated .loom file reference one source, so this
  // avoids rescanning the context source table on every op while keeping cache
  // invalidation trivial.
  struct {
    iree_string_view_t source_name;
    loom_source_id_t source_id;
  } cached_location;

  // Pending block arguments from FUNC_ARGS, BINDING_LIST, or implicit region
  // operands such as loop IVs. The next REGION format element consumes them to
  // seed the entry block and define child-scope-only names for region-local
  // args. For ops with FUNC_ARGS but no REGION (func.decl), loom_parse_op
  // drains their value IDs into regular operands after the format walk.
  loom_parser_pending_block_args_t pending_block_args;

  // Placeholder values created by ARG-mode type parsing inside the one active
  // Scope(...) declaration wrapper.
  loom_parser_unresolved_placeholders_t unresolved_placeholders;

  // One active Scope(...) declaration wrapper, used for func/global signatures.
  // Region/block lexical scopes still nest through |scope|'s parent chain.
  loom_parser_definition_scope_t definition_scope;
} loom_parser_t;

// Returns true while parsing the body of the one active Scope(...) declaration
// wrapper.
static inline bool loom_parser_in_definition_scope(
    const loom_parser_t* parser) {
  return parser->definition_scope.pop_at != UINT16_MAX;
}

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

// Emits a structured diagnostic from |token| with one labeled related location
// anchored at |related_token|.
iree_status_t loom_parser_emit_related(loom_parser_t* parser,
                                       const loom_error_def_t* error,
                                       const loom_diagnostic_param_t* params,
                                       iree_host_size_t param_count,
                                       loom_token_t token,
                                       iree_string_view_t related_label,
                                       loom_token_t related_token);

// Emits ERR_PARSE_003 with |actual_token| rendered from |token.kind| and
// |token.text|, not by slicing the source spelling back out of the file.
iree_status_t loom_parser_emit_unexpected_token(loom_parser_t* parser,
                                                loom_token_t token,
                                                iree_string_view_t expected);

// Emits a one-string parser diagnostic whose token text should be rendered from
// |token.kind| and |token.text|. Intended for syntax errors that want the
// concrete token spelling as a parameter but do not need token-specific error
// definitions yet.
iree_status_t loom_parser_emit_token_text_error(loom_parser_t* parser,
                                                const loom_error_def_t* error,
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

// Resolves an SSA value token and bails out of the enclosing function when
// resolution emits a parser diagnostic. On success: sets |out_value_id| to a
// valid value ID and continues. On undefined name: emits ERR_PARSE_001 and
// returns iree_ok_status() from the enclosing function. On infrastructure
// error: propagates the error.
#define LOOM_PARSE_RESOLVE_VALUE(parser, name_token, out_value_id)          \
  do {                                                                      \
    uint32_t _resolve_errors = (parser)->error_count;                       \
    IREE_RETURN_IF_ERROR(                                                   \
        loom_parser_resolve_value((parser), (name_token), (out_value_id))); \
    if ((parser)->error_count > _resolve_errors) return iree_ok_status();   \
  } while (0)

// Defines an SSA value name in the current scope and bails out of the
// enclosing function when a duplicate-name diagnostic is emitted. On success:
// stores |name_token|'s interned string ID on |value_id|, makes the value
// visible in the current scope, and continues. On duplicate: emits
// ERR_PARSE_002 and returns iree_ok_status() from the enclosing function. On
// infrastructure error: propagates the error.
#define LOOM_PARSE_DEFINE_VALUE_NAME(parser, name_token, value_id)          \
  do {                                                                      \
    uint32_t _define_errors = (parser)->error_count;                        \
    IREE_RETURN_IF_ERROR(                                                   \
        loom_parser_define_value_name((parser), (name_token), (value_id))); \
    if ((parser)->error_count > _define_errors) return iree_ok_status();    \
  } while (0)

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
#define LOOM_PARSED_OP_INLINE_FIELD_SPANS 16

// Accumulates parsed fields during format walk. Pointers start aimed at the
// inline arrays and redirect to parser_arena spill storage on growth. Parser-
// owned frames retain those spill pointers/capacities across logical resets so
// repeated large sibling ops reuse high-water buffers.
//
// Fields are ordered by alignment: pointers, then counts, then inline storage,
// to eliminate padding.
struct loom_parsed_op_t {
  loom_parsed_op_t* next_free;

  loom_value_id_t* operand_ids;
  loom_value_id_t* result_ids;
  loom_token_t* result_name_tokens;
  loom_attribute_t* attributes;
  loom_region_t** regions;
  loom_tied_result_t* tied_results;
  loom_location_field_span_t* field_spans;

  uint16_t operand_count;
  uint16_t operand_capacity;
  uint16_t result_count;
  uint16_t result_capacity;
  uint16_t tied_result_count;
  uint16_t tied_result_capacity;
  uint16_t field_span_count;
  uint16_t field_span_capacity;
  uint8_t attribute_count;
  uint8_t attribute_capacity;
  uint8_t region_count;
  uint8_t region_capacity;
  uint8_t instance_flags;
  uint8_t reserved_;

  loom_value_id_t inline_operand_ids[LOOM_PARSED_OP_INLINE_OPERANDS];
  loom_value_id_t inline_result_ids[LOOM_PARSED_OP_INLINE_RESULTS];
  loom_token_t inline_result_name_tokens[LOOM_PARSED_OP_INLINE_RESULTS];
  loom_attribute_t inline_attributes[LOOM_PARSED_OP_INLINE_ATTRS];
  loom_region_t* inline_regions[LOOM_PARSED_OP_INLINE_REGIONS];
  loom_tied_result_t inline_tied_results[LOOM_PARSED_OP_INLINE_TIED];
  loom_location_field_span_t
      inline_field_spans[LOOM_PARSED_OP_INLINE_FIELD_SPANS];
};

// Initializes a fresh parser-owned parsed-op scratch frame.
void loom_parsed_op_initialize(loom_parsed_op_t* parsed);

// Clears a parsed-op scratch frame for reuse while retaining spill
// pointers/capacities.
void loom_parsed_op_reset(loom_parsed_op_t* parsed);

// Acquires one parser-owned parsed-op scratch frame for the current
// loom_parse_op invocation.
iree_status_t loom_parser_acquire_parsed_op(loom_parser_t* parser,
                                            loom_parsed_op_t** out_parsed);

// Releases a parsed-op scratch frame back to parser->parsed_op_free_list while
// retaining spill pointers/capacities for later reuse.
void loom_parser_release_parsed_op(loom_parser_t* parser,
                                   loom_parsed_op_t* parsed);

// Appends an operand value ID. Spills to arena on overflow.
iree_status_t loom_parsed_op_add_operand(loom_parsed_op_t* parsed,
                                         iree_arena_allocator_t* arena,
                                         loom_value_id_t value_id);

// Sets a parsed operand by storage index, growing the parsed operand list and
// filling any intervening slots with LOOM_VALUE_ID_INVALID. This lets assembly
// formats mention fields in a different order than the op storage layout.
iree_status_t loom_parsed_op_set_operand(loom_parsed_op_t* parsed,
                                         iree_arena_allocator_t* arena,
                                         uint16_t index,
                                         loom_value_id_t value_id);

// Appends a result value ID and its defining token. Parser-synthesized results
// without a user-authored LHS name should pass loom_token_none(). Spills to
// arena on overflow.
iree_status_t loom_parsed_op_add_result(loom_parsed_op_t* parsed,
                                        iree_arena_allocator_t* arena,
                                        loom_value_id_t value_id,
                                        loom_token_t name_token);

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

// Appends one source field span to the parsed-op sidecar. |start_token| is the
// token where the field's source spelling began; |end_line|/|end_column| come
// from the tokenizer's consumed-end cursor after the field parser finished.
// No-op for parser-synthesized fields with LOOM_TOKEN_NONE.
iree_status_t loom_parsed_op_add_field_span(
    loom_parsed_op_t* parsed, iree_arena_allocator_t* arena,
    loom_location_field_kind_t kind, uint16_t index, loom_token_t start_token,
    uint32_t end_line, uint32_t end_column);

// Appends a block arg to the parser's pending block arg list. These entries are
// consumed by the next REGION format element to seed the entry block.
// Arena-allocated with growth.
iree_status_t loom_parser_add_pending_block_arg(loom_parser_t* parser,
                                                loom_value_id_t value_id,
                                                loom_token_t name_token);

//===----------------------------------------------------------------------===//
// Type parsing
//===----------------------------------------------------------------------===//

// Parses a type from the token stream according to |mode|.
iree_status_t loom_parse_type(loom_parser_t* parser,
                              loom_type_parse_mode_t mode,
                              loom_type_t* out_type);

// Reusable parser-owned FAM scratch list for one active composite type parse.
// Released frames retain their element capacity so repeated wide sibling
// signatures only allocate when a new high-water mark is reached.
#define LOOM_PARSER_TYPE_LIST_MIN_CAPACITY 8
struct loom_parser_type_list_t {
  loom_parser_type_list_t* next_free;
  iree_host_size_t count;
  iree_host_size_t capacity;
  loom_type_t types[];
};

// Parses a static encoding reference from a HASH_ATTR token.
//
// Handles `#alias` lookup through parser->aliases and inline
// `#family<param = value, ...>` definitions. When `alias_id` is a valid module
// string ID, the parsed canonical encoding prefers that spelling when it does
// not already have one. Returns the 1-based module encoding ID in
// `*out_encoding_id`.
iree_status_t loom_parse_static_encoding(loom_parser_t* parser,
                                         loom_string_id_t alias_id,
                                         uint16_t* out_encoding_id);

// Reusable parser-owned accumulator for one active inline encoding parameter
// list. Nested static encoding parses lease independent frames from
// parser->encoding_params_free_list.
#define LOOM_PARSER_ENCODING_PARAMS_INLINE_ATTRS 8
struct loom_parser_encoding_params_t {
  loom_parser_encoding_params_t* next_free;
  loom_named_attr_t* attrs;
  iree_host_size_t capacity;
  uint8_t count;

  loom_named_attr_t inline_attrs[LOOM_PARSER_ENCODING_PARAMS_INLINE_ATTRS];
};

//===----------------------------------------------------------------------===//
// Value parsing helpers
//===----------------------------------------------------------------------===//

// Parses an attribute value according to the descriptor's kind
// (integer, string, enum, symbol ref, type ref, etc.).
iree_status_t loom_parse_attr_value(loom_parser_t* parser,
                                    const loom_attr_descriptor_t* descriptor,
                                    loom_attribute_t* out_attr);

// Parses a generic attribute value with the same grammar used by AttrDict and
// encoding parameter lists.
iree_status_t loom_parse_generic_attr_value(loom_parser_t* parser,
                                            uint16_t nesting_depth,
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
iree_status_t loom_parse_region(
    loom_parser_t* parser, const loom_region_descriptor_t* region_descriptor,
    loom_region_t** out_region);

// Emits ERR_PARSE_009 for a result arity mismatch on |vtable| at
// |op_name_token|.
iree_status_t loom_parser_emit_result_count_mismatch(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    loom_token_t op_name_token, uint16_t expected_count, uint16_t actual_count);

//===----------------------------------------------------------------------===//
// Format walker
//===----------------------------------------------------------------------===//

// Walks a vtable's format elements, parsing each according to its kind.
// Populates |parsed| with operands, results, attributes, and regions.
iree_status_t loom_parser_walk_format(loom_parser_t* parser,
                                      const loom_op_vtable_t* vtable,
                                      loom_token_t op_name_token,
                                      loom_parsed_op_t* parsed);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_FORMAT_TEXT_PARSER_INTERNAL_H_
