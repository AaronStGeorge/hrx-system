// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Loom IR graph: Module -> Function -> Block -> Operation -> Value.
//
// The IR is a lightweight, arena-allocated graph that represents loom
// programs. All IR for a module lives in a single arena: creation is
// bump-pointer allocation, destruction is freeing the arena.
//
// ==========================================================================
// Core concepts
// ==========================================================================
//
// Module:     Top-level container. Owns the arena, string table, type
//             table, value table, symbol table. One per file/compilation.
//
// Symbol:     A named module-level entity: function, global, executable.
//             Flat symbol table per module (no nesting). Symbols carry
//             use lists for efficient "find all references" queries.
//
// Function:   A callable symbol with a calling convention, signature,
//             and optionally a body (list of blocks). Functions are the
//             unit of parallel compilation.
//
// Block:      A basic block within a function. Contains a linear
//             sequence of operations. May have block arguments (for
//             function entry, loop carried values, branch targets).
//
// Operation:  A single IR node: an op kind, operands (value IDs),
//             results (value IDs), attributes, regions, and optional
//             tied result bindings.
//
// Value:      An SSA value (op result or block argument). 64-byte
//             cache-line-aligned entry in the module's value table.
//             Carries the type inline (24 bytes) and up to 3 uses
//             inline (no pointer chase for the common case).
//
// Region:     A list of blocks. Used for function bodies, loop bodies,
//             conditional branches. Regions nest (a region's block
//             contains ops that may have their own regions).
//
// ==========================================================================
// ID-based references
// ==========================================================================
//
// All cross-references use integer IDs, not pointers:
//   value_id   -> index into module->values.entries[]
//   op_id      -> index into block->ops[] (block-local)
//   symbol_id  -> index into module->symbols.entries[]
//   string_id  -> index into module->strings.entries[]
//   type_id    -> index into module->types.entries[] (for interned types)
//
// Benefits: stable across serialization, no dangling pointers when
// ops are erased, compact storage, cache-friendly iteration.
//
// ==========================================================================
// Symbol references
// ==========================================================================
//
// A symbol reference is (module_id, symbol_id): 4 bytes total.
// module_id indexes into the context's module list.
// symbol_id indexes into that module's symbol table.
// For local references, module_id is the current module.
//
//   Resolution: context->modules.entries[ref.module_id]
//                 ->symbols.entries[ref.symbol_id]
//
// ==========================================================================
// Tied results (ownership transfer)
// ==========================================================================
//
// A tied result reuses an operand's storage (linear ownership):
//   -> %operand        Same type, operand consumed.
//   -> %operand as T   Type change, operand consumed.
//   -> tensor<...>     Fresh allocation (not tied).
//
// Tied results are stored per-operation as a mapping from result
// index to the tied operand index. The operand is consumed: no uses
// after the consuming op (verified statically).
//
// Well-formed tied metadata has three invariants:
//   - Each result index appears in at most one tied-result entry.
//   - Each operand index appears in at most one tied-result entry.
//   - A tied operand's value appears in exactly one operand slot on
//     the operation, so name-based surface syntax cannot be ambiguous.

#ifndef LOOM_IR_IR_H_
#define LOOM_IR_IR_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/emitter.h"
#include "loom/ir/attribute.h"
#include "loom/ir/types.h"
#include "loom/util/bstring.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_context_t loom_context_t;
typedef struct loom_module_t loom_module_t;
typedef struct loom_symbol_t loom_symbol_t;
typedef struct loom_block_t loom_block_t;
typedef struct loom_op_t loom_op_t;
typedef struct loom_region_t loom_region_t;
typedef struct loom_value_t loom_value_t;
typedef struct loom_encoding_t loom_encoding_t;
typedef struct loom_encoding_vtable_t loom_encoding_vtable_t;
typedef struct loom_op_vtable_t loom_op_vtable_t;
typedef uint8_t loom_symbol_kind_t;
typedef struct loom_operand_descriptor_t loom_operand_descriptor_t;
typedef struct loom_result_descriptor_t loom_result_descriptor_t;
typedef struct loom_attr_descriptor_t loom_attr_descriptor_t;
typedef struct loom_region_descriptor_t loom_region_descriptor_t;
typedef struct loom_format_element_t loom_format_element_t;
typedef struct loom_constraint_t loom_constraint_t;
typedef struct loom_rewriter_t loom_rewriter_t;

//===----------------------------------------------------------------------===//
// References
//===----------------------------------------------------------------------===//

// loom_value_id_t and loom_string_id_t are defined in types.h.

// Packed reference to an op field: 2-bit category (operand, result,
// attr, region) in the top bits, 6-bit index in the bottom bits.
// Used by the constraint system, printer field callbacks, and
// diagnostic highlighting.
typedef uint8_t loom_field_ref_t;

//===----------------------------------------------------------------------===//
// Source locations
//===----------------------------------------------------------------------===//
//
// Every IR operation carries a location for diagnostics, source mapping,
// and debug info generation. Locations are interned per module and
// referenced by a 4-byte loom_location_id_t on each operation.
//
// Three kinds of location:
//
//   File:    Source range from a .loom file. Stores file:start:end so
//            tooling can extract exact text spans for diagnostics,
//            diffs, and agent-driven code modification.
//   Fused:   Derived from multiple source locations when a pass creates
//            an op from several inputs (inlining, fusion, rewrites).
//   Opaque:  External system identifier for JIT (torch node ID, JAX
//            trace position). Round-trips through compilation.
//
// File locations are 16 bytes with full range (start line:col to end
// line:col). This enables precise caret diagnostics with underlines:
//
//   error: element types do not match
//     --> model.loom:42:3
//      |
//   42 |   %r = tile.contract %w, %x : tile<4x4xf32>, tile<4xi8>
//      |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//
// Locations can be stripped for release/embedded builds. A stripped
// location preserves the interned ID (for bytecode stability) but
// carries no data. Fusing stripped locations produces stripped.
//
// Text format:
//   loc("model.loom":42:3 to 42:58)
//   loc(fused<"model.loom":42:3, "recipe.loom":15:1>)
//   loc(opaque<"torch", "node_id=42">)

// Index into the module's location table.
// ID 0 is always LOOM_LOCATION_NONE (unknown/absent).
// When locations are stripped, all ops point to ID 0 and the
// location table is empty. No per-entry stripped flag needed.
typedef uint32_t loom_location_id_t;
#define LOOM_LOCATION_UNKNOWN ((loom_location_id_t)0)

// Index into the context's source table.
// The source table is a small, context-level table of unique source
// identifiers: filenames ("model.loom", "recipe_x86.loom"), external
// system tags ("torch", "jax"), or any other provenance label.
// Shared across modules because the same source can be referenced
// by multiple modules. Separate from the per-module string table.
typedef uint16_t loom_source_id_t;
#define LOOM_SOURCE_ID_INVALID ((loom_source_id_t)UINT16_MAX)

// Source table stored on the context. Entries are interned copies
// of source names (filenames, system tags). The table is append-only
// and deduplicates by exact string match.
typedef struct loom_source_table_t {
  iree_host_size_t count;
  iree_host_size_t capacity;
  iree_string_view_t* entries;
} loom_source_table_t;

// Location kind.
typedef enum loom_location_kind_e {
  // No location information. Synthetic or stripped.
  LOOM_LOCATION_NONE = 0,
  // Source range from a .loom file: file + start + end positions.
  LOOM_LOCATION_FILE = 1,
  // Derived from multiple source locations (inlining, fusion).
  // Children are arena-allocated location IDs.
  LOOM_LOCATION_FUSED = 2,
  // External system identifier (torch node ID, JAX trace).
  // Tag + opaque data blob, round-tripped verbatim.
  LOOM_LOCATION_OPAQUE = 3,
  LOOM_LOCATION_COUNT_,
} loom_location_kind_t;

// Location flags.
enum loom_location_flag_bits_e {
  // Compiler-generated op, not from user source.
  LOOM_LOCATION_FLAG_SYNTHETIC = 1u << 0,
};
typedef uint8_t loom_location_flags_t;

// A source location entry. 16 bytes. Tagged union.
//
// The kind field (byte 0) determines which union variant is active.
// All three variants fit in exactly 16 bytes with no wasted space.
//
// File locations (the 90% case) use uint16_t for line numbers,
// supporting up to 65K lines per file. Agent-authored .loom files
// are typically hundreds of lines; linked module dumps are diagnostic
// output, not round-tripped source.
//
// The source_id field indexes into the context's source table, which
// stores filenames, system tags, and any other provenance labels.
// This keeps location entries small (16-bit ID) while supporting
// arbitrary source identifiers.
//
// Text format:
//   loc("model.loom":42:3 to 42:58)
//   loc(fused<"model.loom":42:3, "recipe.loom":15:1>)
//   loc(opaque<"torch", "node_id=42">)
typedef struct loom_location_entry_t {
  loom_location_kind_t kind;
  loom_location_flags_t flags;
  union {
    // LOOM_LOCATION_FILE: source range.
    struct {
      loom_source_id_t source_id;
      uint16_t start_line;
      uint16_t start_col;
      uint16_t end_line;
      uint16_t end_col;
    } file;

    // LOOM_LOCATION_FUSED: multiple source locations.
    struct {
      uint32_t count;
      loom_location_id_t* children;
    } fused;

    // LOOM_LOCATION_OPAQUE: external identifier.
    struct {
      loom_source_id_t source_id;
      uint32_t data_length;
      const uint8_t* data;
    } opaque;
  };
} loom_location_entry_t;

static_assert(sizeof(loom_location_entry_t) == 24,
              "loom_location_entry_t must be 24 bytes");

// Location table stored on the module.
// Entry 0 is always LOOM_LOCATION_NONE.
// When locations are stripped, the table is empty and all ops
// reference LOOM_LOCATION_UNKNOWN (ID 0).
typedef struct loom_location_table_t {
  iree_host_size_t count;
  iree_host_size_t capacity;
  loom_location_entry_t* entries;
} loom_location_table_t;

//===----------------------------------------------------------------------===//
// Location accessors
//===----------------------------------------------------------------------===//

static inline loom_location_kind_t loom_location_get_kind(
    loom_location_entry_t entry) {
  return (loom_location_kind_t)entry.kind;
}

// Construct a file source range location.
static inline loom_location_entry_t loom_location_file_range(
    loom_source_id_t source_id, uint16_t start_line, uint16_t start_col,
    uint16_t end_line, uint16_t end_col) {
  loom_location_entry_t entry = {.kind = LOOM_LOCATION_FILE};
  entry.file.source_id = source_id;
  entry.file.start_line = start_line;
  entry.file.start_col = start_col;
  entry.file.end_line = end_line;
  entry.file.end_col = end_col;
  return entry;
}

//===----------------------------------------------------------------------===//
// Use-def tracking
//===----------------------------------------------------------------------===//
//
// A use entry records "op X uses this value as operand Y." 8 bytes on
// all platforms, stored inline in the value (up to 3) or in an arena-
// allocated overflow array.
//
// On 64-bit: the op pointer and operand index are packed into a single
// uint64_t. Userspace heap pointers on x86-64 and AArch64 use at most
// 48 bits; the upper 16 bits carry the operand index. This gives direct
// pointer access with no table indirection.
//
// On 32-bit (WASM): the natural struct layout is 8 bytes because
// pointers are 4 bytes. No packing needed.
//
// Both representations fit 3 uses inline in the value's 24-byte union,
// preserving the 64-byte cache-line-aligned value layout.

#if defined(IREE_PTR_SIZE_64)

typedef uint64_t loom_use_t;

// Mask for extracting the 48-bit pointer from a tagged use entry.
// Validated at creation time on debug builds. Holds on all current
// x86-64, AArch64, and RISC-V 64-bit hardware. Intel LAM and ARM TBI
// do not affect untagged userspace pointers.
#define LOOM_USE_POINTER_MASK UINT64_C(0x0000FFFFFFFFFFFF)

static inline loom_use_t loom_use_make(loom_op_t* user_op,
                                       uint16_t operand_index) {
  IREE_ASSERT(((uintptr_t)user_op & ~LOOM_USE_POINTER_MASK) == 0,
              "op pointer exceeds 48-bit address space");
  return (uint64_t)(uintptr_t)user_op | ((uint64_t)operand_index << 48);
}

static inline loom_op_t* loom_use_user_op(loom_use_t use) {
  return (loom_op_t*)(uintptr_t)(use & LOOM_USE_POINTER_MASK);
}

static inline uint16_t loom_use_operand_index(loom_use_t use) {
  return (uint16_t)(use >> 48);
}

#else  // 32-bit

typedef struct loom_use_t {
  loom_op_t* user_op;
  uint16_t operand_index;
  uint16_t reserved;
} loom_use_t;

static inline loom_use_t loom_use_make(loom_op_t* user_op,
                                       uint16_t operand_index) {
  loom_use_t use = {user_op, operand_index, 0};
  return use;
}

static inline loom_op_t* loom_use_user_op(loom_use_t use) {
  return use.user_op;
}

static inline uint16_t loom_use_operand_index(loom_use_t use) {
  return use.operand_index;
}

#endif  // IREE_PTR_SIZE

static_assert(sizeof(loom_use_t) == 8, "loom_use_t must be 8 bytes");

//===----------------------------------------------------------------------===//
// Value definition pointer
//===----------------------------------------------------------------------===//
//
// A tagged pointer that records how a value was defined: either as the
// result of an operation (loom_op_t*) or as a block argument
// (loom_block_t*). The LOOM_VALUE_FLAG_BLOCK_ARG flag on the value
// disambiguates which pointer type is stored.
//
// On 64-bit: the pointer and index are packed into a single uint64_t.
// Low 48 bits hold the pointer (same constraint as loom_use_t), high
// 16 bits hold the result_index (op results) or arg_index (block args).
//
// On 32-bit (WASM): natural struct layout, 8 bytes.
//
// This enables O(1) pattern matching: given a value, follow the def
// pointer to the defining op and inspect its kind, operands, or
// attributes without walking the region tree.

#if defined(IREE_PTR_SIZE_64)

typedef uint64_t loom_value_def_t;

// Mask for extracting the 48-bit pointer from a def entry.
// Same constraint as loom_use_t: validated at creation on debug builds.
#define LOOM_VALUE_DEF_POINTER_MASK UINT64_C(0x0000FFFFFFFFFFFF)

// Creates a def pointer for an op result.
static inline loom_value_def_t loom_value_def_make_op(loom_op_t* op,
                                                      uint16_t result_index) {
  IREE_ASSERT(((uintptr_t)op & ~LOOM_VALUE_DEF_POINTER_MASK) == 0,
              "op pointer exceeds 48-bit address space");
  return (uint64_t)(uintptr_t)op | ((uint64_t)result_index << 48);
}

// Creates a def pointer for a block argument.
static inline loom_value_def_t loom_value_def_make_block(loom_block_t* block,
                                                         uint16_t arg_index) {
  IREE_ASSERT(((uintptr_t)block & ~LOOM_VALUE_DEF_POINTER_MASK) == 0,
              "block pointer exceeds 48-bit address space");
  return (uint64_t)(uintptr_t)block | ((uint64_t)arg_index << 48);
}

// Creates an empty def pointer (value not yet associated with a def site).
static inline loom_value_def_t loom_value_def_make_none(void) { return 0; }

// Extracts the defining op pointer. Caller must check that the value
// is not a block argument. Returns NULL if the def is unset.
static inline loom_op_t* loom_def_op(loom_value_def_t def) {
  return (loom_op_t*)(uintptr_t)(def & LOOM_VALUE_DEF_POINTER_MASK);
}

// Extracts the owning block pointer. Caller must check that the value
// is a block argument. Returns NULL if the def is unset.
static inline loom_block_t* loom_def_block(loom_value_def_t def) {
  return (loom_block_t*)(uintptr_t)(def & LOOM_VALUE_DEF_POINTER_MASK);
}

// Extracts the result index (op results) or arg index (block args).
static inline uint16_t loom_def_index(loom_value_def_t def) {
  return (uint16_t)(def >> 48);
}

#else  // 32-bit

typedef struct loom_value_def_t {
  void* pointer;   // loom_op_t* or loom_block_t*.
  uint16_t index;  // result_index or arg_index.
  uint16_t reserved;
} loom_value_def_t;

static inline loom_value_def_t loom_value_def_make_op(loom_op_t* op,
                                                      uint16_t result_index) {
  loom_value_def_t def = {op, result_index, 0};
  return def;
}

static inline loom_value_def_t loom_value_def_make_block(loom_block_t* block,
                                                         uint16_t arg_index) {
  loom_value_def_t def = {block, arg_index, 0};
  return def;
}

static inline loom_value_def_t loom_value_def_make_none(void) {
  loom_value_def_t def = {NULL, 0, 0};
  return def;
}

static inline loom_op_t* loom_def_op(loom_value_def_t def) {
  return (loom_op_t*)def.pointer;
}

static inline loom_block_t* loom_def_block(loom_value_def_t def) {
  return (loom_block_t*)def.pointer;
}

static inline uint16_t loom_def_index(loom_value_def_t def) {
  return def.index;
}

#endif  // IREE_PTR_SIZE

static_assert(sizeof(loom_value_def_t) == 8,
              "loom_value_def_t must be 8 bytes");

//===----------------------------------------------------------------------===//
// Value flags
//===----------------------------------------------------------------------===//

// Maximum number of uses stored inline in a loom_value_t before
// falling back to an arena-allocated overflow array.
// 3 covers the vast majority: most op results are used 1-2 times.
#define LOOM_VALUE_INLINE_USE_COUNT 3

// Individual flag bits for loom_value_flags_t.
enum loom_value_flag_bits_e {
  // This value is a block argument (entry to a function, loop carried,
  // branch target), not an operation result. When set, the def field
  // stores a loom_block_t* (use loom_value_def_block to extract).
  LOOM_VALUE_FLAG_BLOCK_ARG = 1u << 0,

  // This value has been consumed by a tied operand (linear ownership
  // transfer). Any use of this value after the consuming op is a
  // verification error. Set by the verifier or during IR construction.
  LOOM_VALUE_FLAG_CONSUMED = 1u << 1,

  // The use list has overflowed inline storage. When set, access uses
  // through overflow_uses pointer instead of inline_uses array.
  // Check: if use_count > LOOM_VALUE_INLINE_USE_COUNT, this must be set.
  LOOM_VALUE_FLAG_OVERFLOW_USES = 1u << 2,
};
typedef uint16_t loom_value_flags_t;

//===----------------------------------------------------------------------===//
// Value — 64-byte cache-line-aligned
//===----------------------------------------------------------------------===//

// An SSA value: either an operation result or a block argument.
//
// Values live in the module's value table (module->values), accessed
// by loom_value_id_t. The entire value (name, type, definition site,
// and first 3 uses) fits in a single 64-byte cache line.
//
// Thread safety: values are owned by their module. Read access is safe
// from any thread when the module is immutable (during parallel
// compilation phases). Mutation requires exclusive module access.
//
// Lifetime: values are arena-allocated and live until the module is
// destroyed. Individual values are never freed. Erased operations
// leave their result values in the table (with use_count == 0).
//
// The type field is stored inline (24 bytes) to avoid a pointer chase.
// For the hot path of type checking during pattern matching, this
// means one cache-line fetch gives you everything about a value.
typedef iree_alignas(64) struct loom_value_t {
  // Interned bare SSA name (e.g., "x", "tile", "0") without sigil.
  // The printer adds '%' when emitting. Used for round-trip printing.
  // The name is a property of the value, not the type: two values can
  // have different names but the same type.
  loom_string_id_t name_id;

  // Number of operations that use this value as an operand.
  // When 0, the value is dead (candidate for dead code elimination).
  uint16_t use_count;

  // Bitfield of loom_value_flag_bits_e.
  loom_value_flags_t flags;

  // --- 8 bytes ---

  // Full type, stored inline. 24 bytes. No pointer chase.
  // Includes shape dims (static sizes or SSA value IDs for dynamic
  // dims), element type, and encoding reference.
  loom_type_t type;

  // --- 32 bytes ---

  // Tagged pointer to the defining site. For op results, stores the
  // loom_op_t* and result index. For block arguments, stores the
  // loom_block_t* and arg index. Use loom_value_def_op/block/index
  // to extract (checking LOOM_VALUE_FLAG_BLOCK_ARG first).
  // Set by loom_builder_finalize_op (op results) or
  // loom_block_add_arg (block arguments). Zero until set.
  loom_value_def_t def;

  // --- 40 bytes ---

  // Inline use storage (common path) or overflow pointer.
  //
  // When use_count <= LOOM_VALUE_INLINE_USE_COUNT:
  //   Uses are stored directly in inline_uses[0..use_count-1].
  //   No pointer chase, no arena allocation.
  //
  // When use_count > LOOM_VALUE_INLINE_USE_COUNT:
  //   LOOM_VALUE_FLAG_OVERFLOW_USES is set.
  //   overflow_uses points to an arena-allocated array of
  //   overflow_capacity entries. When use_count reaches
  //   overflow_capacity, a new 2x array is arena-allocated and
  //   the old one is abandoned (arena frees all at module destruction).
  union {
    loom_use_t inline_uses[LOOM_VALUE_INLINE_USE_COUNT];
    struct {
      loom_use_t* overflow_uses;
      uint32_t overflow_capacity;
      uint32_t _reserved_0;
      uint64_t _reserved_1;
    };
  };

  // --- 64 bytes ---
} loom_value_t;

static_assert(sizeof(loom_value_t) == 64, "loom_value_t must be 64 bytes");

//===----------------------------------------------------------------------===//
// Value accessors
//===----------------------------------------------------------------------===//

static inline bool loom_value_is_block_arg(const loom_value_t* value) {
  return iree_any_bit_set(value->flags, LOOM_VALUE_FLAG_BLOCK_ARG);
}

static inline bool loom_value_is_consumed(const loom_value_t* value) {
  return iree_any_bit_set(value->flags, LOOM_VALUE_FLAG_CONSUMED);
}

static inline bool loom_value_has_overflow_uses(const loom_value_t* value) {
  return iree_any_bit_set(value->flags, LOOM_VALUE_FLAG_OVERFLOW_USES);
}

// Returns a const pointer to the use array (inline or overflow).
// Valid for iteration over use_count entries.
static inline const loom_use_t* loom_value_uses(const loom_value_t* value) {
  if (loom_value_has_overflow_uses(value)) {
    return value->overflow_uses;
  }
  return value->inline_uses;
}

// Returns a mutable pointer to the use array (inline or overflow).
// Used by use-list maintenance functions (add_use, remove_use).
static inline loom_use_t* loom_value_uses_mutable(loom_value_t* value) {
  if (loom_value_has_overflow_uses(value)) {
    return value->overflow_uses;
  }
  return value->inline_uses;
}

// Returns true if the value has no uses (dead, candidate for DCE).
static inline bool loom_value_has_no_uses(const loom_value_t* value) {
  return value->use_count == 0;
}

// Returns true if the value has exactly one use.
static inline bool loom_value_has_single_use(const loom_value_t* value) {
  return value->use_count == 1;
}

// Returns the operation that defines this value. The value must be an
// op result (not a block argument). Returns NULL if the def pointer
// has not been set yet (value was just created, finalize_op not called).
//
// Usage (pattern matching — "is this value defined by a constant?"):
//   loom_op_t* def_op = loom_value_def_op(value);
//   if (def_op && loom_test_constant_isa(def_op)) { ... }
static inline loom_op_t* loom_value_def_op(const loom_value_t* value) {
  IREE_ASSERT(!loom_value_is_block_arg(value));
  return loom_def_op(value->def);
}

// Returns the block that owns this block argument. The value must be
// a block argument (LOOM_VALUE_FLAG_BLOCK_ARG set).
static inline loom_block_t* loom_value_def_block(const loom_value_t* value) {
  IREE_ASSERT(loom_value_is_block_arg(value));
  return loom_def_block(value->def);
}

// Returns the result index (for op results) or argument index (for
// block arguments) of this value within its defining op or block.
static inline uint16_t loom_value_def_index(const loom_value_t* value) {
  return loom_def_index(value->def);
}

//===----------------------------------------------------------------------===//
// Tied result
//===----------------------------------------------------------------------===//

// Binding between a result and the operand whose storage it reuses.
// Stored in the operation's tied result list.
//
// The operand is consumed (linear ownership transfer). The result
// occupies the operand's buffer. If has_type_change is set, the
// result has a different type than the operand (e.g., a reshape or
// encoding change over the same storage).
typedef struct loom_tied_result_t {
  // TODO(benvanik): pack this better - that bool is costing bytes.
  uint16_t result_index;
  uint16_t operand_index;
  bool has_type_change;
} loom_tied_result_t;

//===----------------------------------------------------------------------===//
// Operation
//===----------------------------------------------------------------------===//

// Op kind: (dialect_id << 8 | op_index). The high byte identifies the
// dialect, the low byte identifies the op within the dialect.
typedef uint16_t loom_op_kind_t;

// Each dialect has a unique 8-bit ID. The high byte of loom_op_kind_t
// carries the dialect ID, the low byte the op index within that
// dialect. This gives 256 dialects x 256 ops each = 65536 total.
//
//   0x00       = unknown/invalid
//   0x01..0x7F = internal dialects (assigned by loom)
//   0x80..0xFE = external/user dialects (assigned at registration)
//   0xFF       = reserved
typedef enum loom_dialect_id_e {
  LOOM_DIALECT_UNKNOWN = 0x00,
  LOOM_DIALECT_TEST = 0x01,
  LOOM_DIALECT_SCALAR = 0x02,
  LOOM_DIALECT_TILE = 0x03,
  LOOM_DIALECT_TENSOR = 0x04,
  LOOM_DIALECT_SCF = 0x05,
  LOOM_DIALECT_FUNC = 0x06,
  LOOM_DIALECT_HAL = 0x07,
  LOOM_DIALECT_VM = 0x08,
  LOOM_DIALECT_ENCODING = 0x09,
  LOOM_DIALECT_POOL = 0x0A,
  LOOM_DIALECT_GLOBAL = 0x0B,
  LOOM_DIALECT_RESERVED = 0xFF,
} loom_dialect_id_t;
#define LOOM_OP_KIND_UNKNOWN ((loom_op_kind_t)0)

// Constructs a loom_op_kind_t from a dialect ID and an op index.
// Both must fit in a uint8_t. The result is a uint16_t.
#define LOOM_OP_KIND(dialect, index) \
  ((loom_op_kind_t)(((dialect) << 8) | (index)))

// Maximum number of built-in dialects. Dialect IDs must be less than
// this value. Matches the size of the dialect vtable registry array.
#define LOOM_DIALECT_BUILTIN_COUNT_ 16

// Extracts the dialect ID (high byte) from an op kind.
static inline uint8_t loom_op_dialect_id(loom_op_kind_t kind) {
  return (uint8_t)(kind >> 8);
}

// Extracts the op index (low byte) from an op kind.
static inline uint8_t loom_op_dialect_index(loom_op_kind_t kind) {
  return (uint8_t)(kind & 0xFF);
}

enum loom_op_flag_bits_e {
  // The operation has been logically erased. Enumeration macros skip
  // dead ops. The memory is not freed (arena-owned) but the op is
  // invisible to passes and will not be serialized.
  LOOM_OP_FLAG_DEAD = 1u << 0,

  // The operation is on the rewriter's worklist. Set when added,
  // cleared when popped. Prevents duplicate entries in the worklist
  // (O(1) dedup instead of hash set lookup). Harmless if stale —
  // overwritten on the next driver invocation.
  LOOM_OP_FLAG_ON_WORKLIST = 1u << 1,
};
typedef uint8_t loom_op_flags_t;

// Trait bitfield on the op vtable. Enables fast checks in passes:
//   vtable->traits & LOOM_TRAIT_PURE
enum loom_trait_bits_e {
  LOOM_TRAIT_PURE = 1u << 0,
  LOOM_TRAIT_COMMUTATIVE = 1u << 1,
  LOOM_TRAIT_IDEMPOTENT = 1u << 2,
  LOOM_TRAIT_INVOLUTION = 1u << 3,
  LOOM_TRAIT_TERMINATOR = 1u << 4,
  LOOM_TRAIT_CONSTANT_LIKE = 1u << 5,
  LOOM_TRAIT_ELEMENTWISE = 1u << 6,
  LOOM_TRAIT_DECOMPOSABLE = 1u << 7,
  // Op defines a named symbol (function, global) rather than producing
  // SSA values. The printer omits the LHS result list (%name =) for
  // symbol-defining ops even when the op carries result values (which
  // hold return type information for the printer's ResultTypeList).
  LOOM_TRAIT_SYMBOL_DEFINE = 1u << 8,
  // Op reads from at least one resource operand (has LOOM_OPERAND_READS).
  // Derived by the generator from per-operand effect flags.
  LOOM_TRAIT_READS_MEMORY = 1u << 9,
  // Op writes to at least one resource operand (has LOOM_OPERAND_WRITES).
  // Derived by the generator from per-operand effect flags.
  LOOM_TRAIT_WRITES_MEMORY = 1u << 10,
  // Op may produce different results for identical inputs (RNG,
  // timestamps, hardware state queries). Prevents CSE and LICM but
  // does not prevent DCE — a non-deterministic op with unused results
  // and no write effects is dead.
  LOOM_TRAIT_NON_DETERMINISTIC = 1u << 11,
  // Op has dynamic effects that depend on runtime state (e.g., func.call
  // effects depend on the callee). Passes treat this conservatively as
  // both READS_MEMORY and WRITES_MEMORY.
  LOOM_TRAIT_UNKNOWN_EFFECTS = 1u << 12,
  // Op's regions cannot reference values defined outside the op.
  // Values enter the region only through block arguments. CSE and
  // other passes must not substitute inner values with outer ones.
  LOOM_TRAIT_ISOLATED_FROM_ABOVE = 1u << 13,
  // Each execution produces a result with a distinct identity, even
  // when operands and attributes are identical. Two identical allocation
  // ops must not be merged. Prevents CSE but allows DCE (an unused
  // allocation with no write effects is dead) and LICM (a
  // loop-invariant allocation can be hoisted out of the loop). Derived
  // automatically by the generator when any result has LOOM_RESULT_ALLOCATES.
  LOOM_TRAIT_UNIQUE_IDENTITY = 1u << 14,
};
typedef uint32_t loom_trait_flags_t;

// Returns true if the trait flags indicate the op may write to memory
// or has unknown effects. Works on raw bitfields to avoid redundant
// vtable lookups when the vtable is already resolved.
static inline bool loom_traits_may_write(loom_trait_flags_t traits) {
  return (traits & (LOOM_TRAIT_WRITES_MEMORY | LOOM_TRAIT_UNKNOWN_EFFECTS)) !=
         0;
}

// Returns true if the trait flags indicate the op may produce different
// results for identical inputs, writes memory, or has unknown effects.
static inline bool loom_traits_has_side_effects(loom_trait_flags_t traits) {
  return (traits & (LOOM_TRAIT_WRITES_MEMORY | LOOM_TRAIT_UNKNOWN_EFFECTS |
                    LOOM_TRAIT_NON_DETERMINISTIC)) != 0;
}

// Returns true if the trait flags indicate the op's regions cannot
// reference values defined outside the op.
static inline bool loom_traits_is_isolated(loom_trait_flags_t traits) {
  return (traits & LOOM_TRAIT_ISOLATED_FROM_ABOVE) != 0;
}

// Returns true if each execution of the op produces a result with a
// distinct identity. Prevents CSE but not DCE or LICM.
static inline bool loom_traits_has_unique_identity(loom_trait_flags_t traits) {
  return (traits & LOOM_TRAIT_UNIQUE_IDENTITY) != 0;
}

// Structural flags on the op vtable (shared by all instances of an op kind).
enum loom_op_vtable_flag_bits_e {
  LOOM_OP_VTABLE_VARIADIC_OPERANDS = 1u << 0,
  LOOM_OP_VTABLE_VARIADIC_RESULTS = 1u << 1,
  LOOM_OP_VTABLE_HAS_INSTANCE_FLAGS = 1u << 2,
};
typedef uint8_t loom_op_vtable_flags_t;

// Op-specific verification callback. Called after the standard
// table-driven checks have established the op's structural invariants.
typedef iree_status_t (*loom_op_verify_fn_t)(const loom_module_t* module,
                                             const loom_op_t* op,
                                             iree_diagnostic_emitter_t emitter);

// Canonicalization callback. Attempts to simplify |op| using the
// rewriter.
typedef iree_status_t (*loom_canonicalize_fn_t)(loom_op_t* op,
                                                loom_rewriter_t* rewriter);

// Per-instance effective traits callback. Returns the trait flags for
// a specific op instance, incorporating instance-specific state like
// instance_flags. NULL means "use vtable->traits as-is" (the common
// case). Only ops whose traits depend on instance-specific state
// implement this — e.g., func.call uses it to report PURE when the
// canonicalizer has determined the callee is pure.
//
// Instance flags are dialect-specific per op kind: bit 0 on func.call
// means "callee pure" but bit 0 on scalar.addf means a fast-math flag.
// The callback gives each op kind control over interpretation.
typedef loom_trait_flags_t (*loom_effective_traits_fn_t)(const loom_op_t* op);

typedef struct loom_value_facts_t loom_value_facts_t;

// Fold callback. Computes output facts from operand facts for value
// analysis and constant folding. Called by the rewriter to propagate
// facts through operations and to fold ops to constants when all
// output facts are exact.
//
// Writes one entry per result into |result_facts|. Single-result ops
// write result_facts[0]. Multi-result ops fill the entire array.
// |module| provides type access (int vs float detection, bitwidth
// queries via loom_module_value_type). Must not mutate IR.
typedef void (*loom_op_fold_fn_t)(const loom_module_t* module,
                                  const loom_op_t* op,
                                  const loom_value_facts_t* operand_facts,
                                  loom_value_facts_t* result_facts);

//===----------------------------------------------------------------------===//
// FuncLike interface vtable
//===----------------------------------------------------------------------===//

// Sentinel values for interface vtable index fields.
#define LOOM_ATTR_INDEX_NONE ((uint8_t)0xFF)
#define LOOM_REGION_INDEX_NONE ((uint8_t)0xFF)

// Interface descriptor for function-like ops. Pure .rodata — a struct
// of attr/region indices that let generic code read the right fields
// from any implementing op without knowing its specific kind.
//
// Generated by c_tables.py from FuncLikeInterface declarations in the
// Python DSL. One instance per implementing op kind.
typedef struct loom_func_like_vtable_t {
  // Index of the symbol ref attr that names this function.
  uint8_t callee_attr_index;

  // Indices of optional enum attrs. LOOM_ATTR_INDEX_NONE if absent.
  uint8_t visibility_attr_index;
  uint8_t cc_attr_index;
  uint8_t purity_attr_index;

  // Index of the predicate list attr. LOOM_ATTR_INDEX_NONE if absent.
  uint8_t predicates_attr_index;

  // Body region index. LOOM_REGION_INDEX_NONE for bodyless ops
  // (func.decl, func.ukernel) that only declare a signature.
  uint8_t body_region_index;

  // Index of the implements string attr (for template/ukernel dispatch).
  // LOOM_ATTR_INDEX_NONE for def/decl.
  uint8_t implements_attr_index;

  // Index of the priority i64 attr (for template/ukernel dispatch).
  // LOOM_ATTR_INDEX_NONE for def/decl.
  uint8_t priority_attr_index;

  // When true, function arguments are stored as the op's operands
  // rather than as block arguments of the body region. Used for ops
  // that declare a signature without providing a body — the parser
  // stores FUNC_ARGS as operands when no REGION follows.
  bool args_as_operands;
} loom_func_like_vtable_t;

// Fat reference to a function-like op: pairs the op with its interface
// vtable. 16 bytes, passed by value. Constructed by the pass manager
// when iterating symbols, or by any code that has an op and knows it
// implements FuncLike.
typedef struct loom_func_like_t {
  loom_op_t* op;
  const loom_func_like_vtable_t* vtable;
} loom_func_like_t;

//===----------------------------------------------------------------------===//
// Op vtable
//===----------------------------------------------------------------------===//

// Per-op metadata in .rodata. One vtable per op kind.
//
// Contains everything the printer, parser, verifier, and diagnostics
// need to handle an op without op-specific code. Vtables are registered
// per-dialect at context creation time.
//
// The name field is a B-string: [total_len][ns_len]"namespace.opname\0".
// Full name = name+2 for name[0] bytes. Namespace = name+2 for
// name[1] bytes. Short name = name+2+name[1]+1 for the remainder.
// Layout is cache-optimized for compiler pass inner loops:
//
//   Cache line 1 (bytes 0-63): scalars and pointers touched by every
//   pass on every op — traits, counts, canonicalize/fold fn pointers,
//   func_like interface, attr/operand descriptors.
//
//   Cache line 2 (bytes 64-127): verification, parse/print, and
//   diagnostics — descriptor arrays only needed by the verifier,
//   format tables only needed by the parser/printer, and the name
//   string only needed by diagnostics.
//
// With ~500 op kinds, keeping the hot path in one cache line avoids
// ~500 × 64B = 32KB of cold .rodata fetches per pass.
struct loom_op_vtable_t {
  // --- Cache line 1: compiler pass hot path (0-63) ---

  loom_trait_flags_t traits;
  uint8_t fixed_operand_count;
  uint8_t fixed_result_count;
  uint8_t attribute_count;
  uint8_t region_count;
  loom_op_vtable_flags_t vtable_flags;
  // Symbol kind for ops with LOOM_TRAIT_SYMBOL_DEFINE. Set by the
  // Python generator from each op's declared symbol kind. Zero
  // (LOOM_SYMBOL_NONE) for ops that do not define symbols.
  loom_symbol_kind_t symbol_kind;
  uint8_t constraint_count;
  // 5 bytes padding to align pointers at offset 16.

  loom_canonicalize_fn_t canonicalize;
  loom_op_fold_fn_t fold;
  loom_effective_traits_fn_t effective_traits;
  const loom_func_like_vtable_t* func_like;
  const loom_attr_descriptor_t* attr_descriptors;
  const loom_operand_descriptor_t* operand_descriptors;

  // --- Cache line 2: verify + parse/print + diagnostics (64-127) ---

  const loom_result_descriptor_t* result_descriptors;
  const loom_region_descriptor_t* region_descriptors;
  const loom_constraint_t* constraints;
  loom_op_verify_fn_t verify;
  const uint8_t* name;
  const loom_format_element_t* format_elements;
  const loom_bstring_t* instance_flags_case_names;
  uint16_t format_element_count;
  uint8_t instance_flags_case_count;
  // 5 bytes padding to 128.
};

// Returns the full dotted name as a string view (e.g., "test.addi").
static inline iree_string_view_t loom_op_vtable_name(
    const loom_op_vtable_t* vtable) {
  return iree_make_string_view((const char*)(vtable->name + 2),
                               vtable->name[0]);
}

// Returns the namespace portion of the name (e.g., "test").
static inline iree_string_view_t loom_op_vtable_namespace(
    const loom_op_vtable_t* vtable) {
  return iree_make_string_view((const char*)(vtable->name + 2),
                               vtable->name[1]);
}

// Returns the short name after the dot (e.g., "addi").
static inline iree_string_view_t loom_op_vtable_short_name(
    const loom_op_vtable_t* vtable) {
  uint8_t namespace_length = vtable->name[1];
  return iree_make_string_view(
      (const char*)(vtable->name + 2 + namespace_length + 1),
      vtable->name[0] - namespace_length - 1);
}

// An IR operation.
//
// Operations are arena-allocated within their owning block. Operands
// and results are stored as value IDs (indices into the module's value
// table) in variable-length trailing data accessed through helpers.
//
// Thread safety: operations are owned by their module. Same rules as
// loom_value_t: read-safe when module is immutable, mutation requires
// exclusive access.
//
// Lifetime: arena-allocated, lives until module destruction. Erased
// ops are flagged LOOM_OP_FLAG_DEAD but not freed.
typedef struct loom_op_t {
  loom_op_kind_t kind;
  uint16_t operand_count;
  uint16_t result_count;
  uint16_t tied_result_count;
  uint8_t region_count;
  uint8_t attribute_count;
  loom_op_flags_t flags;
  // Per-op-instance flags: fast-math flags for float ops, overflow
  // flags for integer ops. Declared via the Flags format element in
  // the Python DSL. Zero when the op has no flags or uses strict
  // semantics. Bit layout is dialect-specific (each flag enum defines
  // its own mask constants in the per-dialect ops.h).
  uint8_t instance_flags;
  // Source location (index into module's location table).
  // Carries the full source range for diagnostics and debug info.
  // Value locations are derived: value -> def -> op -> location.
  loom_location_id_t location;
  // Op whose region contains this op's block. NULL for module-level
  // ops (direct children of the module body). Set during construction
  // by the builder and maintained by the rewriter on op movement.
  // Enables O(depth) ancestry queries without full IR walks.
  struct loom_op_t* parent_op;
  // Block that contains this op. NULL until inserted into a block.
  // Set by loom_builder_allocate_op and maintained by the rewriter.
  // Enables O(1) same-block checks for canonicalization patterns.
  loom_block_t* parent_block;
  // Variable-length trailing data (arena-allocated, accessed via helpers).
  // Pointer-sized fields first for natural alignment:
  //   loom_region_t*     regions[region_count]
  //   loom_value_id_t    operands[operand_count]
  //   loom_value_id_t    results[result_count]
  //   loom_tied_result_t tied_results[tied_result_count]
  //   <padding to alignof(loom_attribute_t)>
  //   loom_attribute_t   attributes[attribute_count]
} loom_op_t;

static_assert(sizeof(loom_op_t) == 32, "loom_op_t must be 32 bytes");

//===----------------------------------------------------------------------===//
// Op trailing data accessors
//===----------------------------------------------------------------------===//
//
// Operations store variable-length data contiguously after the loom_op_t
// header (arena-allocated in one bump). Pointer-sized fields come first
// so they are naturally aligned after the header:
//
//   loom_region_t*     regions[region_count]       (8 bytes each, aligned)
//   loom_value_id_t    operands[operand_count]     (4 bytes each)
//   loom_value_id_t    results[result_count]        (4 bytes each)
//   loom_tied_result_t tied_results[tied_result_count]
//   <padding to alignof(loom_attribute_t)>
//   loom_attribute_t   attributes[attribute_count]  (16 bytes each)

// Returns a pointer to the region pointer array (first in trailing data).
static inline loom_region_t** loom_op_regions(const loom_op_t* op) {
  return (loom_region_t**)((uint8_t*)op + sizeof(loom_op_t));
}

// Returns a pointer to the operand value ID array (after regions).
// The non-const variant accepts const loom_op_t* for backward
// compatibility (C trailing data doesn't propagate const). Prefer
// loom_op_const_operands when mutation is not needed.
static inline loom_value_id_t* loom_op_operands(const loom_op_t* op) {
  return (loom_value_id_t*)(loom_op_regions(op) + op->region_count);
}
static inline const loom_value_id_t* loom_op_const_operands(
    const loom_op_t* op) {
  return (const loom_value_id_t*)(loom_op_regions(op) + op->region_count);
}

// Returns a pointer to the result value ID array (after operands).
static inline loom_value_id_t* loom_op_results(const loom_op_t* op) {
  return loom_op_operands(op) + op->operand_count;
}
static inline const loom_value_id_t* loom_op_const_results(
    const loom_op_t* op) {
  return loom_op_const_operands(op) + op->operand_count;
}

// Returns a pointer to the tied result binding array (after results).
static inline loom_tied_result_t* loom_op_tied_results(const loom_op_t* op) {
  return (loom_tied_result_t*)(loom_op_results(op) + op->result_count);
}

// Returns a pointer to the attribute array. Aligned to
// alignof(loom_attribute_t) because the union contains int64_t/double.
static inline loom_attribute_t* loom_op_attrs(const loom_op_t* op) {
  uintptr_t after_tied =
      (uintptr_t)(loom_op_tied_results(op) + op->tied_result_count);
  return (loom_attribute_t*)iree_host_align(after_tied,
                                            iree_alignof(loom_attribute_t));
}

//===----------------------------------------------------------------------===//
// Block
//===----------------------------------------------------------------------===//

// A basic block: a linear sequence of operations with optional
// block arguments.
//
// Block arguments serve as the entry interface (function args, loop
// carried values, branch target parameters). They are stored as
// value IDs into the module's value table.
typedef struct loom_block_t {
  loom_string_id_t label_id;
  uint16_t arg_count;
  uint16_t arg_capacity;
  uint16_t op_count;
  uint16_t op_capacity;
  uint16_t flags;
  uint16_t reserved;
  loom_value_id_t* arg_ids;
  loom_op_t** ops;
} loom_block_t;

static_assert(sizeof(loom_block_t) == 32, "loom_block_t must be 32 bytes");

// Returns the |arg_index|-th block argument value ID.
static inline loom_value_id_t loom_block_arg_id(const loom_block_t* block,
                                                uint16_t arg_index) {
  IREE_ASSERT(arg_index < block->arg_count);
  return block->arg_ids[arg_index];
}

// Returns the |op_index|-th operation in |block|.
static inline const loom_op_t* loom_block_const_op(const loom_block_t* block,
                                                   uint16_t op_index) {
  IREE_ASSERT(op_index < block->op_count);
  return block->ops[op_index];
}
static inline loom_op_t* loom_block_op(loom_block_t* block, uint16_t op_index) {
  return (loom_op_t*)loom_block_const_op(block, op_index);
}

// Returns the terminator/last op in |block|. |block| must be non-empty.
static inline const loom_op_t* loom_block_const_last_op(
    const loom_block_t* block) {
  IREE_ASSERT(block->op_count > 0);
  return loom_block_const_op(block, (uint16_t)(block->op_count - 1));
}

//===----------------------------------------------------------------------===//
// Region
//===----------------------------------------------------------------------===//

// Per-instance region flags. These are stored on each loom_region_t
// and describe the runtime state of a specific region, as opposed to
// the vtable's loom_region_flag_bits_e which describe the op-kind's
// static requirements. Stored per-region (not per-op-kind) because
// the same op kind may own both structured and CFG regions at
// different points in the pipeline, and mixed IR (some functions
// structured, others CFG) is valid on input.
enum loom_region_instance_flag_bits_e {
  // Region uses unstructured control flow with explicit branch
  // terminators and successor blocks. When clear, dominance is
  // structural: entry block dominates all siblings, nesting gives
  // the dominator tree. When set, dominance must be computed from
  // predecessor edges (dominator tree construction).
  LOOM_REGION_INSTANCE_FLAG_CFG = 1u << 0,
};
typedef uint16_t loom_region_instance_flags_t;

// A region: an ordered list of blocks. Used for function bodies,
// loop bodies (scf.for), conditional branches (scf.if then/else).
// Regions nest: a region's block contains ops that may have their
// own regions.
//
// Block 0 is embedded directly in the region for the common single-block
// case and is always the entry block when block_count > 0. Additional blocks
// are arena-allocated individually and referenced through the growable
// |blocks| table, so appending blocks does not relocate existing block
// objects or invalidate loom_value_def_t / loom_op_t block pointers.
typedef struct loom_region_t {
  uint16_t block_count;
  uint16_t block_capacity;
  loom_region_instance_flags_t flags;
  uint16_t reserved;
  loom_block_t entry_block;
  loom_block_t** blocks;
  loom_block_t* inline_blocks[1];
} loom_region_t;

static_assert(sizeof(loom_region_t) == 56, "loom_region_t must be 56 bytes");

// Returns the block at |block_index| in |region|.
static inline loom_block_t* loom_region_block(loom_region_t* region,
                                              uint16_t block_index) {
  IREE_ASSERT(block_index < region->block_count);
  return region->blocks[block_index];
}
static inline const loom_block_t* loom_region_const_block(
    const loom_region_t* region, uint16_t block_index) {
  IREE_ASSERT(block_index < region->block_count);
  return region->blocks[block_index];
}

// Returns the entry block for |region|.
static inline loom_block_t* loom_region_entry_block(loom_region_t* region) {
  return loom_region_block(region, 0);
}
static inline const loom_block_t* loom_region_const_entry_block(
    const loom_region_t* region) {
  return loom_region_const_block(region, 0);
}

// Returns the number of arguments on the entry block of |region|. |region|
// must be non-empty.
static inline uint16_t loom_region_entry_arg_count(
    const loom_region_t* region) {
  return loom_region_const_entry_block(region)->arg_count;
}

// Returns the |arg_index|-th argument ID from the entry block of |region|.
// |region| must be non-empty and |arg_index| must be in range.
static inline loom_value_id_t loom_region_entry_arg_id(
    const loom_region_t* region, uint16_t arg_index) {
  return loom_block_arg_id(loom_region_const_entry_block(region), arg_index);
}

//===----------------------------------------------------------------------===//
// Function kind, flags, and calling convention
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Symbol flags and kinds
//===----------------------------------------------------------------------===//

// Symbol kind stored as a fixed-size integer for explicit control
// over struct layout. The enum defines the values.
//
// LOOM_SYMBOL_NONE (0) is the zero-initialized default for unlinked
// symbols. This is intentional: symbols are created when the parser
// first encounters a SYMBOL_REF (which may be a forward reference)
// and linked later when the SYMBOL_DEFINE op is finalized. Code that
// iterates symbols must handle NONE gracefully.
typedef enum loom_symbol_kind_e {
  // Unlinked or unknown. Zero-initialized symbols start here.
  LOOM_SYMBOL_NONE = 0,
  LOOM_SYMBOL_FUNC_DEF = 1,
  LOOM_SYMBOL_FUNC_DECL = 2,
  LOOM_SYMBOL_FUNC_TEMPLATE = 3,
  LOOM_SYMBOL_FUNC_UKERNEL = 4,
  // Sentinel: first non-function-like symbol kind.
  LOOM_SYMBOL_FUNC_COUNT_ = 5,
  LOOM_SYMBOL_GLOBAL = 5,
  LOOM_SYMBOL_EXECUTABLE = 6,
  LOOM_SYMBOL_COUNT_,
} loom_symbol_kind_e;

// Returns true if the symbol kind is a function-like (def, decl,
// template, or ukernel). Function-like symbols carry a defining_op
// pointer to the op that implements them.
static inline bool loom_symbol_kind_is_function_like(loom_symbol_kind_t kind) {
  return kind >= LOOM_SYMBOL_FUNC_DEF && kind < LOOM_SYMBOL_FUNC_COUNT_;
}

enum loom_symbol_flag_bits_e {
  // Symbol is visible outside the module (exported for linking).
  LOOM_SYMBOL_FLAG_PUBLIC = 1u << 0,
  // Use list has overflowed inline storage.
  LOOM_SYMBOL_FLAG_OVERFLOW_USES = 1u << 1,
};
typedef uint16_t loom_symbol_flags_t;

//===----------------------------------------------------------------------===//
// Symbol
//===----------------------------------------------------------------------===//

// A symbol use: records which operation references this symbol.
typedef struct loom_symbol_use_t {
  uint32_t user_op_id;
  uint16_t block_id;
  uint16_t operand_index;
} loom_symbol_use_t;

// A module-level named symbol with use tracking.
//
// Symbols are the top-level entities in a module: functions, globals,
// and executables. Each has a name (interned string), a kind tag, and a
// pointer to the defining op. Symbols maintain incremental use lists so
// "find all references to @foo" is O(uses), not O(module size).
//
// For function-like symbols (def, decl, template, ukernel), the defining op
// is a func.* op. All function properties — signature, purity, calling
// convention, body region — are read through the loom_func_like_t interface
// on the defining op, which eliminates redundant copies and desync risk.
//
// Thread safety: symbols are owned by their module. Same rules as other IR
// nodes.
typedef struct loom_symbol_t {
  loom_string_id_t name_id;
  loom_symbol_kind_t kind;
  loom_symbol_flags_t flags;
  uint16_t use_count;
  // The op that defines this symbol. Set by loom_builder_finalize_op() when
  // an op with LOOM_TRAIT_SYMBOL_DEFINE is finalized. NULL until then (e.g.,
  // during parsing before the op is fully built).
  loom_op_t* defining_op;
  union {
    // Typed pointers for future non-func symbol kinds. Both are void* until
    // the corresponding IR nodes are added (globals and executables).
    void* global;
    void* executable;
  };
  union {
    loom_symbol_use_t inline_uses[2];
    loom_symbol_use_t* overflow_uses;
  };
} loom_symbol_t;

//===----------------------------------------------------------------------===//
// Tables
//===----------------------------------------------------------------------===//

// Interned string table. All strings in a module are deduplicated here.
// String IDs are stable across the module's lifetime.
//
// Lookup by content (for interning during construction) uses a hash
// map. Lookup by ID (for printing) is a direct array index.
typedef struct loom_string_table_t {
  iree_host_size_t count;
  iree_host_size_t capacity;
  iree_string_view_t* entries;
} loom_string_table_t;

// Value table. Cache-line-aligned entries for fast iteration.
// All values (op results and block arguments) in a module live here.
typedef struct loom_value_table_t {
  iree_host_size_t count;
  iree_host_size_t capacity;
  loom_value_t* entries;  // 64-byte aligned.
} loom_value_table_t;

// Symbol table. Flat (no nesting), one per module.
typedef struct loom_symbol_table_t {
  iree_host_size_t count;
  iree_host_size_t capacity;
  loom_symbol_t* entries;
} loom_symbol_table_t;

// Type table. Interned types for pointer-equality comparison.
typedef struct loom_type_table_t {
  iree_host_size_t count;
  iree_host_size_t capacity;
  loom_type_t* entries;
} loom_type_table_t;

// Encoding descriptor in the module's encoding table.
//
// Each unique encoding parameterization in a program gets one entry.
// The encoding_id field in loom_type_t indexes into this table
// (1-based, 0 = no encoding).
//
// The name and alias are interned string IDs shared across all
// encoding instances with the same name. Parameters are named
// attributes (loom_named_attr_t from op_defs.h) — the same
// tagged-union attribute type used by ops, so encoding params can
// be integers, strings, booleans, etc.
//
// In the textual format, encodings print as #name or #name<k=v, ...>.
// When alias_id is set, the printer uses the alias (#enc) instead.
// Aliases are defined at file level: #enc = #q8_0<block=32>.
struct loom_encoding_t {
  loom_string_id_t name_id;
  loom_string_id_t alias_id;  // LOOM_STRING_ID_INVALID for none.
  uint8_t attribute_count;
  uint8_t reserved[3];
  // Arena-allocated array of named attributes. NULL when count is 0.
  const loom_named_attr_t* attributes;
};

// Encoding table. Each unique encoding parameterization in a module
// gets one entry, deduplicated by name + attribute equality.
typedef struct loom_encoding_table_t {
  iree_host_size_t count;
  iree_host_size_t capacity;
  loom_encoding_t* entries;
} loom_encoding_table_t;

// Open-addressing hash table for deduplicating strings and types during
// construction. Arena-allocated, freed when the module is destroyed.
// Lazy-initialized: capacity 0 means uninitialized, first use allocates.
typedef struct loom_intern_table_t {
  iree_host_size_t count;
  iree_host_size_t capacity;
  uint32_t* hashes;
  uint32_t* indices;
} loom_intern_table_t;

//===----------------------------------------------------------------------===//
// Module flags
//===----------------------------------------------------------------------===//

enum loom_module_flag_bits_e {
  // Declaration-only module: contains function signatures and type
  // definitions but no function bodies. Used for fast symbol
  // resolution during linking without parsing the bulk IR.
  LOOM_MODULE_FLAG_DECLARATION = 1u << 0,
};
typedef uint16_t loom_module_flags_t;

//===----------------------------------------------------------------------===//
// Module
//===----------------------------------------------------------------------===//

// A loom module: the top-level IR container.
//
// Owns all IR through an arena allocator. Creating a module allocates
// the arena. Destroying the module frees the arena and all IR within
// it in O(1) time. No individual deallocation of IR nodes.
//
// Thread safety: a module is single-owner. During parallel compilation
// phases, the module is immutable (const access from worker threads).
// Mutation (adding ops, modifying values) requires exclusive access.
// The module is NOT thread-safe for concurrent mutation.
//
// Lifetime: the module owns everything referenced through its tables.
// Values, operations, blocks, regions, strings, types: all arena-
// allocated, all freed when the module is destroyed. Pointers into
// the module's IR are invalid after module destruction.
typedef struct loom_module_t {
  // Flags (accessed frequently, placed first for cache locality).
  loom_module_flags_t flags;
  uint16_t reserved;

  // Module name. Every module is named (required for linking).
  loom_string_id_t name_id;

  // Owning context (provides vtables for op resolution).
  loom_context_t* context;

  // Allocator used to allocate and free the module struct itself.
  iree_allocator_t allocator;

  // Arena backing all IR storage. Bump-pointer allocation during
  // construction. O(1) destruction. Grows in large blocks (~64KB).
  iree_arena_allocator_t arena;

  // Interned strings (SSA names, function names, attribute keys).
  loom_string_table_t strings;

  // Interned types. Pointer equality works for interned types.
  loom_type_table_t types;

  // Encoding instances (e.g., q8_0 with block=32). Indexed 1-based
  // by the encoding_id field in loom_type_t (0 = no encoding).
  loom_encoding_table_t encodings;

  // All SSA values (op results and block arguments).
  // 64-byte aligned for cache-line access.
  loom_value_table_t values;

  // Module-level named symbols (functions, globals, executables).
  loom_symbol_table_t symbols;

  // Source locations for diagnostics, debug info, and source mapping.
  // Entry 0 is always LOOM_LOCATION_NONE (unknown/absent). Ops
  // reference entries by loom_location_id_t index. When locations are
  // stripped (e.g., release builds), the table is empty and all ops
  // reference LOOM_LOCATION_UNKNOWN (0).
  loom_location_table_t locations;

  // Module body: a region with a single entry block. All top-level ops
  // (function definitions, globals, executables) live in this block.
  // Created automatically by loom_module_create.
  loom_region_t* body;

  // Intern hash tables for deduplicating strings and types during
  // construction. Arena-allocated, lazy-initialized on first use.
  loom_intern_table_t string_intern;
  loom_intern_table_t type_intern;
} loom_module_t;

//===----------------------------------------------------------------------===//
// Context
//===----------------------------------------------------------------------===//

// Module list within the context (for cross-module linking).
typedef struct loom_module_list_t {
  iree_host_size_t count;
  loom_module_t** entries;
} loom_module_list_t;

// Encoding vtable list (one per encoding kind).
typedef struct loom_encoding_vtable_list_t {
  iree_host_size_t count;
  const loom_encoding_vtable_t** entries;
} loom_encoding_vtable_list_t;

//===----------------------------------------------------------------------===//
// Enumeration macros
//===----------------------------------------------------------------------===//
//
// Linux kernel-style iteration macros for walking IR structures.
// Each macro declares a typed loop variable and handles skipping
// dead/invalid entries. Stack-allocated, no heap allocation.
//
// Usage:
//
//   loom_op_t* op = NULL;
//   loom_block_for_each_op(block, op) {
//     if (loom_op_kind(op) == MY_OP_KIND) {
//       // Process op.
//     }
//   }
//
//   const loom_use_t* use = NULL;
//   loom_value_for_each_use(value, use) {
//     loom_op_t* user = loom_use_user_op(*use);
//     // Process user.
//   }

// Walk all live operations in a block (skips LOOM_OP_FLAG_DEAD).
#define loom_block_for_each_op(block, op_var)                          \
  for (iree_host_size_t _op_i = 0; _op_i < (block)->op_count; ++_op_i) \
    if (((op_var) = (block)->ops[_op_i]) &&                            \
        !iree_any_bit_set((op_var)->flags, LOOM_OP_FLAG_DEAD))

// Walk all blocks in a region.
#define loom_region_for_each_block(region, block_var)                         \
  for (iree_host_size_t _blk_i = 0; _blk_i < (region)->block_count; ++_blk_i) \
    if (((block_var) = (region)->blocks[_blk_i]))

// Walk all uses of a value.
#define loom_value_for_each_use(value, use_var)                            \
  for (iree_host_size_t _use_i = 0; _use_i < (value)->use_count; ++_use_i) \
    if (((use_var) = &loom_value_uses(value)[_use_i]))

// Walk all symbols in a module.
#define loom_module_for_each_symbol(module, sym_var)                  \
  for (iree_host_size_t _sym_i = 0; _sym_i < (module)->symbols.count; \
       ++_sym_i)                                                      \
    if (((sym_var) = &(module)->symbols.entries[_sym_i]))

#ifdef __cplusplus
}
#endif

#endif  // LOOM_IR_IR_H_
