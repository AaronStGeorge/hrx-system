// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core type definitions for the loom op infrastructure.
//
// This header defines the types that both hand-written and generated dialect
// code depend on: attributes, format elements, field descriptors, op vtables,
// and supporting enums. Generated per-dialect ops.h files include this header.
// Pass authors include the per-dialect ops.h, which transitively includes this.
//
// The op system is table-driven: each op kind has a vtable in .rodata
// containing format element arrays (4-byte instructions for the printer/parser
// interpreter), field descriptors, trait bits, and a B-string name. Adding ops
// adds .rodata tables, not .text code.

#ifndef LOOM_OPS_OP_DEFS_H_
#define LOOM_OPS_OP_DEFS_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/ir/types.h"
#include "loom/util/bstring.h"

// Annotation for parameters that may be NULL or zero. No-op macro that
// improves readability at call sites and in generated declarations.
#define loom_optional

// Annotation for operand parameters that may be consumed by a tied result.
// The operand's storage may be reused by a result of this op (linear
// ownership transfer). After calling the builder, the annotated operand
// must not be used if it was tied to a result.
#define loom_may_consume

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Value slices
//===----------------------------------------------------------------------===//

// A typed range of value IDs. Returned by variadic operand/result accessors
// and used by generic pass infrastructure. One type for all variadic fields
// across all ops.
typedef struct loom_value_slice_t {
  loom_value_id_t* values;
  uint16_t count;
} loom_value_slice_t;

// Returns the value ID at |index| in the slice.
static inline loom_value_id_t loom_value_slice_get(loom_value_slice_t slice,
                                                   uint16_t index) {
  return slice.values[index];
}

// Sets the value ID at |index| in the slice.
static inline void loom_value_slice_set(loom_value_slice_t slice,
                                        uint16_t index, loom_value_id_t value) {
  slice.values[index] = value;
}

// Replaces all occurrences of |old_value| with |new_value| in the slice.
static inline void loom_value_slice_replace(loom_value_slice_t slice,
                                            loom_value_id_t old_value,
                                            loom_value_id_t new_value) {
  for (uint16_t i = 0; i < slice.count; ++i) {
    if (slice.values[i] == old_value) {
      slice.values[i] = new_value;
    }
  }
}

//===----------------------------------------------------------------------===//
// Format elements
//===----------------------------------------------------------------------===//

// Instructions for the format-element-walking printer and parser. The generic
// printer has one switch statement with ~17 cases. Each op's format element
// array is the instruction stream for that switch. Adding ops adds .rodata
// format arrays, not .text code.
typedef enum loom_format_kind_e {
  // Single operand reference: %name.
  LOOM_FORMAT_KIND_OPERAND_REF = 0,
  // Variadic operand references: %a, %b, %c.
  LOOM_FORMAT_KIND_OPERAND_REFS = 1,
  // Attribute value: 42, 3.14, slt, "hello".
  LOOM_FORMAT_KIND_ATTR_VALUE = 2,
  // Symbol reference attribute: @name.
  LOOM_FORMAT_KIND_SYMBOL_REF = 3,
  // Type of an operand: f32, tile<4xf32>.
  LOOM_FORMAT_KIND_OPERAND_TYPE = 4,
  // Type of a result: f32, tile<4xf32>.
  LOOM_FORMAT_KIND_RESULT_TYPE = 5,
  // Types of a variadic operand: f32, tile<4xf32>, i32.
  LOOM_FORMAT_KIND_OPERAND_TYPES = 6,
  // Result type list with tied handling: -> (type, %op as type).
  LOOM_FORMAT_KIND_RESULT_TYPE_LIST = 7,
  // Literal keyword token: , : -> to step else.
  LOOM_FORMAT_KIND_KEYWORD = 8,
  // Optional attribute dictionary: {key = value, ...}.
  LOOM_FORMAT_KIND_ATTR_DICT = 9,
  // Nested region: { block+ }.
  LOOM_FORMAT_KIND_REGION = 10,
  // Mixed static/dynamic index list: [0, %x, 4].
  // field_index = dynamic operand index, data = static attr index.
  LOOM_FORMAT_KIND_INDEX_LIST = 11,
  // Named value bindings: (%a = %x : type, ...).
  // data = binding kind (CAPTURE or ELEMENT).
  LOOM_FORMAT_KIND_BINDING_LIST = 12,
  // Function argument definitions: (%a: type, %b: type).
  LOOM_FORMAT_KIND_FUNC_ARGS = 13,
  // Where-clause predicates: [mul(%M, 16), ...].
  LOOM_FORMAT_KIND_PREDICATE_LIST = 14,
  // Optional group marker. field_index = anchor field index.
  // data = (skip_count << 2) | anchor_category.
  // The walker skips |skip_count| elements when the anchor is absent.
  LOOM_FORMAT_KIND_OPTIONAL_GROUP = 15,
  // Suppress space before the next token.
  LOOM_FORMAT_KIND_GLUE = 16,
  // Per-instance flags in angle brackets: <flag1|flag2>.
  // Glued to the preceding token (op name). field_index indexes into
  // the vtable's instance_flags_case_names. Reads/writes op->instance_flags.
  LOOM_FORMAT_KIND_FLAGS = 17,

  // Op kind reference in angle brackets: <tile.contract>.
  // Glued to the preceding token (op name). The field_index references
  // a string attribute storing the op name.
  LOOM_FORMAT_KIND_OP_REF = 18,

  // Single result type without parentheses: type.
  // For ops with exactly one non-variadic result where parenthesized
  // list syntax would be misleading. No tied-result support — use
  // RESULT_TYPE_LIST for ops that need tied-result syntax.
  LOOM_FORMAT_KIND_RESULT_TYPE_SINGLE = 19,

  // Scoped group of format elements. Pushes a new name scope before
  // processing children and pops it after. Within the scope, type
  // parsing uses definition mode: [%name] creates new index-typed
  // values rather than requiring existing names. Used for global
  // definitions where type annotations introduce named type variables.
  // data = child_count (number of following format elements in scope).
  LOOM_FORMAT_KIND_SCOPE = 20,
};
typedef uint8_t loom_format_kind_t;

// A 4-byte printer/parser instruction. An op with 12 format elements uses
// 48 bytes of .rodata. For 200 ops, total format tables are ~10KB.
//
// The kind field determines the instruction. The field_index selects which
// operand, result, attribute, or region this element references. The data
// field is kind-specific:
//   KEYWORD:        keyword ID (loom_keyword_id_t).
//   INDEX_LIST:     static attribute field index.
//   BINDING_LIST:   binding kind (CAPTURE=0, ELEMENT=1).
//   OPTIONAL_GROUP: (skip_count << 2) | anchor_category.
typedef struct loom_format_element_t {
  loom_format_kind_t kind;
  uint8_t field_index;
  uint16_t data;
} loom_format_element_t;

static_assert(sizeof(loom_format_element_t) == 4,
              "loom_format_element_t must be exactly 4 bytes");

// Data field flags for RESULT_TYPE_LIST elements. Stored in the
// element's data field.
enum loom_result_type_list_flag_bits_e {
  // Wrap result types in parentheses: (type, type).
  // When clear, result types are bare: type, type.
  LOOM_RESULT_TYPE_LIST_PARENS = 1u << 0,
};

// Anchor categories for OPTIONAL_GROUP elements. Encoded in the low
// 2 bits of the data field. Tells the format walker what kind of field
// to check for presence.
enum loom_anchor_category_e {
  // Variadic operand: present if operand_count > fixed_operand_count.
  LOOM_ANCHOR_OPERAND = 0,
  // Optional attribute: present if attribute value is non-zero.
  LOOM_ANCHOR_ATTR = 1,
  // Region: present if region pointer is non-null.
  LOOM_ANCHOR_REGION = 2,
  // Results: present if result_count > 0.
  LOOM_ANCHOR_RESULTS = 3,
};

//===----------------------------------------------------------------------===//
// Keywords
//===----------------------------------------------------------------------===//

// Format keywords (punctuation + text). All format specs across all dialects
// reference keywords by ID. Generated from KEYWORD_MAP in c_tables.py —
// do not edit manually. Append new keywords to KEYWORD_MAP and regenerate.
typedef enum loom_keyword_id_e {
#include "loom/ops/keyword_enum.inc"
  LOOM_KW_COUNT_,
} loom_keyword_id_t;

// Keyword B-string table indexed by loom_keyword_id_t.
// E.g., LOOM_KW_TO → "\x02to", LOOM_KW_STEP → "\x04step".
extern const loom_bstring_t loom_keyword_bstrings[LOOM_KW_COUNT_];

//===----------------------------------------------------------------------===//
// Type constraints, traits, and binding kinds
//===----------------------------------------------------------------------===//

// Abstract type categories for operand/result descriptors. The verifier
// checks that the actual type of a value satisfies the constraint.
typedef enum loom_type_constraint_e {
  LOOM_TYPE_CONSTRAINT_TILE = 0,
  LOOM_TYPE_CONSTRAINT_TENSOR = 1,
  LOOM_TYPE_CONSTRAINT_INTEGER = 2,
  LOOM_TYPE_CONSTRAINT_FLOAT = 3,
  LOOM_TYPE_CONSTRAINT_SCALAR = 4,
  LOOM_TYPE_CONSTRAINT_INDEX = 5,
  LOOM_TYPE_CONSTRAINT_ANY = 6,
  LOOM_TYPE_CONSTRAINT_GROUP = 7,
  LOOM_TYPE_CONSTRAINT_ENCODING = 8,
  LOOM_TYPE_CONSTRAINT_POOL = 9,
  // Exactly i1. Used for comparison results and boolean predicates.
  // Unlike INTEGER (which accepts any integer width), this requires
  // the type to be specifically i1.
  LOOM_TYPE_CONSTRAINT_I1 = 10,
  LOOM_TYPE_CONSTRAINT_COUNT_,
} loom_type_constraint_t;

// Returns the display name for a type constraint (e.g., "tile", "integer").
const char* loom_type_constraint_name(loom_type_constraint_t constraint);

// Returns true if |type| satisfies the abstract |constraint|.
// Used by the verifier for operand/result type checking and by
// builders for debug-mode assertions.
static inline bool loom_type_satisfies_constraint(
    loom_type_t type, loom_type_constraint_t constraint) {
  switch (constraint) {
    case LOOM_TYPE_CONSTRAINT_ANY:
      return true;
    case LOOM_TYPE_CONSTRAINT_TILE:
      return loom_type_is_tile(type);
    case LOOM_TYPE_CONSTRAINT_TENSOR:
      return loom_type_is_tensor(type);
    case LOOM_TYPE_CONSTRAINT_SCALAR:
      return loom_type_is_scalar(type);
    case LOOM_TYPE_CONSTRAINT_INDEX:
      return loom_type_is_scalar(type) &&
             loom_type_element_type(type) == LOOM_SCALAR_TYPE_INDEX;
    case LOOM_TYPE_CONSTRAINT_INTEGER:
      return loom_type_is_scalar(type) &&
             loom_scalar_type_is_integer(loom_type_element_type(type));
    case LOOM_TYPE_CONSTRAINT_FLOAT:
      return loom_type_is_scalar(type) &&
             loom_scalar_type_is_float(loom_type_element_type(type));
    case LOOM_TYPE_CONSTRAINT_GROUP:
      return loom_type_kind(type) == LOOM_TYPE_GROUP;
    case LOOM_TYPE_CONSTRAINT_ENCODING:
      return loom_type_is_encoding(type);
    case LOOM_TYPE_CONSTRAINT_POOL:
      return loom_type_is_pool(type);
    case LOOM_TYPE_CONSTRAINT_I1:
      return loom_type_is_scalar(type) &&
             loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1;
    default:
      return false;
  }
}

//===----------------------------------------------------------------------===//
// Semantic constraints
//===----------------------------------------------------------------------===//
//
// Constraints express relationships between an op's fields that the
// verifier checks. Each op's vtable points to an array of constraint
// entries. The verifier walks the array, interpreting each by kind.
//
// Per-op cost: 16 bytes .rodata per constraint, zero .text. The
// interpreter is one switch statement (~12 cases). Adding constraint
// kinds adds a case to the switch, not code per op.
//
// Field reference categories.
enum loom_field_category_e {
  LOOM_FIELD_OPERAND = 0,
  LOOM_FIELD_RESULT = 1,
  LOOM_FIELD_ATTR = 2,
  LOOM_FIELD_REGION = 3,
};

// Packs a (category, index) pair into a field reference.
#define LOOM_FIELD_REF(category, index) \
  ((loom_field_ref_t)(((category) << 6) | ((index) & 0x3F)))

// Extracts the category (0-3) from a packed field reference.
#define LOOM_FIELD_REF_CATEGORY(ref) ((ref) >> 6)

// Extracts the index (0-63) from a packed field reference.
#define LOOM_FIELD_REF_INDEX(ref) ((ref) & 0x3F)

// Constraint kinds for the table-driven constraint interpreter.
// Constraint relation: how values are compared.
enum loom_constraint_relation_e {
  LOOM_RELATION_PAIRWISE_EQ = 0,
  LOOM_RELATION_ALL_SAME = 1,
  LOOM_RELATION_COUNT_MATCHES_RANK = 2,
  LOOM_RELATION_ATTR_IN_RANGE_RANK = 3,
  LOOM_RELATION_REGION_ARG_COUNT = 4,
  LOOM_RELATION_REGION_ARG_MATCH = 5,
  LOOM_RELATION_YIELD_COUNT = 6,
  LOOM_RELATION_YIELD_MATCH = 7,
  LOOM_RELATION_COUNT_,
};
typedef uint8_t loom_constraint_relation_t;

// Constraint property: which aspect of a type to compare.
enum loom_constraint_property_e {
  LOOM_PROPERTY_TYPE = 0,
  LOOM_PROPERTY_ELEMENT_TYPE = 1,
  LOOM_PROPERTY_ENCODING = 2,
  LOOM_PROPERTY_SHAPE = 3,
  LOOM_PROPERTY_RANK = 4,
  LOOM_PROPERTY_COUNT_,
};
typedef uint8_t loom_constraint_property_t;

const char* loom_constraint_relation_name(loom_constraint_relation_t relation);
const char* loom_constraint_property_name(loom_constraint_property_t property);

// Forward-declared error definition. Defined in error/error_defs.h.
typedef struct loom_error_def_t loom_error_def_t;

// A table-driven semantic constraint entry. 16 bytes.
//
// Each op's vtable points to an array of these. The verifier walks
// the array, interpreting each constraint by (relation, property).
// Per-op cost: 16 bytes .rodata per constraint, zero .text code.
typedef struct loom_constraint_t {
  loom_constraint_relation_t relation;
  loom_constraint_property_t property;
  uint8_t arg_count;
  uint8_t reserved;
  loom_field_ref_t args[4];
  const loom_error_def_t* error;
} loom_constraint_t;

static_assert(sizeof(loom_constraint_t) == 16,
              "loom_constraint_t must be 16 bytes");

//===----------------------------------------------------------------------===//
// Field descriptors
//===----------------------------------------------------------------------===//

enum loom_operand_flag_bits_e {
  LOOM_OPERAND_VARIADIC = 1u << 0,
  LOOM_OPERAND_OPTIONAL = 1u << 1,
  LOOM_OPERAND_READS = 1u << 2,
  LOOM_OPERAND_WRITES = 1u << 3,
};
typedef uint8_t loom_operand_flags_t;

enum loom_result_flag_bits_e {
  LOOM_RESULT_VARIADIC = 1u << 0,
  LOOM_RESULT_ALLOCATES = 1u << 1,
};
typedef uint8_t loom_result_flags_t;

enum loom_attr_flag_bits_e {
  LOOM_ATTR_OPTIONAL = 1u << 0,
};
typedef uint8_t loom_attr_flags_t;

enum loom_region_flag_bits_e {
  LOOM_REGION_SINGLE_BLOCK = 1u << 0,
};
typedef uint8_t loom_region_flags_t;

// Per-operand metadata in the op vtable.
typedef struct loom_operand_descriptor_t {
  loom_type_constraint_t type_constraint;
  loom_operand_flags_t flags;
} loom_operand_descriptor_t;

// Per-result metadata in the op vtable.
typedef struct loom_result_descriptor_t {
  loom_type_constraint_t type_constraint;
  loom_result_flags_t flags;
} loom_result_descriptor_t;

// Per-attribute metadata in the op vtable.
typedef struct loom_attr_descriptor_t {
  loom_bstring_t name;
  loom_attr_kind_t attr_kind;
  loom_attr_flags_t flags;
  uint8_t enum_case_count;
  const loom_bstring_t* enum_case_names;
} loom_attr_descriptor_t;

// Returns the attribute name as a string view.
static inline iree_string_view_t loom_attr_descriptor_name(
    const loom_attr_descriptor_t* descriptor) {
  return loom_bstring_view(descriptor->name);
}

// Per-region metadata in the op vtable.
typedef struct loom_region_descriptor_t {
  loom_region_flags_t flags;
} loom_region_descriptor_t;

// Binding kind for BindingList format elements.
typedef enum loom_binding_kind_e {
  // Block arg has the same type as the operand.
  LOOM_BINDING_CAPTURE = 0,
  // Block arg has the element type of the operand.
  LOOM_BINDING_ELEMENT = 1,
} loom_binding_kind_t;

//===----------------------------------------------------------------------===//
// Value dereference helpers
//===----------------------------------------------------------------------===//

// Resolves an op's operand value ID to the value struct in the module's
// value table. |index| is the operand position (0-based).
//
// Usage:
//   loom_value_t* lhs = loom_op_operand_value(module, addi_op, 0);
//   if (loom_type_kind(lhs->type) == LOOM_TYPE_TILE) { ... }
static inline loom_value_t* loom_op_operand_value(const loom_module_t* module,
                                                  const loom_op_t* op,
                                                  uint16_t index) {
  return &module->values.entries[loom_op_operands(op)[index]];
}

// Resolves an op's result value ID to the value struct in the module's
// value table. |index| is the result position (0-based).
//
// Usage:
//   loom_value_t* result = loom_op_result_value(module, addi_op, 0);
//   if (result->use_count == 0) { /* dead result, candidate for DCE */ }
static inline loom_value_t* loom_op_result_value(const loom_module_t* module,
                                                 const loom_op_t* op,
                                                 uint16_t index) {
  return &module->values.entries[loom_op_results(op)[index]];
}

// Returns a pointer to the single use entry if the value has exactly
// one use, or NULL if it has zero or more than one use. The returned
// pointer is valid until the next use-list mutation on this value.
//
// Usage (fusion pattern — "does this tile feed exactly one consumer?"):
//   const loom_use_t* use = loom_value_single_use(tile_value);
//   if (use && loom_test_map_isa(loom_use_user_op(*use))) {
//     // Fuse into the map.
//   }
static inline const loom_use_t* loom_value_single_use(
    const loom_value_t* value) {
  if (value->use_count != 1) return NULL;
  return &loom_value_uses(value)[0];
}

//===----------------------------------------------------------------------===//
// Effect query helpers
//===----------------------------------------------------------------------===//

// Returns the effective trait flags for |op|, incorporating per-instance
// state via the vtable's effective_traits callback when present. Falls
// back to the vtable's static traits when no callback is registered.
// Returns 0 (no traits) when the vtable cannot be resolved.
loom_trait_flags_t loom_op_effective_traits(const loom_module_t* module,
                                            const loom_op_t* op);

// Returns true if |op| may write to a resource or has unknown effects.
bool loom_op_may_write(const loom_module_t* module, const loom_op_t* op);

// Returns true if every result of |op| has zero uses.
bool loom_op_results_unused(const loom_module_t* module, const loom_op_t* op);

// Returns true if |op| is trivially dead: it has results, does not
// write to any resource, has no unknown effects, and every result is
// unused. Read-only and non-deterministic ops without writes are dead
// when unused — a read with no observer is a no-op.
bool loom_op_is_trivially_dead(const loom_module_t* module,
                               const loom_op_t* op);

//===----------------------------------------------------------------------===//
// FuncLike interface helpers
//===----------------------------------------------------------------------===//

// Returns true if |func| refers to a valid func-like op. A cast via
// loom_func_like_cast() returns {NULL, NULL} on failure. All accessor
// helpers below tolerate a NULL vtable and return safe defaults (NULL,
// 0, or empty) so callers do not need to check loom_func_like_isa()
// before every call.
static inline bool loom_func_like_isa(loom_func_like_t func) {
  return func.op != NULL;
}

// Casts |op| to loom_func_like_t if it implements the FuncLike interface.
// Returns {NULL, NULL} if |op| is NULL or does not implement it. Safe to
// call unconditionally — callers check the result with loom_func_like_isa().
loom_func_like_t loom_func_like_cast(const loom_module_t* module,
                                     loom_op_t* op);

// Returns the body region of a func-like op, or NULL for bodyless ops
// (func.decl, func.ukernel) or if |func| is not valid.
static inline loom_region_t* loom_func_like_body(loom_func_like_t func) {
  if (!func.vtable) return NULL;
  if (func.vtable->body_region_index == LOOM_REGION_INDEX_NONE) return NULL;
  return loom_op_regions(func.op)[func.vtable->body_region_index];
}

// Returns the purity attr value (0 = unspecified, nonzero = pure).
static inline uint8_t loom_func_like_purity(loom_func_like_t func) {
  if (!func.vtable) return 0;
  if (func.vtable->purity_attr_index == LOOM_ATTR_INDEX_NONE) return 0;
  return loom_attr_as_enum(
      loom_op_attrs(func.op)[func.vtable->purity_attr_index]);
}

// Returns the visibility attr value (0 = private, nonzero = public).
static inline uint8_t loom_func_like_visibility(loom_func_like_t func) {
  if (!func.vtable) return 0;
  if (func.vtable->visibility_attr_index == LOOM_ATTR_INDEX_NONE) return 0;
  return loom_attr_as_enum(
      loom_op_attrs(func.op)[func.vtable->visibility_attr_index]);
}

// Returns the calling convention attr value (0 = default/host).
static inline uint8_t loom_func_like_cc(loom_func_like_t func) {
  if (!func.vtable) return 0;
  if (func.vtable->cc_attr_index == LOOM_ATTR_INDEX_NONE) return 0;
  return loom_attr_as_enum(loom_op_attrs(func.op)[func.vtable->cc_attr_index]);
}

// Returns the callee symbol ref for a func-like op, or {0, 0} if
// |func| is not valid.
static inline loom_symbol_ref_t loom_func_like_callee(loom_func_like_t func) {
  if (!func.vtable) return (loom_symbol_ref_t){0};
  return loom_attr_as_symbol(
      loom_op_attrs(func.op)[func.vtable->callee_attr_index]);
}

// Returns the function argument value IDs and their count. For ops
// with a body region, args are the entry block's block arguments.
// For declaration-style ops, args are stored as the op's operands.
// Returns NULL and sets |out_count| to 0 if |func| is not valid.
static inline const loom_value_id_t* loom_func_like_arg_ids(
    loom_func_like_t func, uint16_t* out_count) {
  if (!func.vtable) {
    *out_count = 0;
    return NULL;
  }
  if (!func.vtable->args_as_operands) {
    loom_region_t* body = loom_func_like_body(func);
    if (body && body->block_count > 0) {
      loom_block_t* entry = loom_region_entry_block(body);
      *out_count = entry->arg_count;
      return entry->arg_ids;
    }
    *out_count = 0;
    return NULL;
  }
  *out_count = func.op->operand_count;
  return loom_op_operands(func.op);
}

// Returns the predicate list and count for a func-like op. Sets |out_count|
// to 0 and returns NULL for ops with no predicate list attr or if |func| is
// not valid.
static inline const loom_predicate_t* loom_func_like_predicates(
    loom_func_like_t func, uint16_t* out_count) {
  if (!func.vtable) {
    *out_count = 0;
    return NULL;
  }
  if (func.vtable->predicates_attr_index == LOOM_ATTR_INDEX_NONE) {
    *out_count = 0;
    return NULL;
  }
  loom_attribute_t attr =
      loom_op_attrs(func.op)[func.vtable->predicates_attr_index];
  *out_count = attr.count;
  return attr.predicate_list;
}

// Returns the implements string ID for template/ukernel ops — the name of the
// op kind this function provides an implementation for. Returns
// LOOM_STRING_ID_INVALID for def/decl ops, ops with no implements attr, or
// if |func| is not valid.
static inline loom_string_id_t loom_func_like_implements(
    loom_func_like_t func) {
  if (!func.vtable) return LOOM_STRING_ID_INVALID;
  if (func.vtable->implements_attr_index == LOOM_ATTR_INDEX_NONE) {
    return LOOM_STRING_ID_INVALID;
  }
  return loom_attr_as_string_id(
      loom_op_attrs(func.op)[func.vtable->implements_attr_index]);
}

// Returns the dispatch priority for template/ukernel ops. Returns 0 for
// def/decl ops, ops with no priority attr, or if |func| is not valid.
static inline int64_t loom_func_like_priority(loom_func_like_t func) {
  if (!func.vtable) return 0;
  if (func.vtable->priority_attr_index == LOOM_ATTR_INDEX_NONE) return 0;
  return loom_attr_as_i64(
      loom_op_attrs(func.op)[func.vtable->priority_attr_index]);
}

//===----------------------------------------------------------------------===//
// Op definition macros
//===----------------------------------------------------------------------===//
//
// Generated per-dialect ops.h files use these macros to define typed
// inline accessor functions for each op.
//
// Usage in generated ops.h:
//
//   // LOOM_OP_TEST_ADDI: Test binary integer op.
//   // %result = test.addi %lhs, %rhs : type
//   LOOM_DEFINE_ISA(loom_test_addi_isa, LOOM_OP_TEST_ADDI)
//   LOOM_DEFINE_OPERAND(loom_test_addi_lhs, 0)
//   LOOM_DEFINE_OPERAND(loom_test_addi_rhs, 1)
//   LOOM_DEFINE_RESULT(loom_test_addi_result, 0)

// Defines a function that checks if an op is of a specific kind.
#define LOOM_DEFINE_ISA(func_name, kind_enum)         \
  static inline bool func_name(const loom_op_t* op) { \
    return op->kind == (kind_enum);                   \
  }

// Defines a function that reads a fixed operand by index.
#define LOOM_DEFINE_OPERAND(func_name, index)                    \
  static inline loom_value_id_t func_name(const loom_op_t* op) { \
    return loom_op_operands(op)[(index)];                        \
  }

// Defines a function that reads a fixed result by index.
#define LOOM_DEFINE_RESULT(func_name, index)                     \
  static inline loom_value_id_t func_name(const loom_op_t* op) { \
    return loom_op_results(op)[(index)];                         \
  }

// Defines a function that returns the variadic operand tail as a value
// slice. |fixed_count| is the number of non-variadic operands before
// the variadic tail.
#define LOOM_DEFINE_VARIADIC_OPERANDS(func_name, fixed_count)       \
  static inline loom_value_slice_t func_name(const loom_op_t* op) { \
    loom_value_slice_t slice;                                       \
    slice.values = loom_op_operands(op) + (fixed_count);            \
    slice.count = (uint16_t)(op->operand_count - (fixed_count));    \
    return slice;                                                   \
  }

// Defines a function that returns the variadic result tail as a value
// slice. |fixed_count| is the number of non-variadic results before
// the variadic tail.
#define LOOM_DEFINE_VARIADIC_RESULTS(func_name, fixed_count)        \
  static inline loom_value_slice_t func_name(const loom_op_t* op) { \
    loom_value_slice_t slice;                                       \
    slice.values = loom_op_results(op) + (fixed_count);             \
    slice.count = (uint16_t)(op->result_count - (fixed_count));     \
    return slice;                                                   \
  }

// Defines a function that reads a region by index.
#define LOOM_DEFINE_REGION(func_name, index)                    \
  static inline loom_region_t* func_name(const loom_op_t* op) { \
    return loom_op_regions(op)[(index)];                        \
  }

// Each LOOM_DEFINE_ATTR_* macro defines both a typed accessor function
// and a compile-time constant for the attribute's index in the attr
// array: func_name##_ATTR_INDEX. This lets canonicalize callbacks use
// loom_rewriter_set_attr(rewriter, op, loom_foo_bar_ATTR_INDEX, value)
// without hardcoding magic numbers.

// Defines a function that reads an i64 attribute by index.
#define LOOM_DEFINE_ATTR_I64(func_name, index)           \
  enum { func_name##_ATTR_INDEX = (index) };             \
  static inline int64_t func_name(const loom_op_t* op) { \
    return loom_attr_as_i64(loom_op_attrs(op)[(index)]); \
  }

// Defines a function that reads an f64 attribute by index.
#define LOOM_DEFINE_ATTR_F64(func_name, index)           \
  enum { func_name##_ATTR_INDEX = (index) };             \
  static inline double func_name(const loom_op_t* op) {  \
    return loom_attr_as_f64(loom_op_attrs(op)[(index)]); \
  }

// Defines a function that reads an enum attribute by index.
// Returns the enum case index as a uint8_t.
#define LOOM_DEFINE_ATTR_ENUM(func_name, index)           \
  enum { func_name##_ATTR_INDEX = (index) };              \
  static inline uint8_t func_name(const loom_op_t* op) {  \
    return loom_attr_as_enum(loom_op_attrs(op)[(index)]); \
  }

// Defines a function that reads a symbol attribute by index.
#define LOOM_DEFINE_ATTR_SYMBOL(func_name, index)                  \
  enum { func_name##_ATTR_INDEX = (index) };                       \
  static inline loom_symbol_ref_t func_name(const loom_op_t* op) { \
    return loom_attr_as_symbol(loom_op_attrs(op)[(index)]);        \
  }

// Defines a function that reads a string attribute by index.
#define LOOM_DEFINE_ATTR_STRING(func_name, index)                 \
  enum { func_name##_ATTR_INDEX = (index) };                      \
  static inline loom_string_id_t func_name(const loom_op_t* op) { \
    return loom_attr_as_string_id(loom_op_attrs(op)[(index)]);    \
  }

// Defines a function that reads a bool attribute by index.
#define LOOM_DEFINE_ATTR_BOOL(func_name, index)           \
  enum { func_name##_ATTR_INDEX = (index) };              \
  static inline bool func_name(const loom_op_t* op) {     \
    return loom_attr_as_bool(loom_op_attrs(op)[(index)]); \
  }

// Defines a function that reads the per-instance flags byte.
// Used for fast-math flags (float ops) and overflow flags (integer ops).
#define LOOM_DEFINE_INSTANCE_FLAGS(func_name)            \
  static inline uint8_t func_name(const loom_op_t* op) { \
    return op->instance_flags;                           \
  }

//===----------------------------------------------------------------------===//
// Builder
//===----------------------------------------------------------------------===//

// Saved insertion point for nested region construction. Stack-allocated.
// Use loom_builder_save/loom_builder_restore to switch between blocks.
// The parent_op field is saved/restored so that ops created inside
// nested regions get the correct ancestry on their parent_op pointer.
typedef struct loom_builder_ip_t {
  // Block to insert into.
  loom_block_t* block;
  // Op whose region contains |block|. NULL at the module level.
  // Stamped onto every op created at this insertion point as
  // op->parent_op, giving O(depth) ancestry queries.
  loom_op_t* parent_op;
  // Position within the block's ops array. Ops are inserted before
  // this index. UINT16_MAX means append to the end of the block.
  uint16_t index;
} loom_builder_ip_t;

// Callback invoked when an op is fully constructed (after finalize_op).
// Used by the rewriter to add newly created ops to its worklist.
typedef iree_status_t (*loom_builder_op_fn_t)(void* user_data, loom_op_t* op);
typedef struct loom_builder_callback_t {
  loom_builder_op_fn_t fn;
  void* user_data;
} loom_builder_callback_t;

// The builder is the single API surface for IR construction.
//
// It bundles the target module (for value/string/type tables), the
// allocation arena, and the current insertion point into one object.
// Every generated op builder takes a loom_builder_t* as its first
// parameter. Generic helpers (inlining, cloning, pattern rewriting)
// take a builder without knowing the source or target module.
//
// The arena is separate from the module's arena so that callers can
// control storage lifetime: linking into a fresh arena, per-thread
// arenas during parallel compilation, etc.
//
// The optional on_op_finalized callback fires after finalize_op
// completes (uses registered, def pointers set). NULL when unused.
typedef struct loom_builder_t {
  loom_module_t* module;
  iree_arena_allocator_t* arena;
  loom_builder_ip_t ip;
  loom_builder_callback_t on_op_finalized;
  // Pre-allocated result value_ids for the next op build. When
  // reserved_result_count > 0, loom_builder_define_value consumes
  // from this array instead of allocating new value_ids. Cleared
  // by loom_builder_finalize_op after verifying all were consumed.
  const loom_value_id_t* reserved_result_ids;
  iree_host_size_t reserved_result_count;
  iree_host_size_t reserved_result_next;
} loom_builder_t;

// Initializes a builder that appends to |block|.
void loom_builder_initialize(loom_module_t* module,
                             iree_arena_allocator_t* arena, loom_block_t* block,
                             loom_builder_t* out_builder);

// Moves the insertion point to before |op| in the current block.
// Used during rewrites: insert replacement ops, then erase the original.
void loom_builder_set_before(loom_builder_t* builder, const loom_op_t* op);

// Moves the insertion point to after |op| in the current block.
void loom_builder_set_after(loom_builder_t* builder, const loom_op_t* op);

// Moves the insertion point to the end of |block| (append mode).
// Does not change parent_op — use loom_builder_enter_region when
// entering a nested region to ensure correct parent ancestry.
void loom_builder_set_block(loom_builder_t* builder, loom_block_t* block);

// Enters a nested region for building ops inside it. Saves the
// current insertion point (use loom_builder_restore to return),
// sets the insertion block to the region's entry block, and sets
// parent_op so that ops created inside inherit correct ancestry.
//
//   loom_builder_ip_t saved = loom_builder_enter_region(
//       &builder, parent_op, region);
//   // ... build ops inside region ...
//   loom_builder_restore(&builder, saved);
loom_builder_ip_t loom_builder_enter_region(loom_builder_t* builder,
                                            loom_op_t* parent_op,
                                            loom_region_t* region);

// Saves the current insertion point. Returns a value that can be
// passed to loom_builder_restore to return to this position.
loom_builder_ip_t loom_builder_save(const loom_builder_t* builder);

// Restores a previously saved insertion point.
void loom_builder_restore(loom_builder_t* builder, loom_builder_ip_t ip);

// Pre-allocates |count| result value_ids in the module's value table.
// The values are real entries with uninitialized types. The next |count|
// calls to loom_builder_define_value (typically from a generated builder)
// will assign types to these values instead of allocating fresh ones.
// loom_builder_finalize_op verifies all reserved results were consumed.
//
// This enables constructing result types that reference other results
// by value_id before the build call:
//
//   loom_value_id_t result_ids[2];
//   loom_builder_reserve_results(&builder, 2, result_ids);
//   loom_type_t output_type = loom_type_shaped_1d(
//       LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32,
//       loom_dim_pack_dynamic(result_ids[1]), 0);
//   loom_type_t result_types[] = {output_type, index_type};
//   loom_test_deflate_build(&builder, input, result_types, 2, ...);
//
iree_status_t loom_builder_reserve_results(loom_builder_t* builder,
                                           iree_host_size_t count,
                                           loom_value_id_t* out_result_ids);

// Creates a fresh value in the module's value table with the given type.
// Returns the value ID. The value has no defining op yet (set by the
// builder when the op is inserted). If results were reserved via
// loom_builder_reserve_results, consumes the next reserved id and
// assigns the type to it.
iree_status_t loom_builder_define_value(loom_builder_t* builder,
                                        loom_type_t type,
                                        loom_value_id_t* out_value_id);

// Interns a string in the module's string table. Returns the string ID.
// Identical strings share the same ID.
iree_status_t loom_builder_intern_string(loom_builder_t* builder,
                                         iree_string_view_t string,
                                         loom_string_id_t* out_string_id);

// Creates a fresh value with the given type and adds it as a block
// argument. Convenience wrapper for the define_value + block_add_arg
// sequence that generated builders use when auto-creating regions.
iree_status_t loom_builder_define_block_arg(loom_builder_t* builder,
                                            loom_block_t* block,
                                            loom_type_t type,
                                            loom_value_id_t* out_value_id);

// Allocates an op with the given field counts and inserts it at the
// builder's current insertion point. This is the low-level primitive
// that generated builders call. The caller fills in trailing data
// (operands, results, regions, tied results, attributes) through the
// accessor functions above.
iree_status_t loom_builder_allocate_op(
    loom_builder_t* builder, loom_op_kind_t kind, uint16_t operand_count,
    uint16_t result_count, uint8_t region_count, uint16_t tied_result_count,
    uint8_t attribute_count, loom_location_id_t location, loom_op_t** out_op);

// Erases an op: removes all use records for the op's operands, verifies
// that every result has use_count == 0 (caller must RAUW results first),
// then marks the op dead. Dead ops are skipped by enumeration macros
// and will not be serialized. The memory is not freed (arena-owned).
// Returns IREE_STATUS_FAILED_PRECONDITION if any result still has uses.
iree_status_t loom_op_erase(loom_module_t* module, loom_op_t* op);

//===----------------------------------------------------------------------===//
// Use-def list maintenance
//===----------------------------------------------------------------------===//
//
// These functions keep value use lists always-correct. All operand
// mutations must go through these functions (or the builder finalize
// path) to maintain the invariant that every operand has a corresponding
// use entry on the referenced value.

// Adds a use record: |user_op| uses value |value_id| at |operand_index|.
// Handles inline-to-overflow transition via arena allocation on the module.
iree_status_t loom_value_add_use(loom_module_t* module,
                                 loom_value_id_t value_id, loom_op_t* user_op,
                                 uint16_t operand_index);

// Removes a use record: |user_op| no longer uses |value_id| at
// |operand_index|. Scans the use list for the matching entry, swaps
// with last, decrements use_count. Returns IREE_STATUS_NOT_FOUND if
// no matching entry exists (indicates a use-list bookkeeping bug).
// No overflow-to-inline transition (arena cannot free the overflow
// array; loom_module_compute_uses handles repack).
iree_status_t loom_value_remove_use(loom_module_t* module,
                                    loom_value_id_t value_id,
                                    loom_op_t* user_op, uint16_t operand_index);

// Finalizes a newly-built op: registers all operand uses and performs
// any other per-op bookkeeping. Called as the tail return from every
// builder: `return loom_builder_finalize_op(builder, *out_op);`
iree_status_t loom_builder_finalize_op(loom_builder_t* builder, loom_op_t* op);

// Links a symbol-defining op to its symbol table entry. Scans the
// op's attributes (via the vtable's attr descriptors) for the first
// LOOM_ATTR_SYMBOL, then sets defining_op and kind on the
// corresponding symbol entry. Idempotent.
void loom_module_link_symbol_defining_op(loom_module_t* module, loom_op_t* op,
                                         const loom_op_vtable_t* vtable);

// Changes an operand on an existing op, maintaining use lists. Removes
// the use from the old value, writes the new value ID, and adds a use
// to the new value. Skips LOOM_VALUE_ID_INVALID for both old and new.
iree_status_t loom_op_set_operand(loom_module_t* module, loom_op_t* op,
                                  uint16_t operand_index,
                                  loom_value_id_t new_value_id);

// Replaces all uses of |old_id| with |new_id|. Walks old's use list,
// patches each user op's operand slot, and bulk-transfers the use
// entries to new's list. No-op if old_id == new_id.
iree_status_t loom_value_replace_all_uses_with(loom_module_t* module,
                                               loom_value_id_t old_id,
                                               loom_value_id_t new_id);

// Same as replace_all_uses_with, but skips uses where the user op is
// |except_op|. Used during pattern rewrites where the replacement op
// also references the old value.
iree_status_t loom_value_replace_all_uses_except(loom_module_t* module,
                                                 loom_value_id_t old_id,
                                                 loom_value_id_t new_id,
                                                 const loom_op_t* except_op);

// Predicate-based RAUW. Replaces uses of |old_id| with |new_id| only
// where |predicate| returns true for the user op.
typedef bool (*loom_use_predicate_fn)(const loom_op_t* user_op,
                                      void* user_data);
iree_status_t loom_value_replace_uses_if(loom_module_t* module,
                                         loom_value_id_t old_id,
                                         loom_value_id_t new_id,
                                         loom_use_predicate_fn predicate,
                                         void* user_data);

// Rebuilds all use lists from scratch by walking every live op in the
// module. Clears all values' use data, then re-adds uses from operands.
// Used after parsing (the parser fills operands but not use lists) and
// as a recovery path after bulk IR mutations.
iree_status_t loom_module_compute_uses(loom_module_t* module);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_OP_DEFS_H_
