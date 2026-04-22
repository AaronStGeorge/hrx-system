// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Table-driven source-to-target-low lowering rules.
//
// Targets use these tables to describe the common case: source op guards over
// operands, results, attributes, and descriptor availability followed by one or
// more descriptor-backed low packet emissions. The interpreter owns the shared
// mechanics so target packages can grow generated .rodata instead of per-op
// callback dispatchers.

#ifndef LOOM_CODEGEN_LOW_LOWER_RULES_H_
#define LOOM_CODEGEN_LOW_LOWER_RULES_H_

#include "iree/base/api.h"
#include "loom/codegen/low/lower.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns a scalar element-type bit for type-pattern masks.
#define LOOM_LOW_LOWER_SCALAR_TYPE_BIT(type) (UINT64_C(1) << (uint32_t)(type))

// Bitset of fields checked by a type pattern.
typedef uint16_t loom_low_lower_type_pattern_flags_t;

// Type kind must match type_kind.
#define LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND ((uint16_t)1u << 0)
// Scalar or shaped element type must be in element_type_mask.
#define LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT ((uint16_t)1u << 1)
// Shaped rank must match rank.
#define LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_RANK ((uint16_t)1u << 2)
// First shaped dimension must be statically equal to static_dim0.
#define LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0 ((uint16_t)1u << 3)
// First shaped dimension must be inside [static_dim0_min, static_dim0_max].
#define LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0_RANGE ((uint16_t)1u << 4)

typedef struct loom_low_lower_type_pattern_t {
  // Type fields this pattern checks.
  loom_low_lower_type_pattern_flags_t flags;
  // Required type kind when the KIND flag is set.
  loom_type_kind_t type_kind;
  // Allowed element scalar types when the ELEMENT flag is set.
  uint64_t element_type_mask;
  // Required rank when the RANK flag is set.
  uint8_t rank;
  // Required static dimension 0 when the STATIC_DIM0 flag is set.
  int64_t static_dim0;
  // Inclusive minimum static dimension 0 when STATIC_DIM0_RANGE is set.
  int64_t static_dim0_min;
  // Inclusive maximum static dimension 0 when STATIC_DIM0_RANGE is set.
  int64_t static_dim0_max;
} loom_low_lower_type_pattern_t;

typedef enum loom_low_lower_value_ref_kind_e {
  // Invalid or uninitialized value reference.
  LOOM_LOW_LOWER_VALUE_REF_INVALID = 0,
  // Source op operand at |index|.
  LOOM_LOW_LOWER_VALUE_REF_OPERAND = 1,
  // Source op result at |index|.
  LOOM_LOW_LOWER_VALUE_REF_RESULT = 2,
  // Rule-local temporary low value at |index|.
  LOOM_LOW_LOWER_VALUE_REF_TEMPORARY = 3,
} loom_low_lower_value_ref_kind_t;

// Returns true when the materializer can produce a low value for the source
// value without emitting IR. Selection uses this to keep diagnostics tied to
// the same value-ref row consumed during emission.
typedef bool (*loom_low_lower_can_materialize_value_fn_t)(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id);

// Emits or returns the low value consumed by a descriptor operand for the
// source value. The callback only runs when a value-ref row explicitly names
// it.
typedef iree_status_t (*loom_low_lower_materialize_value_fn_t)(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id, loom_value_id_t* out_low_value_id);

typedef struct loom_low_lower_value_materializer_t {
  // Selection-time predicate proving this materializer can handle the source
  // value without emitting IR.
  loom_low_lower_can_materialize_value_fn_t can_materialize;
  // Emission-time callback that returns the low value used by descriptor ops.
  loom_low_lower_materialize_value_fn_t materialize;
} loom_low_lower_value_materializer_t;

typedef struct loom_low_lower_value_ref_t {
  // Source value namespace being referenced.
  loom_low_lower_value_ref_kind_t kind;
  // Operand or result ordinal in the source op.
  uint16_t index;
  // One-based materializer table row used when this source ref is consumed as a
  // low operand. Zero means direct source-to-low value lookup.
  uint16_t materializer_index;
} loom_low_lower_value_ref_t;

typedef struct loom_low_lower_attr_copy_t {
  // Target low packet attribute name to emit.
  iree_string_view_t target_name;
  // Source op attribute ordinal copied into the emitted low packet.
  uint16_t source_attr_index;
} loom_low_lower_attr_copy_t;

typedef struct loom_low_lower_diagnostic_t {
  // Diagnostic subject category, such as "type", "attr", or "descriptor".
  iree_string_view_t subject_kind;
  // Diagnostic subject name within subject_kind.
  iree_string_view_t subject_name;
  // Human-readable rejection reason.
  iree_string_view_t reason;
} loom_low_lower_diagnostic_t;

#define LOOM_LOW_LOWER_DIAGNOSTIC_NONE UINT16_MAX

typedef enum loom_low_lower_guard_kind_e {
  // Invalid or uninitialized guard.
  LOOM_LOW_LOWER_GUARD_INVALID = 0,
  // Source operand/result type must match a type pattern.
  LOOM_LOW_LOWER_GUARD_VALUE_TYPE = 1,
  // Source attribute kind must match attr_kind.
  LOOM_LOW_LOWER_GUARD_ATTR_KIND = 2,
  // Source enum attribute value must match u64.
  LOOM_LOW_LOWER_GUARD_ATTR_ENUM_EQ = 3,
  // Source i64 attribute value must fall in [minimum_i64, maximum_i64].
  LOOM_LOW_LOWER_GUARD_ATTR_I64_RANGE = 4,
  // Selected descriptor set must contain descriptor_id.
  LOOM_LOW_LOWER_GUARD_DESCRIPTOR_AVAILABLE = 5,
  // Source value ref must be accepted by its configured materializer.
  LOOM_LOW_LOWER_GUARD_VALUE_MATERIALIZABLE = 6,
  // Source value ref must map to a low register with register_class_id.
  LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_CLASS = 7,
} loom_low_lower_guard_kind_t;

typedef struct loom_low_lower_guard_t {
  // Guard operation to evaluate.
  loom_low_lower_guard_kind_t kind;
  // Source value-ref table index used by VALUE_TYPE guards.
  uint16_t value_ref_index;
  // Source attribute ordinal used by attribute guards.
  uint16_t attr_index;
  // Type-pattern table index used by VALUE_TYPE guards.
  uint16_t type_pattern_index;
  // Diagnostic table index emitted when this guard rejects.
  uint16_t diagnostic_index;
  // Required attribute kind for ATTR_KIND guards.
  loom_attr_kind_t attr_kind;
  // Required enum value for ATTR_ENUM_EQ guards.
  uint64_t u64;
  // Stable descriptor ID used by DESCRIPTOR_AVAILABLE guards.
  uint64_t descriptor_id;
  // Descriptor-set register-class ID used by LOW_VALUE_REGISTER_CLASS guards.
  uint16_t register_class_id;
  // Inclusive lower i64 bound for ATTR_I64_RANGE guards.
  int64_t minimum_i64;
  // Inclusive upper i64 bound for ATTR_I64_RANGE guards.
  int64_t maximum_i64;
} loom_low_lower_guard_t;

typedef enum loom_low_lower_emit_kind_e {
  // Invalid or uninitialized emit action.
  LOOM_LOW_LOWER_EMIT_INVALID = 0,
  // Emits a descriptor-backed low.op.
  LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP = 1,
  // Emits a descriptor-backed low.const with copied attributes.
  LOOM_LOW_LOWER_EMIT_DESCRIPTOR_CONST = 2,
  // Slices register-range operands, emits one descriptor-backed low.op per
  // register lane, and concatenates the lane results.
  LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE = 3,
  // Slices register-range operands, emits one descriptor-backed low.op per
  // register lane, and threads one scalar accumulator operand through the
  // emitted results.
  LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_ACCUMULATE_LANES = 4,
} loom_low_lower_emit_kind_t;

typedef uint16_t loom_low_lower_emit_flags_t;

// Swaps emitted descriptor operands 0 and 1 after operand lookup/copy/slicing.
#define LOOM_LOW_LOWER_EMIT_FLAG_SWAP_OPERANDS_0_1 ((uint16_t)1u << 0)
// Binds emitted low results to result_bind_ref_start instead of
// result_ref_start.
#define LOOM_LOW_LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS ((uint16_t)1u << 1)

typedef struct loom_low_lower_emit_t {
  // Emit action to perform.
  loom_low_lower_emit_kind_t kind;
  // Emit behavior flags.
  loom_low_lower_emit_flags_t flags;
  // Stable low descriptor ID consumed by the selected descriptor set.
  uint64_t descriptor_id;
  // First value-ref table row copied as a low operand.
  uint16_t operand_ref_start;
  // Number of low operands to copy from value-ref rows.
  uint16_t operand_ref_count;
  // Bitmask of emitted low operand ordinals copied through low.copy before the
  // descriptor op consumes them. Used for destructive/tied packet operands
  // without clobbering the source SSA value's later uses.
  uint16_t copy_operand_mask;
  // Operand ordinal that carries the threaded scalar accumulator for
  // DESCRIPTOR_OP_ACCUMULATE_LANES.
  uint16_t accumulator_operand_index;
  // First value-ref table row mapped as a low result.
  //
  // Result type refs must address source results. When
  // BIND_RESULTS_TO_REFS is set, result_bind_ref_start controls where the
  // emitted low results are bound.
  uint16_t result_ref_start;
  // Number of low results to map from value-ref rows.
  uint16_t result_ref_count;
  // First value-ref table row receiving emitted low results when
  // BIND_RESULTS_TO_REFS is set.
  uint16_t result_bind_ref_start;
  // First attr-copy table row emitted onto the low packet.
  uint16_t attr_copy_start;
  // Number of attributes copied onto the low packet.
  uint16_t attr_copy_count;
  // First tied-result table row forwarded to the low packet builder.
  uint16_t tied_result_start;
  // Number of tied-result rows forwarded to the low packet builder.
  uint16_t tied_result_count;
} loom_low_lower_emit_t;

typedef struct loom_low_lower_rule_t {
  // Source op kind this rule accepts.
  loom_op_kind_t source_op_kind;
  // Number of rule-local temporary low values available while emitting this
  // rule.
  uint16_t temporary_count;
  // First guard table row for this rule.
  uint16_t guard_start;
  // Number of guard rows for this rule.
  uint16_t guard_count;
  // First emit-program table row for this rule.
  uint16_t emit_start;
  // Number of emit-program rows for this rule.
  uint16_t emit_count;
} loom_low_lower_rule_t;

typedef struct loom_low_lower_rule_span_t {
  // Source op kind covered by this contiguous rule range.
  loom_op_kind_t source_op_kind;
  // First rule table row for source_op_kind.
  uint16_t rule_start;
  // Number of rules for source_op_kind.
  uint16_t rule_count;
} loom_low_lower_rule_span_t;

typedef struct loom_low_lower_rule_set_t {
  // Source op kind to rule-span lookup table sorted by source_op_kind.
  const loom_low_lower_rule_span_t* spans;
  // Number of rows in spans.
  uint16_t span_count;
  // Rule rows referenced by spans.
  const loom_low_lower_rule_t* rules;
  // Number of rows in rules.
  uint16_t rule_count;
  // Type-pattern rows referenced by guards.
  const loom_low_lower_type_pattern_t* type_patterns;
  // Number of rows in type_patterns.
  uint16_t type_pattern_count;
  // Source value-reference rows referenced by guards and emits.
  const loom_low_lower_value_ref_t* value_refs;
  // Number of rows in value_refs.
  uint16_t value_ref_count;
  // Target-owned value materializers referenced by one-based value refs.
  const loom_low_lower_value_materializer_t* materializers;
  // Number of rows in materializers.
  uint16_t materializer_count;
  // Guard rows referenced by rules.
  const loom_low_lower_guard_t* guards;
  // Number of rows in guards.
  uint16_t guard_count;
  // Attribute-copy rows referenced by emits.
  const loom_low_lower_attr_copy_t* attr_copies;
  // Number of rows in attr_copies.
  uint16_t attr_copy_count;
  // Tied-result rows referenced by emits.
  const loom_tied_result_t* tied_results;
  // Number of rows in tied_results.
  uint16_t tied_result_count;
  // Emit-program rows referenced by rules.
  const loom_low_lower_emit_t* emits;
  // Number of rows in emits.
  uint16_t emit_count;
  // Diagnostic rows referenced by guards.
  const loom_low_lower_diagnostic_t* diagnostics;
  // Number of rows in diagnostics.
  uint16_t diagnostic_count;
} loom_low_lower_rule_set_t;

typedef struct loom_low_lower_rule_selection_t {
  // Selected rule row, or NULL when no rule accepted the source op.
  const loom_low_lower_rule_t* rule;
  // True when the rule set had at least one rule span for the source op kind.
  bool has_source_op_span;
  // Diagnostic row describing the best failed guard when |rule| is NULL.
  uint16_t diagnostic_index;
  // Number of guards matched by the best failed rule candidate.
  uint16_t matched_guard_count;
} loom_low_lower_rule_selection_t;

// Selects the exact lowering rule for |source_op| without emitting user
// diagnostics. Callers that compose rule tables with custom target callbacks
// can use the recorded failure detail if every lowering path rejects the op.
iree_status_t loom_low_lower_rule_set_select(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_selection_t* out_selection);

// Emits the diagnostic described by a failed selection. If the rule set did not
// cover the source op kind, emits the generic no-mapping diagnostic.
iree_status_t loom_low_lower_rule_set_emit_selection_failure(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_selection_t selection);

// Selects the exact lowering rule for |source_op| and emits a user diagnostic
// when no rule accepts it. The selected rule is trusted generated data and must
// be executed unchanged by loom_low_lower_rule_set_emit_rule.
iree_status_t loom_low_lower_rule_set_select_op(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_t** out_rule);

// Emits target-low packets for |source_op| using a previously selected rule.
iree_status_t loom_low_lower_rule_set_emit_rule(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_t* rule);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_RULES_H_
