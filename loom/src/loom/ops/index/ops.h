// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables
// clang-format off

#ifndef LOOM_OPS_INDEX_OPS_H_
#define LOOM_OPS_INDEX_OPS_H_

#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_INDEX_CONSTANT = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 0),
  LOOM_OP_INDEX_CAST = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 1),
  LOOM_OP_INDEX_ASSUME = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 2),
  LOOM_OP_INDEX_ADD = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 3),
  LOOM_OP_INDEX_SUB = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 4),
  LOOM_OP_INDEX_MUL = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 5),
  LOOM_OP_INDEX_DIV = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 6),
  LOOM_OP_INDEX_REM = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 7),
  LOOM_OP_INDEX_MADD = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 8),
  LOOM_OP_INDEX_CMP = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 9),
  LOOM_OP_INDEX_COUNT_ = 10,
};

// Address-domain comparison predicates.
typedef enum loom_index_cmp_predicate_e {
  LOOM_INDEX_CMP_PREDICATE_EQ = 0,
  LOOM_INDEX_CMP_PREDICATE_NE = 1,
  LOOM_INDEX_CMP_PREDICATE_SLT = 2,
  LOOM_INDEX_CMP_PREDICATE_SLE = 3,
  LOOM_INDEX_CMP_PREDICATE_SGT = 4,
  LOOM_INDEX_CMP_PREDICATE_SGE = 5,
  LOOM_INDEX_CMP_PREDICATE_ULT = 6,
  LOOM_INDEX_CMP_PREDICATE_ULE = 7,
  LOOM_INDEX_CMP_PREDICATE_UGT = 8,
  LOOM_INDEX_CMP_PREDICATE_UGE = 9,
  LOOM_INDEX_CMP_PREDICATE_COUNT_ = 10,
} loom_index_cmp_predicate_t;

// LOOM_OP_INDEX_CONSTANT: Materialize a compile-time address-domain constant. The result type must be index for logical coordinates or offset for physical byte counts; fixed-width integer payload constants remain scalar.constant.
// %c0 = index.constant 0 : index
LOOM_DEFINE_ISA(loom_index_constant_isa, LOOM_OP_INDEX_CONSTANT)
LOOM_DEFINE_RESULT(loom_index_constant_result, 0)
LOOM_DEFINE_ATTR_ANY(loom_index_constant_value, 0)
iree_status_t loom_index_constant_build(
    loom_builder_t* builder,
    loom_attribute_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_index_constant_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_CAST: Explicit conversion at an address boundary. At least one side must be index or offset; pure integer width changes use scalar.extsi, scalar.extui, or scalar.trunci.
// %i = index.cast %n : i64 to index
LOOM_DEFINE_ISA(loom_index_cast_isa, LOOM_OP_INDEX_CAST)
LOOM_DEFINE_OPERAND(loom_index_cast_input, 0)
LOOM_DEFINE_RESULT(loom_index_cast_result, 0)
iree_status_t loom_index_cast_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_index_cast_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_cast_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_index_cast_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_INDEX_ASSUME: Identity with predicate constraints on index or offset results.
// %n2 = index.assume %n [mul(%n, 16)] : index
LOOM_DEFINE_ISA(loom_index_assume_isa, LOOM_OP_INDEX_ASSUME)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_index_assume_values, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_index_assume_results, 0)
iree_status_t loom_index_assume_build(
    loom_builder_t* builder,
    const loom_value_id_t* values,
    iree_host_size_t values_count,
    const loom_predicate_t* predicates,
    iree_host_size_t predicates_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_index_assume_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_index_assume_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_INDEX_ADD: Address-domain addition. Operands and result must all be index or all offset.
// %r = index.add %lhs, %rhs : index
LOOM_DEFINE_ISA(loom_index_add_isa, LOOM_OP_INDEX_ADD)
LOOM_DEFINE_OPERAND(loom_index_add_lhs, 0)
LOOM_DEFINE_OPERAND(loom_index_add_rhs, 1)
LOOM_DEFINE_RESULT(loom_index_add_result, 0)
iree_status_t loom_index_add_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_index_add_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_add_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_SUB: Address-domain subtraction. Operands and result must all be index or all offset.
// %r = index.sub %lhs, %rhs : index
LOOM_DEFINE_ISA(loom_index_sub_isa, LOOM_OP_INDEX_SUB)
LOOM_DEFINE_OPERAND(loom_index_sub_lhs, 0)
LOOM_DEFINE_OPERAND(loom_index_sub_rhs, 1)
LOOM_DEFINE_RESULT(loom_index_sub_result, 0)
iree_status_t loom_index_sub_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_index_sub_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_sub_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_MUL: Logical coordinate multiplication. Offsets are physical byte counts and cannot be multiplied with this op.
// %r = index.mul %lhs, %rhs : index
LOOM_DEFINE_ISA(loom_index_mul_isa, LOOM_OP_INDEX_MUL)
LOOM_DEFINE_OPERAND(loom_index_mul_lhs, 0)
LOOM_DEFINE_OPERAND(loom_index_mul_rhs, 1)
LOOM_DEFINE_RESULT(loom_index_mul_result, 0)
iree_status_t loom_index_mul_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_index_mul_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_mul_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_DIV: Logical coordinate quotient for non-negative index values with a positive divisor. Offsets are physical byte counts and cannot be divided with this op; use an explicit layout or storage mapping before deriving physical address pieces.
// %q = index.div %lane, %group_size : index
LOOM_DEFINE_ISA(loom_index_div_isa, LOOM_OP_INDEX_DIV)
LOOM_DEFINE_OPERAND(loom_index_div_lhs, 0)
LOOM_DEFINE_OPERAND(loom_index_div_rhs, 1)
LOOM_DEFINE_RESULT(loom_index_div_result, 0)
iree_status_t loom_index_div_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_index_div_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_div_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_REM: Logical coordinate remainder for non-negative index values with a positive divisor. Offsets are physical byte counts and cannot use remainder with this op; use an explicit layout or storage mapping before deriving physical address pieces.
// %r = index.rem %lane, %group_size : index
LOOM_DEFINE_ISA(loom_index_rem_isa, LOOM_OP_INDEX_REM)
LOOM_DEFINE_OPERAND(loom_index_rem_lhs, 0)
LOOM_DEFINE_OPERAND(loom_index_rem_rhs, 1)
LOOM_DEFINE_RESULT(loom_index_rem_result, 0)
iree_status_t loom_index_rem_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_index_rem_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_rem_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_MADD: Logical coordinate multiply-add: a*b + c. Offsets are physical byte counts and cannot be multiplied with this op.
// %r = index.madd %a, %b, %c : index
LOOM_DEFINE_ISA(loom_index_madd_isa, LOOM_OP_INDEX_MADD)
LOOM_DEFINE_OPERAND(loom_index_madd_a, 0)
LOOM_DEFINE_OPERAND(loom_index_madd_b, 1)
LOOM_DEFINE_OPERAND(loom_index_madd_c, 2)
LOOM_DEFINE_RESULT(loom_index_madd_result, 0)
iree_status_t loom_index_madd_build(
    loom_builder_t* builder,
    loom_value_id_t a,
    loom_value_id_t b,
    loom_value_id_t c,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_index_madd_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_madd_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_INDEX_CMP: Address-domain comparison. Operands must both be index or both be offset.
// %p = index.cmp slt, %i, %n : index
LOOM_DEFINE_ISA(loom_index_cmp_isa, LOOM_OP_INDEX_CMP)
LOOM_DEFINE_OPERAND(loom_index_cmp_lhs, 0)
LOOM_DEFINE_OPERAND(loom_index_cmp_rhs, 1)
LOOM_DEFINE_RESULT(loom_index_cmp_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_index_cmp_predicate, 0, loom_index_cmp_predicate_t)
iree_status_t loom_index_cmp_build(
    loom_builder_t* builder, uint8_t predicate,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t operand_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_index_cmp_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_index_cmp_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// Returns the vtable array for the index dialect.
const loom_op_vtable_t* const* loom_index_dialect_vtables(
    iree_host_size_t* out_count);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_INDEX_OPS_H_
