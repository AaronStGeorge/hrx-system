// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_SANITIZER_OPS_H_
#define LOOM_OPS_SANITIZER_OPS_H_

#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_SANITIZER_ASSERT_VALUE = LOOM_OP_KIND(LOOM_DIALECT_SANITIZER, 0),
  LOOM_OP_SANITIZER_COUNT_ = 1,
};

// LOOM_OP_SANITIZER_ASSERT_VALUE: Assert predicate constraints over SSA values and return checked identity aliases. Unlike assume ops, this op is executable: failing the assertion reports the site and aborts execution. Passing the assertion refines facts for the returned aliases.
// %n_checked = sanitizer.assert.value %n [range(%n, 0, 4096), mul(%n, 16)] : index
LOOM_DEFINE_ISA(loom_sanitizer_assert_value_isa, LOOM_OP_SANITIZER_ASSERT_VALUE)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_sanitizer_assert_value_values, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_sanitizer_assert_value_results, 0)
LOOM_DEFINE_ATTR_PREDICATE_LIST(loom_sanitizer_assert_value_predicates, 0)
iree_status_t loom_sanitizer_assert_value_build(
    loom_builder_t* builder,
    const loom_value_id_t* values,
    iree_host_size_t values_count,
    const loom_predicate_t* predicates,
    iree_host_size_t predicates_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_sanitizer_assert_value_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_sanitizer_assert_value_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_sanitizer_assert_value_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Returns the vtable array for the sanitizer dialect.
const loom_op_vtable_t* const* loom_sanitizer_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the sanitizer dialect.
const loom_op_semantics_t* loom_sanitizer_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a sanitizer op kind, or empty metadata.
loom_op_semantics_t loom_sanitizer_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_SANITIZER_OPS_H_
