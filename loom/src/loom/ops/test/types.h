// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// clang-format off
#ifndef LOOM_OPS_TEST_TYPES_H_
#define LOOM_OPS_TEST_TYPES_H_

#include "loom/ops/type_registry.h"
#include "loom/ops/test/type_builders.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const loom_parameterized_type_descriptor_t loom_test_scope_type_parameterized_descriptor;
extern const loom_parameterized_type_descriptor_t loom_test_matrix_type_parameterized_descriptor;
extern const loom_parameterized_type_descriptor_t loom_test_compact_matrix_type_parameterized_descriptor;
extern const loom_parameterized_type_descriptor_t loom_test_array_type_parameterized_descriptor;
extern const loom_parameterized_type_descriptor_t loom_test_variant_set_type_parameterized_descriptor;

// Generated type descriptors owned by this dialect.
extern const loom_type_registry_entry_t loom_test_type_registry_entries[];
extern const iree_host_size_t loom_test_type_registry_entry_count;
// Immutable indirect type-family bindings, in generated key order.
extern const loom_parameterized_type_descriptor_t* const loom_test_type_registry_parameterized_descriptors[];
extern const iree_host_size_t loom_test_type_registry_parameterized_count;

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_TEST_TYPES_H_
