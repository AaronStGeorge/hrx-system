// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_KERNEL_KERNEL_SHAPE_REQUIREMENTS_H_
#define LOOM_TRANSFORMS_KERNEL_KERNEL_SHAPE_REQUIREMENTS_H_

#include "loom/ir/type_dependencies.h"
#include "loom/transforms/symbol/template_decision_model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Scalar dependencies of explicitly static structure in a callable body.
// This describes dependent vector/tile types, constexpr, and authored loop
// expansion controls. It does not determine template provider applicability.
typedef struct loom_kernel_shape_requirement_t {
  // Values consumed as structural extents in this function, including actual
  // arguments supplying a direct callee's structural requirements.
  loom_value_set_id_t values;
  // Formal argument identities transitively supplying those values.
  loom_value_set_id_t arguments;
} loom_kernel_shape_requirement_t;

typedef struct loom_kernel_shape_values_t loom_kernel_shape_values_t;

// Immutable summaries over one module snapshot. Mutating bodies invalidates
// the result. The owning arena retains the sets and symbol-indexed summaries.
typedef struct loom_kernel_shape_requirements_t {
  // Canonical membership shared by all summaries.
  loom_value_set_index_t* sets;
  // Sparse identities and formal dependencies of values visited in bodies.
  const loom_kernel_shape_values_t* values;
  // One summary per module symbol, empty for non-callable symbols.
  loom_kernel_shape_requirement_t* symbols;
} loom_kernel_shape_requirements_t;

// Returns the retained formal dependencies of a visited body value.
loom_value_set_id_t loom_kernel_shape_value_arguments(
    const loom_kernel_shape_requirements_t* requirements,
    loom_value_id_t value);

// Builds the summaries once. Exact calls substitute each callee's formal
// dependencies into the caller; runtime data arguments are not requirements.
// A family with one local implementation has the same dependency summary as
// an exact call; this does not bypass its applicability checks. Multi-provider
// applications remain separate header decisions at publication.
iree_status_t loom_kernel_shape_requirements_build(
    const loom_module_t* module,
    const loom_template_decision_model_catalog_t* decision_models,
    iree_arena_allocator_t* arena,
    loom_kernel_shape_requirements_t* out_requirements);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_KERNEL_KERNEL_SHAPE_REQUIREMENTS_H_
