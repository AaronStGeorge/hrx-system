// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Low function target binding.
//
// This layer connects ordinary Loom target records to dense low descriptor
// tables. The descriptor table ABI itself remains IR-agnostic; this file owns
// the codegen contract that says a low.func target record selects one
// descriptor set key and feature bitset before low.op packet verification,
// scheduling, allocation feedback, or emission run.

#ifndef LOOM_CODEGEN_LOW_TARGET_BINDING_H_
#define LOOM_CODEGEN_LOW_TARGET_BINDING_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resolved low target context for one low function.
typedef struct loom_low_resolved_target_t {
  // Symbol defining the low function target record.
  const loom_symbol_t* target_symbol;
  // Defining op for |target_symbol|; currently expected to be target.bundle.
  const loom_op_t* target_op;
  // target.config op selected by the target bundle.
  const loom_op_t* config_op;
  // Borrowed target symbol name without the leading '@'.
  iree_string_view_t target_name;
  // Borrowed descriptor-set key selected by |config_op|.
  iree_string_view_t descriptor_set_key;
  // Feature bitset selected by |config_op|.
  uint64_t feature_bits;
  // Descriptor set found in the caller-provided registry.
  const loom_low_descriptor_set_t* descriptor_set;
} loom_low_resolved_target_t;

// Resolves the target bundle/config and descriptor set for |low_func_op|.
// User IR failures are emitted through |emitter| and leave
// out_target->descriptor_set NULL. Infrastructure failures are returned as
// status. |low_func_op| must be low.func.def or low.func.decl.
iree_status_t loom_low_resolve_function_target(
    const loom_module_t* module, const loom_op_t* low_func_op,
    const loom_low_descriptor_registry_t* registry,
    iree_diagnostic_emitter_t emitter, loom_low_resolved_target_t* out_target);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_TARGET_BINDING_H_
