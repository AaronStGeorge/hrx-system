// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lowering from Loom IR into the structured LLVM IR module model.
//
// This layer is deliberately above text and bitcode serialization: it consumes
// Loom modules and constructs loom_llvmir_module_t records. Text, bitcode, and
// external LLVM tool verification all observe the same structured target IR.

#ifndef LOOM_TARGET_LLVMIR_LOWER_H_
#define LOOM_TARGET_LLVMIR_LOWER_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/target/emit/llvmir/module.h"
#include "loom/target/emit/llvmir/target_env.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_llvmir_lowering_state_t loom_llvmir_lowering_state_t;
typedef struct loom_llvmir_lowering_provider_t loom_llvmir_lowering_provider_t;

typedef iree_status_t (*loom_llvmir_lowering_try_op_fn_t)(
    const loom_llvmir_lowering_provider_t* provider,
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op, bool* out_handled);

struct loom_llvmir_lowering_provider_t {
  // Stable provider name used for diagnostics.
  iree_string_view_t name;
  // Attempts to lower one source op. Sets |out_handled| false when the op does
  // not belong to this provider.
  loom_llvmir_lowering_try_op_fn_t try_lower_op;
};

typedef struct loom_llvmir_lowering_options_t {
  // Target/ABI profile that owns triples, data layout, index widths, linkage,
  // calling convention, and future kernel ABI policy.
  const loom_llvmir_target_profile_t* target_profile;
  // Optional LLVM source_filename. Empty uses the Loom module name.
  iree_string_view_t source_name;
  // Optional target-specific lowering providers consulted by core lowering.
  const loom_llvmir_lowering_provider_t* const* providers;
  // Number of lowering provider pointers in |providers|.
  iree_host_size_t provider_count;
} loom_llvmir_lowering_options_t;

// Lowers |source_module| into a new structured LLVM IR module.
//
// The produced module is owned by the caller and must be freed with
// loom_llvmir_module_free(). Unsupported Loom ops, types, ABI cases, or
// control-flow forms fail with UNIMPLEMENTED diagnostics that name the gap.
iree_status_t loom_llvmir_lower_module(
    const loom_module_t* source_module,
    const loom_llvmir_lowering_options_t* options, iree_allocator_t allocator,
    loom_llvmir_module_t** out_module);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_LLVMIR_LOWER_H_
