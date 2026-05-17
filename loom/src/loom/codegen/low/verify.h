// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Descriptor-local verification for target-low functions.
//
// This verifier is intentionally separate from the generic structural verifier:
// it needs an explicit linked descriptor registry selected by the caller, while
// generic IR verification must stay target-package agnostic.

#ifndef LOOM_CODEGEN_LOW_VERIFY_H_
#define LOOM_CODEGEN_LOW_VERIFY_H_

#include "iree/base/api.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_verify_options_t {
  // Descriptor registry available to this verification run. The registry is
  // target-owned static data; IR verification only uses it to resolve selected
  // descriptor sets and packet semantics.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Optional runtime/device target overlay applied when compatible with a low
  // function's target record.
  loom_target_selection_t target_selection;
  // Structured diagnostic emitter for user IR failures.
  iree_diagnostic_emitter_t emitter;
  // Maximum number of errors to emit before aborting the walk. 0 means no
  // limit.
  uint32_t max_errors;
} loom_low_verify_options_t;

typedef struct loom_low_verify_result_t {
  // Number of error diagnostics emitted.
  uint32_t error_count;
  // Number of warning diagnostics emitted.
  uint32_t warning_count;
} loom_low_verify_result_t;

typedef struct loom_low_verify_scratch_t {
  // Required value-id indexed u32 scratch storage for register-part masks.
  loom_value_u32_scratch_t* value_scratch;
} loom_low_verify_scratch_t;

// Returns low verification scratch backed by |module|'s reusable scratch.
static inline loom_low_verify_scratch_t loom_low_verify_scratch_for_module(
    loom_module_t* module) {
  return (loom_low_verify_scratch_t){
      .value_scratch = &module->scratch.values,
  };
}

// Verifies descriptor-local low function bodies in |module|. This checks that
// each low.func target resolves to a descriptor set and that descriptor-backed
// low.op/low.const packets match the selected descriptor rows. Verification is
// logically read-only on IR but mutates |scratch|.
iree_status_t loom_low_verify_module(const loom_module_t* module,
                                     const loom_low_verify_options_t* options,
                                     loom_low_verify_scratch_t* scratch,
                                     loom_low_verify_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_VERIFY_H_
