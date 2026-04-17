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

typedef uint32_t loom_low_verify_flags_t;

enum loom_low_verify_flag_bits_e {
  // Validate descriptor registry/table integrity before walking IR. This is
  // useful for tests and package bring-up; hot compiler paths should usually
  // verify linked descriptor packages once when constructing the target
  // configuration instead.
  LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY = 1u << 0,
};

typedef struct loom_low_verify_options_t {
  // Verification behavior flags.
  loom_low_verify_flags_t flags;
  // Descriptor registry available to this verification run. Unless
  // LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY is set, callers must pass a
  // registry already accepted by loom_low_descriptor_registry_verify.
  const loom_low_descriptor_registry_t* descriptor_registry;
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

// Verifies descriptor-local low function bodies in |module|. This checks that
// each low.func target resolves to a descriptor set and that descriptor-backed
// low.op/low.const packets match the selected descriptor rows.
iree_status_t loom_low_verify_module(const loom_module_t* module,
                                     const loom_low_verify_options_t* options,
                                     loom_low_verify_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_VERIFY_H_
