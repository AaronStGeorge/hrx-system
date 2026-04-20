// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compiled Loom execution candidates.

#ifndef LOOM_TOOLING_EXECUTION_CANDIDATE_H_
#define LOOM_TOOLING_EXECUTION_CANDIDATE_H_

#include "iree/base/api.h"
#include "loom/target/emit/ireevm/module_archive.h"
#include "loom/tooling/execution/hal_backend.h"
#include "loom/tooling/execution/session.h"
#include "loom/verify/verify.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_run_candidate_kind_e {
  LOOM_RUN_CANDIDATE_KIND_UNINITIALIZED = 0,
  LOOM_RUN_CANDIDATE_KIND_VM_ARCHIVE = 1,
  LOOM_RUN_CANDIDATE_KIND_HAL_EXECUTABLE = 2,
} loom_run_candidate_kind_t;

typedef struct loom_run_candidate_compile_options_t {
  // VM module name stored in VM bytecode archives. Empty uses "loom".
  iree_string_view_t module_name;
  // Optional target.bundle symbol to compile. Empty requires one compatible
  // target for the selected backend.
  iree_string_view_t target_symbol;
  // Diagnostic sink used for verification, lowering, scheduling, and
  // allocation diagnostics.
  loom_diagnostic_sink_t diagnostic_sink;
  // Source resolver used to render carets for op-backed diagnostics.
  loom_source_resolver_t source_resolver;
  // Maximum diagnostics to emit before stopping. Zero uses a conservative
  // default.
  uint32_t max_errors;
} loom_run_candidate_compile_options_t;

typedef struct loom_run_candidate_t {
  // Compiled artifact kind held by this candidate.
  loom_run_candidate_kind_t kind;
  // Host allocator used for owned candidate storage.
  iree_allocator_t host_allocator;
  // HAL backend that produced |hal_executable|.
  const loom_run_hal_backend_t* hal_backend;
  // HAL target selected during candidate compilation.
  loom_run_hal_selected_target_t hal_target;
  // HAL executable bytes produced by |hal_backend|.
  loom_run_hal_executable_t hal_executable;
  // VM bytecode archive produced by the IREE VM compiler path.
  loom_ireevm_module_archive_t vm_archive;
} loom_run_candidate_t;

// Initializes compile options with stderr diagnostics and a small error cap.
void loom_run_candidate_compile_options_initialize(
    loom_run_candidate_compile_options_t* out_options);

// Compiles |run_module| to an IREE VM archive candidate. Compilation may
// mutate the parsed module by expanding target records and adding lowered IR.
iree_status_t loom_run_candidate_compile_vm(
    loom_run_module_t* run_module,
    const loom_run_candidate_compile_options_t* options,
    iree_allocator_t allocator, loom_run_candidate_t* out_candidate);

// Selects a HAL target through |backend| and compiles |run_module| to a HAL
// executable candidate. Compilation may mutate the parsed module by expanding
// target records and adding lowered IR.
iree_status_t loom_run_candidate_compile_hal(
    const loom_run_hal_backend_t* backend,
    const loom_run_hal_runtime_t* runtime, loom_run_module_t* run_module,
    const loom_run_candidate_compile_options_t* options,
    iree_allocator_t allocator, loom_run_candidate_t* out_candidate);

// Releases all artifact storage owned by |candidate|.
void loom_run_candidate_deinitialize(loom_run_candidate_t* candidate);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_CANDIDATE_H_
