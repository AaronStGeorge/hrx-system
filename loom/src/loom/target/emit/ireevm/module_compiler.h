// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IREE VM archive compilation for parsed Loom modules.
//
// This mutates a parsed module through the current IREE VM target path:
// target preset expansion, generic verification, source-to-low lowering, low
// descriptor verification, packetization, VM function-body bytecode emission,
// and VM bytecode module archive wrapping. The output archive is the same
// runtime artifact consumed by iree_vm_bytecode_module_create.

#ifndef LOOM_TARGET_EMIT_IREEVM_MODULE_COMPILER_H_
#define LOOM_TARGET_EMIT_IREEVM_MODULE_COMPILER_H_

#include "iree/base/api.h"
#include "loom/error/diagnostic.h"
#include "loom/ir/ir.h"
#include "loom/target/emit/ireevm/module_archive.h"
#include "loom/verify/verify.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_ireevm_module_compile_options_t {
  // VM module name stored in the emitted archive. Empty uses "loom".
  iree_string_view_t module_name;
  // Optional target.bundle symbol to compile. Empty requires exactly one
  // IREE-VM-compatible target bundle after preset expansion. A leading '@' is
  // accepted for command-line ergonomics.
  iree_string_view_t target_symbol;
  // Diagnostic sink used for verification, lowering, scheduling, and
  // allocation diagnostics. A NULL callback still counts diagnostics.
  loom_diagnostic_sink_t diagnostic_sink;
  // Source resolver used to render carets for op-backed diagnostics.
  loom_source_resolver_t source_resolver;
  // Maximum diagnostics to emit before the active subsystem stops walking.
  // Zero uses a conservative default.
  uint32_t max_errors;
} loom_ireevm_module_compile_options_t;

// Compiles |module| into an allocator-owned IREE VM bytecode module archive.
//
// |module| is mutated in place: compact target.preset records are expanded to
// explicit target records and the selected source function gains sibling low
// IR produced by source-to-low lowering. The caller owns |out_archive| and must
// release it with loom_ireevm_module_archive_deinitialize.
iree_status_t loom_ireevm_compile_module_archive(
    loom_module_t* module, const loom_ireevm_module_compile_options_t* options,
    iree_allocator_t allocator, loom_ireevm_module_archive_t* out_archive);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_IREEVM_MODULE_COMPILER_H_
