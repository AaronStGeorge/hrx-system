// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU HAL executable compilation for parsed Loom modules.
//
// This mutates a parsed module through the current AMDGPU target-low native
// path: target-record resolution, verification, ABI resource materialization,
// pass-managed low preparation, packetization, native HSACO writing, and IREE
// HAL executable wrapping.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_MODULE_COMPILER_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_MODULE_COMPILER_H_

#include "iree/base/api.h"
#include "loom/error/diagnostic.h"
#include "loom/ir/ir.h"
#include "loom/target/compile_report.h"
#include "loom/target/emit/native/amdgpu/hal_executable.h"
#include "loom/verify/verify.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_module_compile_options_t {
  // Optional function symbol to compile. Empty requires exactly one AMDGPU
  // HAL-native-compatible function with a target record. A leading '@' is
  // accepted for command-line ergonomics.
  iree_string_view_t entry_symbol;
  // Optional target.artifact symbol to compile as one multi-export executable.
  // Empty preserves single-entry selection. A leading '@' is accepted for
  // command-line ergonomics.
  iree_string_view_t artifact_symbol;
  // Optional AMDHSA processor name such as `gfx1100` overriding the selected
  // target snapshot CPU. This preserves the target record's descriptor-set
  // family while letting JIT runners specialize to the concrete HAL device ISA.
  iree_string_view_t target_cpu;
  // Diagnostic sink used for verification, materialization, scheduling, and
  // allocation diagnostics. A NULL callback still counts diagnostics.
  loom_diagnostic_sink_t diagnostic_sink;
  // Source resolver used to render carets for op-backed diagnostics.
  loom_source_resolver_t source_resolver;
  // Maximum diagnostics to emit before the active subsystem stops walking.
  // Zero uses a conservative default.
  uint32_t max_errors;
  // Optional caller-owned structured compile report to populate.
  loom_target_compile_report_t* report;
  // Optional caller-owned row storage for detailed compile report rows.
  loom_target_compile_report_row_storage_t report_row_storage;
} loom_amdgpu_module_compile_options_t;

// Compiles |module| into an allocator-owned IREE HAL AMDGPU executable.
//
// |module| is mutated in place: source functions may gain sibling low IR and
// HAL low.resource imports in the selected target-low function are materialized
// before scheduling. Target records are resolved through the linked descriptor
// registry without materializing companion target records in the IR.
// |out_compiled| is false when target preflight or compiler diagnostics
// rejected the module; status remains reserved for infrastructure failures. The
// caller owns |out_executable| when |out_compiled| is true and must release it
// with loom_amdgpu_hal_executable_deinitialize.
iree_status_t loom_amdgpu_compile_hal_executable(
    loom_module_t* module, const loom_amdgpu_module_compile_options_t* options,
    iree_allocator_t allocator, bool* out_compiled,
    loom_amdgpu_hal_executable_t* out_executable);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_MODULE_COMPILER_H_
