// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Minimal WebAssembly module binary emission from target-low packet sidecars.
//
// This target-owned layer wraps one emitted Wasm function body in the real Wasm
// binary module envelope: type, function, optional memory, export, and code
// sections. It intentionally stops at artifact construction; disassembly,
// validation, and execution stay external target tooling concerns.

#ifndef LOOM_TARGET_EMIT_WASM_MODULE_BINARY_H_
#define LOOM_TARGET_EMIT_WASM_MODULE_BINARY_H_

#include "iree/base/api.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_wasm_module_binary_t {
  // Allocator-owned Wasm module binary bytes.
  uint8_t* data;
  // Number of bytes in |data|.
  iree_host_size_t data_length;
  // True when the module defines one default linear memory.
  bool has_memory;
} loom_wasm_module_binary_t;

// Releases storage owned by |module|. Safe to call on a zero-initialized
// module object.
void loom_wasm_module_binary_deinitialize(loom_wasm_module_binary_t* module,
                                          iree_allocator_t allocator);

// Emits a complete Wasm binary module containing one exported function lowered
// from the provided target-low sidecars. The sidecars must describe one
// wasm.core.simd128 low.func.def supported by loom_wasm_emit_function_body.
// Memory is defined only when scheduled descriptor packets carry load/store
// schedule-class flags.
iree_status_t loom_wasm_emit_single_function_module(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation,
    iree_string_view_t export_name, iree_allocator_t allocator,
    loom_wasm_module_binary_t* out_module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_WASM_MODULE_BINARY_H_
