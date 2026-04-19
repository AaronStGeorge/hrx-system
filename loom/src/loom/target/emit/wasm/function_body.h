// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// WebAssembly function-body emission from target-low packet sidecars.
//
// This emits the size-prefixed function body stored in a Wasm code section,
// not a complete Wasm module. Module sections, imports/exports, validation
// tools, and runtime execution adapters are target-owned layers above this
// body emitter.

#ifndef LOOM_TARGET_EMIT_WASM_FUNCTION_BODY_H_
#define LOOM_TARGET_EMIT_WASM_FUNCTION_BODY_H_

#include "iree/base/api.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_wasm_function_body_t {
  // Allocator-owned size-prefixed Wasm function-body bytes.
  uint8_t* data;
  // Number of bytes in |data|, including the leading body-size LEB.
  iree_host_size_t data_length;
  // Number of semantic body bytes after the leading body-size LEB.
  iree_host_size_t body_length;
  // Number of Wasm function parameters.
  uint32_t parameter_count;
  // Number of Wasm local indices including parameters.
  uint32_t local_count;
} loom_wasm_function_body_t;

// Releases storage owned by |body|. Safe to call on a zero-initialized body.
void loom_wasm_function_body_deinitialize(loom_wasm_function_body_t* body,
                                          iree_allocator_t allocator);

// Emits a size-prefixed Wasm code-section function body for one scheduled and
// allocated low.func.def. The sidecars must describe the same function and the
// wasm.core.simd128 descriptor set. The current emitter supports the straight-
// line scalar i32 and SIMD128 subset with unspilled target-id allocation;
// unsupported packets fail loud instead of producing partial Wasm.
iree_status_t loom_wasm_emit_function_body(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation, iree_allocator_t allocator,
    loom_wasm_function_body_t* out_body);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_WASM_FUNCTION_BODY_H_
