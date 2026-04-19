// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IREE VM bytecode module archive emission.
//
// This wraps emitted VM function-body bytecode in the size-prefixed
// BytecodeModuleDef FlatBuffer envelope consumed by
// iree_vm_bytecode_module_create. The archive bytes are intentionally the real
// runtime artifact shape, not a testing or diagnostic surrogate.

#ifndef LOOM_TARGET_EMIT_IREEVM_MODULE_ARCHIVE_H_
#define LOOM_TARGET_EMIT_IREEVM_MODULE_ARCHIVE_H_

#include "iree/base/api.h"
#include "loom/target/emit/ireevm/function_bytecode.h"

#ifdef __cplusplus
extern "C" {
#endif

// Allocator-owned VM bytecode module archive bytes.
typedef struct loom_ireevm_module_archive_t {
  // Allocator-owned size-prefixed VMFB archive contents.
  uint8_t* data;
  // Number of bytes in |data|.
  iree_host_size_t data_length;
} loom_ireevm_module_archive_t;

// One local function body to place in a VM bytecode module archive.
typedef struct loom_ireevm_module_archive_function_t {
  // Exported local function name in the VM module namespace.
  iree_string_view_t export_name;
  // VM calling convention string, such as "0ii_i".
  iree_string_view_t calling_convention;
  // Emitted function-body bytecode and descriptor metadata.
  const loom_ireevm_function_bytecode_t* bytecode;
} loom_ireevm_module_archive_function_t;

// Releases storage owned by |archive|. Safe to call on a zero-initialized
// archive object.
void loom_ireevm_module_archive_deinitialize(
    loom_ireevm_module_archive_t* archive, iree_allocator_t allocator);

// Emits a size-prefixed IREE VM bytecode module archive containing the provided
// functions. All functions are exported 1:1, while imports, globals, rodata,
// and debug data are empty; unsupported bytecode-level references fail through
// the normal VM module verifier.
iree_status_t loom_ireevm_emit_module_archive(
    iree_string_view_t module_name,
    const loom_ireevm_module_archive_function_t* functions,
    iree_host_size_t function_count, iree_allocator_t allocator,
    loom_ireevm_module_archive_t* out_archive);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_IREEVM_MODULE_ARCHIVE_H_
