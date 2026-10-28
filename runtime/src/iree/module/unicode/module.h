// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULE_UNICODE_MODULE_H_
#define IREE_MODULE_UNICODE_MODULE_H_

#include "iree/vm/module.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates the synchronous "unicode" module using the runtime's Unicode 15.1
// properties and locale-independent full case mappings:
//
//   to_upper(i32 codepoint) -> (i32 count, i32 first, i32 second, i32 third)
//   to_lower(i32 codepoint) -> (i32 count, i32 first, i32 second)
//   is_decimal(i32 codepoint) -> i32
//   is_digit(i32 codepoint) -> i32
//   is_whitespace(i32 codepoint) -> i32
//
// Codepoints are unsigned 32-bit values. Case mappings return at least one
// codepoint, zero unused slots, and preserve unmapped or invalid input.
// Predicates return 0 or 1; decimal selects Numeric_Type=Decimal, digit selects
// Decimal or Digit, and whitespace selects White_Space. Language-specific
// policies such as Python's additional whitespace controls remain with the
// caller.
//
// Calls allocate nothing, retain no references, and use no per-process state.
// The module is independent of a VM environment or serving host. It can be
// shared by programs and concurrent processes. |out_module| is set null before
// construction; the caller releases the returned owner after linking.
IREE_API_EXPORT iree_status_t iree_unicode_module_create(
    iree_allocator_t host_allocator, iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MODULE_UNICODE_MODULE_H_
