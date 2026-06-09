// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test/tooling context helpers.
//
// These helpers register the complete in-tree Loom dialect surface without
// depending on loom-check's execution engine. Use them only for full corpus,
// production front-door fuzzing, registry composition, and tool integration
// tests. Unit tests and generated synthetic fuzzers should register the minimal
// dialect set they exercise so accidental production-dialect dependencies stay
// visible in BUILD files.

#ifndef LOOM_TESTING_CONTEXT_H_
#define LOOM_TESTING_CONTEXT_H_

#include "iree/base/api.h"
#include "loom/ir/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registers all checked-in dialect vtables and built-in encoding families.
//
// The context must have been initialized and must not have been finalized yet.
iree_status_t loom_testing_context_register_all_dialects(
    loom_context_t* context);

// Initializes |out_context|, registers all checked-in dialects/encodings, and
// finalizes op-name lookup tables. On failure the partially initialized context
// is deinitialized before returning.
iree_status_t loom_testing_context_initialize_all(iree_allocator_t allocator,
                                                  loom_context_t* out_context);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TESTING_CONTEXT_H_
