// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/testing/context.h"

#include "loom/ops/op_registry.h"

iree_status_t loom_testing_context_register_all_dialects(
    loom_context_t* context) {
  return loom_op_registry_register_all_dialects(context);
}

iree_status_t loom_testing_context_initialize_all(iree_allocator_t allocator,
                                                  loom_context_t* out_context) {
  return loom_op_registry_initialize_context(allocator, out_context);
}
