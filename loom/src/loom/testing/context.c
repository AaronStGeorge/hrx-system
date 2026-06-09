// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/testing/context.h"

#include "loom/ops/op_registry.h"
#include "loom/ops/test/registry.h"

iree_status_t loom_testing_context_register_all_dialects(
    loom_context_t* context) {
  IREE_RETURN_IF_ERROR(loom_op_registry_register_all_dialects(context));
  return loom_test_dialect_register(context);
}

iree_status_t loom_testing_context_initialize_all(iree_allocator_t allocator,
                                                  loom_context_t* out_context) {
  loom_context_initialize(allocator, out_context);
  iree_status_t status =
      loom_testing_context_register_all_dialects(out_context);
  if (iree_status_is_ok(status)) {
    status = loom_context_finalize(out_context);
  }
  if (!iree_status_is_ok(status)) {
    loom_context_deinitialize(out_context);
  }
  return status;
}
