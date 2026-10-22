// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"

#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_STACK_TRACE) != 0 && \
    defined(IREE_HAVE_LIBBACKTRACE)

extern "C" IREE_ATTRIBUTE_NOINLINE iree_status_t
iree_status_stack_trace_test_capture_cold() {
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "concurrent symbolization test");
}

#endif  // stack traces with libbacktrace
