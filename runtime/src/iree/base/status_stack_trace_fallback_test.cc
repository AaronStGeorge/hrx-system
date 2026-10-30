// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/base/api.h"
#include "iree/base/status_cc.h"
#include "iree/testing/gtest.h"

#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_STACK_TRACE) != 0 && \
    defined(IREE_HAVE_LIBBACKTRACE)
#include <iree_libbacktrace.h>

static int unknown_frame_count = 0;

// Model libbacktrace receiving an address whose debug information cannot be
// resolved. Both the pinned and externally supplied library use this
// callback contract; the loader still owns the executable's exported symbols.
#if defined(IREE_LIBBACKTRACE_PRIVATE)
#define WRAPPED_PCINFO __wrap_iree_private_backtrace_pcinfo
#else
#define WRAPPED_PCINFO __wrap_backtrace_pcinfo
#endif
extern "C" int WRAPPED_PCINFO(struct backtrace_state*, uintptr_t pc,
                              backtrace_full_callback callback,
                              backtrace_error_callback, void* data) {
  ++unknown_frame_count;
  return callback(data, pc, nullptr, 0, nullptr);
}
#undef WRAPPED_PCINFO

extern "C" __attribute__((visibility("default"))) IREE_ATTRIBUTE_NOINLINE
    iree_status_t iree_status_stack_trace_test_capture_exported() {
  iree_status_t status =
      iree_make_status(IREE_STATUS_INTERNAL, "unnamed frame");
  return iree_status_annotate(status, IREE_SV("captured frame"));
}
#endif  // stack traces with libbacktrace

TEST(StatusStackTraceTest, UsesLoaderSymbolForUnnamedDebugFrame) {
#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_STACK_TRACE) != 0 && \
    defined(IREE_HAVE_LIBBACKTRACE)
  iree::Status status(iree_status_stack_trace_test_capture_exported());
  std::string message = status.ToString();
  EXPECT_GT(unknown_frame_count, 0);
  EXPECT_NE(message.find("iree_status_stack_trace_test_capture_exported"),
            std::string::npos)
      << message;
#else
  GTEST_SKIP() << "Symbolized status stack traces are disabled.";
#endif
}
