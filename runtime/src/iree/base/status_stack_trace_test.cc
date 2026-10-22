// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <atomic>
#include <string>
#include <thread>
#include <utility>

#include "iree/base/api.h"
#include "iree/base/status_cc.h"
#include "iree/testing/gtest.h"

#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_STACK_TRACE) != 0 && \
    defined(IREE_HAVE_LIBBACKTRACE)

extern "C" IREE_ATTRIBUTE_NOINLINE iree_status_t
iree_status_stack_trace_test_capture_cold();

namespace iree {
namespace {

IREE_ATTRIBUTE_NOINLINE iree_status_t CaptureWarmStatus() {
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "concurrent symbolization test");
}

TEST(StatusStackTraceTest, ConcurrentlySymbolizesColdCompilationUnit) {
  Status warm_status(CaptureWarmStatus());
  ASSERT_NE(warm_status.ToString().find("CaptureWarmStatus"),
            std::string::npos);

  constexpr size_t kThreadCount = 2;
  std::array<Status, kThreadCount> statuses = {
      Status(iree_status_stack_trace_test_capture_cold()),
      Status(iree_status_stack_trace_test_capture_cold()),
  };

  std::atomic<size_t> ready_count{0};
  std::atomic<bool> start{false};
  std::atomic<bool> all_symbolized{true};
  std::array<std::thread, kThreadCount> threads;
  for (size_t i = 0; i < kThreadCount; ++i) {
    threads[i] = std::thread([&, i] {
      ready_count.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      Status status(std::move(statuses[i]));
      if (status.ToString().find("iree_status_stack_trace_test_capture_cold") ==
          std::string::npos) {
        all_symbolized.store(false, std::memory_order_relaxed);
      }
    });
  }

  while (ready_count.load(std::memory_order_acquire) != kThreadCount) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (std::thread& thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(all_symbolized.load(std::memory_order_relaxed));
}

}  // namespace
}  // namespace iree

#endif  // stack traces with libbacktrace
