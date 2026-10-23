// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>
#include <string>
#include <thread>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/testing/gtest.h"

#if defined(IREE_SANITIZER_ADDRESS) || defined(__SANITIZE_ADDRESS__)
// This test intentionally issues an impossible allocation and requires the
// standard realloc contract: return NULL while preserving the old allocation.
extern "C" const char* __asan_default_options() {
  return "allocator_may_return_null=1";
}
#endif  // IREE_SANITIZER_ADDRESS || __SANITIZE_ADDRESS__
#if defined(IREE_SANITIZER_THREAD)
extern "C" const char* __tsan_default_options() {
  return "allocator_may_return_null=1";
}
#endif  // IREE_SANITIZER_THREAD

namespace {

TEST(ConsoleTracingTest, DynamicNameBoundaries) {
  for (size_t length : {0, 1, 63, 64, 65, 1024}) {
    SCOPED_TRACE(length);
    std::string name(length, 'x');
    ::testing::internal::CaptureStderr();
    IREE_TRACE_ZONE_BEGIN_NAMED_DYNAMIC(zone, name.data(), name.size());
    // The caller may reuse the name storage before ending the zone.
    name.assign(length, 'y');
    IREE_TRACE_ZONE_END(zone);
    std::string output = ::testing::internal::GetCapturedStderr();
    EXPECT_THAT(output, ::testing::HasSubstr("► " + std::string(length, 'x') +
                                             " (console_test.cc:"));
#if IREE_TRACING_CONSOLE_TIMING
    const size_t retained_length = length < 64 ? length : 64;
    EXPECT_THAT(output,
                ::testing::HasSubstr("◄ " + std::string(retained_length, 'x') +
                                     " t = "));
#else
    EXPECT_THAT(output, ::testing::Not(::testing::HasSubstr("◄")));
#endif  // IREE_TRACING_CONSOLE_TIMING
  }
}

TEST(ConsoleTracingTest, NestedNames) {
  ::testing::internal::CaptureStderr();
  IREE_TRACE_ZONE_BEGIN_NAMED(outer, "outer");
  IREE_TRACE_ZONE_BEGIN_NAMED(inner, "inner");
  IREE_TRACE_ZONE_END(inner);
  IREE_TRACE_ZONE_END(outer);
  std::string output = ::testing::internal::GetCapturedStderr();
  EXPECT_THAT(output, ::testing::HasSubstr("► outer (console_test.cc:"));
  EXPECT_THAT(output, ::testing::HasSubstr(" ► inner (console_test.cc:"));
#if IREE_TRACING_CONSOLE_TIMING
  EXPECT_THAT(output, ::testing::HasSubstr(" ◄ inner t = "));
  EXPECT_THAT(output, ::testing::HasSubstr("◄ outer t = "));
  EXPECT_LT(output.find("◄ inner"), output.find("◄ outer"));
#else
  EXPECT_THAT(output, ::testing::Not(::testing::HasSubstr("◄")));
#endif  // IREE_TRACING_CONSOLE_TIMING
}

TEST(ConsoleTracingTest, SectionsOverlapAndMigrateWithoutChangingZoneDepth) {
  ::testing::internal::CaptureStderr();
  IREE_TRACE_SECTION_SETUP(42, "model-loading");
  const char category[] = {'j', 'i', 't'};
  IREE_TRACE_SECTION_SETUP_DYNAMIC(43, category, sizeof(category));
  IREE_TRACE_ZONE_BEGIN_NAMED(zone, "synchronous");
  const iree_section_id_t first = IREE_TRACE_SECTION_ENTER(42, "first");
  char name[] = {'1', '0', '0', '%'};
  const iree_section_id_t second =
      IREE_TRACE_SECTION_ENTER_DYNAMIC(43, name, sizeof(name));
  const iree_section_id_t tagged =
      IREE_TRACE_SECTION_ENTER_TAGGED(42, "model-prepare", name, sizeof(name));
  name[0] = 'x';
  EXPECT_NE(first, 0u);
  EXPECT_NE(second, 0u);
  EXPECT_NE(first, second);
  std::thread([first] { IREE_TRACE_SECTION_LEAVE(first); }).join();
  IREE_TRACE_SECTION_LEAVE(second);
  IREE_TRACE_SECTION_LEAVE(tagged);
  IREE_TRACE_SECTION_LEAVE(0);
  IREE_TRACE_ZONE_END(zone);
  const std::string output = ::testing::internal::GetCapturedStderr();
  EXPECT_THAT(output,
              ::testing::HasSubstr("section category=42 model-loading\n"));
  EXPECT_THAT(output, ::testing::HasSubstr("section category=43 jit\n"));
  const std::string first_id = "section " + std::to_string(first);
  const std::string second_id = "section " + std::to_string(second);
  EXPECT_THAT(output, ::testing::HasSubstr(first_id + " enter category=42 t="));
  EXPECT_THAT(output,
              ::testing::HasSubstr(second_id + " enter category=43 t="));
  EXPECT_THAT(output, ::testing::HasSubstr("ns 100%\n"));
  EXPECT_THAT(output, ::testing::HasSubstr("ns model-prepare:100%\n"));
  EXPECT_LT(output.find(second_id + " enter"),
            output.find(first_id + " leave"));
  EXPECT_LT(output.find(first_id + " leave"),
            output.find(second_id + " leave"));
  EXPECT_THAT(output, ::testing::Not(::testing::HasSubstr("section 0 leave")));
  EXPECT_THAT(output,
              ::testing::HasSubstr("] ► synchronous (console_test.cc:"));
#if IREE_TRACING_CONSOLE_TIMING
  EXPECT_THAT(output, ::testing::HasSubstr("] ◄ synchronous t = "));
#endif  // IREE_TRACING_CONSOLE_TIMING
}

TEST(ConsoleTracingTest, MessagesAndAllocations) {
  std::string allocation(128, 'x');
  ::testing::internal::CaptureStderr();
  IREE_TRACE_MESSAGE(INFO, "embedded message");
  IREE_TRACE_MESSAGE_DYNAMIC(INFO, allocation.data(), 8);
  IREE_TRACE_ALLOC_WAIT_FOR_REALLOCATIONS();
  IREE_TRACE_ALLOC(allocation.data(), allocation.size());
  IREE_TRACE_FREE(allocation.data());
  IREE_TRACE_APP_EXIT(0);
  std::string output = ::testing::internal::GetCapturedStderr();
  EXPECT_THAT(output, ::testing::HasSubstr("i embedded message\n"));
  EXPECT_THAT(output, ::testing::HasSubstr("i xxxxxxxx\n"));
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
  EXPECT_THAT(output, ::testing::HasSubstr("heap alloc "));
  EXPECT_THAT(output, ::testing::HasSubstr("heap free "));
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING
}

TEST(ConsoleTracingTest, FailedReallocationPreservesOwnership) {
  void* allocation = nullptr;
  iree_status_t status = iree_allocator_malloc_uninitialized(
      iree_allocator_system(), 64, &allocation);
  ASSERT_TRUE(iree_status_is_ok(status));
  ASSERT_NE(allocation, nullptr);
  static_cast<char*>(allocation)[0] = 42;

  void* original_allocation = allocation;
  char address[2 + 2 * sizeof(void*) + 1] = {0};
  std::snprintf(address, sizeof(address), "%p", original_allocation);
  const std::string allocation_event = std::string("heap alloc ") + address;
  const std::string free_event = std::string("heap free ") + address;

  ::testing::internal::CaptureStderr();
  status = iree_allocator_realloc(iree_allocator_system(), IREE_HOST_SIZE_MAX,
                                  &allocation);
  const std::string output = ::testing::internal::GetCapturedStderr();
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_EQ(allocation, original_allocation);
  EXPECT_EQ(output.find(allocation_event), std::string::npos);
  EXPECT_EQ(output.find(free_event), std::string::npos);

  iree_status_ignore(status);
  EXPECT_EQ(static_cast<char*>(allocation)[0], 42);
  iree_allocator_free(iree_allocator_system(), allocation);
}

}  // namespace

int main(int argc, char** argv) {
  // Model an embedding application that does not initialize runtime tracing.
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
