// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/report.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/pass/registry.h"

namespace loom {
namespace {

struct ReportAllocator {
  // Zero-based allocation attempt to fail, or the host maximum for no failure.
  iree_host_size_t fail_at = IREE_HOST_SIZE_MAX;
  // Number of allocation attempts.
  iree_host_size_t attempts = 0;
  // Number of allocations that have not been freed.
  iree_host_size_t live = 0;

  iree_allocator_t allocator() { return {this, Control}; }

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** pointer) {
    auto* state = static_cast<ReportAllocator*>(self);
    const bool had_allocation = (command == IREE_ALLOCATOR_COMMAND_FREE ||
                                 command == IREE_ALLOCATOR_COMMAND_REALLOC) &&
                                *pointer != nullptr;
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        state->attempts++ == state->fail_at) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "report allocation rejected");
    }
    const auto system = iree_allocator_system();
    iree_status_t status = system.ctl(system.self, command, params, pointer);
    if (iree_status_is_ok(status)) {
      if (command == IREE_ALLOCATOR_COMMAND_FREE) {
        state->live -= had_allocation;
      } else {
        state->live += !had_allocation;
      }
    }
    return status;
  }
};

class PassReportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, allocator_.allocator(),
                                     &block_pool_);
    loom_pass_report_initialize(&block_pool_, &report_);
  }
  void TearDown() override {
    loom_pass_report_deinitialize(&report_);
    iree_arena_block_pool_deinitialize(&block_pool_);
    EXPECT_EQ(allocator_.live, 0u);
  }

  iree_status_t Append(iree_host_size_t index,
                       iree_string_view_t symbol = IREE_SV("function"),
                       loom_pass_t* pass = nullptr) {
    static const loom_pass_statistic_field_t field = {
        .name = IREE_SVL("changed-ops"),
        .offset = 0,
    };
    static const loom_pass_statistic_layout_t layout = {
        .storage_size = sizeof(int64_t),
        .fields = &field,
        .field_count = 1,
    };
    static const loom_pass_info_t info = {
        .kind = LOOM_PASS_FUNCTION,
        .statistic_layout = &layout,
    };
    static const loom_pass_descriptor_t descriptor = {.key = IREE_SVL("test")};
    int64_t statistic = 42;
    const loom_pass_report_invocation_options_t options = {
        .descriptor = &descriptor,
        .info = &info,
        .instruction_index = index,
        .anchor_kind = LOOM_PASS_FUNCTION,
        .pipeline_symbol = IREE_SV("pipeline"),
        .symbol_name = symbol,
        .duration_nanoseconds = 123,
        .changed = true,
        .status_code = IREE_STATUS_OUT_OF_RANGE,
        .statistic_storage = &statistic,
        .detail_head = pass ? pass->report_detail_head : nullptr,
        .detail_count = pass ? pass->report_detail_count : 0,
    };
    return loom_pass_report_append_invocation(&report_, &options);
  }

  // Allocator observing only backing block requests, not individual records.
  ReportAllocator allocator_;
  // Reusable blocks shared by successive report scopes.
  iree_arena_block_pool_t block_pool_ = {};
  // Report under test.
  loom_pass_report_t report_ = {};
};

TEST_F(PassReportTest, EmptyAndDisabledDoNotAllocate) {
  loom_pass_t pass = {};
  IREE_ASSERT_OK(
      loom_pass_report_append_detail(&pass, IREE_SV("unused"), nullptr, 0));
  EXPECT_EQ(allocator_.attempts, 0u);
  EXPECT_EQ(report_.invocation_head, nullptr);
  EXPECT_EQ(report_.invocation_tail, nullptr);
  EXPECT_EQ(report_.invocation_count, 0u);
}

TEST_F(PassReportTest, OwnsMutableNamesStatisticsAndDetails) {
  loom_pass_t pass = {.report = &report_};
  {
    std::string name = "function";
    std::string key = "event";
    const loom_pass_report_detail_field_t fields[] = {
        loom_pass_report_detail_string_field(
            iree_make_cstring_view(key.c_str()),
            iree_make_cstring_view(name.c_str())),
        loom_pass_report_detail_int64_field(IREE_SV("signed"), -7),
        loom_pass_report_detail_uint64_field(IREE_SV("unsigned"), UINT64_MAX),
        loom_pass_report_detail_bool_field(IREE_SV("changed"), true),
    };
    IREE_ASSERT_OK(loom_pass_report_append_detail(
        &pass, iree_make_cstring_view(key.c_str()), fields,
        IREE_ARRAYSIZE(fields)));
    IREE_ASSERT_OK(Append(7, iree_make_cstring_view(name.c_str()), &pass));
    name.assign(name.size(), 'x');
    key.assign(key.size(), 'y');
  }
  const auto* invocation = report_.invocation_head;
  ASSERT_NE(invocation, nullptr);
  EXPECT_EQ(invocation, report_.invocation_tail);
  EXPECT_EQ(invocation->next, nullptr);
  EXPECT_TRUE(
      iree_string_view_equal(invocation->symbol_name, IREE_SV("function")));
  EXPECT_EQ(invocation->instruction_index, 7u);
  EXPECT_EQ(invocation->duration_nanoseconds, 123);
  EXPECT_TRUE(invocation->changed);
  EXPECT_EQ(invocation->status_code, IREE_STATUS_OUT_OF_RANGE);
  ASSERT_EQ(invocation->statistic_count, 1u);
  EXPECT_EQ(invocation->statistics[0].value, 42);
  ASSERT_EQ(invocation->detail_count, 1u);
  const auto* detail = invocation->details;
  EXPECT_TRUE(iree_string_view_equal(detail->category, IREE_SV("event")));
  ASSERT_EQ(detail->field_count, 4u);
  EXPECT_TRUE(iree_string_view_equal(detail->fields[0].name, IREE_SV("event")));
  EXPECT_TRUE(iree_string_view_equal(detail->fields[0].string_value,
                                     IREE_SV("function")));
  EXPECT_EQ(detail->fields[1].int64_value, -7);
  EXPECT_EQ(detail->fields[2].uint64_value, UINT64_MAX);
  EXPECT_TRUE(detail->fields[3].bool_value);
  loom_pass_report_discard_pass_details(&pass);
  EXPECT_EQ(pass.report_detail_head, nullptr);
  EXPECT_EQ(pass.report_detail_tail, nullptr);
  EXPECT_EQ(pass.report_detail_count, 0u);
  EXPECT_EQ(invocation->details, detail);
}

TEST_F(PassReportTest, GrowthPreservesOrderAndReusesPoolBlocks) {
  for (int iteration = 0; iteration < 2; ++iteration) {
    const auto previous_allocations = allocator_.attempts;
    for (iree_host_size_t i = 0; i < 2048; ++i) {
      IREE_ASSERT_OK(Append(i));
    }
    if (iteration != 0) {
      EXPECT_EQ(allocator_.attempts, previous_allocations);
    }
    EXPECT_LT(allocator_.attempts, 128u);
    EXPECT_EQ(report_.arena.allocation_head, nullptr);
    iree_host_size_t count = 0;
    for (const auto* invocation = report_.invocation_head; invocation;
         invocation = invocation->next) {
      EXPECT_EQ(invocation->instruction_index, count++);
    }
    EXPECT_EQ(count, 2048u);
    EXPECT_EQ(count, report_.invocation_count);
    EXPECT_EQ(report_.invocation_tail->instruction_index, count - 1);
    loom_pass_report_deinitialize(&report_);
    loom_pass_report_initialize(&block_pool_, &report_);
  }
}

TEST_F(PassReportTest,
       FailedAllocationsLeaveCompletedRecordsAndReleaseStorage) {
  for (iree_host_size_t fail_at = 0;; ++fail_at) {
    SCOPED_TRACE(fail_at);
    allocator_.fail_at = fail_at;
    allocator_.attempts = 0;
    loom_pass_t pass = {.report = &report_};
    const auto field =
        loom_pass_report_detail_string_field(IREE_SV("key"), IREE_SV("value"));
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0; i < 64 && iree_status_is_ok(status); ++i) {
      status =
          loom_pass_report_append_detail(&pass, IREE_SV("event"), &field, 1);
      if (iree_status_is_ok(status)) {
        status = Append(i, IREE_SV("function"), &pass);
      }
      loom_pass_report_discard_pass_details(&pass);
    }
    iree_host_size_t completed = 0;
    for (const auto* invocation = report_.invocation_head; invocation;
         invocation = invocation->next) {
      EXPECT_EQ(invocation->instruction_index, completed++);
      EXPECT_EQ(invocation->detail_count, 1u);
      EXPECT_EQ(invocation->statistics[0].value, 42);
    }
    EXPECT_EQ(completed, report_.invocation_count);
    const bool succeeded = iree_status_is_ok(status);
    loom_pass_report_deinitialize(&report_);
    iree_arena_block_pool_trim(&block_pool_);
    EXPECT_EQ(allocator_.live, 0u);
    loom_pass_report_initialize(&block_pool_, &report_);
    if (succeeded) {
      EXPECT_EQ(completed, 64u);
      break;
    }
    IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  }
}

}  // namespace
}  // namespace loom
