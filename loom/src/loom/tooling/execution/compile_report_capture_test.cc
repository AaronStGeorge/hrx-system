// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/compile_report_capture.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

TEST(CompileReportCaptureTest, ConfiguresDetailedRowStorage) {
  loom_run_compile_report_capture_options_t options = {};
  loom_run_compile_report_capture_options_initialize(&options);
  options.mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS;
  options.row_limit = 2;

  loom_run_compile_report_capture_t capture = {};
  IREE_ASSERT_OK(loom_run_compile_report_capture_initialize(
      &options, iree_allocator_system(), &capture));

  loom_run_candidate_compile_options_t compile_options = {};
  loom_run_candidate_compile_options_initialize(&compile_options);
  loom_run_compile_report_capture_configure_compile_options(&capture,
                                                            &compile_options);
  EXPECT_EQ(compile_options.report, &capture.report);
  EXPECT_EQ(compile_options.report_row_storage.pressure_rows,
            capture.pressure_rows);
  EXPECT_EQ(compile_options.report_row_storage.pressure_row_capacity, 2u);
  EXPECT_EQ(compile_options.report_row_storage.spill_rows, capture.spill_rows);
  EXPECT_EQ(compile_options.report_row_storage.spill_row_capacity, 2u);
  EXPECT_EQ(compile_options.report_row_storage.source_low_rows,
            capture.source_low_rows);
  EXPECT_EQ(compile_options.report_row_storage.source_low_row_capacity, 2u);

  loom_run_compile_report_capture_deinitialize(&capture);
}

TEST(CompileReportCaptureTest, AppendsWithSeparator) {
  loom_run_compile_report_capture_options_t options = {};
  loom_run_compile_report_capture_options_initialize(&options);
  options.mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY;

  loom_run_compile_report_capture_t capture = {};
  IREE_ASSERT_OK(loom_run_compile_report_capture_initialize(
      &options, iree_allocator_system(), &capture));
  capture.report.artifact_kind = LOOM_TARGET_COMPILE_ARTIFACT_KIND_VM_ARCHIVE;

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(
      iree_string_builder_append_string(&builder, IREE_SV("output")));
  IREE_ASSERT_OK(
      loom_run_compile_report_capture_append_text(&capture, &builder));

  iree_string_view_t output = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("output\nCOMPILE-REPORT"), 0),
            IREE_STRING_VIEW_NPOS);

  iree_string_builder_deinitialize(&builder);
  loom_run_compile_report_capture_deinitialize(&capture);
}

}  // namespace
}  // namespace loom
