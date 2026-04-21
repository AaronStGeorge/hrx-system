// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/compile_report_format.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/types.h"

namespace loom {
namespace {

TEST(CompileReportFormatTest, FormatsSummaryAndDetails) {
  loom_target_compile_report_pressure_row_t pressure_rows[] = {
      {
          .register_class = IREE_SVL("test.i32"),
          .type_kind = LOOM_TYPE_REGISTER,
          .element_type = LOOM_SCALAR_TYPE_I32,
          .peak_live_units = 7,
          .peak_live_values = 4,
          .peak_point = 3,
          .peak_block_name = IREE_SVL("entry"),
          .peak_operation_name = IREE_SVL("low.op<test.add.i32>"),
      },
  };
  loom_target_compile_report_spill_row_t spill_rows[] = {
      {
          .value_name = IREE_SVL("rhs"),
          .register_class = IREE_SVL("test.i32"),
          .type_kind = LOOM_TYPE_REGISTER,
          .element_type = LOOM_SCALAR_TYPE_I32,
          .assignment_index = 2,
          .slot_index = 1,
          .slot_space = IREE_SVL("stack"),
          .byte_size = 4,
          .byte_alignment = 4,
          .store_count = 1,
          .reload_count = 2,
      },
  };
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report);
  report.artifact_kind = LOOM_TARGET_COMPILE_ARTIFACT_KIND_VM_ARCHIVE;
  report.entry_symbol = IREE_SVL("branchy");
  report.target_bundle_name = IREE_SVL("vm_target");
  report.lowered_symbol = IREE_SVL("branchy");
  loom_target_compile_report_record_artifact_size(&report, 128);
  loom_target_compile_report_record_schedule(&report, 5, 5, 4, 2, 1, 1, 1, 7);
  loom_target_compile_report_record_allocation(&report, 6, 1, 1, 2, 0);
  loom_target_compile_report_record_emission(&report, 8, 64, 80);
  loom_target_compile_report_record_memory(&report, 16, 32);
  report.pressure_rows = pressure_rows;
  report.pressure_row_count = IREE_ARRAYSIZE(pressure_rows);
  report.pressure_row_total_count = 2;
  report.spill_rows = spill_rows;
  report.spill_row_count = IREE_ARRAYSIZE(spill_rows);
  report.spill_row_total_count = 3;
  report.detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS |
                         LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS;

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  const loom_target_compile_report_format_options_t options = {
      .mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS,
  };
  IREE_ASSERT_OK(
      loom_target_compile_report_format_text(&report, &options, &builder));

  iree_string_view_t output = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("artifact=vm-archive"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("entry=branchy"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("pressure_classes=1"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output,
                                  IREE_SV("pressure_rows copied=1 total=2"), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(output, IREE_SV("pressure[0] class=test.i32"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("spill[0] value=rhs"), 0),
            IREE_STRING_VIEW_NPOS);

  iree_string_builder_deinitialize(&builder);
}

TEST(CompileReportFormatTest, ParsesModes) {
  loom_target_compile_report_format_mode_t mode =
      LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE;
  IREE_ASSERT_OK(
      loom_target_compile_report_format_mode_parse(IREE_SV("summary"), &mode));
  EXPECT_EQ(mode, LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_compile_report_format_mode_parse(IREE_SV("verbose"), &mode));
}

}  // namespace
}  // namespace loom
