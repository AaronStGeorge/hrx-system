// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/diagnostics.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"

namespace loom {
namespace {

TEST(ScheduleDiagnosticsTest, PressureContributorsUseSparseLiveness) {
  const loom_liveness_value_class_t value_class = {
      .type_kind = LOOM_TYPE_REGISTER,
      .register_class_id = 0,
      .register_descriptor_set_stable_id = 1,
  };
  const loom_value_id_t value_ids[] = {10, 11, 12, 13};
  const uint32_t value_interval_indices[] = {0, 1, 2, UINT32_MAX};
  const loom_liveness_interval_t intervals[] = {
      {
          .value_id = 10,
          .start_point = 0,
          .end_point = 3,
          .value_class = value_class,
          .unit_count = 1,
      },
      {
          .value_id = 11,
          .start_point = 2,
          .end_point = 3,
          .value_class = value_class,
          .unit_count = 2,
      },
      {
          .value_id = 12,
          .start_point = 0,
          .end_point = 8,
          .value_class = value_class,
          .unit_count = 1,
      },
  };
  const loom_liveness_segment_t segments[] = {
      {.start_point = 0, .end_point = 3},
      {.start_point = 2, .end_point = 3},
      // The enclosing interval for value 12 covers the peak, but the first
      // half-open segment ends at it and the second begins later.
      {.start_point = 0, .end_point = 2},
      {.start_point = 3, .end_point = 8},
  };
  const loom_liveness_segment_range_t segment_ranges[] = {
      {.start = 0, .count = 1},
      {.start = 1, .count = 1},
      {.start = 2, .count = 2},
      {},
  };
  const loom_liveness_pressure_summary_t pressure_summary = {
      .value_class = value_class,
      .peak_live_units = 3,
      .peak_live_values = 2,
      .peak_point = 2,
  };
  const uint32_t pressure_budget = 2;
  const loom_low_schedule_table_t table = {
      .liveness =
          {
              .intervals = intervals,
              .interval_count = IREE_ARRAYSIZE(intervals),
              .value_ids = value_ids,
              .value_count = IREE_ARRAYSIZE(value_ids),
              .value_interval_indices = value_interval_indices,
              .pressure_summaries = &pressure_summary,
              .pressure_summary_count = 1,
              .segments = segments,
              .segment_count = IREE_ARRAYSIZE(segments),
              .value_segment_ranges = segment_ranges,
          },
      .pressure_summary_budgets = &pressure_budget,
  };
  struct Capture {
    iree_host_size_t emission_count = 0;
    iree_host_size_t contributor_count = 0;
    uint32_t peak_live_units = 0;
  } capture;
  const iree_diagnostic_emitter_t emitter = {
      .fn =
          [](void* user_data, const loom_diagnostic_emission_t* emission) {
            auto* capture = static_cast<Capture*>(user_data);
            ++capture->emission_count;
            EXPECT_EQ(emission->error, LOOM_ERR_BACKEND_003);
            EXPECT_EQ(emission->param_count, 10u);
            if (emission->param_count == 10) {
              EXPECT_EQ(emission->params[6].kind, LOOM_PARAM_U32);
              capture->peak_live_units = emission->params[6].u32;
              EXPECT_EQ(emission->params[9].kind, LOOM_PARAM_STRING_LIST);
              capture->contributor_count =
                  emission->params[9].string_list.count;
            }
            return iree_ok_status();
          },
      .user_data = &capture,
  };

  IREE_ASSERT_OK(loom_low_schedule_diagnostics_emit(
      &table, LOOM_LOW_SCHEDULE_DIAGNOSTIC_PRESSURE_PEAKS, emitter));
  EXPECT_EQ(capture.emission_count, 1u);
  EXPECT_EQ(capture.peak_live_units, 3u);
  EXPECT_EQ(capture.contributor_count, 2u);
}

}  // namespace
}  // namespace loom
