// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/pressure.h"

#include "iree/testing/gtest.h"
#include "loom/target/residency.h"

namespace loom {
namespace {

TEST(PressureTest, SourceCliffFloorRetainsReadWriteHighWater) {
  const loom_low_reg_class_t reg_class = {.alloc_unit_bits = 32};
  const loom_low_descriptor_set_t descriptors = {
      .reg_classes = &reg_class,
      .reg_class_count = 1,
  };
  const iree_string_view_t resource_name = IREE_SV("registers");
  const loom_target_residency_cliff_t cliffs[] = {
      {0, 4, 4, 3}, {0, 8, 3, 2}, {0, 12, 2, 1}};
  const loom_target_residency_cliff_range_t range = {
      .start = 0,
      .count = IREE_ARRAYSIZE(cliffs),
  };
  const loom_target_residency_direct_resource_table_t resources = {
      .names = &resource_name,
      .cliffs = cliffs,
      .cliff_count = IREE_ARRAYSIZE(cliffs),
      .cliff_ranges = &range,
      .resource_count = 1,
  };
  const struct {
    // Units read by the node, shared by both operand occurrences.
    uint32_t input_units;
    // Units written by the node and live at its successor.
    uint32_t output_units;
    // Whether the inputs remain live through the result write.
    bool late_read;
    // First cliff strictly above the source-order high-water mark.
    uint32_t first_cliff;
  } cases[] = {
      {1, 3, false, 0}, {1, 3, true, 1},  {4, 4, false, 1},
      {4, 4, true, 2},  {8, 4, false, 2}, {8, 4, true, 3},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message()
                 << test_case.input_units << " -> " << test_case.output_units
                 << " late=" << test_case.late_read);
    loom_low_schedule_value_record_t values[] = {
        {.unit_count = test_case.input_units, .register_class_id = 0},
        {.unit_count = test_case.output_units, .register_class_id = 0},
    };
    loom_local_value_domain_t domain = {.value_count = IREE_ARRAYSIZE(values)};
    const loom_liveness_block_relation_t block_liveness = {};
    const loom_liveness_dataflow_t liveness = {
        .blocks = &block_liveness,
        .block_count = 1,
    };
    loom_low_schedule_build_state_t state = {
        .pressure_cliffs = &resources,
        .target = {.descriptor_set = &descriptors},
        .value_domain = &domain,
        .liveness_dataflow = &liveness,
        .values = values,
    };
    uint64_t live_units[] = {0};
    uint32_t packing_reserve[] = {0};
    uint32_t first_cliffs[] = {0};
    uint16_t block_classes[1] = {};
    uint8_t block_class_touched[1] = {};
    loom_value_ordinal_t block_values[IREE_ARRAYSIZE(values)] = {};
    loom_low_schedule_pressure_state_t pressure = {
        .current_live_units_by_reg_class = live_units,
        .packing_reserve_units_by_reg_class = packing_reserve,
        .first_actionable_pressure_cliff_indices = first_cliffs,
        .block_reg_class_ids = block_classes,
        .block_reg_class_touched_flags = block_class_touched,
        .block_value_ordinals = block_values,
    };
    const loom_low_schedule_node_t consumer = {
        .operand_count = 1,
        .value_ordinals = {.inline_value_ordinals = {1}},
    };
    loom_low_schedule_node_t producer = {
        .operand_count = 2,
        .result_count = 1,
        .value_ordinals = {.inline_value_ordinals = {0, 0, 1, 3}},
    };
    if (test_case.late_read) {
      producer.flags |= LOOM_LOW_SCHEDULE_NODE_FLAG_LATE_READS;
    }
    loom_low_schedule_reverse_source_pressure_node(&state, &pressure,
                                                   &consumer);
    EXPECT_EQ(pressure.current_live_units, test_case.output_units);
    loom_low_schedule_reverse_source_pressure_node(&state, &pressure,
                                                   &producer);
    EXPECT_EQ(pressure.current_live_units, test_case.input_units);
    EXPECT_EQ(live_units[0], test_case.input_units);
    EXPECT_EQ(values[0].live_unit_count, test_case.input_units);
    EXPECT_EQ(values[1].live_unit_count, 0u);
    EXPECT_EQ(pressure.block_value_count, 2u);
    EXPECT_EQ(first_cliffs[0], test_case.first_cliff);

    // Starting another block clears live state, but cannot make a pressure
    // threshold already reached by the authored order actionable again.
    loom_low_schedule_initialize_source_pressure_block(&state, &pressure, 0);
    EXPECT_EQ(pressure.current_live_units, 0u);
    EXPECT_EQ(live_units[0], 0u);
    EXPECT_EQ(pressure.block_value_count, 0u);
    EXPECT_EQ(pressure.block_reg_class_count, 0u);
    EXPECT_EQ(block_class_touched[0], 0u);
    for (const auto& value : values) {
      EXPECT_EQ(value.flags, 0u);
      EXPECT_EQ(value.live_unit_count, 0u);
    }
    loom_low_schedule_reverse_source_pressure_node(&state, &pressure,
                                                   &consumer);
    EXPECT_EQ(first_cliffs[0], test_case.first_cliff);
    EXPECT_EQ(pressure.current_live_units, test_case.output_units);
    loom_low_schedule_reset_source_pressure_sweep(&state, &pressure);
    EXPECT_EQ(pressure.current_live_units, 0u);
    EXPECT_EQ(first_cliffs[0], test_case.first_cliff);
  }
}

}  // namespace
}  // namespace loom
