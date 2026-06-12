// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/compile_report_low.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/registers.h"

namespace loom {
namespace {

TEST(CompileReportLowTest, RecordsPressureAndSpillRows) {
  constexpr uint32_t kSourceAssignmentIndex = 0;
  constexpr uint32_t kResultAssignmentIndex = 1;
  constexpr uint32_t kEdgeCopyCount = 1;
  constexpr uint32_t kRegisterCopyTagOffset = 0;
  constexpr uint32_t kMemoryGlobalTagOffset = 18;
  constexpr uint32_t kMatrixWmmaTagOffset = 41;
  constexpr uint32_t kRegisterClassGprOffset = 57;
  static const uint8_t kDescriptorStringTable[] =
      "\x11"
      "register.copy.b32"
      "\x16"
      "memory.global.load.u32"
      "\x0f"
      "matrix.wmma.f32"
      "\x08"
      "test.gpr";
  const loom_low_descriptor_t descriptors[] = {
      {
          .semantic_tag_string_offset = kRegisterCopyTagOffset,
      },
      {
          .semantic_tag_string_offset = kMemoryGlobalTagOffset,
      },
      {
          .semantic_tag_string_offset = kMatrixWmmaTagOffset,
      },
  };
  const loom_low_reg_class_t reg_classes[] = {
      {
          .name_string_offset = kRegisterClassGprOffset,
          .flags = LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY,
          .alloc_unit_bits = 32,
          .spill_class_id = LOOM_LOW_REG_CLASS_NONE,
          .full_register_part_mask = 1,
          .spill_slot_space = LOOM_LOW_SPILL_SLOT_SPACE_STACK,
      },
  };
  const loom_low_descriptor_set_t descriptor_set = {
      .stable_id = 1,
      .string_table =
          {
              .data = kDescriptorStringTable,
              .data_length = sizeof(kDescriptorStringTable) - 1,
          },
      .descriptors = descriptors,
      .descriptor_count = IREE_ARRAYSIZE(descriptors),
      .reg_classes = reg_classes,
      .reg_class_count = IREE_ARRAYSIZE(reg_classes),
  };
  loom_low_schedule_node_t schedule_nodes[13] = {};
  loom_value_t module_values[6] = {};
  loom_value_ordinal_t value_ordinals[6] = {
      LOOM_VALUE_ORDINAL_INVALID, LOOM_VALUE_ORDINAL_INVALID,
      LOOM_VALUE_ORDINAL_INVALID, LOOM_VALUE_ORDINAL_INVALID,
      LOOM_VALUE_ORDINAL_INVALID, LOOM_VALUE_ORDINAL_INVALID,
  };
  loom_module_t module = {
      .values =
          {
              .count = IREE_ARRAYSIZE(module_values),
              .capacity = IREE_ARRAYSIZE(module_values),
              .entries = module_values,
          },
      .scratch =
          {
              .values =
                  {
                      .values_by_value_id = value_ordinals,
                      .capacity = IREE_ARRAYSIZE(value_ordinals),
                      .state = LOOM_VALUE_U32_SCRATCH_STATE_UNACQUIRED_ORDINALS,
                  },
          },
  };
  module_values[4].type = loom_low_register_type(/*descriptor_set_stable_id=*/1,
                                                 /*register_class_id=*/0, 1);
  module_values[5].type = loom_low_register_type(/*descriptor_set_stable_id=*/1,
                                                 /*register_class_id=*/0, 2);
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS;

  const loom_op_t peak_op = {};
  const loom_liveness_pressure_summary_t pressure_summaries[] = {
      {
          .value_class =
              {
                  .type_kind = LOOM_TYPE_REGISTER,
                  .element_type = LOOM_SCALAR_TYPE_I32,
                  .register_descriptor_set_stable_id = 1,
                  .register_class_id = 0,
              },
          .peak_live_units = 7,
          .peak_live_values = 5,
          .peak_point = 3,
      },
      {
          .value_class =
              {
                  .type_kind = LOOM_TYPE_REGISTER,
                  .element_type = LOOM_SCALAR_TYPE_F32,
                  .register_descriptor_set_stable_id = 1,
                  .register_class_id = 0,
              },
          .peak_live_units = 11,
          .peak_live_values = 2,
          .peak_op = &peak_op,
          .peak_point = 9,
      },
  };
  const loom_low_allocation_assignment_t assignments[] = {
      {
          .value_id = 4,
          .value_class = pressure_summaries[0].value_class,
          .descriptor_reg_class_id = 0,
          .location_kind = LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT,
          .location_base = 0,
          .location_count = 1,
      },
      {
          .value_id = 5,
          .value_class = pressure_summaries[1].value_class,
          .descriptor_reg_class_id = 0,
          .location_kind = LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT,
          .location_base = 1,
          .location_count = 1,
      },
  };
  const loom_value_id_t liveness_value_ids[] = {
      4,
      5,
  };
  const uint32_t assignment_indices_by_value_ordinal[] = {
      0,
      1,
  };
  const loom_low_allocation_copy_decision_t copy_decisions[] = {
      {
          .source_value_id = 4,
          .result_value_id = 5,
          .source_assignment_index = kSourceAssignmentIndex,
          .result_assignment_index = kResultAssignmentIndex,
          .kind = LOOM_LOW_ALLOCATION_COPY_MATERIALIZED,
      },
  };
  const loom_low_allocation_edge_copy_t edge_copies[kEdgeCopyCount] = {
      {
          .payload_index = 0,
          .source_value_id = 4,
          .destination_value_id = 5,
          .source_assignment_index = kSourceAssignmentIndex,
          .destination_assignment_index = kResultAssignmentIndex,
          .source_unit_offset = 0,
          .destination_unit_offset = 0,
          .unit_count = 1,
      },
  };
  const loom_low_allocation_edge_copy_group_t edge_copy_groups[] = {
      {
          .copy_start = 0,
          .copy_count = kEdgeCopyCount,
      },
  };
  const loom_low_allocation_spill_plan_t spill_plans[] = {
      {
          .value_id = 4,
          .assignment_index = 0,
          .slot_index = 0,
          .slot_space = LOOM_LOW_SPILL_SLOT_SPACE_STACK,
          .byte_size = 16,
          .byte_alignment = 8,
          .store_count = 1,
          .reload_count = 2,
      },
      {
          .value_id = 5,
          .assignment_index = 1,
          .slot_index = 1,
          .slot_space = LOOM_LOW_SPILL_SLOT_SPACE_SCRATCH,
          .byte_size = 32,
          .byte_alignment = 16,
          .store_count = 3,
          .reload_count = 4,
      },
  };
  const loom_op_t copy_op = {
      .kind = LOOM_OP_KIND_UNKNOWN,
      .operand_count = 1,
      .result_count = 1,
  };
  schedule_nodes[0] = (loom_low_schedule_node_t){
      .op = &copy_op,
      .kind = LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR,
      .descriptor = &descriptors[0],
      .schedule_class_name = IREE_SVL("amdgpu.valu"),
      .operand_count = 1,
      .result_count = 1,
      .value_ordinals =
          {
              .inline_value_ordinals = {0, 1},
          },
  };
  schedule_nodes[1] = (loom_low_schedule_node_t){
      .kind = LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR,
      .descriptor = &descriptors[1],
      .schedule_class_name = IREE_SVL("amdgpu.vmem.load"),
  };
  schedule_nodes[2] = (loom_low_schedule_node_t){
      .kind = LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR,
      .descriptor = &descriptors[2],
      .schedule_class_name = IREE_SVL("amdgpu.wmma"),
  };
  const loom_low_emission_frame_t frame = {
      .target =
          {
              .descriptor_set = &descriptor_set,
          },
      .schedule =
          {
              .module = &module,
              .target =
                  {
                      .descriptor_set = &descriptor_set,
                  },
              .nodes = schedule_nodes,
              .node_count = 13,
              .dependency_count = 6,
              .scheduled_node_count = 12,
              .resource_use_count = 4,
              .hazard_gap_count = 2,
              .model_summary_count = 1,
          },
      .allocation =
          {
              .module = &module,
              .target =
                  {
                      .descriptor_set = &descriptor_set,
                  },
              .liveness =
                  {
                      .value_ids = liveness_value_ids,
                      .value_count = IREE_ARRAYSIZE(liveness_value_ids),
                      .pressure_summaries = pressure_summaries,
                      .pressure_summary_count =
                          IREE_ARRAYSIZE(pressure_summaries),
                  },
              .assignments = assignments,
              .assignment_count = IREE_ARRAYSIZE(assignments),
              .assignment_indices_by_value_ordinal =
                  assignment_indices_by_value_ordinal,
              .spill_plans = spill_plans,
              .spill_plan_count = IREE_ARRAYSIZE(spill_plans),
              .copy_decisions = copy_decisions,
              .copy_decision_count = IREE_ARRAYSIZE(copy_decisions),
              .edge_copies = edge_copies,
              .edge_copy_count = IREE_ARRAYSIZE(edge_copies),
              .edge_copy_groups = edge_copy_groups,
              .edge_copy_group_count = IREE_ARRAYSIZE(edge_copy_groups),
              .spill_count = IREE_ARRAYSIZE(spill_plans),
              .coalesced_copy_count = 3,
              .materialized_copy_count = 1,
          },
  };

  IREE_ASSERT_OK(
      loom_target_compile_report_record_low_emission_frame(&report, &frame));

  EXPECT_TRUE(iree_all_bits_set(report.detail_flags,
                                LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE));
  EXPECT_TRUE(iree_all_bits_set(report.detail_flags,
                                LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION));
  EXPECT_TRUE(iree_all_bits_set(
      report.detail_flags,
      LOOM_TARGET_COMPILE_REPORT_DETAIL_STATIC_INSTRUCTION_MIX));
  EXPECT_TRUE(iree_all_bits_set(report.detail_flags,
                                LOOM_TARGET_COMPILE_REPORT_DETAIL_MOVE_CAUSES));
  EXPECT_TRUE(iree_all_bits_set(
      report.detail_flags, LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS));
  EXPECT_TRUE(iree_all_bits_set(report.detail_flags,
                                LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS));
  EXPECT_EQ(report.schedule_node_count, 13u);
  EXPECT_EQ(report.register_pressure_summary_count, 2u);
  EXPECT_EQ(report.register_pressure_peak_live_units, 11u);
  EXPECT_EQ(report.allocation_spill_count, 2u);
  EXPECT_EQ(report.static_instruction_mix.descriptor_count, 3u);
  EXPECT_EQ(report.static_instruction_mix.vector_alu_count, 1u);
  EXPECT_EQ(report.static_instruction_mix.global_memory_count, 1u);
  EXPECT_EQ(report.static_instruction_mix.matrix_count, 1u);
  EXPECT_EQ(report.static_instruction_mix.wmma_count, 1u);
  EXPECT_EQ(report.static_instruction_mix.register_move_count, 1u);
  EXPECT_EQ(report.static_instruction_mix.unknown_count, 0u);
  EXPECT_EQ(report.move_causes[LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_COPY]
                .packet_count,
            1u);
  EXPECT_EQ(report.move_causes[LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_COPY]
                .unit_count,
            1u);
  EXPECT_EQ(
      report.move_causes[LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_BRANCH_EDGE]
          .packet_count,
      1u);
  EXPECT_EQ(
      report.move_causes[LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_BRANCH_EDGE]
          .unit_count,
      1u);
  EXPECT_EQ(
      report
          .move_causes
              [LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_OPERAND_BANK_MATERIALIZATION]
          .packet_count,
      1u);
  EXPECT_EQ(
      report
          .move_causes
              [LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_OPERAND_BANK_MATERIALIZATION]
          .unit_count,
      2u);
  EXPECT_EQ(report.pressure_rows.count, 2u);
  ASSERT_NE(report.pressure_rows.head, nullptr);
  const auto* pressure_rows =
      static_cast<const loom_target_compile_report_pressure_row_t*>(
          loom_target_compile_report_vec_const_rows(report.pressure_rows.head));
  EXPECT_EQ(pressure_rows[0].peak_live_units, 7u);
  EXPECT_EQ(pressure_rows[0].peak_live_values, 5u);
  EXPECT_TRUE(iree_string_view_equal(pressure_rows[0].peak_operation_name,
                                     IREE_SV("<block-boundary>")));
  EXPECT_EQ(pressure_rows[1].peak_live_units, 11u);
  EXPECT_EQ(report.spill_rows.count, 2u);
  ASSERT_NE(report.spill_rows.head, nullptr);
  const auto* spill_rows =
      static_cast<const loom_target_compile_report_spill_row_t*>(
          loom_target_compile_report_vec_const_rows(report.spill_rows.head));
  EXPECT_EQ(spill_rows[0].assignment_index, 0u);
  EXPECT_EQ(spill_rows[0].slot_index, 0u);
  EXPECT_TRUE(
      iree_string_view_equal(spill_rows[0].slot_space, IREE_SV("stack")));
  EXPECT_EQ(spill_rows[0].byte_size, 16u);
  EXPECT_EQ(spill_rows[0].store_count, 1u);
  EXPECT_EQ(spill_rows[0].reload_count, 2u);
  EXPECT_EQ(spill_rows[1].slot_index, 1u);
  loom_target_compile_report_deinitialize(&report);
}

}  // namespace
}  // namespace loom
