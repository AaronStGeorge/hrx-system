// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/packet_asm.h"

#include <inttypes.h>

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/target/test/low_registry.h"

namespace loom {
namespace {

class LowPacketAsmTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_test_low_descriptor_registry_initialize(&target_registry_);
  }

  void TearDown() override {
    if (module_) {
      loom_module_free(module_);
      module_ = nullptr;
    }
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* ParseSource(const std::string& source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;

    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(
        loom_text_parse(iree_make_string_view(source.data(), source.size()),
                        IREE_SV("low_packet_asm_test.loom"), &context_,
                        &block_pool_, &parse_options, &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  loom_op_t* FindFirstLowFunction(loom_module_t* module) {
    loom_block_t* block = loom_module_block(module);
    loom_op_t* op = nullptr;
    loom_block_for_each_op(block, op) {
      if (loom_low_func_def_isa(op)) {
        return op;
      }
    }
    return nullptr;
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  void ParseAndVerify(const char* body) {
    std::string source = "target.profile @test_target preset(\"test-low\")\n";
    source += body;
    module_ = ParseSource(source);
    ASSERT_NE(module_, nullptr);

    loom_low_verify_options_t verify_options = {
        .flags = LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY,
        .descriptor_registry = &target_registry_.registry,
        .descriptor_requirements =
            LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
        .max_errors = 20,
    };
    loom_low_verify_result_t verify_result = {};
    IREE_ASSERT_OK(
        loom_low_verify_module(module_, &verify_options, &verify_result));
    EXPECT_EQ(verify_result.error_count, 0u);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_target_low_descriptor_registry_t target_registry_ = {};
};

iree_status_t FormatNamedAllocatedValue(
    void* user_data, const loom_low_allocation_table_t* allocation,
    loom_value_id_t value_id,
    const loom_low_allocation_assignment_t* assignment,
    iree_host_size_t assignment_index, iree_string_builder_t* builder) {
  (void)user_data;
  (void)assignment_index;
  if (assignment->location_kind == LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test value is unassigned");
  }
  const loom_module_t* module = allocation->module;
  if (value_id < module->values.count) {
    const loom_value_t* value = loom_module_value(module, value_id);
    if (value->name_id != LOOM_STRING_ID_INVALID &&
        value->name_id < module->strings.count) {
      return iree_string_builder_append_string(
          builder, module->strings.entries[value->name_id]);
    }
  }
  return iree_string_builder_append_format(builder, "v%" PRIu32, value_id);
}

loom_low_packet_asm_form_table_t MakeCanonicalAsmFormTable(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    std::vector<uint32_t>* selected_asm_forms) {
  selected_asm_forms->assign(schedule->scheduled_node_count,
                             LOOM_LOW_ASM_FORM_ORDINAL_NONE);
  for (iree_host_size_t i = 0; i < schedule->scheduled_node_count; ++i) {
    loom_low_packet_view_t packet = {};
    IREE_EXPECT_OK(loom_low_packet_view_at(schedule, allocation, i, &packet));
    if (packet.descriptor == nullptr) {
      continue;
    }
    uint32_t asm_form_ordinal = LOOM_LOW_ASM_FORM_ORDINAL_NONE;
    IREE_EXPECT_OK(loom_low_descriptor_set_lookup_canonical_asm_form(
        schedule->target.descriptor_set, packet.node->descriptor_ordinal,
        &asm_form_ordinal));
    (*selected_asm_forms)[i] = asm_form_ordinal;
  }
  return loom_low_packet_asm_form_table_t{
      .module = schedule->module,
      .function_op = schedule->function_op,
      .target = schedule->target,
      .asm_form_ordinals = selected_asm_forms->data(),
      .asm_form_ordinal_count = selected_asm_forms->size(),
  };
}

TEST_F(LowPacketAsmTest, FormatsScheduledPacketsWithCanonicalAsmForms) {
  ParseAndVerify(
      "low.func.def target(@test_target) @packet_asm(%lhs: reg<test.i32>, "
      "%rhs: reg<test.i32>) -> (reg<test.i32>) {\n"
      "  %c7 = low.const<test.const.i32> {i32_value = 7} : reg<test.i32>\n"
      "  %sum = low.op<test.add.i32>(%lhs, %c7) : "
      "(reg<test.i32>, reg<test.i32>) -> reg<test.i32>\n"
      "  %copy = low.copy %sum : reg<test.i32> -> reg<test.i32>\n"
      "  %cmp = low.op<test.cmp.eq.i32>(%copy, %rhs) : "
      "(reg<test.i32>, reg<test.i32>) -> reg<test.i32>\n"
      "  low.return %cmp : reg<test.i32>\n"
      "}\n");
  loom_op_t* low_function = FindFirstLowFunction(module_);
  ASSERT_NE(low_function, nullptr);

  iree_arena_allocator_t table_arena;
  iree_arena_initialize(&block_pool_, &table_arena);
  loom_low_emission_frame_t frame = {};
  loom_low_emission_frame_options_t frame_options = {
      .descriptor_registry = &target_registry_.registry,
  };
  IREE_ASSERT_OK(loom_low_emission_frame_build(
      module_, low_function, &frame_options, &table_arena, &frame));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  std::vector<uint32_t> selected_asm_form_ordinals;
  loom_low_packet_asm_form_table_t selected_asm_forms =
      MakeCanonicalAsmFormTable(&frame.schedule, &frame.allocation,
                                &selected_asm_form_ordinals);
  loom_low_packet_asm_options_t asm_options = {
      .selected_asm_forms = &selected_asm_forms,
      .format_value =
          {
              .fn = FormatNamedAllocatedValue,
              .user_data = nullptr,
          },
  };
  IREE_ASSERT_OK(loom_low_packet_asm_format(&frame.schedule, &frame.allocation,
                                            &asm_options, &builder));
  EXPECT_EQ(std::string(iree_string_builder_view(&builder).data,
                        iree_string_builder_view(&builder).size),
            "^bb0(lhs, rhs):\n"
            "  c7 = test.const.i32 {i32_value = 7}\n"
            "  sum = test.add.i32 lhs, c7\n"
            "  copy = copy sum\n"
            "  cmp = test.cmp.eq.i32 copy, rhs\n"
            "  return cmp\n");
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&table_arena);
}

TEST_F(LowPacketAsmTest, FormatsStructuralBranches) {
  ParseAndVerify(
      "low.func.def target(@test_target) @packet_asm(%lhs: reg<test.i32>, "
      "%rhs: reg<test.i32>) -> (reg<test.i32>) {\n"
      "  %cmp = low.op<test.cmp.eq.i32>(%lhs, %rhs) : "
      "(reg<test.i32>, reg<test.i32>) -> reg<test.i32>\n"
      "  low.cond_br %cmp, ^then, ^else : reg<test.i32>\n"
      "^then:\n"
      "  %sum = low.op<test.add.i32>(%lhs, %rhs) : "
      "(reg<test.i32>, reg<test.i32>) -> reg<test.i32>\n"
      "  low.br ^join(%sum: reg<test.i32>)\n"
      "^else:\n"
      "  low.br ^join(%rhs: reg<test.i32>)\n"
      "^join(%result: reg<test.i32>):\n"
      "  low.return %result : reg<test.i32>\n"
      "}\n");
  loom_op_t* low_function = FindFirstLowFunction(module_);
  ASSERT_NE(low_function, nullptr);

  iree_arena_allocator_t table_arena;
  iree_arena_initialize(&block_pool_, &table_arena);
  loom_low_emission_frame_t frame = {};
  loom_low_emission_frame_options_t frame_options = {
      .descriptor_registry = &target_registry_.registry,
  };
  IREE_ASSERT_OK(loom_low_emission_frame_build(
      module_, low_function, &frame_options, &table_arena, &frame));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_low_packet_asm_options_t asm_options = {
      .format_value =
          {
              .fn = FormatNamedAllocatedValue,
              .user_data = nullptr,
          },
  };
  IREE_ASSERT_OK(loom_low_packet_asm_format(&frame.schedule, &frame.allocation,
                                            &asm_options, &builder));
  const std::string output(iree_string_builder_view(&builder).data,
                           iree_string_builder_view(&builder).size);
  EXPECT_NE(output.find("cond_br cmp, ^bb1, ^bb2"), std::string::npos);
  EXPECT_NE(output.find("^bb3(result):"), std::string::npos);
  EXPECT_NE(output.find("br ^bb3(sum)"), std::string::npos);
  EXPECT_NE(output.find("return result"), std::string::npos);
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&table_arena);
}

TEST_F(LowPacketAsmTest, FormatsStructuralConcat) {
  ParseAndVerify(
      "low.func.def target(@test_target) @packet_asm(%lo: reg<test.i32>, "
      "%hi: reg<test.i32>) -> (reg<test.i32 x2>) {\n"
      "  %pair = low.concat(%lo, %hi) : (reg<test.i32>, reg<test.i32>) -> "
      "reg<test.i32 x2>\n"
      "  low.return %pair : reg<test.i32 x2>\n"
      "}\n");
  loom_op_t* low_function = FindFirstLowFunction(module_);
  ASSERT_NE(low_function, nullptr);

  iree_arena_allocator_t table_arena;
  iree_arena_initialize(&block_pool_, &table_arena);
  loom_low_emission_frame_t frame = {};
  loom_low_emission_frame_options_t frame_options = {
      .descriptor_registry = &target_registry_.registry,
  };
  IREE_ASSERT_OK(loom_low_emission_frame_build(
      module_, low_function, &frame_options, &table_arena, &frame));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_low_packet_asm_options_t asm_options = {
      .format_value =
          {
              .fn = FormatNamedAllocatedValue,
              .user_data = nullptr,
          },
  };
  IREE_ASSERT_OK(loom_low_packet_asm_format(&frame.schedule, &frame.allocation,
                                            &asm_options, &builder));
  EXPECT_EQ(std::string(iree_string_builder_view(&builder).data,
                        iree_string_builder_view(&builder).size),
            "^bb0(lo, hi):\n"
            "  pair = concat(lo, hi)\n"
            "  return pair\n");
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&table_arena);
}

TEST_F(LowPacketAsmTest, FormatsStructuralSlice) {
  ParseAndVerify(
      "low.func.def target(@test_target) @packet_asm(%pair: reg<test.i32 "
      "x2>) -> (reg<test.i32>) {\n"
      "  %lane = low.slice %pair[1] : reg<test.i32 x2> -> reg<test.i32>\n"
      "  low.return %lane : reg<test.i32>\n"
      "}\n");
  loom_op_t* low_function = FindFirstLowFunction(module_);
  ASSERT_NE(low_function, nullptr);

  iree_arena_allocator_t table_arena;
  iree_arena_initialize(&block_pool_, &table_arena);
  loom_low_emission_frame_t frame = {};
  loom_low_emission_frame_options_t frame_options = {
      .descriptor_registry = &target_registry_.registry,
  };
  IREE_ASSERT_OK(loom_low_emission_frame_build(
      module_, low_function, &frame_options, &table_arena, &frame));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_low_packet_asm_options_t asm_options = {
      .format_value =
          {
              .fn = FormatNamedAllocatedValue,
              .user_data = nullptr,
          },
  };
  IREE_ASSERT_OK(loom_low_packet_asm_format(&frame.schedule, &frame.allocation,
                                            &asm_options, &builder));
  EXPECT_EQ(std::string(iree_string_builder_view(&builder).data,
                        iree_string_builder_view(&builder).size),
            "^bb0(pair):\n"
            "  lane = slice pair[1]\n"
            "  return lane\n");
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&table_arena);
}

TEST_F(LowPacketAsmTest, RejectsMissingValueFormatter) {
  ParseAndVerify(
      "low.func.def target(@test_target) @packet_asm() {\n"
      "  low.return\n"
      "}\n");
  loom_op_t* low_function = FindFirstLowFunction(module_);
  ASSERT_NE(low_function, nullptr);

  iree_arena_allocator_t table_arena;
  iree_arena_initialize(&block_pool_, &table_arena);
  loom_low_emission_frame_t frame = {};
  loom_low_emission_frame_options_t frame_options = {
      .descriptor_registry = &target_registry_.registry,
  };
  IREE_ASSERT_OK(loom_low_emission_frame_build(
      module_, low_function, &frame_options, &table_arena, &frame));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_low_packet_asm_options_t asm_options = {};
  iree_status_t status = loom_low_packet_asm_format(
      &frame.schedule, &frame.allocation, &asm_options, &builder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&table_arena);
}

TEST_F(LowPacketAsmTest, RejectsInvalidSelectedAsmForms) {
  ParseAndVerify(
      "low.func.def target(@test_target) @packet_asm() {\n"
      "  low.return\n"
      "}\n");
  loom_op_t* low_function = FindFirstLowFunction(module_);
  ASSERT_NE(low_function, nullptr);

  iree_arena_allocator_t table_arena;
  iree_arena_initialize(&block_pool_, &table_arena);
  loom_low_emission_frame_t frame = {};
  loom_low_emission_frame_options_t frame_options = {
      .descriptor_registry = &target_registry_.registry,
  };
  IREE_ASSERT_OK(loom_low_emission_frame_build(
      module_, low_function, &frame_options, &table_arena, &frame));

  const uint32_t selected_asm_form_ordinals[1] = {
      LOOM_LOW_ASM_FORM_ORDINAL_NONE,
  };
  loom_low_packet_asm_form_table_t selected_asm_forms = {
      .module = frame.schedule.module,
      .function_op = frame.schedule.function_op,
      .target = frame.schedule.target,
      .asm_form_ordinals = selected_asm_form_ordinals,
      .asm_form_ordinal_count = 0,
  };
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_low_packet_asm_options_t asm_options = {
      .selected_asm_forms = &selected_asm_forms,
      .format_value =
          {
              .fn = FormatNamedAllocatedValue,
              .user_data = nullptr,
          },
  };
  iree_status_t status = loom_low_packet_asm_format(
      &frame.schedule, &frame.allocation, &asm_options, &builder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&table_arena);
}

}  // namespace
}  // namespace loom
