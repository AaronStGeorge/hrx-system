// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/liveness.h"

#include <memory>
#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/liveness_json.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

struct ModuleDeleter {
  void operator()(loom_module_t* module) const { loom_module_free(module); }
};
using ModulePtr = std::unique_ptr<loom_module_t, ModuleDeleter>;

class LivenessTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_CFG, loom_cfg_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &analysis_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("liveness_test.loom"), &context_,
                                  &block_pool_, &options, &module));
    return ModulePtr(module);
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

  loom_func_like_t FindFunction(const loom_module_t* module,
                                iree_string_view_t name) {
    loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    uint16_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    loom_op_t* op = module->symbols.entries[symbol_id].defining_op;
    loom_func_like_t func = loom_func_like_cast(module, op);
    IREE_ASSERT(func.op != NULL);
    return func;
  }

  loom_liveness_analysis_t AnalyzeBody(const loom_module_t* module,
                                       loom_func_like_t func) {
    loom_liveness_analysis_t analysis;
    IREE_CHECK_OK(loom_liveness_analyze_region(
        module, loom_func_like_body(func), &analysis_arena_, &analysis));
    return analysis;
  }

  static bool ContainsValue(const loom_value_id_t* values,
                            iree_host_size_t count, loom_value_id_t value) {
    for (iree_host_size_t i = 0; i < count; ++i) {
      if (values[i] == value) return true;
    }
    return false;
  }

  static const loom_liveness_pressure_summary_t* FindRegisterPressure(
      const loom_liveness_analysis_t& analysis, loom_string_id_t class_id) {
    for (iree_host_size_t i = 0; i < analysis.pressure_summary_count; ++i) {
      const auto* summary = &analysis.pressure_summaries[i];
      if (summary->value_class.type_kind == LOOM_TYPE_REGISTER &&
          summary->value_class.register_class_id == class_id) {
        return summary;
      }
    }
    return nullptr;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  iree_arena_allocator_t analysis_arena_;
};

TEST_F(LivenessTest, SingleBlockIntervalsAndDeadDefs) {
  ModulePtr module = ParseModule(R"(
func.def @linear(%a: i32, %b: i32) -> (i32) {
  %sum = scalar.addi %a, %b : i32
  %dead = scalar.addi %a, %b : i32
  func.return %sum : i32
}
)");
  loom_func_like_t func = FindFunction(module.get(), IREE_SV("linear"));
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 2u);

  loom_liveness_analysis_t analysis = AnalyzeBody(module.get(), func);
  ASSERT_EQ(analysis.block_count, 1u);
  EXPECT_EQ(analysis.blocks[0].live_in_count, 0u);
  EXPECT_EQ(analysis.blocks[0].live_out_count, 0u);

  const loom_block_t* entry =
      loom_region_const_entry_block(loom_func_like_body(func));
  const loom_op_t* add = loom_block_const_op(entry, 0);
  const loom_op_t* dead_add = loom_block_const_op(entry, 1);
  loom_value_id_t sum = loom_op_const_results(add)[0];
  loom_value_id_t dead = loom_op_const_results(dead_add)[0];

  const loom_liveness_interval_t* a_interval =
      loom_liveness_interval_for_value(&analysis, args[0]);
  const loom_liveness_interval_t* sum_interval =
      loom_liveness_interval_for_value(&analysis, sum);
  const loom_liveness_interval_t* dead_interval =
      loom_liveness_interval_for_value(&analysis, dead);
  ASSERT_NE(a_interval, nullptr);
  ASSERT_NE(sum_interval, nullptr);
  ASSERT_NE(dead_interval, nullptr);
  EXPECT_EQ(a_interval->start_point, 0u);
  EXPECT_EQ(a_interval->end_point, 2u);
  EXPECT_EQ(sum_interval->start_point, 1u);
  EXPECT_EQ(sum_interval->end_point, 3u);
  EXPECT_EQ(dead_interval->start_point, 2u);
  EXPECT_EQ(dead_interval->end_point, 2u);
}

TEST_F(LivenessTest, CfgLiveInOutUsesSuccessorEdgesAndBranchOperands) {
  ModulePtr module = ParseModule(R"(
func.def @cfg_select(%cond: i1, %a: i32, %b: i32) -> (i32) {
  cfg.cond_br %cond, ^then, ^else : i1
^then:
  cfg.br ^join(%a : i32)
^else:
  cfg.br ^join(%b : i32)
^join(%result : i32):
  func.return %result : i32
}
)");
  loom_func_like_t func = FindFunction(module.get(), IREE_SV("cfg_select"));
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 3u);

  loom_liveness_analysis_t analysis = AnalyzeBody(module.get(), func);
  ASSERT_TRUE(analysis.is_cfg);
  ASSERT_EQ(analysis.block_count, 4u);

  const loom_liveness_block_info_t& entry = analysis.blocks[0];
  EXPECT_FALSE(
      ContainsValue(entry.live_out_values, entry.live_out_count, args[0]));
  EXPECT_TRUE(
      ContainsValue(entry.live_out_values, entry.live_out_count, args[1]));
  EXPECT_TRUE(
      ContainsValue(entry.live_out_values, entry.live_out_count, args[2]));

  const loom_liveness_block_info_t& then_block = analysis.blocks[1];
  const loom_liveness_block_info_t& else_block = analysis.blocks[2];
  const loom_liveness_block_info_t& join_block = analysis.blocks[3];
  EXPECT_TRUE(ContainsValue(then_block.live_in_values, then_block.live_in_count,
                            args[1]));
  EXPECT_TRUE(ContainsValue(else_block.live_in_values, else_block.live_in_count,
                            args[2]));
  EXPECT_EQ(join_block.live_in_count, 0u);
}

TEST_F(LivenessTest, CfgLoopPropagatesFixedPointLiveness) {
  ModulePtr module = ParseModule(R"(
func.def @cfg_loop(%cond: i1, %x: i32) -> (i32) {
  cfg.br ^loop(%x : i32)
^loop(%iter : i32):
  cfg.cond_br %cond, ^body, ^exit : i1
^body:
  %next = scalar.addi %iter, %x : i32
  cfg.br ^loop(%next : i32)
^exit:
  func.return %iter : i32
}
)");
  loom_func_like_t func = FindFunction(module.get(), IREE_SV("cfg_loop"));
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 2u);

  loom_liveness_analysis_t analysis = AnalyzeBody(module.get(), func);
  ASSERT_TRUE(analysis.is_cfg);
  ASSERT_EQ(analysis.block_count, 4u);

  const loom_liveness_block_info_t& entry = analysis.blocks[0];
  const loom_liveness_block_info_t& loop = analysis.blocks[1];
  const loom_liveness_block_info_t& body = analysis.blocks[2];
  const loom_liveness_block_info_t& exit = analysis.blocks[3];
  EXPECT_TRUE(
      ContainsValue(entry.live_out_values, entry.live_out_count, args[0]));
  EXPECT_TRUE(
      ContainsValue(entry.live_out_values, entry.live_out_count, args[1]));
  EXPECT_TRUE(ContainsValue(loop.live_in_values, loop.live_in_count, args[0]));
  EXPECT_TRUE(ContainsValue(loop.live_in_values, loop.live_in_count, args[1]));
  EXPECT_TRUE(ContainsValue(body.live_in_values, body.live_in_count, args[1]));
  EXPECT_EQ(exit.live_in_count, 1u);
}

TEST_F(LivenessTest, OperandTypeReferencesKeepDynamicDimsLive) {
  ModulePtr module = ParseModule(R"(
func.def @type_ref(%N: index, %v: vector<[%N]xi32>) -> (vector<[%N]xi32>) {
  func.return %v : vector<[%N]xi32>
}
)");
  loom_func_like_t func = FindFunction(module.get(), IREE_SV("type_ref"));
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 2u);

  loom_liveness_analysis_t analysis = AnalyzeBody(module.get(), func);
  const loom_liveness_interval_t* dim_interval =
      loom_liveness_interval_for_value(&analysis, args[0]);
  const loom_liveness_interval_t* vector_interval =
      loom_liveness_interval_for_value(&analysis, args[1]);
  ASSERT_NE(dim_interval, nullptr);
  ASSERT_NE(vector_interval, nullptr);
  EXPECT_EQ(dim_interval->end_point, vector_interval->end_point);
}

TEST_F(LivenessTest, TiedResultOperandIsLiveThroughConsumingOp) {
  ModulePtr module = ParseModule(R"(
func.def @tied_update(%tile: tile<4xf32>, %tensor: tensor<4xf32>, %off: index) -> (tensor<4xf32>) {
  %updated = test.update %tile, %tensor[%off] : tile<4xf32> -> (%tensor as tensor<4xf32>)
  func.return %updated : tensor<4xf32>
}
)");
  loom_func_like_t func = FindFunction(module.get(), IREE_SV("tied_update"));
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 3u);

  loom_liveness_analysis_t analysis = AnalyzeBody(module.get(), func);
  const loom_liveness_interval_t* tensor_interval =
      loom_liveness_interval_for_value(&analysis, args[1]);
  ASSERT_NE(tensor_interval, nullptr);
  EXPECT_LT(tensor_interval->start_point, tensor_interval->end_point);
}

TEST_F(LivenessTest, RegisterPressureGroupsByRegisterClassUnits) {
  ModulePtr module = ParseModule(R"(
func.def @register_pressure(%a: reg<vm.i32>, %b: reg<vm.i32>, %c: reg<vm.i32>) -> (reg<vm.i32>) {
  %ab = low.copy %a : reg<vm.i32> -> reg<vm.i32>
  %bc = low.copy %b : reg<vm.i32> -> reg<vm.i32>
  %cc = low.copy %c : reg<vm.i32> -> reg<vm.i32>
  func.return %ab : reg<vm.i32>
}
)");
  loom_func_like_t func =
      FindFunction(module.get(), IREE_SV("register_pressure"));
  loom_liveness_analysis_t analysis = AnalyzeBody(module.get(), func);

  loom_string_id_t vm_i32 =
      loom_module_lookup_string(module.get(), IREE_SV("vm.i32"));
  ASSERT_NE(vm_i32, LOOM_STRING_ID_INVALID);
  const loom_liveness_pressure_summary_t* pressure =
      FindRegisterPressure(analysis, vm_i32);
  ASSERT_NE(pressure, nullptr);
  EXPECT_EQ(pressure->peak_live_units, 3u);
  EXPECT_EQ(pressure->peak_live_values, 3u);
}

TEST_F(LivenessTest, PressureBudgetReportsHighUnrolledRegisterUse) {
  ModulePtr module = ParseModule(R"(
func.def @high_pressure(%a0: reg<vm.i32>, %a1: reg<vm.i32>, %a2: reg<vm.i32>, %a3: reg<vm.i32>, %a4: reg<vm.i32>, %a5: reg<vm.i32>) -> (reg<vm.i32>) {
  %r0 = low.copy %a0 : reg<vm.i32> -> reg<vm.i32>
  %r1 = low.copy %a1 : reg<vm.i32> -> reg<vm.i32>
  %r2 = low.copy %a2 : reg<vm.i32> -> reg<vm.i32>
  %r3 = low.copy %a3 : reg<vm.i32> -> reg<vm.i32>
  %r4 = low.copy %a4 : reg<vm.i32> -> reg<vm.i32>
  %r5 = low.copy %a5 : reg<vm.i32> -> reg<vm.i32>
  func.return %r0 : reg<vm.i32>
}
)");
  loom_func_like_t func = FindFunction(module.get(), IREE_SV("high_pressure"));
  loom_liveness_analysis_t analysis = AnalyzeBody(module.get(), func);

  loom_string_id_t vm_i32 =
      loom_module_lookup_string(module.get(), IREE_SV("vm.i32"));
  ASSERT_NE(vm_i32, LOOM_STRING_ID_INVALID);
  const loom_liveness_pressure_summary_t* pressure =
      FindRegisterPressure(analysis, vm_i32);
  ASSERT_NE(pressure, nullptr);
  EXPECT_EQ(pressure->peak_live_units, 6u);

  loom_liveness_pressure_budget_t budget = {
      .value_class = pressure->value_class,
      .max_live_units = 4,
      .max_live_values = 4,
  };
  const loom_liveness_pressure_budget_violation_t* violations = nullptr;
  iree_host_size_t violation_count = 0;
  IREE_ASSERT_OK(loom_liveness_collect_pressure_budget_violations(
      &analysis, &budget, 1, &analysis_arena_, &violations, &violation_count));
  ASSERT_EQ(violation_count, 1u);
  EXPECT_EQ(violations[0].budget_index, 0u);
  EXPECT_EQ(violations[0].summary, pressure);
  EXPECT_EQ(violations[0].violation_bits,
            LOOM_LIVENESS_PRESSURE_BUDGET_VIOLATION_LIVE_UNITS |
                LOOM_LIVENESS_PRESSURE_BUDGET_VIOLATION_LIVE_VALUES);
}

TEST_F(LivenessTest, FormatsMachineReadableJsonSummary) {
  ModulePtr module = ParseModule(R"(
func.def @json_pressure(%a: reg<vm.i32>, %b: reg<vm.i32>) -> (reg<vm.i32>) {
  %r = low.copy %a : reg<vm.i32> -> reg<vm.i32>
  %dead = low.copy %b : reg<vm.i32> -> reg<vm.i32>
  func.return %r : reg<vm.i32>
}
)");
  loom_func_like_t func = FindFunction(module.get(), IREE_SV("json_pressure"));
  loom_liveness_analysis_t analysis = AnalyzeBody(module.get(), func);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_liveness_format_json(&analysis, &builder));
  std::string json(iree_string_builder_buffer(&builder),
                   iree_string_builder_size(&builder));
  EXPECT_NE(json.find("\"format\":\"loom.liveness.v0\""), std::string::npos);
  EXPECT_NE(json.find("\"blocks\""), std::string::npos);
  EXPECT_NE(json.find("\"intervals\""), std::string::npos);
  EXPECT_NE(json.find("\"pressure_summaries\""), std::string::npos);
  EXPECT_NE(json.find("\"register_class\":\"vm.i32\""), std::string::npos);
  EXPECT_NE(json.find("\"peak_live_units\":2"), std::string::npos);
  iree_string_builder_deinitialize(&builder);
}

}  // namespace
}  // namespace loom
