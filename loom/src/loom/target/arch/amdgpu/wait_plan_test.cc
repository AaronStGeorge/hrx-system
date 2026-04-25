// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/wait_plan.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/schedule.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/amdgpu/low_registry.h"
#include "loom/testing/module_ptr.h"
#include "loom/verify/verify.h"

namespace {

using ModulePtr = ::loom::testing::ModulePtr;

std::string ToString(const iree_string_builder_t& builder) {
  if (iree_string_builder_size(&builder) == 0) {
    return std::string();
  }
  return std::string(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
}

std::string TargetPreamble(const char* target_symbol, const char* preset_key) {
  std::string source = "target.profile @";
  source += target_symbol;
  source += " preset(\"";
  source += preset_key;
  source += "\")\n\n";
  return source;
}

class AmdgpuWaitPlanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    block_pool_initialized_ = true;

    loom_context_initialize(iree_allocator_system(), &context_);
    context_initialized_ = true;
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));

    loom_amdgpu_low_descriptor_registry_initialize(&low_registry_);
  }

  void TearDown() override {
    if (context_initialized_) {
      loom_context_deinitialize(&context_);
    }
    if (block_pool_initialized_) {
      iree_arena_block_pool_deinitialize(&block_pool_);
    }
  }

  iree_status_t ParseAndVerify(iree_string_view_t source,
                               ModulePtr* out_module) {
    out_module->reset();
    loom_text_parse_options_t parse_options = {};
    loom_low_descriptor_text_asm_environment_initialize(
        &low_registry_.registry, &parse_options.low_asm_environment);

    loom_module_t* raw_module = nullptr;
    IREE_RETURN_IF_ERROR(
        loom_text_parse(source, IREE_SV("amdgpu_wait_plan_test.loom"),
                        &context_, &block_pool_, &parse_options, &raw_module));
    if (!raw_module) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "test source did not parse");
    }
    ModulePtr module(raw_module);

    loom_verify_options_t verify_options = {};
    loom_verify_result_t verify_result = {};
    IREE_RETURN_IF_ERROR(
        loom_verify_module(module.get(), &verify_options, &verify_result));
    if (verify_result.error_count != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "test source failed generic verification");
    }

    loom_low_verify_options_t low_verify_options = {
        .flags = LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY,
        .descriptor_registry = &low_registry_.registry,
        .max_errors = 100,
    };
    loom_low_verify_result_t low_verify_result = {};
    IREE_RETURN_IF_ERROR(loom_low_verify_module(
        module.get(), &low_verify_options, &low_verify_result));
    if (low_verify_result.error_count != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "test source failed low verification");
    }

    *out_module = std::move(module);
    return iree_ok_status();
  }

  const loom_op_t* FirstLowFunction(const loom_module_t* module) {
    loom_block_t* module_block = loom_module_block((loom_module_t*)module);
    const loom_op_t* op = nullptr;
    loom_block_for_each_op(module_block, op) {
      if (loom_low_func_def_isa(op)) {
        return op;
      }
    }
    return nullptr;
  }

  iree_status_t BuildWaitPlan(const loom_module_t* module,
                              const loom_op_t* low_function,
                              iree_arena_allocator_t* arena,
                              loom_low_schedule_sidecar_t* out_schedule,
                              loom_amdgpu_wait_plan_t* out_plan) {
    loom_low_schedule_options_t schedule_options = {
        .descriptor_registry = &low_registry_.registry,
        .strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
    };
    IREE_RETURN_IF_ERROR(loom_low_schedule_function(
        module, low_function, &schedule_options, arena, out_schedule));
    return loom_amdgpu_wait_plan_build(out_schedule, arena, out_plan);
  }

  iree_arena_block_pool_t block_pool_ = {};
  loom_context_t context_ = {};
  loom_target_low_descriptor_registry_t low_registry_ = {};
  bool block_pool_initialized_ = false;
  bool context_initialized_ = false;
};

TEST_F(AmdgpuWaitPlanTest, PlansLoadUseAndRecordsExplicitWaits) {
  std::string source = TargetPreamble("gfx11_target", "amdgpu-gfx11");
  source += R"(
low.func.def target(@gfx11_target) @gfx11_func(%s0: reg<amdgpu.sgpr>, %resource: reg<amdgpu.sgpr x4>, %soffset: reg<amdgpu.sgpr>, %vaddr: reg<amdgpu.vgpr>, %v0: reg<amdgpu.vgpr>) -> (reg<amdgpu.sgpr>, reg<amdgpu.vgpr>) {
  %smem = low.op<amdgpu.s_buffer_load_dword>(%resource, %soffset) {offset = 0} : (reg<amdgpu.sgpr x4>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %vmem = low.op<amdgpu.buffer_load_dword>(%resource, %vaddr, %soffset) {offset = 4} : (reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.vgpr>
  %s_mix = low.op<amdgpu.s_add_u32>(%s0, %smem) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %v_mix = low.op<amdgpu.v_add_u32>(%v0, %vmem) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  low.op<amdgpu.buffer_store_dword>(%v_mix, %resource, %vaddr, %soffset) {offset = 8} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)
  low.op<amdgpu.s_waitcnt>() {vmcnt = 0, lgkmcnt = 0} : ()
  low.op<amdgpu.s_waitcnt_vscnt>() {vscnt = 0} : ()
  low.op<amdgpu.s_waitcnt_depctr>() {depctr = 0} : ()
  low.return %s_mix, %v_mix : reg<amdgpu.sgpr>, reg<amdgpu.vgpr>
}
)";

  ModulePtr module;
  IREE_ASSERT_OK(ParseAndVerify(
      iree_make_string_view(source.data(), source.size()), &module));
  const loom_op_t* low_function = FirstLowFunction(module.get());
  ASSERT_NE(low_function, nullptr);

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_schedule_sidecar_t schedule = {};
  loom_amdgpu_wait_plan_t plan = {};
  IREE_ASSERT_OK(
      BuildWaitPlan(module.get(), low_function, &arena, &schedule, &plan));

  bool saw_planned_vmem_load_use = false;
  bool saw_planned_smem_use = false;
  bool saw_explicit_store_wait = false;
  bool saw_explicit_alu_wait = false;
  for (iree_host_size_t i = 0; i < plan.action_count; ++i) {
    const loom_amdgpu_wait_plan_action_t& action = plan.actions[i];
    if (action.kind == LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED &&
        action.reason == LOOM_AMDGPU_WAIT_PLAN_REASON_SSA_USE &&
        action.counter_id == LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD &&
        action.producer_node != LOOM_LOW_SCHEDULE_NODE_NONE &&
        action.consumer_node != LOOM_LOW_SCHEDULE_NODE_NONE &&
        action.outstanding_before == 1) {
      saw_planned_vmem_load_use = true;
    }
    if (action.kind == LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED &&
        action.reason == LOOM_AMDGPU_WAIT_PLAN_REASON_SSA_USE &&
        action.counter_id == LOOM_AMDGPU_WAIT_COUNTER_SMEM &&
        action.producer_node != LOOM_LOW_SCHEDULE_NODE_NONE &&
        action.consumer_node != LOOM_LOW_SCHEDULE_NODE_NONE &&
        action.outstanding_before == 1) {
      saw_planned_smem_use = true;
    }
    if (action.kind == LOOM_AMDGPU_WAIT_PLAN_ACTION_EXPLICIT &&
        action.reason == LOOM_AMDGPU_WAIT_PLAN_REASON_EXPLICIT_PACKET &&
        action.counter_id == LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE &&
        action.outstanding_before == 1) {
      saw_explicit_store_wait = true;
    }
    if (action.kind == LOOM_AMDGPU_WAIT_PLAN_ACTION_EXPLICIT &&
        action.reason == LOOM_AMDGPU_WAIT_PLAN_REASON_EXPLICIT_PACKET &&
        action.counter_id == LOOM_AMDGPU_WAIT_COUNTER_ALU) {
      saw_explicit_alu_wait = true;
    }
  }
  EXPECT_TRUE(saw_planned_vmem_load_use);
  EXPECT_TRUE(saw_planned_smem_use);
  EXPECT_TRUE(saw_explicit_store_wait);
  EXPECT_TRUE(saw_explicit_alu_wait);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_amdgpu_wait_plan_format_json(&plan, &builder));
  std::string json = ToString(builder);
  iree_string_builder_deinitialize(&builder);

  EXPECT_NE(json.find("\"format\":\"loom.amdgpu.wait_plan.v0\""),
            std::string::npos);
  EXPECT_NE(json.find("\"reason_name\":\"ssa_use\""), std::string::npos);
  EXPECT_NE(json.find("\"counter_name\":\"vmem_load\""), std::string::npos);
  EXPECT_NE(json.find("\"counter_name\":\"vmem_store\""), std::string::npos);
  EXPECT_NE(json.find("\"counter_name\":\"smem\""), std::string::npos);
  EXPECT_NE(json.find("\"counter_name\":\"alu\""), std::string::npos);
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuWaitPlanTest, PlansCombinedWaitcntForGfx950) {
  std::string source = TargetPreamble("gfx950_target", "amdgpu-gfx950");
  source += R"(
low.func.def target(@gfx950_target) @gfx950_func(%s0: reg<amdgpu.sgpr>, %resource: reg<amdgpu.sgpr x4>, %soffset: reg<amdgpu.sgpr>, %vaddr: reg<amdgpu.vgpr>, %v0: reg<amdgpu.vgpr>) -> (reg<amdgpu.sgpr>, reg<amdgpu.vgpr>) {
  %smem = low.op<amdgpu.s_buffer_load_dword>(%resource, %soffset) {offset = 0} : (reg<amdgpu.sgpr x4>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %vmem = low.op<amdgpu.buffer_load_dword>(%resource, %vaddr, %soffset) {offset = 4} : (reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.vgpr>
  %s_mix = low.op<amdgpu.s_add_u32>(%s0, %smem) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>
  %v_mix = low.op<amdgpu.v_add_u32>(%v0, %vmem) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  low.op<amdgpu.buffer_store_dword>(%v_mix, %resource, %vaddr, %soffset) {offset = 8} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)
  low.op<amdgpu.s_waitcnt>() {vmcnt = 0, lgkmcnt = 0} : ()
  low.return %s_mix, %v_mix : reg<amdgpu.sgpr>, reg<amdgpu.vgpr>
}
)";

  ModulePtr module;
  IREE_ASSERT_OK(ParseAndVerify(
      iree_make_string_view(source.data(), source.size()), &module));
  const loom_op_t* low_function = FirstLowFunction(module.get());
  ASSERT_NE(low_function, nullptr);

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_schedule_sidecar_t schedule = {};
  loom_amdgpu_wait_plan_t plan = {};
  IREE_ASSERT_OK(
      BuildWaitPlan(module.get(), low_function, &arena, &schedule, &plan));

  bool saw_planned_vmem_load_use = false;
  bool saw_planned_smem_use = false;
  bool saw_explicit_store_wait = false;
  for (iree_host_size_t i = 0; i < plan.action_count; ++i) {
    const loom_amdgpu_wait_plan_action_t& action = plan.actions[i];
    if (action.kind == LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED &&
        action.reason == LOOM_AMDGPU_WAIT_PLAN_REASON_SSA_USE &&
        action.counter_id == LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD &&
        action.outstanding_before == 1) {
      saw_planned_vmem_load_use = true;
    }
    if (action.kind == LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED &&
        action.reason == LOOM_AMDGPU_WAIT_PLAN_REASON_SSA_USE &&
        action.counter_id == LOOM_AMDGPU_WAIT_COUNTER_SMEM &&
        action.outstanding_before == 1) {
      saw_planned_smem_use = true;
    }
    if (action.kind == LOOM_AMDGPU_WAIT_PLAN_ACTION_EXPLICIT &&
        action.reason == LOOM_AMDGPU_WAIT_PLAN_REASON_EXPLICIT_PACKET &&
        action.counter_id == LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE &&
        action.outstanding_before == 1) {
      saw_explicit_store_wait = true;
    }
  }
  EXPECT_TRUE(saw_planned_vmem_load_use);
  EXPECT_TRUE(saw_planned_smem_use);
  EXPECT_TRUE(saw_explicit_store_wait);
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuWaitPlanTest, PlansSplitWaitcntForGfx12AndGfx1250) {
  const char* preset_keys[] = {
      "amdgpu-gfx12",
      "amdgpu-gfx1250",
  };
  for (const char* preset_key : preset_keys) {
    std::string source = TargetPreamble("target", preset_key);
    source += R"(
low.func.def target(@target) @func(%resource: reg<amdgpu.sgpr x4>, %soffset: reg<amdgpu.sgpr>, %vaddr: reg<amdgpu.vgpr>, %v0: reg<amdgpu.vgpr>) -> (reg<amdgpu.vgpr>) {
  %vmem = low.op<amdgpu.buffer_load_dword>(%resource, %vaddr, %soffset) {offset = 0} : (reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.vgpr>
  %v_mix = low.op<amdgpu.v_add_u32>(%v0, %vmem) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  low.op<amdgpu.buffer_store_dword>(%v_mix, %resource, %vaddr, %soffset) {offset = 4} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)
  low.op<amdgpu.s_wait_loadcnt>() {loadcnt = 0} : ()
  low.op<amdgpu.s_wait_storecnt>() {storecnt = 0} : ()
  low.op<amdgpu.s_wait_alu>() {depctr = 0} : ()
  low.return %v_mix : reg<amdgpu.vgpr>
}
)";

    ModulePtr module;
    IREE_ASSERT_OK(ParseAndVerify(
        iree_make_string_view(source.data(), source.size()), &module))
        << preset_key;
    const loom_op_t* low_function = FirstLowFunction(module.get());
    ASSERT_NE(low_function, nullptr) << preset_key;

    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool_, &arena);
    loom_low_schedule_sidecar_t schedule = {};
    loom_amdgpu_wait_plan_t plan = {};
    IREE_ASSERT_OK(
        BuildWaitPlan(module.get(), low_function, &arena, &schedule, &plan))
        << preset_key;

    bool saw_planned_load_use = false;
    bool saw_explicit_load_wait = false;
    bool saw_explicit_store_wait = false;
    bool saw_explicit_alu_wait = false;
    for (iree_host_size_t i = 0; i < plan.action_count; ++i) {
      const loom_amdgpu_wait_plan_action_t& action = plan.actions[i];
      if (action.kind == LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED &&
          action.reason == LOOM_AMDGPU_WAIT_PLAN_REASON_SSA_USE &&
          action.counter_id == LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD &&
          action.outstanding_before == 1) {
        saw_planned_load_use = true;
      }
      if (action.kind == LOOM_AMDGPU_WAIT_PLAN_ACTION_EXPLICIT &&
          action.reason == LOOM_AMDGPU_WAIT_PLAN_REASON_EXPLICIT_PACKET &&
          action.counter_id == LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD) {
        saw_explicit_load_wait = true;
      }
      if (action.kind == LOOM_AMDGPU_WAIT_PLAN_ACTION_EXPLICIT &&
          action.reason == LOOM_AMDGPU_WAIT_PLAN_REASON_EXPLICIT_PACKET &&
          action.counter_id == LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE &&
          action.outstanding_before == 1) {
        saw_explicit_store_wait = true;
      }
      if (action.kind == LOOM_AMDGPU_WAIT_PLAN_ACTION_EXPLICIT &&
          action.reason == LOOM_AMDGPU_WAIT_PLAN_REASON_EXPLICIT_PACKET &&
          action.counter_id == LOOM_AMDGPU_WAIT_COUNTER_ALU) {
        saw_explicit_alu_wait = true;
      }
    }
    EXPECT_TRUE(saw_planned_load_use) << preset_key;
    EXPECT_TRUE(saw_explicit_load_wait) << preset_key;
    EXPECT_TRUE(saw_explicit_store_wait) << preset_key;
    EXPECT_TRUE(saw_explicit_alu_wait) << preset_key;
    iree_arena_deinitialize(&arena);
  }
}

TEST_F(AmdgpuWaitPlanTest, PlansStoreDrainAtBlockExit) {
  std::string source = TargetPreamble("gfx11_target", "amdgpu-gfx11");
  source += R"(
low.func.def target(@gfx11_target) @gfx11_func(%value: reg<amdgpu.vgpr>, %resource: reg<amdgpu.sgpr x4>, %soffset: reg<amdgpu.sgpr>, %vaddr: reg<amdgpu.vgpr>) -> (reg<amdgpu.vgpr>) {
  low.op<amdgpu.buffer_store_dword>(%value, %resource, %vaddr, %soffset) {offset = 0} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)
  low.return %value : reg<amdgpu.vgpr>
}
)";

  ModulePtr module;
  IREE_ASSERT_OK(ParseAndVerify(
      iree_make_string_view(source.data(), source.size()), &module));
  const loom_op_t* low_function = FirstLowFunction(module.get());
  ASSERT_NE(low_function, nullptr);

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_schedule_sidecar_t schedule = {};
  loom_amdgpu_wait_plan_t plan = {};
  IREE_ASSERT_OK(
      BuildWaitPlan(module.get(), low_function, &arena, &schedule, &plan));

  bool saw_block_exit_store_wait = false;
  for (iree_host_size_t i = 0; i < plan.action_count; ++i) {
    const loom_amdgpu_wait_plan_action_t& action = plan.actions[i];
    if (action.kind == LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED &&
        action.reason == LOOM_AMDGPU_WAIT_PLAN_REASON_BLOCK_EXIT &&
        action.counter_id == LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE &&
        action.outstanding_before == 1 &&
        action.consumer_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
      saw_block_exit_store_wait = true;
    }
  }
  EXPECT_TRUE(saw_block_exit_store_wait);
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuWaitPlanTest, PlansWorkgroupBarrierDrain) {
  std::string source = TargetPreamble("gfx11_target", "amdgpu-gfx11");
  source += R"(
low.func.def target(@gfx11_target) @gfx11_func(%addr: reg<amdgpu.vgpr>, %value: reg<amdgpu.vgpr x4>) -> (reg<amdgpu.vgpr x4>) {
  low.op<amdgpu.ds_write_b128>(%addr, %value) {offset = 0} : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr x4>)
  low.op<amdgpu.s_barrier>() : ()
  %loaded = low.op<amdgpu.ds_read_b128>(%addr) {offset = 0} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x4>
  low.return %loaded : reg<amdgpu.vgpr x4>
}
)";

  ModulePtr module;
  IREE_ASSERT_OK(ParseAndVerify(
      iree_make_string_view(source.data(), source.size()), &module));
  const loom_op_t* low_function = FirstLowFunction(module.get());
  ASSERT_NE(low_function, nullptr);

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_schedule_sidecar_t schedule = {};
  loom_amdgpu_wait_plan_t plan = {};
  IREE_ASSERT_OK(
      BuildWaitPlan(module.get(), low_function, &arena, &schedule, &plan));

  bool saw_barrier_store_wait = false;
  bool saw_return_load_wait = false;
  for (iree_host_size_t i = 0; i < plan.action_count; ++i) {
    const loom_amdgpu_wait_plan_action_t& action = plan.actions[i];
    if (action.kind == LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED &&
        action.reason == LOOM_AMDGPU_WAIT_PLAN_REASON_BARRIER &&
        action.counter_id == LOOM_AMDGPU_WAIT_COUNTER_LDS &&
        action.outstanding_before == 1) {
      saw_barrier_store_wait = true;
    }
    if (action.kind == LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED &&
        action.reason == LOOM_AMDGPU_WAIT_PLAN_REASON_SSA_USE &&
        action.counter_id == LOOM_AMDGPU_WAIT_COUNTER_LDS &&
        action.outstanding_before == 1) {
      saw_return_load_wait = true;
    }
  }
  EXPECT_TRUE(saw_barrier_store_wait);
  EXPECT_TRUE(saw_return_load_wait);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_amdgpu_wait_plan_format_json(&plan, &builder));
  std::string json = ToString(builder);
  iree_string_builder_deinitialize(&builder);
  EXPECT_NE(json.find("\"reason_name\":\"barrier\""), std::string::npos);
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuWaitPlanTest, RejectsUnknownWaitCounterId) {
  loom_low_schedule_node_t node = {};
  loom_low_schedule_hazard_use_t hazard = {
      .node_index = 0,
      .kind = LOOM_LOW_HAZARD_KIND_WAIT_COUNTER,
      .reference_kind = LOOM_LOW_HAZARD_REFERENCE_KIND_COUNTER,
      .reference_id = 99,
  };
  loom_low_schedule_sidecar_t schedule = {
      .nodes = &node,
      .node_count = 1,
      .hazard_uses = &hazard,
      .hazard_use_count = 1,
  };

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_amdgpu_wait_plan_t plan = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_amdgpu_wait_plan_build(&schedule, &arena, &plan));
  iree_arena_deinitialize(&arena);
}

}  // namespace
