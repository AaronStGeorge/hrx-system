// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/wait_packets.h"

#include <memory>
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
#include "loom/target/presets.h"
#include "loom/verify/verify.h"

namespace {

struct ModuleDeleter {
  void operator()(loom_module_t* module) const {
    if (module) {
      loom_module_free(module);
    }
  }
};

using ModulePtr = std::unique_ptr<loom_module_t, ModuleDeleter>;

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

std::string ToString(const iree_string_builder_t& builder) {
  if (iree_string_builder_size(&builder) == 0) {
    return std::string();
  }
  return std::string(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
}

std::string TargetPreamble(const char* target_symbol, const char* preset_key,
                           const char* function_symbol) {
  std::string source = "target.preset @";
  source += target_symbol;
  source += " {key = \"";
  source += preset_key;
  source += "\", source = @";
  source += function_symbol;
  source += "}\n\n";
  return source;
}

class AmdgpuWaitPacketsTest : public ::testing::Test {
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
        loom_text_parse(source, IREE_SV("amdgpu_wait_packets_test.loom"),
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

    const loom_target_preset_registry_t preset_registry =
        loom_target_low_descriptor_registry_presets(&low_registry_);
    iree_host_size_t expanded_preset_count = 0;
    IREE_RETURN_IF_ERROR(loom_target_expand_presets(
        module.get(), &preset_registry, &expanded_preset_count));
    if (expanded_preset_count != 1) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "test source expanded %zu target presets, "
                              "expected exactly one",
                              expanded_preset_count);
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

  iree_status_t BuildWaitPackets(
      const loom_module_t* module, const loom_op_t* low_function,
      iree_arena_allocator_t* arena, loom_low_schedule_sidecar_t* out_schedule,
      loom_amdgpu_wait_plan_t* out_wait_plan,
      loom_amdgpu_wait_packet_plan_t* out_packet_plan) {
    loom_low_schedule_options_t schedule_options = {
        .descriptor_registry = &low_registry_.registry,
        .strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
    };
    IREE_RETURN_IF_ERROR(loom_low_schedule_function(
        module, low_function, &schedule_options, arena, out_schedule));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_wait_plan_build(out_schedule, arena, out_wait_plan));
    return loom_amdgpu_wait_packet_plan_build(out_wait_plan, arena,
                                              out_packet_plan);
  }

  iree_status_t LookupDescriptorSet(
      iree_string_view_t key,
      const loom_low_descriptor_set_t** out_descriptor_set) {
    return loom_low_descriptor_registry_lookup(&low_registry_.registry, key,
                                               out_descriptor_set);
  }

  iree_arena_block_pool_t block_pool_ = {};
  loom_context_t context_ = {};
  loom_target_low_descriptor_registry_t low_registry_ = {};
  bool block_pool_initialized_ = false;
  bool context_initialized_ = false;
};

TEST_F(AmdgpuWaitPacketsTest, CoalescesCombinedMemoryWaitForGfx950AndGfx11) {
  const char* preset_keys[] = {
      "amdgpu-gfx950",
      "amdgpu-gfx11",
  };
  for (const char* preset_key : preset_keys) {
    std::string source = TargetPreamble("target", preset_key, "func");
    source += R"(
low.func.def target(@target) @func(%value : reg<amdgpu.vgpr>, %resource : reg<amdgpu.sgpr x4>, %soffset : reg<amdgpu.sgpr>, %vaddr : reg<amdgpu.vgpr>) -> (reg<amdgpu.vgpr>) {
  %loaded = low.op<amdgpu.buffer_load_dword>(%resource, %vaddr, %soffset) {offset = 0} : (reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.vgpr>
  low.op<amdgpu.buffer_store_dword>(%value, %resource, %vaddr, %soffset) {offset = 4} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)
  low.return %loaded : reg<amdgpu.vgpr>
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
    loom_amdgpu_wait_plan_t wait_plan = {};
    loom_amdgpu_wait_packet_plan_t packet_plan = {};
    IREE_ASSERT_OK(BuildWaitPackets(module.get(), low_function, &arena,
                                    &schedule, &wait_plan, &packet_plan))
        << preset_key;

    EXPECT_EQ(wait_plan.action_count, 2u) << preset_key;
    ASSERT_EQ(packet_plan.packet_count, 1u) << preset_key;
    const loom_amdgpu_wait_packet_t& packet = packet_plan.packets[0];
    EXPECT_EQ(ToString(packet.descriptor_key), "amdgpu.s_waitcnt")
        << preset_key;
    EXPECT_EQ(packet.counter_mask, LOOM_AMDGPU_WAIT_COUNTER_MASK_MEMORY)
        << preset_key;
    EXPECT_EQ(packet.source_action_count, 2u) << preset_key;
    ASSERT_EQ(packet.immediate_count, 2u) << preset_key;
    EXPECT_EQ(packet_plan.immediates[packet.immediate_start].value, 0u)
        << preset_key;
    EXPECT_EQ(packet_plan.immediates[packet.immediate_start + 1].value, 0u)
        << preset_key;

    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    IREE_ASSERT_OK(
        loom_amdgpu_wait_packet_plan_format_json(&packet_plan, &builder))
        << preset_key;
    std::string json = ToString(builder);
    iree_string_builder_deinitialize(&builder);

    EXPECT_NE(json.find("\"format\":\"loom.amdgpu.wait_packet_plan.v0\""),
              std::string::npos)
        << preset_key;
    EXPECT_NE(json.find("\"descriptor\":\"amdgpu.s_waitcnt\""),
              std::string::npos)
        << preset_key;
    EXPECT_NE(json.find("\"source_action_count\":2"), std::string::npos)
        << preset_key;
    EXPECT_NE(json.find("\"name\":\"vmcnt\""), std::string::npos) << preset_key;
    EXPECT_NE(json.find("\"name\":\"lgkmcnt\""), std::string::npos)
        << preset_key;
    iree_arena_deinitialize(&arena);
  }
}

TEST_F(AmdgpuWaitPacketsTest, MaterializesSplitMemoryWaitsForGfx12AndGfx1250) {
  const char* preset_keys[] = {
      "amdgpu-gfx12",
      "amdgpu-gfx1250",
  };
  for (const char* preset_key : preset_keys) {
    std::string source = TargetPreamble("target", preset_key, "func");
    source += R"(
low.func.def target(@target) @func(%value : reg<amdgpu.vgpr>, %resource : reg<amdgpu.sgpr x4>, %soffset : reg<amdgpu.sgpr>, %vaddr : reg<amdgpu.vgpr>) -> (reg<amdgpu.vgpr>) {
  %loaded = low.op<amdgpu.buffer_load_dword>(%resource, %vaddr, %soffset) {offset = 0} : (reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.vgpr>
  low.op<amdgpu.buffer_store_dword>(%value, %resource, %vaddr, %soffset) {offset = 4} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)
  low.return %loaded : reg<amdgpu.vgpr>
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
    loom_amdgpu_wait_plan_t wait_plan = {};
    loom_amdgpu_wait_packet_plan_t packet_plan = {};
    IREE_ASSERT_OK(BuildWaitPackets(module.get(), low_function, &arena,
                                    &schedule, &wait_plan, &packet_plan))
        << preset_key;

    EXPECT_EQ(wait_plan.action_count, 2u) << preset_key;
    ASSERT_EQ(packet_plan.packet_count, 2u) << preset_key;
    EXPECT_EQ(ToString(packet_plan.packets[0].descriptor_key),
              "amdgpu.s_wait_loadcnt")
        << preset_key;
    EXPECT_EQ(packet_plan.packets[0].counter_mask,
              LOOM_AMDGPU_WAIT_COUNTER_MASK_LOAD)
        << preset_key;
    EXPECT_EQ(packet_plan.packets[0].source_action_count, 2u) << preset_key;
    EXPECT_EQ(ToString(packet_plan.packets[1].descriptor_key),
              "amdgpu.s_wait_storecnt")
        << preset_key;
    EXPECT_EQ(packet_plan.packets[1].counter_mask,
              LOOM_AMDGPU_WAIT_COUNTER_MASK_STORE)
        << preset_key;
    EXPECT_EQ(packet_plan.packets[1].source_action_count, 2u) << preset_key;
    iree_arena_deinitialize(&arena);
  }
}

TEST_F(AmdgpuWaitPacketsTest, SelectsAluWaitDescriptorPerTarget) {
  struct Case {
    const char* descriptor_set_key;
    const char* expected_descriptor;
  };
  const Case cases[] = {
      {"amdgpu.gfx11.core", "amdgpu.s_waitcnt_depctr"},
      {"amdgpu.gfx12.core", "amdgpu.s_wait_alu"},
      {"amdgpu.gfx1250.core", "amdgpu.s_wait_alu"},
  };
  for (const Case& test_case : cases) {
    const loom_low_descriptor_set_t* descriptor_set = nullptr;
    IREE_ASSERT_OK(LookupDescriptorSet(
        iree_make_cstring_view(test_case.descriptor_set_key), &descriptor_set));
    ASSERT_NE(descriptor_set, nullptr) << test_case.descriptor_set_key;

    loom_low_schedule_sidecar_t schedule = {};
    schedule.target.descriptor_set = descriptor_set;
    loom_amdgpu_wait_plan_action_t action = {
        .kind = LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED,
        .reason = LOOM_AMDGPU_WAIT_PLAN_REASON_SSA_USE,
        .counter_id = LOOM_AMDGPU_WAIT_COUNTER_ALU,
        .target_count = 0,
    };
    loom_amdgpu_wait_plan_t wait_plan = {
        .schedule = &schedule,
        .actions = &action,
        .action_count = 1,
    };

    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool_, &arena);
    loom_amdgpu_wait_packet_plan_t packet_plan = {};
    IREE_ASSERT_OK(
        loom_amdgpu_wait_packet_plan_build(&wait_plan, &arena, &packet_plan))
        << test_case.descriptor_set_key;
    ASSERT_EQ(packet_plan.packet_count, 1u) << test_case.descriptor_set_key;
    EXPECT_EQ(ToString(packet_plan.packets[0].descriptor_key),
              test_case.expected_descriptor)
        << test_case.descriptor_set_key;
    EXPECT_EQ(packet_plan.packets[0].counter_mask,
              LOOM_AMDGPU_WAIT_COUNTER_MASK_ALU)
        << test_case.descriptor_set_key;
    iree_arena_deinitialize(&arena);
  }
}

TEST_F(AmdgpuWaitPacketsTest, RejectsUnrepresentableAluWaitOnGfx950) {
  const loom_low_descriptor_set_t* descriptor_set = nullptr;
  IREE_ASSERT_OK(
      LookupDescriptorSet(IREE_SV("amdgpu.gfx950.core"), &descriptor_set));
  ASSERT_NE(descriptor_set, nullptr);

  loom_low_schedule_sidecar_t schedule = {};
  schedule.target.descriptor_set = descriptor_set;
  loom_amdgpu_wait_plan_action_t action = {
      .kind = LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED,
      .reason = LOOM_AMDGPU_WAIT_PLAN_REASON_SSA_USE,
      .counter_id = LOOM_AMDGPU_WAIT_COUNTER_ALU,
      .target_count = 0,
  };
  loom_amdgpu_wait_plan_t wait_plan = {
      .schedule = &schedule,
      .actions = &action,
      .action_count = 1,
  };

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_amdgpu_wait_packet_plan_t packet_plan = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      loom_amdgpu_wait_packet_plan_build(&wait_plan, &arena, &packet_plan));
  iree_arena_deinitialize(&arena);
}

}  // namespace
