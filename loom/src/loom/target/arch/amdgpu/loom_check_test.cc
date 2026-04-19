// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amdgpu/low_registry.h"
#include "loom/tools/loom-check/execute.h"
#include "loom/tools/loom-check/test_util.h"

namespace loom {
namespace {

iree_status_t InitializeAmdgpuLowDescriptorRegistry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  (void)user_data;
  loom_amdgpu_low_descriptor_registry_initialize(out_registry);
  return iree_ok_status();
}

const loom_check_environment_t kAmdgpuLoomCheckEnvironment = {
    .register_context =
        {
            .fn = ::loom::testing::LoomCheckRegisterProductionContext,
            .user_data = nullptr,
        },
    .initialize_low_descriptor_registry =
        {
            .fn = InitializeAmdgpuLowDescriptorRegistry,
            .user_data = nullptr,
        },
};

class AmdgpuLoomCheckTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(harness_.Initialize(&kAmdgpuLoomCheckEnvironment));
  }

  ::loom::testing::LoomCheckHarness harness_;
};

constexpr const char* kAmdgpuGfx11MixedLowFunction =
    "target.preset @gfx11_target {key = \"amdgpu-gfx11\", source = "
    "@gfx11_mix}\n"
    "\n"
    "low.func.def target(@gfx11_target) @gfx11_mix(%s0 : reg<amdgpu.sgpr>, "
    "%s1 : reg<amdgpu.sgpr>, %v0 : reg<amdgpu.vgpr>, "
    "%v1 : reg<amdgpu.vgpr>, %a : reg<amdgpu.vgpr x4>, "
    "%b : reg<amdgpu.vgpr x4>, %acc : reg<amdgpu.vgpr x8>, "
    "%resource : reg<amdgpu.sgpr x4>, %soffset : reg<amdgpu.sgpr>, "
    "%vaddr : reg<amdgpu.vgpr>) -> "
    "(reg<amdgpu.sgpr>, reg<amdgpu.vgpr>, reg<amdgpu.vgpr x8>) {\n"
    "  %s_sum = low.op<amdgpu.s_add_u32>(%s0, %s1) : "
    "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
    "  %v_sum = low.op<amdgpu.v_add_u32>(%v0, %v1) : "
    "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
    "  %smem = low.op<amdgpu.s_buffer_load_dword>(%resource, %soffset) "
    "{offset = 0} : (reg<amdgpu.sgpr x4>, reg<amdgpu.sgpr>) -> "
    "reg<amdgpu.sgpr>\n"
    "  %vmem = low.op<amdgpu.buffer_load_dword>(%resource, %vaddr, "
    "%soffset) {offset = 4} : (reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, "
    "reg<amdgpu.sgpr>) -> reg<amdgpu.vgpr>\n"
    "  %s_mix = low.op<amdgpu.s_add_u32>(%s_sum, %smem) : "
    "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
    "  %v_mix = low.op<amdgpu.v_add_u32>(%v_sum, %vmem) : "
    "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
    "  %matrix0 = low.op<amdgpu.v_wmma_f32_16x16x16_f16>(%a, %b, %acc) : "
    "(reg<amdgpu.vgpr x4>, reg<amdgpu.vgpr x4>, reg<amdgpu.vgpr x8>) -> "
    "reg<amdgpu.vgpr x8>\n"
    "  %matrix1 = low.op<amdgpu.v_wmma_f32_16x16x16_f16>(%a, %b, "
    "%matrix0) : (reg<amdgpu.vgpr x4>, reg<amdgpu.vgpr x4>, "
    "reg<amdgpu.vgpr x8>) -> reg<amdgpu.vgpr x8>\n"
    "  low.op<amdgpu.buffer_store_dword>(%v_mix, %resource, %vaddr, "
    "%soffset) {offset = 8} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x4>, "
    "reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
    "  low.op<amdgpu.s_waitcnt>() {vmcnt = 0, lgkmcnt = 0} : ()\n"
    "  low.op<amdgpu.s_waitcnt_depctr>() {depctr = 0} : ()\n"
    "  low.op<amdgpu.s_wait_idle>() : ()\n"
    "  low.return %s_mix, %v_mix, %matrix1 : reg<amdgpu.sgpr>, "
    "reg<amdgpu.vgpr>, reg<amdgpu.vgpr x8>\n"
    "}\n";

TEST_F(AmdgpuLoomCheckTest, DescriptorManifestUsesGfx11RegistryPackage) {
  loom_check_result_t result;
  IREE_ASSERT_OK(harness_.ExecuteFirst(
      IREE_SV("// RUN: emit low-descriptor-manifest amdgpu.gfx11.core\n"
              "func.def @unused() {\n"
              "}\n"
              "// ----\n"),
      IREE_SV("amdgpu_low.loom-test"), &result));

  EXPECT_TRUE(result.has_actual_output);
  EXPECT_EQ(result.diagnostic_count, 0u);
  const std::string actual_output = harness_.ActualOutputString(result);
  EXPECT_NE(actual_output.find("\"key\":\"amdgpu.gfx11.core\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"target\":\"amdgpu\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"key\":\"amdgpu.s_add_u32\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"key\":\"amdgpu.v_add_u32\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"key\":\"amdgpu.v_wmma_f32_16x16x16_f16\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"key\":\"amdgpu.s_waitcnt\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"key\":\"amdgpu.s_waitcnt_depctr\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"key\":\"amdgpu.s_wait_idle\""),
            std::string::npos);
  EXPECT_EQ(actual_output.find("\"test.low.core\""), std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(AmdgpuLoomCheckTest, ScheduleJsonUsesGfx11ResourcesAndLiveness) {
  loom_check_result_t result;
  std::string source = "// RUN: emit low-schedule-json @gfx11_mix\n";
  source += kAmdgpuGfx11MixedLowFunction;
  source += "// ----\n";
  IREE_ASSERT_OK(
      harness_.ExecuteFirst(iree_make_string_view(source.data(), source.size()),
                            IREE_SV("amdgpu_low.loom-test"), &result));

  EXPECT_TRUE(result.has_actual_output);
  EXPECT_EQ(result.diagnostic_count, 0u);
  const std::string actual_output = harness_.ActualOutputString(result);
  EXPECT_NE(actual_output.find("\"function\":\"gfx11_mix\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor_set\":\"amdgpu.gfx11.core\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"amdgpu.s_add_u32\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"amdgpu.v_add_u32\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"amdgpu.s_buffer_load_dword\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"amdgpu.buffer_load_dword\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"amdgpu.buffer_store_dword\""),
            std::string::npos);
  EXPECT_NE(
      actual_output.find("\"descriptor\":\"amdgpu.v_wmma_f32_16x16x16_f16\""),
      std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"amdgpu.s_waitcnt\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"amdgpu.s_waitcnt_depctr\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"amdgpu.s_wait_idle\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"resource_name\":\"amdgpu.salu\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"resource_name\":\"amdgpu.valu\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"resource_name\":\"amdgpu.smem\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"resource_name\":\"amdgpu.vmem.load\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"resource_name\":\"amdgpu.vmem.store\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"resource_name\":\"amdgpu.wmma\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"schedule_class\":\"amdgpu.wait.memory\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"schedule_class\":\"amdgpu.wait.alu\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"schedule_class\":\"amdgpu.wait.idle\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"resource_name\":\"amdgpu.control\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"scheduled_hazard_uses\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"kind_name\":\"wait_counter\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"kind_name\":\"min_distance\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"reference_kind_name\":\"counter\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"reference_kind_name\":\"resource\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"scheduled_hazard_gaps\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"required_delay\":1"), std::string::npos);
  EXPECT_NE(actual_output.find("\"register_class\":\"amdgpu.sgpr\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"register_class\":\"amdgpu.vgpr\""),
            std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(AmdgpuLoomCheckTest, AllocationJsonUsesGfx11PhysicalRegisterClasses) {
  loom_check_result_t result;
  std::string source =
      "// RUN: emit low-allocation-json @gfx11_mix amdgpu.vgpr=64\n";
  source += kAmdgpuGfx11MixedLowFunction;
  source += "// ----\n";
  IREE_ASSERT_OK(
      harness_.ExecuteFirst(iree_make_string_view(source.data(), source.size()),
                            IREE_SV("amdgpu_low.loom-test"), &result));

  EXPECT_TRUE(result.has_actual_output);
  EXPECT_EQ(result.diagnostic_count, 0u);
  const std::string actual_output = harness_.ActualOutputString(result);
  EXPECT_NE(actual_output.find("\"format\":\"loom.low.allocation.v0\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"function\":\"gfx11_mix\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor_set\":\"amdgpu.gfx11.core\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"register_class\":\"amdgpu.sgpr\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"register_class\":\"amdgpu.vgpr\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"location\":{\"kind\":\"physical_register\""),
            std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(AmdgpuLoomCheckTest, PacketJsonUsesGfx11DescriptorsAndAllocation) {
  loom_check_result_t result;
  std::string source =
      "// RUN: emit low-packet-json @gfx11_mix amdgpu.vgpr=64\n";
  source += kAmdgpuGfx11MixedLowFunction;
  source += "// ----\n";
  IREE_ASSERT_OK(
      harness_.ExecuteFirst(iree_make_string_view(source.data(), source.size()),
                            IREE_SV("amdgpu_low.loom-test"), &result));

  EXPECT_TRUE(result.has_actual_output);
  EXPECT_EQ(result.diagnostic_count, 0u);
  const std::string actual_output = harness_.ActualOutputString(result);
  EXPECT_NE(actual_output.find("\"format\":\"loom.low.packet.v0\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"function\":\"gfx11_mix\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor_set\":\"amdgpu.gfx11.core\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"amdgpu.s_add_u32\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"amdgpu.v_add_u32\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"amdgpu.s_buffer_load_dword\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"amdgpu.buffer_load_dword\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"amdgpu.buffer_store_dword\""),
            std::string::npos);
  EXPECT_NE(
      actual_output.find("\"descriptor\":\"amdgpu.v_wmma_f32_16x16x16_f16\""),
      std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"amdgpu.s_waitcnt\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"amdgpu.s_waitcnt_depctr\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"amdgpu.s_wait_idle\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor_ordinal\":"), std::string::npos);
  EXPECT_NE(actual_output.find("\"encoding_id\":"), std::string::npos);
  EXPECT_NE(actual_output.find("\"name\":\"vmcnt\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"name\":\"lgkmcnt\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"kind\":\"unsigned\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"value\":0"), std::string::npos);
  EXPECT_NE(actual_output.find("\"value\":4"), std::string::npos);
  EXPECT_NE(actual_output.find("\"value\":8"), std::string::npos);
  EXPECT_NE(actual_output.find("\"hazard_gap_count\":"), std::string::npos);
  EXPECT_NE(actual_output.find("\"hazard_gaps\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"required_delay\":1"), std::string::npos);
  EXPECT_NE(actual_output.find("\"kind\":\"physical_register\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"type\":\"reg<amdgpu.vgpr x8>\""),
            std::string::npos);
  EXPECT_EQ(actual_output.find("\"test.low.core\""), std::string::npos);
  loom_check_result_deinitialize(&result);
}

}  // namespace
}  // namespace loom
