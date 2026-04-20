// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/x86/low_registry.h"
#include "loom/target/arch/x86/lower.h"
#include "loom/tools/loom-check/execute.h"
#include "loom/tools/loom-check/test_util.h"

namespace loom {
namespace {

iree_status_t InitializeX86LowDescriptorRegistry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  (void)user_data;
  loom_x86_low_descriptor_registry_initialize(out_registry);
  return iree_ok_status();
}

iree_status_t InitializeX86LowLowerPolicyRegistry(
    void* user_data, loom_low_lower_policy_registry_t* out_registry) {
  (void)user_data;
  loom_x86_low_lower_policy_registry_initialize(out_registry);
  return iree_ok_status();
}

const loom_check_environment_t kX86LoomCheckEnvironment = {
    .register_context =
        {
            .fn = ::loom::testing::LoomCheckRegisterProductionContext,
            .user_data = nullptr,
        },
    .initialize_low_descriptor_registry =
        {
            .fn = InitializeX86LowDescriptorRegistry,
            .user_data = nullptr,
        },
    .initialize_low_lower_policy_registry =
        {
            .fn = InitializeX86LowLowerPolicyRegistry,
            .user_data = nullptr,
        },
};

class X86LoomCheckTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(harness_.Initialize(&kX86LoomCheckEnvironment));
  }

  ::loom::testing::LoomCheckHarness harness_;
};

constexpr const char* kX86MixedLowFunction =
    "target.preset @x86_target {key = \"x86-avx512\", source = @x86_mix}\n"
    "\n"
    "low.func.def target(@x86_target) @x86_mix(%base : reg<x86.gpr64>, "
    "%acc : reg<x86.zmm>, %lhs : reg<x86.zmm>, %rhs : reg<x86.zmm>) -> "
    "(reg<x86.zmm>) {\n"
    "  %loaded = low.op<x86.avx512.vmovdqu32.load.zmm>(%base) "
    "{disp32 = 0} : (reg<x86.gpr64>) -> reg<x86.zmm>\n"
    "  %sum = low.op<x86.avx512.vpaddd.zmm>(%loaded, %lhs) : "
    "(reg<x86.zmm>, reg<x86.zmm>) -> reg<x86.zmm>\n"
    "  %dot = low.op<x86.avx512.vpdpbusd.zmm>(%acc, %sum, %rhs) : "
    "(reg<x86.zmm>, reg<x86.zmm>, reg<x86.zmm>) -> reg<x86.zmm>\n"
    "  low.op<x86.avx512.vmovdqu32.store.zmm>(%dot, %base) {disp32 = 64} : "
    "(reg<x86.zmm>, reg<x86.gpr64>)\n"
    "  low.return %dot : reg<x86.zmm>\n"
    "}\n";

TEST_F(X86LoomCheckTest, DescriptorManifestUsesX86RegistryPackage) {
  loom_check_result_t result;
  IREE_ASSERT_OK(harness_.ExecuteFirst(
      IREE_SV("// RUN: emit low-descriptor-manifest x86.avx512.core\n"
              "func.def @unused() {\n"
              "}\n"
              "// ----\n"),
      IREE_SV("x86_low.loom-test"), &result));

  EXPECT_TRUE(result.has_actual_output);
  EXPECT_EQ(result.diagnostic_count, 0u)
      << harness_.DetailString(result) << "\n"
      << harness_.DiagnosticJsonString(result);
  const std::string actual_output = harness_.ActualOutputString(result);
  EXPECT_NE(actual_output.find("\"key\":\"x86.avx512.core\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"target\":\"x86\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"key\":\"x86.avx512.vpaddd.zmm\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"key\":\"x86.avx512.vmovdqu32.load.zmm\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"key\":\"x86.avx512.vpdpbusd.zmm\""),
            std::string::npos);
  EXPECT_EQ(actual_output.find("\"test.low.core\""), std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(X86LoomCheckTest, DescriptorManifestCanUsePackedDotPackage) {
  loom_check_result_t result;
  IREE_ASSERT_OK(harness_.ExecuteFirst(
      IREE_SV("// RUN: emit low-descriptor-manifest x86.packed_dot.core\n"
              "func.def @unused() {\n"
              "}\n"
              "// ----\n"),
      IREE_SV("x86_low.loom-test"), &result));

  EXPECT_TRUE(result.has_actual_output);
  EXPECT_EQ(result.diagnostic_count, 0u)
      << harness_.DetailString(result) << "\n"
      << harness_.DiagnosticJsonString(result);
  const std::string actual_output = harness_.ActualOutputString(result);
  EXPECT_NE(actual_output.find("\"key\":\"x86.packed_dot.core\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"key\":\"x86.avx512-vnni.vpdpbusd.512\""),
            std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(X86LoomCheckTest, ScheduleJsonUsesX86ResourcesAndLiveness) {
  loom_check_result_t result;
  std::string source = "// RUN: emit low-schedule-json @x86_mix\n";
  source += kX86MixedLowFunction;
  source += "// ----\n";
  IREE_ASSERT_OK(
      harness_.ExecuteFirst(iree_make_string_view(source.data(), source.size()),
                            IREE_SV("x86_low.loom-test"), &result));

  EXPECT_TRUE(result.has_actual_output);
  EXPECT_EQ(result.diagnostic_count, 0u)
      << harness_.DetailString(result) << "\n"
      << harness_.DiagnosticJsonString(result);
  const std::string actual_output = harness_.ActualOutputString(result);
  EXPECT_NE(actual_output.find("\"function\":\"x86_mix\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor_set\":\"x86.avx512.core\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"x86.avx512.vpaddd.zmm\""),
            std::string::npos);
  EXPECT_NE(
      actual_output.find("\"descriptor\":\"x86.avx512.vmovdqu32.load.zmm\""),
      std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"x86.avx512.vpdpbusd.zmm\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"resource_name\":\"x86.address\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"resource_name\":\"x86.load\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"resource_name\":\"x86.vector.dot\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"resource_name\":\"x86.store\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"schedule_class\":\"x86.vector.i32.512\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"schedule_class\":\"x86.vector.dot.512\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"capacity_per_cycle\":4"), std::string::npos);
  EXPECT_NE(actual_output.find("\"contention_group\":1"), std::string::npos);
  EXPECT_NE(actual_output.find("\"peak_units_per_cycle\":4"),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"register_class\":\"x86.zmm\""),
            std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(X86LoomCheckTest, AllocationJsonUsesX86PhysicalRegisterClasses) {
  loom_check_result_t result;
  std::string source = "// RUN: emit low-allocation-json @x86_mix x86.zmm=2\n";
  source += kX86MixedLowFunction;
  source += "// ----\n";
  IREE_ASSERT_OK(
      harness_.ExecuteFirst(iree_make_string_view(source.data(), source.size()),
                            IREE_SV("x86_low.loom-test"), &result));

  EXPECT_TRUE(result.has_actual_output);
  EXPECT_EQ(result.diagnostic_count, 0u)
      << harness_.DetailString(result) << "\n"
      << harness_.DiagnosticJsonString(result);
  const std::string actual_output = harness_.ActualOutputString(result);
  EXPECT_NE(actual_output.find("\"format\":\"loom.low.allocation.v0\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"function\":\"x86_mix\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor_set\":\"x86.avx512.core\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"register_class\":\"x86.gpr64\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"register_class\":\"x86.zmm\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"location\":{\"kind\":\"physical_register\""),
            std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(X86LoomCheckTest, PacketJsonUsesX86DescriptorsAndAllocation) {
  loom_check_result_t result;
  std::string source = "// RUN: emit low-packet-json @x86_mix x86.zmm=2\n";
  source += kX86MixedLowFunction;
  source += "// ----\n";
  IREE_ASSERT_OK(
      harness_.ExecuteFirst(iree_make_string_view(source.data(), source.size()),
                            IREE_SV("x86_low.loom-test"), &result));

  EXPECT_TRUE(result.has_actual_output);
  EXPECT_EQ(result.diagnostic_count, 0u)
      << harness_.DetailString(result) << "\n"
      << harness_.DiagnosticJsonString(result);
  const std::string actual_output = harness_.ActualOutputString(result);
  EXPECT_NE(actual_output.find("\"format\":\"loom.low.packet.v0\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"function\":\"x86_mix\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor_set\":\"x86.avx512.core\""),
            std::string::npos);
  EXPECT_NE(
      actual_output.find("\"descriptor\":\"x86.avx512.vmovdqu32.load.zmm\""),
      std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"x86.avx512.vpaddd.zmm\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"x86.avx512.vpdpbusd.zmm\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"x86.avx512.vmovdqu32.store."
                               "zmm\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor_ordinal\":"), std::string::npos);
  EXPECT_NE(actual_output.find("\"encoding_id\":"), std::string::npos);
  EXPECT_NE(actual_output.find("\"name\":\"disp32\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"kind\":\"signed\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"bit_width\":32"), std::string::npos);
  EXPECT_NE(actual_output.find("\"value\":64"), std::string::npos);
  EXPECT_NE(actual_output.find("\"kind\":\"physical_register\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"type\":\"reg<x86.zmm>\""), std::string::npos);
  EXPECT_EQ(actual_output.find("\"test.low.core\""), std::string::npos);
  loom_check_result_deinitialize(&result);
}

}  // namespace
}  // namespace loom
