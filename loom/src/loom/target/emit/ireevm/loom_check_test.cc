// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/emit/ireevm/low_registry.h"
#include "loom/target/emit/ireevm/lower.h"
#include "loom/tools/loom-check/execute.h"
#include "loom/tools/loom-check/test_util.h"

namespace loom {
namespace {

iree_status_t InitializeIreeVmLowDescriptorRegistry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  (void)user_data;
  loom_ireevm_low_descriptor_registry_initialize(out_registry);
  return iree_ok_status();
}

iree_status_t InitializeIreeVmLowLowerPolicyRegistry(
    void* user_data, loom_low_lower_policy_registry_t* out_registry) {
  (void)user_data;
  loom_ireevm_low_lower_policy_registry_initialize(out_registry);
  return iree_ok_status();
}

const loom_check_environment_t kIreeVmLoomCheckEnvironment = {
    .register_context =
        {
            .fn = ::loom::testing::LoomCheckRegisterProductionContext,
            .user_data = nullptr,
        },
    .initialize_low_descriptor_registry =
        {
            .fn = InitializeIreeVmLowDescriptorRegistry,
            .user_data = nullptr,
        },
    .initialize_low_lower_policy_registry =
        {
            .fn = InitializeIreeVmLowLowerPolicyRegistry,
            .user_data = nullptr,
        },
};

class IreeVmLoomCheckTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(harness_.Initialize(&kIreeVmLoomCheckEnvironment));
  }

  ::loom::testing::LoomCheckHarness harness_;
};

constexpr const char* kVmMixedLowFunction =
    "target.preset @vm_target {key = \"iree-vm\", source = @vm_mix}\n"
    "\n"
    "low.func.def target(@vm_target) @vm_mix(%lhs : reg<vm.i32>, "
    "%rhs : reg<vm.i32>) -> (reg<vm.i32>) {\n"
    "  %c0 = low.const<iree.vm.const.i32> {i32_value = 7} : reg<vm.i32>\n"
    "  %sum = low.op<iree.vm.add.i32>(%lhs, %c0) : "
    "(reg<vm.i32>, reg<vm.i32>) -> reg<vm.i32>\n"
    "  %eq = low.op<iree.vm.cmp.eq.i32>(%sum, %rhs) : "
    "(reg<vm.i32>, reg<vm.i32>) -> reg<vm.i32>\n"
    "  low.cond_br %eq, ^then, ^else : reg<vm.i32>\n"
    "^then:\n"
    "  %call = low.op<iree.vm.call.import.i32>(%sum) "
    "{callee_ordinal = 0} : (reg<vm.i32>) -> reg<vm.i32>\n"
    "  low.br ^join(%call : reg<vm.i32>)\n"
    "^else:\n"
    "  %sub = low.op<iree.vm.sub.i32>(%rhs, %sum) : "
    "(reg<vm.i32>, reg<vm.i32>) -> reg<vm.i32>\n"
    "  low.br ^join(%sub : reg<vm.i32>)\n"
    "^join(%result : reg<vm.i32>):\n"
    "  low.return %result : reg<vm.i32>\n"
    "}\n";

TEST_F(IreeVmLoomCheckTest, DescriptorManifestUsesVmRegistryPackage) {
  loom_check_result_t result;
  IREE_ASSERT_OK(harness_.ExecuteFirst(
      IREE_SV("// RUN: emit low-descriptor-manifest iree.vm.core\n"
              "func.def @unused() {\n"
              "}\n"
              "// ----\n"),
      IREE_SV("ireevm_low.loom-test"), &result));

  EXPECT_TRUE(result.has_actual_output);
  EXPECT_EQ(result.diagnostic_count, 0u);
  const std::string actual_output = harness_.ActualOutputString(result);
  EXPECT_FALSE(actual_output.empty());
  loom_check_result_deinitialize(&result);
}

TEST_F(IreeVmLoomCheckTest, ScheduleJsonUsesVmResourcesAndLiveness) {
  loom_check_result_t result;
  std::string source = "// RUN: emit low-schedule-json @vm_mix\n";
  source += kVmMixedLowFunction;
  source += "// ----\n";
  IREE_ASSERT_OK(
      harness_.ExecuteFirst(iree_make_string_view(source.data(), source.size()),
                            IREE_SV("ireevm_low.loom-test"), &result));

  EXPECT_TRUE(result.has_actual_output);
  EXPECT_EQ(result.diagnostic_count, 0u);
  const std::string actual_output = harness_.ActualOutputString(result);
  EXPECT_NE(actual_output.find("\"function\":\"vm_mix\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor_set\":\"iree.vm.core\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"iree.vm.add.i32\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"iree.vm.call.import.i32\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"resource_name\":\"vm.alu\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"resource_name\":\"vm.call\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"is_cfg\":true"), std::string::npos);
  EXPECT_NE(actual_output.find("\"register_class\":\"vm.i32\""),
            std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(IreeVmLoomCheckTest, AllocationJsonUsesVmRegisterClasses) {
  loom_check_result_t result;
  std::string source = "// RUN: emit low-allocation-json @vm_mix vm.i32=2\n";
  source += kVmMixedLowFunction;
  source += "// ----\n";
  IREE_ASSERT_OK(
      harness_.ExecuteFirst(iree_make_string_view(source.data(), source.size()),
                            IREE_SV("ireevm_low.loom-test"), &result));

  EXPECT_TRUE(result.has_actual_output);
  EXPECT_EQ(result.diagnostic_count, 0u);
  const std::string actual_output = harness_.ActualOutputString(result);
  EXPECT_NE(actual_output.find("\"format\":\"loom.low.allocation.v0\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"function\":\"vm_mix\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor_set\":\"iree.vm.core\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"register_class\":\"vm.i32\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"location\":{\"kind\":\"spill_slot\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"spill_plan_count\":"), std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(IreeVmLoomCheckTest, PacketJsonUsesVmDescriptorsAndAllocation) {
  loom_check_result_t result;
  std::string source = "// RUN: emit low-packet-json @vm_mix vm.i32=2\n";
  source += kVmMixedLowFunction;
  source += "// ----\n";
  IREE_ASSERT_OK(
      harness_.ExecuteFirst(iree_make_string_view(source.data(), source.size()),
                            IREE_SV("ireevm_low.loom-test"), &result));

  EXPECT_TRUE(result.has_actual_output);
  EXPECT_EQ(result.diagnostic_count, 0u);
  const std::string actual_output = harness_.ActualOutputString(result);
  EXPECT_NE(actual_output.find("\"format\":\"loom.low.packet.v0\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"function\":\"vm_mix\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor_set\":\"iree.vm.core\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"iree.vm.add.i32\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"iree.vm.call.import.i32\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor_ordinal\":"), std::string::npos);
  EXPECT_NE(actual_output.find("\"encoding_id\":"), std::string::npos);
  EXPECT_NE(actual_output.find("\"name\":\"i32_value\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"kind\":\"signed\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"bit_width\":32"), std::string::npos);
  EXPECT_NE(actual_output.find("\"value\":7"), std::string::npos);
  EXPECT_NE(actual_output.find("\"kind\":\"spill_slot\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"successors\":[{\"index\":0,\"block\":1},"
                               "{\"index\":1,\"block\":2}]"),
            std::string::npos);
  loom_check_result_deinitialize(&result);
}

}  // namespace
}  // namespace loom
