// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/wasm/low_registry.h"
#include "loom/tools/loom-check/execute.h"
#include "loom/tools/loom-check/test_util.h"

namespace loom {
namespace {

iree_status_t InitializeWasmLowDescriptorRegistry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  (void)user_data;
  loom_wasm_low_descriptor_registry_initialize(out_registry);
  return iree_ok_status();
}

const loom_check_environment_t kWasmLoomCheckEnvironment = {
    .register_context =
        {
            .fn = ::loom::testing::LoomCheckRegisterProductionContext,
            .user_data = nullptr,
        },
    .initialize_low_descriptor_registry =
        {
            .fn = InitializeWasmLowDescriptorRegistry,
            .user_data = nullptr,
        },
};

class WasmLoomCheckTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(harness_.Initialize(&kWasmLoomCheckEnvironment));
  }

  ::loom::testing::LoomCheckHarness harness_;
};

constexpr const char* kWasmMixedLowFunction =
    "target.preset @wasm_target {key = \"wasm-simd128\", source = "
    "@wasm_mix}\n"
    "\n"
    "low.func.def target(@wasm_target) @wasm_mix(%addr : reg<wasm.i32>, "
    "%lhs : reg<wasm.v128>, %rhs : reg<wasm.v128>) -> (reg<wasm.v128>) {\n"
    "  %loaded = low.op<wasm.v128.load>(%addr) : (reg<wasm.i32>) -> "
    "reg<wasm.v128>\n"
    "  %sum = low.op<wasm.i32x4.add>(%loaded, %lhs) : "
    "(reg<wasm.v128>, reg<wasm.v128>) -> reg<wasm.v128>\n"
    "  low.op<wasm.v128.store>(%addr, %sum) : "
    "(reg<wasm.i32>, reg<wasm.v128>)\n"
    "  %out = low.op<wasm.i32x4.mul>(%sum, %rhs) : "
    "(reg<wasm.v128>, reg<wasm.v128>) -> reg<wasm.v128>\n"
    "  low.return %out : reg<wasm.v128>\n"
    "}\n";

TEST_F(WasmLoomCheckTest, DescriptorManifestUsesWasmRegistryPackage) {
  loom_check_result_t result;
  IREE_ASSERT_OK(harness_.ExecuteFirst(
      IREE_SV("// RUN: emit low-descriptor-manifest wasm.core.simd128\n"
              "func.def @unused() {\n"
              "}\n"
              "// ----\n"),
      IREE_SV("wasm_low.loom-test"), &result));

  EXPECT_TRUE(result.has_actual_output);
  EXPECT_EQ(result.diagnostic_count, 0u);
  const std::string actual_output = harness_.ActualOutputString(result);
  EXPECT_NE(actual_output.find("\"key\":\"wasm.core.simd128\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"target\":\"wasm\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"key\":\"wasm.i32x4.add\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"key\":\"wasm.v128.load\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"key\":\"wasm.v128.store\""),
            std::string::npos);
  EXPECT_EQ(actual_output.find("\"test.low.core\""), std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(WasmLoomCheckTest, ScheduleJsonUsesWasmResourcesAndLiveness) {
  loom_check_result_t result;
  std::string source = "// RUN: emit low-schedule-json @wasm_mix\n";
  source += kWasmMixedLowFunction;
  source += "// ----\n";
  IREE_ASSERT_OK(
      harness_.ExecuteFirst(iree_make_string_view(source.data(), source.size()),
                            IREE_SV("wasm_low.loom-test"), &result));

  EXPECT_TRUE(result.has_actual_output);
  EXPECT_EQ(result.diagnostic_count, 0u);
  const std::string actual_output = harness_.ActualOutputString(result);
  EXPECT_NE(actual_output.find("\"function\":\"wasm_mix\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor_set\":\"wasm.core.simd128\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"wasm.i32x4.add\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"wasm.v128.load\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"wasm.v128.store\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"resource_name\":\"wasm.simd\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"resource_name\":\"wasm.load\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"resource_name\":\"wasm.store\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"register_class\":\"wasm.v128\""),
            std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(WasmLoomCheckTest, AllocationJsonUsesWasmRegisterClasses) {
  loom_check_result_t result;
  std::string source = "// RUN: emit low-allocation-json @wasm_mix\n";
  source += kWasmMixedLowFunction;
  source += "// ----\n";
  IREE_ASSERT_OK(
      harness_.ExecuteFirst(iree_make_string_view(source.data(), source.size()),
                            IREE_SV("wasm_low.loom-test"), &result));

  EXPECT_TRUE(result.has_actual_output);
  EXPECT_EQ(result.diagnostic_count, 0u);
  const std::string actual_output = harness_.ActualOutputString(result);
  EXPECT_NE(actual_output.find("\"format\":\"loom.low.allocation.v0\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"function\":\"wasm_mix\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor_set\":\"wasm.core.simd128\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"register_class\":\"wasm.i32\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"register_class\":\"wasm.v128\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"location\":{\"kind\":\"target_id\""),
            std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(WasmLoomCheckTest, PacketJsonUsesWasmDescriptorsAndAllocation) {
  loom_check_result_t result;
  std::string source = "// RUN: emit low-packet-json @wasm_mix\n";
  source += kWasmMixedLowFunction;
  source += "// ----\n";
  IREE_ASSERT_OK(
      harness_.ExecuteFirst(iree_make_string_view(source.data(), source.size()),
                            IREE_SV("wasm_low.loom-test"), &result));

  EXPECT_TRUE(result.has_actual_output);
  EXPECT_EQ(result.diagnostic_count, 0u);
  const std::string actual_output = harness_.ActualOutputString(result);
  EXPECT_NE(actual_output.find("\"format\":\"loom.low.packet.v0\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"function\":\"wasm_mix\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor_set\":\"wasm.core.simd128\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"wasm.v128.load\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"wasm.i32x4.add\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"wasm.v128.store\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"descriptor\":\"wasm.i32x4.mul\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"kind\":\"target_id\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"type\":\"reg<wasm.v128>\""),
            std::string::npos);
  EXPECT_EQ(actual_output.find("\"test.low.core\""), std::string::npos);
  loom_check_result_deinitialize(&result);
}

}  // namespace
}  // namespace loom
