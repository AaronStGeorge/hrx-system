// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amdgpu/low_registry.h"
#include "loom/target/arch/amdgpu/lower.h"
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

iree_status_t InitializeAmdgpuLowLowerPolicyRegistry(
    void* user_data, loom_low_lower_policy_registry_t* out_registry) {
  (void)user_data;
  loom_amdgpu_low_lower_policy_registry_initialize(out_registry);
  return iree_ok_status();
}

const loom_target_low_legality_provider_t* const kAmdgpuLowLegalityProviders[] =
    {
        &loom_amdgpu_low_legality_provider_storage,
};

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
    .initialize_low_lower_policy_registry =
        {
            .fn = InitializeAmdgpuLowLowerPolicyRegistry,
            .user_data = nullptr,
        },
    .low_legality_provider_list =
        {
            .count = IREE_ARRAYSIZE(kAmdgpuLowLegalityProviders),
            .values = kAmdgpuLowLegalityProviders,
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

static std::string AmdgpuB128CopySource(iree_string_view_t target_symbol,
                                        iree_string_view_t target_key) {
  std::string source = "// RUN: emit source-low @";
  source.append(target_symbol.data, target_symbol.size);
  source += " output=low\n";
  source += "target.preset @";
  source.append(target_symbol.data, target_symbol.size);
  source += " {key = \"";
  source.append(target_key.data, target_key.size);
  source +=
      "\", source = @copy_b128}\n"
      "\n"
      "func.def @copy_b128(%input: buffer, %output: buffer) {\n"
      "  %tid = kernel.workitem.id<x> : index\n"
      "  %zero = index.constant 0 : offset\n"
      "  %input_view = buffer.view %input[%zero] : buffer -> "
      "view<64x4xi32, #dense>\n"
      "  %output_view = buffer.view %output[%zero] : buffer -> "
      "view<64x4xi32, #dense>\n"
      "  %loaded = vector.load %input_view[%tid, 0] : "
      "view<64x4xi32, #dense> -> vector<4xi32>\n"
      "  vector.store %loaded, %output_view[%tid, 0] : "
      "vector<4xi32>, view<64x4xi32, #dense>\n"
      "  func.return\n"
      "}\n"
      "// ----\n";
  return source;
}

static std::string AmdgpuGfx11SourceLowCase(const char* output,
                                            const char* body) {
  std::string source = "// RUN: emit source-low @gfx11_target output=";
  source += output;
  source +=
      "\n"
      "target.preset @gfx11_target {key = \"amdgpu-gfx11\", source = @case}\n"
      "\n"
      "func.def @case(%input: buffer, %output: buffer) {\n";
  source += body;
  source +=
      "  func.return\n"
      "}\n"
      "// ----\n";
  return source;
}

static std::string AmdgpuGfx11SourceLowCase(const char* body) {
  return AmdgpuGfx11SourceLowCase("none", body);
}

static void ExpectAmdgpuSourceLowRejects(
    ::loom::testing::LoomCheckHarness* harness, const std::string& source,
    const char* expected_detail) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      harness->ExecuteFirst(iree_make_string_view(source.data(), source.size()),
                            IREE_SV("amdgpu_source_low.loom-test"), &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL)
      << harness->ActualOutputString(result) << "\n"
      << harness->DetailString(result) << "\n"
      << harness->DiagnosticJsonString(result);
  EXPECT_GT(result.diagnostic_count, 0u);
  const std::string diagnostic_json = harness->DiagnosticJsonString(result);
  EXPECT_NE(diagnostic_json.find("\"error_id\":\"ERR_BACKEND_001\""),
            std::string::npos)
      << diagnostic_json;
  EXPECT_NE(diagnostic_json.find(expected_detail), std::string::npos)
      << diagnostic_json;
  loom_check_result_deinitialize(&result);
}

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
  EXPECT_FALSE(actual_output.empty());
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

TEST_F(AmdgpuLoomCheckTest,
       SourceLowerSelectsTargetLegalB128BufferDescriptorNames) {
  struct B128Case {
    // Symbol name for the target preset under test.
    iree_string_view_t target_symbol;
    // Target preset key that selects the descriptor set.
    iree_string_view_t target_key;
    // Load packet spelling expected after source lowering.
    const char* expected_load;
    // Store packet spelling expected after source lowering.
    const char* expected_store;
    // Load packet spelling that must not be selected for this target.
    const char* forbidden_load;
    // Store packet spelling that must not be selected for this target.
    const char* forbidden_store;
  };
  const B128Case cases[] = {
      {
          .target_symbol = IREE_SV("gfx11_copy_b128_target"),
          .target_key = IREE_SV("amdgpu-gfx11"),
          .expected_load = "low.op<amdgpu.buffer_load_b128>",
          .expected_store = "low.op<amdgpu.buffer_store_b128>",
          .forbidden_load = "low.op<amdgpu.buffer_load_dwordx4>",
          .forbidden_store = "low.op<amdgpu.buffer_store_dwordx4>",
      },
      {
          .target_symbol = IREE_SV("gfx12_copy_b128_target"),
          .target_key = IREE_SV("amdgpu-gfx12"),
          .expected_load = "low.op<amdgpu.buffer_load_b128>",
          .expected_store = "low.op<amdgpu.buffer_store_b128>",
          .forbidden_load = "low.op<amdgpu.buffer_load_dwordx4>",
          .forbidden_store = "low.op<amdgpu.buffer_store_dwordx4>",
      },
      {
          .target_symbol = IREE_SV("gfx1250_copy_b128_target"),
          .target_key = IREE_SV("amdgpu-gfx1250"),
          .expected_load = "low.op<amdgpu.buffer_load_b128>",
          .expected_store = "low.op<amdgpu.buffer_store_b128>",
          .forbidden_load = "low.op<amdgpu.buffer_load_dwordx4>",
          .forbidden_store = "low.op<amdgpu.buffer_store_dwordx4>",
      },
      {
          .target_symbol = IREE_SV("gfx950_copy_b128_target"),
          .target_key = IREE_SV("amdgpu-gfx950"),
          .expected_load = "low.op<amdgpu.buffer_load_dwordx4>",
          .expected_store = "low.op<amdgpu.buffer_store_dwordx4>",
          .forbidden_load = "low.op<amdgpu.buffer_load_b128>",
          .forbidden_store = "low.op<amdgpu.buffer_store_b128>",
      },
  };
  for (const B128Case& test_case : cases) {
    loom_check_result_t result;
    std::string source =
        AmdgpuB128CopySource(test_case.target_symbol, test_case.target_key);
    IREE_ASSERT_OK(harness_.ExecuteFirst(
        iree_make_string_view(source.data(), source.size()),
        IREE_SV("amdgpu_source_low.loom-test"), &result));
    EXPECT_TRUE(result.has_actual_output);
    EXPECT_EQ(result.diagnostic_count, 0u);
    const std::string output = harness_.ActualOutputString(result);
    const std::string target_key(test_case.target_key.data,
                                 test_case.target_key.size);
    EXPECT_NE(output.find(test_case.expected_load), std::string::npos)
        << target_key;
    EXPECT_NE(output.find(test_case.expected_store), std::string::npos)
        << target_key;
    EXPECT_EQ(output.find(test_case.forbidden_load), std::string::npos)
        << target_key;
    EXPECT_EQ(output.find(test_case.forbidden_store), std::string::npos)
        << target_key;
    loom_check_result_deinitialize(&result);
  }
}

TEST_F(AmdgpuLoomCheckTest, SourceLowerRejectsDynamicBufferViewByteOffset) {
  const std::string source = AmdgpuGfx11SourceLowCase(
      "  %lhs = index.constant 4 : offset\n"
      "  %rhs = index.constant 8 : offset\n"
      "  %base = index.add %lhs, %rhs : offset\n"
      "  %view = buffer.view %input[%base] : buffer -> view<4xi32, #dense>\n");
  ExpectAmdgpuSourceLowRejects(
      &harness_, source,
      "AMDGPU HAL buffer views currently require exact non-negative static "
      "byte offsets");
}

TEST_F(AmdgpuLoomCheckTest, SourceLowerRejectsNegativeBufferViewByteOffset) {
  const std::string source = AmdgpuGfx11SourceLowCase(
      "  %base = index.constant -4 : offset\n"
      "  %view = buffer.view %input[%base] : buffer -> view<4xi32, #dense>\n");
  ExpectAmdgpuSourceLowRejects(
      &harness_, source,
      "AMDGPU HAL buffer views currently require exact non-negative static "
      "byte offsets");
}

TEST_F(AmdgpuLoomCheckTest, SourceLowerFoldsStaticBufferViewByteOffset) {
  const std::string source = AmdgpuGfx11SourceLowCase(
      "low",
      "  %base = index.constant 16 : offset\n"
      "  %zero = index.constant 0 : offset\n"
      "  %input_view = buffer.view %input[%base] : buffer -> "
      "view<8xi32, #dense>\n"
      "  %output_view = buffer.view %output[%zero] : buffer -> "
      "view<8xi32, #dense>\n"
      "  %loaded = vector.load %input_view[1] : view<8xi32, #dense> -> "
      "vector<1xi32>\n"
      "  vector.store %loaded, %output_view[0] : vector<1xi32>, "
      "view<8xi32, #dense>\n");
  loom_check_result_t result;
  IREE_ASSERT_OK(
      harness_.ExecuteFirst(iree_make_string_view(source.data(), source.size()),
                            IREE_SV("amdgpu_source_low.loom-test"), &result));
  EXPECT_TRUE(result.has_actual_output);
  EXPECT_EQ(result.diagnostic_count, 0u);
  const std::string actual_output = harness_.ActualOutputString(result);
  EXPECT_NE(actual_output.find(
                "low.resource<hal_buffer_resource> {index = 0, semantic_type "
                "= hal.buffer, valid_byte_count = 48}"),
            std::string::npos)
      << actual_output;
  EXPECT_NE(actual_output.find(
                "low.resource<hal_buffer_resource> {index = 1, semantic_type "
                "= hal.buffer, valid_byte_count = 32}"),
            std::string::npos)
      << actual_output;
  EXPECT_NE(actual_output.find("low.op<amdgpu.buffer_load_dword>"),
            std::string::npos)
      << actual_output;
  EXPECT_NE(actual_output.find("{offset = 20}"), std::string::npos)
      << actual_output;
  loom_check_result_deinitialize(&result);
}

TEST_F(AmdgpuLoomCheckTest, SourceLowerOmitsResourceExtentForUnknownBufferUse) {
  const std::string source = AmdgpuGfx11SourceLowCase(
      "low",
      "  %base = index.constant 16 : offset\n"
      "  %zero = index.constant 0 : offset\n"
      "  %unused = buffer.assume.memory_space %input {memory_space = global} : "
      "buffer\n"
      "  %input_view = buffer.view %input[%base] : buffer -> "
      "view<8xi32, #dense>\n"
      "  %output_view = buffer.view %output[%zero] : buffer -> "
      "view<8xi32, #dense>\n"
      "  %loaded = vector.load %input_view[1] : view<8xi32, #dense> -> "
      "vector<1xi32>\n"
      "  vector.store %loaded, %output_view[0] : vector<1xi32>, "
      "view<8xi32, #dense>\n");
  loom_check_result_t result;
  IREE_ASSERT_OK(
      harness_.ExecuteFirst(iree_make_string_view(source.data(), source.size()),
                            IREE_SV("amdgpu_source_low.loom-test"), &result));
  EXPECT_TRUE(result.has_actual_output);
  EXPECT_EQ(result.diagnostic_count, 0u);
  const std::string actual_output = harness_.ActualOutputString(result);
  EXPECT_NE(actual_output.find(
                "low.resource<hal_buffer_resource> {index = 0, semantic_type "
                "= hal.buffer}"),
            std::string::npos)
      << actual_output;
  EXPECT_EQ(actual_output.find(
                "low.resource<hal_buffer_resource> {index = 0, semantic_type "
                "= hal.buffer, valid_byte_count"),
            std::string::npos)
      << actual_output;
  EXPECT_NE(actual_output.find(
                "low.resource<hal_buffer_resource> {index = 1, semantic_type "
                "= hal.buffer, valid_byte_count = 32}"),
            std::string::npos)
      << actual_output;
  loom_check_result_deinitialize(&result);
}

TEST_F(AmdgpuLoomCheckTest, SourceLowerRejectsMisalignedB128StaticOffset) {
  const std::string source = AmdgpuGfx11SourceLowCase(
      "  %zero = index.constant 0 : offset\n"
      "  %view = buffer.view %input[%zero] : buffer -> view<8xi32, #dense>\n"
      "  %loaded = vector.load %view[1] : view<8xi32, #dense> -> "
      "vector<4xi32>\n"
      "  vector.store %loaded, %view[0] : vector<4xi32>, "
      "view<8xi32, #dense>\n");
  ExpectAmdgpuSourceLowRejects(
      &harness_, source,
      "128-bit AMDGPU buffer memory accesses currently require 16-byte "
      "aligned static byte offsets");
}

TEST_F(AmdgpuLoomCheckTest, SourceLowerSplitsStaticOffsetIntoSoffset) {
  const std::string source = AmdgpuGfx11SourceLowCase(
      "low",
      "  %zero = index.constant 0 : offset\n"
      "  %view = buffer.view %input[%zero] : buffer -> view<2048xi32, #dense>\n"
      "  %loaded = vector.load %view[1200] : view<2048xi32, #dense> -> "
      "vector<1xi32>\n"
      "  vector.store %loaded, %view[0] : vector<1xi32>, "
      "view<2048xi32, #dense>\n");
  loom_check_result_t result;
  IREE_ASSERT_OK(
      harness_.ExecuteFirst(iree_make_string_view(source.data(), source.size()),
                            IREE_SV("amdgpu_source_low.loom-test"), &result));
  EXPECT_TRUE(result.has_actual_output);
  EXPECT_EQ(result.diagnostic_count, 0u);
  const std::string actual_output = harness_.ActualOutputString(result);
  EXPECT_NE(actual_output.find("low.const<amdgpu.s_mov_b32> {imm32 = 705}"),
            std::string::npos)
      << actual_output;
  EXPECT_NE(actual_output.find("low.op<amdgpu.buffer_load_dword>"),
            std::string::npos)
      << actual_output;
  EXPECT_NE(actual_output.find("{offset = 4095}"), std::string::npos)
      << actual_output;
  loom_check_result_deinitialize(&result);
}

TEST_F(AmdgpuLoomCheckTest, SourceLowerRejectsOutOfRangeStaticBufferOffset) {
  const std::string source = AmdgpuGfx11SourceLowCase(
      "  %zero = index.constant 0 : offset\n"
      "  %view = buffer.view %input[%zero] : buffer -> "
      "view<1073742849xi32, #dense>\n"
      "  %loaded = vector.load %view[1073742848] : "
      "view<1073742849xi32, #dense> -> vector<1xi32>\n"
      "  vector.store %loaded, %view[0] : vector<1xi32>, "
      "view<1073742849xi32, #dense>\n");
  ExpectAmdgpuSourceLowRejects(
      &harness_, source,
      "AMDGPU buffer memory static byte offset is outside the selected "
      "descriptor's immediate plus scalar SOFFSET range");
}

TEST_F(AmdgpuLoomCheckTest, SourceLowerRejectsNonWorkitemDynamicIndex) {
  const std::string source = AmdgpuGfx11SourceLowCase(
      "  %zero = index.constant 0 : offset\n"
      "  %index = index.constant 0 : index\n"
      "  %view = buffer.view %input[%zero] : buffer -> "
      "view<64x4xi32, #dense>\n"
      "  %loaded = vector.load %view[%index, 0] : "
      "view<64x4xi32, #dense> -> vector<4xi32>\n"
      "  vector.store %loaded, %view[0, 0] : vector<4xi32>, "
      "view<64x4xi32, #dense>\n");
  ExpectAmdgpuSourceLowRejects(
      &harness_, source,
      "AMDGPU buffer memory lowering currently requires dynamic indices to "
      "come from kernel.workitem.id<x>");
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
