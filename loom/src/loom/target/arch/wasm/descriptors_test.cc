// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/wasm/descriptors.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm_roundtrip_test_util.h"
#include "loom/codegen/low/text_asm_test_util.h"

namespace loom {
namespace {

using ::loom::testing::LowFuncAsmRoundTripHarness;
using ::loom::testing::LowTextAsmRoundTripHarness;
using ::loom::testing::LowTextAsmTypeInferenceHarness;

constexpr uint16_t kWasmOpcodeI32Add = 0x6A;
constexpr uint16_t kWasmOpcodeV128Load = 0xFD00;
constexpr uint16_t kWasmOpcodeI32x4Add = 0xFDAE;
constexpr uint16_t kWasmOpcodeI32x4Sub = 0xFDB1;
constexpr uint16_t kWasmOpcodeF32x4Add = 0xFDE4;
constexpr uint16_t kWasmOpcodeF32x4Mul = 0xFDE6;
constexpr uint16_t kWasmOpcodeV128Store = 0xFD0B;

TEST(WasmDescriptorsTest, CoreSimd128DescriptorSetVerifies) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_wasm_core_simd128_descriptor_set();
  ASSERT_NE(descriptor_set, nullptr);
  IREE_ASSERT_OK(loom_low_descriptor_set_verify(descriptor_set));

  iree_string_view_t set_key = iree_string_view_empty();
  IREE_ASSERT_OK(loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset, &set_key));
  EXPECT_TRUE(iree_string_view_equal(set_key, IREE_SV("wasm.core.simd128")));

  EXPECT_GE(descriptor_set->descriptor_count, 12u);
  EXPECT_EQ(descriptor_set->descriptor_ref_count,
            descriptor_set->descriptor_count);
  EXPECT_GE(descriptor_set->reg_class_count, 5u);
  EXPECT_GE(descriptor_set->schedule_class_count, 6u);
}

TEST(WasmDescriptorsTest, CoreSimd128DescriptorLookupUsesStableKeys) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_wasm_core_simd128_descriptor_set();

  uint32_t add_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_ASSERT_OK(loom_low_descriptor_set_lookup_descriptor(
      descriptor_set, IREE_SV("wasm.i32x4.add"), &add_ordinal));
  EXPECT_NE(add_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  const loom_low_descriptor_t* add_descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, add_ordinal);
  ASSERT_NE(add_descriptor, nullptr);
  iree_string_view_t add_key = iree_string_view_empty();
  IREE_ASSERT_OK(loom_low_descriptor_set_string(
      descriptor_set, add_descriptor->key_string_offset, &add_key));
  EXPECT_TRUE(iree_string_view_equal(add_key, IREE_SV("wasm.i32x4.add")));
  EXPECT_EQ(add_descriptor->operand_count, 3u);
  EXPECT_EQ(add_descriptor->result_count, 1u);
  EXPECT_EQ(add_descriptor->encoding_id, kWasmOpcodeI32x4Add);

  uint32_t subtract_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_ASSERT_OK(loom_low_descriptor_set_lookup_descriptor(
      descriptor_set, IREE_SV("wasm.i32x4.sub"), &subtract_ordinal));
  const loom_low_descriptor_t* subtract_descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, subtract_ordinal);
  ASSERT_NE(subtract_descriptor, nullptr);
  EXPECT_EQ(subtract_descriptor->operand_count, 3u);
  EXPECT_EQ(subtract_descriptor->result_count, 1u);
  EXPECT_EQ(subtract_descriptor->encoding_id, kWasmOpcodeI32x4Sub);

  uint32_t f32_add_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_ASSERT_OK(loom_low_descriptor_set_lookup_descriptor(
      descriptor_set, IREE_SV("wasm.f32x4.add"), &f32_add_ordinal));
  const loom_low_descriptor_t* f32_add_descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, f32_add_ordinal);
  ASSERT_NE(f32_add_descriptor, nullptr);
  EXPECT_EQ(f32_add_descriptor->operand_count, 3u);
  EXPECT_EQ(f32_add_descriptor->result_count, 1u);
  EXPECT_EQ(f32_add_descriptor->encoding_id, kWasmOpcodeF32x4Add);

  uint32_t f32_multiply_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_ASSERT_OK(loom_low_descriptor_set_lookup_descriptor(
      descriptor_set, IREE_SV("wasm.f32x4.mul"), &f32_multiply_ordinal));
  const loom_low_descriptor_t* f32_multiply_descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set,
                                            f32_multiply_ordinal);
  ASSERT_NE(f32_multiply_descriptor, nullptr);
  EXPECT_EQ(f32_multiply_descriptor->operand_count, 3u);
  EXPECT_EQ(f32_multiply_descriptor->result_count, 1u);
  EXPECT_EQ(f32_multiply_descriptor->encoding_id, kWasmOpcodeF32x4Mul);

  uint32_t scalar_add_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_ASSERT_OK(loom_low_descriptor_set_lookup_descriptor(
      descriptor_set, IREE_SV("wasm.i32.add"), &scalar_add_ordinal));
  const loom_low_descriptor_t* scalar_add_descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, scalar_add_ordinal);
  ASSERT_NE(scalar_add_descriptor, nullptr);
  EXPECT_EQ(scalar_add_descriptor->encoding_id, kWasmOpcodeI32Add);

  uint32_t load_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_ASSERT_OK(loom_low_descriptor_set_lookup_descriptor(
      descriptor_set, IREE_SV("wasm.v128.load"), &load_ordinal));
  const loom_low_descriptor_t* load_descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, load_ordinal);
  ASSERT_NE(load_descriptor, nullptr);
  EXPECT_EQ(load_descriptor->encoding_id, kWasmOpcodeV128Load);

  uint32_t store_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_ASSERT_OK(loom_low_descriptor_set_lookup_descriptor(
      descriptor_set, IREE_SV("wasm.v128.store"), &store_ordinal));
  const loom_low_descriptor_t* store_descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, store_ordinal);
  ASSERT_NE(store_descriptor, nullptr);
  EXPECT_EQ(store_descriptor->operand_count, 2u);
  EXPECT_EQ(store_descriptor->result_count, 0u);
  EXPECT_EQ(store_descriptor->effect_count, 1u);
  EXPECT_NE(store_descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING,
            0u);
  EXPECT_EQ(store_descriptor->encoding_id, kWasmOpcodeV128Store);
}

TEST(WasmDescriptorsTest, LowAsmInfersV128ResultType) {
  LowTextAsmTypeInferenceHarness harness;
  IREE_ASSERT_OK(harness.Initialize(loom_wasm_core_simd128_descriptor_set));

  loom_text_low_asm_packet_descriptor_t packet = {};
  IREE_ASSERT_OK(harness.LookupPacket(IREE_SV("wasm.core.simd128"),
                                      IREE_SV("i32x4.add"), &packet));

  loom_type_t result_type = loom_type_none();
  iree_string_view_t diagnostic_detail = iree_string_view_empty();
  IREE_ASSERT_OK(harness.InferResultType(
      &packet, /*operands=*/nullptr, /*operand_count=*/0, /*result_index=*/0,
      &result_type, &diagnostic_detail));
  EXPECT_TRUE(iree_string_view_is_empty(diagnostic_detail));
  EXPECT_TRUE(harness.RegisterTypeEquals(result_type, IREE_SV("wasm.v128"), 1));
}

TEST(WasmDescriptorsTest, LowAsmRegionRoundTrips) {
  LowTextAsmRoundTripHarness harness;
  IREE_ASSERT_OK(harness.Initialize(loom_wasm_core_simd128_descriptor_set));

  const char* source =
      "test.low_asm_region asm<wasm.core.simd128> {\n"
      "  %addr = i32.const 16\n"
      "  %lhs = v128.const 1, 2\n"
      "  %rhs = v128.const 3, 4\n"
      "  %sum = i32x4.add %lhs, %rhs\n"
      "  %diff = i32x4.sub %sum, %rhs\n"
      "  %fsum = f32x4.add %lhs, %rhs\n"
      "  %fproduct = f32x4.mul %fsum, %rhs\n"
      "  %loaded = v128.load %addr\n"
      "  v128.store %addr, %loaded\n"
      "  return %fproduct\n"
      "}\n";
  std::string printed;
  IREE_ASSERT_OK(harness.RoundTrip(IREE_SV(source),
                                   IREE_SV("wasm.core.simd128"), &printed));
  EXPECT_EQ(printed, source);
}

TEST(WasmDescriptorsTest, LowFuncAsmRoundTripsMemoryPacketsWithArguments) {
  LowFuncAsmRoundTripHarness harness;
  IREE_ASSERT_OK(harness.Initialize(loom_wasm_core_simd128_descriptor_set));

  const char* source =
      "target.snapshot @wasm32 {codegen_format = wasm, target_triple = "
      "\"wasm32-unknown-unknown\", data_layout = \"\", artifact_format = "
      "wasm_binary, target_cpu = \"generic\", target_features = "
      "\"+simd128\", default_pointer_bitwidth = 32, index_bitwidth = 32, "
      "offset_bitwidth = 32, memory_space_generic = 0, memory_space_global = "
      "0, memory_space_workgroup = 4294967295, memory_space_constant = 0, "
      "memory_space_private = 4294967295, memory_space_host = 4294967295, "
      "memory_space_descriptor = 4294967295}\n"
      "target.export @wasm_export {export_symbol = \"memory\", abi = "
      "wasm_function, linkage = default, hal_binding_alignment = 0, "
      "hal_workgroup_size_x = 0, hal_workgroup_size_y = 0, "
      "hal_workgroup_size_z = 0, hal_flat_workgroup_size_min = 0, "
      "hal_flat_workgroup_size_max = 0, hal_buffer_resource_flags = 0}\n"
      "target.config @wasm_config {contract_set_key = "
      "\"wasm.core.simd128\", contract_feature_bits = 7}\n"
      "target.bundle @wasm_target {snapshot = @wasm32, export_plan = "
      "@wasm_export, config = @wasm_config}\n"
      "low.func.def target(@wasm_target) @memory(%addr : reg<wasm.i32>, "
      "%lhs : reg<wasm.v128>, %rhs : reg<wasm.v128>) -> (reg<wasm.v128>) "
      "asm<wasm.core.simd128> {\n"
      "  %loaded = v128.load %addr\n"
      "  %sum = i32x4.add %lhs, %rhs\n"
      "  %diff = i32x4.sub %sum, %rhs\n"
      "  %fsum = f32x4.add %lhs, %rhs\n"
      "  %fproduct = f32x4.mul %fsum, %rhs\n"
      "  v128.store %addr, %loaded\n"
      "  return %fproduct\n"
      "}\n";
  std::string printed;
  IREE_ASSERT_OK(harness.RoundTripAndVerify(
      IREE_SV(source), IREE_SV("wasm.core.simd128"), &printed));
  EXPECT_NE(printed.find("v128.load %addr"), std::string::npos);
  EXPECT_NE(printed.find("i32x4.sub %sum, %rhs"), std::string::npos);
  EXPECT_NE(printed.find("f32x4.mul %fsum, %rhs"), std::string::npos);
  EXPECT_NE(printed.find("v128.store %addr, %loaded"), std::string::npos);
}

TEST(WasmDescriptorsTest, ManifestNamesSimdAndMemoryPackets) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_wasm_core_simd128_descriptor_set();

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(
      loom_low_descriptor_set_format_manifest_json(descriptor_set, &builder));
  std::string json(iree_string_builder_buffer(&builder),
                   iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);

  EXPECT_NE(json.find("\"key\":\"wasm.core.simd128\""), std::string::npos);
  EXPECT_NE(json.find("\"key\":\"wasm.i32x4.add\""), std::string::npos);
  EXPECT_NE(json.find("\"key\":\"wasm.f32x4.mul\""), std::string::npos);
  EXPECT_NE(json.find("\"key\":\"wasm.v128.load\""), std::string::npos);
  EXPECT_NE(json.find("\"descriptor_refs\""), std::string::npos);
}

}  // namespace
}  // namespace loom
