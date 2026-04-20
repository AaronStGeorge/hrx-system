// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/encoding.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/packetization.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/low_registry.h"
#include "loom/target/presets.h"
#include "loom/testing/context.h"

namespace loom {
namespace {

uint32_t ReadU32LE(const uint8_t* data) {
  return (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
         ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

bool IsSop1SMovB32(uint32_t word) {
  return (word & UINT32_C(0xFF80FF00)) == UINT32_C(0xBE800000);
}

class AmdgpuEncodingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_testing_context_initialize_all(iree_allocator_system(),
                                                       &context_));
    loom_amdgpu_low_descriptor_registry_initialize(&target_registry_);
  }

  void TearDown() override {
    ResetModule();
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void ResetModule() {
    if (module_ != nullptr) {
      loom_module_free(module_);
      module_ = nullptr;
    }
  }

  loom_module_t* ParseSource(const std::string& source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;

    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(
        loom_text_parse(iree_make_string_view(source.data(), source.size()),
                        IREE_SV("amdgpu_encoding_test.loom"), &context_,
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

  loom_value_id_t FindValueIdByName(const char* name) {
    iree_string_view_t expected_name = iree_make_cstring_view(name);
    for (iree_host_size_t i = 0; i < module_->values.count; ++i) {
      const loom_value_t* value =
          loom_module_value(module_, (loom_value_id_t)i);
      if (value->name_id == LOOM_STRING_ID_INVALID ||
          value->name_id >= module_->strings.count) {
        continue;
      }
      iree_string_view_t value_name = module_->strings.entries[value->name_id];
      if (iree_string_view_equal(value_name, expected_name)) {
        return (loom_value_id_t)i;
      }
    }
    return LOOM_VALUE_ID_INVALID;
  }

  void BuildShiftedCopySidecars(iree_arena_allocator_t* arena,
                                loom_low_packetization_t* out_packetization) {
    const char* body =
        "low.func.def target(@gfx_target) @gfx_kernel(%source : "
        "reg<amdgpu.sgpr x3>, %tail : reg<amdgpu.sgpr>, %value : "
        "reg<amdgpu.vgpr>, %vaddr : reg<amdgpu.vgpr>, %soffset : "
        "reg<amdgpu.sgpr>) {\n"
        "  %shifted = low.copy %source : reg<amdgpu.sgpr x3> -> "
        "reg<amdgpu.sgpr x3>\n"
        "  %resource = low.concat(%shifted, %tail) : "
        "(reg<amdgpu.sgpr x3>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x4>\n"
        "  low.op<amdgpu.buffer_store_dword>(%value, %resource, %vaddr, "
        "%soffset) {offset = 0} : (reg<amdgpu.vgpr>, "
        "reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
        "  low.return\n"
        "}\n";
    std::string source =
        "target.preset @gfx_target {key = \"amdgpu-gfx11\", source = "
        "@gfx_kernel}\n";
    source += body;
    ResetModule();
    module_ = ParseSource(source);
    ASSERT_NE(module_, nullptr);

    const loom_target_preset_registry_t preset_registry =
        loom_target_low_descriptor_registry_presets(&target_registry_);
    iree_host_size_t expanded_preset_count = 0;
    IREE_ASSERT_OK(loom_target_expand_presets(module_, &preset_registry,
                                              &expanded_preset_count));
    EXPECT_EQ(expanded_preset_count, 1u);

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

    loom_value_id_t source_value = FindValueIdByName("source");
    loom_value_id_t shifted_value = FindValueIdByName("shifted");
    ASSERT_NE(source_value, LOOM_VALUE_ID_INVALID);
    ASSERT_NE(shifted_value, LOOM_VALUE_ID_INVALID);
    const loom_low_allocation_fixed_value_t fixed_values[] = {
        {
            .value_id = source_value,
            .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
            .location_base = 0,
            .location_count = 3,
        },
        {
            .value_id = shifted_value,
            .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
            .location_base = 1,
            .location_count = 3,
        },
    };
    loom_op_t* low_function = FindFirstLowFunction(module_);
    ASSERT_NE(low_function, nullptr);
    loom_low_packetization_options_t packetization_options = {
        .descriptor_registry = &target_registry_.registry,
        .allocation_fixed_values = fixed_values,
        .allocation_fixed_value_count = IREE_ARRAYSIZE(fixed_values),
    };
    IREE_ASSERT_OK(loom_low_packetize_function(module_, low_function,
                                               &packetization_options, arena,
                                               out_packetization));
  }

  void BuildSidecarsForPreset(const char* preset_key, const char* body,
                              iree_arena_allocator_t* arena,
                              loom_low_packetization_t* out_packetization) {
    std::string source = "target.preset @gfx_target {key = \"";
    source += preset_key;
    source += "\", source = @gfx_kernel}\n";
    source += body;
    ResetModule();
    module_ = ParseSource(source);
    ASSERT_NE(module_, nullptr);

    const loom_target_preset_registry_t preset_registry =
        loom_target_low_descriptor_registry_presets(&target_registry_);
    iree_host_size_t expanded_preset_count = 0;
    IREE_ASSERT_OK(loom_target_expand_presets(module_, &preset_registry,
                                              &expanded_preset_count));
    EXPECT_EQ(expanded_preset_count, 1u);

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

    loom_op_t* low_function = FindFirstLowFunction(module_);
    ASSERT_NE(low_function, nullptr);
    loom_low_packetization_options_t packetization_options = {
        .descriptor_registry = &target_registry_.registry,
    };
    IREE_ASSERT_OK(loom_low_packetize_function(module_, low_function,
                                               &packetization_options, arena,
                                               out_packetization));
  }

  void BuildGfx11Sidecars(const char* body, iree_arena_allocator_t* arena,
                          loom_low_packetization_t* out_packetization) {
    BuildSidecarsForPreset("amdgpu-gfx11", body, arena, out_packetization);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_target_low_descriptor_registry_t target_registry_ = {};
};

TEST_F(AmdgpuEncodingTest, EncodesInlineScalarConstantAndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_packetization_t packetization = {};
  BuildGfx11Sidecars(
      "low.func.def target(@gfx_target) @gfx_kernel(%resource : "
      "reg<amdgpu.sgpr x4>) {\n"
      "  %c0 = low.op<amdgpu.s_mov_b32>() {imm32 = 7} : () -> "
      "reg<amdgpu.sgpr>\n"
      "  %loaded = low.op<amdgpu.s_buffer_load_dword>(%resource, %c0) "
      "{offset = 0} : (reg<amdgpu.sgpr x4>, reg<amdgpu.sgpr>) -> "
      "reg<amdgpu.sgpr>\n"
      "  low.return\n"
      "}\n",
      &arena, &packetization);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &packetization.schedule, &packetization.allocation, &text, &arena));

  ASSERT_GE(text.data_length, 8u);
  EXPECT_TRUE(IsSop1SMovB32(ReadU32LE(text.data + 0)));
  EXPECT_EQ(ReadU32LE(text.data + 0) & UINT32_C(0xFF), UINT32_C(0x87));
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesLiteralScalarConstantAndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_packetization_t packetization = {};
  BuildGfx11Sidecars(
      "low.func.def target(@gfx_target) @gfx_kernel(%resource : "
      "reg<amdgpu.sgpr x4>) {\n"
      "  %c0 = low.op<amdgpu.s_mov_b32>() {imm32 = 305419896} : () -> "
      "reg<amdgpu.sgpr>\n"
      "  %loaded = low.op<amdgpu.s_buffer_load_dword>(%resource, %c0) "
      "{offset = 0} : (reg<amdgpu.sgpr x4>, reg<amdgpu.sgpr>) -> "
      "reg<amdgpu.sgpr>\n"
      "  low.return\n"
      "}\n",
      &arena, &packetization);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &packetization.schedule, &packetization.allocation, &text, &arena));

  ASSERT_GE(text.data_length, 12u);
  EXPECT_TRUE(IsSop1SMovB32(ReadU32LE(text.data + 0)));
  EXPECT_EQ(ReadU32LE(text.data + 0) & UINT32_C(0xFF), UINT32_C(0xFF));
  EXPECT_EQ(ReadU32LE(text.data + 4), UINT32_C(0x12345678));
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesLiteralVectorConstantAndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_packetization_t packetization = {};
  BuildGfx11Sidecars(
      "low.func.def target(@gfx_target) @gfx_kernel(%resource : "
      "reg<amdgpu.sgpr x4>, %vaddr : reg<amdgpu.vgpr>, %soffset : "
      "reg<amdgpu.sgpr>) {\n"
      "  %v0 = low.const<amdgpu.v_mov_b32> {imm32 = 42} : "
      "reg<amdgpu.vgpr>\n"
      "  low.op<amdgpu.buffer_store_dword>(%v0, %resource, %vaddr, "
      "%soffset) {offset = 0} : (reg<amdgpu.vgpr>, "
      "reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
      "  low.return\n"
      "}\n",
      &arena, &packetization);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &packetization.schedule, &packetization.allocation, &text, &arena));

  ASSERT_GE(text.data_length, 12u);
  EXPECT_EQ(ReadU32LE(text.data + 0) & UINT32_C(0xFE01FFFF),
            UINT32_C(0x7E0002FF));
  EXPECT_EQ(ReadU32LE(text.data + 4), UINT32_C(42));
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesLiveInAsNonEmittingPacket) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_packetization_t packetization = {};
  BuildGfx11Sidecars(
      "low.func.def target(@gfx_target) @gfx_kernel() {\n"
      "  %kernarg = low.live_in<amdgpu.kernarg_segment_ptr> : "
      "reg<amdgpu.sgpr x2>\n"
      "  low.return\n"
      "}\n",
      &arena, &packetization);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &packetization.schedule, &packetization.allocation, &text, &arena));

  ASSERT_EQ(text.data_length, 4u);
  EXPECT_EQ(ReadU32LE(text.data), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, SequencesOverlappingCopyBeforeClobber) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_packetization_t packetization = {};
  BuildShiftedCopySidecars(&arena, &packetization);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &packetization.schedule, &packetization.allocation, &text, &arena));

  ASSERT_GE(text.data_length, 16u);
  EXPECT_EQ(ReadU32LE(text.data + 0), UINT32_C(0xBE830002));
  EXPECT_EQ(ReadU32LE(text.data + 4), UINT32_C(0xBE820001));
  EXPECT_EQ(ReadU32LE(text.data + 8), UINT32_C(0xBE810000));
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesConcatRegisterCopies) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_packetization_t packetization = {};
  BuildGfx11Sidecars(
      "low.func.def target(@gfx_target) @gfx_kernel(%r0 : reg<amdgpu.sgpr>, "
      "%r1 : reg<amdgpu.sgpr>, %r2 : reg<amdgpu.sgpr>, %r3 : "
      "reg<amdgpu.sgpr>, %value : reg<amdgpu.vgpr>, %vaddr : "
      "reg<amdgpu.vgpr>) {\n"
      "  %resource = low.concat(%r0, %r1, %r2, %r3) : (reg<amdgpu.sgpr>, "
      "reg<amdgpu.sgpr>, reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> "
      "reg<amdgpu.sgpr x4>\n"
      "  %sum0 = low.op<amdgpu.s_add_u32>(%r0, %r1) : "
      "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
      "  %sum1 = low.op<amdgpu.s_add_u32>(%r2, %r3) : "
      "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
      "  %soffset = low.op<amdgpu.s_add_u32>(%sum0, %sum1) : "
      "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
      "  low.op<amdgpu.buffer_store_dword>(%value, %resource, %vaddr, "
      "%soffset) {offset = 0} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x4>, "
      "reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
      "  low.return\n"
      "}\n",
      &arena, &packetization);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &packetization.schedule, &packetization.allocation, &text, &arena));

  ASSERT_GE(text.data_length, 8u);
  ASSERT_EQ(text.data_length % 4, 0u);
  bool saw_concat_copy = false;
  for (iree_host_size_t i = 0; i < text.data_length; i += 4) {
    saw_concat_copy |= IsSop1SMovB32(ReadU32LE(text.data + i));
  }
  EXPECT_TRUE(saw_concat_copy);
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesInitialGfx11Allowlist) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_packetization_t packetization = {};
  BuildGfx11Sidecars(
      "low.func.def target(@gfx_target) @gfx_kernel(%s0 : "
      "reg<amdgpu.sgpr>, %s1 : reg<amdgpu.sgpr>, %v0 : "
      "reg<amdgpu.vgpr>, %v1 : reg<amdgpu.vgpr>, %resource : "
      "reg<amdgpu.sgpr x4>, %vaddr : reg<amdgpu.vgpr>) {\n"
      "  %s_sum = low.op<amdgpu.s_add_u32>(%s0, %s1) : "
      "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
      "  %v_sum = low.op<amdgpu.v_add_u32>(%v0, %v1) : "
      "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
      "  %v_product = low.op<amdgpu.v_mul_lo_u32>(%v_sum, %v1) : "
      "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
      "  low.op<amdgpu.buffer_store_dword>(%v_product, %resource, %vaddr, "
      "%s_sum) {offset = 0} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x4>, "
      "reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
      "  low.op<amdgpu.s_waitcnt>() {vmcnt = 0, lgkmcnt = 0} : ()\n"
      "  low.op<amdgpu.s_wait_idle>() : ()\n"
      "  low.return\n"
      "}\n",
      &arena, &packetization);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &packetization.schedule, &packetization.allocation, &text, &arena));

  ASSERT_GE(text.data_length, 36u);
  EXPECT_EQ(ReadU32LE(text.data + 0), UINT32_C(0x80000100));
  EXPECT_EQ(ReadU32LE(text.data + 4), UINT32_C(0x4A000300));
  EXPECT_EQ(ReadU32LE(text.data + 8), UINT32_C(0xD72C0000));
  EXPECT_EQ(ReadU32LE(text.data + 12), UINT32_C(0x00020300));
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 12), UINT32_C(0xBF890007));
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 8), UINT32_C(0xBF8A0000));
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesGfx11MubufStoreAndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_packetization_t packetization = {};
  BuildGfx11Sidecars(
      "low.func.def target(@gfx_target) @gfx_kernel(%value : "
      "reg<amdgpu.vgpr>, %resource : reg<amdgpu.sgpr x4>, %vaddr : "
      "reg<amdgpu.vgpr>, %soffset : reg<amdgpu.sgpr>) {\n"
      "  low.op<amdgpu.buffer_store_dword>(%value, %resource, %vaddr, "
      "%soffset) {offset = 8} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x4>, "
      "reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
      "  low.return\n"
      "}\n",
      &arena, &packetization);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &packetization.schedule, &packetization.allocation, &text, &arena));

  ASSERT_EQ(text.data_length, 12u);
  EXPECT_EQ(ReadU32LE(text.data + 0), UINT32_C(0xE0680008));
  EXPECT_EQ(ReadU32LE(text.data + 4), UINT32_C(0x04400001));
  EXPECT_EQ(ReadU32LE(text.data + 8), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesGfx11MubufLoadAndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_packetization_t packetization = {};
  BuildGfx11Sidecars(
      "low.func.def target(@gfx_target) @gfx_kernel(%resource : "
      "reg<amdgpu.sgpr x4>, %vaddr : reg<amdgpu.vgpr>, %soffset : "
      "reg<amdgpu.sgpr>) {\n"
      "  %loaded = low.op<amdgpu.buffer_load_dword>(%resource, %vaddr, "
      "%soffset) {offset = 12} : (reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, "
      "reg<amdgpu.sgpr>) -> reg<amdgpu.vgpr>\n"
      "  low.return\n"
      "}\n",
      &arena, &packetization);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &packetization.schedule, &packetization.allocation, &text, &arena));

  ASSERT_EQ(text.data_length, 12u);
  EXPECT_EQ(ReadU32LE(text.data + 0), UINT32_C(0xE050000C));
  EXPECT_EQ(ReadU32LE(text.data + 4), UINT32_C(0x04400000));
  EXPECT_EQ(ReadU32LE(text.data + 8), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesGfx11MubufB128LoadStoreAndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_packetization_t packetization = {};
  BuildGfx11Sidecars(
      "low.func.def target(@gfx_target) @gfx_kernel(%resource : "
      "reg<amdgpu.sgpr x4>, %vaddr : reg<amdgpu.vgpr>, %soffset : "
      "reg<amdgpu.sgpr>) {\n"
      "  %loaded = low.op<amdgpu.buffer_load_b128>(%resource, %vaddr, "
      "%soffset) {offset = 16} : (reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, "
      "reg<amdgpu.sgpr>) -> reg<amdgpu.vgpr x4>\n"
      "  low.op<amdgpu.buffer_store_b128>(%loaded, %resource, %vaddr, "
      "%soffset) {offset = 32} : (reg<amdgpu.vgpr x4>, "
      "reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
      "  low.return\n"
      "}\n",
      &arena, &packetization);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &packetization.schedule, &packetization.allocation, &text, &arena));

  ASSERT_EQ(text.data_length, 20u);
  EXPECT_EQ(ReadU32LE(text.data + 0), UINT32_C(0xE05C0010));
  EXPECT_EQ(ReadU32LE(text.data + 4), UINT32_C(0x04400400));
  EXPECT_EQ(ReadU32LE(text.data + 8), UINT32_C(0xE0740020));
  EXPECT_EQ(ReadU32LE(text.data + 12), UINT32_C(0x04400400));
  EXPECT_EQ(ReadU32LE(text.data + 16), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesReturnForCurrentAmdgpuFamilies) {
  struct Case {
    // Target preset used to select the low descriptor set.
    const char* preset_key;
    // Expected little-endian SOPP instruction word for `s_endpgm 0`.
    uint32_t expected_return_word;
  };
  const Case cases[] = {
      {"amdgpu-gfx950", UINT32_C(0xBF810000)},
      {"amdgpu-gfx11", UINT32_C(0xBFB00000)},
      {"amdgpu-gfx12", UINT32_C(0xBFB00000)},
      {"amdgpu-gfx1250", UINT32_C(0xBFB00000)},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.preset_key);
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool_, &arena);
    loom_low_packetization_t packetization = {};
    BuildSidecarsForPreset(test_case.preset_key,
                           "low.func.def target(@gfx_target) @gfx_kernel() {\n"
                           "  low.return\n"
                           "}\n",
                           &arena, &packetization);

    iree_const_byte_span_t text = iree_const_byte_span_empty();
    IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
        &packetization.schedule, &packetization.allocation, &text, &arena));

    ASSERT_EQ(text.data_length, 4u);
    EXPECT_EQ(ReadU32LE(text.data), test_case.expected_return_word);
    iree_arena_deinitialize(&arena);
  }
}

TEST_F(AmdgpuEncodingTest, RejectsDescriptorPacketsOutsideImplementedProfile) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_packetization_t packetization = {};
  BuildSidecarsForPreset("amdgpu-gfx12",
                         "low.func.def target(@gfx_target) @gfx_kernel() {\n"
                         "  low.op<amdgpu.s_wait_idle>() : ()\n"
                         "  low.return\n"
                         "}\n",
                         &arena, &packetization);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  iree_status_t status = loom_amdgpu_encode_instruction_stream(
      &packetization.schedule, &packetization.allocation, &text, &arena);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);
  EXPECT_TRUE(iree_const_byte_span_is_empty(text));
  iree_arena_deinitialize(&arena);
}

}  // namespace
}  // namespace loom
