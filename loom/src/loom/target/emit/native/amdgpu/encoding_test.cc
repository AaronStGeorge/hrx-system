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
#include "loom/target/arch/amdgpu/encoding.h"
#include "loom/target/arch/amdgpu/low_registry.h"
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
        "target.profile @gfx_target preset(\"amdgpu-gfx11\")\n";
    source += body;
    ResetModule();
    module_ = ParseSource(source);
    ASSERT_NE(module_, nullptr);

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
    std::string source = "target.profile @gfx_target preset(\"";
    source += preset_key;
    source += "\")\n";
    source += body;
    ResetModule();
    module_ = ParseSource(source);
    ASSERT_NE(module_, nullptr);

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

TEST_F(AmdgpuEncodingTest, PacksGeneratedVectorRegisterMove) {
  const loom_amdgpu_encoding_table_t* table =
      loom_amdgpu_encoding_table_for_descriptor_set_id(
          loom_low_descriptor_stable_id_from_key(IREE_SV("amdgpu.gfx11.core")));
  ASSERT_NE(table, nullptr);

  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(
      loom_amdgpu_encoding_pack_v_mov_b32_vgpr(table, 7, 3, &packet));
  ASSERT_EQ(packet.word_count, 1u);
  const uint32_t word = packet.words[0];
  EXPECT_EQ(word & UINT32_C(0x1FF), UINT32_C(0x103));
  EXPECT_EQ((word >> 9) & UINT32_C(0xFF), table->v_mov_b32_opcode);
  EXPECT_EQ((word >> 17) & UINT32_C(0xFF), UINT32_C(7));
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

TEST_F(AmdgpuEncodingTest, EncodesGenericSoppCacheControl) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_packetization_t packetization = {};
  BuildGfx11Sidecars(
      "low.func.def target(@gfx_target) @gfx_kernel() {\n"
      "  low.op<amdgpu.s_icache_inv>() : ()\n"
      "  low.return\n"
      "}\n",
      &arena, &packetization);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &packetization.schedule, &packetization.allocation, &text, &arena));

  ASSERT_GE(text.data_length, 8u);
  ASSERT_EQ(text.data_length % 4, 0u);
  bool saw_icache_invalidate = false;
  for (iree_host_size_t i = 0; i + 4 <= text.data_length; i += 4) {
    saw_icache_invalidate |= ReadU32LE(text.data + i) == UINT32_C(0xBFBC0000);
  }
  EXPECT_TRUE(saw_icache_invalidate);
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesGfx12SmemPrefetchPackets) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_packetization_t packetization = {};
  BuildSidecarsForPreset(
      "amdgpu-gfx12",
      "low.func.def target(@gfx_target) @gfx_kernel(%base : "
      "reg<amdgpu.sgpr x2>, %resource : reg<amdgpu.sgpr x4>, "
      "%soffset : reg<amdgpu.sgpr>) {\n"
      "  low.op<amdgpu.s_prefetch_data>(%base, %soffset) {offset = 64, "
      "count = 2} : (reg<amdgpu.sgpr x2>, reg<amdgpu.sgpr>)\n"
      "  low.op<amdgpu.s_buffer_prefetch_data>(%resource, %soffset) "
      "{offset = 128, count = 1} : (reg<amdgpu.sgpr x4>, "
      "reg<amdgpu.sgpr>)\n"
      "  low.op<amdgpu.s_prefetch_inst_pc_rel>(%soffset) {offset = 0, "
      "count = 1} : (reg<amdgpu.sgpr>)\n"
      "  low.return\n"
      "}\n",
      &arena, &packetization);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &packetization.schedule, &packetization.allocation, &text, &arena));

  ASSERT_GE(text.data_length, 16u);
  ASSERT_EQ(text.data_length % 4, 0u);
  EXPECT_NE(ReadU32LE(text.data), UINT32_C(0));
  EXPECT_NE(ReadU32LE(text.data + 4), UINT32_C(0));
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBFB00000));
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

TEST_F(AmdgpuEncodingTest,
       EncodesScalarAndVectorBasicsForCurrentAmdgpuFamilies) {
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
    BuildSidecarsForPreset(
        test_case.preset_key,
        "low.func.def target(@gfx_target) @gfx_kernel(%s0 : "
        "reg<amdgpu.sgpr>, %v0 : "
        "reg<amdgpu.vgpr>, %v1 : reg<amdgpu.vgpr>, %resource : "
        "reg<amdgpu.sgpr x4>, %soffset : reg<amdgpu.sgpr>, %vaddr : "
        "reg<amdgpu.vgpr>) {\n"
        "  %loaded = low.op<amdgpu.s_buffer_load_dword>(%resource, "
        "%soffset) {offset = 0} : (reg<amdgpu.sgpr x4>, "
        "reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
        "  %s_sum = low.op<amdgpu.s_add_u32>(%s0, %loaded) : "
        "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
        "  %v_sum = low.op<amdgpu.v_add_u32>(%v0, %v1) : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        "  %converted = low.op<amdgpu.v_cvt_f32_i32>(%s_sum) : "
        "(reg<amdgpu.sgpr>) -> reg<amdgpu.vgpr>\n"
        "  %v_min = low.op<amdgpu.v_min_f32>(%v_sum, %converted) : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        "  %v_max = low.op<amdgpu.v_max_f32>(%v_min, %converted) : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        "  %v_mix = low.op<amdgpu.v_add_u32>(%v_max, %converted) : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        "  %dot_s = low.op<amdgpu.v_dot4_i32_i8>(%v0, %v1, %v_mix) : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> "
        "reg<amdgpu.vgpr>\n"
        "  %dot_u = low.op<amdgpu.v_dot4_u32_u8>(%v0, %v1, %dot_s) : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> "
        "reg<amdgpu.vgpr>\n"
        "  low.op<amdgpu.buffer_store_dword>(%dot_u, %resource, %vaddr, "
        "%s_sum) {offset = 0} : (reg<amdgpu.vgpr>, "
        "reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
        "  low.return\n"
        "}\n",
        &arena, &packetization);

    iree_const_byte_span_t text = iree_const_byte_span_empty();
    IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
        &packetization.schedule, &packetization.allocation, &text, &arena));

    ASSERT_GT(text.data_length, 4u);
    EXPECT_EQ(text.data_length % 4, 0u);
    EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4),
              test_case.expected_return_word);
    iree_arena_deinitialize(&arena);
  }
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

TEST_F(AmdgpuEncodingTest, EncodesGfx11MubufOffZeroLoadStoreAndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_packetization_t packetization = {};
  BuildGfx11Sidecars(
      "low.func.def target(@gfx_target) @gfx_kernel(%resource : "
      "reg<amdgpu.sgpr x4>) {\n"
      "  %loaded = low.op<amdgpu.buffer_load_dword_off_zero>(%resource) "
      "{offset = 12} : (reg<amdgpu.sgpr x4>) -> reg<amdgpu.vgpr>\n"
      "  low.op<amdgpu.buffer_store_dword_off_zero>(%loaded, %resource) "
      "{offset = 16} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x4>)\n"
      "  low.return\n"
      "}\n",
      &arena, &packetization);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &packetization.schedule, &packetization.allocation, &text, &arena));

  ASSERT_EQ(text.data_length, 20u);
  EXPECT_EQ(ReadU32LE(text.data + 0), UINT32_C(0xE050000C));
  EXPECT_EQ(ReadU32LE(text.data + 4), UINT32_C(0x80000000));
  EXPECT_EQ(ReadU32LE(text.data + 8), UINT32_C(0xE0680010));
  EXPECT_EQ(ReadU32LE(text.data + 12), UINT32_C(0x80000000));
  EXPECT_EQ(ReadU32LE(text.data + 16), UINT32_C(0xBFB00000));
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

TEST_F(AmdgpuEncodingTest, EncodesBufferB128ForCurrentAmdgpuFamilies) {
  struct Case {
    // Target preset used to select the low descriptor set.
    const char* preset_key;
    // Descriptor key for a 128-bit buffer load on this descriptor set.
    const char* load_key;
    // Descriptor key for a 128-bit buffer store on this descriptor set.
    const char* store_key;
    // Expected little-endian SOPP instruction word for `s_endpgm 0`.
    uint32_t expected_return_word;
  };
  const Case cases[] = {
      {"amdgpu-gfx950", "amdgpu.buffer_load_dwordx4",
       "amdgpu.buffer_store_dwordx4", UINT32_C(0xBF810000)},
      {"amdgpu-gfx11", "amdgpu.buffer_load_b128", "amdgpu.buffer_store_b128",
       UINT32_C(0xBFB00000)},
      {"amdgpu-gfx12", "amdgpu.buffer_load_b128", "amdgpu.buffer_store_b128",
       UINT32_C(0xBFB00000)},
      {"amdgpu-gfx1250", "amdgpu.buffer_load_b128", "amdgpu.buffer_store_b128",
       UINT32_C(0xBFB00000)},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.preset_key);
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool_, &arena);
    loom_low_packetization_t packetization = {};
    std::string body =
        "low.func.def target(@gfx_target) @gfx_kernel(%resource : "
        "reg<amdgpu.sgpr x4>, %vaddr : reg<amdgpu.vgpr>, %soffset : "
        "reg<amdgpu.sgpr>) {\n"
        "  %loaded = low.op<";
    body += test_case.load_key;
    body +=
        ">(%resource, %vaddr, %soffset) {offset = 16} : "
        "(reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>) -> "
        "reg<amdgpu.vgpr x4>\n"
        "  low.op<";
    body += test_case.store_key;
    body +=
        ">(%loaded, %resource, %vaddr, %soffset) {offset = 32} : "
        "(reg<amdgpu.vgpr x4>, reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, "
        "reg<amdgpu.sgpr>)\n"
        "  low.return\n"
        "}\n";
    BuildSidecarsForPreset(test_case.preset_key, body.c_str(), &arena,
                           &packetization);

    iree_const_byte_span_t text = iree_const_byte_span_empty();
    IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
        &packetization.schedule, &packetization.allocation, &text, &arena));

    ASSERT_GE(text.data_length, 20u);
    EXPECT_EQ(text.data_length % 4, 0u);
    EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4),
              test_case.expected_return_word);
    iree_arena_deinitialize(&arena);
  }
}

TEST_F(AmdgpuEncodingTest, EncodesGlobalPointerB128ForCurrentAmdgpuFamilies) {
  struct Case {
    // Target preset used to select the low descriptor set.
    const char* preset_key;
    // Whether the target descriptor exposes m0 as an architectural input.
    bool uses_m0;
    // Expected little-endian SOPP instruction word for `s_endpgm 0`.
    uint32_t expected_return_word;
  };
  const Case cases[] = {
      {"amdgpu-gfx950", true, UINT32_C(0xBF810000)},
      {"amdgpu-gfx11", false, UINT32_C(0xBFB00000)},
      {"amdgpu-gfx12", false, UINT32_C(0xBFB00000)},
      {"amdgpu-gfx1250", false, UINT32_C(0xBFB00000)},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.preset_key);
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool_, &arena);
    loom_low_packetization_t packetization = {};
    std::string body =
        "low.func.def target(@gfx_target) @gfx_kernel(%addr : "
        "reg<amdgpu.vgpr x2>";
    if (test_case.uses_m0) {
      body += ", %m0 : reg<amdgpu.m0>";
    }
    body +=
        ") {\n"
        "  %loaded = low.op<amdgpu.global_load_b128>(%addr";
    if (test_case.uses_m0) {
      body += ", %m0";
    }
    body += ") {offset = -16} : (reg<amdgpu.vgpr x2>";
    if (test_case.uses_m0) {
      body += ", reg<amdgpu.m0>";
    }
    body +=
        ") -> reg<amdgpu.vgpr x4>\n"
        "  low.op<amdgpu.global_store_b128>(%addr, %loaded";
    if (test_case.uses_m0) {
      body += ", %m0";
    }
    body += ") {offset = 32} : (reg<amdgpu.vgpr x2>, reg<amdgpu.vgpr x4>";
    if (test_case.uses_m0) {
      body += ", reg<amdgpu.m0>";
    }
    body +=
        ")\n"
        "  low.return\n"
        "}\n";
    BuildSidecarsForPreset(test_case.preset_key, body.c_str(), &arena,
                           &packetization);

    iree_const_byte_span_t text = iree_const_byte_span_empty();
    IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
        &packetization.schedule, &packetization.allocation, &text, &arena));

    ASSERT_GT(text.data_length, 4u);
    EXPECT_EQ(text.data_length % 4, 0u);
    EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4),
              test_case.expected_return_word);
    iree_arena_deinitialize(&arena);
  }
}

TEST_F(AmdgpuEncodingTest, EncodesGlobalSaddrB128ForCurrentAmdgpuFamilies) {
  struct Case {
    // Target preset used to select the low descriptor set.
    const char* preset_key;
    // Whether the target descriptor exposes m0 as an architectural input.
    bool uses_m0;
    // Expected little-endian SOPP instruction word for `s_endpgm 0`.
    uint32_t expected_return_word;
  };
  const Case cases[] = {
      {"amdgpu-gfx950", true, UINT32_C(0xBF810000)},
      {"amdgpu-gfx11", false, UINT32_C(0xBFB00000)},
      {"amdgpu-gfx12", false, UINT32_C(0xBFB00000)},
      {"amdgpu-gfx1250", false, UINT32_C(0xBFB00000)},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.preset_key);
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool_, &arena);
    loom_low_packetization_t packetization = {};
    std::string body =
        "low.func.def target(@gfx_target) @gfx_kernel(%addr : "
        "reg<amdgpu.vgpr>, %saddr : reg<amdgpu.sgpr x2>";
    if (test_case.uses_m0) {
      body += ", %m0 : reg<amdgpu.m0>";
    }
    body +=
        ") {\n"
        "  %loaded = low.op<amdgpu.global_load_b128_saddr>(%addr, %saddr";
    if (test_case.uses_m0) {
      body += ", %m0";
    }
    body += ") {offset = 4} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>";
    if (test_case.uses_m0) {
      body += ", reg<amdgpu.m0>";
    }
    body +=
        ") -> reg<amdgpu.vgpr x4>\n"
        "  low.op<amdgpu.global_store_b128_saddr>(%addr, %loaded, %saddr";
    if (test_case.uses_m0) {
      body += ", %m0";
    }
    body +=
        ") {offset = 8} : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr x4>, "
        "reg<amdgpu.sgpr x2>";
    if (test_case.uses_m0) {
      body += ", reg<amdgpu.m0>";
    }
    body +=
        ")\n"
        "  low.return\n"
        "}\n";
    BuildSidecarsForPreset(test_case.preset_key, body.c_str(), &arena,
                           &packetization);

    iree_const_byte_span_t text = iree_const_byte_span_empty();
    IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
        &packetization.schedule, &packetization.allocation, &text, &arena));

    ASSERT_GT(text.data_length, 4u);
    EXPECT_EQ(text.data_length % 4, 0u);
    EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4),
              test_case.expected_return_word);
    iree_arena_deinitialize(&arena);
  }
}

TEST_F(AmdgpuEncodingTest, EncodesGfx11MubufB64LoadStoreAndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_packetization_t packetization = {};
  BuildGfx11Sidecars(
      "low.func.def target(@gfx_target) @gfx_kernel(%resource : "
      "reg<amdgpu.sgpr x4>, %vaddr : reg<amdgpu.vgpr>, %soffset : "
      "reg<amdgpu.sgpr>) {\n"
      "  %loaded = low.op<amdgpu.buffer_load_b64>(%resource, %vaddr, "
      "%soffset) {offset = 8} : (reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, "
      "reg<amdgpu.sgpr>) -> reg<amdgpu.vgpr x2>\n"
      "  low.op<amdgpu.buffer_store_b64>(%loaded, %resource, %vaddr, "
      "%soffset) {offset = 16} : (reg<amdgpu.vgpr x2>, "
      "reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
      "  low.return\n"
      "}\n",
      &arena, &packetization);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &packetization.schedule, &packetization.allocation, &text, &arena));

  ASSERT_GE(text.data_length, 12u);
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesDsMemoryForCurrentAmdgpuFamilies) {
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
    BuildSidecarsForPreset(
        test_case.preset_key,
        "low.func.def target(@gfx_target) @gfx_kernel(%addr : "
        "reg<amdgpu.vgpr>) {\n"
        "  %loaded32 = low.op<amdgpu.ds_read_b32>(%addr) {offset = 4} : "
        "(reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        "  low.op<amdgpu.ds_write_b32>(%addr, %loaded32) {offset = 4} : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>)\n"
        "  %loaded64 = low.op<amdgpu.ds_read_b64>(%addr) {offset = 8} : "
        "(reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x2>\n"
        "  low.op<amdgpu.ds_write_b64>(%addr, %loaded64) {offset = 8} : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr x2>)\n"
        "  %loaded96 = low.op<amdgpu.ds_read_b96>(%addr) {offset = 12} : "
        "(reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x3>\n"
        "  low.op<amdgpu.ds_write_b96>(%addr, %loaded96) {offset = 12} : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr x3>)\n"
        "  %loaded128 = low.op<amdgpu.ds_read_b128>(%addr) {offset = 16} : "
        "(reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x4>\n"
        "  low.op<amdgpu.ds_write_b128>(%addr, %loaded128) {offset = 16} : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr x4>)\n"
        "  low.return\n"
        "}\n",
        &arena, &packetization);

    iree_const_byte_span_t text = iree_const_byte_span_empty();
    IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
        &packetization.schedule, &packetization.allocation, &text, &arena));

    ASSERT_GT(text.data_length, 4u);
    EXPECT_EQ(text.data_length % 4, 0u);
    EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4),
              test_case.expected_return_word);
    iree_arena_deinitialize(&arena);
  }
}

TEST_F(AmdgpuEncodingTest, EncodesGfx11DsMemoryBarrierAndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_packetization_t packetization = {};
  BuildGfx11Sidecars(
      "low.func.def target(@gfx_target) @gfx_kernel(%addr : "
      "reg<amdgpu.vgpr>, %value64 : reg<amdgpu.vgpr x2>, %value96 : "
      "reg<amdgpu.vgpr x3>, %value128 : reg<amdgpu.vgpr x4>) {\n"
      "  low.op<amdgpu.ds_write_b64>(%addr, %value64) {offset = 8} : "
      "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr x2>)\n"
      "  low.op<amdgpu.ds_write_b96>(%addr, %value96) {offset = 12} : "
      "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr x3>)\n"
      "  low.op<amdgpu.ds_write_b128>(%addr, %value128) {offset = 16} : "
      "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr x4>)\n"
      "  low.op<amdgpu.s_barrier>() : ()\n"
      "  %loaded64 = low.op<amdgpu.ds_read_b64>(%addr) {offset = 8} : "
      "(reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x2>\n"
      "  %loaded96 = low.op<amdgpu.ds_read_b96>(%addr) {offset = 12} : "
      "(reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x3>\n"
      "  %loaded = low.op<amdgpu.ds_read_b128>(%addr) {offset = 16} : "
      "(reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x4>\n"
      "  low.return\n"
      "}\n",
      &arena, &packetization);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &packetization.schedule, &packetization.allocation, &text, &arena));

  ASSERT_GE(text.data_length, 16u);
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesGfx11Ds2AddrMemoryAndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_packetization_t packetization = {};
  BuildGfx11Sidecars(
      "low.func.def target(@gfx_target) @gfx_kernel(%addr : "
      "reg<amdgpu.vgpr>, %value32a : reg<amdgpu.vgpr>, %value32b : "
      "reg<amdgpu.vgpr>, %value64a : reg<amdgpu.vgpr x2>, "
      "%value64b : reg<amdgpu.vgpr x2>) {\n"
      "  %loaded32 = low.op<amdgpu.ds_read2_b32>(%addr) {offset0 = 1, "
      "offset1 = 2} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x2>\n"
      "  low.op<amdgpu.ds_write2_b32>(%addr, %value32a, %value32b) "
      "{offset0 = 3, offset1 = 4} : (reg<amdgpu.vgpr>, "
      "reg<amdgpu.vgpr>, reg<amdgpu.vgpr>)\n"
      "  %loaded64st64 = low.op<amdgpu.ds_read2st64_b64>(%addr) "
      "{offset0 = 5, offset1 = 6} : (reg<amdgpu.vgpr>) -> "
      "reg<amdgpu.vgpr x4>\n"
      "  low.op<amdgpu.ds_write2st64_b64>(%addr, %value64a, %value64b) "
      "{offset0 = 7, offset1 = 8} : (reg<amdgpu.vgpr>, "
      "reg<amdgpu.vgpr x2>, reg<amdgpu.vgpr x2>)\n"
      "  low.return\n"
      "}\n",
      &arena, &packetization);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &packetization.schedule, &packetization.allocation, &text, &arena));

  ASSERT_GT(text.data_length, 4u);
  EXPECT_EQ(text.data_length % 4, 0u);
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesDsAddtidMemoryForCurrentAmdgpuFamilies) {
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
    BuildSidecarsForPreset(
        test_case.preset_key,
        "low.func.def target(@gfx_target) @gfx_kernel(%m0 : reg<amdgpu.m0>, "
        "%value : reg<amdgpu.vgpr>) {\n"
        "  %loaded = low.op<amdgpu.ds_read_addtid_b32>(%m0) {offset = 16} : "
        "(reg<amdgpu.m0>) -> reg<amdgpu.vgpr>\n"
        "  low.op<amdgpu.ds_write_addtid_b32>(%value, %m0) {offset = 20} : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.m0>)\n"
        "  low.return\n"
        "}\n",
        &arena, &packetization);

    iree_const_byte_span_t text = iree_const_byte_span_empty();
    IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
        &packetization.schedule, &packetization.allocation, &text, &arena));

    ASSERT_GT(text.data_length, 4u);
    EXPECT_EQ(text.data_length % 4, 0u);
    EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4),
              test_case.expected_return_word);
    iree_arena_deinitialize(&arena);
  }
}

TEST_F(AmdgpuEncodingTest, EncodesDsCrosslaneForCurrentAmdgpuFamilies) {
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
    BuildSidecarsForPreset(
        test_case.preset_key,
        "low.func.def target(@gfx_target) @gfx_kernel(%addr : "
        "reg<amdgpu.vgpr>) {\n"
        "  %swizzled = low.op<amdgpu.ds_swizzle_b32>(%addr) {offset = "
        "32} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        "  %permuted = low.op<amdgpu.ds_permute_b32>(%addr, %swizzled) "
        "{offset = 4} : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> "
        "reg<amdgpu.vgpr>\n"
        "  %bpermuted = low.op<amdgpu.ds_bpermute_b32>(%addr, "
        "%permuted) {offset = 8} : (reg<amdgpu.vgpr>, "
        "reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        "  low.op<amdgpu.ds_write_b32>(%addr, %bpermuted) {offset = "
        "12} : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>)\n"
        "  low.return\n"
        "}\n",
        &arena, &packetization);

    iree_const_byte_span_t text = iree_const_byte_span_empty();
    IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
        &packetization.schedule, &packetization.allocation, &text, &arena));

    ASSERT_GT(text.data_length, 4u);
    EXPECT_EQ(text.data_length % 4, 0u);
    EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4),
              test_case.expected_return_word);
    iree_arena_deinitialize(&arena);
  }
}

TEST_F(AmdgpuEncodingTest, EncodesGfx12DsCrosslaneFetchInvalidAndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_packetization_t packetization = {};
  BuildSidecarsForPreset(
      "amdgpu-gfx12",
      "low.func.def target(@gfx_target) @gfx_kernel(%addr : "
      "reg<amdgpu.vgpr>, %value : reg<amdgpu.vgpr>) {\n"
      "  %bpermuted = low.op<amdgpu.ds_bpermute_fi_b32>(%addr, "
      "%value) {offset = 16} : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) "
      "-> reg<amdgpu.vgpr>\n"
      "  low.op<amdgpu.ds_write_b32>(%addr, %bpermuted) {offset = 20} "
      ": (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>)\n"
      "  low.return\n"
      "}\n",
      &arena, &packetization);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &packetization.schedule, &packetization.allocation, &text, &arena));

  ASSERT_GT(text.data_length, 4u);
  EXPECT_EQ(text.data_length % 4, 0u);
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesGfx950DsTransposeReadsAndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_packetization_t packetization = {};
  BuildSidecarsForPreset(
      "amdgpu-gfx950",
      "low.func.def target(@gfx_target) @gfx_kernel(%addr : "
      "reg<amdgpu.vgpr>) {\n"
      "  %loaded_b4 = low.op<amdgpu.ds_read_b64_tr_b4>(%addr) "
      "{offset = 0} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x2>\n"
      "  %loaded_b6 = low.op<amdgpu.ds_read_b96_tr_b6>(%addr) "
      "{offset = 16} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x3>\n"
      "  %loaded_b8 = low.op<amdgpu.ds_read_b64_tr_b8>(%addr) "
      "{offset = 32} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x2>\n"
      "  %loaded_b16 = low.op<amdgpu.ds_read_b64_tr_b16>(%addr) "
      "{offset = 48} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x2>\n"
      "  low.op<amdgpu.ds_write_b64>(%addr, %loaded_b4) {offset = 64} "
      ": (reg<amdgpu.vgpr>, reg<amdgpu.vgpr x2>)\n"
      "  low.op<amdgpu.ds_write_b96>(%addr, %loaded_b6) {offset = 80} "
      ": (reg<amdgpu.vgpr>, reg<amdgpu.vgpr x3>)\n"
      "  low.op<amdgpu.ds_write_b64>(%addr, %loaded_b8) {offset = 96} "
      ": (reg<amdgpu.vgpr>, reg<amdgpu.vgpr x2>)\n"
      "  low.op<amdgpu.ds_write_b64>(%addr, %loaded_b16) {offset = "
      "112} : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr x2>)\n"
      "  low.return\n"
      "}\n",
      &arena, &packetization);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &packetization.schedule, &packetization.allocation, &text, &arena));

  ASSERT_GT(text.data_length, 4u);
  EXPECT_EQ(text.data_length % 4, 0u);
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBF810000));
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

TEST_F(AmdgpuEncodingTest, EncodesRdnaWmmaPacketAndReturn) {
  struct Case {
    // Target preset used to select the low descriptor set.
    const char* preset_key;
    // Expected little-endian SOPP instruction word for `s_endpgm 0`.
    uint32_t expected_return_word;
  };
  const Case cases[] = {
      {"amdgpu-gfx11", UINT32_C(0xBFB00000)},
      {"amdgpu-gfx12", UINT32_C(0xBFB00000)},
      {"amdgpu-gfx1250", UINT32_C(0xBFB00000)},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.preset_key);
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool_, &arena);
    loom_low_packetization_t packetization = {};
    BuildSidecarsForPreset(
        test_case.preset_key,
        "low.func.def target(@gfx_target) @gfx_kernel(%a : "
        "reg<amdgpu.vgpr x4>, %b : reg<amdgpu.vgpr x4>, %acc : "
        "reg<amdgpu.vgpr x8>, %resource : reg<amdgpu.sgpr x4>, "
        "%vaddr : reg<amdgpu.vgpr>, %soffset : reg<amdgpu.sgpr>) {\n"
        "  %out = low.op<amdgpu.v_wmma_f32_16x16x16_f16>(%a, %b, %acc) : "
        "(reg<amdgpu.vgpr x4>, reg<amdgpu.vgpr x4>, reg<amdgpu.vgpr x8>) "
        "-> %acc as reg<amdgpu.vgpr x8>\n"
        "  %out_low = low.slice %out[0] : reg<amdgpu.vgpr x8> -> "
        "reg<amdgpu.vgpr x4>\n"
        "  low.op<amdgpu.buffer_store_b128>(%out_low, %resource, %vaddr, "
        "%soffset) {offset = 0} : (reg<amdgpu.vgpr x4>, "
        "reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
        "  low.return\n"
        "}\n",
        &arena, &packetization);

    iree_const_byte_span_t text = iree_const_byte_span_empty();
    IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
        &packetization.schedule, &packetization.allocation, &text, &arena));

    ASSERT_GT(text.data_length, 12u);
    EXPECT_EQ(text.data_length % 4, 0u);
    EXPECT_NE(ReadU32LE(text.data), UINT32_C(0));
    EXPECT_NE(ReadU32LE(text.data + 4), UINT32_C(0));
    EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4),
              test_case.expected_return_word);
    iree_arena_deinitialize(&arena);
  }
}

TEST_F(AmdgpuEncodingTest, EncodesGfx950MfmaPacketAndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_packetization_t packetization = {};
  BuildSidecarsForPreset(
      "amdgpu-gfx950",
      "low.func.def target(@gfx_target) @gfx_kernel(%a : "
      "reg<amdgpu.vgpr x2>, %b : reg<amdgpu.vgpr x2>, %acc : "
      "reg<amdgpu.vgpr x4>, %vaddr : reg<amdgpu.vgpr>) {\n"
      "  %out = low.op<amdgpu.v_mfma_f32_16x16x16_f16>(%a, %b, %acc) : "
      "(reg<amdgpu.vgpr x2>, reg<amdgpu.vgpr x2>, reg<amdgpu.vgpr x4>) "
      "-> %acc as reg<amdgpu.vgpr x4>\n"
      "  low.op<amdgpu.ds_write_b128>(%vaddr, %out) {offset = 0} : "
      "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr x4>)\n"
      "  low.return\n"
      "}\n",
      &arena, &packetization);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &packetization.schedule, &packetization.allocation, &text, &arena));

  ASSERT_GT(text.data_length, 12u);
  EXPECT_EQ(text.data_length % 4, 0u);
  EXPECT_NE(ReadU32LE(text.data), UINT32_C(0));
  EXPECT_NE(ReadU32LE(text.data + 4), UINT32_C(0));
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBF810000));
  iree_arena_deinitialize(&arena);
}

}  // namespace
}  // namespace loom
