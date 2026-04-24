// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/assembly.h"

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
#include "loom/testing/context.h"

namespace loom {
namespace {

class AmdgpuAssemblyTest : public ::testing::Test {
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
                        IREE_SV("amdgpu_assembly_test.loom"), &context_,
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
        "low.func.def target(@gfx11_target) @gfx11_fragment(%source : "
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
        "target.profile @gfx11_target preset(\"amdgpu-gfx11\")\n";
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

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_target_low_descriptor_registry_t target_registry_ = {};
};

TEST_F(AmdgpuAssemblyTest, SequencesOverlappingCopyBeforeClobber) {
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  BuildShiftedCopySidecars(&sidecar_arena, &packetization);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_amdgpu_emit_assembly_fragment(
      &packetization.schedule, &packetization.allocation, &builder));
  const std::string output(iree_string_builder_view(&builder).data,
                           iree_string_builder_view(&builder).size);
  const size_t move_s3 = output.find("s_mov_b32 s3, s2");
  const size_t move_s2 = output.find("s_mov_b32 s2, s1");
  const size_t move_s1 = output.find("s_mov_b32 s1, s0");
  EXPECT_NE(move_s3, std::string::npos);
  EXPECT_NE(move_s2, std::string::npos);
  EXPECT_NE(move_s1, std::string::npos);
  EXPECT_LT(move_s3, move_s2);
  EXPECT_LT(move_s2, move_s1);
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&sidecar_arena);
}

}  // namespace
}  // namespace loom
