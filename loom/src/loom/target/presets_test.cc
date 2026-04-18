// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/presets.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/target/ops.h"
#include "loom/target/ir_records.h"
#include "loom/target/low_descriptor_registry.h"

namespace loom {
namespace {

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

class TargetPresetsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_op_registry_initialize_context(iree_allocator_system(),
                                                       &context_));
    loom_target_low_descriptor_registry_initialize(&registry_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* ParseSource(const char* source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;

    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(
        iree_make_cstring_view(source), IREE_SV("target_presets_test.loom"),
        &context_, &block_pool_, &parse_options, &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  loom_op_t* FindFirstOp(loom_module_t* module, loom_op_kind_t kind) {
    loom_block_t* block = loom_module_block(module);
    loom_op_t* op = nullptr;
    loom_block_for_each_op(block, op) {
      if (op->kind == kind) return op;
    }
    return nullptr;
  }

  bool HasSymbol(loom_module_t* module, iree_string_view_t name) {
    loom_string_id_t name_id = loom_module_lookup_string(module, name);
    if (name_id == LOOM_STRING_ID_INVALID) return false;
    return loom_module_find_symbol(module, name_id) != LOOM_SYMBOL_ID_INVALID;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t registry_;
};

TEST_F(TargetPresetsTest, ExpandsPresetToConcreteTargetRecords) {
  const char source[] =
      "target.preset @test_target {key = \"test-low\", source = @sched}\n"
      "func.def @sched() {\n"
      "  func.return\n"
      "}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);

  iree_host_size_t expanded_count = 0;
  const loom_target_preset_registry_t preset_registry =
      loom_target_low_descriptor_registry_presets(&registry_);
  IREE_ASSERT_OK(
      loom_target_expand_presets(module, &preset_registry, &expanded_count));
  EXPECT_EQ(expanded_count, 1u);
  EXPECT_EQ(FindFirstOp(module, LOOM_OP_TARGET_PRESET), nullptr);
  ASSERT_NE(FindFirstOp(module, LOOM_OP_TARGET_SNAPSHOT), nullptr);
  ASSERT_NE(FindFirstOp(module, LOOM_OP_TARGET_EXPORT), nullptr);
  ASSERT_NE(FindFirstOp(module, LOOM_OP_TARGET_CONFIG), nullptr);
  ASSERT_NE(FindFirstOp(module, LOOM_OP_TARGET_BUNDLE), nullptr);

  loom_target_ir_bundle_storage_t storage = {};
  IREE_ASSERT_OK(loom_target_ir_bundle_from_symbol_name(
      module, IREE_SV("test_target"), &storage));
  EXPECT_EQ(ToString(storage.bundle.name), "test_target");
  EXPECT_EQ(storage.snapshot.codegen_format,
            LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE);
  EXPECT_EQ(storage.snapshot.artifact_format, LOOM_TARGET_ARTIFACT_FORMAT_ELF);
  EXPECT_EQ(storage.export_plan.abi_kind, LOOM_TARGET_ABI_OBJECT_FUNCTION);
  EXPECT_EQ(ToString(storage.export_plan.source_symbol), "sched");
  EXPECT_EQ(ToString(storage.export_plan.export_symbol), "sched");
  EXPECT_EQ(ToString(storage.config.contract_set_key), "test.low.core");

  loom_module_free(module);
}

TEST_F(TargetPresetsTest, RejectsUnknownPresetKey) {
  const char source[] =
      "target.preset @test_target {key = \"missing-target\", source = @sched}\n"
      "func.def @sched() {\n"
      "  func.return\n"
      "}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);

  iree_host_size_t expanded_count = 0;
  const loom_target_preset_registry_t preset_registry =
      loom_target_low_descriptor_registry_presets(&registry_);
  iree_status_t status =
      loom_target_expand_presets(module, &preset_registry, &expanded_count);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
  EXPECT_EQ(expanded_count, 0u);

  loom_module_free(module);
}

TEST_F(TargetPresetsTest, RejectsDuplicateGeneratedSymbols) {
  const char source[] =
      "target.config @test_target__snapshot {contract_set_key = "
      "\"test.low.core\", contract_feature_bits = 0}\n"
      "target.preset @test_target {key = \"test-low\", source = @sched}\n"
      "func.def @sched() {\n"
      "  func.return\n"
      "}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);

  iree_host_size_t expanded_count = 0;
  const loom_target_preset_registry_t preset_registry =
      loom_target_low_descriptor_registry_presets(&registry_);
  iree_status_t status =
      loom_target_expand_presets(module, &preset_registry, &expanded_count);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS, status);
  EXPECT_EQ(expanded_count, 0u);

  loom_module_free(module);
}

TEST_F(TargetPresetsTest, RejectsLaterDuplicateWithoutAddingEarlierSiblings) {
  const char source[] =
      "target.config @test_target__config {contract_set_key = "
      "\"test.low.core\", contract_feature_bits = 0}\n"
      "target.preset @test_target {key = \"test-low\", source = @sched}\n"
      "func.def @sched() {\n"
      "  func.return\n"
      "}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);

  iree_host_size_t expanded_count = 0;
  const loom_target_preset_registry_t preset_registry =
      loom_target_low_descriptor_registry_presets(&registry_);
  iree_status_t status =
      loom_target_expand_presets(module, &preset_registry, &expanded_count);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS, status);
  EXPECT_EQ(expanded_count, 0u);
  EXPECT_FALSE(HasSymbol(module, IREE_SV("test_target__snapshot")));
  EXPECT_FALSE(HasSymbol(module, IREE_SV("test_target__export")));
  EXPECT_TRUE(HasSymbol(module, IREE_SV("test_target__config")));

  loom_module_free(module);
}

TEST_F(TargetPresetsTest, RejectsUnresolvedExportSymbol) {
  const char source[] =
      "target.preset @test_target {key = \"test-low\", source = @missing}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);

  iree_host_size_t expanded_count = 0;
  const loom_target_preset_registry_t preset_registry =
      loom_target_low_descriptor_registry_presets(&registry_);
  iree_status_t status =
      loom_target_expand_presets(module, &preset_registry, &expanded_count);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
  EXPECT_EQ(expanded_count, 0u);

  loom_module_free(module);
}

}  // namespace
}  // namespace loom
