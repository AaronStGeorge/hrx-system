// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/pipeline.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/pass/ops.h"
#include "loom/pass/builder.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

static iree_status_t ContributeMockPipeline(
    const loom_target_pipeline_contribution_t* contribution) {
  iree_string_view_t pass_key = iree_string_view_empty();
  switch (contribution->phase) {
    case LOOM_TARGET_PIPELINE_PHASE_SOURCE_NORMALIZATION:
      pass_key = IREE_SV("mock-source-normalization");
      break;
    case LOOM_TARGET_PIPELINE_PHASE_SOURCE_TO_LOW:
      pass_key = IREE_SV("mock-source-to-low");
      break;
    case LOOM_TARGET_PIPELINE_PHASE_TARGET_LOW_MATERIALIZATION:
      pass_key = IREE_SV("mock-target-low-materialization");
      break;
    case LOOM_TARGET_PIPELINE_PHASE_TARGET_LOW_PREPARATION:
      pass_key = IREE_SV("mock-target-low-preparation");
      break;
    case LOOM_TARGET_PIPELINE_PHASE_COUNT_:
      return iree_ok_status();
  }
  loom_op_t* run_op = nullptr;
  return loom_pass_ir_build_run(contribution->builder, pass_key,
                                loom_named_attr_slice_empty(), &run_op);
}

static const loom_target_provider_t kMockProvider = {
    .contribute_pipeline = ContributeMockPipeline,
};

static const loom_target_provider_t* const kMockProviders[] = {
    &kMockProvider,
};

static iree_string_view_t RunKey(loom_module_t* module, const loom_op_t* op) {
  return module->strings.entries[loom_pass_run_key(op)];
}

static void ExpectRunKey(loom_module_t* module, const loom_op_t* op,
                         iree_string_view_t expected) {
  ASSERT_TRUE(loom_pass_run_isa(op));
  EXPECT_TRUE(iree_string_view_equal(RunKey(module, op), expected))
      << "expected " << std::string(expected.data, expected.size) << ", got "
      << std::string(RunKey(module, op).data, RunKey(module, op).size);
}

static void ExpectRunKeySequence(loom_module_t* module, loom_block_t* block,
                                 const iree_string_view_t* expected_keys,
                                 iree_host_size_t expected_key_count) {
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->op_count, expected_key_count + 1);

  loom_op_t* op = block->first_op;
  for (iree_host_size_t i = 0; i < expected_key_count; ++i) {
    ASSERT_NE(op, nullptr);
    ExpectRunKey(module, op, expected_keys[i]);
    op = op->next_op;
  }
  ASSERT_NE(op, nullptr);
  EXPECT_TRUE(loom_pass_yield_isa(op));
  EXPECT_EQ(op, block->last_op);
}

class TargetPipelineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_op_registry_initialize_context(iree_allocator_system(),
                                                       &context_));
    provider_set_ = loom_target_provider_set_make(
        kMockProviders, IREE_ARRAYSIZE(kMockProviders));
    IREE_ASSERT_OK(loom_target_environment_initialize(&provider_set_,
                                                      &target_environment_));
  }

  void TearDown() override {
    loom_target_environment_deinitialize(&target_environment_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t AllocateModule(iree_string_view_t name, ModulePtr* out_module) {
    *out_module = nullptr;
    loom_module_t* module = nullptr;
    IREE_RETURN_IF_ERROR(loom_module_allocate(&context_, name, &block_pool_,
                                              nullptr, iree_allocator_system(),
                                              &module));
    *out_module = ModulePtr(module);
    return iree_ok_status();
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_provider_set_t provider_set_;
  loom_target_environment_t target_environment_;
};

TEST_F(TargetPipelineTest, BuildsVisibleSourceLowPipeline) {
  ModulePtr module;
  IREE_ASSERT_OK(AllocateModule(IREE_SV("pipeline"), &module));

  loom_op_t* pipeline_op = nullptr;
  IREE_ASSERT_OK(loom_target_pipeline_build_to_source_low(
      module.get(), IREE_SV("source_low"), /*options=*/nullptr,
      &target_environment_, loom_pass_environment_empty(), &pipeline_op));

  loom_block_t* pipeline_body =
      loom_region_entry_block(loom_pass_pipeline_body(pipeline_op));
  ASSERT_NE(pipeline_body, nullptr);
  ASSERT_EQ(pipeline_body->op_count, 5u);

  loom_op_t* source_for = pipeline_body->first_op;
  ASSERT_TRUE(loom_pass_for_isa(source_for));
  loom_block_t* source_body =
      loom_region_entry_block(loom_pass_for_body(source_for));
  const iree_string_view_t source_keys[] = {
      IREE_SV("mock-source-normalization"),
      IREE_SV("normalize-kernel-resources"),
      IREE_SV("promote-private-fragments"),
      IREE_SV("linearize-view-accesses"),
      IREE_SV("canonicalize"),
      IREE_SV("cse"),
      IREE_SV("scf-to-cfg"),
      IREE_SV("cfg-simplify"),
      IREE_SV("canonicalize"),
      IREE_SV("cse"),
  };
  ExpectRunKeySequence(module.get(), source_body, source_keys,
                       IREE_ARRAYSIZE(source_keys));

  loom_op_t* source_to_low_hook = source_for->next_op;
  ExpectRunKey(module.get(), source_to_low_hook, IREE_SV("mock-source-to-low"));

  loom_op_t* source_to_low = source_to_low_hook->next_op;
  ExpectRunKey(module.get(), source_to_low, IREE_SV("source-to-low"));

  loom_op_t* source_low_cleanup_for = source_to_low->next_op;
  ASSERT_TRUE(loom_pass_for_isa(source_low_cleanup_for));
  loom_block_t* source_low_cleanup_body =
      loom_region_entry_block(loom_pass_for_body(source_low_cleanup_for));
  const iree_string_view_t cleanup_keys[] = {
      IREE_SV("canonicalize"),
      IREE_SV("cse"),
      IREE_SV("low-dce"),
  };
  ExpectRunKeySequence(module.get(), source_low_cleanup_body, cleanup_keys,
                       IREE_ARRAYSIZE(cleanup_keys));
  EXPECT_TRUE(loom_pass_yield_isa(pipeline_body->last_op));
}

TEST_F(TargetPipelineTest, BuildsVisiblePreparedLowPipeline) {
  ModulePtr module;
  IREE_ASSERT_OK(AllocateModule(IREE_SV("pipeline"), &module));

  const loom_target_pipeline_options_t options = {
      .source_to_low_max_errors = 7,
  };
  loom_op_t* pipeline_op = nullptr;
  IREE_ASSERT_OK(loom_target_pipeline_build_to_prepared_low(
      module.get(), IREE_SV("prepare_low"), &options, &target_environment_,
      loom_pass_environment_empty(), &pipeline_op));

  loom_block_t* pipeline_body =
      loom_region_entry_block(loom_pass_pipeline_body(pipeline_op));
  ASSERT_NE(pipeline_body, nullptr);
  ASSERT_EQ(pipeline_body->op_count, 6u);

  loom_op_t* source_for = pipeline_body->first_op;
  ASSERT_TRUE(loom_pass_for_isa(source_for));
  loom_block_t* source_body =
      loom_region_entry_block(loom_pass_for_body(source_for));
  const iree_string_view_t source_keys[] = {
      IREE_SV("mock-source-normalization"),
      IREE_SV("normalize-kernel-resources"),
      IREE_SV("promote-private-fragments"),
      IREE_SV("linearize-view-accesses"),
      IREE_SV("canonicalize"),
      IREE_SV("cse"),
      IREE_SV("scf-to-cfg"),
      IREE_SV("cfg-simplify"),
      IREE_SV("canonicalize"),
      IREE_SV("cse"),
  };
  ExpectRunKeySequence(module.get(), source_body, source_keys,
                       IREE_ARRAYSIZE(source_keys));

  loom_op_t* source_to_low_hook = source_for->next_op;
  ExpectRunKey(module.get(), source_to_low_hook, IREE_SV("mock-source-to-low"));

  loom_op_t* source_to_low = source_to_low_hook->next_op;
  ExpectRunKey(module.get(), source_to_low, IREE_SV("source-to-low"));
  loom_attribute_t options_attr =
      loom_op_attrs(source_to_low)[loom_pass_run_options_ATTR_INDEX];
  ASSERT_EQ(options_attr.kind, LOOM_ATTR_DICT);
  loom_named_attr_slice_t option_attrs = loom_attr_as_dict(options_attr);
  ASSERT_EQ(option_attrs.count, 1u);
  EXPECT_TRUE(iree_string_view_equal(
      module.get()->strings.entries[option_attrs.entries[0].name_id],
      IREE_SV("max-errors")));
  EXPECT_EQ(loom_attr_as_i64(option_attrs.entries[0].value), 7);

  loom_op_t* source_low_cleanup_for = source_to_low->next_op;
  ASSERT_TRUE(loom_pass_for_isa(source_low_cleanup_for));
  loom_block_t* source_low_cleanup_body =
      loom_region_entry_block(loom_pass_for_body(source_low_cleanup_for));
  const iree_string_view_t cleanup_keys[] = {
      IREE_SV("canonicalize"),
      IREE_SV("cse"),
      IREE_SV("low-dce"),
  };
  ExpectRunKeySequence(module.get(), source_low_cleanup_body, cleanup_keys,
                       IREE_ARRAYSIZE(cleanup_keys));

  loom_op_t* low_for = source_low_cleanup_for->next_op;
  ASSERT_TRUE(loom_pass_for_isa(low_for));
  loom_block_t* low_body = loom_region_entry_block(loom_pass_for_body(low_for));
  const iree_string_view_t low_keys[] = {
      IREE_SV("mock-target-low-materialization"),
      IREE_SV("canonicalize"),
      IREE_SV("cse"),
      IREE_SV("mock-target-low-preparation"),
      IREE_SV("cse"),
      IREE_SV("low-select-operand-forms"),
      IREE_SV("low-dce"),
  };
  ExpectRunKeySequence(module.get(), low_body, low_keys,
                       IREE_ARRAYSIZE(low_keys));
  EXPECT_TRUE(loom_pass_yield_isa(pipeline_body->last_op));
}

}  // namespace
}  // namespace loom
