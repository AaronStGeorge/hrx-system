// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/provider.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/pass/builder.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

static iree_status_t ContributeMaterialization(
    const loom_target_pipeline_contribution_t* contribution) {
  if (contribution->phase !=
      LOOM_TARGET_PIPELINE_PHASE_TARGET_LOW_MATERIALIZATION) {
    return iree_ok_status();
  }
  loom_op_t* run_op = nullptr;
  return loom_pass_ir_build_run(contribution->builder,
                                IREE_SV("target-materialize"),
                                loom_named_attr_slice_empty(), &run_op);
}

static iree_status_t ContributePreparation(
    const loom_target_pipeline_contribution_t* contribution) {
  if (contribution->phase !=
      LOOM_TARGET_PIPELINE_PHASE_TARGET_LOW_PREPARATION) {
    return iree_ok_status();
  }
  loom_op_t* run_op = nullptr;
  return loom_pass_ir_build_run(contribution->builder,
                                IREE_SV("target-prepare"),
                                loom_named_attr_slice_empty(), &run_op);
}

struct PipelineBuildData {
  const loom_target_environment_t* environment;
};

static iree_status_t BuildContributedPipeline(loom_builder_t* builder,
                                              void* user_data) {
  const PipelineBuildData* data =
      static_cast<const PipelineBuildData*>(user_data);
  IREE_RETURN_IF_ERROR(loom_target_environment_contribute_pipeline(
      data->environment, LOOM_TARGET_PIPELINE_PHASE_TARGET_LOW_MATERIALIZATION,
      loom_pass_environment_empty(), builder));
  loom_op_t* run_op = nullptr;
  IREE_RETURN_IF_ERROR(
      loom_pass_ir_build_run(builder, IREE_SV("driver-cleanup"),
                             loom_named_attr_slice_empty(), &run_op));
  return loom_target_environment_contribute_pipeline(
      data->environment, LOOM_TARGET_PIPELINE_PHASE_TARGET_LOW_PREPARATION,
      loom_pass_environment_empty(), builder);
}

class TargetProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_op_registry_initialize_context(iree_allocator_system(),
                                                       &context_));
  }

  void TearDown() override {
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

  iree_string_view_t RunKey(loom_module_t* module, const loom_op_t* op) {
    return module->strings.entries[loom_pass_run_key(op)];
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(TargetProviderTest, ContributesPassIrByPhase) {
  static const loom_target_provider_t materialization_provider = {
      .contribute_pipeline = ContributeMaterialization,
  };
  static const loom_target_provider_t preparation_provider = {
      .contribute_pipeline = ContributePreparation,
  };
  static const loom_target_provider_t* const providers[] = {
      &materialization_provider,
      &preparation_provider,
  };
  const loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  loom_target_environment_t environment = {0};
  IREE_ASSERT_OK(
      loom_target_environment_initialize(&provider_set, &environment));

  ModulePtr module;
  IREE_ASSERT_OK(AllocateModule(IREE_SV("pipeline"), &module));
  PipelineBuildData build_data = {
      .environment = &environment,
  };
  loom_op_t* pipeline_op = nullptr;
  IREE_ASSERT_OK(loom_pass_ir_build_pipeline(
      module.get(), IREE_SV("compile"), LOOM_PASS_ANCHOR_MODULE,
      BuildContributedPipeline, &build_data, &pipeline_op));

  loom_block_t* pipeline_body =
      loom_region_entry_block(loom_pass_pipeline_body(pipeline_op));
  ASSERT_NE(pipeline_body, nullptr);
  ASSERT_EQ(pipeline_body->op_count, 4u);

  loom_op_t* materialize_run = pipeline_body->first_op;
  ASSERT_TRUE(loom_pass_run_isa(materialize_run));
  EXPECT_TRUE(iree_string_view_equal(RunKey(module.get(), materialize_run),
                                     IREE_SV("target-materialize")));

  loom_op_t* cleanup_run = materialize_run->next_op;
  ASSERT_TRUE(loom_pass_run_isa(cleanup_run));
  EXPECT_TRUE(iree_string_view_equal(RunKey(module.get(), cleanup_run),
                                     IREE_SV("driver-cleanup")));

  loom_op_t* prepare_run = cleanup_run->next_op;
  ASSERT_TRUE(loom_pass_run_isa(prepare_run));
  EXPECT_TRUE(iree_string_view_equal(RunKey(module.get(), prepare_run),
                                     IREE_SV("target-prepare")));
  EXPECT_TRUE(loom_pass_yield_isa(pipeline_body->last_op));

  loom_target_environment_deinitialize(&environment);
}

}  // namespace
}  // namespace loom
