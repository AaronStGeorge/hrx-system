// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/verify.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/attribute.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/pass/ops.h"
#include "loom/pass/test/registry.h"

namespace loom {
namespace {

static bool allow_one_requirement(void* user_data,
                                  iree_string_view_t requirement) {
  const iree_string_view_t* allowed =
      static_cast<const iree_string_view_t*>(user_data);
  return iree_string_view_equal(*allowed, requirement);
}

class PassVerifyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &scratch_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);

    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = loom_pass_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_PASS,
                                                 vtables, (uint16_t)count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    for (loom_module_t* module : modules_) {
      loom_module_free(module);
    }
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* Parse(iree_string_view_t source) {
    loom_text_parse_options_t options = {
        .diagnostic_sink = {loom_diagnostic_stderr_sink, NULL},
        .max_errors = 20,
    };
    loom_module_t* module = NULL;
    IREE_CHECK_OK(loom_text_parse(source, IREE_SV("pass_verify.loom"),
                                  &context_, &block_pool_, &options, &module));
    if (!module) {
      ADD_FAILURE() << "parser produced no module";
      return nullptr;
    }
    modules_.push_back(module);
    return module;
  }

  iree_status_t Verify(
      iree_string_view_t source,
      loom_pass_requirement_provider_t requirement_provider = {}) {
    loom_module_t* module = Parse(source);
    if (!module) {
      return iree_make_status(IREE_STATUS_INTERNAL, "parse failed");
    }
    return VerifyModule(module, requirement_provider);
  }

  iree_status_t VerifyModule(
      const loom_module_t* module,
      loom_pass_requirement_provider_t requirement_provider = {}) {
    loom_pass_verify_options_t options = {
        .registry = loom_test_pass_registry(),
        .requirement_provider = requirement_provider,
    };
    return loom_pass_verify_module(module, &options, &scratch_arena_);
  }

  void ExpectVerifyStatus(
      iree_status_code_t expected_code, iree_string_view_t source,
      loom_pass_requirement_provider_t requirement_provider = {}) {
    IREE_EXPECT_STATUS_IS(expected_code, Verify(source, requirement_provider));
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t scratch_arena_;
  loom_context_t context_;
  std::vector<loom_module_t*> modules_;
};

TEST_F(PassVerifyTest, VerifiesModuleAndFuncPipelines) {
  IREE_ASSERT_OK(Verify(IREE_SV(
      "pass.pipeline<module> @cleanup pipeline {\n"
      "  test.module-noop\n"
      "  for func {\n"
      "    test.noop\n"
      "    repeat fixed(count = 2) {\n"
      "      test.options(count = 3, mode = beta, string = \"payload\")\n"
      "    }\n"
      "  }\n"
      "}\n"
      "\n"
      "pass.pipeline<func> @function_cleanup pipeline {\n"
      "  where name(value = \"matmul\") {\n"
      "    test.noop\n"
      "  }\n"
      "}\n"
      "\n"
      "pass.pipeline<module> @debug pipeline {\n"
      "  call @cleanup\n"
      "  fail \"expected cleanup to run\"\n"
      "  halt \"inspect IR\"\n"
      "}\n")));
}

TEST_F(PassVerifyTest, AllowsModulesWithoutPassPipelines) {
  IREE_ASSERT_OK(Verify(IREE_SV("")));
}

TEST_F(PassVerifyTest, RejectsNullPipelineOp) {
  loom_module_t* module = Parse(IREE_SV(""));
  ASSERT_NE(module, nullptr);
  loom_pass_verify_options_t options = {
      .registry = loom_test_pass_registry(),
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_verify_pipeline_op(module, NULL, &options, &scratch_arena_));
}

TEST_F(PassVerifyTest, VerifiesSatisfiedDescriptorRequirement) {
  iree_string_view_t allowed_requirement = IREE_SV("target.bundle");
  loom_pass_requirement_provider_t provider = {
      .callback = allow_one_requirement,
      .user_data = &allowed_requirement,
  };
  IREE_ASSERT_OK(Verify(IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                                "  test.requires-target\n"
                                "}\n"),
                        provider));
}

TEST_F(PassVerifyTest, RejectsUnknownPassKey) {
  ExpectVerifyStatus(IREE_STATUS_INVALID_ARGUMENT,
                     IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                             "  definitely-not-a-pass\n"
                             "}\n"));
}

TEST_F(PassVerifyTest, RejectsUnavailablePassKey) {
  ExpectVerifyStatus(IREE_STATUS_FAILED_PRECONDITION,
                     IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                             "  test.unavailable\n"
                             "}\n"));
}

TEST_F(PassVerifyTest, RejectsWrongAnchor) {
  ExpectVerifyStatus(IREE_STATUS_INVALID_ARGUMENT,
                     IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                             "  test.noop\n"
                             "}\n"));
}

TEST_F(PassVerifyTest, RejectsMalformedOptions) {
  ExpectVerifyStatus(IREE_STATUS_INVALID_ARGUMENT,
                     IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                             "  test.options(count = 99)\n"
                             "}\n"));
}

TEST_F(PassVerifyTest, RejectsUnknownOptions) {
  ExpectVerifyStatus(IREE_STATUS_INVALID_ARGUMENT,
                     IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                             "  test.options(unknown = 1)\n"
                             "}\n"));
}

TEST_F(PassVerifyTest, RejectsMissingRequiredOptions) {
  ExpectVerifyStatus(IREE_STATUS_INVALID_ARGUMENT,
                     IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                             "  test.required\n"
                             "}\n"));
}

TEST_F(PassVerifyTest, RejectsMissingDescriptorRequirementProvider) {
  ExpectVerifyStatus(IREE_STATUS_FAILED_PRECONDITION,
                     IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                             "  test.requires-target\n"
                             "}\n"));
}

TEST_F(PassVerifyTest, RejectsUnsatisfiedDescriptorRequirement) {
  iree_string_view_t allowed_requirement = IREE_SV("not-target.bundle");
  loom_pass_requirement_provider_t provider = {
      .callback = allow_one_requirement,
      .user_data = &allowed_requirement,
  };
  ExpectVerifyStatus(IREE_STATUS_FAILED_PRECONDITION,
                     IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                             "  test.requires-target\n"
                             "}\n"),
                     provider);
}

TEST_F(PassVerifyTest, RejectsNestedSameAnchorFor) {
  ExpectVerifyStatus(IREE_STATUS_INVALID_ARGUMENT,
                     IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                             "  for func {\n"
                             "    for func {\n"
                             "      test.noop\n"
                             "    }\n"
                             "  }\n"
                             "}\n"));
}

TEST_F(PassVerifyTest, RejectsEmptyWherePredicate) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                    "  where name() {\n"
                    "    test.noop\n"
                    "  }\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);
  loom_op_t* pipeline_op = loom_block_op(loom_region_entry_block(module->body),
                                         /*op_index=*/0);
  loom_block_t* pipeline_body =
      loom_region_entry_block(loom_pass_pipeline_body(pipeline_op));
  loom_op_t* where_op = loom_block_op(pipeline_body, /*op_index=*/0);
  ASSERT_TRUE(loom_pass_where_isa(where_op));
  loom_string_id_t empty_string_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module, iree_string_view_empty(),
                                           &empty_string_id));
  loom_op_attrs(where_op)[loom_pass_where_predicate_ATTR_INDEX] =
      loom_attr_string(empty_string_id);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, VerifyModule(module));
}

TEST_F(PassVerifyTest, RejectsMissingRegionTerminator) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                    "  test.noop\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);
  loom_op_t* pipeline_op = loom_block_op(loom_region_entry_block(module->body),
                                         /*op_index=*/0);
  loom_block_t* pipeline_body =
      loom_region_entry_block(loom_pass_pipeline_body(pipeline_op));
  ASSERT_TRUE(loom_pass_yield_isa(pipeline_body->last_op));
  pipeline_body->last_op = pipeline_body->first_op;
  pipeline_body->first_op->next_op = NULL;
  pipeline_body->op_count = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, VerifyModule(module));
}

TEST_F(PassVerifyTest, RejectsRepeatMissingCount) {
  ExpectVerifyStatus(IREE_STATUS_INVALID_ARGUMENT,
                     IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                             "  repeat fixed {\n"
                             "    test.noop\n"
                             "  }\n"
                             "}\n"));
}

TEST_F(PassVerifyTest, RejectsRepeatUntilConvergedWithCount) {
  ExpectVerifyStatus(IREE_STATUS_INVALID_ARGUMENT,
                     IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                             "  repeat until_converged(count = 2) {\n"
                             "    test.noop\n"
                             "  }\n"
                             "}\n"));
}

TEST_F(PassVerifyTest, RejectsUnresolvedCall) {
  ExpectVerifyStatus(IREE_STATUS_INVALID_ARGUMENT,
                     IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                             "  call @missing\n"
                             "}\n"));
}

TEST_F(PassVerifyTest, RejectsCallAnchorMismatch) {
  ExpectVerifyStatus(
      IREE_STATUS_INVALID_ARGUMENT,
      IREE_SV("pass.pipeline<func> @function_cleanup pipeline {\n"
              "  test.noop\n"
              "}\n"
              "\n"
              "pass.pipeline<module> @pipeline pipeline {\n"
              "  call @function_cleanup\n"
              "}\n"));
}

TEST_F(PassVerifyTest, RejectsCallCycle) {
  ExpectVerifyStatus(IREE_STATUS_INVALID_ARGUMENT,
                     IREE_SV("pass.pipeline<module> @a pipeline {\n"
                             "  call @b\n"
                             "}\n"
                             "\n"
                             "pass.pipeline<module> @b pipeline {\n"
                             "  call @a\n"
                             "}\n"));
}

}  // namespace
}  // namespace loom
