// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/linker.h"

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

class LinkerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_op_registry_initialize_context(iree_allocator_system(),
                                                       &context_));
  }

  void TearDown() override {
    for (loom_module_t* module : modules_) {
      loom_module_free(module);
    }
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* Parse(iree_string_view_t source,
                       iree_string_view_t filename = IREE_SV("test.loom")) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t parse_options = {
        .max_errors = 20,
    };
    IREE_EXPECT_OK(loom_text_parse(source, filename, &context_, &block_pool_,
                                   &parse_options, &module));
    EXPECT_NE(module, nullptr);
    if (module) {
      modules_.push_back(module);
    }
    return module;
  }

  loom_module_t* Link(std::initializer_list<loom_module_t*> source_modules) {
    loom_module_t* linked_module = nullptr;
    IREE_CHECK_OK(LinkStatus(source_modules, &linked_module));
    modules_.push_back(linked_module);
    return linked_module;
  }

  iree_status_t LinkStatus(std::initializer_list<loom_module_t*> source_modules,
                           loom_module_t** out_module) {
    std::vector<const loom_module_t*> inputs;
    inputs.reserve(source_modules.size());
    for (loom_module_t* module : source_modules) {
      inputs.push_back(module);
    }
    loom_link_options_t options = {
        .module_name = IREE_SV("linked"),
    };
    iree_status_t status = loom_link_materialized_modules(
        inputs.data(), inputs.size(), &options, &block_pool_,
        iree_allocator_system(), out_module);
    return status;
  }

  std::string Print(const loom_module_t* module) {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    IREE_EXPECT_OK(loom_text_print_module_to_builder(module, &builder,
                                                     LOOM_TEXT_PRINT_DEFAULT));
    std::string result(iree_string_builder_buffer(&builder),
                       iree_string_builder_size(&builder));
    iree_string_builder_deinitialize(&builder);
    return result;
  }

  void Verify(const loom_module_t* module) {
    loom_verify_options_t options = {
        .max_errors = 100,
    };
    loom_verify_result_t result = {};
    IREE_ASSERT_OK(loom_verify_module(module, &options, &result));
    EXPECT_EQ(result.error_count, 0u);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_ = {};
  std::vector<loom_module_t*> modules_;
};

TEST_F(LinkerTest, ResolvesForwardReferenceFromLaterCorpusModule) {
  loom_module_t* harness = Parse(IREE_SV(R"(
func.def @caller(%x: i32) -> (i32) {
  %y = func.call @identity(%x) : (i32) -> (i32)
  func.return %y : i32
}
)"));
  loom_module_t* corpus = Parse(IREE_SV(R"(
func.def @identity(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  loom_module_t* linked = Link({harness, corpus});
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_NE(text.find("func.def @caller"), std::string::npos);
  EXPECT_NE(text.find("func.call @identity"), std::string::npos);
  EXPECT_NE(text.find("func.def @identity"), std::string::npos);
}

TEST_F(LinkerTest, ConcreteDefinitionSupersedesDeclaration) {
  loom_module_t* harness = Parse(IREE_SV(R"(
func.decl @identity(%x: i32) -> (i32)

func.def @caller(%x: i32) -> (i32) {
  %y = func.call @identity(%x) : (i32) -> (i32)
  func.return %y : i32
}
)"));
  loom_module_t* corpus = Parse(IREE_SV(R"(
func.def @identity(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  loom_module_t* linked = Link({harness, corpus});
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_EQ(text.find("func.decl @identity"), std::string::npos);
  EXPECT_NE(text.find("func.def @identity"), std::string::npos);
}

TEST_F(LinkerTest, MergesDeclarationTargetContractIntoDefinition) {
  loom_module_t* harness = Parse(IREE_SV(R"(
target.profile @test_target preset("test-low")

func.decl target(@test_target) abi(object_function) export("identity_export") @identity(%x: i32) -> (i32)
)"));
  loom_module_t* corpus = Parse(IREE_SV(R"(
func.def @identity(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  loom_module_t* linked = Link({harness, corpus});
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_EQ(text.find("func.decl @identity"), std::string::npos);
  EXPECT_NE(text.find("func.def target(@test_target) abi(object_function) "
                      "export(\"identity_export\") @identity"),
            std::string::npos);
}

TEST_F(LinkerTest, MergesDeclarationPredicatesIntoDefinition) {
  loom_module_t* harness = Parse(IREE_SV(R"(
target.profile @test_target preset("test-low")

func.decl target(@test_target) @bounded(%m: index, %x: tensor<[%m]xf32>) -> (tensor<[%m]xf32>) where [mul(%m, 16)]
)"));
  loom_module_t* corpus = Parse(IREE_SV(R"(
func.def @bounded(%m: index, %x: tensor<[%m]xf32>) -> (tensor<[%m]xf32>) {
  func.return %x : tensor<[%m]xf32>
}
)"));

  loom_module_t* linked = Link({harness, corpus});
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_EQ(text.find("func.decl @bounded"), std::string::npos);
  EXPECT_NE(text.find("func.def target(@test_target) @bounded"),
            std::string::npos);
  EXPECT_NE(text.find("where [mul(%m, 16)]"), std::string::npos);
}

TEST_F(LinkerTest, RejectsDeclarationDefinitionTargetConflict) {
  loom_module_t* harness = Parse(IREE_SV(R"(
target.profile @decl_target preset("test-low")

func.decl target(@decl_target) @identity(%x: i32) -> (i32)
)"));
  loom_module_t* corpus = Parse(IREE_SV(R"(
target.profile @def_target preset("test-low")

func.def target(@def_target) @identity(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  loom_module_t* linked = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        LinkStatus({harness, corpus}, &linked));
  EXPECT_EQ(linked, nullptr);
}

TEST_F(LinkerTest, RejectsDeclarationDefinitionSignatureConflict) {
  loom_module_t* harness = Parse(IREE_SV(R"(
func.decl @identity(%x: i64) -> (i64)
)"));
  loom_module_t* corpus = Parse(IREE_SV(R"(
func.def @identity(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  loom_module_t* linked = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        LinkStatus({harness, corpus}, &linked));
  EXPECT_EQ(linked, nullptr);
}

TEST_F(LinkerTest, KeepsDeclarationWhenNoDefinitionExists) {
  loom_module_t* harness = Parse(IREE_SV(R"(
func.decl @external_identity(%m: index, %x: tensor<[%m]xf32>) -> (tensor<[%m]xf32>) where [mul(%m, 16)]
)"));

  loom_module_t* linked = Link({harness});
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_NE(text.find("func.decl @external_identity"), std::string::npos);
  EXPECT_NE(text.find("tensor<[%m]xf32"), std::string::npos);
  EXPECT_NE(text.find("where [mul(%m, 16)]"), std::string::npos);
}

TEST_F(LinkerTest, RejectsDuplicateConcreteDefinitions) {
  loom_module_t* first = Parse(IREE_SV(R"(
func.def @identity(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  loom_module_t* second = Parse(IREE_SV(R"(
func.def @identity(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  const loom_module_t* inputs[] = {first, second};
  loom_module_t* linked = nullptr;
  loom_link_options_t options = {
      .module_name = IREE_SV("linked"),
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        loom_link_materialized_modules(
                            inputs, IREE_ARRAYSIZE(inputs), &options,
                            &block_pool_, iree_allocator_system(), &linked));
  EXPECT_EQ(linked, nullptr);
}

}  // namespace
}  // namespace loom
