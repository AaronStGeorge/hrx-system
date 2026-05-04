// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/artifact_plan.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/error/error_catalog.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/target/facts.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/types.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/testing/module_ptr.h"
#include "loom/util/call_graph.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class ArtifactPlanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_symbol_fact_table_initialize(&fact_table_, &analysis_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("artifact_plan_test.loom"), &context_,
                                  &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  loom_symbol_id_t FindSymbol(const loom_module_t* module,
                              iree_string_view_t name) {
    loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    return symbol_id;
  }

  loom_symbol_ref_t SymbolRef(const loom_module_t* module,
                              iree_string_view_t name) {
    return (loom_symbol_ref_t){
        .module_id = 0,
        .symbol_id = FindSymbol(module, name),
    };
  }

  iree_status_t BuildArtifactPlan(const loom_module_t* module,
                                  iree_string_view_t artifact_name,
                                  testing::DiagnosticEmissionCapture* capture,
                                  bool* out_valid,
                                  loom_target_artifact_plan_t* out_plan) {
    loom_call_graph_t call_graph;
    IREE_RETURN_IF_ERROR(
        loom_call_graph_build(module, &analysis_arena_, &call_graph));
    iree_diagnostic_emitter_t diagnostic_emitter =
        capture ? capture->emitter() : iree_diagnostic_emitter_t{};
    return loom_target_artifact_plan_build(
        module, SymbolRef(module, artifact_name), &fact_table_, &call_graph,
        diagnostic_emitter, &analysis_arena_, out_valid, out_plan);
  }

  void ExpectArtifactPlanError(const loom_module_t* module,
                               const loom_error_def_t* error) {
    testing::DiagnosticEmissionCapture capture;
    loom_target_artifact_plan_t plan;
    bool valid = true;
    IREE_ASSERT_OK(
        BuildArtifactPlan(module, IREE_SV("module"), &capture, &valid, &plan));
    EXPECT_FALSE(valid);
    ASSERT_EQ(capture.emissions.size(), 1u);
    EXPECT_EQ(capture.emissions[0].error, error);
  }

  // Block pool shared by parser, module allocation, and analysis storage.
  iree_arena_block_pool_t block_pool_;

  // Context with the target and function dialects registered.
  loom_context_t context_;

  // Arena for symbol facts, call graph, and artifact plan storage.
  iree_arena_allocator_t analysis_arena_;

  // Dense symbol fact table under test.
  loom_symbol_fact_table_t fact_table_;
};

TEST_F(ArtifactPlanTest, CollectsExportedEntryAndPrivateClosure) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @test_target {contract_feature_bits = 1}
target.artifact @module target(@test_target)

func.def target(@test_target) abi(object_function) export("entry", {artifact = @module}) @entry() {
  func.call @helper() : ()
  func.return
}

func.def target(@test_target) abi(object_function) @helper() {
  func.return
}

func.def target(@test_target) abi(object_function) @unused() {
  func.return
}
)");

  loom_target_artifact_plan_t plan;
  bool valid = false;
  IREE_ASSERT_OK(BuildArtifactPlan(module.get(), IREE_SV("module"), nullptr,
                                   &valid, &plan));
  ASSERT_TRUE(valid);

  EXPECT_EQ(plan.entry_count, 1u);
  EXPECT_EQ(plan.entry_symbol_ids[0],
            FindSymbol(module.get(), IREE_SV("entry")));
  EXPECT_EQ(plan.func_count, 2u);
  EXPECT_EQ(plan.func_symbol_ids[0],
            FindSymbol(module.get(), IREE_SV("entry")));
  EXPECT_EQ(plan.func_symbol_ids[1],
            FindSymbol(module.get(), IREE_SV("helper")));
}

TEST_F(ArtifactPlanTest, OrdersEntriesByDenseExportOrdinal) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @test_target {contract_feature_bits = 1}
target.artifact @module target(@test_target)

func.def target(@test_target) abi(object_function) export("second", {artifact = @module, ordinal = 1}) @second() {
  func.return
}

func.def target(@test_target) abi(object_function) export("first", {artifact = @module, ordinal = 0}) @first() {
  func.return
}
)");

  loom_target_artifact_plan_t plan;
  bool valid = false;
  IREE_ASSERT_OK(BuildArtifactPlan(module.get(), IREE_SV("module"), nullptr,
                                   &valid, &plan));
  ASSERT_TRUE(valid);

  ASSERT_EQ(plan.entry_count, 2u);
  EXPECT_EQ(plan.entry_symbol_ids[0],
            FindSymbol(module.get(), IREE_SV("first")));
  EXPECT_EQ(plan.entry_symbol_ids[1],
            FindSymbol(module.get(), IREE_SV("second")));
}

TEST_F(ArtifactPlanTest, RejectsMixedExportOrdinalPolicy) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @test_target {contract_feature_bits = 1}
target.artifact @module target(@test_target)

func.def target(@test_target) abi(object_function) export("first", {artifact = @module, ordinal = 0}) @first() {
  func.return
}

func.def target(@test_target) abi(object_function) export("second", {artifact = @module}) @second() {
  func.return
}
)");

  ExpectArtifactPlanError(module.get(), LOOM_ERR_TARGET_018);
}

TEST_F(ArtifactPlanTest, RejectsEntryTargetMismatchingArtifactTarget) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @test_target {contract_feature_bits = 1}
test.target<low_core> @other {contract_feature_bits = 1}
target.artifact @module target(@test_target)

func.def target(@other) abi(object_function) export("entry", {artifact = @module}) @entry() {
  func.return
}
)");

  ExpectArtifactPlanError(module.get(), LOOM_ERR_TARGET_017);
}

TEST_F(ArtifactPlanTest, RejectsClosureCrossingIntoAnotherArtifact) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @test_target {contract_feature_bits = 1}
target.artifact @module target(@test_target)
target.artifact @other_module target(@test_target)

func.def target(@test_target) abi(object_function) export("entry", {artifact = @module}) @entry() {
  func.call @other() : ()
  func.return
}

func.def target(@test_target) abi(object_function) export("other", {artifact = @other_module}) @other() {
  func.return
}
)");

  ExpectArtifactPlanError(module.get(), LOOM_ERR_TARGET_015);
}

TEST_F(ArtifactPlanTest, AllowsExternalDeclarationCalls) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @test_target {contract_feature_bits = 1}
target.artifact @module target(@test_target)

func.def target(@test_target) abi(object_function) export("entry", {artifact = @module}) @entry() {
  func.call @external() : ()
  func.return
}

func.decl import("env") @external()
)");

  loom_target_artifact_plan_t plan;
  bool valid = false;
  IREE_ASSERT_OK(BuildArtifactPlan(module.get(), IREE_SV("module"), nullptr,
                                   &valid, &plan));
  ASSERT_TRUE(valid);

  EXPECT_EQ(plan.entry_count, 1u);
  EXPECT_EQ(plan.func_count, 1u);
  EXPECT_EQ(plan.func_symbol_ids[0],
            FindSymbol(module.get(), IREE_SV("entry")));
}

}  // namespace
}  // namespace loom
