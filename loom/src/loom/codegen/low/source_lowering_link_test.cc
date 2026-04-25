// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/lower.h"
#include "loom/codegen/low/source_selection.h"
#include "loom/codegen/low/testing/ir_match_test_util.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/link/linker.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/lower.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using EmissionCollector = ::loom::testing::DiagnosticEmissionCapture;
using ModulePtr = ::loom::testing::ModulePtr;

class SourceLoweringLinkTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_test_low_descriptor_registry_initialize(&registry_);
    loom_test_low_lower_policy_registry_initialize(&policy_registry_);
    IREE_ASSERT_OK(loom_low_lower_policy_registry_verify(&policy_registry_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseSource(const char* source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("source_lowering_link_test.loom"),
                                  &context_, &block_pool_, &parse_options,
                                  &module));
    IREE_ASSERT(module != nullptr);
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

  ModulePtr LinkModules(std::initializer_list<loom_module_t*> source_modules) {
    std::vector<const loom_module_t*> inputs;
    inputs.reserve(source_modules.size());
    for (loom_module_t* module : source_modules) {
      inputs.push_back(module);
    }
    loom_link_options_t options = {
        .module_name = IREE_SV("linked"),
    };
    loom_module_t* linked_module = nullptr;
    IREE_CHECK_OK(loom_link_materialized_modules(
        inputs.data(), inputs.size(), &options, &block_pool_,
        iree_allocator_system(), &linked_module));
    return ModulePtr(linked_module);
  }

  void LowerTargetedSource(loom_module_t* module, EmissionCollector* collector,
                           loom_low_lower_result_t* out_result) {
    iree_arena_allocator_t selection_arena;
    iree_arena_initialize(module->arena.block_pool, &selection_arena);
    const loom_low_source_selection_options_t selection_options = {
        .descriptor_registry = &registry_.registry,
        .policy_registry = &policy_registry_,
    };
    loom_low_source_selection_t selection = {};
    IREE_CHECK_OK(loom_low_select_source_func(module, &selection_options,
                                              &selection_arena, &selection));
    const loom_low_lower_options_t options = {
        .target_ref = selection.target_ref,
        .bundle = selection.target_bundle,
        .descriptor_registry = &registry_.registry,
        .descriptor_requirements =
            LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
        .policy = selection.policy,
        .emitter = collector->emitter(),
        .max_errors = 20,
    };
    IREE_CHECK_OK(
        loom_low_lower_function(module, selection.func, &options, out_result));
    iree_arena_deinitialize(&selection_arena);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t registry_ = {};
  loom_low_lower_policy_registry_t policy_registry_ = {};
};

TEST_F(SourceLoweringLinkTest,
       LowersLinkedTargetNeutralDefinitionWithDeclarationContract) {
  ModulePtr harness = ParseSource(
      "target.profile @test_target preset(\"test-low\")\n"
      "\n"
      "func.decl target(@test_target) @add(%lhs: i32, %rhs: i32) -> (i32)\n");
  ModulePtr corpus = ParseSource(
      "func.def @add(%lhs: i32, %rhs: i32) -> (i32) {\n"
      "  %sum = scalar.addi %lhs, %rhs : i32\n"
      "  func.return %sum : i32\n"
      "}\n");
  ModulePtr linked = LinkModules({harness.get(), corpus.get()});

  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  LowerTargetedSource(linked.get(), &lower_collector, &lower_result);

  EXPECT_EQ(lower_result.error_count, 0u);
  EXPECT_EQ(lower_result.remark_count, 0u);
  EXPECT_NE(lower_result.low_func_op, nullptr);
  EXPECT_TRUE(lower_collector.emissions.empty());

  EXPECT_EQ(
      loom::testing::FindModuleSymbolDefiningOp(linked.get(), IREE_SV("add")),
      lower_result.low_func_op);
  EXPECT_TRUE(loom_low_func_def_isa(lower_result.low_func_op));
  EXPECT_EQ(loom_low_func_def_abi(lower_result.low_func_op),
            LOOM_LOW_ABI_OBJECT_FUNCTION);
}

}  // namespace
}  // namespace loom
