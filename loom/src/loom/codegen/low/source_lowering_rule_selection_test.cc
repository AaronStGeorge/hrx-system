// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/lower.h"
#include "loom/codegen/low/lower_rules.h"
#include "loom/codegen/low/source_selection.h"
#include "loom/codegen/low/testing/ir_match_test_util.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/target/test/descriptors.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/lower.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

enum TestRejectedScalarAddValueRef : uint16_t {
  kTestRejectedScalarAddOperand0,
  kTestRejectedScalarAddOperand1,
  kTestRejectedScalarAddResult0,
};

static const loom_low_lower_value_ref_t kTestRejectedScalarAddValueRefs[] = {
    {
        .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
        .index = 0,
    },
    {
        .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
        .index = 1,
    },
    {
        .kind = LOOM_LOW_LOWER_VALUE_REF_RESULT,
        .index = 0,
    },
};

static const loom_low_lower_type_pattern_t kTestRejectedScalarAddTypes[] = {
    {
        .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                 LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT |
                 LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_RANK |
                 LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0_RANGE,
        .type_kind = LOOM_TYPE_VECTOR,
        .element_type_mask =
            LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_I32),
        .rank = 1,
        .static_dim0_min = 4,
        .static_dim0_max = 4,
    },
};

static const loom_low_lower_diagnostic_t kTestRejectedScalarAddDiagnostics[] = {
    {
        .subject_kind = IREE_SVL("type"),
        .subject_name = IREE_SVL("vector<4xi32>"),
        .reason =
            IREE_SVL("test lowering requires vector<4xi32> vector values"),
    },
};

static const loom_low_lower_guard_t kTestRejectedScalarAddGuards[] = {
    {
        .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
        .value_ref_index = kTestRejectedScalarAddOperand0,
        .type_pattern_index = 0,
        .diagnostic_index = 0,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
        .value_ref_index = kTestRejectedScalarAddOperand1,
        .type_pattern_index = 0,
        .diagnostic_index = 0,
    },
    {
        .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
        .value_ref_index = kTestRejectedScalarAddResult0,
        .type_pattern_index = 0,
        .diagnostic_index = 0,
    },
};

static const loom_low_lower_emit_t kTestRejectedScalarAddEmits[] = {
    {
        .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE,
        .descriptor_id = TEST_LOW_CORE_DESCRIPTOR_ID_TEST_ADD_I32,
        .operand_ref_start = kTestRejectedScalarAddOperand0,
        .operand_ref_count = 2,
        .result_ref_start = kTestRejectedScalarAddResult0,
        .result_ref_count = 1,
    },
};

static const loom_low_lower_rule_t kTestRejectedScalarAddRules[] = {
    {
        .source_op_kind = LOOM_OP_SCALAR_ADDI,
        .guard_start = 0,
        .guard_count = 3,
        .emit_start = 0,
        .emit_count = 1,
    },
};

static const loom_low_lower_rule_span_t kTestRejectedScalarAddSpans[] = {
    {
        .source_op_kind = LOOM_OP_SCALAR_ADDI,
        .rule_start = 0,
        .rule_count = 1,
    },
};

static const loom_low_lower_rule_set_t kTestRejectedScalarAddRuleSet = {
    .spans = kTestRejectedScalarAddSpans,
    .span_count = IREE_ARRAYSIZE(kTestRejectedScalarAddSpans),
    .rules = kTestRejectedScalarAddRules,
    .rule_count = IREE_ARRAYSIZE(kTestRejectedScalarAddRules),
    .type_patterns = kTestRejectedScalarAddTypes,
    .type_pattern_count = IREE_ARRAYSIZE(kTestRejectedScalarAddTypes),
    .value_refs = kTestRejectedScalarAddValueRefs,
    .value_ref_count = IREE_ARRAYSIZE(kTestRejectedScalarAddValueRefs),
    .guards = kTestRejectedScalarAddGuards,
    .guard_count = IREE_ARRAYSIZE(kTestRejectedScalarAddGuards),
    .emits = kTestRejectedScalarAddEmits,
    .emit_count = IREE_ARRAYSIZE(kTestRejectedScalarAddEmits),
    .diagnostics = kTestRejectedScalarAddDiagnostics,
    .diagnostic_count = IREE_ARRAYSIZE(kTestRejectedScalarAddDiagnostics),
};

static const loom_low_lower_rule_set_t* const kTestComposedRuleSets[] = {
    &kTestRejectedScalarAddRuleSet,
    &loom_test_low_lower_rule_set,
};

static const loom_low_lower_policy_t kTestComposedLowerPolicy = {
    .name = IREE_SVL("test-composed-lower-policy"),
    .map_type = {.fn = loom_test_low_lower_map_type, .user_data = nullptr},
    .map_contract_value = {.fn = loom_test_low_lower_map_contract_value,
                           .user_data = nullptr},
    .map_argument = {.fn = loom_test_low_lower_map_argument,
                     .user_data = nullptr},
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kTestComposedRuleSets),
            .values = kTestComposedRuleSets,
        },
};

static loom_low_lower_policy_registry_t MakeTestComposedPolicyRegistry() {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("test.low.core"),
          .policy = &kTestComposedLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_t registry = {};
  loom_low_lower_policy_registry_initialize_from_entries(
      &registry, kEntries, IREE_ARRAYSIZE(kEntries));
  return registry;
}

class SourceLoweringRuleSelectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables,
                    loom_target_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables,
                    loom_func_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables,
                    loom_scalar_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables,
                    loom_low_dialect_op_semantics);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_test_low_descriptor_registry_initialize(&registry_);
    policy_registry_ = MakeTestComposedPolicyRegistry();
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
    IREE_CHECK_OK(
        loom_text_parse(iree_make_cstring_view(source),
                        IREE_SV("source_lowering_rule_selection.loom"),
                        &context_, &block_pool_, &parse_options, &module));
    IREE_ASSERT(module != nullptr);
    return ModulePtr(module);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);
  using DialectSemanticsFn = const loom_op_semantics_t* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id, DialectVtablesFn dialect_vtables_fn,
                       DialectSemanticsFn dialect_semantics_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    iree_host_size_t semantics_count = 0;
    const loom_op_semantics_t* semantics =
        dialect_semantics_fn(&semantics_count);
    ASSERT_EQ(semantics_count, count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
    IREE_ASSERT_OK(loom_context_register_dialect_semantics(
        &context_, dialect_id, semantics, (uint16_t)semantics_count));
  }

  void LowerTargetedSource(loom_module_t* module,
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
    loom_pass_value_fact_owner_t value_facts = {};
    loom_pass_value_fact_owner_initialize(module->arena.block_pool,
                                          &value_facts);
    loom_value_fact_table_t* fact_table = nullptr;
    IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
        &value_facts, module,
        loom_pass_value_fact_scope_function_for_target(selection.func,
                                                       selection.target_bundle),
        &fact_table));
    const loom_low_lower_options_t options = {
        .target_ref = selection.target_ref,
        .bundle = selection.target_bundle,
        .descriptor_registry = &registry_.registry,
        .descriptor_requirements =
            LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
        .policy = selection.policy,
        .fact_table = fact_table,
        .max_errors = 20,
    };
    IREE_CHECK_OK(
        loom_low_lower_function(module, selection.func, &options, out_result));
    loom_pass_value_fact_owner_deinitialize(&value_facts);
    iree_arena_deinitialize(&selection_arena);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t registry_ = {};
  loom_low_lower_policy_registry_t policy_registry_ = {};
};

TEST_F(SourceLoweringRuleSelectionTest,
       ContinuesAfterRejectedRuleSetCandidate) {
  loom_low_lower_result_t lower_result = {};
  ModulePtr module = ParseSource(
      "target.profile @test_target preset(\"test-low\")\n"
      "func.def target(@test_target) @add(%lhs: i32, %rhs: i32) -> (i32) {\n"
      "  %sum = scalar.addi %lhs, %rhs : i32\n"
      "  func.return %sum : i32\n"
      "}\n");
  LowerTargetedSource(module.get(), &lower_result);

  EXPECT_EQ(lower_result.error_count, 0u);
  EXPECT_EQ(lower_result.remark_count, 0u);
  EXPECT_NE(lower_result.low_func_op, nullptr);

  EXPECT_EQ(
      loom::testing::FindModuleSymbolDefiningOp(module.get(), IREE_SV("add")),
      lower_result.low_func_op);
  EXPECT_TRUE(loom_low_func_def_isa(lower_result.low_func_op));
  EXPECT_NE(
      loom::testing::FindLowFuncDescriptorOp(
          lower_result.low_func_op, TEST_LOW_CORE_DESCRIPTOR_ID_TEST_ADD_I32),
      nullptr);
}

}  // namespace
}  // namespace loom
