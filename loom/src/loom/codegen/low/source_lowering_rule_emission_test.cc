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
#include "loom/codegen/low/verify.h"
#include "loom/error/error_defs.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/target/test/descriptors.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/lower.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using CollectedEmission = ::loom::testing::CapturedDiagnosticEmission;
using EmissionCollector = ::loom::testing::DiagnosticEmissionCapture;
using ModulePtr = ::loom::testing::ModulePtr;

static bool IsI32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
}

static const loom_low_lower_rule_set_t* const kTestLowerRuleSets[] = {
    &loom_test_low_lower_rule_set,
};

struct TestCallbackPlan {
  // Descriptor row emitted by the callback-selected low op.
  loom_low_lower_resolved_descriptor_t descriptor;
};

static iree_status_t TestSelectCallbackOp(void* user_data,
                                          loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_low_lower_plan_t* out_plan) {
  (void)user_data;
  *out_plan = loom_low_lower_plan_empty();
  if (source_op->kind != LOOM_OP_SCALAR_DIVSI) {
    return iree_ok_status();
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!IsI32(
          loom_module_value_type(module, loom_scalar_divsi_lhs(source_op))) ||
      !IsI32(
          loom_module_value_type(module, loom_scalar_divsi_rhs(source_op))) ||
      !IsI32(loom_module_value_type(module,
                                    loom_scalar_divsi_result(source_op)))) {
    return iree_ok_status();
  }
  TestCallbackPlan* plan_data = nullptr;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*plan_data), (void**)&plan_data));
  IREE_RETURN_IF_ERROR(loom_low_lower_resolve_descriptor(
      context, TEST_LOW_CORE_DESCRIPTOR_ID_TEST_ADD_I32,
      &plan_data->descriptor));
  *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);
  return iree_ok_status();
}

static iree_status_t TestEmitCallbackOp(void* user_data,
                                        loom_low_lower_context_t* context,
                                        const loom_op_t* source_op,
                                        loom_low_lower_plan_t plan) {
  (void)user_data;
  IREE_ASSERT_EQ(plan.id, LOOM_OP_SCALAR_DIVSI);
  const auto* plan_data =
      static_cast<const TestCallbackPlan*>(plan.target_data);
  IREE_ASSERT_NE(plan_data, nullptr);
  loom_value_id_t operands[2] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_scalar_divsi_lhs(source_op), &operands[0]));
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_scalar_divsi_rhs(source_op), &operands[1]));
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
      context, source_op, loom_scalar_divsi_result(source_op), &result_type));
  loom_op_t* low_op = nullptr;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan_data->descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &result_type, 1, nullptr, 0,
      source_op->location, &low_op));
  return loom_low_lower_bind_value(
      context, loom_scalar_divsi_result(source_op),
      loom_value_slice_get(loom_low_op_results(low_op), 0));
}

static const loom_low_lower_policy_t kTestHybridLowerPolicy = {
    .name = IREE_SVL("test-hybrid-lower-policy"),
    .map_type = {.fn = loom_test_low_lower_map_type, .user_data = nullptr},
    .map_contract_value = {.fn = loom_test_low_lower_map_contract_value,
                           .user_data = nullptr},
    .map_argument = {.fn = loom_test_low_lower_map_argument,
                     .user_data = nullptr},
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kTestLowerRuleSets),
            .values = kTestLowerRuleSets,
        },
    .select_op = {.fn = TestSelectCallbackOp, .user_data = nullptr},
    .emit_op = {.fn = TestEmitCallbackOp, .user_data = nullptr},
};

static loom_low_lower_policy_registry_t MakeTestHybridPolicyRegistry() {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("test.low.core"),
          .policy = &kTestHybridLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_t registry = {};
  loom_low_lower_policy_registry_initialize_from_entries(
      &registry, kEntries, IREE_ARRAYSIZE(kEntries));
  return registry;
}

class SourceLoweringRuleEmissionTest : public ::testing::Test {
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
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables,
                    loom_index_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables,
                    loom_vector_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables,
                    loom_low_dialect_op_semantics);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_test_low_descriptor_registry_initialize(&registry_);
    loom_test_low_lower_policy_registry_initialize(&policy_registry_);
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
                                  IREE_SV("source_lowering_rule_test.loom"),
                                  &context_, &block_pool_, &parse_options,
                                  &module));
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
        .policy = selection.policy,
        .fact_table = fact_table,
        .emitter = collector->emitter(),
        .max_errors = 20,
    };
    IREE_CHECK_OK(
        loom_low_lower_function(module, selection.func, &options, out_result));
    loom_pass_value_fact_owner_deinitialize(&value_facts);
    iree_arena_deinitialize(&selection_arena);
  }

  ModulePtr ParseAndLowerTargetedSource(const char* source,
                                        EmissionCollector* collector,
                                        loom_low_lower_result_t* out_result) {
    ModulePtr module = ParseSource(source);
    LowerTargetedSource(module.get(), collector, out_result);
    return module;
  }

  iree_status_t VerifyLowModule(loom_module_t* module,
                                EmissionCollector* collector,
                                loom_low_verify_result_t* out_result) {
    const loom_low_verify_options_t options = {
        .descriptor_registry = &registry_.registry,
        .emitter = collector->emitter(),
        .max_errors = 20,
    };
    return loom_low_verify_module(module, &options, out_result);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t registry_ = {};
  loom_low_lower_policy_registry_t policy_registry_ = {};
};

TEST_F(SourceLoweringRuleEmissionTest, CopiesTiedOperandsBeforeDestructiveOp) {
  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  ModulePtr module = ParseAndLowerTargetedSource(
      "target.profile @test_target preset(\"test-low\")\n"
      "func.def target(@test_target) @destructive(%lhs: i32, %rhs: i32) -> "
      "(i32) {\n"
      "  %result = scalar.subi %lhs, %rhs : i32\n"
      "  func.return %result : i32\n"
      "}\n",
      &lower_collector, &lower_result);

  EXPECT_EQ(lower_result.error_count, 0u);
  EXPECT_EQ(lower_result.remark_count, 0u);
  EXPECT_NE(lower_result.low_func_op, nullptr);
  EXPECT_TRUE(lower_collector.emissions.empty());

  EmissionCollector verify_collector;
  loom_low_verify_result_t verify_result = {};
  IREE_ASSERT_OK(
      VerifyLowModule(module.get(), &verify_collector, &verify_result));
  EXPECT_EQ(verify_result.error_count, 0u);
  EXPECT_TRUE(verify_collector.emissions.empty());

  const loom_block_t* entry_block = loom_region_const_entry_block(
      loom_low_func_def_body(lower_result.low_func_op));
  const loom_op_t* copy_op = nullptr;
  const loom_op_t* tied_op = nullptr;
  loom_op_t* op = nullptr;
  loom_block_for_each_op(entry_block, op) {
    if (copy_op == nullptr && loom_low_copy_isa(op)) {
      copy_op = op;
    }
    if (loom_low_op_isa(op) && (uint64_t)loom_low_op_descriptor_id(op) ==
                                   TEST_LOW_CORE_DESCRIPTOR_ID_TEST_TIED_ANY) {
      tied_op = op;
      break;
    }
  }
  ASSERT_NE(copy_op, nullptr);
  ASSERT_NE(tied_op, nullptr);
  EXPECT_LT(copy_op->block_ordinal, tied_op->block_ordinal);
  EXPECT_EQ(tied_op->tied_result_count, 1u);
  const loom_tied_result_t* ties = loom_op_tied_results(tied_op);
  EXPECT_EQ(ties[0].result_index, 0u);
  EXPECT_EQ(ties[0].operand_index, 0u);
}

TEST_F(SourceLoweringRuleEmissionTest, CarriesTemporaryBetweenDescriptorOps) {
  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  ModulePtr module = ParseAndLowerTargetedSource(
      "target.profile @test_target preset(\"test-low\")\n"
      "func.def target(@test_target) @madd(%a: index, %b: index, %c: index) "
      "-> (index) {\n"
      "  %result = index.madd %a, %b, %c : index\n"
      "  func.return %result : index\n"
      "}\n",
      &lower_collector, &lower_result);

  EXPECT_EQ(lower_result.error_count, 0u);
  EXPECT_EQ(lower_result.remark_count, 0u);
  EXPECT_NE(lower_result.low_func_op, nullptr);
  EXPECT_TRUE(lower_collector.emissions.empty());

  EmissionCollector verify_collector;
  loom_low_verify_result_t verify_result = {};
  IREE_ASSERT_OK(
      VerifyLowModule(module.get(), &verify_collector, &verify_result));
  EXPECT_EQ(verify_result.error_count, 0u);
  EXPECT_TRUE(verify_collector.emissions.empty());

  loom_region_t* low_body = loom_low_func_def_body(lower_result.low_func_op);
  ASSERT_NE(low_body, nullptr);
  ASSERT_GT(low_body->block_count, 0u);
  const loom_block_t* entry_block = loom_region_const_entry_block(low_body);
  ASSERT_GE(entry_block->arg_count, 3u);

  const loom_op_t* first_add = nullptr;
  const loom_op_t* second_add = nullptr;
  loom_op_t* op = nullptr;
  loom_block_for_each_op(entry_block, op) {
    if (!loom_low_op_isa(op) || loom_low_op_descriptor_id(op) !=
                                    TEST_LOW_CORE_DESCRIPTOR_ID_TEST_ADD_I32) {
      continue;
    }
    if (first_add == nullptr) {
      first_add = op;
    } else {
      second_add = op;
      break;
    }
  }
  ASSERT_NE(first_add, nullptr);
  ASSERT_NE(second_add, nullptr);

  loom_value_id_t temporary_result =
      loom_value_slice_get(loom_low_op_results(first_add), 0);
  loom_value_slice_t second_operands = loom_low_op_operands(second_add);
  ASSERT_EQ(second_operands.count, 2u);
  EXPECT_EQ(second_operands.values[0], temporary_result);
  const loom_op_t* materialized_operand = loom_value_def_op(
      loom_module_value(module.get(), second_operands.values[1]));
  ASSERT_TRUE(loom_low_copy_isa(materialized_operand));
  EXPECT_EQ(loom_low_copy_source(materialized_operand),
            entry_block->arg_ids[2]);
}

TEST_F(SourceLoweringRuleEmissionTest, SwapsPerLaneOperands) {
  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  ModulePtr module = ParseAndLowerTargetedSource(
      "target.profile @test_target preset(\"test-low\")\n"
      "func.def target(@test_target) @sub_vector(%lhs: vector<4xi32>, "
      "%rhs: vector<4xi32>) -> (vector<4xi32>) {\n"
      "  %result = vector.subi %lhs, %rhs : vector<4xi32>\n"
      "  func.return %result : vector<4xi32>\n"
      "}\n",
      &lower_collector, &lower_result);

  EXPECT_EQ(lower_result.error_count, 0u);
  EXPECT_NE(lower_result.low_func_op, nullptr);
  EXPECT_TRUE(lower_collector.emissions.empty());

  EmissionCollector verify_collector;
  loom_low_verify_result_t verify_result = {};
  IREE_ASSERT_OK(
      VerifyLowModule(module.get(), &verify_collector, &verify_result));
  EXPECT_EQ(verify_result.error_count, 0u);
  EXPECT_TRUE(verify_collector.emissions.empty());

  loom_region_t* low_body = loom_low_func_def_body(lower_result.low_func_op);
  ASSERT_NE(low_body, nullptr);
  ASSERT_GT(low_body->block_count, 0u);
  const loom_block_t* entry_block = loom_region_const_entry_block(low_body);
  ASSERT_GE(entry_block->arg_count, 2u);
  const loom_value_id_t low_lhs = entry_block->arg_ids[0];
  const loom_value_id_t low_rhs = entry_block->arg_ids[1];

  const loom_op_t* first_lane_op = nullptr;
  loom_op_t* op = nullptr;
  loom_block_for_each_op(entry_block, op) {
    if (!loom_low_op_isa(op) || loom_low_op_descriptor_id(op) !=
                                    TEST_LOW_CORE_DESCRIPTOR_ID_TEST_ADD_I32) {
      continue;
    }
    first_lane_op = op;
    break;
  }
  ASSERT_NE(first_lane_op, nullptr);

  loom_value_slice_t operands = loom_low_op_operands(first_lane_op);
  ASSERT_EQ(operands.count, 2u);
  const loom_op_t* first_operand_def =
      loom_value_def_op(loom_module_value(module.get(), operands.values[0]));
  const loom_op_t* second_operand_def =
      loom_value_def_op(loom_module_value(module.get(), operands.values[1]));
  ASSERT_TRUE(loom_low_slice_isa(first_operand_def));
  ASSERT_TRUE(loom_low_slice_isa(second_operand_def));
  EXPECT_EQ(loom_low_slice_source(first_operand_def), low_rhs);
  EXPECT_EQ(loom_low_slice_source(second_operand_def), low_lhs);
}

TEST_F(SourceLoweringRuleEmissionTest,
       ProjectsI64ArrayElementIntoDescriptorImmediate) {
  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  ModulePtr module = ParseAndLowerTargetedSource(
      "target.profile @test_target preset(\"test-low\")\n"
      "func.def target(@test_target) @extract_lane(%v: vector<4xi32>) -> "
      "(i32) {\n"
      "  %lane = vector.extract %v[2] : vector<4xi32> -> i32\n"
      "  func.return %lane : i32\n"
      "}\n",
      &lower_collector, &lower_result);

  EXPECT_EQ(lower_result.error_count, 0u);
  EXPECT_EQ(lower_result.remark_count, 0u);
  EXPECT_NE(lower_result.low_func_op, nullptr);
  EXPECT_TRUE(lower_collector.emissions.empty());

  EmissionCollector verify_collector;
  loom_low_verify_result_t verify_result = {};
  IREE_ASSERT_OK(
      VerifyLowModule(module.get(), &verify_collector, &verify_result));
  EXPECT_EQ(verify_result.error_count, 0u);
  EXPECT_TRUE(verify_collector.emissions.empty());

  const loom_op_t* const_op = loom::testing::FindLowFuncDescriptorOp(
      lower_result.low_func_op, TEST_LOW_CORE_DESCRIPTOR_ID_TEST_CONST_I32);
  ASSERT_NE(const_op, nullptr);
  ASSERT_TRUE(loom_low_const_isa(const_op));
  loom_named_attr_slice_t attrs = loom_low_const_attrs(const_op);
  ASSERT_EQ(attrs.count, 1u);
  EXPECT_TRUE(loom::testing::ModuleStringEquals(
      module.get(), attrs.entries[0].name_id, IREE_SV("i32_value")));
  ASSERT_EQ(attrs.entries[0].value.kind, LOOM_ATTR_I64);
  EXPECT_EQ(attrs.entries[0].value.i64, 2);
}

TEST_F(SourceLoweringRuleEmissionTest,
       RejectsDynamicLaneProjectionWithRuleDiagnostic) {
  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  (void)ParseAndLowerTargetedSource(
      "target.profile @test_target preset(\"test-low\")\n"
      "func.def target(@test_target) @extract_lane(%v: vector<4xi32>, "
      "%lane_index: index) -> (i32) {\n"
      "  %lane = vector.extract %v[%lane_index] : vector<4xi32> -> i32\n"
      "  func.return %lane : i32\n"
      "}\n",
      &lower_collector, &lower_result);

  EXPECT_EQ(lower_result.error_count, 1u);
  EXPECT_EQ(lower_result.remark_count, 0u);
  EXPECT_EQ(lower_result.low_func_op, nullptr);
  ASSERT_EQ(lower_collector.emissions.size(), 1u);
  const CollectedEmission& emission = lower_collector.emissions[0];
  ASSERT_EQ(emission.string_params.size(), 7u);
  EXPECT_EQ(emission.string_params[4], "attr");
  EXPECT_EQ(emission.string_params[5], "static_indices");
  EXPECT_EQ(emission.string_params[6],
            "test lowering requires one static lane in [0, 3]");
}

TEST_F(SourceLoweringRuleEmissionTest,
       PacksI64ArrayElementsIntoDescriptorImmediate) {
  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  ModulePtr module = ParseAndLowerTargetedSource(
      "target.profile @test_target preset(\"test-low\")\n"
      "func.def target(@test_target) @shuffle(%v: vector<4xi32>) -> "
      "(vector<4xi32>) {\n"
      "  %shuffled = vector.shuffle<[3, 2, 1, 0]> %v : vector<4xi32>\n"
      "  func.return %shuffled : vector<4xi32>\n"
      "}\n",
      &lower_collector, &lower_result);

  EXPECT_EQ(lower_result.error_count, 0u);
  EXPECT_EQ(lower_result.remark_count, 0u);
  EXPECT_NE(lower_result.low_func_op, nullptr);
  EXPECT_TRUE(lower_collector.emissions.empty());

  EmissionCollector verify_collector;
  loom_low_verify_result_t verify_result = {};
  IREE_ASSERT_OK(
      VerifyLowModule(module.get(), &verify_collector, &verify_result));
  EXPECT_EQ(verify_result.error_count, 0u);
  EXPECT_TRUE(verify_collector.emissions.empty());

  const loom_op_t* shuffle_op = loom::testing::FindLowFuncDescriptorOp(
      lower_result.low_func_op, TEST_LOW_CORE_DESCRIPTOR_ID_TEST_SHUFFLE_V4I32);
  ASSERT_NE(shuffle_op, nullptr);
  ASSERT_TRUE(loom_low_op_isa(shuffle_op));
  loom_named_attr_slice_t attrs = loom_low_op_attrs(shuffle_op);
  ASSERT_EQ(attrs.count, 1u);
  EXPECT_TRUE(loom::testing::ModuleStringEquals(
      module.get(), attrs.entries[0].name_id, IREE_SV("shuffle_control")));
  ASSERT_EQ(attrs.entries[0].value.kind, LOOM_ATTR_I64);
  EXPECT_EQ(attrs.entries[0].value.i64, 27);
}

TEST_F(SourceLoweringRuleEmissionTest,
       RejectsPackedI64ArrayElementOutOfRangeWithRuleDiagnostic) {
  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  (void)ParseAndLowerTargetedSource(
      "target.profile @test_target preset(\"test-low\")\n"
      "func.def target(@test_target) @shuffle(%v: vector<4xi32>) -> "
      "(vector<4xi32>) {\n"
      "  %shuffled = vector.shuffle<[3, 2, 1, 4]> %v : vector<4xi32>\n"
      "  func.return %shuffled : vector<4xi32>\n"
      "}\n",
      &lower_collector, &lower_result);

  EXPECT_EQ(lower_result.error_count, 1u);
  EXPECT_EQ(lower_result.remark_count, 0u);
  EXPECT_EQ(lower_result.low_func_op, nullptr);
  ASSERT_EQ(lower_collector.emissions.size(), 1u);
  const CollectedEmission& emission = lower_collector.emissions[0];
  ASSERT_EQ(emission.string_params.size(), 7u);
  EXPECT_EQ(emission.string_params[4], "attr");
  EXPECT_EQ(emission.string_params[5], "source_lanes");
  EXPECT_EQ(emission.string_params[6],
            "test lowering requires four shuffle lanes in [0, 3]");
}

TEST_F(SourceLoweringRuleEmissionTest, HybridPolicyUsesCallbackForUncoveredOp) {
  policy_registry_ = MakeTestHybridPolicyRegistry();

  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  ModulePtr module = ParseAndLowerTargetedSource(
      "target.profile @test_target preset(\"test-low\")\n"
      "func.def target(@test_target) @div(%lhs: i32, %rhs: i32) -> (i32) {\n"
      "  %quotient = scalar.divsi %lhs, %rhs : i32\n"
      "  func.return %quotient : i32\n"
      "}\n",
      &lower_collector, &lower_result);

  EXPECT_EQ(lower_result.error_count, 0u);
  EXPECT_EQ(lower_result.remark_count, 0u);
  EXPECT_NE(lower_result.low_func_op, nullptr);
  EXPECT_TRUE(lower_collector.emissions.empty());

  EmissionCollector verify_collector;
  loom_low_verify_result_t verify_result = {};
  IREE_ASSERT_OK(
      VerifyLowModule(module.get(), &verify_collector, &verify_result));
  EXPECT_EQ(verify_result.error_count, 0u);
  EXPECT_TRUE(verify_collector.emissions.empty());

  EXPECT_EQ(
      loom::testing::FindModuleSymbolDefiningOp(module.get(), IREE_SV("div")),
      lower_result.low_func_op);
  EXPECT_TRUE(loom_low_func_def_isa(lower_result.low_func_op));
  EXPECT_NE(
      loom::testing::FindLowFuncDescriptorOp(
          lower_result.low_func_op, TEST_LOW_CORE_DESCRIPTOR_ID_TEST_ADD_I32),
      nullptr);
}

TEST_F(SourceLoweringRuleEmissionTest,
       HybridPolicyKeepsRuleDiagnosticsForCoveredOps) {
  policy_registry_ = MakeTestHybridPolicyRegistry();

  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  (void)ParseAndLowerTargetedSource(
      "target.profile @test_target preset(\"test-low\")\n"
      "func.def target(@test_target) @bad_const() -> (i32) {\n"
      "  %c = scalar.constant 1.0 : i32\n"
      "  func.return %c : i32\n"
      "}\n",
      &lower_collector, &lower_result);

  EXPECT_EQ(lower_result.error_count, 1u);
  EXPECT_EQ(lower_result.remark_count, 0u);
  EXPECT_EQ(lower_result.low_func_op, nullptr);
  ASSERT_EQ(lower_collector.emissions.size(), 1u);
  const CollectedEmission& emission = lower_collector.emissions[0];
  ASSERT_EQ(emission.string_params.size(), 7u);
  EXPECT_EQ(emission.string_params[4], "attr");
  EXPECT_EQ(emission.string_params[5], "value");
  EXPECT_EQ(emission.string_params[6],
            "test constant lowering requires an i64 value");
}

TEST_F(SourceLoweringRuleEmissionTest,
       UnsupportedSourceOpEmitsDiagnosticAndNoLowFunction) {
  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  ModulePtr module = ParseAndLowerTargetedSource(
      "target.profile @test_target preset(\"test-low\")\n"
      "func.def target(@test_target) @div(%lhs: i32, %rhs: i32) -> (i32) {\n"
      "  %quotient = scalar.divsi %lhs, %rhs : i32\n"
      "  func.return %quotient : i32\n"
      "}\n",
      &lower_collector, &lower_result);

  EXPECT_EQ(lower_result.error_count, 1u);
  EXPECT_EQ(lower_result.remark_count, 0u);
  EXPECT_EQ(lower_result.low_func_op, nullptr);
  EXPECT_FALSE(loom_symbol_ref_is_valid(lower_result.low_func_ref));
  ASSERT_EQ(lower_collector.emissions.size(), 1u);
  const CollectedEmission& emission = lower_collector.emissions[0];
  EXPECT_EQ(emission.error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 1));
  ASSERT_EQ(emission.string_params.size(), 7u);
  EXPECT_EQ(emission.string_params[0], "test_target");
  EXPECT_EQ(emission.string_params[3], "div");
  EXPECT_EQ(emission.string_params[4], "op");
  EXPECT_EQ(emission.string_params[5], "scalar.divsi");
  EXPECT_NE(emission.string_params[6].find("descriptor mapping"),
            std::string::npos);

  const loom_op_t* div_def =
      loom::testing::FindModuleSymbolDefiningOp(module.get(), IREE_SV("div"));
  ASSERT_NE(div_def, nullptr);
  EXPECT_TRUE(loom_func_def_isa(div_def));
}

}  // namespace
}  // namespace loom
