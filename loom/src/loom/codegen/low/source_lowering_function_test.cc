// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/lower.h"
#include "loom/codegen/low/source_selection.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/lower.h"

namespace loom {
namespace {

struct CollectedEmission {
  const loom_error_def_t* error = nullptr;
  const loom_op_t* op = nullptr;
  std::vector<std::string> string_params;
};

struct ModuleDeleter {
  void operator()(loom_module_t* module) const { loom_module_free(module); }
};

using ModulePtr = std::unique_ptr<loom_module_t, ModuleDeleter>;

struct EmissionCollector {
  std::vector<CollectedEmission> emissions;

  iree_diagnostic_emitter_t emitter() {
    return iree_diagnostic_emitter_t{
        .fn = Collect,
        .user_data = this,
    };
  }

 private:
  static std::string CopyString(iree_string_view_t value) {
    return std::string(value.data, value.size);
  }

  static iree_status_t Collect(void* user_data,
                               const loom_diagnostic_emission_t* emission) {
    auto* collector = static_cast<EmissionCollector*>(user_data);
    CollectedEmission entry;
    entry.error = emission->error;
    entry.op = emission->op;
    for (iree_host_size_t i = 0; i < emission->param_count; ++i) {
      const loom_diagnostic_param_t* param = &emission->params[i];
      if (param->kind == LOOM_PARAM_STRING) {
        entry.string_params.push_back(CopyString(param->string));
      }
    }
    collector->emissions.push_back(std::move(entry));
    return iree_ok_status();
  }
};

static iree_status_t MakeRegisterType(loom_low_lower_context_t* context,
                                      iree_string_view_t register_class,
                                      uint32_t unit_count,
                                      loom_type_t* out_type) {
  loom_string_id_t register_class_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(loom_low_lower_context_module(context),
                                register_class, &register_class_id));
  *out_type = loom_type_register(register_class_id, unit_count);
  return iree_ok_status();
}

static iree_status_t TestEmitPreamble(void* user_data,
                                      loom_low_lower_context_t* context) {
  (void)user_data;
  loom_string_id_t source_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(loom_low_lower_context_module(context),
                                IREE_SV("test.thread_id"), &source_id));
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      MakeRegisterType(context, IREE_SV("test.i32"), 1, &result_type));
  loom_op_t* live_in_op = nullptr;
  return loom_low_live_in_build(
      loom_low_lower_context_builder(context), source_id,
      loom_make_named_attr_slice(NULL, 0), result_type,
      loom_low_lower_context_low_function(context)->location, &live_in_op);
}

static const loom_low_lower_rule_set_t* const kTestLowerRuleSets[] = {
    &loom_test_low_lower_rule_set,
};

static const loom_low_lower_policy_t kTestPreambleLowerPolicy = {
    .name = IREE_SVL("test-preamble-lower-policy"),
    .map_type = {.fn = loom_test_low_lower_map_type, .user_data = nullptr},
    .map_argument = {.fn = loom_test_low_lower_map_argument,
                     .user_data = nullptr},
    .emit_preamble = {.fn = TestEmitPreamble, .user_data = nullptr},
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kTestLowerRuleSets),
            .values = kTestLowerRuleSets,
        },
};

static loom_low_lower_policy_registry_t MakeTestPreamblePolicyRegistry() {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("test.low.core"),
          .policy = &kTestPreambleLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_t registry = {};
  loom_low_lower_policy_registry_initialize_from_entries(
      &registry, kEntries, IREE_ARRAYSIZE(kEntries));
  return registry;
}

class SourceLoweringFunctionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_CFG, loom_cfg_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_BUFFER, loom_buffer_dialect_vtables);
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
                                  IREE_SV("source_lowering_function_test.loom"),
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
        .flags = LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY,
        .descriptor_registry = &registry_.registry,
        .emitter = collector->emitter(),
        .max_errors = 20,
    };
    return loom_low_verify_module(module, &options, out_result);
  }

  iree_status_t PrintModule(const loom_module_t* module,
                            std::string* out_text) {
    out_text->clear();
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    iree_status_t status = loom_text_print_module_to_builder(
        module, &builder, LOOM_TEXT_PRINT_DEFAULT);
    if (iree_status_is_ok(status)) {
      *out_text = std::string(iree_string_builder_buffer(&builder),
                              iree_string_builder_size(&builder));
    }
    iree_string_builder_deinitialize(&builder);
    return status;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t registry_ = {};
  loom_low_lower_policy_registry_t policy_registry_ = {};
};

TEST_F(SourceLoweringFunctionTest, LowersScalarFunction) {
  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  ModulePtr module = ParseAndLowerTargetedSource(
      "target.profile @test_target preset(\"test-low\")\n"
      "func.def target(@test_target) @add_const(%lhs: i32) -> (i32) {\n"
      "  %c7 = scalar.constant 7 : i32\n"
      "  %sum = scalar.addi %lhs, %c7 : i32\n"
      "  func.return %sum : i32\n"
      "}\n",
      &lower_collector, &lower_result);

  EXPECT_EQ(lower_result.error_count, 0u);
  EXPECT_EQ(lower_result.remark_count, 0u);
  EXPECT_NE(lower_result.descriptor_set, nullptr);
  EXPECT_NE(lower_result.low_func_op, nullptr);
  EXPECT_TRUE(loom_symbol_ref_is_valid(lower_result.low_func_ref));
  EXPECT_TRUE(lower_collector.emissions.empty());

  EmissionCollector verify_collector;
  loom_low_verify_result_t verify_result = {};
  IREE_ASSERT_OK(
      VerifyLowModule(module.get(), &verify_collector, &verify_result));
  EXPECT_EQ(verify_result.error_count, 0u);
  EXPECT_TRUE(verify_collector.emissions.empty());

  std::string text;
  IREE_ASSERT_OK(PrintModule(module.get(), &text));
  EXPECT_NE(text.find("low.func.def target(@test_target) "
                      "abi(object_function)"),
            std::string::npos);
  EXPECT_NE(text.find("@add_const"), std::string::npos);
  EXPECT_EQ(text.find("\nfunc.def target(@test_target) @add_const"),
            std::string::npos);
  EXPECT_EQ(text.find("source(@"), std::string::npos);
  EXPECT_NE(text.find("test.const.i32"), std::string::npos);
  EXPECT_NE(text.find("test.add.i32"), std::string::npos);
  EXPECT_EQ(text.find("low.abi"), std::string::npos);
}

TEST_F(SourceLoweringFunctionTest, CarriesTargetProfileAbiAndExport) {
  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  ModulePtr module = ParseAndLowerTargetedSource(
      "target.profile @test_target preset(\"test-low\")\n"
      "func.def target(@test_target) abi(vm_module_function) "
      "export(\"add_const_export\") @add_const(%lhs: i32) -> (i32) {\n"
      "  %c7 = scalar.constant 7 : i32\n"
      "  %sum = scalar.addi %lhs, %c7 : i32\n"
      "  func.return %sum : i32\n"
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

  std::string text;
  IREE_ASSERT_OK(PrintModule(module.get(), &text));
  EXPECT_NE(text.find("target.profile @test_target preset(\"test-low\")"),
            std::string::npos);
  EXPECT_NE(text.find("low.func.def target(@test_target) "
                      "abi(vm_module_function) export(\"add_const_export\")"),
            std::string::npos);
  EXPECT_NE(text.find("@add_const"), std::string::npos);
  EXPECT_EQ(text.find("\nfunc.def target(@test_target) "
                      "abi(vm_module_function) export(\"add_const_export\") "
                      "@add_const"),
            std::string::npos);
  EXPECT_NE(text.find("test.const.i32"), std::string::npos);
  EXPECT_NE(text.find("test.add.i32"), std::string::npos);
  EXPECT_EQ(text.find("target.bundle"), std::string::npos);
}

TEST_F(SourceLoweringFunctionTest, LowersVectorCfgFunction) {
  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  ModulePtr module = ParseAndLowerTargetedSource(
      "target.profile @test_target preset(\"test-low\")\n"
      "func.def target(@test_target) @select_vector(%cond: i1, %a: "
      "vector<4xi32>, "
      "%b: vector<4xi32>) -> (vector<4xi32>) {\n"
      "  cfg.cond_br %cond, ^then, ^else : i1\n"
      "^then:\n"
      "  %sum = vector.addi %a, %b : vector<4xi32>\n"
      "  cfg.br ^join(%sum: vector<4xi32>)\n"
      "^else:\n"
      "  %sum2 = vector.addi %b, %a : vector<4xi32>\n"
      "  cfg.br ^join(%sum2: vector<4xi32>)\n"
      "^join(%result: vector<4xi32>):\n"
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

  std::string text;
  IREE_ASSERT_OK(PrintModule(module.get(), &text));
  EXPECT_NE(text.find("@select_vector"), std::string::npos);
  EXPECT_EQ(text.find("\nfunc.def target(@test_target) @select_vector"),
            std::string::npos);
  EXPECT_EQ(text.find("source(@"), std::string::npos);
  EXPECT_NE(text.find("reg<test.i32 x4>"), std::string::npos);
  EXPECT_NE(text.find("low.slice"), std::string::npos);
  EXPECT_NE(text.find("low.concat"), std::string::npos);
  EXPECT_NE(text.find("test.add.i32"), std::string::npos);
  EXPECT_NE(text.find("low.cond_br"), std::string::npos);
  EXPECT_NE(text.find("low.br"), std::string::npos);
  EXPECT_EQ(text.find("low.abi"), std::string::npos);
}

TEST_F(SourceLoweringFunctionTest, LowersResourceArgumentsAsImports) {
  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  ModulePtr module = ParseAndLowerTargetedSource(
      "target.profile @test_target preset(\"test-low\")\n"
      "func.def target(@test_target) @resource_entry(%buffer: buffer) {\n"
      "  func.return\n"
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

  std::string text;
  IREE_ASSERT_OK(PrintModule(module.get(), &text));
  EXPECT_NE(text.find("low.func.def target(@test_target) "
                      "abi(object_function)"),
            std::string::npos);
  EXPECT_NE(text.find("@resource_entry()"), std::string::npos);
  EXPECT_EQ(text.find("\nfunc.def target(@test_target) @resource_entry"),
            std::string::npos);
  EXPECT_NE(text.find("%buffer = low.resource<native_pointer>"),
            std::string::npos);
  EXPECT_NE(text.find("semantic_type = buffer"), std::string::npos);
  EXPECT_EQ(text.find("source_arg"), std::string::npos);
  EXPECT_EQ(text.find("low.abi"), std::string::npos);
  EXPECT_EQ(text.find("low.abi.operand"), std::string::npos);
}

TEST_F(SourceLoweringFunctionTest, EmitsPreambleBeforeResourceImports) {
  policy_registry_ = MakeTestPreamblePolicyRegistry();
  IREE_ASSERT_OK(loom_low_lower_policy_registry_verify(&policy_registry_));

  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  ModulePtr module = ParseAndLowerTargetedSource(
      "target.profile @test_target preset(\"test-low\")\n"
      "func.def target(@test_target) @resource_entry(%buffer: buffer) {\n"
      "  func.return\n"
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

  std::string text;
  IREE_ASSERT_OK(PrintModule(module.get(), &text));
  const size_t live_in_position = text.find("low.live_in<test.thread_id>");
  const size_t resource_position = text.find("low.resource");
  ASSERT_NE(live_in_position, std::string::npos);
  ASSERT_NE(resource_position, std::string::npos);
  EXPECT_LT(live_in_position, resource_position);
}

}  // namespace
}  // namespace loom
