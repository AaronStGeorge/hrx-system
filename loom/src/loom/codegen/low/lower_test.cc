// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower.h"

#include <memory>
#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/lower_rules.h"
#include "loom/codegen/low/source_selection.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/codegen/low/verify.h"
#include "loom/error/error_defs.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/link/linker.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/test/descriptors.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/lower.h"
#include "loom/testing/context.h"

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

static bool IsI32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
}

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

static const loom_low_lower_rule_set_t* const kTestLowerRuleSets[] = {
    &loom_test_low_lower_rule_set,
};

static const loom_low_lower_rule_set_t* const kTestComposedRuleSets[] = {
    &kTestRejectedScalarAddRuleSet,
    &loom_test_low_lower_rule_set,
};

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

struct TestCallbackPlan {
  // Descriptor emitted by the callback-selected low op.
  uint64_t descriptor_id;
};

static iree_status_t TestSelectCallbackOp(void* user_data,
                                          loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_low_lower_plan_t* out_plan) {
  (void)user_data;
  *out_plan = loom_low_lower_plan_empty();
  if (source_op->kind != LOOM_OP_SCALAR_MULI) {
    return iree_ok_status();
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!IsI32(loom_module_value_type(module, loom_scalar_muli_lhs(source_op))) ||
      !IsI32(loom_module_value_type(module, loom_scalar_muli_rhs(source_op))) ||
      !IsI32(
          loom_module_value_type(module, loom_scalar_muli_result(source_op)))) {
    return iree_ok_status();
  }
  TestCallbackPlan* plan_data = nullptr;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*plan_data), (void**)&plan_data));
  plan_data->descriptor_id = TEST_LOW_CORE_DESCRIPTOR_ID_TEST_ADD_I32;
  *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);
  return iree_ok_status();
}

static iree_status_t TestEmitCallbackOp(void* user_data,
                                        loom_low_lower_context_t* context,
                                        const loom_op_t* source_op,
                                        loom_low_lower_plan_t plan) {
  (void)user_data;
  IREE_ASSERT_EQ(plan.id, LOOM_OP_SCALAR_MULI);
  const auto* plan_data =
      static_cast<const TestCallbackPlan*>(plan.target_data);
  IREE_ASSERT_NE(plan_data, nullptr);
  loom_value_id_t operands[2] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_scalar_muli_lhs(source_op), &operands[0]));
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_scalar_muli_rhs(source_op), &operands[1]));
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
      context, source_op, loom_scalar_muli_result(source_op), &result_type));
  loom_op_t* low_op = nullptr;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_descriptor_op(
      context, plan_data->descriptor_id, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &result_type, 1, nullptr, 0,
      source_op->location, &low_op));
  return loom_low_lower_bind_value(
      context, loom_scalar_muli_result(source_op),
      loom_value_slice_get(loom_low_op_results(low_op), 0));
}

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

static const loom_low_lower_policy_t kTestHybridLowerPolicy = {
    .name = IREE_SVL("test-hybrid-lower-policy"),
    .map_type = {.fn = loom_test_low_lower_map_type, .user_data = nullptr},
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

static const loom_low_lower_policy_t kTestComposedLowerPolicy = {
    .name = IREE_SVL("test-composed-lower-policy"),
    .map_type = {.fn = loom_test_low_lower_map_type, .user_data = nullptr},
    .map_argument = {.fn = loom_test_low_lower_map_argument,
                     .user_data = nullptr},
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kTestComposedRuleSets),
            .values = kTestComposedRuleSets,
        },
};

static loom_low_lower_policy_registry_t MakeTestPolicyRegistry() {
  loom_low_lower_policy_registry_t registry = {};
  loom_test_low_lower_policy_registry_initialize(&registry);
  return registry;
}

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

TEST(LowLowerPolicyRegistryTest, LooksUpPolicyByContractKey) {
  loom_low_lower_policy_registry_t registry = MakeTestPolicyRegistry();
  IREE_ASSERT_OK(loom_low_lower_policy_registry_verify(&registry));

  const loom_low_lower_policy_t* policy = nullptr;
  IREE_ASSERT_OK(loom_low_lower_policy_registry_lookup(
      &registry, IREE_SV("test.low.core"), &policy));
  EXPECT_EQ(policy, loom_test_low_lower_policy());
}

TEST(LowLowerPolicyRegistryTest, LooksUpPolicyForTargetBundle) {
  loom_low_lower_policy_registry_t registry = MakeTestPolicyRegistry();
  loom_target_config_t config = {};
  config.contract_set_key = IREE_SV("test.low.core");
  loom_target_bundle_t bundle = {};
  bundle.name = IREE_SV("test-low");
  bundle.config = &config;

  const loom_low_lower_policy_t* policy = nullptr;
  IREE_ASSERT_OK(loom_low_lower_policy_registry_lookup_for_bundle(
      &registry, &bundle, &policy));
  EXPECT_EQ(policy, loom_test_low_lower_policy());
}

TEST(LowLowerPolicyRegistryTest, RejectsMissingContractKey) {
  loom_low_lower_policy_registry_t registry = MakeTestPolicyRegistry();

  const loom_low_lower_policy_t* policy = nullptr;
  iree_status_t status = loom_low_lower_policy_registry_lookup(
      &registry, IREE_SV("missing-target"), &policy);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
  EXPECT_EQ(policy, nullptr);
}

TEST(LowLowerPolicyRegistryTest, RejectsMalformedRegistries) {
  loom_low_lower_policy_registry_t missing_entries = {};
  missing_entries.entry_count = 1;
  iree_status_t status =
      loom_low_lower_policy_registry_verify(&missing_entries);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);

  const loom_low_lower_policy_registry_entry_t empty_key_entries[] = {
      {
          .contract_set_key = IREE_SVL(""),
          .policy = loom_test_low_lower_policy(),
      },
  };
  loom_low_lower_policy_registry_t empty_key_registry = {};
  loom_low_lower_policy_registry_initialize_from_entries(
      &empty_key_registry, empty_key_entries,
      IREE_ARRAYSIZE(empty_key_entries));
  status = loom_low_lower_policy_registry_verify(&empty_key_registry);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);

  const loom_low_lower_policy_t incomplete_policy = {
      .name = IREE_SVL("incomplete"),
  };
  const loom_low_lower_policy_registry_entry_t incomplete_entries[] = {
      {
          .contract_set_key = IREE_SVL("test.low.core"),
          .policy = &incomplete_policy,
      },
  };
  loom_low_lower_policy_registry_t incomplete_registry = {};
  loom_low_lower_policy_registry_initialize_from_entries(
      &incomplete_registry, incomplete_entries,
      IREE_ARRAYSIZE(incomplete_entries));
  status = loom_low_lower_policy_registry_verify(&incomplete_registry);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowLowerPolicyRegistryTest, RejectsDuplicateContractKeys) {
  const loom_low_lower_policy_registry_entry_t entries[] = {
      {
          .contract_set_key = IREE_SVL("test.low.core"),
          .policy = loom_test_low_lower_policy(),
      },
      {
          .contract_set_key = IREE_SVL("test.low.core"),
          .policy = loom_test_low_lower_policy(),
      },
  };
  loom_low_lower_policy_registry_t registry = {};
  loom_low_lower_policy_registry_initialize_from_entries(
      &registry, entries, IREE_ARRAYSIZE(entries));

  iree_status_t status = loom_low_lower_policy_registry_verify(&registry);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS, status);

  const loom_low_lower_policy_t* policy = nullptr;
  status = loom_low_lower_policy_registry_lookup(
      &registry, IREE_SV("test.low.core"), &policy);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS, status);
  EXPECT_EQ(policy, nullptr);
}

class LowLowerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_testing_context_initialize_all(iree_allocator_system(),
                                                       &context_));
    loom_test_low_descriptor_registry_initialize(&registry_);
    policy_registry_ = MakeTestPolicyRegistry();
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
                                  IREE_SV("low_lower_test.loom"), &context_,
                                  &block_pool_, &parse_options, &module));
    IREE_ASSERT(module != nullptr);
    return ModulePtr(module);
  }

  ModulePtr ParseSource(const std::string& source,
                        const loom_text_low_asm_environment_t* environment) {
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;
    if (environment != nullptr) {
      parse_options.low_asm_environment = *environment;
    }
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(
        loom_text_parse(iree_make_string_view(source.data(), source.size()),
                        IREE_SV("low_lower_test_roundtrip.loom"), &context_,
                        &block_pool_, &parse_options, &module));
    IREE_ASSERT(module != nullptr);
    return ModulePtr(module);
  }

  loom_func_like_t FirstFunction(loom_module_t* module) {
    loom_op_t* op = nullptr;
    loom_block_for_each_op(loom_module_block(module), op) {
      loom_func_like_t function = loom_func_like_cast(module, op);
      if (loom_func_like_isa(function)) {
        return function;
      }
    }
    ADD_FAILURE() << "expected module to contain a function";
    return (loom_func_like_t){0};
  }

  const loom_op_t* FirstBodyOpOfKind(loom_module_t* module,
                                     loom_op_kind_t kind) {
    loom_func_like_t function = FirstFunction(module);
    loom_region_t* body = loom_func_like_body(function);
    if (body == nullptr) {
      ADD_FAILURE() << "expected function to have a body";
      return nullptr;
    }
    const loom_block_t* entry_block = loom_region_const_entry_block(body);
    loom_op_t* op = nullptr;
    loom_block_for_each_op(entry_block, op) {
      if (op->kind == kind) {
        return op;
      }
    }
    ADD_FAILURE() << "expected function body to contain op kind " << kind;
    return nullptr;
  }

  loom_symbol_ref_t SymbolRef(loom_module_t* module,
                              iree_string_view_t symbol_name) {
    loom_string_id_t symbol_name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(
        loom_module_intern_string(module, symbol_name, &symbol_name_id));
    uint16_t symbol_id = loom_module_find_symbol(module, symbol_name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    return loom_symbol_ref_t{.module_id = 0, .symbol_id = symbol_id};
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

  ModulePtr ParseAndLowerTargetedSource(const char* source,
                                        EmissionCollector* collector,
                                        loom_low_lower_result_t* out_result) {
    ModulePtr module = ParseSource(source);
    LowerTargetedSource(module.get(), collector, out_result);
    return module;
  }

  ModulePtr ParseAndLowerProfileTarget(const char* source,
                                       EmissionCollector* collector,
                                       loom_low_lower_result_t* out_result) {
    return ParseAndLowerTargetedSource(source, collector, out_result);
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

  loom_text_low_asm_environment_t LowAsmEnvironment() {
    loom_text_low_asm_environment_t environment = {};
    loom_low_descriptor_text_asm_environment_initialize(&registry_.registry,
                                                        &environment);
    return environment;
  }

  iree_status_t PrintModule(const loom_module_t* module,
                            const loom_text_print_options_t* print_options,
                            std::string* out_text) {
    out_text->clear();
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    iree_status_t status = loom_text_print_module_to_builder_with_options(
        module, &builder, print_options);
    if (iree_status_is_ok(status)) {
      *out_text = std::string(iree_string_builder_buffer(&builder),
                              iree_string_builder_size(&builder));
    }
    iree_string_builder_deinitialize(&builder);
    return status;
  }

  iree_status_t WriteModule(const loom_module_t* module,
                            std::vector<uint8_t>* out_bytes) {
    iree_io_stream_t* stream = nullptr;
    IREE_RETURN_IF_ERROR(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    iree_status_t status =
        loom_bytecode_write_module(module, stream, nullptr, &block_pool_);
    if (iree_status_is_ok(status)) {
      iree_io_stream_pos_t length = iree_io_stream_length(stream);
      out_bytes->resize((size_t)length);
      status = iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0);
    }
    if (iree_status_is_ok(status)) {
      status = iree_io_stream_read(stream, out_bytes->size(), out_bytes->data(),
                                   nullptr);
    }
    iree_io_stream_release(stream);
    return status;
  }

  iree_status_t ReadModule(const std::vector<uint8_t>& bytes,
                           loom_module_t** out_module) {
    const loom_bytecode_read_options_t options = {
        .verify_module = true,
        .verify_max_errors = 20,
    };
    loom_bytecode_read_result_t result = {};
    IREE_RETURN_IF_ERROR(loom_bytecode_read_module(
        iree_make_const_byte_span(bytes.data(), bytes.size()),
        IREE_SV("low_lower_test.loombc"), &context_, &block_pool_, &options,
        &result, out_module, iree_allocator_system()));
    if (result.error_count != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "bytecode read emitted %u errors",
                              result.error_count);
    }
    return iree_ok_status();
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t registry_ = {};
  loom_low_lower_policy_registry_t policy_registry_ = {};
};

TEST_F(LowLowerTest, RuleSelectionWithMatchContextSelectsVectorRule) {
  ModulePtr module = ParseSource(
      "func.def @add_vectors(%lhs: vector<4xi32>, %rhs: vector<4xi32>) -> "
      "(vector<4xi32>) {\n"
      "  %sum = vector.addi %lhs, %rhs : vector<4xi32>\n"
      "  func.return %sum : vector<4xi32>\n"
      "}\n");
  const loom_op_t* source_op =
      FirstBodyOpOfKind(module.get(), LOOM_OP_VECTOR_ADDI);
  ASSERT_NE(source_op, nullptr);

  const loom_low_lower_rule_match_context_t match_context = {
      .module = module.get(),
      .descriptor_set = loom_test_low_core_descriptor_set(),
      .map_value =
          {
              .fn = loom_test_low_lower_rule_match_map_value,
              .user_data = nullptr,
          },
  };
  loom_low_lower_rule_selection_t selection = {};
  IREE_ASSERT_OK(loom_low_lower_rule_set_select_with_match_context(
      &match_context, &loom_test_low_lower_rule_set, source_op, &selection));

  EXPECT_TRUE(selection.has_source_op_span);
  ASSERT_NE(selection.rule, nullptr);
  EXPECT_EQ(selection.diagnostic_index, LOOM_LOW_LOWER_DIAGNOSTIC_NONE);
  EXPECT_EQ(loom_low_lower_rule_set_selection_diagnostic(
                &loom_test_low_lower_rule_set, selection),
            nullptr);
  EXPECT_EQ(loom_low_lower_rule_first_descriptor_id(
                &loom_test_low_lower_rule_set, selection.rule),
            TEST_LOW_CORE_DESCRIPTOR_ID_TEST_ADD_I32);
}

TEST_F(LowLowerTest, RuleSelectionWithMatchContextUsesMaterializerCallback) {
  ModulePtr module = ParseSource(
      "func.def @madd(%lhs: index, %rhs: index, %acc: index) -> (index) {\n"
      "  %result = index.madd %lhs, %rhs, %acc : index\n"
      "  func.return %result : index\n"
      "}\n");
  const loom_op_t* source_op =
      FirstBodyOpOfKind(module.get(), LOOM_OP_INDEX_MADD);
  ASSERT_NE(source_op, nullptr);

  loom_low_lower_rule_match_context_t match_context = {
      .module = module.get(),
      .descriptor_set = loom_test_low_core_descriptor_set(),
      .map_value =
          {
              .fn = loom_test_low_lower_rule_match_map_value,
              .user_data = nullptr,
          },
  };
  loom_low_lower_rule_selection_t selection = {};
  IREE_ASSERT_OK(loom_low_lower_rule_set_select_with_match_context(
      &match_context, &loom_test_low_lower_rule_set, source_op, &selection));
  EXPECT_EQ(selection.rule, nullptr);
  const loom_low_lower_diagnostic_t* diagnostic =
      loom_low_lower_rule_set_selection_diagnostic(
          &loom_test_low_lower_rule_set, selection);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_TRUE(iree_string_view_equal(diagnostic->subject_kind,
                                     IREE_SV("materializer")));
  EXPECT_TRUE(
      iree_string_view_equal(diagnostic->subject_name, IREE_SV("copy")));
  EXPECT_TRUE(iree_string_view_equal(
      diagnostic->reason,
      IREE_SV("test lowering requires copy-materializable values")));

  match_context.can_materialize.fn =
      loom_test_low_lower_rule_match_can_materialize;
  match_context.can_materialize.user_data = nullptr;
  IREE_ASSERT_OK(loom_low_lower_rule_set_select_with_match_context(
      &match_context, &loom_test_low_lower_rule_set, source_op, &selection));

  EXPECT_TRUE(selection.has_source_op_span);
  ASSERT_NE(selection.rule, nullptr);
  EXPECT_EQ(selection.diagnostic_index, LOOM_LOW_LOWER_DIAGNOSTIC_NONE);
  EXPECT_EQ(loom_low_lower_rule_first_descriptor_id(
                &loom_test_low_lower_rule_set, selection.rule),
            TEST_LOW_CORE_DESCRIPTOR_ID_TEST_ADD_I32);
}

TEST_F(LowLowerTest, LowersScalarFunctionAndSurvivesTextAndBytecodeRoundTrip) {
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

  loom_text_low_asm_environment_t environment = LowAsmEnvironment();
  const loom_text_print_options_t print_options = {
      .flags = LOOM_TEXT_PRINT_DEFAULT,
      .low_asm_environment = environment,
      .low_asm_descriptor_set_key = IREE_SV("test.low.core"),
  };
  std::string text;
  IREE_ASSERT_OK(PrintModule(module.get(), &print_options, &text));
  EXPECT_NE(text.find("low.func.def target(@test_target) "
                      "abi(object_function)"),
            std::string::npos);
  EXPECT_NE(text.find("@add_const__low"), std::string::npos);
  EXPECT_EQ(text.find("source(@"), std::string::npos);
  EXPECT_NE(text.find("asm<test.low.core>"), std::string::npos);
  EXPECT_NE(text.find("test.const.i32"), std::string::npos);
  EXPECT_NE(text.find("test.add.i32"), std::string::npos);
  EXPECT_EQ(text.find("low.abi"), std::string::npos);

  std::string invoked_text = text;
  invoked_text.append(
      "\nfunc.def @call_lowered(%lhs: i32) -> (i32) {\n"
      "  %result = low.invoke @add_const__low(%lhs) : (i32) -> (i32)\n"
      "  func.return %result : i32\n"
      "}\n");
  ModulePtr invoked_module = ParseSource(invoked_text, &environment);
  EmissionCollector invoked_verify_collector;
  loom_low_verify_result_t invoked_verify_result = {};
  IREE_ASSERT_OK(VerifyLowModule(
      invoked_module.get(), &invoked_verify_collector, &invoked_verify_result));
  EXPECT_EQ(invoked_verify_result.error_count, 0u);
  EXPECT_TRUE(invoked_verify_collector.emissions.empty());

  ModulePtr reparsed_module = ParseSource(text, &environment);
  EmissionCollector reparse_verify_collector;
  loom_low_verify_result_t reparse_verify_result = {};
  IREE_ASSERT_OK(VerifyLowModule(reparsed_module.get(),
                                 &reparse_verify_collector,
                                 &reparse_verify_result));
  EXPECT_EQ(reparse_verify_result.error_count, 0u);
  EXPECT_TRUE(reparse_verify_collector.emissions.empty());

  std::vector<uint8_t> bytes;
  IREE_ASSERT_OK(WriteModule(reparsed_module.get(), &bytes));
  ASSERT_FALSE(bytes.empty());

  loom_module_t* read_module_raw = nullptr;
  IREE_ASSERT_OK(ReadModule(bytes, &read_module_raw));
  ModulePtr read_module(read_module_raw);
  EmissionCollector read_verify_collector;
  loom_low_verify_result_t read_verify_result = {};
  IREE_ASSERT_OK(VerifyLowModule(read_module.get(), &read_verify_collector,
                                 &read_verify_result));
  EXPECT_EQ(read_verify_result.error_count, 0u);
  EXPECT_TRUE(read_verify_collector.emissions.empty());

  std::string disassembled_text;
  IREE_ASSERT_OK(
      PrintModule(read_module.get(), &print_options, &disassembled_text));
  EXPECT_NE(disassembled_text.find("@add_const__low"), std::string::npos);
  EXPECT_NE(disassembled_text.find("asm<test.low.core>"), std::string::npos);
  EXPECT_NE(disassembled_text.find("test.add.i32"), std::string::npos);
}

TEST_F(LowLowerTest,
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

  std::string text;
  IREE_ASSERT_OK(PrintModule(linked.get(), nullptr, &text));
  EXPECT_EQ(text.find("func.decl @add"), std::string::npos);
  EXPECT_NE(text.find("func.def target(@test_target) @add"), std::string::npos);
  EXPECT_NE(text.find("low.func.def target(@test_target) "
                      "abi(object_function) @add__low"),
            std::string::npos);
}

TEST_F(LowLowerTest, ComposedRuleSetsContinueAfterRejectedOverlap) {
  policy_registry_ = MakeTestComposedPolicyRegistry();
  IREE_ASSERT_OK(loom_low_lower_policy_registry_verify(&policy_registry_));

  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  ModulePtr module = ParseAndLowerTargetedSource(
      "target.profile @test_target preset(\"test-low\")\n"
      "func.def target(@test_target) @add(%lhs: i32, %rhs: i32) -> (i32) {\n"
      "  %sum = scalar.addi %lhs, %rhs : i32\n"
      "  func.return %sum : i32\n"
      "}\n",
      &lower_collector, &lower_result);

  EXPECT_EQ(lower_result.error_count, 0u);
  EXPECT_EQ(lower_result.remark_count, 0u);
  EXPECT_NE(lower_result.low_func_op, nullptr);
  EXPECT_TRUE(lower_collector.emissions.empty());

  const loom_text_print_options_t print_options = {
      .flags = LOOM_TEXT_PRINT_DEFAULT,
  };
  std::string text;
  IREE_ASSERT_OK(PrintModule(module.get(), &print_options, &text));
  EXPECT_NE(text.find("@add__low"), std::string::npos);
  EXPECT_NE(text.find("low.op<test.add.i32>"), std::string::npos);
}

TEST_F(LowLowerTest, LowersScalarFunctionTargetingProfile) {
  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  ModulePtr module = ParseAndLowerProfileTarget(
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

  const loom_text_print_options_t print_options = {
      .flags = LOOM_TEXT_PRINT_DEFAULT,
  };
  std::string text;
  IREE_ASSERT_OK(PrintModule(module.get(), &print_options, &text));
  EXPECT_NE(text.find("target.profile @test_target preset(\"test-low\")"),
            std::string::npos);
  EXPECT_NE(text.find("low.func.def target(@test_target) "
                      "abi(vm_module_function) export(\"add_const_export\")"),
            std::string::npos);
  EXPECT_NE(text.find("@add_const__low"), std::string::npos);
  EXPECT_NE(text.find("test.const.i32"), std::string::npos);
  EXPECT_NE(text.find("test.add.i32"), std::string::npos);
  EXPECT_EQ(text.find("target.bundle"), std::string::npos);
}

TEST_F(LowLowerTest, RuleEmissionCopiesTiedOperandsBeforeDestructiveOp) {
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

  const loom_text_print_options_t print_options = {
      .flags = LOOM_TEXT_PRINT_DEFAULT,
  };
  std::string text;
  IREE_ASSERT_OK(PrintModule(module.get(), &print_options, &text));
  const size_t copy_position = text.find("low.copy %lhs");
  const size_t tied_position = text.find("low.op<test.tied.any>");
  ASSERT_NE(copy_position, std::string::npos);
  ASSERT_NE(tied_position, std::string::npos);
  EXPECT_LT(copy_position, tied_position);
  EXPECT_NE(text.find("-> %"), std::string::npos);
  EXPECT_NE(text.find(" as reg<test.i32>"), std::string::npos);
}

TEST_F(LowLowerTest, RuleEmissionCarriesTemporaryBetweenDescriptorOps) {
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

TEST_F(LowLowerTest, LowersVectorCfgFunction) {
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
      "  cfg.br ^join(%sum : vector<4xi32>)\n"
      "^else:\n"
      "  %sum2 = vector.addi %b, %a : vector<4xi32>\n"
      "  cfg.br ^join(%sum2 : vector<4xi32>)\n"
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

  const loom_text_print_options_t print_options = {
      .flags = LOOM_TEXT_PRINT_DEFAULT,
  };
  std::string text;
  IREE_ASSERT_OK(PrintModule(module.get(), &print_options, &text));
  EXPECT_NE(text.find("@select_vector__low"), std::string::npos);
  EXPECT_EQ(text.find("source(@"), std::string::npos);
  EXPECT_NE(text.find("reg<test.i32 x4>"), std::string::npos);
  EXPECT_NE(text.find("low.slice"), std::string::npos);
  EXPECT_NE(text.find("low.concat"), std::string::npos);
  EXPECT_NE(text.find("test.add.i32"), std::string::npos);
  EXPECT_NE(text.find("low.cond_br"), std::string::npos);
  EXPECT_NE(text.find("low.br"), std::string::npos);
  EXPECT_EQ(text.find("low.abi"), std::string::npos);
}

TEST_F(LowLowerTest, RuleEmissionSwapsPerLaneOperands) {
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

TEST_F(LowLowerTest, LowersResourceArgumentsWithoutDirectAbiOperands) {
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

  const loom_text_print_options_t print_options = {
      .flags = LOOM_TEXT_PRINT_DEFAULT,
  };
  std::string text;
  IREE_ASSERT_OK(PrintModule(module.get(), &print_options, &text));
  EXPECT_NE(text.find("low.func.def target(@test_target) "
                      "abi(object_function)"),
            std::string::npos);
  EXPECT_NE(text.find("@resource_entry__low()"), std::string::npos);
  EXPECT_NE(text.find("%buffer = low.resource<native_pointer>"),
            std::string::npos);
  EXPECT_NE(text.find("semantic_type = buffer"), std::string::npos);
  EXPECT_EQ(text.find("source_arg"), std::string::npos);
  EXPECT_EQ(text.find("low.abi"), std::string::npos);
  EXPECT_EQ(text.find("low.abi.operand"), std::string::npos);
}

TEST_F(LowLowerTest, EmitsPreambleBeforeResourceImports) {
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

  const loom_text_print_options_t print_options = {
      .flags = LOOM_TEXT_PRINT_DEFAULT,
  };
  std::string text;
  IREE_ASSERT_OK(PrintModule(module.get(), &print_options, &text));
  const size_t live_in_position = text.find("low.live_in<test.thread_id>");
  const size_t resource_position = text.find("low.resource");
  ASSERT_NE(live_in_position, std::string::npos);
  ASSERT_NE(resource_position, std::string::npos);
  EXPECT_LT(live_in_position, resource_position);
}

TEST_F(LowLowerTest, HybridPolicyUsesCallbackForOpsNotCoveredByRuleTable) {
  policy_registry_ = MakeTestHybridPolicyRegistry();
  IREE_ASSERT_OK(loom_low_lower_policy_registry_verify(&policy_registry_));

  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  ModulePtr module = ParseAndLowerTargetedSource(
      "target.profile @test_target preset(\"test-low\")\n"
      "func.def target(@test_target) @mul(%lhs: i32, %rhs: i32) -> (i32) {\n"
      "  %product = scalar.muli %lhs, %rhs : i32\n"
      "  func.return %product : i32\n"
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

  const loom_text_print_options_t print_options = {
      .flags = LOOM_TEXT_PRINT_DEFAULT,
  };
  std::string text;
  IREE_ASSERT_OK(PrintModule(module.get(), &print_options, &text));
  EXPECT_NE(text.find("@mul__low"), std::string::npos);
  EXPECT_NE(text.find("low.op<test.add.i32>"), std::string::npos);
}

TEST_F(LowLowerTest, HybridPolicyKeepsRuleDiagnosticsForCoveredOps) {
  policy_registry_ = MakeTestHybridPolicyRegistry();
  IREE_ASSERT_OK(loom_low_lower_policy_registry_verify(&policy_registry_));

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

TEST_F(LowLowerTest, UnsupportedSourceOpEmitsDiagnosticAndNoLowFunction) {
  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  ModulePtr module = ParseAndLowerTargetedSource(
      "target.profile @test_target preset(\"test-low\")\n"
      "func.def target(@test_target) @mul(%lhs: i32, %rhs: i32) -> (i32) {\n"
      "  %product = scalar.muli %lhs, %rhs : i32\n"
      "  func.return %product : i32\n"
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
  EXPECT_EQ(emission.string_params[3], "mul");
  EXPECT_EQ(emission.string_params[4], "op");
  EXPECT_EQ(emission.string_params[5], "scalar.muli");
  EXPECT_NE(emission.string_params[6].find("descriptor mapping"),
            std::string::npos);

  loom_string_id_t low_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module.get(), IREE_SV("mul__low"),
                                           &low_name_id));
  EXPECT_EQ(loom_module_find_symbol(module.get(), low_name_id),
            LOOM_SYMBOL_ID_INVALID);
}

}  // namespace
}  // namespace loom
