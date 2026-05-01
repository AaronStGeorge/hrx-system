// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/passes/source_to_low.h"

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/lower_rules.h"
#include "loom/codegen/low/pass_environment.h"
#include "loom/error/error_defs.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/lower.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

struct DiagnosticEmissionCollector {
  int count = 0;
  std::string subject_name;
  std::string reason;
};

static std::string CopyString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

static iree_status_t CollectDiagnosticEmission(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  auto* collector = static_cast<DiagnosticEmissionCollector*>(user_data);
  ++collector->count;
  if (emission->error == loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 1) &&
      emission->param_count >= 7) {
    collector->subject_name = CopyString(emission->params[5].string);
    collector->reason = CopyString(emission->params[6].string);
  }
  return iree_ok_status();
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

static iree_status_t EmitInvalidPreamble(void* user_data,
                                         loom_low_lower_context_t* context) {
  (void)user_data;
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      MakeRegisterType(context, IREE_SV("test.i32"), 1, &result_type));
  loom_op_t* copy_op = nullptr;
  return loom_low_copy_build(
      loom_low_lower_context_builder(context), LOOM_VALUE_ID_INVALID,
      result_type, loom_low_lower_context_low_function(context)->location,
      &copy_op);
}

static const loom_low_lower_rule_set_t* const kInvalidPreambleRuleSets[] = {
    &loom_test_low_lower_rule_set,
};

static const loom_low_lower_policy_t kInvalidPreamblePolicy = {
    .name = IREE_SVL("invalid-preamble-policy"),
    .map_type = {.fn = loom_test_low_lower_map_type, .user_data = nullptr},
    .map_contract_value = {.fn = loom_test_low_lower_map_contract_value,
                           .user_data = nullptr},
    .map_argument = {.fn = loom_test_low_lower_map_argument,
                     .user_data = nullptr},
    .emit_preamble = {.fn = EmitInvalidPreamble, .user_data = nullptr},
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kInvalidPreambleRuleSets),
            .values = kInvalidPreambleRuleSets,
        },
};

static loom_low_lower_policy_registry_t MakeInvalidPreamblePolicyRegistry() {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("test.low.core"),
          .policy = &kInvalidPreamblePolicy,
      },
  };
  loom_low_lower_policy_registry_t registry = {};
  loom_low_lower_policy_registry_initialize_from_entries(
      &registry, kEntries, IREE_ARRAYSIZE(kEntries));
  return registry;
}

class LowLowerPassTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_test_low_descriptor_registry_initialize(&registry_);
    invalid_preamble_policy_registry_ = MakeInvalidPreamblePolicyRegistry();
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
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

  ModulePtr Parse(iree_string_view_t source) {
    loom_text_parse_options_t parse_options = {
        .max_errors = 20,
    };
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(source, IREE_SV("source_to_low_test.loom"),
                                  &context_, &block_pool_, &parse_options,
                                  &module));
    IREE_ASSERT(module != nullptr);
    return ModulePtr(module);
  }

  iree_status_t RunSourceToLow(
      loom_low_lower_policy_registry_t* policy_registry, loom_module_t* module,
      DiagnosticEmissionCollector* collector = nullptr) {
    iree_arena_allocator_t instance_arena;
    iree_arena_initialize(&block_pool_, &instance_arena);
    loom_pass_value_fact_owner_t value_facts = {};
    loom_pass_value_fact_owner_initialize(&block_pool_, &value_facts);
    const loom_pass_info_t* pass_info = loom_low_source_to_low_pass_info();
    std::vector<int64_t> statistics(pass_info->statistic_count, 0);
    loom_low_pass_environment_storage_t low_pass_environment_storage;
    loom_pass_environment_t environment =
        loom_low_pass_environment_storage_initialize(
            &registry_.registry, policy_registry, nullptr,
            &low_pass_environment_storage);
    loom_pass_t pass = {
        .info = pass_info,
        .module_run = loom_low_source_to_low_run,
        .instance_arena = &instance_arena,
        .arena = &instance_arena,
        .statistics = statistics.data(),
        .environment = &environment,
        .value_facts = &value_facts,
    };
    if (collector != nullptr) {
      pass.diagnostic_emitter = {
          .fn = CollectDiagnosticEmission,
          .user_data = collector,
      };
    }

    iree_status_t status =
        loom_low_source_to_low_create(&pass, iree_string_view_empty());
    if (iree_status_is_ok(status)) {
      status = loom_low_source_to_low_run(&pass, module);
    }
    loom_pass_value_fact_owner_deinitialize(&value_facts);
    iree_arena_deinitialize(&instance_arena);
    return status;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t registry_ = {};
  loom_low_lower_policy_registry_t invalid_preamble_policy_registry_ = {};
};

TEST_F(LowLowerPassTest, VerifiesLoweredModuleBeforeReturningSuccess) {
  ModulePtr module = Parse(IREE_SV(
      "target.profile @test_target preset(\"test-low\")\n"
      "func.def target(@test_target) @identity(%value: i32) -> (i32) {\n"
      "  func.return %value : i32\n"
      "}\n"));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      RunSourceToLow(&invalid_preamble_policy_registry_, module.get()));
}

TEST_F(LowLowerPassTest, ContractTableDrivesRuleSelectionWithoutLegacySpans) {
  ModulePtr module = Parse(IREE_SV(
      "target.profile @test_target preset(\"test-low\")\n"
      "func.def target(@test_target) @add(%lhs: i32, %rhs: i32) -> (i32) {\n"
      "  %sum = scalar.addi %lhs, %rhs : i32\n"
      "  func.return %sum : i32\n"
      "}\n"));

  loom_low_lower_rule_set_t no_span_rule_set = loom_test_low_lower_rule_set;
  no_span_rule_set.spans = nullptr;
  no_span_rule_set.span_count = 0;
  const loom_low_lower_rule_set_t* rule_sets[] = {
      &no_span_rule_set,
  };
  loom_low_lower_policy_t policy = *loom_test_low_lower_policy();
  policy.rule_sets = {
      .count = IREE_ARRAYSIZE(rule_sets),
      .values = rule_sets,
  };
  const loom_low_lower_policy_registry_entry_t entries[] = {
      {
          .contract_set_key = IREE_SVL("test.low.core"),
          .policy = &policy,
      },
  };
  loom_low_lower_policy_registry_t policy_registry = {};
  loom_low_lower_policy_registry_initialize_from_entries(
      &policy_registry, entries, IREE_ARRAYSIZE(entries));

  DiagnosticEmissionCollector collector;
  iree_status_t status =
      RunSourceToLow(&policy_registry, module.get(), &collector);
  EXPECT_EQ(collector.count, 0)
      << collector.subject_name << ": " << collector.reason;
  IREE_ASSERT_OK(status);
}

}  // namespace
}  // namespace loom
