// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower_pass.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/lower.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

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
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_test_low_descriptor_registry_initialize(&registry_);
    invalid_preamble_policy_registry_ = MakeInvalidPreamblePolicyRegistry();
    IREE_ASSERT_OK(loom_low_lower_policy_registry_verify(
        &invalid_preamble_policy_registry_));
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
    IREE_CHECK_OK(loom_text_parse(source, IREE_SV("lower_pass_test.loom"),
                                  &context_, &block_pool_, &parse_options,
                                  &module));
    IREE_ASSERT(module != nullptr);
    return ModulePtr(module);
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

  iree_arena_allocator_t instance_arena;
  iree_arena_initialize(&block_pool_, &instance_arena);
  const loom_pass_info_t* pass_info = loom_low_source_to_low_pass_info();
  std::vector<int64_t> statistics(pass_info->statistic_count, 0);
  loom_low_source_to_low_pass_config_t config = {
      .descriptor_registry = &registry_.registry,
      .policy_registry = &invalid_preamble_policy_registry_,
  };
  loom_pass_t pass = {
      .info = pass_info,
      .module_run = loom_low_source_to_low_run,
      .instance_arena = &instance_arena,
      .arena = &instance_arena,
      .statistics = statistics.data(),
      .user_data = &config,
  };

  IREE_ASSERT_OK(
      loom_low_source_to_low_create(&pass, iree_string_view_empty()));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_low_source_to_low_run(&pass, module.get()));
  iree_arena_deinitialize(&instance_arena);
}

}  // namespace
}  // namespace loom
