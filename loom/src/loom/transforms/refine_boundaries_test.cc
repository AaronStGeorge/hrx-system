// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/refine_boundaries.h"

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_defs.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"

namespace loom {
namespace {

using DialectVtablesFn = const loom_op_vtable_t* const* (*)(iree_host_size_t*);

struct DiagnosticEmissionCollector {
  // Number of structured diagnostics collected.
  int count = 0;

  // Last structured error definition emitted.
  const loom_error_def_t* error = nullptr;

  // Last pass name parameter.
  std::string pass_name;

  // Last scope parameter.
  std::string scope;

  // Last reason parameter.
  std::string reason;
};

static std::string CopyString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

static iree_status_t CollectDiagnosticEmission(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  auto* collector = static_cast<DiagnosticEmissionCollector*>(user_data);
  ++collector->count;
  collector->error = emission->error;
  if (emission->error == loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 2) &&
      emission->param_count >= 3) {
    collector->pass_name = CopyString(emission->params[0].string);
    collector->scope = CopyString(emission->params[1].string);
    collector->reason = CopyString(emission->params[2].string);
  }
  return iree_ok_status();
}

class RefineBoundariesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void RegisterDialect(uint8_t dialect_id, DialectVtablesFn vtables_fn) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = vtables_fn(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)vtable_count));
  }

  iree_status_t ParseModule(const char* source, loom_module_t** out_module) {
    loom_text_parse_options_t parse_options = {};
    return loom_text_parse(iree_make_cstring_view(source),
                           IREE_SV("refine_boundaries_test.loom"), &context_,
                           &block_pool_, &parse_options, out_module);
  }

  iree_status_t RunRefineBoundaries(
      loom_module_t* module, std::vector<int64_t>* statistics,
      const loom_refine_boundaries_options_t* options = nullptr,
      DiagnosticEmissionCollector* collector = nullptr) {
    const loom_pass_info_t* info = loom_refine_boundaries_pass_info();
    statistics->assign(info->statistic_count, 0);

    iree_arena_allocator_t pass_arena;
    iree_arena_initialize(&block_pool_, &pass_arena);
    loom_pass_t pass = {};
    pass.info = info;
    pass.instance_arena = &pass_arena;
    pass.arena = &pass_arena;
    pass.statistics = statistics->data();
    if (collector) {
      pass.diagnostic_emitter.fn = CollectDiagnosticEmission;
      pass.diagnostic_emitter.user_data = collector;
    }
    iree_status_t status =
        loom_refine_boundaries_run_with_options(&pass, module, options);
    iree_arena_deinitialize(&pass_arena);
    return status;
  }

  int64_t StatisticValue(const std::vector<int64_t>& statistics,
                         iree_string_view_t name) {
    const loom_pass_info_t* info = loom_refine_boundaries_pass_info();
    for (uint16_t i = 0; i < info->statistic_count; ++i) {
      if (iree_string_view_equal(info->statistic_defs[i].name, name)) {
        return statistics[i];
      }
    }
    return -1;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(RefineBoundariesTest, SkipsUnchangedFunctionFacts) {
  const char* source = R"(
func.def @identity(%value: index) -> (index) {
  func.return %value : index
}

func.def @stable(%value: index) -> (index) {
  %two = index.constant 2 : index
  %result = index.add %value, %two : index
  func.return %result : index
}

func.def public @caller(%dynamic: index) -> (index, index) {
  %zero = index.constant 0 : index
  %left = func.call @identity(%zero) : (index) -> (index)
  %right = func.call @stable(%dynamic) : (index) -> (index)
  func.return %left, %right : index, index
}
)";

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseModule(source, &module));
  ASSERT_NE(module, nullptr);

  std::vector<int64_t> statistics;
  iree_status_t status = RunRefineBoundaries(module, &statistics);
  loom_module_free(module);
  IREE_ASSERT_OK(status);

  EXPECT_GT(StatisticValue(statistics, IREE_SV("function-fact-cache-hits")), 0);
  EXPECT_GT(StatisticValue(statistics, IREE_SV("functions-canonicalized")), 0);
}

TEST_F(RefineBoundariesTest, EmitsDiagnosticWhenFactsDoNotConverge) {
  const char* source = R"(
func.def @identity(%value: index) -> (index) {
  func.return %value : index
}

func.def public @caller() -> (index) {
  %zero = index.constant 0 : index
  %result = func.call @identity(%zero) : (index) -> (index)
  func.return %result : index
}
)";

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseModule(source, &module));
  ASSERT_NE(module, nullptr);

  loom_refine_boundaries_options_t options = {};
  options.max_iterations = 1;
  std::vector<int64_t> statistics;
  DiagnosticEmissionCollector collector;
  iree_status_t status =
      RunRefineBoundaries(module, &statistics, &options, &collector);
  loom_module_free(module);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);

  EXPECT_EQ(collector.count, 1);
  EXPECT_EQ(collector.error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 2));
  EXPECT_EQ(collector.pass_name, "refine-boundaries");
  EXPECT_EQ(collector.scope, "module");
  EXPECT_NE(collector.reason.find("did not converge"), std::string::npos);
}

}  // namespace
}  // namespace loom
