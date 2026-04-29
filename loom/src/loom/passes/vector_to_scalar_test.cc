// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/passes/vector_to_scalar.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_defs.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/families.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/pass/types.h"
#include "loom/pass/value_facts.h"

namespace loom {
namespace {

using DialectVtablesFn = const loom_op_vtable_t* const* (*)(iree_host_size_t*);

struct DiagnosticEmissionCollector {
  // Last structured error definition emitted.
  const loom_error_def_t* error = nullptr;
  // Last rendered lowering reason parameter.
  std::string reason;
  // Number of diagnostic emissions collected.
  int count = 0;
};

static std::string CopyString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

static iree_status_t CollectDiagnosticEmission(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  auto* collector = static_cast<DiagnosticEmissionCollector*>(user_data);
  ++collector->count;
  collector->error = emission->error;
  if (emission->error == loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 1) &&
      emission->param_count >= 3) {
    collector->reason = CopyString(emission->params[2].string);
  }
  return iree_ok_status();
}

class VectorToScalarTransformTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_ENCODING, loom_encoding_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCF, loom_scf_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables);
    IREE_ASSERT_OK(loom_context_register_builtin_encoding_vtables(&context_));
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

  iree_status_t RunSingleFunction(const char* source,
                                  DiagnosticEmissionCollector* collector) {
    loom_module_t* module = NULL;
    loom_text_parse_options_t parse_options = {};
    iree_status_t status = loom_text_parse(
        iree_make_cstring_view(source), IREE_SV("vector_to_scalar_test.loom"),
        &context_, &block_pool_, &parse_options, &module);
    if (!iree_status_is_ok(status)) {
      if (module) loom_module_free(module);
      return status;
    }
    if (!module) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "parser did not produce a module");
    }

    loom_func_like_t function = {};
    for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
      loom_symbol_t* symbol = &module->symbols.entries[i];
      if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
        continue;
      }
      function = loom_func_like_cast(module, symbol->defining_op);
      if (loom_func_like_body(function)) break;
    }
    if (!loom_func_like_body(function)) {
      loom_module_free(module);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "module does not contain a function body");
    }

    iree_arena_allocator_t pass_arena;
    iree_arena_initialize(&block_pool_, &pass_arena);
    loom_pass_t pass = {};
    pass.info = loom_vector_to_scalar_pass_info();
    pass.instance_arena = &pass_arena;
    pass.arena = &pass_arena;
    pass.diagnostic_emitter.fn = CollectDiagnosticEmission;
    pass.diagnostic_emitter.user_data = collector;
    loom_pass_value_fact_owner_t value_facts = {};
    loom_pass_value_fact_owner_initialize(&block_pool_, &value_facts);
    pass.value_facts = &value_facts;
    status = loom_vector_to_scalar_run(&pass, module, function);
    loom_pass_value_fact_owner_deinitialize(&value_facts);
    iree_arena_deinitialize(&pass_arena);
    loom_module_free(module);
    return status;
  }

  // Block pool shared by parser, module allocation, and pass arenas.
  iree_arena_block_pool_t block_pool_;

  // Context with the dialects needed by vector.transform lowering tests.
  loom_context_t context_;
};

TEST_F(VectorToScalarTransformTest,
       RejectsOutOfBoundsIotaTransformPermutation) {
  const char* source = R"(
func.def @test(%v: vector<2xf32>, %signs: vector<2xi1>) -> (vector<2xf32>) {
  %one = index.constant 1 : index
  %permutation = vector.iota %one step %one : vector<2xindex>
  %xf = encoding.define #numeric_transform<family="sign_permute_hadamard", input_elems=2, output_elems=2> {permutation = %permutation : vector<2xindex>, signs = %signs : vector<2xi1>} : encoding<transform>
  %r = vector.transform %v, %xf : vector<2xf32>, encoding<transform> -> vector<2xf32>
  func.return %r : vector<2xf32>
}
)";

  DiagnosticEmissionCollector collector;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        RunSingleFunction(source, &collector));
  EXPECT_EQ(collector.count, 1);
  EXPECT_EQ(collector.error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 1));
  EXPECT_NE(collector.reason.find("in-bounds source lanes"), std::string::npos);
}

TEST_F(VectorToScalarTransformTest, RejectsDuplicateUniformPermutation) {
  const char* source = R"(
func.def @test(%v: vector<2xf32>, %signs: vector<2xi1>) -> (vector<2xf32>) {
  %permutation = vector.constant 0 : vector<2xindex>
  %xf = encoding.define #numeric_transform<family="sign_permute_hadamard", input_elems=2, output_elems=2> {permutation = %permutation : vector<2xindex>, signs = %signs : vector<2xi1>} : encoding<transform>
  %r = vector.transform %v, %xf : vector<2xf32>, encoding<transform> -> vector<2xf32>
  func.return %r : vector<2xf32>
}
)";

  DiagnosticEmissionCollector collector;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        RunSingleFunction(source, &collector));
  EXPECT_EQ(collector.count, 1);
  EXPECT_EQ(collector.error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 1));
  EXPECT_NE(collector.reason.find("name each source lane once"),
            std::string::npos);
}

}  // namespace
}  // namespace loom
