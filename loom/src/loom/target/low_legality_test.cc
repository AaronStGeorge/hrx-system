// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/low_legality.h"

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_defs.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/test/low_registry.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using CollectedEmission = ::loom::testing::CapturedDiagnosticEmission;
using EmissionCollector = ::loom::testing::DiagnosticEmissionCapture;
using ModulePtr = ::loom::testing::ModulePtr;

static iree_status_t IgnoreProviderOp(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  (void)context;
  (void)op;
  *out_handled = false;
  return iree_ok_status();
}

TEST(TargetLowLegalityProviderListTest, Empty) {
  const loom_target_low_legality_provider_list_t list =
      loom_target_low_legality_provider_list_empty();
  EXPECT_TRUE(loom_target_low_legality_provider_list_is_empty(list));
  IREE_EXPECT_OK(loom_target_low_legality_provider_list_verify(list));
}

TEST(TargetLowLegalityProviderListTest, VerifiesValues) {
  const loom_target_low_legality_provider_t provider = {
      .name = IREE_SVL("test-provider"),
      .try_verify_op = IgnoreProviderOp,
  };
  const loom_target_low_legality_provider_t* values[] = {&provider};
  const loom_target_low_legality_provider_list_t list =
      loom_target_low_legality_provider_list_make(values,
                                                  IREE_ARRAYSIZE(values));
  EXPECT_FALSE(loom_target_low_legality_provider_list_is_empty(list));
  IREE_EXPECT_OK(loom_target_low_legality_provider_list_verify(list));
}

TEST(TargetLowLegalityProviderListTest, RejectsMissingValues) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_low_legality_provider_list_verify(
          (loom_target_low_legality_provider_list_t){.count = 1}));
}

class TargetLowLegalityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables,
                    loom_func_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_BUFFER, loom_buffer_dialect_vtables,
                    loom_buffer_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_ENCODING, loom_encoding_dialect_vtables,
                    loom_encoding_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables,
                    loom_scalar_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables,
                    loom_index_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables,
                    loom_vector_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_CFG, loom_cfg_dialect_vtables,
                    loom_cfg_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_KERNEL, loom_kernel_dialect_vtables,
                    loom_kernel_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_SCF, loom_scf_dialect_vtables,
                    loom_scf_dialect_op_semantics);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_test_low_descriptor_registry_initialize(&registry_);
    IREE_ASSERT_OK(loom_target_low_descriptor_registry_lookup_bundle(
        &registry_, IREE_SV("test-low"), &bundle_));
    ASSERT_NE(bundle_, nullptr);
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
                                  IREE_SV("low_legality_test.loom"), &context_,
                                  &block_pool_, &parse_options, &module));
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

  loom_func_like_t FirstFunction(loom_module_t* module) {
    loom_block_t* block = loom_module_block(module);
    loom_op_t* op = nullptr;
    loom_block_for_each_op(block, op) {
      loom_func_like_t function = loom_func_like_cast(module, op);
      if (loom_func_like_isa(function)) {
        return function;
      }
    }
    ADD_FAILURE() << "expected module to contain a function";
    return (loom_func_like_t){0};
  }

  iree_status_t Verify(loom_module_t* module, EmissionCollector* collector,
                       loom_target_low_legality_result_t* result) {
    const loom_target_low_legality_options_t options = {
        .bundle = bundle_,
        .descriptor_registry = &registry_.registry,
        .descriptor_requirements =
            LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
        .emitter = collector->emitter(),
    };
    return loom_target_low_verify_function_legality(
        module, FirstFunction(module), &options, result);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t registry_ = {};
  const loom_target_bundle_t* bundle_ = nullptr;
};

TEST_F(TargetLowLegalityTest, AcceptsScalarSourceFunction) {
  ModulePtr module = ParseSource(
      "func.def @add(%lhs: i32, %rhs: i32) -> (i32) {\n"
      "  %sum = scalar.addi %lhs, %rhs : i32\n"
      "  func.return %sum : i32\n"
      "}\n");

  EmissionCollector collector;
  loom_target_low_legality_result_t result = {};
  IREE_ASSERT_OK(Verify(module.get(), &collector, &result));

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_EQ(result.remark_count, 0u);
  EXPECT_NE(result.descriptor_set, nullptr);
  EXPECT_TRUE(collector.emissions.empty());
}

TEST_F(TargetLowLegalityTest, AcceptsVectorCfgSourceFunction) {
  ModulePtr module = ParseSource(
      "func.def @select_vector(%cond: i1, %a: vector<4xi32>, "
      "%b: vector<4xi32>) -> (vector<4xi32>) {\n"
      "  cfg.cond_br %cond, ^then, ^else : i1\n"
      "^then:\n"
      "  %sum = vector.addi %a, %b : vector<4xi32>\n"
      "  cfg.br ^join(%sum: vector<4xi32>)\n"
      "^else:\n"
      "  %diff = vector.subi %a, %b : vector<4xi32>\n"
      "  cfg.br ^join(%diff: vector<4xi32>)\n"
      "^join(%result: vector<4xi32>):\n"
      "  func.return %result : vector<4xi32>\n"
      "}\n");

  EmissionCollector collector;
  loom_target_low_legality_result_t result = {};
  IREE_ASSERT_OK(Verify(module.get(), &collector, &result));

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_EQ(result.remark_count, 0u);
  EXPECT_NE(result.descriptor_set, nullptr);
  EXPECT_TRUE(collector.emissions.empty());
}

TEST_F(TargetLowLegalityTest, AcceptsKernelWorkitemIdSourceFunction) {
  ModulePtr module = ParseSource(
      "func.def @kernel_coordinate() -> (index) {\n"
      "  %tid = kernel.workitem.id<x> : index\n"
      "  func.return %tid : index\n"
      "}\n");

  EmissionCollector collector;
  loom_target_low_legality_result_t result = {};
  IREE_ASSERT_OK(Verify(module.get(), &collector, &result));

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_EQ(result.remark_count, 0u);
  EXPECT_NE(result.descriptor_set, nullptr);
  EXPECT_TRUE(collector.emissions.empty());
}

TEST_F(TargetLowLegalityTest, RejectsScfBeforeCfgLowering) {
  ModulePtr module = ParseSource(
      "func.def @structured(%cond: i1, %a: i32, %b: i32) -> (i32) {\n"
      "  %selected = scf.if %cond -> (i32) {\n"
      "    scf.yield %a : i32\n"
      "  } else {\n"
      "    scf.yield %b : i32\n"
      "  }\n"
      "  func.return %selected : i32\n"
      "}\n");

  EmissionCollector collector;
  loom_target_low_legality_result_t result = {};
  IREE_ASSERT_OK(Verify(module.get(), &collector, &result));

  ASSERT_EQ(result.error_count, 1u);
  ASSERT_EQ(collector.emissions.size(), 1u);
  const CollectedEmission& emission = collector.emissions[0];
  EXPECT_EQ(emission.error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 1));
  ASSERT_EQ(emission.string_params.size(), 7u);
  EXPECT_EQ(emission.string_params[3], "structured");
  EXPECT_EQ(emission.string_params[4], "op");
  EXPECT_EQ(emission.string_params[5], "scf.if");
  EXPECT_NE(emission.string_params[6].find("CFG"), std::string::npos);
}

TEST_F(TargetLowLegalityTest, AcceptsDotSourceFunctionWithoutProvider) {
  ModulePtr module = ParseSource(
      "func.def @dot(%lhs: vector<4xi8>, %rhs: vector<4xi8>, "
      "%acc: vector<1xi32>) -> (vector<1xi32>) {\n"
      "  %result = vector.dot4i<s8s8> %lhs, %rhs, %acc : "
      "vector<4xi8>, vector<4xi8>, vector<1xi32>\n"
      "  func.return %result : vector<1xi32>\n"
      "}\n");

  EmissionCollector collector;
  loom_target_low_legality_result_t result = {};
  IREE_ASSERT_OK(Verify(module.get(), &collector, &result));

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_EQ(result.remark_count, 0u);
  EXPECT_TRUE(collector.emissions.empty());
}

TEST_F(TargetLowLegalityTest, RejectsProviderRequiredOpWithoutProvider) {
  ModulePtr module = ParseSource(
      "func.def @iota(%base: i32, %step: i32) -> (vector<4xi32>) {\n"
      "  %lanes = vector.iota %base step %step : vector<4xi32>\n"
      "  func.return %lanes : vector<4xi32>\n"
      "}\n");

  EmissionCollector collector;
  loom_target_low_legality_result_t result = {};
  IREE_ASSERT_OK(Verify(module.get(), &collector, &result));

  ASSERT_EQ(result.error_count, 1u);
  ASSERT_EQ(collector.emissions.size(), 1u);
  const CollectedEmission& emission = collector.emissions[0];
  EXPECT_EQ(emission.error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 1));
  ASSERT_EQ(emission.string_params.size(), 7u);
  EXPECT_EQ(emission.string_params[3], "iota");
  EXPECT_EQ(emission.string_params[4], "op");
  EXPECT_EQ(emission.string_params[5], "vector.iota");
  EXPECT_NE(emission.string_params[6].find("provider"), std::string::npos);
}

static iree_status_t AcceptVectorIotaContract(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  (void)context;
  if (!loom_vector_iota_isa(op)) {
    *out_handled = false;
    return iree_ok_status();
  }
  *out_handled = true;
  return iree_ok_status();
}

static iree_status_t AcceptAnyContractOp(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  loom_op_semantics_t semantics =
      loom_op_semantics(loom_target_low_legality_module(context), op);
  *out_handled = semantics.contract_families != 0;
  return iree_ok_status();
}

TEST_F(TargetLowLegalityTest, ProviderMayClaimProviderRequiredOp) {
  ModulePtr module = ParseSource(
      "func.def @iota(%base: i32, %step: i32) -> (vector<4xi32>) {\n"
      "  %lanes = vector.iota %base step %step : vector<4xi32>\n"
      "  func.return %lanes : vector<4xi32>\n"
      "}\n");

  const loom_target_low_legality_provider_t provider = {
      .name = IREE_SVL("test-provider"),
      .try_verify_op = AcceptVectorIotaContract,
  };
  const loom_target_low_legality_provider_t* providers[] = {&provider};
  const loom_target_low_legality_provider_list_t provider_list =
      loom_target_low_legality_provider_list_make(providers,
                                                  IREE_ARRAYSIZE(providers));
  EmissionCollector collector;
  const loom_target_low_legality_options_t options = {
      .bundle = bundle_,
      .descriptor_registry = &registry_.registry,
      .descriptor_requirements =
          LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
      .provider_list = provider_list,
      .emitter = collector.emitter(),
  };
  loom_target_low_legality_result_t result = {};
  IREE_ASSERT_OK(loom_target_low_verify_function_legality(
      module.get(), FirstFunction(module.get()), &options, &result));

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_EQ(result.remark_count, 0u);
  EXPECT_TRUE(collector.emissions.empty());
}

TEST_F(TargetLowLegalityTest, ProviderMayClaimTensorAsyncContractValues) {
  ModulePtr module = ParseSource(
      "func.def @tensor_async(%source_buffer: buffer, %byte_offset: offset) {\n"
      "  %zero = index.constant 0 : offset\n"
      "  %bytes = index.constant 16384 : offset\n"
      "  %layout = encoding.layout.dense : encoding<layout>\n"
      "  %i32_zero = scalar.constant 0 : i32\n"
      "  %d0 = vector.splat %i32_zero : vector<4xi32>\n"
      "  %d1 = vector.splat %i32_zero : vector<8xi32>\n"
      "  %desc = kernel.tensor.lds.descriptor dgroups(%d0, %d1) : "
      "vector<4xi32>, vector<8xi32> -> kernel.tensor.lds.descriptor\n"
      "  %global = buffer.assume.memory_space %source_buffer {memory_space = "
      "global} : buffer\n"
      "  %source = buffer.view %global[%byte_offset] : buffer -> "
      "view<64x64xf32, %layout>\n"
      "  %scratch = buffer.alloca %bytes {base_alignment = 256, memory_space = "
      "workgroup} : buffer\n"
      "  %dest = buffer.view %scratch[%zero] : buffer -> "
      "view<64x64xf32, %layout>\n"
      "  %copy = kernel.async.tensor.load.to.lds %source to %dest using %desc "
      "{cache_scope = cu, cache_temporal = regular} : "
      "view<64x64xf32, %layout> to view<64x64xf32, %layout>, "
      "kernel.tensor.lds.descriptor -> kernel.async.token\n"
      "  %group = kernel.async.group %copy : kernel.async.token -> "
      "kernel.async.group\n"
      "  kernel.async.wait %group {newer_groups = 0} : kernel.async.group\n"
      "  func.return\n"
      "}\n");

  const loom_target_low_legality_provider_t provider = {
      .name = IREE_SVL("test-provider"),
      .try_verify_op = AcceptAnyContractOp,
  };
  const loom_target_low_legality_provider_t* providers[] = {&provider};
  const loom_target_low_legality_provider_list_t provider_list =
      loom_target_low_legality_provider_list_make(providers,
                                                  IREE_ARRAYSIZE(providers));
  EmissionCollector collector;
  const loom_target_low_legality_options_t options = {
      .bundle = bundle_,
      .descriptor_registry = &registry_.registry,
      .descriptor_requirements =
          LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
      .provider_list = provider_list,
      .emitter = collector.emitter(),
  };
  loom_target_low_legality_result_t result = {};
  IREE_ASSERT_OK(loom_target_low_verify_function_legality(
      module.get(), FirstFunction(module.get()), &options, &result));

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_EQ(result.remark_count, 0u);
  EXPECT_TRUE(collector.emissions.empty());
}

static iree_status_t RejectScalarAddiContract(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  if (op->kind != LOOM_OP_SCALAR_ADDI) {
    *out_handled = false;
    return iree_ok_status();
  }
  *out_handled = true;
  return loom_target_low_legality_reject(
      context, provider, op, IREE_SV("op"), IREE_SV("scalar.addi"),
      IREE_SV("test provider rejected scalar.addi"));
}

TEST_F(TargetLowLegalityTest, ProviderMayClaimCoreSourceOp) {
  ModulePtr module = ParseSource(
      "func.def @add(%lhs: i32, %rhs: i32) -> (i32) {\n"
      "  %sum = scalar.addi %lhs, %rhs : i32\n"
      "  func.return %sum : i32\n"
      "}\n");

  const loom_target_low_legality_provider_t provider = {
      .name = IREE_SVL("test-provider"),
      .try_verify_op = RejectScalarAddiContract,
  };
  const loom_target_low_legality_provider_t* providers[] = {&provider};
  const loom_target_low_legality_provider_list_t provider_list =
      loom_target_low_legality_provider_list_make(providers,
                                                  IREE_ARRAYSIZE(providers));
  EmissionCollector collector;
  const loom_target_low_legality_options_t options = {
      .bundle = bundle_,
      .descriptor_registry = &registry_.registry,
      .descriptor_requirements =
          LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
      .provider_list = provider_list,
      .emitter = collector.emitter(),
  };
  loom_target_low_legality_result_t result = {};
  IREE_ASSERT_OK(loom_target_low_verify_function_legality(
      module.get(), FirstFunction(module.get()), &options, &result));

  ASSERT_EQ(result.error_count, 1u);
  ASSERT_EQ(collector.emissions.size(), 1u);
  const CollectedEmission& emission = collector.emissions[0];
  EXPECT_EQ(emission.error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 1));
  ASSERT_EQ(emission.string_params.size(), 7u);
  EXPECT_EQ(emission.string_params[3], "add");
  EXPECT_EQ(emission.string_params[4], "op");
  EXPECT_EQ(emission.string_params[5], "scalar.addi");
  EXPECT_EQ(emission.string_params[6], "test provider rejected scalar.addi");
}

static iree_status_t RecordDot4iContract(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  if (!loom_vector_dot4i_isa(op)) {
    *out_handled = false;
    return iree_ok_status();
  }
  *out_handled = true;
  return loom_target_low_legality_record_contract(
      context, provider, op, IREE_SV("test.dot4i"), IREE_SV("selected"),
      IREE_SV("test provider accepted the dot4i contract"));
}

TEST_F(TargetLowLegalityTest, ProviderRecordsContractDecision) {
  ModulePtr module = ParseSource(
      "func.def @dot(%lhs: vector<4xi8>, %rhs: vector<4xi8>, "
      "%acc: vector<1xi32>) -> (vector<1xi32>) {\n"
      "  %result = vector.dot4i<s8s8> %lhs, %rhs, %acc : "
      "vector<4xi8>, vector<4xi8>, vector<1xi32>\n"
      "  func.return %result : vector<1xi32>\n"
      "}\n");

  const loom_target_low_legality_provider_t provider = {
      .name = IREE_SVL("test-provider"),
      .try_verify_op = RecordDot4iContract,
  };
  const loom_target_low_legality_provider_t* providers[] = {&provider};
  const loom_target_low_legality_provider_list_t provider_list =
      loom_target_low_legality_provider_list_make(providers,
                                                  IREE_ARRAYSIZE(providers));
  EmissionCollector collector;
  const loom_target_low_legality_options_t options = {
      .bundle = bundle_,
      .descriptor_registry = &registry_.registry,
      .descriptor_requirements =
          LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
      .provider_list = provider_list,
      .emitter = collector.emitter(),
  };
  loom_target_low_legality_result_t result = {};
  IREE_ASSERT_OK(loom_target_low_verify_function_legality(
      module.get(), FirstFunction(module.get()), &options, &result));

  EXPECT_EQ(result.error_count, 0u);
  ASSERT_EQ(result.remark_count, 1u);
  ASSERT_EQ(collector.emissions.size(), 1u);
  const CollectedEmission& emission = collector.emissions[0];
  EXPECT_EQ(emission.error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 2));
  ASSERT_EQ(emission.string_params.size(), 7u);
  EXPECT_EQ(emission.string_params[0], "test-low");
  EXPECT_EQ(emission.string_params[3], "dot");
  EXPECT_EQ(emission.string_params[4], "test.dot4i");
  EXPECT_EQ(emission.string_params[5], "selected");
}

}  // namespace
}  // namespace loom
