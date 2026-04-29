// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/passes/vector_memory_footprint.h"

#include <string.h>

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_defs.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/families.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/pool/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/pass/types.h"
#include "loom/pass/value_facts.h"

namespace loom {
namespace {

using DialectVtablesFn = const loom_op_vtable_t* const* (*)(iree_host_size_t*);

struct DiagnosticEmissionCollector {
  // Last reported view axis for footprint failures.
  int64_t view_axis = -1;
  // Last reported vector axis mapped to view_axis.
  int64_t vector_axis = -1;
  // Number of diagnostic emissions collected.
  int count = 0;
  // Last rendered origin parameter.
  std::string origin;
  // Last rendered proof-failure reason parameter.
  std::string reason;
  // Last rendered required-condition parameter.
  std::string required;
  // Last structured error definition emitted.
  const loom_error_def_t* error = nullptr;
};

static std::string CopyString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

static iree_status_t CollectDiagnosticEmission(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  auto* collector = static_cast<DiagnosticEmissionCollector*>(user_data);
  ++collector->count;
  collector->error = emission->error;
  if (emission->error == loom_error_def_lookup(LOOM_ERROR_DOMAIN_SUBRANGE, 5) &&
      emission->param_count >= 6) {
    collector->view_axis = emission->params[1].i64;
    collector->vector_axis = emission->params[2].i64;
    collector->origin = CopyString(emission->params[3].string);
    collector->reason = CopyString(emission->params[4].string);
    collector->required = CopyString(emission->params[5].string);
  } else if (emission->error ==
                 loom_error_def_lookup(LOOM_ERROR_DOMAIN_SUBRANGE, 6) &&
             emission->param_count >= 4) {
    collector->origin = CopyString(emission->params[1].string);
    collector->reason = CopyString(emission->params[2].string);
    collector->required = CopyString(emission->params[3].string);
  }
  return iree_ok_status();
}

class VectorMemoryFootprintTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_ENCODING, loom_encoding_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_POOL, loom_pool_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_GLOBAL, loom_global_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCF, loom_scf_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_BUFFER, loom_buffer_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_VIEW, loom_view_dialect_vtables);
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

  iree_status_t RunPipeline(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t parse_options = {};
    iree_status_t status = loom_text_parse(
        iree_make_cstring_view(source), IREE_SV("vector_memory_footprint.loom"),
        &context_, &block_pool_, &parse_options, &module);
    if (!iree_status_is_ok(status)) {
      if (module) loom_module_free(module);
      return status;
    }
    if (!module) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "parser did not produce a module");
    }

    iree_arena_allocator_t instance_arena;
    iree_arena_initialize(&block_pool_, &instance_arena);
    iree_arena_allocator_t function_arena;
    iree_arena_initialize(&block_pool_, &function_arena);
    loom_pass_value_fact_owner_t value_facts = {};
    loom_pass_value_fact_owner_initialize(&block_pool_, &value_facts);

    const loom_pass_info_t* info = loom_vector_memory_footprint_pass_info();
    loom_pass_t pass = {};
    pass.info = info;
    pass.function_run = loom_vector_memory_footprint_run;
    pass.instance_arena = &instance_arena;
    pass.arena = &function_arena;
    pass.value_facts = &value_facts;
    if (info->statistic_count > 0) {
      iree_host_size_t statistics_size =
          (iree_host_size_t)info->statistic_count * sizeof(*pass.statistics);
      status = iree_arena_allocate(&instance_arena, statistics_size,
                                   (void**)&pass.statistics);
      if (iree_status_is_ok(status)) {
        memset(pass.statistics, 0, statistics_size);
      }
    }
    if (iree_status_is_ok(status)) {
      for (iree_host_size_t i = 0;
           i < module->symbols.count && iree_status_is_ok(status); ++i) {
        loom_symbol_t* symbol = &module->symbols.entries[i];
        if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
          continue;
        }
        loom_func_like_t function =
            loom_func_like_cast(module, symbol->defining_op);
        if (!loom_func_like_body(function)) {
          continue;
        }
        iree_arena_reset(&function_arena);
        status = loom_vector_memory_footprint_run(&pass, module, function);
      }
    }
    pass.arena = pass.instance_arena;
    loom_pass_value_fact_owner_deinitialize(&value_facts);
    iree_arena_deinitialize(&function_arena);
    iree_arena_deinitialize(&instance_arena);
    loom_module_free(module);
    return status;
  }

  iree_status_t RunSingleFunction(const char* source,
                                  DiagnosticEmissionCollector* collector) {
    loom_module_t* module = NULL;
    loom_text_parse_options_t parse_options = {};
    iree_status_t status = loom_text_parse(
        iree_make_cstring_view(source), IREE_SV("vector_memory_footprint.loom"),
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
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "module does not contain a function body");
      loom_module_free(module);
      return status;
    }

    iree_arena_allocator_t pass_arena;
    iree_arena_initialize(&block_pool_, &pass_arena);
    loom_pass_value_fact_owner_t value_facts = {};
    loom_pass_value_fact_owner_initialize(&block_pool_, &value_facts);
    loom_pass_t pass = {};
    pass.info = loom_vector_memory_footprint_pass_info();
    pass.instance_arena = &pass_arena;
    pass.arena = &pass_arena;
    pass.diagnostic_emitter.fn = CollectDiagnosticEmission;
    pass.diagnostic_emitter.user_data = collector;
    pass.value_facts = &value_facts;
    status = loom_vector_memory_footprint_run(&pass, module, function);
    loom_pass_value_fact_owner_deinitialize(&value_facts);
    iree_arena_deinitialize(&pass_arena);
    loom_module_free(module);
    return status;
  }

  // Block pool shared by parser, module allocation, and pass execution arenas.
  iree_arena_block_pool_t block_pool_;

  // Context with the full Loom dialect set needed by production text IR.
  loom_context_t context_;
};

TEST_F(VectorMemoryFootprintTest, StaticDenseLoadInBounds) {
  const char* source = R"(
func.def @test(%buffer: buffer, %base: offset) {
  %layout = encoding.layout.dense : encoding<layout>
  %view = buffer.view %buffer[%base] : buffer -> view<8xf32, %layout>
  %loaded = vector.load %view[4] : view<8xf32, %layout> -> vector<4xf32>
  func.return
}
)";

  IREE_EXPECT_OK(RunPipeline(source));
}

TEST_F(VectorMemoryFootprintTest, StaticDenseStoreInBounds) {
  const char* source = R"(
func.def @test(%buffer: buffer, %base: offset, %value: vector<4xf32>) {
  %layout = encoding.layout.dense : encoding<layout>
  %view = buffer.view %buffer[%base] : buffer -> view<8xf32, %layout>
  vector.store %value, %view[4] : vector<4xf32>, view<8xf32, %layout>
  func.return
}
)";

  IREE_EXPECT_OK(RunPipeline(source));
}

TEST_F(VectorMemoryFootprintTest, StaticDenseLoadOutOfBoundsFails) {
  const char* source = R"(
func.def @test(%buffer: buffer, %base: offset) {
  %layout = encoding.layout.dense : encoding<layout>
  %view = buffer.view %buffer[%base] : buffer -> view<8xf32, %layout>
  %loaded = vector.load %view[5] : view<8xf32, %layout> -> vector<4xf32>
  func.return
}
)";

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, RunPipeline(source));
}

TEST_F(VectorMemoryFootprintTest, FailureEmitsStructuredDiagnostic) {
  const char* source = R"(
func.def @test(%buffer: buffer, %base: offset) {
  %layout = encoding.layout.dense : encoding<layout>
  %view = buffer.view %buffer[%base] : buffer -> view<8xf32, %layout>
  %loaded = vector.load %view[5] : view<8xf32, %layout> -> vector<4xf32>
  func.return
}
)";

  DiagnosticEmissionCollector collector;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        RunSingleFunction(source, &collector));
  EXPECT_EQ(collector.count, 1);
  EXPECT_EQ(collector.error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_SUBRANGE, 5));
  EXPECT_EQ(collector.view_axis, 0);
  EXPECT_EQ(collector.vector_axis, 0);
  EXPECT_EQ(collector.origin, "5");
  EXPECT_EQ(collector.reason, "upper bound proof failed");
  EXPECT_EQ(collector.required, "origin + extent <= bound");
}

TEST_F(VectorMemoryFootprintTest, DynamicFullVectorAtZeroIsProven) {
  const char* source = R"(
func.def @test(%buffer: buffer, %base: offset, %n: index) {
  %layout = encoding.layout.dense : encoding<layout>
  %c0 = index.constant 0 : index
  %view = buffer.view %buffer[%base] : buffer -> view<[%n]xf32, %layout>
  %loaded = vector.load %view[%c0] : view<[%n]xf32, %layout> -> vector<[%n]xf32>
  func.return
}
)";

  IREE_EXPECT_OK(RunPipeline(source));
}

TEST_F(VectorMemoryFootprintTest, DynamicUnprovenTailFails) {
  const char* source = R"(
func.def @test(%buffer: buffer, %base: offset, %n: index, %iv: index) {
  %layout = encoding.layout.dense : encoding<layout>
  %iv2 = index.assume %iv [range(%iv, 0, 4096)] : index
  %view = buffer.view %buffer[%base] : buffer -> view<[%n]xf32, %layout>
  %loaded = vector.load %view[%iv2] : view<[%n]xf32, %layout> -> vector<16xf32>
  func.return
}
)";

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, RunPipeline(source));
}

TEST_F(VectorMemoryFootprintTest, MaskRangeTailUsesUpperBound) {
  const char* source = R"(
func.def @test(%buffer: buffer, %base: offset, %n: index, %iv: index, %old: vector<16xf32>) {
  %layout = encoding.layout.dense : encoding<layout>
  %c1 = index.constant 1 : index
  %iv2 = index.assume %iv [range(%iv, 0, 4096)] : index
  %view = buffer.view %buffer[%base] : buffer -> view<[%n]xf32, %layout>
  %mask = vector.mask.range [%iv2 to %n step %c1] : index -> vector<16xi1>
  %loaded = vector.load.mask %view[%iv2], %mask, %old : view<[%n]xf32, %layout>, vector<16xi1>, vector<16xf32>
  func.return
}
)";

  IREE_EXPECT_OK(RunPipeline(source));
}

TEST_F(VectorMemoryFootprintTest, MaskedStoreTailUsesUpperBound) {
  const char* source = R"(
func.def @test(%buffer: buffer, %base: offset, %n: index, %iv: index, %value: vector<16xf32>) {
  %layout = encoding.layout.dense : encoding<layout>
  %c1 = index.constant 1 : index
  %iv2 = index.assume %iv [range(%iv, 0, 4096)] : index
  %view = buffer.view %buffer[%base] : buffer -> view<[%n]xf32, %layout>
  %mask = vector.mask.range [%iv2 to %n step %c1] : index -> vector<16xi1>
  vector.store.mask %value, %view[%iv2], %mask : vector<16xf32>, view<[%n]xf32, %layout>, vector<16xi1>
  func.return
}
)";

  IREE_EXPECT_OK(RunPipeline(source));
}

TEST_F(VectorMemoryFootprintTest, ExpandCompressTailUsesUpperBound) {
  const char* source = R"(
func.def @test(%buffer: buffer, %base: offset, %n: index, %iv: index, %old: vector<16xf32>, %value: vector<16xf32>) {
  %layout = encoding.layout.dense : encoding<layout>
  %c1 = index.constant 1 : index
  %iv2 = index.assume %iv [range(%iv, 0, 4096)] : index
  %view = buffer.view %buffer[%base] : buffer -> view<[%n]xf32, %layout>
  %mask = vector.mask.range [%iv2 to %n step %c1] : index -> vector<16xi1>
  %loaded = vector.load.expand %view[%iv2], %mask, %old : view<[%n]xf32, %layout>, vector<16xi1>, vector<16xf32>
  vector.store.compress %value, %view[%iv2], %mask : vector<16xf32>, view<[%n]xf32, %layout>, vector<16xi1>
  func.return
}
)";

  IREE_EXPECT_OK(RunPipeline(source));
}

TEST_F(VectorMemoryFootprintTest, IotaGatherInBounds) {
  const char* source = R"(
func.def @test(%buffer: buffer, %base: offset) {
  %layout = encoding.layout.dense : encoding<layout>
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %view = buffer.view %buffer[%base] : buffer -> view<4xf32, %layout>
  %offsets = vector.iota %c0 step %c1 : vector<4xindex>
  %loaded = vector.gather %view[%c0][%offsets] : view<4xf32, %layout>, vector<4xindex> -> vector<4xf32>
  func.return
}
)";

  IREE_EXPECT_OK(RunPipeline(source));
}

TEST_F(VectorMemoryFootprintTest, IotaScatterInBounds) {
  const char* source = R"(
func.def @test(%buffer: buffer, %base: offset, %value: vector<4xf32>) {
  %layout = encoding.layout.dense : encoding<layout>
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %view = buffer.view %buffer[%base] : buffer -> view<4xf32, %layout>
  %offsets = vector.iota %c0 step %c1 : vector<4xindex>
  vector.scatter %value, %view[%c0][%offsets] : vector<4xf32>, view<4xf32, %layout>, vector<4xindex>
  func.return
}
)";

  IREE_EXPECT_OK(RunPipeline(source));
}

TEST_F(VectorMemoryFootprintTest, IotaGatherRankTwoCrossesDenseRowInBounds) {
  const char* source = R"(
func.def @test(%buffer: buffer, %base: offset) {
  %layout = encoding.layout.dense : encoding<layout>
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %view = buffer.view %buffer[%base] : buffer -> view<2x4xf32, %layout>
  %offsets = vector.iota %c0 step %c1 : vector<2xindex>
  %loaded = vector.gather %view[0, 3][%offsets] : view<2x4xf32, %layout>, vector<2xindex> -> vector<2xf32>
  func.return
}
)";

  IREE_EXPECT_OK(RunPipeline(source));
}

TEST_F(VectorMemoryFootprintTest, IotaGatherRankTwoLinearOutOfBoundsFails) {
  const char* source = R"(
func.def @test(%buffer: buffer, %base: offset) {
  %layout = encoding.layout.dense : encoding<layout>
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %view = buffer.view %buffer[%base] : buffer -> view<2x4xf32, %layout>
  %offsets = vector.iota %c0 step %c1 : vector<2xindex>
  %loaded = vector.gather %view[1, 3][%offsets] : view<2x4xf32, %layout>, vector<2xindex> -> vector<2xf32>
  func.return
}
)";

  DiagnosticEmissionCollector collector;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        RunSingleFunction(source, &collector));
  EXPECT_EQ(collector.count, 1);
  EXPECT_EQ(collector.error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_SUBRANGE, 6));
  EXPECT_EQ(collector.origin, "<linearized origin>");
  EXPECT_EQ(collector.reason, "upper bound proof failed");
  EXPECT_EQ(collector.required,
            "linearized_origin + max(offsets) + 1 <= view storage span");
}

TEST_F(VectorMemoryFootprintTest, IotaGatherOutOfBoundsFails) {
  const char* source = R"(
func.def @test(%buffer: buffer, %base: offset) {
  %layout = encoding.layout.dense : encoding<layout>
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %view = buffer.view %buffer[%base] : buffer -> view<4xf32, %layout>
  %offsets = vector.iota %c1 step %c1 : vector<4xindex>
  %loaded = vector.gather %view[%c0][%offsets] : view<4xf32, %layout>, vector<4xindex> -> vector<4xf32>
  func.return
}
)";

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, RunPipeline(source));
}

TEST_F(VectorMemoryFootprintTest, AtomicReduceFootprintIsChecked) {
  const char* source = R"(
func.def @test(%buffer: buffer, %base: offset, %value: vector<4xi32>) {
  %layout = encoding.layout.dense : encoding<layout>
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %view = buffer.view %buffer[%base] : buffer -> view<4xi32, %layout>
  %offsets = vector.iota %c0 step %c1 : vector<4xindex>
  vector.atomic.reduce<addi> %value, %view[%c0][%offsets] {ordering = relaxed, scope = workgroup} : vector<4xi32>, view<4xi32, %layout>, vector<4xindex>
  func.return
}
)";

  IREE_EXPECT_OK(RunPipeline(source));
}

TEST_F(VectorMemoryFootprintTest, AtomicRmwFootprintIsChecked) {
  const char* source = R"(
func.def @test(%buffer: buffer, %base: offset, %value: vector<4xi32>) -> (vector<4xi32>) {
  %layout = encoding.layout.dense : encoding<layout>
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %view = buffer.view %buffer[%base] : buffer -> view<4xi32, %layout>
  %offsets = vector.iota %c0 step %c1 : vector<4xindex>
  %old = vector.atomic.rmw<xchgi> %value, %view[%c0][%offsets] {ordering = acq_rel, scope = workgroup} : vector<4xi32>, view<4xi32, %layout>, vector<4xindex>
  func.return %old : vector<4xi32>
}
)";

  IREE_EXPECT_OK(RunPipeline(source));
}

TEST_F(VectorMemoryFootprintTest, MaskedOffsetOpsUniformFalseSkip) {
  const char* source = R"(
func.def @test(%buffer: buffer, %base: offset, %value: vector<4xi32>) {
  %layout = encoding.layout.dense : encoding<layout>
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %mask = vector.constant false : vector<4xi1>
  %old = vector.constant 0 : vector<4xi32>
  %view = buffer.view %buffer[%base] : buffer -> view<4xi32, %layout>
  %offsets = vector.iota %c1 step %c1 : vector<4xindex>
  %gathered = vector.gather.mask %view[%c0][%offsets], %mask, %old : view<4xi32, %layout>, vector<4xindex>, vector<4xi1>, vector<4xi32>
  vector.scatter.mask %value, %view[%c0][%offsets], %mask : vector<4xi32>, view<4xi32, %layout>, vector<4xindex>, vector<4xi1>
  vector.atomic.reduce.mask<addi> %value, %view[%c0][%offsets], %mask {ordering = relaxed, scope = workgroup} : vector<4xi32>, view<4xi32, %layout>, vector<4xindex>, vector<4xi1>
  %rmw = vector.atomic.rmw.mask<xchgi> %value, %view[%c0][%offsets], %mask, %old {ordering = acq_rel, scope = workgroup} : vector<4xi32>, view<4xi32, %layout>, vector<4xindex>, vector<4xi1>, vector<4xi32>
  func.return
}
)";

  IREE_EXPECT_OK(RunPipeline(source));
}

TEST_F(VectorMemoryFootprintTest, UnresolvedLayoutFails) {
  const char* source = R"(
func.def @test(%buffer: buffer, %base: offset, %layout: encoding<layout>) {
  %view = buffer.view %buffer[%base] : buffer -> view<4xf32, %layout>
  %loaded = vector.load %view[0] : view<4xf32, %layout> -> vector<4xf32>
  func.return
}
)";

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, RunPipeline(source));
}

}  // namespace
}  // namespace loom
