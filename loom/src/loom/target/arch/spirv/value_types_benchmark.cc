// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/target/arch/spirv/ops/types.h"

namespace {

class CooperativeMatrixBenchmark {
 public:
  CooperativeMatrixBenchmark() {
    iree_arena_block_pool_initialize(65536, iree_allocator_system(), &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_CHECK_OK(loom_context_register_parameterized_types(
        &context_, LOOM_DIALECT_SPIRV,
        loom_spirv_type_registry_parameterized_descriptors,
        loom_spirv_type_registry_parameterized_count));
    IREE_CHECK_OK(loom_type_registry_register_types(
        &context_, loom_spirv_type_registry_entries,
        loom_spirv_type_registry_entry_count));
    IREE_CHECK_OK(loom_context_finalize(&context_));
    ResetModule();
  }

  ~CooperativeMatrixBenchmark() {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void ResetModule() {
    loom_module_free(module_);
    module_ = nullptr;
    IREE_CHECK_OK(loom_module_allocate(&context_, IREE_SV("matrix_types"),
                                       &pool_, nullptr, iree_allocator_system(),
                                       &module_));
  }

  loom_type_t Make(int64_t columns) {
    loom_type_t type = {};
    IREE_CHECK_OK(loom_spirv_cooperative_matrix_type_make(
        module_, 16, columns, LOOM_SPIRV_SCALAR_TYPE_F16,
        LOOM_SPIRV_SCOPE_SUBGROUP,
        LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MATRIX_AKHR, &type));
    return type;
  }

 private:
  // Recycled allocation blocks shared by successive benchmark modules.
  iree_arena_block_pool_t pool_;
  // Registered SPIR-V type descriptors outlive each module.
  loom_context_t context_;
  // Current module owns interned cooperative-matrix types.
  loom_module_t* module_ = nullptr;
};

static void BM_MakeExistingCooperativeMatrix(benchmark::State& state) {
  CooperativeMatrixBenchmark fixture;
  const int64_t count = state.range(0);
  for (int64_t i = 0; i < count; ++i) {
    fixture.Make(16 + i);
  }
  int64_t index = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(fixture.Make(16 + index));
    if (++index == count) {
      index = 0;
    }
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_MakeExistingCooperativeMatrix)->Arg(1)->Arg(64);

static void BM_MakeNewCooperativeMatrices(benchmark::State& state) {
  CooperativeMatrixBenchmark fixture;
  const int64_t count = state.range(0);
  for (auto _ : state) {
    state.PauseTiming();
    fixture.ResetModule();
    state.ResumeTiming();
    for (int64_t i = 0; i < count; ++i) {
      benchmark::DoNotOptimize(fixture.Make(16 + i));
    }
  }
  state.SetItemsProcessed(state.iterations() * count);
}
BENCHMARK(BM_MakeNewCooperativeMatrices)->Arg(64)->Arg(1024);

}  // namespace
