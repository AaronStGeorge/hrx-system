// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/type_registry.h"

namespace {

enum class ChannelAccessKind { Channel, Read, Write };

class TypeInternBenchmark {
 public:
  TypeInternBenchmark() {
    iree_arena_block_pool_initialize(65536, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_CHECK_OK(loom_context_finalize(&context_));
    IREE_CHECK_OK(loom_module_allocate(&context_, IREE_SV("type_intern"),
                                       &block_pool_, nullptr,
                                       iree_allocator_system(), &module_));
  }

  ~TypeInternBenchmark() {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_type_t Intern(loom_type_t type) {
    loom_type_t interned_type = {};
    IREE_CHECK_OK(loom_module_intern_type(module_, type, &interned_type));
    return interned_type;
  }

  void PreparePayloads(iree_host_size_t count) {
    loom_module_free(module_);
    module_ = nullptr;
    IREE_CHECK_OK(loom_module_allocate(&context_, IREE_SV("type_intern"),
                                       &block_pool_, nullptr,
                                       iree_allocator_system(), &module_));
    payloads_.resize(count);
    for (iree_host_size_t i = 0; i < count; ++i) {
      const loom_type_t payload =
          loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F16,
                              loom_dim_pack_static((int64_t)(16 + i)), 0);
      IREE_CHECK_OK(
          loom_module_intern_type_id(module_, payload, &payloads_[i]));
    }
  }

  template <ChannelAccessKind Family>
  loom_type_t Make(iree_host_size_t index) {
    loom_type_t type = {};
    if constexpr (Family == ChannelAccessKind::Channel) {
      IREE_CHECK_OK(loom_channel_type_make(module_, payloads_[index], &type));
    } else if constexpr (Family == ChannelAccessKind::Read) {
      IREE_CHECK_OK(loom_read_type_make(
          module_, LOOM_READ_TYPE_BUILD_FLAG_HAS_MODE, payloads_[index],
          LOOM_READ_TYPE_MODE_MUTABLE, &type));
    } else {
      IREE_CHECK_OK(loom_write_type_make(module_, payloads_[index], &type));
    }
    return type;
  }

 private:
  // Recycled allocation blocks shared by successive benchmark modules.
  iree_arena_block_pool_t block_pool_;
  // Descriptor owner retained across module resets.
  loom_context_t context_;
  // Current module owns interned payload and channel types.
  loom_module_t* module_ = nullptr;
  // Prepared child types exclude their construction from the timed operation.
  std::vector<loom_type_id_t> payloads_;
};

static void BM_InternRepeatedScalar(benchmark::State& state) {
  TypeInternBenchmark fixture;
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  fixture.Intern(type);
  for (auto _ : state) {
    benchmark::DoNotOptimize(fixture.Intern(type));
  }
}
BENCHMARK(BM_InternRepeatedScalar);

static void BM_InternAlternatingScalars(benchmark::State& state) {
  TypeInternBenchmark fixture;
  const loom_type_t types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_F16),
      loom_type_scalar(LOOM_SCALAR_TYPE_F32),
  };
  fixture.Intern(types[0]);
  fixture.Intern(types[1]);
  iree_host_size_t type_index = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(fixture.Intern(types[type_index]));
    type_index ^= 1;
  }
}
BENCHMARK(BM_InternAlternatingScalars);

static void BM_InternRepeatedStaticVector(benchmark::State& state) {
  TypeInternBenchmark fixture;
  const loom_type_t type = loom_type_shaped_2d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F16, loom_dim_pack_static(16),
      loom_dim_pack_static(16), /*encoding_id=*/0);
  fixture.Intern(type);
  for (auto _ : state) {
    benchmark::DoNotOptimize(fixture.Intern(type));
  }
}
BENCHMARK(BM_InternRepeatedStaticVector);

template <ChannelAccessKind Family>
static void BM_MakeExistingParameterizedType(benchmark::State& state) {
  TypeInternBenchmark fixture;
  const auto count = (iree_host_size_t)state.range(0);
  fixture.PreparePayloads(count);
  for (iree_host_size_t i = 0; i < count; ++i) {
    fixture.Make<Family>(i);
  }
  iree_host_size_t index = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(fixture.Make<Family>(index));
    if (++index == count) {
      index = 0;
    }
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK_TEMPLATE(BM_MakeExistingParameterizedType, ChannelAccessKind::Channel)
    ->Arg(1)
    ->Arg(64);
BENCHMARK_TEMPLATE(BM_MakeExistingParameterizedType, ChannelAccessKind::Read)
    ->Arg(1)
    ->Arg(64);
BENCHMARK_TEMPLATE(BM_MakeExistingParameterizedType, ChannelAccessKind::Write)
    ->Arg(1)
    ->Arg(64);

template <ChannelAccessKind Family>
static void BM_MakeNewParameterizedTypes(benchmark::State& state) {
  TypeInternBenchmark fixture;
  const auto count = (iree_host_size_t)state.range(0);
  for (auto _ : state) {
    state.PauseTiming();
    fixture.PreparePayloads(count);
    state.ResumeTiming();
    for (iree_host_size_t i = 0; i < count; ++i) {
      benchmark::DoNotOptimize(fixture.Make<Family>(i));
    }
  }
  state.SetItemsProcessed(state.iterations() * count);
}
BENCHMARK_TEMPLATE(BM_MakeNewParameterizedTypes, ChannelAccessKind::Channel)
    ->Arg(64)
    ->Arg(1024);
BENCHMARK_TEMPLATE(BM_MakeNewParameterizedTypes, ChannelAccessKind::Read)
    ->Arg(64)
    ->Arg(1024);
BENCHMARK_TEMPLATE(BM_MakeNewParameterizedTypes, ChannelAccessKind::Write)
    ->Arg(64)
    ->Arg(1024);

}  // namespace
