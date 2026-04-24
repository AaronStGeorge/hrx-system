// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks generated source-to-low lowering through the synthetic test-low
// target.
//
// The measured path generates a deterministic source module, verifies it,
// selects the target profile, lowers to low.func.def, low-verifies, packetizes,
// schedules, and allocates. This is intentionally a pipeline-level throughput
// signal for generated vector/scalar source bodies; narrower microbenchmarks
// should sit below this helper when perf points at a specific stage.

#include <cstdint>

#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/source_workload.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/lower.h"

namespace {

class SourceLoweringBenchmark {
 public:
  SourceLoweringBenchmark() {
    iree_arena_block_pool_initialize(65536, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_CHECK_OK(loom_low_source_workload_register_dialects(&context_));
    IREE_CHECK_OK(loom_context_finalize(&context_));
    loom_test_low_descriptor_registry_initialize(&descriptor_registry_);
    loom_test_low_lower_policy_registry_initialize(&policy_registry_);
  }

  SourceLoweringBenchmark(const SourceLoweringBenchmark&) = delete;
  SourceLoweringBenchmark& operator=(const SourceLoweringBenchmark&) = delete;

  ~SourceLoweringBenchmark() {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t Run(uint64_t seed, uint32_t scale,
                    loom_low_source_workload_pipeline_counters_t* counters) {
    loom_module_t* module = nullptr;
    loom_low_source_workload_config_t workload_config =
        loom_low_source_workload_config_make(IREE_SV("test-low"), scale);
    iree_status_t status = loom_low_source_workload_generate_seeded_module(
        seed, &workload_config, &context_, &block_pool_, &module);
    if (iree_status_is_ok(status)) {
      const loom_low_source_workload_pipeline_options_t pipeline_options = {
          .descriptor_registry = &descriptor_registry_.registry,
          .policy_registry = &policy_registry_,
          .descriptor_requirements =
              LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
          .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE,
      };
      status = loom_low_source_workload_run_pipeline(module, &pipeline_options,
                                                     &block_pool_, counters);
    }
    if (module) {
      loom_module_free(module);
    }
    return status;
  }

 private:
  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t descriptor_registry_ = {};
  loom_low_lower_policy_registry_t policy_registry_ = {};
};

static uint64_t SourceOpCount(
    const loom_low_source_workload_pipeline_counters_t& counters) {
  return counters.source_counts.scalar_integer_op_count +
         counters.source_counts.scalar_constant_count +
         counters.source_counts.vector_integer_op_count +
         counters.source_counts.vector_reduce_op_count +
         counters.source_counts.vector_extract_op_count +
         counters.source_counts.vector_shuffle_op_count +
         counters.source_counts.vector_load_op_count +
         counters.source_counts.vector_store_op_count +
         counters.source_counts.index_madd_op_count;
}

static double Average(uint64_t total, int64_t iterations) {
  return iterations == 0
             ? 0.0
             : static_cast<double>(total) / static_cast<double>(iterations);
}

static void BM_SourceLowPipeline(benchmark::State& state) {
  const uint32_t scale = static_cast<uint32_t>(state.range(0));
  SourceLoweringBenchmark benchmark_fixture;
  uint64_t seed = 0;
  uint64_t total_source_ops = 0;
  uint64_t total_low_packets = 0;
  uint64_t total_schedule_nodes = 0;
  uint64_t total_dependencies = 0;
  uint64_t total_assignments = 0;
  uint64_t total_module_bytes = 0;
  uint64_t total_lowering_bytes = 0;
  uint64_t total_packet_bytes = 0;

  for (auto _ : state) {
    loom_low_source_workload_pipeline_counters_t counters = {};
    iree_status_t status = benchmark_fixture.Run(seed++, scale, &counters);
    if (!iree_status_is_ok(status)) {
      iree_status_abort(status);
    }
    total_source_ops += SourceOpCount(counters);
    total_low_packets += counters.low_descriptor_op_count;
    total_schedule_nodes += counters.schedule_node_count;
    total_dependencies += counters.schedule_dependency_count;
    total_assignments += counters.allocation_assignment_count;
    total_module_bytes += counters.module_arena_used_bytes;
    total_lowering_bytes += counters.lowering_arena_used_bytes;
    total_packet_bytes += counters.packet_arena_used_bytes;
    benchmark::DoNotOptimize(counters.low_descriptor_op_count);
  }

  const int64_t iterations = state.iterations();
  state.SetItemsProcessed(static_cast<int64_t>(total_source_ops));
  state.counters["source_ops"] = Average(total_source_ops, iterations);
  state.counters["low_packets"] = Average(total_low_packets, iterations);
  state.counters["schedule_nodes"] = Average(total_schedule_nodes, iterations);
  state.counters["dependencies"] = Average(total_dependencies, iterations);
  state.counters["assignments"] = Average(total_assignments, iterations);
  state.counters["module_bytes"] = Average(total_module_bytes, iterations);
  state.counters["lowering_bytes"] = Average(total_lowering_bytes, iterations);
  state.counters["packet_bytes"] = Average(total_packet_bytes, iterations);
}
BENCHMARK(BM_SourceLowPipeline)->Arg(1)->Arg(2)->Arg(4)->Arg(8);

}  // namespace
