// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generated source workloads for source-to-low tests, fuzzers, and benchmarks.
//
// This is target-agnostic test infrastructure: it builds ordinary Loom source
// functions with a caller-selected target.profile preset, but it does not link
// or query any concrete target descriptor registry. Consumers choose a target
// provider separately when they lower or execute the generated module.

#ifndef LOOM_CODEGEN_LOW_TESTING_SOURCE_WORKLOAD_H_
#define LOOM_CODEGEN_LOW_TESTING_SOURCE_WORKLOAD_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/lower.h"
#include "loom/codegen/low/packetization.h"
#include "loom/ir/ir.h"
#include "loom/testing/gen.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_arena_block_pool_t iree_arena_block_pool_t;
typedef struct loom_context_t loom_context_t;
typedef struct loom_module_t loom_module_t;

typedef struct loom_low_source_workload_config_t {
  // Module name assigned to the generated compilation unit. Empty uses a stable
  // default.
  iree_string_view_t module_name;
  // Symbol name for the generated target.profile. Empty uses a stable default.
  iree_string_view_t target_symbol_name;
  // Preset key written into target.profile. This is required.
  iree_string_view_t target_preset;
  // Symbol name for the generated source function. Empty uses a stable default.
  iree_string_view_t function_symbol_name;
  // Number of source ops to emit in the generated function body.
  uint16_t op_count;
} loom_low_source_workload_config_t;

typedef struct loom_low_source_workload_counts_t {
  // Number of generated scalar integer arithmetic ops.
  uint32_t scalar_integer_op_count;
  // Number of generated scalar constants.
  uint32_t scalar_constant_count;
  // Number of generated lanewise vector integer arithmetic ops.
  uint32_t vector_integer_op_count;
  // Number of generated vector.reduce ops.
  uint32_t vector_reduce_op_count;
  // Number of generated vector dot ops.
  uint32_t vector_dot_op_count;
  // Number of generated vector.extract ops.
  uint32_t vector_extract_op_count;
  // Number of generated vector.shuffle ops.
  uint32_t vector_shuffle_op_count;
  // Number of generated vector.load ops.
  uint32_t vector_load_op_count;
  // Number of generated vector.store ops.
  uint32_t vector_store_op_count;
  // Number of generated index.madd ops.
  uint32_t index_madd_op_count;
} loom_low_source_workload_counts_t;

typedef struct loom_low_source_workload_pipeline_options_t {
  // Low descriptor registry linked into the caller.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Target lowering policies linked into the caller.
  const loom_low_lower_policy_registry_t* policy_registry;
  // Descriptor payload requirements needed after lowering.
  loom_low_descriptor_requirement_flags_t descriptor_requirements;
  // Candidate selection strategy used by packetization.
  loom_low_schedule_strategy_t schedule_strategy;
} loom_low_source_workload_pipeline_options_t;

typedef struct loom_low_source_workload_pipeline_counters_t {
  // Source op category counts before lowering.
  loom_low_source_workload_counts_t source_counts;
  // Number of low.op and low.const packets emitted by lowering.
  uint32_t low_descriptor_op_count;
  // Number of source-to-low error diagnostics emitted by lowering.
  uint32_t lower_error_count;
  // Number of source-to-low remark diagnostics emitted by lowering.
  uint32_t lower_remark_count;
  // Number of schedule nodes produced by packetization.
  iree_host_size_t schedule_node_count;
  // Number of schedule dependency edges produced by packetization.
  iree_host_size_t schedule_dependency_count;
  // Number of descriptor resource-use rows produced by packetization.
  iree_host_size_t schedule_resource_use_count;
  // Number of schedule hazard-gap rows produced by packetization.
  iree_host_size_t schedule_hazard_gap_count;
  // Number of register-allocation assignments produced by packetization.
  iree_host_size_t allocation_assignment_count;
  // Number of assignments placed in spill slots.
  iree_host_size_t allocation_spill_count;
  // Number of low.copy ops coalesced by allocation.
  iree_host_size_t allocation_coalesced_copy_count;
  // Number of low.copy ops left materialized by allocation.
  iree_host_size_t allocation_materialized_copy_count;
  // Module arena bytes used after generation and lowering.
  iree_host_size_t module_arena_used_bytes;
  // Module arena bytes reserved after generation and lowering.
  iree_host_size_t module_arena_allocated_bytes;
  // Lowering scratch arena bytes used before cleanup.
  iree_host_size_t lowering_arena_used_bytes;
  // Packetization scratch arena bytes used before cleanup.
  iree_host_size_t packet_arena_used_bytes;
} loom_low_source_workload_pipeline_counters_t;

// Returns a generated source-to-low workload config using |target_preset|.
// Scale 1 produces a compact body; larger scales increase only the op count.
loom_low_source_workload_config_t loom_low_source_workload_config_make(
    iree_string_view_t target_preset, uint32_t scale);

// Registers the exact source and low dialects used by generated source-low
// workloads. Callers own context initialization and finalization so they can
// compose this with target-owned descriptor/policy registries without pulling
// the full production op registry into fuzzers and benchmarks.
iree_status_t loom_low_source_workload_register_dialects(
    loom_context_t* context);

// Generates one targeted source function suitable for source-to-low lowering.
//
// The generated function has buffer, i32, vector<4xi32>, vector<16xi8>, and
// index arguments/results and emits only source ops intentionally supported by
// the foundation lowering path.
iree_status_t loom_low_source_workload_generate_module(
    loom_test_gen_t* gen, const loom_low_source_workload_config_t* config,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    loom_module_t** out_module);

// Generates one targeted source function using a deterministic PRNG seed.
iree_status_t loom_low_source_workload_generate_seeded_module(
    uint64_t seed, const loom_low_source_workload_config_t* config,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    loom_module_t** out_module);

// Counts source ops emitted by loom_low_source_workload_generate_module().
void loom_low_source_workload_count_func_ops(
    const loom_op_t* func_op, loom_low_source_workload_counts_t* out_counts);

// Adds |source_counts| into |target_counts|.
void loom_low_source_workload_counts_accumulate(
    loom_low_source_workload_counts_t* target_counts,
    const loom_low_source_workload_counts_t* source_counts);

// Returns the total number of counted source ops.
uint64_t loom_low_source_workload_counts_total(
    const loom_low_source_workload_counts_t* counts);

// Runs the generated workload through source verification, source-to-low
// lowering, low verification, packetization, scheduling, and allocation.
iree_status_t loom_low_source_workload_run_pipeline(
    loom_module_t* module,
    const loom_low_source_workload_pipeline_options_t* options,
    iree_arena_block_pool_t* block_pool,
    loom_low_source_workload_pipeline_counters_t* out_counters);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_TESTING_SOURCE_WORKLOAD_H_
