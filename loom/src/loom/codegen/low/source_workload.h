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

#ifndef LOOM_CODEGEN_LOW_SOURCE_WORKLOAD_H_
#define LOOM_CODEGEN_LOW_SOURCE_WORKLOAD_H_

#include "iree/base/api.h"
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
  // Number of generated index.madd ops.
  uint32_t index_madd_op_count;
} loom_low_source_workload_counts_t;

// Returns a generated source-to-low workload config using |target_preset|.
// Scale 1 produces a compact body; larger scales increase only the op count.
loom_low_source_workload_config_t loom_low_source_workload_config_make(
    iree_string_view_t target_preset, uint32_t scale);

// Generates one targeted source function suitable for source-to-low lowering.
//
// The generated function has i32, vector<4xi32>, and index arguments/results
// and emits only source ops intentionally supported by the foundation lowering
// path. |out_func_ref| receives the source function symbol so callers can lower
// that exact function without text lookups.
iree_status_t loom_low_source_workload_generate_module(
    loom_test_gen_t* gen, const loom_low_source_workload_config_t* config,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    loom_module_t** out_module, loom_symbol_ref_t* out_func_ref);

// Generates one targeted source function using a deterministic PRNG seed.
iree_status_t loom_low_source_workload_generate_seeded_module(
    uint64_t seed, const loom_low_source_workload_config_t* config,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    loom_module_t** out_module, loom_symbol_ref_t* out_func_ref);

// Counts source ops emitted by loom_low_source_workload_generate_module().
void loom_low_source_workload_count_func_ops(
    const loom_op_t* func_op, loom_low_source_workload_counts_t* out_counts);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SOURCE_WORKLOAD_H_
