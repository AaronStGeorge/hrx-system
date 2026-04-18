// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent scheduler sidecar for target-low functions.
//
// This layer consumes ordinary Loom IR plus descriptor tables and produces a
// deterministic schedule sidecar. The first scheduler is intentionally
// conservative: it builds the dependency graph and records a source-priority
// topological order without mutating IR. Target hazard insertion, allocation,
// and pressure-sensitive scoring plug into this sidecar instead of creating a
// second low-level IR container.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sentinel for absent schedule node indices.
#define LOOM_LOW_SCHEDULE_NODE_NONE UINT32_MAX

typedef enum loom_low_schedule_node_kind_e {
  // Ordinary structural low op such as low.copy, low.spill, or low.reload.
  LOOM_LOW_SCHEDULE_NODE_STRUCTURAL = 0,
  // Descriptor-backed packet such as low.op or low.const.
  LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR = 1,
  // Block terminator kept fixed after all schedulable block contents.
  LOOM_LOW_SCHEDULE_NODE_TERMINATOR = 2,
} loom_low_schedule_node_kind_t;

typedef enum loom_low_schedule_dependency_kind_e {
  // Unknown or uninitialized dependency kind.
  LOOM_LOW_SCHEDULE_DEPENDENCY_UNKNOWN = 0,
  // SSA producer-to-consumer dependency.
  LOOM_LOW_SCHEDULE_DEPENDENCY_SSA = 1,
  // Conservative side-effect ordering dependency.
  LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT = 2,
  // Block-control dependency keeping terminators after block contents.
  LOOM_LOW_SCHEDULE_DEPENDENCY_CONTROL = 3,
} loom_low_schedule_dependency_kind_t;

// One scheduled operation in a low function body.
typedef struct loom_low_schedule_node_t {
  // Operation represented by this node.
  const loom_op_t* op;
  // Block containing |op|.
  const loom_block_t* block;
  // Region block ordinal containing |op|.
  uint32_t block_index;
  // Source-order ordinal within the whole low function body.
  uint32_t source_ordinal;
  // Scheduled ordinal within |block| after topological scheduling.
  uint32_t scheduled_ordinal;
  // Kind of schedule node.
  loom_low_schedule_node_kind_t kind;
  // Effective traits used for conservative structural ordering.
  loom_trait_flags_t traits;
  // Descriptor ordinal for descriptor-backed nodes, or NONE.
  uint32_t descriptor_ordinal;
  // Borrowed descriptor key for descriptor-backed nodes.
  iree_string_view_t descriptor_key;
  // Schedule-class id for descriptor-backed nodes, or NONE.
  uint16_t schedule_class_id;
  // Borrowed schedule-class name for descriptor-backed nodes.
  iree_string_view_t schedule_class_name;
  // Descriptor schedule latency in cycles.
  uint16_t latency_cycles;
  // Descriptor latency interpretation.
  loom_low_latency_kind_t latency_kind;
  // Descriptor schedule-model quality.
  loom_low_model_quality_t model_quality;
  // Number of issue-resource rows consumed by the schedule class.
  uint16_t issue_use_count;
  // Number of descriptor effect rows.
  uint16_t effect_count;
} loom_low_schedule_node_t;

// One dependency edge between two schedule nodes.
typedef struct loom_low_schedule_dependency_t {
  // Producer node index.
  uint32_t producer_node;
  // Consumer node index.
  uint32_t consumer_node;
  // Dependency kind.
  loom_low_schedule_dependency_kind_t kind;
  // Operand index for SSA dependencies, or UINT32_MAX.
  uint32_t operand_index;
} loom_low_schedule_dependency_t;

// Schedule metadata for one low function block.
typedef struct loom_low_schedule_block_t {
  // Region block represented by this record.
  const loom_block_t* block;
  // First source-order node index owned by this block.
  uint32_t node_start;
  // Number of nodes owned by this block.
  uint32_t node_count;
  // First entry in the sidecar scheduled-node-index array.
  uint32_t scheduled_node_start;
  // Number of scheduled-node-index entries owned by this block.
  uint32_t scheduled_node_count;
} loom_low_schedule_block_t;

// Options controlling low schedule construction.
typedef struct loom_low_schedule_options_t {
  // Descriptor registry available to the scheduler.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Structured diagnostic emitter for user IR failures.
  iree_diagnostic_emitter_t emitter;
} loom_low_schedule_options_t;

// Schedule sidecar for one low.func.def body. All arrays are arena-owned by the
// caller-provided arena passed to loom_low_schedule_function.
typedef struct loom_low_schedule_sidecar_t {
  // Module containing the scheduled low function.
  const loom_module_t* module;
  // low.func.def operation scheduled by this sidecar.
  const loom_op_t* function_op;
  // Resolved target context selected by |function_op|.
  loom_low_resolved_target_t target;
  // Liveness analysis for the scheduled low function body.
  loom_liveness_analysis_t liveness;
  // Per-block schedule records in region block order.
  const loom_low_schedule_block_t* blocks;
  // Number of block records.
  iree_host_size_t block_count;
  // Per-op schedule nodes in source order.
  const loom_low_schedule_node_t* nodes;
  // Number of schedule nodes.
  iree_host_size_t node_count;
  // Dependency edges between schedule nodes.
  const loom_low_schedule_dependency_t* dependencies;
  // Number of dependency edges.
  iree_host_size_t dependency_count;
  // Node indices in scheduled order, grouped by block.
  const uint32_t* scheduled_node_indices;
  // Number of scheduled node indices.
  iree_host_size_t scheduled_node_count;
} loom_low_schedule_sidecar_t;

// Schedules one low.func.def body and writes an arena-owned sidecar. The caller
// must keep |module| immutable and |arena| alive for as long as |out_sidecar|
// is used. This function performs descriptor target resolution and liveness
// analysis; malformed user IR is reported through |options->emitter| when
// provided and otherwise fails loud with status.
iree_status_t loom_low_schedule_function(
    const loom_module_t* module, const loom_op_t* low_func_op,
    const loom_low_schedule_options_t* options, iree_arena_allocator_t* arena,
    loom_low_schedule_sidecar_t* out_sidecar);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_H_
