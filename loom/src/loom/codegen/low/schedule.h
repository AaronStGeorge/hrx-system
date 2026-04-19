// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent scheduler sidecar for target-low functions.
//
// This layer consumes ordinary Loom IR plus descriptor tables and produces a
// deterministic schedule sidecar. The default scheduler is intentionally
// conservative: it builds the dependency graph and records a source-priority
// topological order without mutating IR. Optional strategies can score the same
// dependency-ready sets for register pressure while still keeping target hazard
// insertion, allocation, and diagnostics on this sidecar instead of creating a
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

enum loom_low_schedule_diagnostic_bits_e {
  // Emits one BACKEND/003 remark per hard-bounded register-pressure summary.
  LOOM_LOW_SCHEDULE_DIAGNOSTIC_PRESSURE_PEAKS = 1u << 0,
  // Emits BACKEND/013 remarks for resources tied at the schedule bottleneck.
  LOOM_LOW_SCHEDULE_DIAGNOSTIC_RESOURCE_BOTTLENECKS = 1u << 1,
  // Emits BACKEND/014 remarks for required delay/wait hazard gaps.
  LOOM_LOW_SCHEDULE_DIAGNOSTIC_HAZARD_GAPS = 1u << 2,
  // Emits BACKEND/015 remarks for pressure-scheduler candidate choices.
  LOOM_LOW_SCHEDULE_DIAGNOSTIC_CANDIDATE_DECISIONS = 1u << 3,
  // Emits BACKEND/016 remarks for non-exact schedule model quality.
  LOOM_LOW_SCHEDULE_DIAGNOSTIC_MODEL_QUALITY = 1u << 4,
};
typedef uint32_t loom_low_schedule_diagnostic_flags_t;

typedef enum loom_low_schedule_strategy_e {
  // Chooses the first ready node in source order.
  LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY = 0,
  // Chooses ready nodes using a target-independent register-pressure score.
  LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE = 1,
} loom_low_schedule_strategy_t;

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
  // Number of hazard rows attached to the schedule class.
  uint16_t hazard_count;
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

// Pressure-model step recorded while scheduling one node. This is an aggregate
// target-independent register-pressure estimate across all register classes,
// not a replacement for source-order liveness or target occupancy analysis.
typedef struct loom_low_schedule_pressure_step_t {
  // Scheduled node represented by this step.
  uint32_t node_index;
  // Region block containing |node_index|.
  uint32_t block_index;
  // Scheduled ordinal within |block_index|.
  uint32_t scheduled_ordinal;
  // Aggregate register live units before scheduling the node.
  uint64_t live_units_before;
  // Register live units killed by the node.
  uint64_t killed_live_units;
  // Register live units produced by the node.
  uint64_t produced_live_units;
  // Aggregate register live units after scheduling the node.
  uint64_t live_units_after;
} loom_low_schedule_pressure_step_t;

// Pressure-strategy candidate decision recorded when a ready set has an
// alternative to reject. This is intentionally compact: it captures the chosen
// candidate and the best rejected alternative, which is enough to explain local
// pressure tradeoffs without recording every ready-set member in large blocks.
typedef struct loom_low_schedule_candidate_decision_t {
  // Region block containing the decision.
  uint32_t block_index;
  // Scheduled ordinal within |block_index|.
  uint32_t scheduled_ordinal;
  // Number of dependency-ready candidates scored at this ordinal.
  uint32_t ready_candidate_count;
  // Chosen schedule node.
  uint32_t chosen_node;
  // Best rejected schedule node, or LOOM_LOW_SCHEDULE_NODE_NONE.
  uint32_t rejected_node;
  // Chosen aggregate live register units after scheduling the node.
  uint64_t chosen_projected_live_units;
  // Chosen live register units killed by scheduling the node.
  uint64_t chosen_killed_live_units;
  // Chosen live register units produced by scheduling the node.
  uint64_t chosen_produced_live_units;
  // Best rejected aggregate live register units after scheduling the node.
  uint64_t rejected_projected_live_units;
  // Best rejected live register units killed by scheduling the node.
  uint64_t rejected_killed_live_units;
  // Best rejected live register units produced by scheduling the node.
  uint64_t rejected_produced_live_units;
} loom_low_schedule_candidate_decision_t;

// Descriptor resource use recorded in scheduled order. This is an issue-model
// trace, not a cycle-accurate reservation table; target overlays can refine it
// into wait insertion, occupancy, or port-pressure diagnostics.
typedef struct loom_low_schedule_resource_use_t {
  // Scheduled node using the resource.
  uint32_t node_index;
  // Region block containing |node_index|.
  uint32_t block_index;
  // Scheduled ordinal within |block_index|.
  uint32_t scheduled_ordinal;
  // Issue-use row ordinal within the node's schedule class.
  uint16_t issue_use_ordinal;
  // Target resource table identifier consumed by this issue use.
  uint16_t resource_id;
  // Borrowed stable resource name.
  iree_string_view_t resource_name;
  // Abstract resource kind used by generic diagnostics.
  loom_low_resource_kind_t resource_kind;
  // Generic resource flags from the descriptor table.
  loom_low_resource_flags_t resource_flags;
  // Resource units available per cycle in the descriptor model.
  uint16_t capacity_per_cycle;
  // Contention group identifier shared by related resources.
  uint16_t contention_group_id;
  // Pipeline stage associated with this use.
  uint16_t stage;
  // Number of cycles the resource is occupied.
  uint16_t cycles;
  // Number of resource units consumed per cycle.
  uint16_t units;
} loom_low_schedule_resource_use_t;

// Descriptor effect row recorded in scheduled order. Effects describe memory,
// counter, call, barrier, and control visibility used by dependency
// construction and target-owned policies such as AMDGPU waitcnt planning.
typedef struct loom_low_schedule_effect_use_t {
  // Scheduled node carrying the effect row.
  uint32_t node_index;
  // Region block containing |node_index|.
  uint32_t block_index;
  // Scheduled ordinal within |block_index|.
  uint32_t scheduled_ordinal;
  // Effect row ordinal within the node's descriptor.
  uint16_t effect_ordinal;
  // Effect kind used by dependency and legality construction.
  loom_low_effect_kind_t kind;
  // Memory space or external resource touched by the effect.
  loom_low_memory_space_t memory_space;
  // Target-owned scope identifier for ordering and visibility.
  uint16_t scope_id;
  // Effect flags used by scheduling and verification.
  loom_low_effect_flags_t effect_flags;
  // Target counter identifier for counter effects.
  uint16_t counter_id;
  // Access width in bits, or zero when not width-specific.
  uint16_t width_bits;
} loom_low_schedule_effect_use_t;

// Descriptor hazard row recorded in scheduled order. These rows are passive
// facts: target overlays consume them to insert waits, enforce distances, or
// record richer backend diagnostics.
typedef struct loom_low_schedule_hazard_use_t {
  // Scheduled node carrying the hazard row.
  uint32_t node_index;
  // Region block containing |node_index|.
  uint32_t block_index;
  // Scheduled ordinal within |block_index|.
  uint32_t scheduled_ordinal;
  // Hazard row ordinal within the node's schedule class.
  uint16_t hazard_ordinal;
  // Hazard kind used by schedule policy and verification.
  loom_low_hazard_kind_t kind;
  // Interpretation of reference_id.
  loom_low_hazard_reference_kind_t reference_kind;
  // Resource, counter, or target-owned hazard identifier.
  uint16_t reference_id;
  // Borrowed stable resource name when reference_kind is RESOURCE.
  iree_string_view_t resource_name;
  // Producer pipeline stage participating in the hazard.
  uint16_t producer_stage;
  // Consumer pipeline stage participating in the hazard.
  uint16_t consumer_stage;
  // Required distance or target-owned hazard value.
  uint16_t distance;
  // Hazard flags for target-owned refinements.
  loom_low_hazard_flags_t hazard_flags;
} loom_low_schedule_hazard_use_t;

// Minimum-distance hazard gap recorded after scheduling. These rows identify
// where the chosen order needs an abstract delay/wait before final emission.
// They are not inserted operations; target overlays decide whether a gap
// becomes a wait packet, a delay packet, or a diagnostic.
typedef struct loom_low_schedule_hazard_gap_t {
  // Producer node carrying the previous hazard row.
  uint32_t producer_node;
  // Consumer node carrying the hazard row that requires additional distance.
  uint32_t consumer_node;
  // Region block containing both nodes.
  uint32_t block_index;
  // Scheduled ordinal of the producer node within |block_index|.
  uint32_t producer_scheduled_ordinal;
  // Scheduled ordinal of the consumer node within |block_index|.
  uint32_t consumer_scheduled_ordinal;
  // Hazard row ordinal within the producer node's schedule class.
  uint16_t producer_hazard_ordinal;
  // Hazard row ordinal within the consumer node's schedule class.
  uint16_t consumer_hazard_ordinal;
  // Hazard kind that produced this gap.
  loom_low_hazard_kind_t kind;
  // Interpretation of reference_id.
  loom_low_hazard_reference_kind_t reference_kind;
  // Resource, counter, or target-owned hazard identifier.
  uint16_t reference_id;
  // Borrowed stable resource name when reference_kind is RESOURCE.
  iree_string_view_t resource_name;
  // Producer pipeline stage participating in the hazard.
  uint16_t producer_stage;
  // Consumer pipeline stage participating in the hazard.
  uint16_t consumer_stage;
  // Required minimum distance in abstract issue slots.
  uint16_t required_distance;
  // Actual scheduled distance in abstract issue slots.
  uint32_t actual_distance;
  // Additional abstract issue slots needed before the consumer.
  uint16_t required_delay;
  // Hazard flags for target-owned refinements.
  loom_low_hazard_flags_t hazard_flags;
} loom_low_schedule_hazard_gap_t;

// Schedule-class model quality summary for schedule classes used by this
// function. These summaries make model uncertainty visible without repeating
// the same schedule-class facts on every packet.
typedef struct loom_low_schedule_model_summary_t {
  // Schedule node where this class first appears, or
  // LOOM_LOW_SCHEDULE_NODE_NONE.
  uint32_t first_node;
  // Target descriptor schedule-class identifier.
  uint32_t schedule_class_id;
  // Borrowed stable schedule-class name.
  iree_string_view_t schedule_class_name;
  // Descriptor schedule latency in cycles.
  uint16_t latency_cycles;
  // Descriptor latency interpretation.
  loom_low_latency_kind_t latency_kind;
  // Descriptor schedule-model quality.
  loom_low_model_quality_t model_quality;
  // Number of issue-resource rows consumed by the schedule class.
  uint16_t issue_use_count;
  // Number of hazard rows attached to the schedule class.
  uint16_t hazard_count;
  // Number of scheduled nodes using this schedule class.
  uint32_t use_count;
} loom_low_schedule_model_summary_t;

// Aggregate descriptor resource pressure for one target resource. Summaries are
// emitted in resource-id order and only include resources used by the schedule.
typedef struct loom_low_schedule_resource_summary_t {
  // Target resource table identifier summarized by this record.
  uint16_t resource_id;
  // Borrowed stable resource name.
  iree_string_view_t resource_name;
  // Abstract resource kind used by generic diagnostics.
  loom_low_resource_kind_t resource_kind;
  // Generic resource flags from the descriptor table.
  loom_low_resource_flags_t resource_flags;
  // Resource units available per cycle in the descriptor model.
  uint16_t capacity_per_cycle;
  // Contention group identifier shared by related resources.
  uint16_t contention_group_id;
  // Number of issue-use rows accumulated for this resource.
  uint32_t use_count;
  // Sum of occupied cycles across all issue-use rows.
  uint64_t total_occupied_cycles;
  // Sum of cycles * units across all issue-use rows.
  uint64_t total_unit_cycles;
  // Ceiling of total_unit_cycles / capacity_per_cycle.
  uint64_t estimated_min_cycles;
  // Largest single issue-use units value observed for this resource.
  uint16_t peak_units_per_cycle;
} loom_low_schedule_resource_summary_t;

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
  // Optional backend feedback diagnostics to emit after scheduling analysis.
  loom_low_schedule_diagnostic_flags_t diagnostic_flags;
  // Candidate selection strategy used within each dependency-ready set.
  loom_low_schedule_strategy_t strategy;
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
  // Pressure-model steps in scheduled order when the selected strategy records
  // them. Empty for the default source-priority strategy.
  const loom_low_schedule_pressure_step_t* pressure_steps;
  // Number of pressure-model steps.
  iree_host_size_t pressure_step_count;
  // Pressure candidate decisions in scheduled order when requested by
  // diagnostic flags. Empty for source-priority scheduling and ordinary
  // pressure scheduling without candidate diagnostics.
  const loom_low_schedule_candidate_decision_t* candidate_decisions;
  // Number of pressure candidate decision records.
  iree_host_size_t candidate_decision_count;
  // Descriptor resource uses in scheduled order. Empty when scheduled nodes do
  // not reference descriptor issue-use rows.
  const loom_low_schedule_resource_use_t* resource_uses;
  // Number of resource-use records.
  iree_host_size_t resource_use_count;
  // Descriptor effects in scheduled order.
  const loom_low_schedule_effect_use_t* effect_uses;
  // Number of effect-use records.
  iree_host_size_t effect_use_count;
  // Descriptor hazards in scheduled order. Empty when scheduled nodes do not
  // reference descriptor hazard rows.
  const loom_low_schedule_hazard_use_t* hazard_uses;
  // Number of hazard-use records.
  iree_host_size_t hazard_use_count;
  // Minimum-distance hazard gaps in scheduled order.
  const loom_low_schedule_hazard_gap_t* hazard_gaps;
  // Number of minimum-distance hazard gaps.
  iree_host_size_t hazard_gap_count;
  // Schedule-class model-quality summaries in descriptor schedule-class order.
  const loom_low_schedule_model_summary_t* model_summaries;
  // Number of schedule-class model-quality summaries.
  iree_host_size_t model_summary_count;
  // Per-resource aggregate schedule pressure in resource-id order.
  const loom_low_schedule_resource_summary_t* resource_summaries;
  // Number of resource summary records.
  iree_host_size_t resource_summary_count;
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
