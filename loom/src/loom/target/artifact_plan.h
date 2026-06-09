// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target artifact compile-unit planning.
//
// Artifact plans are derived data: target.artifact symbols describe packaging,
// exported func facts assign entry points to artifacts, and the call graph
// provides private func closure. Artifacts never own entry lists in IR.

#ifndef LOOM_TARGET_ARTIFACT_PLAN_H_
#define LOOM_TARGET_ARTIFACT_PLAN_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/ops/target/facts.h"
#include "loom/util/call_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

// Derived compile-unit plan for one target.artifact symbol.
typedef struct loom_target_artifact_plan_t {
  // Module-local artifact symbol this plan was built for.
  loom_symbol_ref_t artifact_symbol;

  // Resolved artifact facts for artifact_symbol.
  const loom_target_artifact_symbol_facts_t* artifact_facts;

  // Exported entry func symbol IDs assigned to artifact_symbol.
  const loom_symbol_id_t* entry_symbol_ids;

  // Number of entry func symbol IDs.
  uint16_t entry_count;

  // Body func symbol IDs included in the artifact compile unit.
  const loom_symbol_id_t* func_symbol_ids;

  // Number of body func symbol IDs.
  uint16_t func_count;
} loom_target_artifact_plan_t;

// Builds a compile-unit plan for one target.artifact.
//
// |fact_table| must contain the target-record facts referenced by exported
// functions. |call_graph| must have been built for |module|. All plan arrays
// are allocated from |arena| and remain valid for the arena lifetime. Returns
// status only for infrastructure failures. Invalid user IR emits a structured
// diagnostic, sets |out_valid| to false, and returns OK.
iree_status_t loom_target_artifact_plan_build(
    const loom_module_t* module, loom_symbol_ref_t artifact_symbol,
    loom_symbol_fact_table_t* fact_table, const loom_call_graph_t* call_graph,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    bool* out_valid, loom_target_artifact_plan_t* out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARTIFACT_PLAN_H_
