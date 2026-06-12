// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Collects target-neutral artifact manifest facts from Loom IR analyses.

#ifndef LOOM_TARGET_ARTIFACT_MANIFEST_COLLECT_H_
#define LOOM_TARGET_ARTIFACT_MANIFEST_COLLECT_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_dependencies.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/ir/ir.h"
#include "loom/target/artifact_manifest.h"
#include "loom/target/artifact_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t loom_target_artifact_manifest_collect_flags_t;

enum loom_target_artifact_manifest_collect_flag_bits_e {
  // Indicates that artifact_byte_length carries the emitted artifact byte
  // length. The collector preserves zero when this flag is set.
  LOOM_TARGET_ARTIFACT_MANIFEST_COLLECT_FLAG_ARTIFACT_BYTE_LENGTH = 1u << 0,
};

typedef struct loom_target_artifact_manifest_collect_options_t {
  // Selected manifest detail mode.
  loom_target_artifact_manifest_mode_t mode;

  // Bitfield selecting optional collector inputs that are present.
  loom_target_artifact_manifest_collect_flags_t flags;

  // Emitted artifact byte length when ARTIFACT_BYTE_LENGTH is set.
  uint64_t artifact_byte_length;
} loom_target_artifact_manifest_collect_options_t;

// Initializes collection options with manifest collection disabled.
void loom_target_artifact_manifest_collect_options_initialize(
    loom_target_artifact_manifest_collect_options_t* out_options);

// Collects a manifest for |plan| into arena-owned storage.
//
// NONE mode only clears |out_manifest|. SUMMARY mode collects cheap structural
// facts already present in the artifact plan, symbol facts, target facts, and
// symbol dependency table. DETAILS and ANALYSIS modes additionally collect
// per-parameter rows from function signatures. Expensive artifact byte scans,
// hashing, disassembly, target-specific static analysis, and pass reports are
// intentionally outside this shared collector.
//
// Manifest arrays are allocated from |arena|. String views borrow storage from
// |module|, |fact_table|, and static literals and must not outlive them.
iree_status_t loom_target_artifact_manifest_collect_from_plan(
    const loom_module_t* module, const loom_target_artifact_plan_t* plan,
    loom_symbol_fact_table_t* fact_table,
    const loom_symbol_dependency_table_t* dependency_table,
    const loom_target_artifact_manifest_collect_options_t* options,
    iree_arena_allocator_t* arena,
    loom_target_artifact_manifest_t* out_manifest);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARTIFACT_MANIFEST_COLLECT_H_
