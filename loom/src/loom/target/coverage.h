// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-low execution coverage reporting.

#ifndef LOOM_TARGET_COVERAGE_H_
#define LOOM_TARGET_COVERAGE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t loom_target_coverage_phase_flags_t;
enum {
  // No target-low execution phase is covered.
  LOOM_TARGET_COVERAGE_PHASE_NONE = 0u,
  // Descriptor rows exist for the target category.
  LOOM_TARGET_COVERAGE_PHASE_DESCRIPTOR = 1u << 0,
  // Descriptor-backed target-low IR can be parsed and verified.
  LOOM_TARGET_COVERAGE_PHASE_LOW_PARSE = 1u << 1,
  // Source vector/scalar IR can lower into the target category.
  LOOM_TARGET_COVERAGE_PHASE_SOURCE_LOWER = 1u << 2,
  // Generic target-low scheduling can model the target category.
  LOOM_TARGET_COVERAGE_PHASE_SCHEDULE = 1u << 3,
  // Generic target-low allocation can assign storage for the target category.
  LOOM_TARGET_COVERAGE_PHASE_ALLOCATE = 1u << 4,
  // Human-readable target assembly can be emitted for the target category.
  LOOM_TARGET_COVERAGE_PHASE_TEXT_EMIT = 1u << 5,
  // Target instruction or module bytes can be encoded.
  LOOM_TARGET_COVERAGE_PHASE_BINARY_ENCODE = 1u << 6,
  // Encoded bytes can be wrapped into the target artifact container.
  LOOM_TARGET_COVERAGE_PHASE_ARTIFACT_WRAP = 1u << 7,
  // Produced artifacts can be inspected or validated by a target tool.
  LOOM_TARGET_COVERAGE_PHASE_INSPECT = 1u << 8,
  // Produced artifacts can be executed through an in-process or external
  // runner.
  LOOM_TARGET_COVERAGE_PHASE_RUN = 1u << 9,
};

// All phase bits known to this coverage schema.
#define LOOM_TARGET_COVERAGE_PHASE_ALL                                         \
  (LOOM_TARGET_COVERAGE_PHASE_DESCRIPTOR |                                     \
   LOOM_TARGET_COVERAGE_PHASE_LOW_PARSE |                                      \
   LOOM_TARGET_COVERAGE_PHASE_SOURCE_LOWER |                                   \
   LOOM_TARGET_COVERAGE_PHASE_SCHEDULE | LOOM_TARGET_COVERAGE_PHASE_ALLOCATE | \
   LOOM_TARGET_COVERAGE_PHASE_TEXT_EMIT |                                      \
   LOOM_TARGET_COVERAGE_PHASE_BINARY_ENCODE |                                  \
   LOOM_TARGET_COVERAGE_PHASE_ARTIFACT_WRAP |                                  \
   LOOM_TARGET_COVERAGE_PHASE_INSPECT | LOOM_TARGET_COVERAGE_PHASE_RUN)

// One target-owned coverage fact row.
//
// Rows are intentionally coarse: each row records a semantic category and the
// target-low execution phases expected for that category. They are a tool/test
// projection for planning work, not a durable compiler artifact. All string
// views are borrowed from static target tables and must outlive the provider.
typedef struct loom_target_coverage_row_t {
  // Stable target family or bundle key owning the row.
  iree_string_view_t target_key;
  // Low descriptor-set key associated with the row.
  iree_string_view_t descriptor_set_key;
  // Descriptor or kernel category, such as "vector.arithmetic".
  iree_string_view_t category;
  // Semantic capability within |category|, such as "i32x4.add".
  iree_string_view_t semantic_tag;
  // Phases expected before the capability is considered executable.
  loom_target_coverage_phase_flags_t expected_phases;
  // Expected phases currently implemented by the linked provider.
  loom_target_coverage_phase_flags_t supported_phases;
  // Short reason key for any missing expected phase, or empty when complete.
  iree_string_view_t gap_key;
} loom_target_coverage_row_t;

// Target-owned static coverage rows.
typedef struct loom_target_coverage_provider_t {
  // Stable provider name used in diagnostics and manifests.
  iree_string_view_t name;
  // Borrowed coverage row table.
  const loom_target_coverage_row_t* rows;
  // Number of entries in |rows|.
  iree_host_size_t row_count;
} loom_target_coverage_provider_t;

// Linked coverage provider set selected by a tool or embedding.
typedef struct loom_target_coverage_provider_set_t {
  // Borrowed provider pointer table.
  const loom_target_coverage_provider_t* const* providers;
  // Number of entries in |providers|.
  iree_host_size_t provider_count;
} loom_target_coverage_provider_set_t;

// Verifies the linked coverage provider set and every target-owned row.
iree_status_t loom_target_coverage_provider_set_verify(
    const loom_target_coverage_provider_set_t* provider_set);

// Appends a compact JSON manifest for |provider_set| to |builder|. The manifest
// lists provider-owned rows, expected/supported/missing phases, and short gap
// keys without embedding descriptor tables or target prose.
iree_status_t loom_target_coverage_provider_set_format_manifest_json(
    const loom_target_coverage_provider_set_t* provider_set,
    iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_COVERAGE_H_
