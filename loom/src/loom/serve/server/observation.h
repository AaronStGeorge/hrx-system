// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_OBSERVATION_H_
#define LOOM_SERVE_SERVER_OBSERVATION_H_

#include "loom/serve/server/registry.h"
#include "loom/serve/server/resources.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_observation_t loom_serve_observation_t;

// Revision of one copied source, independent of transport or reader lifetime.
typedef struct loom_serve_observation_version_t {
  // Latest sample time, including samples whose content did not change.
  iree_time_t time_ns;
  // Content revision; zero identifies an empty model range.
  uint64_t revision;
  // Revision immediately preceding the latest content change.
  uint64_t previous_revision;
  // Last sample of previous_revision, not the time of its first publication.
  iree_time_t previous_time_ns;
} loom_serve_observation_version_t;

// Allocates shared latest-state metadata, with one entry per registry slot plus
// server totals and immutable device sources. Resources remain borrowed through
// observation destruction. Content size is bounded by the formatting extent;
// storage grows geometrically and is reused across samples and readers. No
// service-owned pointers survive sampling. The serialized registry owner
// advances sampling and reads pages.
iree_status_t loom_serve_observation_allocate(
    loom_serve_registry_t* registry, const loom_serve_resources_t* resources,
    iree_allocator_t host_allocator,
    loom_serve_observation_t** out_observation);
void loom_serve_observation_free(loom_serve_observation_t* observation);

// Samples server totals at cursor zero, then one device per call, then at most
// 64 slots and one model per call. Returning cursor zero completes a sweep.
// Scratch is caller-owned fixed storage; its capacity bounds each source.
// Exact byte comparison identifies unchanged state. A failed sample leaves its
// cursor unchanged and propagates output/allocation failure to the observation
// owner, never a model service.
iree_status_t loom_serve_observation_sample(
    loom_serve_observation_t* observation, iree_time_t now,
    iree_host_size_t* cursor, iree_string_builder_t* scratch);

// Changes only when a sampled source changes, including removal/incarnation.
uint64_t loom_serve_observation_revision(
    const loom_serve_observation_t* observation);

// Writes a page from shared copied state without resampling a service. Each
// model page identifies its scanned half-open slot range; absence is
// authoritative only within that range. Sweep reads do not freeze an inventory.
// Display names are capped at 256 ASCII bytes with explicit truncation; slot
// and incarnation, not the display label, identify the source. All uint64
// counters and identities use decimal strings. Output failure leaves cursor
// unchanged. A source version lets consumers distinguish observed idleness from
// missed revisions, without retaining per-source subscriber metadata here.
iree_status_t loom_serve_observation_write_page(
    const loom_serve_observation_t* observation, iree_host_size_t* cursor,
    iree_string_builder_t* output,
    loom_serve_observation_version_t* out_version);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_SERVER_OBSERVATION_H_
