// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_OBSERVATION_H_
#define LOOM_SERVE_SERVER_OBSERVATION_H_

#include "loom/serve/server/registry.h"

#ifdef __cplusplus
extern "C" {
#endif

// Writes one bounded current-state page on the serialized registry owner.
// Cursor zero samples server-lifetime counters; subsequent pages inspect at
// most 64 registry slots and return at most one model. The cursor returns to
// zero after the last slot. Every model page identifies its scanned half-open
// slot range; absence is authoritative only within that range at capture time.
// Pages do not freeze an inventory or retain model/stream storage across calls.
// Display names are capped at 256 ASCII bytes with explicit truncation; slot
// and incarnation, not the display label, identify the source. All uint64
// counters and identities use decimal strings. Output failure leaves cursor
// unchanged. Caller-owned bounded output makes the byte budget explicit and
// reusable by polling, native collectors, and streaming adapters.
iree_status_t loom_serve_observation_write_page(loom_serve_registry_t* registry,
                                                iree_host_size_t* cursor,
                                                iree_string_builder_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_SERVER_OBSERVATION_H_
