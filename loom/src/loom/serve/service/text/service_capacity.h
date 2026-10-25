// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_SERVICE_CAPACITY_H_
#define LOOM_SERVE_SERVICE_TEXT_SERVICE_CAPACITY_H_

#include "loom/serve/service/text/service.h"
#include "loom/serve/service/text/session.h"

#ifdef __cplusplus
extern "C" {
#endif

// Cold growth borrows the model on a CPU task, then its queue frontier. Public
// inspection continues to see the old directory until the owner publishes the
// replacement. Existing records remain stable throughout either loan.
typedef struct loom_serve_text_service_capacity_t {
  // Requested next capacity, independent of epoch shape and live KV extent.
  iree_host_size_t requested;
  // Initialized prefix of sessions; the old prefix is borrowed, not owned.
  iree_host_size_t count;
  // Owned replacement directory; the service arena owns its stable records.
  loom_serve_text_session_t** sessions;
  // Owned zeroed readiness entries for the next directory.
  loom_serve_ready_span_t* ready;
  // Whether native growth was submitted instead of deferred for a borrower.
  bool submitted;
  // Accepted native work, published by CPU task release.
  loom_serve_execution_frontier_t frontier;
} loom_serve_text_service_capacity_t;

// Called after selection found no idle resident. A successful retired owner
// starts amortized growth; accepted epochs and prepared continuations defer it.
bool loom_serve_text_service_capacity_begin(loom_serve_text_service_t* service);

// After successful CPU readiness, imports the submitted device frontier.
iree_status_t loom_serve_text_service_capacity_await(
    loom_serve_text_service_t* service);

// After successful device readiness, atomically publishes the new directory
// on the owner. Normal request admission still reserves actual memory credit.
void loom_serve_text_service_capacity_publish(
    loom_serve_text_service_t* service);

// Releases unconsumed host preparation after all CPU/device borrowers retire.
void loom_serve_text_service_capacity_discard(
    loom_serve_text_service_t* service);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_SERVICE_CAPACITY_H_
