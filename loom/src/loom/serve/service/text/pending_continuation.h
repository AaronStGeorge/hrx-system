// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_PENDING_CONTINUATION_H_
#define LOOM_SERVE_SERVICE_TEXT_PENDING_CONTINUATION_H_

#include "loom/serve/service/text/prefix_cache.h"
#include "loom/serve/service/text/reservation.h"
#include "loom/serve/service/text/session_cache.h"

#ifdef __cplusplus
extern "C" {
#endif

// Request-owned continuation while independent media work runs. The session
// and optional cache entries are protected from reuse/eviction, but language
// parameters remain cold. Zero initialization owns nothing.
typedef struct loom_serve_text_pending_continuation_t {
  // Borrowed idle destination, marked preparing for this record's lifetime.
  loom_serve_text_session_t* session;
  // Borrowed detached history, protected by its endpoint pending count.
  loom_serve_text_session_cache_entry_t* cached;
  // Borrowed exact prefix, protected by its pending count.
  loom_serve_text_prefix_entry_t* prefix;
  // Owned exact native plan, including a retained checkpoint if selected.
  loom_serve_text_reservation_t reservation;
  // Cold checkpoint payload measured before its first restore.
  iree_host_size_t restored_bytes;
} loom_serve_text_pending_continuation_t;

// The input has been prepared and any outgoing row history retained. Selects
// the same explicit/cache/prefix frontier as ordinary admission. A successful
// empty result denotes logical capacity pressure and protects no partial state.
iree_status_t loom_serve_text_pending_continuation_prepare(
    loom_serve_text_session_t* session,
    loom_serve_text_session_cache_entry_t* cached,
    loom_serve_text_prefix_entry_t* prefix,
    loom_serve_text_checkpoint_t* checkpoint, iree_host_size_t retained_count,
    iree_host_size_t extent,
    loom_serve_text_pending_continuation_t* out_continuation);

// Cancels logical ownership before submission, or drops metadata after the
// submitted row takes ownership. Releases cache/row borrows in either case.
// The request calls this before destroying or consuming its selected history.
void loom_serve_text_pending_continuation_deinitialize(
    loom_serve_text_pending_continuation_t* continuation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_PENDING_CONTINUATION_H_
