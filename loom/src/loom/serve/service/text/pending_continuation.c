// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/pending_continuation.h"

iree_status_t loom_serve_text_pending_continuation_prepare(
    loom_serve_text_session_t* session,
    loom_serve_text_session_cache_entry_t* cached,
    loom_serve_text_prefix_entry_t* prefix,
    loom_serve_text_checkpoint_t* checkpoint, iree_host_size_t retained_count,
    iree_host_size_t extent,
    loom_serve_text_pending_continuation_t* out_continuation) {
  *out_continuation = (loom_serve_text_pending_continuation_t){0};
  if (!checkpoint && retained_count && cached) {
    checkpoint = cached->endpoint.state;
  }
  if (!checkpoint && !retained_count && prefix) {
    checkpoint = prefix->state;
  }
  const loom_serve_text_reserve_mode_t mode =
      checkpoint       ? LOOM_SERVE_TEXT_RESERVE_CHECKPOINT
      : retained_count ? LOOM_SERVE_TEXT_RESERVE_CONTINUE
                       : LOOM_SERVE_TEXT_RESERVE_FRESH;
  IREE_RETURN_IF_ERROR(loom_serve_text_reservation_prepare(
      session->row, mode, checkpoint, extent, &out_continuation->reservation));
  if (out_continuation->reservation.state) {
    out_continuation->session = session;
    out_continuation->cached = cached;
    out_continuation->prefix = prefix;
    out_continuation->restored_bytes =
        checkpoint ? loom_serve_text_checkpoint_suspended_bytes(checkpoint) : 0;
    session->preparing = true;
    if (cached) {
      ++cached->endpoint.pending;
    }
    if (prefix) {
      ++prefix->pending;
    }
  }
  return iree_ok_status();
}

void loom_serve_text_pending_continuation_deinitialize(
    loom_serve_text_pending_continuation_t* continuation) {
  loom_serve_text_reservation_deinitialize(&continuation->reservation);
  if (continuation->session) {
    continuation->session->preparing = false;
  }
  if (continuation->cached) {
    --continuation->cached->endpoint.pending;
  }
  if (continuation->prefix) {
    --continuation->prefix->pending;
  }
  *continuation = (loom_serve_text_pending_continuation_t){0};
}
