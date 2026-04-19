// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TOOLS_LOOM_CHECK_STATUS_UTIL_H_
#define LOOM_TOOLS_LOOM_CHECK_STATUS_UTIL_H_

#include "iree/base/api.h"

// Returns a stable status code name without reading optional status payload
// text.
static inline iree_string_view_t loom_check_status_name(iree_status_t status) {
  if (iree_status_is_ok(status)) return IREE_SV("OK");
  if (iree_status_is_cancelled(status)) return IREE_SV("CANCELLED");
  if (iree_status_is_unknown(status)) return IREE_SV("UNKNOWN");
  if (iree_status_is_invalid_argument(status)) {
    return IREE_SV("INVALID_ARGUMENT");
  }
  if (iree_status_is_deadline_exceeded(status)) {
    return IREE_SV("DEADLINE_EXCEEDED");
  }
  if (iree_status_is_not_found(status)) return IREE_SV("NOT_FOUND");
  if (iree_status_is_already_exists(status)) {
    return IREE_SV("ALREADY_EXISTS");
  }
  if (iree_status_is_permission_denied(status)) {
    return IREE_SV("PERMISSION_DENIED");
  }
  if (iree_status_is_resource_exhausted(status)) {
    return IREE_SV("RESOURCE_EXHAUSTED");
  }
  if (iree_status_is_failed_precondition(status)) {
    return IREE_SV("FAILED_PRECONDITION");
  }
  if (iree_status_is_aborted(status)) return IREE_SV("ABORTED");
  if (iree_status_is_out_of_range(status)) return IREE_SV("OUT_OF_RANGE");
  if (iree_status_is_unimplemented(status)) return IREE_SV("UNIMPLEMENTED");
  if (iree_status_is_internal(status)) return IREE_SV("INTERNAL");
  if (iree_status_is_unavailable(status)) return IREE_SV("UNAVAILABLE");
  if (iree_status_is_data_loss(status)) return IREE_SV("DATA_LOSS");
  if (iree_status_is_unauthenticated(status)) return IREE_SV("UNAUTHENTICATED");
  if (iree_status_is_deferred(status)) return IREE_SV("DEFERRED");
  if (iree_status_is_incompatible(status)) return IREE_SV("INCOMPATIBLE");
  return IREE_SV("UNKNOWN_STATUS");
}

#endif  // LOOM_TOOLS_LOOM_CHECK_STATUS_UTIL_H_
