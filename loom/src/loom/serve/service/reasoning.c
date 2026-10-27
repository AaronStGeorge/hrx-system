// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/reasoning.h"

iree_status_t loom_serve_reasoning_initialize(
    int32_t enabled, int32_t efforts, int32_t default_effort,
    loom_serve_reasoning_t* out_reasoning) {
  memset(out_reasoning, 0, sizeof(*out_reasoning));
  if (enabled < 0 || enabled > 1 || efforts < 0 ||
      efforts >= (1u << LOOM_SERVE_REASONING_EFFORT_COUNT) ||
      default_effort < 0 ||
      default_effort >= LOOM_SERVE_REASONING_EFFORT_COUNT ||
      (!enabled && efforts) ||
      (efforts ? (efforts == 1 || !(efforts & (1u << default_effort)))
               : default_effort != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source chat_reasoning requires a boolean, a supported effort mask, "
        "and a default in that mask; absent efforts require a zero default");
  }
  *out_reasoning = (loom_serve_reasoning_t){
      .support = enabled ? LOOM_SERVE_REASONING_SUPPORTED
                         : LOOM_SERVE_REASONING_UNSUPPORTED,
      .efforts = (uint8_t)efforts,
      .default_effort = (uint8_t)default_effort,
  };
  return iree_ok_status();
}

const char* loom_serve_reasoning_effort_name(
    loom_serve_reasoning_effort_t effort) {
  static const char names[][8] = {"none", "minimal", "low", "medium",
                                  "high", "xhigh",   "max"};
  return names[effort];
}
