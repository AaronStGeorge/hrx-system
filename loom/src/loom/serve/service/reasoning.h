// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_REASONING_H_
#define LOOM_SERVE_SERVICE_REASONING_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Protocol vocabulary, not model-specific policies or token budgets. Source
// chooses which levels to advertise and interprets request values itself.
typedef enum loom_serve_reasoning_effort_e {
  LOOM_SERVE_REASONING_EFFORT_NONE = 0,
  LOOM_SERVE_REASONING_EFFORT_MINIMAL = 1,
  LOOM_SERVE_REASONING_EFFORT_LOW = 2,
  LOOM_SERVE_REASONING_EFFORT_MEDIUM = 3,
  LOOM_SERVE_REASONING_EFFORT_HIGH = 4,
  LOOM_SERVE_REASONING_EFFORT_XHIGH = 5,
  LOOM_SERVE_REASONING_EFFORT_MAX = 6,
  LOOM_SERVE_REASONING_EFFORT_COUNT = 7,
} loom_serve_reasoning_effort_t;

typedef enum loom_serve_reasoning_support_e {
  LOOM_SERVE_REASONING_UNKNOWN = 0,
  LOOM_SERVE_REASONING_UNSUPPORTED = 1,
  LOOM_SERVE_REASONING_SUPPORTED = 2,
} loom_serve_reasoning_support_t;

// Immutable discovery facts copied from source during model preparation.
// Zero initialization means unknown, distinct from an explicit denial.
typedef struct loom_serve_reasoning_t {
  // One loom_serve_reasoning_support_t value.
  uint8_t support;
  // Bit i advertises effort ordinal i. Zero means no effort selector.
  uint8_t efforts;
  // Selected ordinal when a request omits controls; zero when efforts is zero.
  uint8_t default_effort;
} loom_serve_reasoning_t;

// Validates the cold source declaration. Enabled is a boolean; disabled models
// return three zeros. Enabled models may omit an effort selector, or declare a
// ladder containing a reasoning level and its default. Failure leaves unknown.
iree_status_t loom_serve_reasoning_initialize(
    int32_t enabled, int32_t efforts, int32_t default_effort,
    loom_serve_reasoning_t* out_reasoning);

// Process-lifetime spelling of a valid effort ordinal.
const char* loom_serve_reasoning_effort_name(
    loom_serve_reasoning_effort_t effort);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_REASONING_H_
