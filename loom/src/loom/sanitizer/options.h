// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Sanitizer enablement options shared by compiler front doors and target
// pipeline construction.

#ifndef LOOM_SANITIZER_OPTIONS_H_
#define LOOM_SANITIZER_OPTIONS_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

enum loom_sanitizer_check_bit_e {
  // Enables memory access assertions.
  LOOM_SANITIZER_CHECK_ACCESS = 1u << 0,
  // Enables SSA value assertions.
  LOOM_SANITIZER_CHECK_VALUE = 1u << 1,
  // Enables operation-level assertions.
  LOOM_SANITIZER_CHECK_OPERATION = 1u << 2,
};
// Bitset of loom_sanitizer_check_bit_e values selecting assertion classes.
typedef uint64_t loom_sanitizer_checks_t;

enum loom_sanitizer_flag_bit_e {
  // No sanitizer flags are currently defined.
  LOOM_SANITIZER_FLAG_NONE = 0u,
};
// Bitset of loom_sanitizer_flag_bit_e values controlling sanitizer behavior.
typedef uint32_t loom_sanitizer_flags_t;

// Check bits understood by this compiler.
#define LOOM_SANITIZER_CHECKS_KNOWN                        \
  ((loom_sanitizer_checks_t)(LOOM_SANITIZER_CHECK_ACCESS | \
                             LOOM_SANITIZER_CHECK_VALUE |  \
                             LOOM_SANITIZER_CHECK_OPERATION))

// Flag bits understood by this compiler.
#define LOOM_SANITIZER_FLAGS_KNOWN ((loom_sanitizer_flags_t)0u)

typedef struct loom_sanitizer_options_t {
  // Assertion classes enabled by the caller.
  loom_sanitizer_checks_t checks;
  // Additional behavior flags enabled by the caller.
  loom_sanitizer_flags_t flags;
} loom_sanitizer_options_t;

// Returns true when sanitizer instrumentation is enabled.
static inline bool loom_sanitizer_options_is_enabled(
    const loom_sanitizer_options_t* options) {
  return options != NULL && options->checks != 0;
}

// Validates that all sanitizer option bits are understood by this compiler.
static inline iree_status_t loom_sanitizer_options_validate(
    const loom_sanitizer_options_t* options) {
  if (options == NULL) {
    return iree_ok_status();
  }
  if ((options->checks & ~LOOM_SANITIZER_CHECKS_KNOWN) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sanitizer options contain unknown check bits");
  }
  if ((options->flags & ~LOOM_SANITIZER_FLAGS_KNOWN) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sanitizer options contain unknown flag bits");
  }
  return iree_ok_status();
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SANITIZER_OPTIONS_H_
