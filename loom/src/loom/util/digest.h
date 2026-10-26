// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_UTIL_DIGEST_H_
#define LOOM_UTIL_DIGEST_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Complete SHA-256 content identity in big-endian byte order.
typedef struct loom_sha256_t {
  uint8_t bytes[32];
} loom_sha256_t;

// Incremental SHA-256 state. The total input must remain below the algorithm's
// 2^61-byte bit-length limit. Initialize once, update with ordered byte spans,
// then finalize once; no operation allocates.
typedef struct loom_sha256_state_t {
  // Chaining state after complete input blocks.
  uint32_t state[8];
  // Total input length in bytes, excluding final padding.
  uint64_t data_length;
  // Pending input block or final padding block.
  uint8_t block[64];
  // Valid byte count in block.
  uint8_t block_length;
} loom_sha256_state_t;

void loom_sha256_initialize(loom_sha256_state_t* out_state);
void loom_sha256_update(loom_sha256_state_t* state,
                        iree_const_byte_span_t data);
void loom_sha256_finalize(loom_sha256_state_t* state,
                          loom_sha256_t* out_digest);

// One-shot counterpart to the incremental interface.
void loom_sha256_compute(iree_const_byte_span_t data,
                         loom_sha256_t* out_digest);

// Compares all digest bytes without data-dependent early exit.
bool loom_sha256_equal(const loom_sha256_t* lhs, const loom_sha256_t* rhs);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_UTIL_DIGEST_H_
