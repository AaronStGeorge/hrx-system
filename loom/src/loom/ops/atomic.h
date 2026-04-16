// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared atomic operation vocabulary used by scalar view atomics and vector
// atomics. Dialects still generate their own enum names for parser/printer and
// builder APIs, but the numeric values are intentionally identical to these
// shared constants so verification and lowering can use one semantic table.

#ifndef LOOM_OPS_ATOMIC_H_
#define LOOM_OPS_ATOMIC_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Read-modify-write operation kind for view and vector atomics.
typedef enum loom_atomic_kind_e {
  // Integer exchange.
  LOOM_ATOMIC_KIND_XCHGI = 0,
  // Floating-point exchange.
  LOOM_ATOMIC_KIND_XCHGF = 1,
  // Integer addition.
  LOOM_ATOMIC_KIND_ADDI = 2,
  // Floating-point addition.
  LOOM_ATOMIC_KIND_ADDF = 3,
  // Integer subtraction.
  LOOM_ATOMIC_KIND_SUBI = 4,
  // Bitwise AND.
  LOOM_ATOMIC_KIND_ANDI = 5,
  // Bitwise OR.
  LOOM_ATOMIC_KIND_ORI = 6,
  // Bitwise XOR.
  LOOM_ATOMIC_KIND_XORI = 7,
  // Signed integer minimum.
  LOOM_ATOMIC_KIND_MINSI = 8,
  // Signed integer maximum.
  LOOM_ATOMIC_KIND_MAXSI = 9,
  // Unsigned integer minimum.
  LOOM_ATOMIC_KIND_MINUI = 10,
  // Unsigned integer maximum.
  LOOM_ATOMIC_KIND_MAXUI = 11,
  // IEEE 754 floating-point minimum.
  LOOM_ATOMIC_KIND_MINIMUMF = 12,
  // IEEE 754 floating-point maximum.
  LOOM_ATOMIC_KIND_MAXIMUMF = 13,
  // C99 fmin-style floating-point minimum.
  LOOM_ATOMIC_KIND_MINNUMF = 14,
  // C99 fmax-style floating-point maximum.
  LOOM_ATOMIC_KIND_MAXNUMF = 15,
  LOOM_ATOMIC_KIND_COUNT_,
} loom_atomic_kind_t;

// Returns true when |kind| is a known loom_atomic_kind_t value.
bool loom_atomic_kind_is_valid(uint8_t kind);

// Returns true when |kind| operates on integer element types.
bool loom_atomic_kind_accepts_integer(uint8_t kind);

// Returns true when |kind| operates on floating-point element types.
bool loom_atomic_kind_accepts_float(uint8_t kind);

// Returns true when |kind| is an atomic exchange operation.
bool loom_atomic_kind_is_exchange(uint8_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_ATOMIC_H_
