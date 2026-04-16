// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/atomic.h"

bool loom_atomic_kind_is_valid(uint8_t kind) {
  return kind < LOOM_ATOMIC_KIND_COUNT_;
}

bool loom_atomic_kind_accepts_integer(uint8_t kind) {
  switch (kind) {
    case LOOM_ATOMIC_KIND_XCHGI:
    case LOOM_ATOMIC_KIND_ADDI:
    case LOOM_ATOMIC_KIND_SUBI:
    case LOOM_ATOMIC_KIND_ANDI:
    case LOOM_ATOMIC_KIND_ORI:
    case LOOM_ATOMIC_KIND_XORI:
    case LOOM_ATOMIC_KIND_MINSI:
    case LOOM_ATOMIC_KIND_MAXSI:
    case LOOM_ATOMIC_KIND_MINUI:
    case LOOM_ATOMIC_KIND_MAXUI:
      return true;
    default:
      return false;
  }
}

bool loom_atomic_kind_accepts_float(uint8_t kind) {
  switch (kind) {
    case LOOM_ATOMIC_KIND_XCHGF:
    case LOOM_ATOMIC_KIND_ADDF:
    case LOOM_ATOMIC_KIND_MINIMUMF:
    case LOOM_ATOMIC_KIND_MAXIMUMF:
    case LOOM_ATOMIC_KIND_MINNUMF:
    case LOOM_ATOMIC_KIND_MAXNUMF:
      return true;
    default:
      return false;
  }
}

bool loom_atomic_kind_is_exchange(uint8_t kind) {
  return kind == LOOM_ATOMIC_KIND_XCHGI || kind == LOOM_ATOMIC_KIND_XCHGF;
}
