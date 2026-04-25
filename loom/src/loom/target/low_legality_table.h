// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.target_low_legality.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables
// clang-format off
#ifndef LOOM_TARGET_LOW_LEGALITY_TABLE_H_
#define LOOM_TARGET_LOW_LEGALITY_TABLE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t loom_target_low_legality_t;

typedef enum loom_target_low_legality_e {
  LOOM_TARGET_LOW_LEGALITY_UNSUPPORTED = 0,
  LOOM_TARGET_LOW_LEGALITY_CORE = 1,
  LOOM_TARGET_LOW_LEGALITY_PROVIDER = 2,
  LOOM_TARGET_LOW_LEGALITY_SOURCE_ONLY = 3,
  LOOM_TARGET_LOW_LEGALITY_MODULE_METADATA = 4,
} loom_target_low_legality_e;

// Returns the target-low legality class for |kind|.
loom_target_low_legality_t loom_target_low_legality_class(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_LOW_LEGALITY_TABLE_H_
