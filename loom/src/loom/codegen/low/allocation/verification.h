// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocation table consistency verification.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_VERIFICATION_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_VERIFICATION_H_

#include "iree/base/api.h"
#include "loom/codegen/low/allocation/table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Verifies allocation-table consistency. Register-like assignments must not
// overlap on the same physical register or target ID range. Spill slots are not
// treated as registers by this verifier because materialized
// low.spill/low.reload insertion owns their eventual storage reuse policy.
// Tied-result placement is enforced only when both sides are register-like; a
// spill-slot side defers the tie until spill materialization inserts reloads
// and final allocation runs on the materialized IR.
iree_status_t loom_low_allocation_verify_table(
    const loom_low_allocation_table_t* table);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_VERIFICATION_H_
