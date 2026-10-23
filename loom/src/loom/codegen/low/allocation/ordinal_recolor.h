// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Atomic recoloring of hard same-register-ordinal allocation components.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_ORDINAL_RECOLOR_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_ORDINAL_RECOLOR_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation/search.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/local_value_domain.h"

#ifdef __cplusplus
extern "C" {
#endif

// Tries to free a physical-register candidate for |requested_interval| by
// atomically changing the semantic register ordinal of one blocking hard
// same-register-ordinal component. Every published member and every future
// member must remain legal across its complete lifetime. Recoloring applies
// only when exact-storage aliases preserve one physical class and all tuple
// classes share a candidate count, so one ordinal identifies every replacement.
// A successful repair returns the newly available location in
// |out_location_base|.
iree_status_t loom_low_allocation_ordinal_recolor_try(
    const loom_local_value_domain_t* value_domain,
    const loom_low_storage_transport_t* storage_transport,
    loom_low_allocation_search_context_t* search_context,
    loom_low_allocation_assignment_map_t* assignment_map,
    loom_low_allocation_assignment_t* assignments,
    iree_arena_allocator_t* scratch_arena,
    const loom_liveness_interval_t* requested_interval,
    loom_value_ordinal_t requested_value_ordinal,
    const loom_low_allocation_class_capacity_t* requested_capacity,
    uint32_t* out_location_base, bool* out_recolored);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_ORDINAL_RECOLOR_H_
