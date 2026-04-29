// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Function/region-local value domains backed by module ordinal scratch.
//
// A local value domain acquires a dense value_id -> value_ordinal map in the
// module-owned scratch table for one compiler frame. The module owns the large
// reusable scratch storage; the domain owns only the compact ordinal ->
// value_id list needed for cleanup, iteration, diagnostics, and dense phase
// tables.

#ifndef LOOM_IR_LOCAL_VALUE_DOMAIN_H_
#define LOOM_IR_LOCAL_VALUE_DOMAIN_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

enum loom_local_value_domain_flag_bits_e {
  // The domain currently owns the module value-ordinal scratch map.
  LOOM_LOCAL_VALUE_DOMAIN_FLAG_ACQUIRED = 1u << 0,
};
typedef uint16_t loom_local_value_domain_flags_t;

typedef struct loom_local_value_domain_t {
  // Module whose value-ordinal scratch map stores this domain.
  loom_module_t* module;
  // Region whose local values are covered by this domain.
  const loom_region_t* region;
  // Function/region-local value IDs indexed by local value ordinal.
  loom_value_id_t* value_ids;
  // Number of initialized value IDs in value_ids.
  loom_value_ordinal_t value_count;
  // Allocated capacity of value_ids.
  iree_host_size_t value_capacity;
  // Domain lifecycle flags.
  loom_local_value_domain_flags_t flags;
} loom_local_value_domain_t;

// Acquires a local value domain for |region| in |module|'s ordinal scratch.
//
// The module and region must stay semantically immutable until release. The
// domain registers block arguments, op results, op operands, SSA references
// carried by value types, and values captured by nested regions.
iree_status_t loom_local_value_domain_acquire_for_region(
    loom_module_t* module, const loom_region_t* region,
    iree_arena_allocator_t* arena, loom_local_value_domain_t* out_domain);

// Clears all acquired value IDs and releases the module ordinal scratch map.
void loom_local_value_domain_release(loom_local_value_domain_t* domain);

// Returns true while the domain owns the module ordinal scratch map.
static inline bool loom_local_value_domain_is_acquired(
    const loom_local_value_domain_t* domain) {
  IREE_ASSERT_ARGUMENT(domain);
  return iree_any_bit_set(domain->flags, LOOM_LOCAL_VALUE_DOMAIN_FLAG_ACQUIRED);
}

// Returns the local ordinal for |value_id|. This is an invariant-checked direct
// array lookup: callers must only ask for values covered by the acquired
// domain.
static inline loom_value_ordinal_t loom_local_value_domain_ordinal(
    const loom_local_value_domain_t* domain, loom_value_id_t value_id) {
  IREE_ASSERT_ARGUMENT(domain);
  IREE_ASSERT(
      iree_any_bit_set(domain->flags, LOOM_LOCAL_VALUE_DOMAIN_FLAG_ACQUIRED));
  IREE_ASSERT_LT(value_id, domain->module->value_ordinal_scratch.capacity);
  const loom_value_ordinal_t value_ordinal =
      domain->module->value_ordinal_scratch.ordinals_by_value_id[value_id];
  IREE_ASSERT_NE(value_ordinal, LOOM_VALUE_ORDINAL_INVALID);
  IREE_ASSERT_LT(value_ordinal, domain->value_count);
  return value_ordinal;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_IR_LOCAL_VALUE_DOMAIN_H_
