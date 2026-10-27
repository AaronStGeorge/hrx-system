// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pass-local interning and publication of anonymous immutable storage.

#ifndef LOOM_OPS_GLOBAL_RODATA_POOL_H_
#define LOOM_OPS_GLOBAL_RODATA_POOL_H_

#include "iree/base/internal/arena.h"
#include "loom/ir/attribute.h"
#include "loom/ir/location.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_rodata_pool_t loom_rodata_pool_t;

// Private payload identity, stable until the pool's arena is released. An ID
// is not a module symbol and cannot appear in IR.
typedef uint32_t loom_rodata_id_t;

// Allocates a pool owned by |arena|. Planning can intern bytes without changing
// any module; publication and finalization must use one destination module.
iree_status_t loom_rodata_pool_allocate(iree_arena_allocator_t* arena,
                                        loom_rodata_pool_t** out_pool);

// Copies a verified BYTES attribute into pool storage. Equal byte contents
// share an ID and retain the strongest power-of-two alignment and explicit
// UTF-8 form, independently of insertion order. Named globals are never pooled.
iree_status_t loom_rodata_pool_intern(loom_rodata_pool_t* pool,
                                      loom_attribute_t contents,
                                      uint64_t minimum_alignment,
                                      loom_location_id_t location,
                                      loom_rodata_id_t* out_id);

// Reserves a module symbol when execution first needs the payload. Planning
// does not publish symbols. The definition is created during finalization.
iree_status_t loom_rodata_pool_reference(loom_rodata_pool_t* pool,
                                         struct loom_module_t* module,
                                         loom_rodata_id_t id,
                                         loom_symbol_ref_t* out_symbol);

// Materializes every interned payload as a global.rodata.def. The destination
// module owns the resulting bytes independently of the pool's arena. Repeated
// finalization has no effect; interning must finish before finalization.
iree_status_t loom_rodata_pool_finalize(loom_rodata_pool_t* pool,
                                        struct loom_module_t* module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_GLOBAL_RODATA_POOL_H_
