// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Descriptor register-class resolution for target-low type checking.
//
// Loom register types store compact target-low descriptor identity. This helper
// validates that a type belongs to a descriptor set and returns the
// descriptor-set-local register-class row.

#ifndef LOOM_CODEGEN_LOW_REGISTER_CLASS_MAP_H_
#define LOOM_CODEGEN_LOW_REGISTER_CLASS_MAP_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_register_class_map_t {
  // Descriptor set defining the resolved descriptor register-class IDs.
  const loom_low_descriptor_set_t* descriptor_set;
} loom_low_register_class_map_t;

// Initializes |out_map| to borrow |descriptor_set|.
iree_status_t loom_low_register_class_map_initialize(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena, loom_low_register_class_map_t* out_map);

// Resolves a Loom register type to a descriptor-set-local register class.
// |out_descriptor_register_class| may be NULL when only the dense descriptor ID
// is needed.
// Returns |out_found| false when |type| is not a register type or its class is
// not defined by the descriptor set.
iree_status_t loom_low_register_class_map_try_resolve_type(
    const loom_low_register_class_map_t* map, loom_type_t type,
    uint16_t* out_descriptor_register_class_id,
    const loom_low_reg_class_t** out_descriptor_register_class,
    bool* out_found);

// Looks up a descriptor-set-local register class by stable register-class name.
// |out_descriptor_register_class| may be NULL when only the dense descriptor ID
// is needed.
// This is intended for textual/user configuration boundaries, not repeated IR
// walks.
iree_status_t loom_low_register_class_try_lookup_name(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t register_class_name,
    uint16_t* out_descriptor_register_class_id,
    const loom_low_reg_class_t** out_descriptor_register_class,
    bool* out_found);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_REGISTER_CLASS_MAP_H_
