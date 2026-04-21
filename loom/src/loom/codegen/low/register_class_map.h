// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Descriptor register-class resolution for target-low type checking.
//
// Loom register types store a module string ID for their register-class name.
// Target-low descriptor consumers need descriptor-set-local integer IDs. This
// helper builds that bridge once per module/descriptor-set pair so verifier,
// scheduler, allocator, and emitters do not perform repeated string identity
// checks while walking IR.

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
  // Module whose string table indexes descriptor_register_class_ids.
  const loom_module_t* module;
  // Descriptor set defining the resolved descriptor register-class IDs.
  const loom_low_descriptor_set_t* descriptor_set;
  // Descriptor register-class IDs indexed by module string ID.
  uint16_t* descriptor_register_class_ids;
  // Number of entries in descriptor_register_class_ids.
  iree_host_size_t descriptor_register_class_id_count;
} loom_low_register_class_map_t;

// Builds |out_map| from the module string table to descriptor-set-local
// register-class IDs. The returned map borrows |module| and |descriptor_set|;
// only the dense ID table is allocated from |arena|.
iree_status_t loom_low_register_class_map_initialize(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena, loom_low_register_class_map_t* out_map);

// Resolves a module string ID to a descriptor-set-local register class.
// |out_descriptor_register_class| may be NULL when only the dense descriptor ID
// is needed.
// Returns |out_found| false when the string ID is invalid or names no register
// class in the descriptor set.
iree_status_t loom_low_register_class_map_try_resolve_string_id(
    const loom_low_register_class_map_t* map,
    loom_string_id_t register_class_string_id,
    uint16_t* out_descriptor_register_class_id,
    const loom_low_reg_class_t** out_descriptor_register_class,
    bool* out_found);

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
