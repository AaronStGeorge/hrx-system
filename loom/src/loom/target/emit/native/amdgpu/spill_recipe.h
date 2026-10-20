// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU register and scratch-packet recipe for one spill access.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_SPILL_RECIPE_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_SPILL_RECIPE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/ir.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/emit/native/amdgpu/storage_layout.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_amdgpu_spill_recipe_kind_e {
  LOOM_AMDGPU_SPILL_RECIPE_KIND_STORE = 0,
  LOOM_AMDGPU_SPILL_RECIPE_KIND_RELOAD = 1,
} loom_amdgpu_spill_recipe_kind_t;

typedef enum loom_amdgpu_spill_register_kind_e {
  LOOM_AMDGPU_SPILL_REGISTER_KIND_UNSUPPORTED = 0,
  LOOM_AMDGPU_SPILL_REGISTER_KIND_SGPR = 1,
  LOOM_AMDGPU_SPILL_REGISTER_KIND_PER_LANE = 2,
} loom_amdgpu_spill_register_kind_t;

typedef uint8_t loom_amdgpu_spill_recipe_chunk_flags_t;
enum {
  // Materializes a VGPR storage address consumed by the memory packet.
  LOOM_AMDGPU_SPILL_RECIPE_CHUNK_FLAG_STORAGE_ADDRESS = 1u << 0,
};

// One aligned memory packet in a spill recipe.
typedef struct loom_amdgpu_spill_recipe_chunk_t {
  // Offset relative to the referenced low storage handle.
  uint64_t storage_byte_offset;
  // Immediate encoded on the selected packet.
  int64_t packet_byte_offset;
  // First allocation unit transferred from or to the spilled value.
  uint32_t unit_start;
  // Exact generated descriptor selected for this memory packet.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
  // Number of 32-bit allocation units transferred by the packet.
  uint8_t unit_count;
  // Additional materialization selected for this chunk.
  loom_amdgpu_spill_recipe_chunk_flags_t flags;
} loom_amdgpu_spill_recipe_chunk_t;

// Complete target decision for one low.spill or low.reload access.
typedef struct loom_amdgpu_spill_recipe_t {
  // Store or reload recipe kind.
  loom_amdgpu_spill_recipe_kind_t kind;
  // Register-bank behavior selected from |value_type|.
  loom_amdgpu_spill_register_kind_t register_kind;
  // Register type transferred by the complete access.
  loom_type_t value_type;
  // Arena-owned chunks in transfer order.
  const loom_amdgpu_spill_recipe_chunk_t* chunks;
  // Number of records in |chunks|.
  uint32_t chunk_count;
} loom_amdgpu_spill_recipe_t;

// Classifies one register type for AMDGPU spill lowering.
loom_amdgpu_spill_register_kind_t loom_amdgpu_spill_recipe_register_kind(
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t type);

// Selects the exact chunk and packet recipe for a validated spill access.
// |operation_byte_offset| and the full value byte length must fit within
// |storage_reference|. The recipe retains stable descriptor references and
// owns its compact chunk array in |arena|.
iree_status_t loom_amdgpu_spill_recipe_build(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_spill_recipe_kind_t kind, loom_type_t value_type,
    const loom_amdgpu_storage_layout_reference_t* storage_reference,
    int64_t operation_byte_offset, iree_arena_allocator_t* arena,
    loom_amdgpu_spill_recipe_t* out_recipe);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_SPILL_RECIPE_H_
