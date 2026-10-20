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

// Logical register value produced while realizing one spill recipe. Recipe
// values are dense and local to one visit; the source value of a store uses a
// distinguished sentinel instead of consuming a recipe value index.
typedef uint32_t loom_amdgpu_spill_recipe_value_index_t;

#define LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE UINT32_MAX
#define LOOM_AMDGPU_SPILL_RECIPE_VALUE_SOURCE (UINT32_MAX - 1u)

typedef enum loom_amdgpu_spill_recipe_step_kind_e {
  // Reads the current EXEC mask into an SGPR pair.
  LOOM_AMDGPU_SPILL_RECIPE_STEP_EXEC_READ = 0,
  // Writes an all-active mask to EXEC.
  LOOM_AMDGPU_SPILL_RECIPE_STEP_EXEC_FULL = 1,
  // Projects one consecutive source-register chunk.
  LOOM_AMDGPU_SPILL_RECIPE_STEP_SLICE = 2,
  // Copies one SGPR unit into a VGPR transport value.
  LOOM_AMDGPU_SPILL_RECIPE_STEP_SGPR_TO_VGPR = 3,
  // Materializes a VGPR scratch address.
  LOOM_AMDGPU_SPILL_RECIPE_STEP_STORAGE_ADDRESS = 4,
  // Stores one register chunk through the selected scratch packet.
  LOOM_AMDGPU_SPILL_RECIPE_STEP_STORE = 5,
  // Loads one register chunk through the selected scratch packet.
  LOOM_AMDGPU_SPILL_RECIPE_STEP_LOAD = 6,
  // Copies the first lane of a VGPR transport value into one SGPR unit.
  LOOM_AMDGPU_SPILL_RECIPE_STEP_VGPR_TO_SGPR = 7,
  // Restores EXEC from the saved SGPR pair.
  LOOM_AMDGPU_SPILL_RECIPE_STEP_EXEC_RESTORE = 8,
  // Concatenates every completed reload chunk in transfer order.
  LOOM_AMDGPU_SPILL_RECIPE_STEP_CONCAT = 9,
} loom_amdgpu_spill_recipe_step_kind_t;

typedef uint8_t loom_amdgpu_spill_recipe_step_result_flags_t;
enum {
  // The result is defined while EXEC is widened beyond the semantic domain.
  LOOM_AMDGPU_SPILL_RECIPE_STEP_RESULT_FLAG_EXPANDED_EXECUTION = 1u << 0,
  // The result is private setup for its one recipe-local consumer.
  LOOM_AMDGPU_SPILL_RECIPE_STEP_RESULT_FLAG_PRIVATE_TO_CONSUMER = 1u << 1,
  // The result is one completed reload chunk retained through concatenation.
  LOOM_AMDGPU_SPILL_RECIPE_STEP_RESULT_FLAG_RELOAD_CHUNK = 1u << 2,
  // The result replaces the generic low.reload result after the recipe.
  LOOM_AMDGPU_SPILL_RECIPE_STEP_RESULT_FLAG_EXPORTED = 1u << 3,
};

// One operation in the exact target recipe expansion. Steps borrow their
// chunk from the compact recipe and are valid only for the duration of the
// visitor callback. CONCAT consumes all prior RELOAD_CHUNK results in transfer
// order; other register operands are listed explicitly.
typedef struct loom_amdgpu_spill_recipe_step_t {
  // Operation selected by the recipe.
  loom_amdgpu_spill_recipe_step_kind_t kind;
  // Facts attached to |result|, or zero for a resultless step.
  loom_amdgpu_spill_recipe_step_result_flags_t result_flags;
  // Number of initialized entries in |operands|.
  uint8_t operand_count;
  // Exact descriptor selected for a descriptor-backed step, or NONE for a
  // structural slice, storage address, or concat.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
  // Dense recipe-local result index, or VALUE_NONE for a resultless step.
  loom_amdgpu_spill_recipe_value_index_t result;
  // Recipe-local register operands or VALUE_SOURCE for the store source.
  loom_amdgpu_spill_recipe_value_index_t operands[2];
  // Chunk described by this step, or NULL for whole-recipe steps.
  const loom_amdgpu_spill_recipe_chunk_t* chunk;
} loom_amdgpu_spill_recipe_step_t;

// Fallible sink invoked once for every recipe step in emission order.
typedef iree_status_t (*loom_amdgpu_spill_recipe_step_fn_t)(
    void* user_data, const loom_amdgpu_spill_recipe_step_t* step);

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

// Visits the exact operation graph selected by |recipe|. The compact chunk
// rows remain the source of truth, and consumers derive the same deterministic
// expansion instead of retaining a second step table. Every produced register
// value receives one dense result index, including structural slice and concat
// results.
iree_status_t loom_amdgpu_spill_recipe_visit(
    const loom_amdgpu_spill_recipe_t* recipe,
    loom_amdgpu_spill_recipe_step_fn_t visit_step, void* user_data);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_SPILL_RECIPE_H_
