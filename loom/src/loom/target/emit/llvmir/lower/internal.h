// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Internal lowering state and declarations shared by the LLVMIR lowering
// implementation files. Not a public API.

#ifndef LOOM_TARGET_LLVMIR_LOWER_INTERNAL_H_
#define LOOM_TARGET_LLVMIR_LOWER_INTERNAL_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/module.h"
#include "loom/target/emit/llvmir/builder.h"
#include "loom/target/emit/llvmir/lower.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_llvmir_lowering_intrinsic_cache_key_t {
  // Source llvmir.intrinsic kind string id.
  loom_string_id_t kind_id;
  // First intrinsic overload discriminator, or zero when unused.
  uint32_t discriminator0;
  // Second intrinsic overload discriminator, or zero when unused.
  uint32_t discriminator1;
  // Third intrinsic overload discriminator, or zero when unused.
  uint32_t discriminator2;
} loom_llvmir_lowering_intrinsic_cache_key_t;

typedef iree_status_t (*loom_llvmir_lowering_provider_intrinsic_decl_fn_t)(
    loom_llvmir_module_t* module, loom_llvmir_function_t** out_function);

typedef enum loom_llvmir_lowering_state_flag_bits_e {
  LOOM_LLVMIR_LOWERING_STATE_FLAG_FUNCTION_FACT_TABLE_INITIALIZED = 1u << 0,
} loom_llvmir_lowering_state_flag_bits_t;
typedef uint32_t loom_llvmir_lowering_state_flags_t;

typedef struct loom_llvmir_lowering_value_state_t {
  // Target LLVMIR value mapped from the source Loom value.
  loom_llvmir_value_id_t target_value_id;
  // Pointer address space, or UINT32_MAX when the value is not a pointer.
  uint32_t pointer_address_space;
  // Pointer minimum alignment, or zero when unknown or not a pointer.
  uint64_t pointer_alignment;
} loom_llvmir_lowering_value_state_t;

struct loom_llvmir_lowering_state_t {
  // Source Loom module being lowered.
  const loom_module_t* source_module;
  // Target structured LLVM IR module being populated.
  loom_llvmir_module_t* target_module;
  // Target/ABI profile selected for this lowering run.
  const loom_llvmir_target_profile_t* target_profile;
  // Host allocator used for temporary lowering maps.
  iree_allocator_t allocator;
  // Optional target-specific lowering providers.
  const loom_llvmir_lowering_provider_t* const* providers;
  // Number of provider pointers in |providers|.
  iree_host_size_t provider_count;
  // Lowering state flags.
  loom_llvmir_lowering_state_flags_t flags;
  // Whole-module direct map from source value ID to LLVMIR lowering state.
  loom_llvmir_lowering_value_state_t* values;
  // Number of entries in |values|.
  iree_host_size_t value_count;
  // Map from source module symbol id to target LLVMIR function object.
  loom_llvmir_function_t** symbol_functions;
  // Number of entries in |symbol_functions|.
  iree_host_size_t symbol_function_count;
  // Block pool for reusable function-scope value-fact storage.
  iree_arena_block_pool_t fact_block_pool;
  // Arena for reusable direct-address value-fact entries.
  iree_arena_allocator_t fact_storage_arena;
  // Arena for function-scope extension payloads and fact inference scratch.
  iree_arena_allocator_t fact_transient_arena;
  // Reusable function-scope fact table populated while lowering a body.
  loom_value_fact_table_t function_fact_table;
  // Function-local fact table active while lowering a body, or NULL.
  const loom_value_fact_table_t* fact_table;
  // Cached profile kernel attribute group, or INVALID until materialized.
  loom_llvmir_attr_group_id_t kernel_attr_group_id;
  // Cached !{i32 1} node used by LLVM's !nontemporal metadata.
  loom_llvmir_metadata_id_t nontemporal_metadata_id;
  // Address spaces for cached llvm.prefetch declarations.
  uint32_t prefetch_function_address_spaces[8];
  // Cached llvm.prefetch declarations keyed by address space.
  loom_llvmir_function_t* prefetch_functions[8];
  // Number of cached llvm.prefetch declarations.
  iree_host_size_t prefetch_function_count;
  // Cached source intrinsic declaration keys.
  loom_llvmir_lowering_intrinsic_cache_key_t intrinsic_function_keys[32];
  // Cached source intrinsic declarations.
  loom_llvmir_function_t* intrinsic_functions[32];
  // Number of cached source intrinsic declarations.
  iree_host_size_t intrinsic_function_count;
  // Cached target-provider intrinsic keys.
  const void* provider_intrinsic_keys[64];
  // Cached target-provider LLVM intrinsic declarations.
  loom_llvmir_function_t* provider_intrinsic_functions[64];
  // Number of cached target-provider LLVM intrinsic declarations.
  iree_host_size_t provider_intrinsic_function_count;
};

iree_string_view_t loom_llvmir_lowering_value_name(
    const loom_llvmir_lowering_state_t* state, loom_value_id_t value_id);

iree_status_t loom_llvmir_lowering_unsupported_op(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    const char* detail);

iree_status_t loom_llvmir_lowering_unsupported_type(
    const loom_llvmir_lowering_state_t* state, loom_type_t type,
    const char* detail);

bool loom_llvmir_lowering_lookup_provider_intrinsic(
    const loom_llvmir_lowering_state_t* state, const void* key,
    loom_llvmir_function_t** out_function);

iree_status_t loom_llvmir_lowering_cache_provider_intrinsic(
    loom_llvmir_lowering_state_t* state, const void* key,
    loom_llvmir_function_t* function);

iree_status_t loom_llvmir_lowering_declare_provider_intrinsic_cached(
    loom_llvmir_lowering_state_t* state, const void* key,
    loom_llvmir_lowering_provider_intrinsic_decl_fn_t declare_fn,
    loom_llvmir_function_t** out_function);

iree_status_t loom_llvmir_lowering_string_attr(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    iree_string_view_t attr_name, loom_string_id_t string_id,
    iree_string_view_t* out_string);

iree_status_t loom_llvmir_lowering_expect_intrinsic_shape(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    iree_host_size_t operand_count, iree_host_size_t result_count,
    const char* detail);

iree_status_t loom_llvmir_lowering_expect_scalar_result(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_scalar_type_t expected_type, const char* detail);

iree_status_t loom_llvmir_lowering_try_provider_op(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op, bool* out_handled);

iree_status_t loom_llvmir_lowering_lower_declared_call(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op, loom_llvmir_function_t* function);

iree_status_t loom_llvmir_lowering_lower_scalar_type(
    loom_llvmir_lowering_state_t* state, loom_scalar_type_t scalar_type,
    loom_llvmir_type_id_t* out_type_id);

iree_status_t loom_llvmir_lowering_get_pointer_type(
    loom_llvmir_lowering_state_t* state, uint32_t address_space,
    loom_llvmir_type_id_t* out_type_id);

iree_status_t loom_llvmir_lowering_lower_type(
    loom_llvmir_lowering_state_t* state, loom_type_t type,
    loom_llvmir_type_id_t* out_type_id);

iree_status_t loom_llvmir_lowering_map_value(
    loom_llvmir_lowering_state_t* state, loom_value_id_t source_value_id,
    loom_llvmir_value_id_t target_value_id);

iree_status_t loom_llvmir_lowering_map_pointer_value(
    loom_llvmir_lowering_state_t* state, loom_value_id_t source_value_id,
    loom_llvmir_value_id_t target_value_id, uint32_t address_space,
    uint64_t minimum_alignment);

iree_status_t loom_llvmir_lowering_lookup_value(
    const loom_llvmir_lowering_state_t* state, loom_value_id_t source_value_id,
    loom_llvmir_value_id_t* out_target_value_id);

iree_status_t loom_llvmir_lowering_lookup_pointer(
    const loom_llvmir_lowering_state_t* state, loom_value_id_t source_value_id,
    loom_llvmir_value_id_t* out_target_value_id, uint32_t* out_address_space,
    uint64_t* out_minimum_alignment);

iree_status_t loom_llvmir_lowering_lookup_operands(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_llvmir_value_id_t* target_operands);

iree_status_t loom_llvmir_lowering_map_single_result(
    loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_llvmir_value_id_t target_value_id);

iree_status_t loom_llvmir_lowering_lower_binop(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_negf(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_constant(
    loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_attribute_t value_attr);

iree_status_t loom_llvmir_lowering_lower_index_madd(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_index_minmax(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_icmp(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op, uint8_t source_predicate);

iree_status_t loom_llvmir_lowering_lower_fcmp(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_cast(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_index_cast(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_select(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_alloca(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_assume_memory_space(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_buffer_view(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_subview(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_view_load(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_view_store(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_vector_load(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_vector_store(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_view_prefetch(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_inline_asm(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_intrinsic(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_vector_binop(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_vector_negf(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_vector_constant(
    loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_attribute_t value_attr);

iree_status_t loom_llvmir_lowering_lower_vector_poison(
    loom_llvmir_lowering_state_t* state, const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_vector_splat(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_vector_from_elements(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_vector_extract(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_vector_insert(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_vector_shuffle(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_vector_icmp(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_vector_fcmp(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_vector_select(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

iree_status_t loom_llvmir_lowering_lower_vector_cast(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_LLVMIR_LOWER_INTERNAL_H_
