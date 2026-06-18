// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/sanitizer.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/module.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/sanitizer/site_table.h"
#include "loom/target/arch/amdgpu/abi/tsan.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_access.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/registers.h"

typedef struct loom_amdgpu_sanitizer_site_map_row_t {
  // Borrowed source operation pointer used for local lookup only.
  const loom_op_t* op;
  // Dense sanitizer site ID assigned by final collection order.
  loom_sanitizer_site_id_t site_id;
} loom_amdgpu_sanitizer_site_map_row_t;

typedef struct loom_amdgpu_sanitizer_site_chunk_t {
  // Next committed site-row chunk, or NULL for the final chunk.
  struct loom_amdgpu_sanitizer_site_chunk_t* next;
  // Rows copied from one successfully lowered source function.
  loom_sanitizer_site_row_t* rows;
  // Number of rows in rows.
  iree_host_size_t row_count;
} loom_amdgpu_sanitizer_site_chunk_t;

typedef struct loom_amdgpu_sanitizer_module_state_t {
  // First committed site-row chunk, or NULL when no sites were committed.
  loom_amdgpu_sanitizer_site_chunk_t* site_chunk_head;
  // Final committed site-row chunk, or NULL when no sites were committed.
  loom_amdgpu_sanitizer_site_chunk_t* site_chunk_tail;
  // Total number of committed site rows across all chunks.
  iree_host_size_t site_row_count;
} loom_amdgpu_sanitizer_module_state_t;

typedef struct loom_amdgpu_sanitizer_lower_state_t {
  // True once the source function sanitizer site collection has been built.
  bool has_site_collection;
  // First module-global site ID assigned to this function's collection.
  loom_sanitizer_site_id_t site_id_base;
  // Function-local sanitizer site rows in report-site order.
  loom_sanitizer_site_collection_t site_collection;
  // Lookup rows sorted by borrowed source operation pointer.
  loom_amdgpu_sanitizer_site_map_row_t* site_map_rows;
  // Number of entries in site_map_rows.
  iree_host_size_t site_map_row_count;
  // True once the runtime feedback config symbol has been looked up or created.
  bool has_feedback_config_symbol;
  // Module-local feedback channel configuration symbol.
  loom_symbol_ref_t feedback_config_symbol;
  // True once the runtime ASAN shadow config symbol has been looked up or
  // created.
  bool has_asan_config_symbol;
  // Module-local ASAN shadow configuration symbol.
  loom_symbol_ref_t asan_config_symbol;
  // True once the runtime TSAN shadow config symbol has been looked up or
  // created.
  bool has_tsan_config_symbol;
  // Module-local TSAN shadow configuration symbol.
  loom_symbol_ref_t tsan_config_symbol;
  // True once read access failures have a shared cold report island.
  bool has_read_island;
  // Shared cold island for read access failures.
  loom_amdgpu_sanitizer_access_report_island_t read_island;
  // True once write access failures have a shared cold report island.
  bool has_write_island;
  // Shared cold island for write access failures.
  loom_amdgpu_sanitizer_access_report_island_t write_island;
  // True once atomic access failures have a shared cold report island.
  bool has_atomic_island;
  // Shared cold island for atomic access failures.
  loom_amdgpu_sanitizer_access_report_island_t atomic_island;
} loom_amdgpu_sanitizer_lower_state_t;

static int loom_amdgpu_sanitizer_lower_state_key;
static int loom_amdgpu_sanitizer_module_state_key;

static int loom_amdgpu_sanitizer_site_map_compare(const void* lhs,
                                                  const void* rhs) {
  const loom_amdgpu_sanitizer_site_map_row_t* lhs_row =
      (const loom_amdgpu_sanitizer_site_map_row_t*)lhs;
  const loom_amdgpu_sanitizer_site_map_row_t* rhs_row =
      (const loom_amdgpu_sanitizer_site_map_row_t*)rhs;
  const uintptr_t lhs_key = (uintptr_t)lhs_row->op;
  const uintptr_t rhs_key = (uintptr_t)rhs_row->op;
  if (lhs_key < rhs_key) return -1;
  if (lhs_key > rhs_key) return 1;
  return 0;
}

static iree_status_t loom_amdgpu_sanitizer_get_or_create_symbol(
    loom_module_t* module, iree_string_view_t name,
    loom_symbol_ref_t* out_symbol_ref) {
  *out_symbol_ref = loom_symbol_ref_null();
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, name, &name_id));
  uint16_t symbol_id = loom_module_find_symbol(module, name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name_id, &symbol_id));
  }
  *out_symbol_ref = (loom_symbol_ref_t){
      .module_id = 0,
      .symbol_id = symbol_id,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_lower_state(
    loom_low_lower_context_t* context,
    loom_amdgpu_sanitizer_lower_state_t** out_state) {
  *out_state = NULL;
  loom_amdgpu_sanitizer_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_sanitizer_lower_state_key, sizeof(*state),
      (void**)&state));
  *out_state = state;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_module_state_from_context(
    loom_low_lower_context_t* context,
    loom_amdgpu_sanitizer_module_state_t** out_state) {
  *out_state = NULL;
  loom_amdgpu_sanitizer_module_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_module_target_state(
      context, &loom_amdgpu_sanitizer_module_state_key, sizeof(*state),
      (void**)&state));
  *out_state = state;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_module_state_from_module_state(
    loom_low_lower_module_state_t* module_state,
    loom_amdgpu_sanitizer_module_state_t** out_state) {
  *out_state = NULL;
  loom_amdgpu_sanitizer_module_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_module_state_get_or_allocate(
      module_state, &loom_amdgpu_sanitizer_module_state_key, sizeof(*state),
      (void**)&state));
  *out_state = state;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_ensure_feedback_config_symbol(
    loom_low_lower_context_t* context,
    loom_amdgpu_sanitizer_lower_state_t* state) {
  if (!state->has_feedback_config_symbol) {
    loom_module_t* module = loom_low_lower_context_module(context);
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_get_or_create_symbol(
        module, IREE_SV("iree_feedback_config"),
        &state->feedback_config_symbol));
    state->has_feedback_config_symbol = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_ensure_asan_config_symbol(
    loom_low_lower_context_t* context,
    loom_amdgpu_sanitizer_lower_state_t* state) {
  if (!state->has_asan_config_symbol) {
    loom_module_t* module = loom_low_lower_context_module(context);
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_get_or_create_symbol(
        module, IREE_SV("iree_asan_config"), &state->asan_config_symbol));
    state->has_asan_config_symbol = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_ensure_tsan_config_symbol(
    loom_low_lower_context_t* context,
    loom_amdgpu_sanitizer_lower_state_t* state) {
  if (!state->has_tsan_config_symbol) {
    loom_module_t* module = loom_low_lower_context_module(context);
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_get_or_create_symbol(
        module, IREE_SV(LOOM_AMDGPU_TSAN_CONFIG_GLOBAL_NAME),
        &state->tsan_config_symbol));
    state->has_tsan_config_symbol = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_ensure_config_symbols(
    loom_low_lower_context_t* context,
    loom_amdgpu_sanitizer_lower_state_t* state) {
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_ensure_feedback_config_symbol(context, state));
  return loom_amdgpu_sanitizer_ensure_asan_config_symbol(context, state);
}

iree_status_t loom_amdgpu_sanitizer_feedback_config_symbol(
    loom_low_lower_context_t* context, loom_symbol_ref_t* out_symbol_ref) {
  *out_symbol_ref = loom_symbol_ref_null();
  loom_amdgpu_sanitizer_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_lower_state(context, &state));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_ensure_feedback_config_symbol(context, state));
  *out_symbol_ref = state->feedback_config_symbol;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_sanitizer_tsan_config_symbol(
    loom_low_lower_context_t* context, loom_symbol_ref_t* out_symbol_ref) {
  *out_symbol_ref = loom_symbol_ref_null();
  loom_amdgpu_sanitizer_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_lower_state(context, &state));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_ensure_tsan_config_symbol(context, state));
  *out_symbol_ref = state->tsan_config_symbol;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_ensure_site_collection(
    loom_low_lower_context_t* context,
    loom_amdgpu_sanitizer_lower_state_t* state) {
  if (state->has_site_collection) return iree_ok_status();

  loom_amdgpu_sanitizer_module_state_t* module_state = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_module_state_from_context(context, &module_state));

  loom_module_t* module = loom_low_lower_context_module(context);
  IREE_RETURN_IF_ERROR(loom_sanitizer_site_collection_build_function(
      module, loom_low_lower_context_source_function(context),
      loom_low_lower_context_scratch_arena(context), &state->site_collection));
  const iree_host_size_t row_count = state->site_collection.row_count;
  iree_host_size_t total_row_count = 0;
  if (!iree_host_size_checked_add(module_state->site_row_count, row_count,
                                  &total_row_count) ||
      total_row_count > LOOM_SANITIZER_SITE_ID_INVALID) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AMDGPU sanitizer site table exceeded max site id %u",
        LOOM_SANITIZER_SITE_ID_INVALID - 1);
  }
  state->site_id_base = (loom_sanitizer_site_id_t)module_state->site_row_count;
  state->site_map_row_count = row_count;
  if (row_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
        context, row_count, sizeof(*state->site_map_rows),
        (void**)&state->site_map_rows));
    for (iree_host_size_t i = 0; i < row_count; ++i) {
      state->site_map_rows[i] = (loom_amdgpu_sanitizer_site_map_row_t){
          .op = state->site_collection.rows[i].op,
          .site_id = (loom_sanitizer_site_id_t)(state->site_id_base + i),
      };
      state->site_collection.rows[i].site_id = state->site_map_rows[i].site_id;
    }
    qsort(state->site_map_rows, row_count, sizeof(*state->site_map_rows),
          loom_amdgpu_sanitizer_site_map_compare);
    for (iree_host_size_t i = 1; i < row_count; ++i) {
      if (state->site_map_rows[i - 1].op == state->site_map_rows[i].op) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "sanitizer site collection mapped one op to multiple site ids");
      }
    }
  }

  state->has_site_collection = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_finalize_sanitizer_function(
    loom_low_lower_context_t* context) {
  loom_amdgpu_sanitizer_lower_state_t* function_state = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_lower_state(context, &function_state));
  if (!function_state->has_site_collection ||
      function_state->site_collection.row_count == 0) {
    return iree_ok_status();
  }

  loom_amdgpu_sanitizer_module_state_t* module_state = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_module_state_from_context(context, &module_state));
  if (module_state->site_row_count != function_state->site_id_base) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU sanitizer site rows committed out of order");
  }

  const iree_host_size_t row_count = function_state->site_collection.row_count;
  loom_low_lower_module_state_t* lower_module_state =
      loom_low_lower_context_module_state(context);
  loom_amdgpu_sanitizer_site_chunk_t* chunk = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_module_state_allocate(
      lower_module_state, sizeof(*chunk), (void**)&chunk));
  memset(chunk, 0, sizeof(*chunk));
  IREE_RETURN_IF_ERROR(loom_low_lower_module_state_allocate_array(
      lower_module_state, row_count, sizeof(*chunk->rows),
      (void**)&chunk->rows));
  for (iree_host_size_t i = 0; i < row_count; ++i) {
    chunk->rows[i] = function_state->site_collection.rows[i];
    chunk->rows[i].op = NULL;
  }
  chunk->row_count = row_count;

  if (module_state->site_chunk_tail != NULL) {
    module_state->site_chunk_tail->next = chunk;
  } else {
    module_state->site_chunk_head = chunk;
  }
  module_state->site_chunk_tail = chunk;
  module_state->site_row_count += row_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_flatten_site_rows(
    const loom_amdgpu_sanitizer_module_state_t* module_state,
    iree_arena_allocator_t* scratch_arena,
    loom_sanitizer_site_collection_t* out_collection) {
  *out_collection = (loom_sanitizer_site_collection_t){0};
  const iree_host_size_t row_count = module_state->site_row_count;
  out_collection->row_count = row_count;
  if (row_count == 0) return iree_ok_status();

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, row_count, sizeof(*out_collection->rows),
      (void**)&out_collection->rows));
  iree_host_size_t row_index = 0;
  for (const loom_amdgpu_sanitizer_site_chunk_t* chunk =
           module_state->site_chunk_head;
       chunk != NULL; chunk = chunk->next) {
    memcpy(out_collection->rows + row_index, chunk->rows,
           chunk->row_count * sizeof(*out_collection->rows));
    row_index += chunk->row_count;
  }
  if (row_index != row_count) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU sanitizer site chunk row count changed during finalization");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_emit_site_table_rodata(
    loom_module_t* module, iree_const_byte_span_t site_table) {
  loom_symbol_ref_t symbol_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_get_or_create_symbol(
      module, IREE_SV(LOOM_SANITIZER_SITE_TABLE_SYMBOL_NAME), &symbol_ref));
  loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->defining_op != NULL) {
    return iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "AMDGPU sanitizer site table symbol is already defined");
  }

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  loom_op_t* rodata_op = NULL;
  return loom_global_rodata_build(
      &builder, LOOM_GLOBAL_RODATA_BUILD_FLAG_HAS_ALIGNMENT, symbol_ref,
      site_table, 8, LOOM_LOCATION_UNKNOWN, &rodata_op);
}

iree_status_t loom_amdgpu_finalize_sanitizer_module(
    loom_module_t* module, loom_low_lower_module_state_t* module_state,
    iree_arena_allocator_t* scratch_arena) {
  loom_amdgpu_sanitizer_module_state_t* sanitizer_state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_module_state_from_module_state(
      module_state, &sanitizer_state));
  if (sanitizer_state->site_row_count == 0) {
    return iree_ok_status();
  }

  loom_sanitizer_site_collection_t collection = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_flatten_site_rows(
      sanitizer_state, scratch_arena, &collection));
  iree_const_byte_span_t site_table = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(loom_sanitizer_site_table_encode(
      module, &collection, scratch_arena, &site_table));
  return loom_amdgpu_sanitizer_emit_site_table_rodata(module, site_table);
}

iree_status_t loom_amdgpu_sanitizer_site_id_for_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_sanitizer_site_id_t* out_site_id) {
  *out_site_id = LOOM_SANITIZER_SITE_ID_INVALID;
  loom_amdgpu_sanitizer_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_lower_state(context, &state));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_ensure_site_collection(context, state));

  const uintptr_t target_key = (uintptr_t)source_op;
  iree_host_size_t low = 0;
  iree_host_size_t high = state->site_map_row_count;
  while (low < high) {
    const iree_host_size_t mid = low + ((high - low) / 2);
    const uintptr_t mid_key = (uintptr_t)state->site_map_rows[mid].op;
    if (mid_key < target_key) {
      low = mid + 1;
    } else if (mid_key > target_key) {
      high = mid;
    } else {
      *out_site_id = state->site_map_rows[mid].site_id;
      return iree_ok_status();
    }
  }

  return iree_make_status(IREE_STATUS_INTERNAL,
                          "sanitizer assertion op missing final site id");
}

static bool loom_amdgpu_sanitizer_assert_access_kinds(
    loom_sanitizer_assert_access_kind_t kind,
    loom_low_source_memory_operation_kind_t* out_source_kind,
    loom_amdgpu_memory_operation_kind_t* out_memory_kind,
    loom_amdgpu_sanitizer_access_kind_t* out_report_kind) {
  switch (kind) {
    case LOOM_SANITIZER_ASSERT_ACCESS_KIND_READ:
      *out_source_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
      *out_memory_kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD;
      *out_report_kind = LOOM_AMDGPU_SANITIZER_ACCESS_KIND_READ;
      return true;
    case LOOM_SANITIZER_ASSERT_ACCESS_KIND_WRITE:
      *out_source_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE;
      *out_memory_kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE;
      *out_report_kind = LOOM_AMDGPU_SANITIZER_ACCESS_KIND_WRITE;
      return true;
    case LOOM_SANITIZER_ASSERT_ACCESS_KIND_READ_WRITE:
      *out_source_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_RMW;
      *out_memory_kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD;
      *out_report_kind = LOOM_AMDGPU_SANITIZER_ACCESS_KIND_ATOMIC;
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_sanitizer_access_payload_type(
    const loom_module_t* module, loom_value_id_t view_value_id,
    loom_type_t* out_vector_type) {
  *out_vector_type = loom_type_none();
  if (view_value_id >= module->values.count) {
    return false;
  }
  const loom_type_t view_type = loom_module_value_type(module, view_value_id);
  if (!loom_type_is_view(view_type)) {
    return false;
  }
  *out_vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, loom_type_element_type(view_type),
                          loom_dim_pack_static(1), /*encoding_id=*/0);
  return true;
}

iree_status_t loom_amdgpu_select_sanitizer_assert_access_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_sanitizer_access_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_sanitizer_access_plan_t){0};
  out_plan->site_id = LOOM_SANITIZER_SITE_ID_INVALID;
  *out_selected = false;
  if (!loom_sanitizer_assert_access_isa(source_op)) {
    return iree_ok_status();
  }

  loom_low_source_memory_operation_kind_t source_kind =
      LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
  loom_amdgpu_memory_operation_kind_t memory_kind =
      LOOM_AMDGPU_MEMORY_OPERATION_LOAD;
  loom_amdgpu_sanitizer_access_kind_t report_kind =
      LOOM_AMDGPU_SANITIZER_ACCESS_KIND_UNKNOWN;
  if (!loom_amdgpu_sanitizer_assert_access_kinds(
          loom_sanitizer_assert_access_kind(source_op), &source_kind,
          &memory_kind, &report_kind)) {
    return iree_ok_status();
  }

  loom_type_t vector_type = loom_type_none();
  const loom_value_id_t view_value_id =
      loom_sanitizer_assert_access_view(source_op);
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_amdgpu_sanitizer_access_payload_type(module, view_value_id,
                                                 &vector_type)) {
    return iree_ok_status();
  }

  loom_low_source_memory_access_plan_t source = {0};
  loom_low_source_memory_access_diagnostic_t source_diagnostic = {0};
  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_low_source_memory_access_plan_build_indexed(
          module, loom_low_lower_context_fact_table(context), source_kind,
          view_value_id, loom_sanitizer_assert_access_indices(source_op),
          loom_sanitizer_assert_access_static_indices(source_op), vector_type,
          cache_policy, &source, &source_diagnostic)) {
    return iree_ok_status();
  }
  const uint64_t access_size =
      (uint64_t)source.element_byte_count * source.vector_lane_count;
  if (access_size == 0 || access_size > 8) {
    return iree_ok_status();
  }

  loom_amdgpu_memory_access_diagnostic_t diagnostic = {0};
  if (!loom_amdgpu_memory_access_select_flat_global_address(
          module, loom_low_lower_context_descriptor_set(context), memory_kind,
          &source, vector_type, &out_plan->address, &diagnostic)) {
    return iree_ok_status();
  }
  out_plan->report_access_kind = report_kind;
  out_plan->access_size = (uint32_t)access_size;
  const loom_sanitizer_reporting_mode_t reporting_mode =
      loom_low_lower_context_sanitizer_reporting_mode(context);
  if (reporting_mode == LOOM_SANITIZER_REPORTING_MODE_DEFAULT ||
      reporting_mode == LOOM_SANITIZER_REPORTING_MODE_REPORT_ONLY) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_site_id_for_op(
        context, source_op, &out_plan->site_id));
  }
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_emit_vgpr_u64_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t value, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t sgpr_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_constant_u64(
      context, source_op, value, &sgpr_value));
  return loom_amdgpu_materialize_low_vgpr_b32_registers(context, source_op,
                                                        sgpr_value, out_value);
}

static iree_status_t loom_amdgpu_sanitizer_emit_sgpr_u32_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t value, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  return loom_amdgpu_emit_const_u32(context, source_op,
                                    LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, value,
                                    sgpr_type, out_value);
}

static iree_status_t loom_amdgpu_sanitizer_emit_vgpr_u32_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t value, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  return loom_amdgpu_emit_const_u32(context, source_op,
                                    LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, value,
                                    vgpr_type, out_value);
}

static loom_amdgpu_sanitizer_access_report_island_t*
loom_amdgpu_sanitizer_island_for_kind(
    loom_amdgpu_sanitizer_lower_state_t* state,
    loom_amdgpu_sanitizer_access_kind_t access_kind, bool** out_has_island) {
  switch (access_kind) {
    case LOOM_AMDGPU_SANITIZER_ACCESS_KIND_READ:
      *out_has_island = &state->has_read_island;
      return &state->read_island;
    case LOOM_AMDGPU_SANITIZER_ACCESS_KIND_WRITE:
      *out_has_island = &state->has_write_island;
      return &state->write_island;
    case LOOM_AMDGPU_SANITIZER_ACCESS_KIND_ATOMIC:
      *out_has_island = &state->has_atomic_island;
      return &state->atomic_island;
    default:
      *out_has_island = NULL;
      return NULL;
  }
}

static iree_status_t loom_amdgpu_sanitizer_get_access_island(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_sanitizer_lower_state_t* state,
    loom_amdgpu_sanitizer_access_kind_t access_kind,
    loom_location_id_t location,
    const loom_amdgpu_sanitizer_access_report_island_t** out_island) {
  *out_island = NULL;
  bool* has_island = NULL;
  loom_amdgpu_sanitizer_access_report_island_t* island =
      loom_amdgpu_sanitizer_island_for_kind(state, access_kind, &has_island);
  if (island == NULL) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "unsupported AMDGPU sanitizer access kind");
  }
  if (!*has_island) {
    loom_builder_ip_t saved_ip = loom_builder_save(builder);
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_access_report_island(
        builder, descriptor_set, saved_ip.block, state->feedback_config_symbol,
        access_kind, LOOM_AMDGPU_SANITIZER_REPORT_FLAG_NONE, location, island));
    loom_builder_restore(builder, saved_ip);
    *has_island = true;
  }
  *out_island = island;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_sanitizer_assert_access(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_access_plan_t* plan) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);

  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_sanitizer_assert_access_view(source_op), &low_resource));
  loom_value_id_t fault_address = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_flat_vaddr(
      context, source_op, &plan->address, low_resource, &fault_address));

  loom_amdgpu_sanitizer_lower_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_lower_state(context, &state));
  const loom_sanitizer_reporting_mode_t reporting_mode =
      loom_low_lower_context_sanitizer_reporting_mode(context);
  switch (reporting_mode) {
    case LOOM_SANITIZER_REPORTING_MODE_DEFAULT: {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_sanitizer_ensure_config_symbols(context, state));
      break;
    }
    case LOOM_SANITIZER_REPORTING_MODE_REPORT_ONLY: {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_sanitizer_ensure_config_symbols(context, state));
      break;
    }
    case LOOM_SANITIZER_REPORTING_MODE_TRAP: {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_sanitizer_ensure_asan_config_symbol(context, state));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported AMDGPU sanitizer reporting mode");
  }
  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_wavefront_size(
      loom_low_lower_context_bundle(context), &wavefront_size));
  loom_amdgpu_sanitizer_access_check_t check = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_sanitizer_access_check(
      builder, descriptor_set, state->asan_config_symbol, fault_address,
      plan->access_size, wavefront_size, source_op->location, &check));

  if (reporting_mode == LOOM_SANITIZER_REPORTING_MODE_TRAP) {
    loom_amdgpu_sanitizer_access_report_failure_branch_t branch = {0};
    return loom_amdgpu_build_sanitizer_trap_failure_mask_branch(
        builder, descriptor_set, check.failure_mask, source_op->location,
        &branch);
  }

  loom_amdgpu_sanitizer_report_source_t source = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_constant_u64(
      context, source_op, 0, &source.dispatch_ptr));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_emit_sgpr_u32_constant(
      context, source_op, 0, &source.workgroup_id_x));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_emit_vgpr_u32_constant(
      context, source_op, 0, &source.workitem_id_x));

  loom_amdgpu_sanitizer_access_report_t report = {
      .access_kind = plan->report_access_kind,
      .flags = LOOM_AMDGPU_SANITIZER_REPORT_FLAG_NONE,
      .fault_address = fault_address,
      .shadow_address = check.shadow_address,
      .shadow_value = check.shadow_value,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_emit_vgpr_u64_constant(
      context, source_op, plan->access_size, &report.access_size));
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_emit_vgpr_u64_constant(
      context, source_op, plan->site_id, &report.site_id));

  loom_amdgpu_sanitizer_access_report_failure_branch_t branch = {0};
  if (reporting_mode == LOOM_SANITIZER_REPORTING_MODE_REPORT_ONLY) {
    return loom_amdgpu_build_sanitizer_access_report_only_failure_mask_branch(
        builder, descriptor_set, state->feedback_config_symbol,
        check.failure_mask, &source, &report, source_op->location, &branch);
  }

  const loom_amdgpu_sanitizer_access_report_island_t* island = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_get_access_island(
      builder, descriptor_set, state, plan->report_access_kind,
      source_op->location, &island));
  return loom_amdgpu_build_sanitizer_access_report_failure_mask_branch(
      builder, descriptor_set, island, check.failure_mask, &source, &report,
      source_op->location, &branch);
}
