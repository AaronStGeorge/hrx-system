// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/arithmetic_lower_rules.h"
#include "loom/target/arch/amdgpu/async_lower_rules.h"
#include "loom/target/arch/amdgpu/compare_lower_rules.h"
#include "loom/target/arch/amdgpu/dot_lower_rules.h"
#include "loom/target/arch/amdgpu/integer_lower_rules.h"
#include "loom/target/arch/amdgpu/lower/internal.h"
#include "loom/target/arch/amdgpu/reduce_lower_rules.h"

typedef struct loom_amdgpu_lower_dispatch_row_t
    loom_amdgpu_lower_dispatch_row_t;

typedef iree_status_t (*loom_amdgpu_lower_select_fn_t)(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_lower_dispatch_row_t* row,
    loom_low_lower_plan_t* out_plan);

typedef iree_status_t (*loom_amdgpu_lower_emit_fn_t)(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_lower_dispatch_row_t* row, loom_low_lower_plan_t plan);

typedef iree_status_t (*loom_amdgpu_lower_verify_fn_t)(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

typedef struct loom_amdgpu_lower_dispatch_row_t {
  // Source op kind covered by this AMDGPU lowering row.
  loom_op_kind_t source_op_kind;
  // Bytes of plan data allocated by typed data selectors. Direct selector
  // hooks keep this zero and own any mixed plan allocation themselves.
  iree_host_size_t plan_data_size;
  // Optional source-to-low plan selection hook.
  loom_amdgpu_lower_select_fn_t select;
  // Optional source-to-low plan emission hook.
  loom_amdgpu_lower_emit_fn_t emit;
  // Optional target-low legality verifier hook.
  loom_amdgpu_lower_verify_fn_t verify;
} loom_amdgpu_lower_dispatch_row_t;

#define LOOM_AMDGPU_DEFINE_DATA_SELECT(name, plan_type, select_fn)             \
  static iree_status_t name(loom_low_lower_context_t* context,                 \
                            const loom_op_t* source_op,                        \
                            const loom_amdgpu_lower_dispatch_row_t* row,       \
                            loom_low_lower_plan_t* out_plan) {                 \
    plan_type* plan_data = NULL;                                               \
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(                    \
        context, row->plan_data_size, (void**)&plan_data));                    \
    bool selected = false;                                                     \
    IREE_RETURN_IF_ERROR(select_fn(context, source_op, plan_data, &selected)); \
    if (selected) {                                                            \
      *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);        \
    }                                                                          \
    return iree_ok_status();                                                   \
  }

#define LOOM_AMDGPU_DEFINE_DATA_EMIT(name, plan_type, emit_fn)              \
  static iree_status_t name(loom_low_lower_context_t* context,              \
                            const loom_op_t* source_op,                     \
                            const loom_amdgpu_lower_dispatch_row_t* row,    \
                            loom_low_lower_plan_t plan) {                   \
    (void)row;                                                              \
    return emit_fn(context, source_op, (const plan_type*)plan.target_data); \
  }

static iree_status_t loom_amdgpu_select_value_dispatch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_lower_dispatch_row_t* row,
    loom_low_lower_plan_t* out_plan) {
  (void)row;
  return loom_amdgpu_select_value_plan(context, source_op, out_plan);
}

static iree_status_t loom_amdgpu_emit_value_dispatch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_lower_dispatch_row_t* row, loom_low_lower_plan_t plan) {
  (void)row;
  return loom_amdgpu_lower_value_op(context, source_op, plan);
}

static iree_status_t loom_amdgpu_select_buffer_dispatch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_lower_dispatch_row_t* row,
    loom_low_lower_plan_t* out_plan) {
  (void)row;
  return loom_amdgpu_select_buffer_plan(context, source_op, out_plan);
}

static iree_status_t loom_amdgpu_emit_buffer_dispatch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_lower_dispatch_row_t* row, loom_low_lower_plan_t plan) {
  (void)row;
  return loom_amdgpu_lower_buffer_op(context, source_op, plan);
}

static iree_status_t loom_amdgpu_select_view_dispatch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_lower_dispatch_row_t* row,
    loom_low_lower_plan_t* out_plan) {
  (void)row;
  return loom_amdgpu_select_view_plan(context, source_op, out_plan);
}

static iree_status_t loom_amdgpu_emit_view_dispatch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_lower_dispatch_row_t* row, loom_low_lower_plan_t plan) {
  (void)row;
  (void)plan;
  return loom_amdgpu_lower_view_op(context, source_op);
}

static iree_status_t loom_amdgpu_select_preamble_dispatch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_lower_dispatch_row_t* row,
    loom_low_lower_plan_t* out_plan) {
  (void)row;
  return loom_amdgpu_select_preamble_plan(context, source_op, out_plan);
}

static iree_status_t loom_amdgpu_emit_preamble_dispatch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_lower_dispatch_row_t* row, loom_low_lower_plan_t plan) {
  (void)row;
  (void)plan;
  return loom_amdgpu_lower_preamble_op(context, source_op);
}

static iree_status_t loom_amdgpu_select_kernel_barrier_dispatch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_lower_dispatch_row_t* row,
    loom_low_lower_plan_t* out_plan) {
  (void)row;
  return loom_amdgpu_select_kernel_barrier_plan(context, source_op, out_plan);
}

static iree_status_t loom_amdgpu_emit_kernel_barrier_dispatch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_lower_dispatch_row_t* row, loom_low_lower_plan_t plan) {
  (void)row;
  (void)plan;
  return loom_amdgpu_lower_kernel_barrier(context, source_op);
}

LOOM_AMDGPU_DEFINE_DATA_SELECT(
    loom_amdgpu_select_kernel_subgroup_broadcast_dispatch,
    loom_amdgpu_subgroup_broadcast_plan_t,
    loom_amdgpu_select_kernel_subgroup_broadcast_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(
    loom_amdgpu_emit_kernel_subgroup_broadcast_dispatch,
    loom_amdgpu_subgroup_broadcast_plan_t,
    loom_amdgpu_lower_kernel_subgroup_broadcast)

LOOM_AMDGPU_DEFINE_DATA_SELECT(
    loom_amdgpu_select_kernel_subgroup_broadcast_first_dispatch,
    loom_amdgpu_subgroup_broadcast_first_plan_t,
    loom_amdgpu_select_kernel_subgroup_broadcast_first_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(
    loom_amdgpu_emit_kernel_subgroup_broadcast_first_dispatch,
    loom_amdgpu_subgroup_broadcast_first_plan_t,
    loom_amdgpu_lower_kernel_subgroup_broadcast_first)

LOOM_AMDGPU_DEFINE_DATA_SELECT(
    loom_amdgpu_select_kernel_subgroup_shuffle_dispatch,
    loom_amdgpu_subgroup_shuffle_plan_t,
    loom_amdgpu_select_kernel_subgroup_shuffle_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_kernel_subgroup_shuffle_dispatch,
                             loom_amdgpu_subgroup_shuffle_plan_t,
                             loom_amdgpu_lower_kernel_subgroup_shuffle)

LOOM_AMDGPU_DEFINE_DATA_SELECT(
    loom_amdgpu_select_kernel_subgroup_reduce_dispatch,
    loom_amdgpu_subgroup_reduce_plan_t,
    loom_amdgpu_select_kernel_subgroup_reduce_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_kernel_subgroup_reduce_dispatch,
                             loom_amdgpu_subgroup_reduce_plan_t,
                             loom_amdgpu_lower_kernel_subgroup_reduce)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_kernel_subgroup_scan_dispatch,
                               loom_amdgpu_subgroup_scan_plan_t,
                               loom_amdgpu_select_kernel_subgroup_scan_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_kernel_subgroup_scan_dispatch,
                             loom_amdgpu_subgroup_scan_plan_t,
                             loom_amdgpu_lower_kernel_subgroup_scan)

LOOM_AMDGPU_DEFINE_DATA_SELECT(
    loom_amdgpu_select_kernel_subgroup_active_mask_dispatch,
    loom_amdgpu_subgroup_active_mask_plan_t,
    loom_amdgpu_select_kernel_subgroup_active_mask_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(
    loom_amdgpu_emit_kernel_subgroup_active_mask_dispatch,
    loom_amdgpu_subgroup_active_mask_plan_t,
    loom_amdgpu_lower_kernel_subgroup_active_mask)

LOOM_AMDGPU_DEFINE_DATA_SELECT(
    loom_amdgpu_select_kernel_subgroup_ballot_dispatch,
    loom_amdgpu_subgroup_ballot_plan_t,
    loom_amdgpu_select_kernel_subgroup_ballot_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_kernel_subgroup_ballot_dispatch,
                             loom_amdgpu_subgroup_ballot_plan_t,
                             loom_amdgpu_lower_kernel_subgroup_ballot)

LOOM_AMDGPU_DEFINE_DATA_SELECT(
    loom_amdgpu_select_kernel_subgroup_vote_any_dispatch,
    loom_amdgpu_subgroup_vote_any_plan_t,
    loom_amdgpu_select_kernel_subgroup_vote_any_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_kernel_subgroup_vote_any_dispatch,
                             loom_amdgpu_subgroup_vote_any_plan_t,
                             loom_amdgpu_lower_kernel_subgroup_vote_any)

LOOM_AMDGPU_DEFINE_DATA_SELECT(
    loom_amdgpu_select_kernel_subgroup_vote_all_dispatch,
    loom_amdgpu_subgroup_vote_all_plan_t,
    loom_amdgpu_select_kernel_subgroup_vote_all_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_kernel_subgroup_vote_all_dispatch,
                             loom_amdgpu_subgroup_vote_all_plan_t,
                             loom_amdgpu_lower_kernel_subgroup_vote_all)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_kernel_async_gather_dispatch,
                               loom_amdgpu_async_gather_plan_t,
                               loom_amdgpu_select_kernel_async_gather_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_kernel_async_gather_dispatch,
                             loom_amdgpu_async_gather_plan_t,
                             loom_amdgpu_lower_kernel_async_gather)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_kernel_async_wait_dispatch,
                               loom_amdgpu_async_wait_plan_t,
                               loom_amdgpu_select_kernel_async_wait_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_kernel_async_wait_dispatch,
                             loom_amdgpu_async_wait_plan_t,
                             loom_amdgpu_lower_kernel_async_wait)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_cmpi_dispatch,
                               loom_amdgpu_vector_compare_plan_t,
                               loom_amdgpu_select_vector_cmpi_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_cmpi_dispatch,
                             loom_amdgpu_vector_compare_plan_t,
                             loom_amdgpu_lower_vector_cmpi)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_cmpf_dispatch,
                               loom_amdgpu_vector_compare_plan_t,
                               loom_amdgpu_select_vector_cmpf_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_cmpf_dispatch,
                             loom_amdgpu_vector_compare_plan_t,
                             loom_amdgpu_lower_vector_cmpf)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_select_dispatch,
                               loom_amdgpu_vector_select_plan_t,
                               loom_amdgpu_select_vector_select_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_select_dispatch,
                             loom_amdgpu_vector_select_plan_t,
                             loom_amdgpu_lower_vector_select)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_mma_dispatch,
                               loom_amdgpu_matrix_mma_plan_t,
                               loom_amdgpu_select_vector_mma_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_mma_dispatch,
                             loom_amdgpu_matrix_mma_plan_t,
                             loom_amdgpu_lower_vector_mma)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_table_lookup_dispatch,
                               loom_amdgpu_table_lookup_plan_t,
                               loom_amdgpu_select_vector_table_lookup_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_table_lookup_dispatch,
                             loom_amdgpu_table_lookup_plan_t,
                             loom_amdgpu_lower_vector_table_lookup)

LOOM_AMDGPU_DEFINE_DATA_SELECT(
    loom_amdgpu_select_vector_bitfield_extract_dispatch,
    loom_amdgpu_bitfield_extract_plan_t,
    loom_amdgpu_select_vector_bitfield_extract_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_bitfield_extract_dispatch,
                             loom_amdgpu_bitfield_extract_plan_t,
                             loom_amdgpu_lower_vector_bitfield_extract)

LOOM_AMDGPU_DEFINE_DATA_SELECT(
    loom_amdgpu_select_vector_bitfield_insert_dispatch,
    loom_amdgpu_bitfield_insert_plan_t,
    loom_amdgpu_select_vector_bitfield_insert_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_bitfield_insert_dispatch,
                             loom_amdgpu_bitfield_insert_plan_t,
                             loom_amdgpu_lower_vector_bitfield_insert)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_bitpack_dispatch,
                               loom_amdgpu_bitpack_plan_t,
                               loom_amdgpu_select_vector_bitpack_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_bitpack_dispatch,
                             loom_amdgpu_bitpack_plan_t,
                             loom_amdgpu_lower_vector_bitpack)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_bitunpack_dispatch,
                               loom_amdgpu_bitunpack_plan_t,
                               loom_amdgpu_select_vector_bitunpack_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_bitunpack_dispatch,
                             loom_amdgpu_bitunpack_plan_t,
                             loom_amdgpu_lower_vector_bitunpack)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_bitcast_dispatch,
                               loom_amdgpu_vector_bitcast_plan_t,
                               loom_amdgpu_select_vector_bitcast_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_bitcast_dispatch,
                             loom_amdgpu_vector_bitcast_plan_t,
                             loom_amdgpu_lower_vector_bitcast)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_slice_dispatch,
                               loom_amdgpu_vector_slice_plan_t,
                               loom_amdgpu_select_vector_slice_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_slice_dispatch,
                             loom_amdgpu_vector_slice_plan_t,
                             loom_amdgpu_lower_vector_slice)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_load_dispatch,
                               loom_amdgpu_memory_access_plan_t,
                               loom_amdgpu_select_vector_load_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_load_dispatch,
                             loom_amdgpu_memory_access_plan_t,
                             loom_amdgpu_lower_vector_load)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_store_dispatch,
                               loom_amdgpu_memory_access_plan_t,
                               loom_amdgpu_select_vector_store_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_store_dispatch,
                             loom_amdgpu_memory_access_plan_t,
                             loom_amdgpu_lower_vector_store)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_view_atomic_dispatch,
                               loom_amdgpu_atomic_plan_t,
                               loom_amdgpu_select_view_atomic_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_view_atomic_dispatch,
                             loom_amdgpu_atomic_plan_t,
                             loom_amdgpu_lower_view_atomic)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_view_prefetch_dispatch,
                               loom_amdgpu_prefetch_plan_t,
                               loom_amdgpu_select_view_prefetch_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_view_prefetch_dispatch,
                             loom_amdgpu_prefetch_plan_t,
                             loom_amdgpu_lower_view_prefetch)

#undef LOOM_AMDGPU_DEFINE_DATA_SELECT
#undef LOOM_AMDGPU_DEFINE_DATA_EMIT

#define LOOM_AMDGPU_OP_INDEX(op_kind) ((uint8_t)((op_kind) & 0xFFu))

#define LOOM_AMDGPU_DIRECT_ROW(op_kind, select_fn, emit_fn, verify_fn) \
  {                                                                    \
      .source_op_kind = (op_kind),                                     \
      .plan_data_size = 0,                                             \
      .select = (select_fn),                                           \
      .emit = (emit_fn),                                               \
      .verify = (verify_fn),                                           \
  }

#define LOOM_AMDGPU_DATA_ROW(op_kind, plan_type, select_fn, emit_fn, \
                             verify_fn)                              \
  {                                                                  \
      .source_op_kind = (op_kind),                                   \
      .plan_data_size = sizeof(plan_type),                           \
      .select = (select_fn),                                         \
      .emit = (emit_fn),                                             \
      .verify = (verify_fn),                                         \
  }

#define LOOM_AMDGPU_LEGALITY_ROW(op_kind, verify_fn) \
  {                                                  \
      .source_op_kind = (op_kind),                   \
      .plan_data_size = 0,                           \
      .select = NULL,                                \
      .emit = NULL,                                  \
      .verify = (verify_fn),                         \
  }

static const loom_amdgpu_lower_dispatch_row_t
    kAmdgpuIndexDispatchRows[LOOM_OP_INDEX_COUNT_] = {
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_INDEX_CONSTANT)] = LOOM_AMDGPU_DIRECT_ROW(
            LOOM_OP_INDEX_CONSTANT, loom_amdgpu_select_value_dispatch,
            loom_amdgpu_emit_value_dispatch, NULL),
};

static const loom_amdgpu_lower_dispatch_row_t
    kAmdgpuScalarDispatchRows[LOOM_OP_SCALAR_COUNT_] = {
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_CONSTANT)] =
            LOOM_AMDGPU_DIRECT_ROW(LOOM_OP_SCALAR_CONSTANT,
                                   loom_amdgpu_select_value_dispatch,
                                   loom_amdgpu_emit_value_dispatch, NULL),
};

static const loom_amdgpu_lower_dispatch_row_t
    kAmdgpuBufferDispatchRows[LOOM_OP_BUFFER_COUNT_] = {
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_BUFFER_ALLOCA)] = LOOM_AMDGPU_DIRECT_ROW(
            LOOM_OP_BUFFER_ALLOCA, loom_amdgpu_select_buffer_dispatch,
            loom_amdgpu_emit_buffer_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_BUFFER_VIEW)] = LOOM_AMDGPU_DIRECT_ROW(
            LOOM_OP_BUFFER_VIEW, loom_amdgpu_select_buffer_dispatch,
            loom_amdgpu_emit_buffer_dispatch,
            loom_amdgpu_low_legality_verify_buffer),
};

static const loom_amdgpu_lower_dispatch_row_t
    kAmdgpuViewDispatchRows[LOOM_OP_VIEW_COUNT_] = {
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VIEW_SUBVIEW)] = LOOM_AMDGPU_DIRECT_ROW(
            LOOM_OP_VIEW_SUBVIEW, loom_amdgpu_select_view_dispatch,
            loom_amdgpu_emit_view_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VIEW_ATOMIC_REDUCE)] =
            LOOM_AMDGPU_DATA_ROW(LOOM_OP_VIEW_ATOMIC_REDUCE,
                                 loom_amdgpu_atomic_plan_t,
                                 loom_amdgpu_select_view_atomic_dispatch,
                                 loom_amdgpu_emit_view_atomic_dispatch,
                                 loom_amdgpu_low_legality_verify_view_atomic),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VIEW_ATOMIC_RMW)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VIEW_ATOMIC_RMW, loom_amdgpu_atomic_plan_t,
            loom_amdgpu_select_view_atomic_dispatch,
            loom_amdgpu_emit_view_atomic_dispatch,
            loom_amdgpu_low_legality_verify_view_atomic),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VIEW_ATOMIC_CMPXCHG)] =
            LOOM_AMDGPU_DATA_ROW(LOOM_OP_VIEW_ATOMIC_CMPXCHG,
                                 loom_amdgpu_atomic_plan_t,
                                 loom_amdgpu_select_view_atomic_dispatch,
                                 loom_amdgpu_emit_view_atomic_dispatch,
                                 loom_amdgpu_low_legality_verify_view_atomic),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VIEW_PREFETCH)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VIEW_PREFETCH, loom_amdgpu_prefetch_plan_t,
            loom_amdgpu_select_view_prefetch_dispatch,
            loom_amdgpu_emit_view_prefetch_dispatch, NULL),
};

static const loom_amdgpu_lower_dispatch_row_t
    kAmdgpuVectorDispatchRows[LOOM_OP_VECTOR_COUNT_] = {
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_CONSTANT)] =
            LOOM_AMDGPU_DIRECT_ROW(LOOM_OP_VECTOR_CONSTANT,
                                   loom_amdgpu_select_value_dispatch,
                                   loom_amdgpu_emit_value_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_IOTA)] = LOOM_AMDGPU_DIRECT_ROW(
            LOOM_OP_VECTOR_IOTA, loom_amdgpu_select_value_dispatch,
            loom_amdgpu_emit_value_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_CMPI)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VECTOR_CMPI, loom_amdgpu_vector_compare_plan_t,
            loom_amdgpu_select_vector_cmpi_dispatch,
            loom_amdgpu_emit_vector_cmpi_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_CMPF)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VECTOR_CMPF, loom_amdgpu_vector_compare_plan_t,
            loom_amdgpu_select_vector_cmpf_dispatch,
            loom_amdgpu_emit_vector_cmpf_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_SELECT)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VECTOR_SELECT, loom_amdgpu_vector_select_plan_t,
            loom_amdgpu_select_vector_select_dispatch,
            loom_amdgpu_emit_vector_select_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_MMA)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VECTOR_MMA, loom_amdgpu_matrix_mma_plan_t,
            loom_amdgpu_select_vector_mma_dispatch,
            loom_amdgpu_emit_vector_mma_dispatch,
            loom_amdgpu_low_legality_verify_vector_mma),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_TABLE_LOOKUP)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_VECTOR_TABLE_LOOKUP, loom_amdgpu_table_lookup_plan_t,
                loom_amdgpu_select_vector_table_lookup_dispatch,
                loom_amdgpu_emit_vector_table_lookup_dispatch,
                loom_amdgpu_low_legality_verify_vector_table),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_BITFIELD_EXTRACTU)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_VECTOR_BITFIELD_EXTRACTU,
                loom_amdgpu_bitfield_extract_plan_t,
                loom_amdgpu_select_vector_bitfield_extract_dispatch,
                loom_amdgpu_emit_vector_bitfield_extract_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_BITFIELD_EXTRACTS)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_VECTOR_BITFIELD_EXTRACTS,
                loom_amdgpu_bitfield_extract_plan_t,
                loom_amdgpu_select_vector_bitfield_extract_dispatch,
                loom_amdgpu_emit_vector_bitfield_extract_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_BITFIELD_INSERT)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_VECTOR_BITFIELD_INSERT,
                loom_amdgpu_bitfield_insert_plan_t,
                loom_amdgpu_select_vector_bitfield_insert_dispatch,
                loom_amdgpu_emit_vector_bitfield_insert_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_BITPACK)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VECTOR_BITPACK, loom_amdgpu_bitpack_plan_t,
            loom_amdgpu_select_vector_bitpack_dispatch,
            loom_amdgpu_emit_vector_bitpack_dispatch,
            loom_amdgpu_low_legality_verify_vector_bitstream),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_BITUNPACKS)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_VECTOR_BITUNPACKS, loom_amdgpu_bitunpack_plan_t,
                loom_amdgpu_select_vector_bitunpack_dispatch,
                loom_amdgpu_emit_vector_bitunpack_dispatch,
                loom_amdgpu_low_legality_verify_vector_bitstream),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_BITUNPACKU)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_VECTOR_BITUNPACKU, loom_amdgpu_bitunpack_plan_t,
                loom_amdgpu_select_vector_bitunpack_dispatch,
                loom_amdgpu_emit_vector_bitunpack_dispatch,
                loom_amdgpu_low_legality_verify_vector_bitstream),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_BITCAST)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VECTOR_BITCAST, loom_amdgpu_vector_bitcast_plan_t,
            loom_amdgpu_select_vector_bitcast_dispatch,
            loom_amdgpu_emit_vector_bitcast_dispatch,
            loom_amdgpu_low_legality_verify_vector_structural),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_SLICE)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VECTOR_SLICE, loom_amdgpu_vector_slice_plan_t,
            loom_amdgpu_select_vector_slice_dispatch,
            loom_amdgpu_emit_vector_slice_dispatch,
            loom_amdgpu_low_legality_verify_vector_structural),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_EXTRACT)] = LOOM_AMDGPU_DIRECT_ROW(
            LOOM_OP_VECTOR_EXTRACT, loom_amdgpu_select_value_dispatch,
            loom_amdgpu_emit_value_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_FROM_ELEMENTS)] =
            LOOM_AMDGPU_DIRECT_ROW(LOOM_OP_VECTOR_FROM_ELEMENTS,
                                   loom_amdgpu_select_value_dispatch,
                                   loom_amdgpu_emit_value_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_LOAD)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VECTOR_LOAD, loom_amdgpu_memory_access_plan_t,
            loom_amdgpu_select_vector_load_dispatch,
            loom_amdgpu_emit_vector_load_dispatch,
            loom_amdgpu_low_legality_verify_vector_memory),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_STORE)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VECTOR_STORE, loom_amdgpu_memory_access_plan_t,
            loom_amdgpu_select_vector_store_dispatch,
            loom_amdgpu_emit_vector_store_dispatch,
            loom_amdgpu_low_legality_verify_vector_memory),
};

static const loom_amdgpu_lower_dispatch_row_t
    kAmdgpuKernelDispatchRows[LOOM_OP_KERNEL_COUNT_] = {
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_BARRIER)] = LOOM_AMDGPU_DIRECT_ROW(
            LOOM_OP_KERNEL_BARRIER, loom_amdgpu_select_kernel_barrier_dispatch,
            loom_amdgpu_emit_kernel_barrier_dispatch,
            loom_amdgpu_low_legality_verify_kernel_barrier),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_WORKITEM_ID)] =
            LOOM_AMDGPU_DIRECT_ROW(
                LOOM_OP_KERNEL_WORKITEM_ID,
                loom_amdgpu_select_preamble_dispatch,
                loom_amdgpu_emit_preamble_dispatch,
                loom_amdgpu_low_legality_verify_kernel_preamble),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_WORKGROUP_ID)] =
            LOOM_AMDGPU_DIRECT_ROW(
                LOOM_OP_KERNEL_WORKGROUP_ID,
                loom_amdgpu_select_preamble_dispatch,
                loom_amdgpu_emit_preamble_dispatch,
                loom_amdgpu_low_legality_verify_kernel_preamble),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_WORKGROUP_SIZE)] =
            LOOM_AMDGPU_DIRECT_ROW(
                LOOM_OP_KERNEL_WORKGROUP_SIZE,
                loom_amdgpu_select_preamble_dispatch,
                loom_amdgpu_emit_preamble_dispatch,
                loom_amdgpu_low_legality_verify_kernel_preamble),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_WORKGROUP_COUNT)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_WORKGROUP_COUNT,
                loom_amdgpu_low_legality_verify_kernel_preamble),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_WORKITEM_DISPATCH_ID)] =
            LOOM_AMDGPU_DIRECT_ROW(
                LOOM_OP_KERNEL_WORKITEM_DISPATCH_ID,
                loom_amdgpu_select_preamble_dispatch,
                loom_amdgpu_emit_preamble_dispatch,
                loom_amdgpu_low_legality_verify_kernel_preamble),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_ID)] =
            LOOM_AMDGPU_DIRECT_ROW(
                LOOM_OP_KERNEL_SUBGROUP_ID,
                loom_amdgpu_select_preamble_dispatch,
                loom_amdgpu_emit_preamble_dispatch,
                loom_amdgpu_low_legality_verify_kernel_preamble),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_COUNT)] =
            LOOM_AMDGPU_DIRECT_ROW(
                LOOM_OP_KERNEL_SUBGROUP_COUNT,
                loom_amdgpu_select_preamble_dispatch,
                loom_amdgpu_emit_preamble_dispatch,
                loom_amdgpu_low_legality_verify_kernel_preamble),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_SIZE)] =
            LOOM_AMDGPU_DIRECT_ROW(
                LOOM_OP_KERNEL_SUBGROUP_SIZE,
                loom_amdgpu_select_preamble_dispatch,
                loom_amdgpu_emit_preamble_dispatch,
                loom_amdgpu_low_legality_verify_kernel_preamble),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_LANE_ID)] =
            LOOM_AMDGPU_DIRECT_ROW(
                LOOM_OP_KERNEL_SUBGROUP_LANE_ID,
                loom_amdgpu_select_preamble_dispatch,
                loom_amdgpu_emit_preamble_dispatch,
                loom_amdgpu_low_legality_verify_kernel_preamble),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_SHUFFLE)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_KERNEL_SUBGROUP_SHUFFLE,
                loom_amdgpu_subgroup_shuffle_plan_t,
                loom_amdgpu_select_kernel_subgroup_shuffle_dispatch,
                loom_amdgpu_emit_kernel_subgroup_shuffle_dispatch,
                loom_amdgpu_low_legality_verify_kernel_subgroup_shuffle),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_BROADCAST)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_KERNEL_SUBGROUP_BROADCAST,
                loom_amdgpu_subgroup_broadcast_plan_t,
                loom_amdgpu_select_kernel_subgroup_broadcast_dispatch,
                loom_amdgpu_emit_kernel_subgroup_broadcast_dispatch,
                loom_amdgpu_low_legality_verify_kernel_subgroup_broadcast),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_BROADCAST_FIRST)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_KERNEL_SUBGROUP_BROADCAST_FIRST,
                loom_amdgpu_subgroup_broadcast_first_plan_t,
                loom_amdgpu_select_kernel_subgroup_broadcast_first_dispatch,
                loom_amdgpu_emit_kernel_subgroup_broadcast_first_dispatch,
                loom_amdgpu_low_legality_verify_kernel_subgroup_broadcast_first),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_REDUCE)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_KERNEL_SUBGROUP_REDUCE,
                loom_amdgpu_subgroup_reduce_plan_t,
                loom_amdgpu_select_kernel_subgroup_reduce_dispatch,
                loom_amdgpu_emit_kernel_subgroup_reduce_dispatch,
                loom_amdgpu_low_legality_verify_kernel_subgroup_reduce),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_SCAN)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_KERNEL_SUBGROUP_SCAN, loom_amdgpu_subgroup_scan_plan_t,
                loom_amdgpu_select_kernel_subgroup_scan_dispatch,
                loom_amdgpu_emit_kernel_subgroup_scan_dispatch,
                loom_amdgpu_low_legality_verify_kernel_subgroup_scan),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_VOTE_ANY)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_KERNEL_SUBGROUP_VOTE_ANY,
                loom_amdgpu_subgroup_vote_any_plan_t,
                loom_amdgpu_select_kernel_subgroup_vote_any_dispatch,
                loom_amdgpu_emit_kernel_subgroup_vote_any_dispatch,
                loom_amdgpu_low_legality_verify_kernel_subgroup_vote_any),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_VOTE_ALL)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_KERNEL_SUBGROUP_VOTE_ALL,
                loom_amdgpu_subgroup_vote_all_plan_t,
                loom_amdgpu_select_kernel_subgroup_vote_all_dispatch,
                loom_amdgpu_emit_kernel_subgroup_vote_all_dispatch,
                loom_amdgpu_low_legality_verify_kernel_subgroup_vote_all),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_VOTE_BALLOT)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_KERNEL_SUBGROUP_VOTE_BALLOT,
                loom_amdgpu_subgroup_ballot_plan_t,
                loom_amdgpu_select_kernel_subgroup_ballot_dispatch,
                loom_amdgpu_emit_kernel_subgroup_ballot_dispatch,
                loom_amdgpu_low_legality_verify_kernel_subgroup_ballot),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_ACTIVE_MASK)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_KERNEL_SUBGROUP_ACTIVE_MASK,
                loom_amdgpu_subgroup_active_mask_plan_t,
                loom_amdgpu_select_kernel_subgroup_active_mask_dispatch,
                loom_amdgpu_emit_kernel_subgroup_active_mask_dispatch,
                loom_amdgpu_low_legality_verify_kernel_subgroup_active_mask),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_MATCH_ANY)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_SUBGROUP_MATCH_ANY,
                loom_amdgpu_low_legality_verify_kernel_subgroup_match),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_MATCH_ALL)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_SUBGROUP_MATCH_ALL,
                loom_amdgpu_low_legality_verify_kernel_subgroup_match),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_WORKGROUP_REDUCE)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_WORKGROUP_REDUCE,
                loom_amdgpu_low_legality_verify_kernel_collective),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_WORKGROUP_VOTE_ANY)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_WORKGROUP_VOTE_ANY,
                loom_amdgpu_low_legality_verify_kernel_collective),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_WORKGROUP_VOTE_ALL)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_WORKGROUP_VOTE_ALL,
                loom_amdgpu_low_legality_verify_kernel_collective),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_WORKGROUP_VOTE_COUNT)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_WORKGROUP_VOTE_COUNT,
                loom_amdgpu_low_legality_verify_kernel_collective),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER,
                loom_amdgpu_low_legality_verify_kernel_async),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER_MASK)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER_MASK,
                loom_amdgpu_low_legality_verify_kernel_async),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_ASYNC_COPY)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_ASYNC_COPY,
                loom_amdgpu_low_legality_verify_kernel_async),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_ASYNC_COPY_MASK)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_ASYNC_COPY_MASK,
                loom_amdgpu_low_legality_verify_kernel_async),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_ASYNC_GATHER)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_KERNEL_ASYNC_GATHER, loom_amdgpu_async_gather_plan_t,
                loom_amdgpu_select_kernel_async_gather_dispatch,
                loom_amdgpu_emit_kernel_async_gather_dispatch,
                loom_amdgpu_low_legality_verify_kernel_async),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_ASYNC_GATHER_MASK)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_ASYNC_GATHER_MASK,
                loom_amdgpu_low_legality_verify_kernel_async),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_ASYNC_GROUP)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_ASYNC_GROUP,
                loom_amdgpu_low_legality_verify_kernel_async),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_ASYNC_TENSOR_LOAD_TO_LDS)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_ASYNC_TENSOR_LOAD_TO_LDS,
                loom_amdgpu_low_legality_verify_kernel_async),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_ASYNC_TENSOR_STORE_FROM_LDS)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_ASYNC_TENSOR_STORE_FROM_LDS,
                loom_amdgpu_low_legality_verify_kernel_async),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_ASYNC_WAIT)] =
            LOOM_AMDGPU_DATA_ROW(LOOM_OP_KERNEL_ASYNC_WAIT,
                                 loom_amdgpu_async_wait_plan_t,
                                 loom_amdgpu_select_kernel_async_wait_dispatch,
                                 loom_amdgpu_emit_kernel_async_wait_dispatch,
                                 loom_amdgpu_low_legality_verify_kernel_async),
};

#undef LOOM_AMDGPU_DIRECT_ROW
#undef LOOM_AMDGPU_DATA_ROW
#undef LOOM_AMDGPU_LEGALITY_ROW
#undef LOOM_AMDGPU_OP_INDEX

static const loom_amdgpu_lower_dispatch_row_t*
loom_amdgpu_lower_dispatch_row_from_table(
    loom_op_kind_t op_kind, const loom_amdgpu_lower_dispatch_row_t* rows,
    iree_host_size_t row_count) {
  const uint8_t op_index = loom_op_dialect_index(op_kind);
  if (op_index >= row_count) {
    return NULL;
  }
  const loom_amdgpu_lower_dispatch_row_t* row = &rows[op_index];
  return row->source_op_kind == op_kind ? row : NULL;
}

static const loom_amdgpu_lower_dispatch_row_t*
loom_amdgpu_find_lower_dispatch_row(loom_low_lower_plan_id_t plan_id) {
  if (plan_id > UINT16_MAX) {
    return NULL;
  }
  const loom_op_kind_t op_kind = (loom_op_kind_t)plan_id;
  switch (loom_op_dialect_id(op_kind)) {
    case LOOM_DIALECT_INDEX:
      return loom_amdgpu_lower_dispatch_row_from_table(
          op_kind, kAmdgpuIndexDispatchRows,
          IREE_ARRAYSIZE(kAmdgpuIndexDispatchRows));
    case LOOM_DIALECT_SCALAR:
      return loom_amdgpu_lower_dispatch_row_from_table(
          op_kind, kAmdgpuScalarDispatchRows,
          IREE_ARRAYSIZE(kAmdgpuScalarDispatchRows));
    case LOOM_DIALECT_BUFFER:
      return loom_amdgpu_lower_dispatch_row_from_table(
          op_kind, kAmdgpuBufferDispatchRows,
          IREE_ARRAYSIZE(kAmdgpuBufferDispatchRows));
    case LOOM_DIALECT_VIEW:
      return loom_amdgpu_lower_dispatch_row_from_table(
          op_kind, kAmdgpuViewDispatchRows,
          IREE_ARRAYSIZE(kAmdgpuViewDispatchRows));
    case LOOM_DIALECT_VECTOR:
      return loom_amdgpu_lower_dispatch_row_from_table(
          op_kind, kAmdgpuVectorDispatchRows,
          IREE_ARRAYSIZE(kAmdgpuVectorDispatchRows));
    case LOOM_DIALECT_KERNEL:
      return loom_amdgpu_lower_dispatch_row_from_table(
          op_kind, kAmdgpuKernelDispatchRows,
          IREE_ARRAYSIZE(kAmdgpuKernelDispatchRows));
    default:
      return NULL;
  }
}

static iree_status_t loom_amdgpu_select_plan_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  const loom_amdgpu_lower_dispatch_row_t* row =
      loom_amdgpu_find_lower_dispatch_row(source_op->kind);
  if (row == NULL || row->select == NULL) {
    return iree_ok_status();
  }
  return row->select(context, source_op, row, out_plan);
}

static iree_status_t loom_amdgpu_select_op(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_low_lower_plan_t* out_plan) {
  (void)user_data;
  return loom_amdgpu_select_plan_id(context, source_op, out_plan);
}

static iree_status_t loom_amdgpu_emit_op(void* user_data,
                                         loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_low_lower_plan_t plan) {
  (void)user_data;
  const loom_amdgpu_lower_dispatch_row_t* row =
      loom_amdgpu_find_lower_dispatch_row(plan.id);
  return row->emit(context, source_op, row, plan);
}

static iree_status_t loom_amdgpu_low_legality_try_verify_op(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  *out_handled = false;
  const loom_amdgpu_lower_dispatch_row_t* row =
      loom_amdgpu_find_lower_dispatch_row(op->kind);
  if (row == NULL || row->verify == NULL) {
    return iree_ok_status();
  }
  return row->verify(provider, context, op, out_handled);
}

// clang-format off
static const loom_low_lower_rule_set_t* const kAmdgpuRuleSets[] = {
  &loom_amdgpu_arithmetic_lower_rule_set,
  &loom_amdgpu_integer_lower_rule_set,
  &loom_amdgpu_compare_lower_rule_set,
  &loom_amdgpu_dot_lower_rule_set,
  &loom_amdgpu_reduce_lower_rule_set,
  &loom_amdgpu_async_lower_rule_set,
};
// clang-format on

static const loom_low_lower_policy_t kAmdgpuLowLowerPolicy = {
    .name = IREE_SVL("amdgpu-register-lower"),
    .map_type = {.fn = loom_amdgpu_map_type, .user_data = NULL},
    .map_value = {.fn = loom_amdgpu_map_value, .user_data = NULL},
    .map_contract_value = {.fn = loom_amdgpu_map_contract_value,
                           .user_data = NULL},
    .map_argument = {.fn = loom_amdgpu_map_argument, .user_data = NULL},
    .emit_preamble = {.fn = loom_amdgpu_emit_preamble, .user_data = NULL},
    .emit_cond_branch = {.fn = loom_amdgpu_emit_cond_branch, .user_data = NULL},
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kAmdgpuRuleSets),
            .values = kAmdgpuRuleSets,
        },
    .select_op = {.fn = loom_amdgpu_select_op, .user_data = NULL},
    .emit_op = {.fn = loom_amdgpu_emit_op, .user_data = NULL},
};

const loom_target_low_legality_provider_t
    loom_amdgpu_low_legality_provider_storage = {
        .name = IREE_SVL("amdgpu"),
        .builtin_dialect_bits =
            (1u << LOOM_DIALECT_BUFFER) | (1u << LOOM_DIALECT_VIEW) |
            (1u << LOOM_DIALECT_VECTOR) | (1u << LOOM_DIALECT_KERNEL),
        .try_verify_op = loom_amdgpu_low_legality_try_verify_op,
};

const loom_low_lower_policy_t* loom_amdgpu_low_lower_policy(void) {
  return &kAmdgpuLowLowerPolicy;
}

const loom_target_low_legality_provider_t* loom_amdgpu_low_legality_provider(
    void) {
  return &loom_amdgpu_low_legality_provider_storage;
}

void loom_amdgpu_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  // clang-format off
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
    {.contract_set_key = IREE_SVL("amdgpu.gfx950.core"), .policy = &kAmdgpuLowLowerPolicy},
    {.contract_set_key = IREE_SVL("amdgpu.gfx11.core"), .policy = &kAmdgpuLowLowerPolicy},
    {.contract_set_key = IREE_SVL("amdgpu.gfx12.core"), .policy = &kAmdgpuLowLowerPolicy},
    {.contract_set_key = IREE_SVL("amdgpu.gfx1250.core"), .policy = &kAmdgpuLowLowerPolicy},
  };
  // clang-format on
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
