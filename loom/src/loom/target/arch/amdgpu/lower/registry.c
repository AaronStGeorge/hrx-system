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
#include "loom/target/arch/amdgpu/lower/internal.h"

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

#define LOOM_AMDGPU_DEFINE_DATA_SELECT(name, plan_type, select_fn)       \
  static iree_status_t name(loom_low_lower_context_t* context,           \
                            const loom_op_t* source_op,                  \
                            const loom_amdgpu_lower_dispatch_row_t* row, \
                            loom_low_lower_plan_t* out_plan) {           \
    IREE_ASSERT_EQ(row->plan_data_size, sizeof(plan_type));              \
    plan_type* plan_data = NULL;                                         \
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(              \
        context, row->plan_data_size, (void**)&plan_data));              \
    if (select_fn(context, source_op, plan_data)) {                      \
      *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);  \
    }                                                                    \
    return iree_ok_status();                                             \
  }

#define LOOM_AMDGPU_DEFINE_DATA_EMIT(name, plan_type, emit_fn)              \
  static iree_status_t name(loom_low_lower_context_t* context,              \
                            const loom_op_t* source_op,                     \
                            const loom_amdgpu_lower_dispatch_row_t* row,    \
                            loom_low_lower_plan_t plan) {                   \
    IREE_ASSERT_EQ(row->plan_data_size, sizeof(plan_type));                 \
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

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_kernel_async_gather_dispatch,
                               loom_amdgpu_async_gather_plan_t,
                               loom_amdgpu_select_kernel_async_gather_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_kernel_async_gather_dispatch,
                             loom_amdgpu_async_gather_plan_t,
                             loom_amdgpu_lower_kernel_async_gather)

static iree_status_t loom_amdgpu_select_kernel_async_wait_dispatch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_lower_dispatch_row_t* row,
    loom_low_lower_plan_t* out_plan) {
  IREE_ASSERT_EQ(row->plan_data_size, sizeof(loom_amdgpu_async_wait_plan_t));
  loom_amdgpu_async_wait_plan_t* plan_data = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, row->plan_data_size, (void**)&plan_data));
  bool selected = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_kernel_async_wait_plan(
      context, source_op, plan_data, &selected));
  if (selected) {
    *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);
  }
  return iree_ok_status();
}

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

static const loom_amdgpu_lower_dispatch_row_t kAmdgpuLowerDispatchRows[] = {
    LOOM_AMDGPU_DIRECT_ROW(LOOM_OP_INDEX_CONSTANT,
                           loom_amdgpu_select_value_dispatch,
                           loom_amdgpu_emit_value_dispatch, NULL),
    LOOM_AMDGPU_DIRECT_ROW(LOOM_OP_SCALAR_CONSTANT,
                           loom_amdgpu_select_value_dispatch,
                           loom_amdgpu_emit_value_dispatch, NULL),
    LOOM_AMDGPU_DIRECT_ROW(LOOM_OP_BUFFER_ALLOCA,
                           loom_amdgpu_select_buffer_dispatch,
                           loom_amdgpu_emit_buffer_dispatch, NULL),
    LOOM_AMDGPU_DIRECT_ROW(LOOM_OP_BUFFER_VIEW,
                           loom_amdgpu_select_buffer_dispatch,
                           loom_amdgpu_emit_buffer_dispatch,
                           loom_amdgpu_low_legality_verify_buffer),
    LOOM_AMDGPU_DIRECT_ROW(LOOM_OP_VIEW_SUBVIEW,
                           loom_amdgpu_select_view_dispatch,
                           loom_amdgpu_emit_view_dispatch, NULL),
    LOOM_AMDGPU_DATA_ROW(LOOM_OP_VIEW_ATOMIC_REDUCE, loom_amdgpu_atomic_plan_t,
                         loom_amdgpu_select_view_atomic_dispatch,
                         loom_amdgpu_emit_view_atomic_dispatch,
                         loom_amdgpu_low_legality_verify_view_atomic),
    LOOM_AMDGPU_DATA_ROW(LOOM_OP_VIEW_ATOMIC_RMW, loom_amdgpu_atomic_plan_t,
                         loom_amdgpu_select_view_atomic_dispatch,
                         loom_amdgpu_emit_view_atomic_dispatch,
                         loom_amdgpu_low_legality_verify_view_atomic),
    LOOM_AMDGPU_DATA_ROW(LOOM_OP_VIEW_ATOMIC_CMPXCHG, loom_amdgpu_atomic_plan_t,
                         loom_amdgpu_select_view_atomic_dispatch,
                         loom_amdgpu_emit_view_atomic_dispatch,
                         loom_amdgpu_low_legality_verify_view_atomic),
    LOOM_AMDGPU_DATA_ROW(LOOM_OP_VIEW_PREFETCH, loom_amdgpu_prefetch_plan_t,
                         loom_amdgpu_select_view_prefetch_dispatch,
                         loom_amdgpu_emit_view_prefetch_dispatch, NULL),
    LOOM_AMDGPU_DIRECT_ROW(LOOM_OP_VECTOR_CONSTANT,
                           loom_amdgpu_select_value_dispatch,
                           loom_amdgpu_emit_value_dispatch, NULL),
    LOOM_AMDGPU_DIRECT_ROW(LOOM_OP_VECTOR_IOTA,
                           loom_amdgpu_select_value_dispatch,
                           loom_amdgpu_emit_value_dispatch, NULL),
    LOOM_AMDGPU_DATA_ROW(LOOM_OP_VECTOR_CMPI, loom_amdgpu_vector_compare_plan_t,
                         loom_amdgpu_select_vector_cmpi_dispatch,
                         loom_amdgpu_emit_vector_cmpi_dispatch, NULL),
    LOOM_AMDGPU_DATA_ROW(LOOM_OP_VECTOR_CMPF, loom_amdgpu_vector_compare_plan_t,
                         loom_amdgpu_select_vector_cmpf_dispatch,
                         loom_amdgpu_emit_vector_cmpf_dispatch, NULL),
    LOOM_AMDGPU_DATA_ROW(LOOM_OP_VECTOR_SELECT,
                         loom_amdgpu_vector_select_plan_t,
                         loom_amdgpu_select_vector_select_dispatch,
                         loom_amdgpu_emit_vector_select_dispatch, NULL),
    LOOM_AMDGPU_DATA_ROW(LOOM_OP_VECTOR_TABLE_LOOKUP,
                         loom_amdgpu_table_lookup_plan_t,
                         loom_amdgpu_select_vector_table_lookup_dispatch,
                         loom_amdgpu_emit_vector_table_lookup_dispatch,
                         loom_amdgpu_low_legality_verify_vector_table),
    LOOM_AMDGPU_LEGALITY_ROW(LOOM_OP_VECTOR_DOTF,
                             loom_amdgpu_low_legality_verify_vector_dot),
    LOOM_AMDGPU_LEGALITY_ROW(LOOM_OP_VECTOR_DOT2F,
                             loom_amdgpu_low_legality_verify_vector_dot),
    LOOM_AMDGPU_LEGALITY_ROW(LOOM_OP_VECTOR_DOT4I,
                             loom_amdgpu_low_legality_verify_vector_dot),
    LOOM_AMDGPU_LEGALITY_ROW(LOOM_OP_VECTOR_DOT8I4,
                             loom_amdgpu_low_legality_verify_vector_dot),
    LOOM_AMDGPU_LEGALITY_ROW(LOOM_OP_VECTOR_DOT4F8,
                             loom_amdgpu_low_legality_verify_vector_dot),
    LOOM_AMDGPU_DATA_ROW(
        LOOM_OP_VECTOR_BITFIELD_EXTRACTU, loom_amdgpu_bitfield_extract_plan_t,
        loom_amdgpu_select_vector_bitfield_extract_dispatch,
        loom_amdgpu_emit_vector_bitfield_extract_dispatch, NULL),
    LOOM_AMDGPU_DATA_ROW(
        LOOM_OP_VECTOR_BITFIELD_EXTRACTS, loom_amdgpu_bitfield_extract_plan_t,
        loom_amdgpu_select_vector_bitfield_extract_dispatch,
        loom_amdgpu_emit_vector_bitfield_extract_dispatch, NULL),
    LOOM_AMDGPU_DATA_ROW(
        LOOM_OP_VECTOR_BITFIELD_INSERT, loom_amdgpu_bitfield_insert_plan_t,
        loom_amdgpu_select_vector_bitfield_insert_dispatch,
        loom_amdgpu_emit_vector_bitfield_insert_dispatch, NULL),
    LOOM_AMDGPU_DATA_ROW(LOOM_OP_VECTOR_BITPACK, loom_amdgpu_bitpack_plan_t,
                         loom_amdgpu_select_vector_bitpack_dispatch,
                         loom_amdgpu_emit_vector_bitpack_dispatch,
                         loom_amdgpu_low_legality_verify_vector_bitstream),
    LOOM_AMDGPU_DATA_ROW(LOOM_OP_VECTOR_BITUNPACKS,
                         loom_amdgpu_bitunpack_plan_t,
                         loom_amdgpu_select_vector_bitunpack_dispatch,
                         loom_amdgpu_emit_vector_bitunpack_dispatch,
                         loom_amdgpu_low_legality_verify_vector_bitstream),
    LOOM_AMDGPU_DATA_ROW(LOOM_OP_VECTOR_BITUNPACKU,
                         loom_amdgpu_bitunpack_plan_t,
                         loom_amdgpu_select_vector_bitunpack_dispatch,
                         loom_amdgpu_emit_vector_bitunpack_dispatch,
                         loom_amdgpu_low_legality_verify_vector_bitstream),
    LOOM_AMDGPU_DATA_ROW(LOOM_OP_VECTOR_BITCAST,
                         loom_amdgpu_vector_bitcast_plan_t,
                         loom_amdgpu_select_vector_bitcast_dispatch,
                         loom_amdgpu_emit_vector_bitcast_dispatch,
                         loom_amdgpu_low_legality_verify_vector_structural),
    LOOM_AMDGPU_DATA_ROW(LOOM_OP_VECTOR_SLICE, loom_amdgpu_vector_slice_plan_t,
                         loom_amdgpu_select_vector_slice_dispatch,
                         loom_amdgpu_emit_vector_slice_dispatch,
                         loom_amdgpu_low_legality_verify_vector_structural),
    LOOM_AMDGPU_DIRECT_ROW(LOOM_OP_VECTOR_EXTRACT,
                           loom_amdgpu_select_value_dispatch,
                           loom_amdgpu_emit_value_dispatch, NULL),
    LOOM_AMDGPU_DIRECT_ROW(LOOM_OP_VECTOR_FROM_ELEMENTS,
                           loom_amdgpu_select_value_dispatch,
                           loom_amdgpu_emit_value_dispatch, NULL),
    LOOM_AMDGPU_DATA_ROW(LOOM_OP_VECTOR_LOAD, loom_amdgpu_memory_access_plan_t,
                         loom_amdgpu_select_vector_load_dispatch,
                         loom_amdgpu_emit_vector_load_dispatch,
                         loom_amdgpu_low_legality_verify_vector_memory),
    LOOM_AMDGPU_DATA_ROW(LOOM_OP_VECTOR_STORE, loom_amdgpu_memory_access_plan_t,
                         loom_amdgpu_select_vector_store_dispatch,
                         loom_amdgpu_emit_vector_store_dispatch,
                         loom_amdgpu_low_legality_verify_vector_memory),
    LOOM_AMDGPU_DIRECT_ROW(LOOM_OP_KERNEL_BARRIER,
                           loom_amdgpu_select_kernel_barrier_dispatch,
                           loom_amdgpu_emit_kernel_barrier_dispatch,
                           loom_amdgpu_low_legality_verify_kernel_barrier),
    LOOM_AMDGPU_DIRECT_ROW(LOOM_OP_KERNEL_WORKITEM_ID,
                           loom_amdgpu_select_preamble_dispatch,
                           loom_amdgpu_emit_preamble_dispatch, NULL),
    LOOM_AMDGPU_DIRECT_ROW(LOOM_OP_KERNEL_WORKGROUP_ID,
                           loom_amdgpu_select_preamble_dispatch,
                           loom_amdgpu_emit_preamble_dispatch, NULL),
    LOOM_AMDGPU_LEGALITY_ROW(LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER,
                             loom_amdgpu_low_legality_verify_kernel_async),
    LOOM_AMDGPU_LEGALITY_ROW(LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER_MASK,
                             loom_amdgpu_low_legality_verify_kernel_async),
    LOOM_AMDGPU_LEGALITY_ROW(LOOM_OP_KERNEL_ASYNC_COPY,
                             loom_amdgpu_low_legality_verify_kernel_async),
    LOOM_AMDGPU_LEGALITY_ROW(LOOM_OP_KERNEL_ASYNC_COPY_MASK,
                             loom_amdgpu_low_legality_verify_kernel_async),
    LOOM_AMDGPU_DATA_ROW(LOOM_OP_KERNEL_ASYNC_GATHER,
                         loom_amdgpu_async_gather_plan_t,
                         loom_amdgpu_select_kernel_async_gather_dispatch,
                         loom_amdgpu_emit_kernel_async_gather_dispatch,
                         loom_amdgpu_low_legality_verify_kernel_async),
    LOOM_AMDGPU_LEGALITY_ROW(LOOM_OP_KERNEL_ASYNC_GATHER_MASK,
                             loom_amdgpu_low_legality_verify_kernel_async),
    LOOM_AMDGPU_LEGALITY_ROW(LOOM_OP_KERNEL_ASYNC_GROUP,
                             loom_amdgpu_low_legality_verify_kernel_async),
    LOOM_AMDGPU_LEGALITY_ROW(LOOM_OP_KERNEL_ASYNC_TENSOR_LOAD_TO_LDS,
                             loom_amdgpu_low_legality_verify_kernel_async),
    LOOM_AMDGPU_LEGALITY_ROW(LOOM_OP_KERNEL_ASYNC_TENSOR_STORE_FROM_LDS,
                             loom_amdgpu_low_legality_verify_kernel_async),
    LOOM_AMDGPU_DATA_ROW(LOOM_OP_KERNEL_ASYNC_WAIT,
                         loom_amdgpu_async_wait_plan_t,
                         loom_amdgpu_select_kernel_async_wait_dispatch,
                         loom_amdgpu_emit_kernel_async_wait_dispatch,
                         loom_amdgpu_low_legality_verify_kernel_async),
};

#undef LOOM_AMDGPU_DIRECT_ROW
#undef LOOM_AMDGPU_DATA_ROW
#undef LOOM_AMDGPU_LEGALITY_ROW

static const loom_amdgpu_lower_dispatch_row_t*
loom_amdgpu_find_lower_dispatch_row(loom_low_lower_plan_id_t plan_id) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kAmdgpuLowerDispatchRows);
       ++i) {
    const loom_amdgpu_lower_dispatch_row_t* row = &kAmdgpuLowerDispatchRows[i];
    if ((loom_low_lower_plan_id_t)row->source_op_kind == plan_id) {
      return row;
    }
  }
  return NULL;
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
  IREE_ASSERT_ARGUMENT(out_plan);
  return loom_amdgpu_select_plan_id(context, source_op, out_plan);
}

static iree_status_t loom_amdgpu_emit_op(void* user_data,
                                         loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_low_lower_plan_t plan) {
  (void)user_data;
  const loom_amdgpu_lower_dispatch_row_t* row =
      loom_amdgpu_find_lower_dispatch_row(plan.id);
  if (row == NULL) {
    return loom_low_lower_emit_reject(
        context, source_op, IREE_SV("plan"), IREE_SV("AMDGPU callback plan"),
        IREE_SV("selected AMDGPU callback plan has no dispatch row"));
  }
  if (row->emit == NULL) {
    return loom_low_lower_emit_reject(
        context, source_op, IREE_SV("plan"), IREE_SV("AMDGPU callback plan"),
        IREE_SV("selected AMDGPU callback plan has no emission hook"));
  }
  if (row->plan_data_size != 0 && plan.target_data == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "selected AMDGPU callback plan is missing plan data");
  }
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

static const loom_low_lower_rule_set_t* const kAmdgpuRuleSets[] = {
    &loom_amdgpu_arithmetic_rule_set, &loom_amdgpu_integer_rule_set,
    &loom_amdgpu_compare_rule_set,    &loom_amdgpu_dot_rule_set,
    &loom_amdgpu_reduce_rule_set,     &loom_amdgpu_async_rule_set,
};

static const loom_low_lower_policy_t kAmdgpuLowLowerPolicy = {
    .name = IREE_SVL("amdgpu-register-lower"),
    .map_type = {.fn = loom_amdgpu_map_type, .user_data = NULL},
    .map_value = {.fn = loom_amdgpu_map_value, .user_data = NULL},
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
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("amdgpu.gfx950.core"),
          .policy = &kAmdgpuLowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("amdgpu.gfx11.core"),
          .policy = &kAmdgpuLowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("amdgpu.gfx12.core"),
          .policy = &kAmdgpuLowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("amdgpu.gfx1250.core"),
          .policy = &kAmdgpuLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
