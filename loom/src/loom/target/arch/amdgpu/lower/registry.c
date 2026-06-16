// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>
#include <stdint.h>

#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/contracts/arithmetic.h"
#include "loom/target/arch/amdgpu/contracts/arithmetic_lower_rules.h"
#include "loom/target/arch/amdgpu/contracts/async.h"
#include "loom/target/arch/amdgpu/contracts/async_lower_rules.h"
#include "loom/target/arch/amdgpu/contracts/buffer.h"
#include "loom/target/arch/amdgpu/contracts/buffer_lower_rules.h"
#include "loom/target/arch/amdgpu/contracts/compare.h"
#include "loom/target/arch/amdgpu/contracts/compare_lower_rules.h"
#include "loom/target/arch/amdgpu/contracts/config.h"
#include "loom/target/arch/amdgpu/contracts/config_lower_rules.h"
#include "loom/target/arch/amdgpu/contracts/dot.h"
#include "loom/target/arch/amdgpu/contracts/dot_lower_rules.h"
#include "loom/target/arch/amdgpu/contracts/integer.h"
#include "loom/target/arch/amdgpu/contracts/integer_lower_rules.h"
#include "loom/target/arch/amdgpu/contracts/matrix.h"
#include "loom/target/arch/amdgpu/contracts/reduce.h"
#include "loom/target/arch/amdgpu/contracts/reduce_lower_rules.h"
#include "loom/target/arch/amdgpu/contracts/view.h"
#include "loom/target/arch/amdgpu/contracts/view_lower_rules.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/lower/abi.h"
#include "loom/target/arch/amdgpu/lower/arithmetic.h"
#include "loom/target/arch/amdgpu/lower/async.h"
#include "loom/target/arch/amdgpu/lower/bitpack.h"
#include "loom/target/arch/amdgpu/lower/buffer.h"
#include "loom/target/arch/amdgpu/lower/control.h"
#include "loom/target/arch/amdgpu/lower/dot.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/mask.h"
#include "loom/target/arch/amdgpu/lower/matrix.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/preamble.h"
#include "loom/target/arch/amdgpu/lower/structural.h"
#include "loom/target/arch/amdgpu/lower/subgroup.h"
#include "loom/target/arch/amdgpu/lower/sync.h"
#include "loom/target/arch/amdgpu/lower/table.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/lower/values.h"
#include "loom/target/arch/amdgpu/lower/workgroup.h"

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

typedef uint8_t loom_amdgpu_storage_policy_t;
typedef uint8_t loom_amdgpu_preselect_policy_t;
typedef uint8_t loom_amdgpu_report_policy_t;
typedef uint8_t loom_amdgpu_lower_policy_bits_t;

enum loom_amdgpu_storage_policy_e {
  // Conservative callback-plan behavior: keep every source operand available.
  LOOM_AMDGPU_STORAGE_SOURCE_OPERANDS = 0,
  // Value lowering owns its source operand demand policy.
  LOOM_AMDGPU_STORAGE_VALUE_PLAN = 1,
  // Target plan data starts with one source value.
  LOOM_AMDGPU_STORAGE_PLAN_SOURCE_ARRAY_1 = 2,
  // Target plan data starts with a 2-value source array.
  LOOM_AMDGPU_STORAGE_PLAN_SOURCE_ARRAY_2 = 3,
  // Target plan data starts with a 3-value source array.
  LOOM_AMDGPU_STORAGE_PLAN_SOURCE_ARRAY_3 = 4,
  // Memory access plans own their source operand demand policy.
  LOOM_AMDGPU_STORAGE_MEMORY_PLAN = 5,
  // Atomic plans own their source operand demand policy.
  LOOM_AMDGPU_STORAGE_ATOMIC = 6,
  // Prefetch plans own their source operand demand policy.
  LOOM_AMDGPU_STORAGE_PREFETCH = 7,
  // Fragment memory plans own their source operand demand policy.
  LOOM_AMDGPU_STORAGE_FRAGMENT_MEMORY = 8,
  // Maximum storage-policy value accepted by dispatch row policy bits.
  LOOM_AMDGPU_STORAGE_MAX = LOOM_AMDGPU_STORAGE_FRAGMENT_MEMORY,
};

enum loom_amdgpu_preselect_policy_e {
  // The row does not need target-owned preselection before generated rules.
  LOOM_AMDGPU_PRESELECT_NONE = 0,
  // Invoke value lowering preselection for conversion and value-shaping plans.
  LOOM_AMDGPU_PRESELECT_VALUE_PLAN = 1,
  // Invoke the ordinary callback selector before generated rules.
  LOOM_AMDGPU_PRESELECT_PLAN_ID = 2,
  // Invoke the ordinary callback selector and emit FMA literal diagnostics when
  // no plan is selected.
  LOOM_AMDGPU_PRESELECT_PLAN_ID_FMA_DIAGNOSTIC = 3,
  // Maximum preselect-policy value accepted by dispatch row policy bits.
  LOOM_AMDGPU_PRESELECT_MAX = 3,
};

enum loom_amdgpu_report_policy_e {
  // The row has no target-owned compile-report plan key.
  LOOM_AMDGPU_REPORT_NONE = 0,
  // Report the workgroup-reduce publication strategy selected by the plan.
  LOOM_AMDGPU_REPORT_WORKGROUP_REDUCE_PUBLICATION = 1,
  // Maximum report-policy value accepted by dispatch row policy bits.
  LOOM_AMDGPU_REPORT_MAX = LOOM_AMDGPU_REPORT_WORKGROUP_REDUCE_PUBLICATION,
};

enum loom_amdgpu_lower_policy_bits_e {
  LOOM_AMDGPU_LOWER_POLICY_STORAGE_MASK = 0x0Fu,
  LOOM_AMDGPU_LOWER_POLICY_PRESELECT_SHIFT = 4u,
  LOOM_AMDGPU_LOWER_POLICY_PRESELECT_MASK = 0x30u,
  LOOM_AMDGPU_LOWER_POLICY_REPORT_SHIFT = 6u,
  LOOM_AMDGPU_LOWER_POLICY_REPORT_MASK = 0xC0u,
};

static_assert((LOOM_AMDGPU_STORAGE_MAX &
               ~LOOM_AMDGPU_LOWER_POLICY_STORAGE_MASK) == 0,
              "AMDGPU storage policy values must fit dispatch row bits");
static_assert(((LOOM_AMDGPU_PRESELECT_MAX
                << LOOM_AMDGPU_LOWER_POLICY_PRESELECT_SHIFT) &
               ~LOOM_AMDGPU_LOWER_POLICY_PRESELECT_MASK) == 0,
              "AMDGPU preselect policy values must fit dispatch row bits");
static_assert(((LOOM_AMDGPU_REPORT_MAX
                << LOOM_AMDGPU_LOWER_POLICY_REPORT_SHIFT) &
               ~LOOM_AMDGPU_LOWER_POLICY_REPORT_MASK) == 0,
              "AMDGPU report policy values must fit dispatch row bits");

typedef struct loom_amdgpu_lower_dispatch_row_t {
  // Source op kind covered by this AMDGPU lowering row.
  loom_op_kind_t source_op_kind;
  // Packed storage, preselection, and compile-report policy bits.
  loom_amdgpu_lower_policy_bits_t policy_bits;
  // Bytes of plan data allocated by typed data selectors. Direct selector
  // hooks keep this zero and own any mixed plan allocation themselves.
  uint16_t plan_data_size;
  // Optional source-to-low plan selection hook.
  loom_amdgpu_lower_select_fn_t select;
  // Optional source-to-low plan emission hook.
  loom_amdgpu_lower_emit_fn_t emit;
  // Optional target-low legality verifier hook.
  loom_amdgpu_lower_verify_fn_t verify;
} loom_amdgpu_lower_dispatch_row_t;
static_assert(sizeof(loom_amdgpu_lower_dispatch_row_t) == 32,
              "AMDGPU lower dispatch rows must stay cache dense");

typedef struct loom_amdgpu_lower_dispatch_table_t {
  // Dialect-local dispatch rows indexed by the low byte of a source op kind.
  const loom_amdgpu_lower_dispatch_row_t* rows;
  // Number of entries in |rows|, or 0 when the dialect is unsupported.
  uint8_t row_count;
} loom_amdgpu_lower_dispatch_table_t;
static_assert(sizeof(loom_amdgpu_lower_dispatch_table_t) == 16,
              "AMDGPU dispatch table references must stay cache dense");

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
  return loom_amdgpu_lower_kernel_barrier(
      context, source_op,
      (const loom_amdgpu_kernel_barrier_plan_t*)plan.target_data);
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

LOOM_AMDGPU_DEFINE_DATA_SELECT(
    loom_amdgpu_select_kernel_workgroup_reduce_dispatch,
    loom_amdgpu_workgroup_reduce_plan_t,
    loom_amdgpu_select_kernel_workgroup_reduce_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_kernel_workgroup_reduce_dispatch,
                             loom_amdgpu_workgroup_reduce_plan_t,
                             loom_amdgpu_lower_kernel_workgroup_reduce)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_kernel_subgroup_scan_dispatch,
                               loom_amdgpu_subgroup_scan_plan_t,
                               loom_amdgpu_select_kernel_subgroup_scan_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_kernel_subgroup_scan_dispatch,
                             loom_amdgpu_subgroup_scan_plan_t,
                             loom_amdgpu_lower_kernel_subgroup_scan)

LOOM_AMDGPU_DEFINE_DATA_SELECT(
    loom_amdgpu_select_kernel_workgroup_scan_dispatch,
    loom_amdgpu_workgroup_scan_plan_t,
    loom_amdgpu_select_kernel_workgroup_scan_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_kernel_workgroup_scan_dispatch,
                             loom_amdgpu_workgroup_scan_plan_t,
                             loom_amdgpu_lower_kernel_workgroup_scan)

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

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_scalar_cmpf_dispatch,
                               loom_amdgpu_vector_compare_plan_t,
                               loom_amdgpu_select_scalar_cmpf_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_scalar_cmpf_dispatch,
                             loom_amdgpu_vector_compare_plan_t,
                             loom_amdgpu_lower_scalar_cmpf)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_dotf_dispatch,
                               loom_amdgpu_dotf_plan_t,
                               loom_amdgpu_select_vector_dotf_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_dotf_dispatch,
                             loom_amdgpu_dotf_plan_t,
                             loom_amdgpu_lower_vector_dotf)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_scalar_clampf_dispatch,
                               loom_amdgpu_clampf_plan_t,
                               loom_amdgpu_select_scalar_clampf_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_scalar_clampf_dispatch,
                             loom_amdgpu_clampf_plan_t,
                             loom_amdgpu_lower_clampf)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_fragment_load_dispatch,
                               loom_amdgpu_fragment_memory_plan_t,
                               loom_amdgpu_select_vector_fragment_load_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_fragment_load_dispatch,
                             loom_amdgpu_fragment_memory_plan_t,
                             loom_amdgpu_lower_vector_fragment_load)

LOOM_AMDGPU_DEFINE_DATA_SELECT(
    loom_amdgpu_select_vector_fragment_store_dispatch,
    loom_amdgpu_fragment_memory_plan_t,
    loom_amdgpu_select_vector_fragment_store_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_fragment_store_dispatch,
                             loom_amdgpu_fragment_memory_plan_t,
                             loom_amdgpu_lower_vector_fragment_store)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_select_dispatch,
                               loom_amdgpu_vector_select_plan_t,
                               loom_amdgpu_select_vector_select_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_select_dispatch,
                             loom_amdgpu_vector_select_plan_t,
                             loom_amdgpu_lower_select)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_scf_select_dispatch,
                               loom_amdgpu_vector_select_plan_t,
                               loom_amdgpu_select_scf_select_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_scf_select_dispatch,
                             loom_amdgpu_vector_select_plan_t,
                             loom_amdgpu_lower_select)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_clampf_dispatch,
                               loom_amdgpu_clampf_plan_t,
                               loom_amdgpu_select_vector_clampf_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_clampf_dispatch,
                             loom_amdgpu_clampf_plan_t,
                             loom_amdgpu_lower_clampf)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_table_lookup_dispatch,
                               loom_amdgpu_table_lookup_plan_t,
                               loom_amdgpu_select_vector_table_lookup_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_table_lookup_dispatch,
                             loom_amdgpu_table_lookup_plan_t,
                             loom_amdgpu_lower_vector_table_lookup)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_scalar_fmaf_mix_dispatch,
                               loom_amdgpu_fma_mix_plan_t,
                               loom_amdgpu_select_scalar_fmaf_mix_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_scalar_fmaf_mix_dispatch,
                             loom_amdgpu_fma_mix_plan_t,
                             loom_amdgpu_lower_scalar_fmaf_mix)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_packed_fmaf_dispatch,
                               loom_amdgpu_packed_ternary_plan_t,
                               loom_amdgpu_select_vector_packed_fmaf_plan)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_packed_fmai_dispatch,
                               loom_amdgpu_packed_ternary_plan_t,
                               loom_amdgpu_select_vector_packed_fmai_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_packed_ternary_dispatch,
                             loom_amdgpu_packed_ternary_plan_t,
                             loom_amdgpu_lower_vector_packed_ternary)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_scalar_mulf_mix_dispatch,
                               loom_amdgpu_mulf_mix_plan_t,
                               loom_amdgpu_select_scalar_mulf_mix_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_scalar_mulf_mix_dispatch,
                             loom_amdgpu_mulf_mix_plan_t,
                             loom_amdgpu_lower_mulf_mix)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_mulf_mix_dispatch,
                               loom_amdgpu_mulf_mix_plan_t,
                               loom_amdgpu_select_vector_mulf_mix_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_mulf_mix_dispatch,
                             loom_amdgpu_mulf_mix_plan_t,
                             loom_amdgpu_lower_mulf_mix)

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

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_concat_dispatch,
                               loom_amdgpu_vector_concat_plan_t,
                               loom_amdgpu_select_vector_concat_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_concat_dispatch,
                             loom_amdgpu_vector_concat_plan_t,
                             loom_amdgpu_lower_vector_concat)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_deinterleave_dispatch,
                               loom_amdgpu_vector_deinterleave_plan_t,
                               loom_amdgpu_select_vector_deinterleave_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_deinterleave_dispatch,
                             loom_amdgpu_vector_deinterleave_plan_t,
                             loom_amdgpu_lower_vector_deinterleave)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_interleave_dispatch,
                               loom_amdgpu_vector_interleave_plan_t,
                               loom_amdgpu_select_vector_interleave_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_interleave_dispatch,
                             loom_amdgpu_vector_interleave_plan_t,
                             loom_amdgpu_lower_vector_interleave)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_shuffle_dispatch,
                               loom_amdgpu_vector_permutation_plan_t,
                               loom_amdgpu_select_vector_shuffle_plan)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_transpose_dispatch,
                               loom_amdgpu_vector_permutation_plan_t,
                               loom_amdgpu_select_vector_transpose_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_permutation_dispatch,
                             loom_amdgpu_vector_permutation_plan_t,
                             loom_amdgpu_lower_vector_permutation)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_slice_dispatch,
                               loom_amdgpu_vector_slice_plan_t,
                               loom_amdgpu_select_vector_slice_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_slice_dispatch,
                             loom_amdgpu_vector_slice_plan_t,
                             loom_amdgpu_lower_vector_slice)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_memory_load_dispatch,
                               loom_amdgpu_memory_access_plan_t,
                               loom_amdgpu_select_memory_load_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_memory_load_dispatch,
                             loom_amdgpu_memory_access_plan_t,
                             loom_amdgpu_lower_memory_load)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_memory_store_dispatch,
                               loom_amdgpu_memory_access_plan_t,
                               loom_amdgpu_select_memory_store_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_memory_store_dispatch,
                             loom_amdgpu_memory_access_plan_t,
                             loom_amdgpu_lower_memory_store)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_atomic_dispatch,
                               loom_amdgpu_atomic_plan_t,
                               loom_amdgpu_select_atomic_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_atomic_dispatch,
                             loom_amdgpu_atomic_plan_t,
                             loom_amdgpu_lower_atomic)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_view_prefetch_dispatch,
                               loom_amdgpu_prefetch_plan_t,
                               loom_amdgpu_select_view_prefetch_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_view_prefetch_dispatch,
                             loom_amdgpu_prefetch_plan_t,
                             loom_amdgpu_lower_view_prefetch)

#undef LOOM_AMDGPU_DEFINE_DATA_SELECT
#undef LOOM_AMDGPU_DEFINE_DATA_EMIT

#define LOOM_AMDGPU_OP_INDEX(op_kind) ((uint8_t)((op_kind) & 0xFFu))

#define LOOM_AMDGPU_PLAN_DATA_SIZE(plan_type) \
  ((uint16_t)(sizeof(plan_type) +             \
              0u * sizeof(char[(sizeof(plan_type) <= UINT16_MAX) ? 1 : -1])))

#define LOOM_AMDGPU_ENUM_MAX_CHECK(value, max_value) \
  (0u * sizeof(char[((value) <= (max_value)) ? 1 : -1]))

#define LOOM_AMDGPU_POLICY_BITS(storage_policy_value, preselect_policy_value,         \
                                report_policy_value)                                  \
  ((loom_amdgpu_lower_policy_bits_t)(((storage_policy_value) |                        \
                                      ((preselect_policy_value)                       \
                                       << LOOM_AMDGPU_LOWER_POLICY_PRESELECT_SHIFT) | \
                                      ((report_policy_value)                          \
                                       << LOOM_AMDGPU_LOWER_POLICY_REPORT_SHIFT)) +   \
                                     LOOM_AMDGPU_ENUM_MAX_CHECK(                      \
                                         storage_policy_value,                        \
                                         LOOM_AMDGPU_STORAGE_MAX) +                   \
                                     LOOM_AMDGPU_ENUM_MAX_CHECK(                      \
                                         preselect_policy_value,                      \
                                         LOOM_AMDGPU_PRESELECT_MAX) +                 \
                                     LOOM_AMDGPU_ENUM_MAX_CHECK(                      \
                                         report_policy_value,                         \
                                         LOOM_AMDGPU_REPORT_MAX)))

#define LOOM_AMDGPU_INTERNAL_ROW(                                             \
    op_kind, storage_policy_value, preselect_policy_value,                    \
    report_policy_value, plan_data_size_value, select_fn, emit_fn, verify_fn) \
  {                                                                           \
      .source_op_kind = (op_kind),                                            \
      .policy_bits = LOOM_AMDGPU_POLICY_BITS(                                 \
          storage_policy_value, preselect_policy_value, report_policy_value), \
      .plan_data_size = (plan_data_size_value),                               \
      .select = (select_fn),                                                  \
      .emit = (emit_fn),                                                      \
      .verify = (verify_fn),                                                  \
  }

#define LOOM_AMDGPU_INTERNAL_DIRECT_POLICY_ROW(                                \
    op_kind, select_fn, emit_fn, verify_fn, storage_policy_value,              \
    preselect_policy_value)                                                    \
  LOOM_AMDGPU_INTERNAL_ROW(op_kind, storage_policy_value,                      \
                           preselect_policy_value, LOOM_AMDGPU_REPORT_NONE, 0, \
                           select_fn, emit_fn, verify_fn)

#define LOOM_AMDGPU_INTERNAL_DIRECT_ROW(op_kind, select_fn, emit_fn, \
                                        verify_fn)                   \
  LOOM_AMDGPU_INTERNAL_DIRECT_POLICY_ROW(                            \
      op_kind, select_fn, emit_fn, verify_fn,                        \
      LOOM_AMDGPU_STORAGE_SOURCE_OPERANDS, LOOM_AMDGPU_PRESELECT_NONE)

#define LOOM_AMDGPU_INTERNAL_DIRECT_STORAGE_ROW(                          \
    op_kind, select_fn, emit_fn, verify_fn, storage_policy_value)         \
  LOOM_AMDGPU_INTERNAL_DIRECT_POLICY_ROW(op_kind, select_fn, emit_fn,     \
                                         verify_fn, storage_policy_value, \
                                         LOOM_AMDGPU_PRESELECT_NONE)

#define LOOM_AMDGPU_INTERNAL_DIRECT_PRESELECT_ROW(                  \
    op_kind, select_fn, emit_fn, verify_fn, preselect_policy_value) \
  LOOM_AMDGPU_INTERNAL_DIRECT_POLICY_ROW(                           \
      op_kind, select_fn, emit_fn, verify_fn,                       \
      LOOM_AMDGPU_STORAGE_SOURCE_OPERANDS, preselect_policy_value)

#define LOOM_AMDGPU_INTERNAL_DATA_POLICY_ROW(                                \
    op_kind, plan_type, select_fn, emit_fn, verify_fn, storage_policy_value, \
    preselect_policy_value)                                                  \
  LOOM_AMDGPU_INTERNAL_ROW(op_kind, storage_policy_value,                    \
                           preselect_policy_value, LOOM_AMDGPU_REPORT_NONE,  \
                           LOOM_AMDGPU_PLAN_DATA_SIZE(plan_type), select_fn, \
                           emit_fn, verify_fn)

#define LOOM_AMDGPU_INTERNAL_DATA_REPORT_ROW(                                \
    op_kind, plan_type, select_fn, emit_fn, verify_fn, report_policy_value)  \
  LOOM_AMDGPU_INTERNAL_ROW(op_kind, LOOM_AMDGPU_STORAGE_SOURCE_OPERANDS,     \
                           LOOM_AMDGPU_PRESELECT_NONE, report_policy_value,  \
                           LOOM_AMDGPU_PLAN_DATA_SIZE(plan_type), select_fn, \
                           emit_fn, verify_fn)

#define LOOM_AMDGPU_INTERNAL_DATA_STORAGE_REPORT_ROW(                        \
    op_kind, plan_type, select_fn, emit_fn, verify_fn, storage_policy_value, \
    report_policy_value)                                                     \
  LOOM_AMDGPU_INTERNAL_ROW(op_kind, storage_policy_value,                    \
                           LOOM_AMDGPU_PRESELECT_NONE, report_policy_value,  \
                           LOOM_AMDGPU_PLAN_DATA_SIZE(plan_type), select_fn, \
                           emit_fn, verify_fn)

#define LOOM_AMDGPU_INTERNAL_DATA_ROW(op_kind, plan_type, select_fn, emit_fn, \
                                      verify_fn)                              \
  LOOM_AMDGPU_INTERNAL_DATA_POLICY_ROW(                                       \
      op_kind, plan_type, select_fn, emit_fn, verify_fn,                      \
      LOOM_AMDGPU_STORAGE_SOURCE_OPERANDS, LOOM_AMDGPU_PRESELECT_NONE)

#define LOOM_AMDGPU_INTERNAL_DATA_STORAGE_ROW(                                 \
    op_kind, plan_type, select_fn, emit_fn, verify_fn, storage_policy_value)   \
  LOOM_AMDGPU_INTERNAL_DATA_POLICY_ROW(op_kind, plan_type, select_fn, emit_fn, \
                                       verify_fn, storage_policy_value,        \
                                       LOOM_AMDGPU_PRESELECT_NONE)

#define LOOM_AMDGPU_INTERNAL_DATA_PRESELECT_ROW(                               \
    op_kind, plan_type, select_fn, emit_fn, verify_fn, preselect_policy_value) \
  LOOM_AMDGPU_INTERNAL_DATA_POLICY_ROW(                                        \
      op_kind, plan_type, select_fn, emit_fn, verify_fn,                       \
      LOOM_AMDGPU_STORAGE_SOURCE_OPERANDS, preselect_policy_value)

#define LOOM_AMDGPU_VALUE_DIRECT_ROW LOOM_AMDGPU_INTERNAL_DIRECT_ROW
#define LOOM_AMDGPU_VALUE_DIRECT_STORAGE_ROW \
  LOOM_AMDGPU_INTERNAL_DIRECT_STORAGE_ROW
#define LOOM_AMDGPU_VALUE_DIRECT_PRESELECT_ROW \
  LOOM_AMDGPU_INTERNAL_DIRECT_PRESELECT_ROW
#define LOOM_AMDGPU_VALUE_DIRECT_POLICY_ROW \
  LOOM_AMDGPU_INTERNAL_DIRECT_POLICY_ROW

#define LOOM_AMDGPU_MEMORY_DATA_ROW LOOM_AMDGPU_INTERNAL_DATA_ROW
#define LOOM_AMDGPU_MEMORY_DATA_STORAGE_ROW \
  LOOM_AMDGPU_INTERNAL_DATA_STORAGE_ROW

#define LOOM_AMDGPU_RECIPE_DIRECT_ROW LOOM_AMDGPU_INTERNAL_DIRECT_ROW
#define LOOM_AMDGPU_RECIPE_DATA_ROW LOOM_AMDGPU_INTERNAL_DATA_ROW
#define LOOM_AMDGPU_RECIPE_DATA_STORAGE_ROW \
  LOOM_AMDGPU_INTERNAL_DATA_STORAGE_ROW
#define LOOM_AMDGPU_RECIPE_DATA_REPORT_ROW LOOM_AMDGPU_INTERNAL_DATA_REPORT_ROW
#define LOOM_AMDGPU_RECIPE_DATA_STORAGE_REPORT_ROW \
  LOOM_AMDGPU_INTERNAL_DATA_STORAGE_REPORT_ROW

#define LOOM_AMDGPU_GENERATED_PRESELECT_DIRECT_ROW \
  LOOM_AMDGPU_INTERNAL_DIRECT_PRESELECT_ROW
#define LOOM_AMDGPU_GENERATED_PRESELECT_DATA_PRESELECT_ROW \
  LOOM_AMDGPU_INTERNAL_DATA_PRESELECT_ROW
#define LOOM_AMDGPU_GENERATED_PRESELECT_DATA_POLICY_ROW \
  LOOM_AMDGPU_INTERNAL_DATA_POLICY_ROW

#define LOOM_AMDGPU_LEGALITY_ROW(op_kind, verify_fn)                     \
  LOOM_AMDGPU_INTERNAL_ROW(op_kind, LOOM_AMDGPU_STORAGE_SOURCE_OPERANDS, \
                           LOOM_AMDGPU_PRESELECT_NONE,                   \
                           LOOM_AMDGPU_REPORT_NONE, 0, NULL, NULL, verify_fn)

#include "loom/target/arch/amdgpu/lower/registry_tables.inl"  // IWYU pragma: keep

#define LOOM_AMDGPU_DISPATCH_TABLE(rows_value)                                 \
  {                                                                            \
      .rows = (rows_value),                                                    \
      .row_count =                                                             \
          (uint8_t)(IREE_ARRAYSIZE(rows_value) +                               \
                    0u * sizeof(char[(IREE_ARRAYSIZE(rows_value) <= UINT8_MAX) \
                                         ? 1                                   \
                                         : -1])),                              \
  }

static const loom_amdgpu_lower_dispatch_table_t
    kAmdgpuDispatchTables[LOOM_DIALECT_BUILTIN_COUNT_] = {
        [LOOM_DIALECT_INDEX] =
            LOOM_AMDGPU_DISPATCH_TABLE(kAmdgpuIndexDispatchRows),
        [LOOM_DIALECT_SCALAR] =
            LOOM_AMDGPU_DISPATCH_TABLE(kAmdgpuScalarDispatchRows),
        [LOOM_DIALECT_SCF] = LOOM_AMDGPU_DISPATCH_TABLE(kAmdgpuScfDispatchRows),
        [LOOM_DIALECT_BUFFER] =
            LOOM_AMDGPU_DISPATCH_TABLE(kAmdgpuBufferDispatchRows),
        [LOOM_DIALECT_VIEW] =
            LOOM_AMDGPU_DISPATCH_TABLE(kAmdgpuViewDispatchRows),
        [LOOM_DIALECT_VECTOR] =
            LOOM_AMDGPU_DISPATCH_TABLE(kAmdgpuVectorDispatchRows),
        [LOOM_DIALECT_KERNEL] =
            LOOM_AMDGPU_DISPATCH_TABLE(kAmdgpuKernelDispatchRows),
};

#undef LOOM_AMDGPU_DISPATCH_TABLE

#undef LOOM_AMDGPU_GENERATED_PRESELECT_DATA_POLICY_ROW
#undef LOOM_AMDGPU_GENERATED_PRESELECT_DATA_PRESELECT_ROW
#undef LOOM_AMDGPU_GENERATED_PRESELECT_DIRECT_ROW
#undef LOOM_AMDGPU_LEGALITY_ROW
#undef LOOM_AMDGPU_RECIPE_DATA_STORAGE_REPORT_ROW
#undef LOOM_AMDGPU_RECIPE_DATA_REPORT_ROW
#undef LOOM_AMDGPU_RECIPE_DATA_STORAGE_ROW
#undef LOOM_AMDGPU_RECIPE_DATA_ROW
#undef LOOM_AMDGPU_RECIPE_DIRECT_ROW
#undef LOOM_AMDGPU_MEMORY_DATA_STORAGE_ROW
#undef LOOM_AMDGPU_MEMORY_DATA_ROW
#undef LOOM_AMDGPU_VALUE_DIRECT_POLICY_ROW
#undef LOOM_AMDGPU_VALUE_DIRECT_PRESELECT_ROW
#undef LOOM_AMDGPU_VALUE_DIRECT_STORAGE_ROW
#undef LOOM_AMDGPU_VALUE_DIRECT_ROW
#undef LOOM_AMDGPU_INTERNAL_DATA_PRESELECT_ROW
#undef LOOM_AMDGPU_INTERNAL_DATA_STORAGE_ROW
#undef LOOM_AMDGPU_INTERNAL_DATA_ROW
#undef LOOM_AMDGPU_INTERNAL_DATA_STORAGE_REPORT_ROW
#undef LOOM_AMDGPU_INTERNAL_DATA_REPORT_ROW
#undef LOOM_AMDGPU_INTERNAL_DATA_POLICY_ROW
#undef LOOM_AMDGPU_INTERNAL_DIRECT_PRESELECT_ROW
#undef LOOM_AMDGPU_INTERNAL_DIRECT_STORAGE_ROW
#undef LOOM_AMDGPU_INTERNAL_DIRECT_ROW
#undef LOOM_AMDGPU_INTERNAL_DIRECT_POLICY_ROW
#undef LOOM_AMDGPU_INTERNAL_ROW
#undef LOOM_AMDGPU_POLICY_BITS
#undef LOOM_AMDGPU_ENUM_MAX_CHECK
#undef LOOM_AMDGPU_PLAN_DATA_SIZE
#undef LOOM_AMDGPU_OP_INDEX

static const loom_amdgpu_lower_dispatch_row_t*
loom_amdgpu_lower_dispatch_row_from_table(
    loom_op_kind_t op_kind, const loom_amdgpu_lower_dispatch_row_t* rows,
    uint8_t row_count) {
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
  const uint8_t dialect_id = loom_op_dialect_id(op_kind);
  if (dialect_id >= IREE_ARRAYSIZE(kAmdgpuDispatchTables)) {
    return NULL;
  }
  const loom_amdgpu_lower_dispatch_table_t* table =
      &kAmdgpuDispatchTables[dialect_id];
  if (table->rows == NULL) {
    return NULL;
  }
  return loom_amdgpu_lower_dispatch_row_from_table(op_kind, table->rows,
                                                   table->row_count);
}

static loom_amdgpu_storage_policy_t loom_amdgpu_dispatch_row_storage_policy(
    const loom_amdgpu_lower_dispatch_row_t* row) {
  if (row == NULL) {
    return LOOM_AMDGPU_STORAGE_SOURCE_OPERANDS;
  }
  return row->policy_bits & LOOM_AMDGPU_LOWER_POLICY_STORAGE_MASK;
}

static loom_amdgpu_preselect_policy_t loom_amdgpu_dispatch_row_preselect_policy(
    const loom_amdgpu_lower_dispatch_row_t* row) {
  if (row == NULL) {
    return LOOM_AMDGPU_PRESELECT_NONE;
  }
  return (row->policy_bits & LOOM_AMDGPU_LOWER_POLICY_PRESELECT_MASK) >>
         LOOM_AMDGPU_LOWER_POLICY_PRESELECT_SHIFT;
}

static loom_amdgpu_report_policy_t loom_amdgpu_dispatch_row_report_policy(
    const loom_amdgpu_lower_dispatch_row_t* row) {
  if (row == NULL) {
    return LOOM_AMDGPU_REPORT_NONE;
  }
  return (row->policy_bits & LOOM_AMDGPU_LOWER_POLICY_REPORT_MASK) >>
         LOOM_AMDGPU_LOWER_POLICY_REPORT_SHIFT;
}

static_assert(offsetof(loom_amdgpu_fma_mix_plan_t, sources) == 0,
              "fma mix plan storage policy reads the leading sources array");
static_assert(
    offsetof(loom_amdgpu_packed_ternary_plan_t, sources) == 0,
    "packed ternary plan storage policy reads the leading sources array");
static_assert(offsetof(loom_amdgpu_mulf_mix_plan_t, sources) == 0,
              "mulf mix plan storage policy reads the leading sources array");

#define LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(plan_type, field, index) \
  static_assert(                                                         \
      offsetof(plan_type, field) == sizeof(loom_value_id_t) * (index),   \
      #plan_type "." #field " must match dispatch row storage policy")

LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_bitpack_plan_t, source, 0);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_bitunpack_plan_t, source,
                                        0);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_dotf_plan_t, lhs, 0);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_dotf_plan_t, rhs, 1);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_dotf_plan_t, init, 2);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_vector_compare_plan_t, lhs,
                                        0);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_vector_compare_plan_t, rhs,
                                        1);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_vector_select_plan_t,
                                        condition, 0);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_vector_select_plan_t,
                                        true_value, 1);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_vector_select_plan_t,
                                        false_value, 2);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_clampf_plan_t, value, 0);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_clampf_plan_t, lower, 1);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_clampf_plan_t, upper, 2);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_vector_bitcast_plan_t,
                                        source, 0);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_vector_deinterleave_plan_t,
                                        source, 0);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_vector_interleave_plan_t,
                                        even, 0);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_vector_interleave_plan_t,
                                        odd, 1);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_vector_permutation_plan_t,
                                        source, 0);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_table_lookup_plan_t, table,
                                        0);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_table_lookup_plan_t,
                                        indices, 1);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_vector_slice_plan_t, source,
                                        0);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_subgroup_shuffle_plan_t,
                                        value, 0);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(
    loom_amdgpu_subgroup_broadcast_first_plan_t, value, 0);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_subgroup_reduce_plan_t,
                                        value, 0);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_workgroup_reduce_plan_t,
                                        value, 0);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_subgroup_scan_plan_t, value,
                                        0);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_workgroup_scan_plan_t,
                                        value, 0);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_subgroup_ballot_plan_t,
                                        predicate, 0);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_subgroup_vote_any_plan_t,
                                        predicate, 0);
LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD(loom_amdgpu_subgroup_vote_all_plan_t,
                                        predicate, 0);

#undef LOOM_AMDGPU_ASSERT_LEADING_SOURCE_FIELD

static void loom_amdgpu_mark_plan_sources_storage(
    loom_low_lower_context_t* context, const void* plan_data,
    uint8_t source_count) {
  const loom_value_id_t* sources = (const loom_value_id_t*)plan_data;
  for (uint8_t i = 0; i < source_count; ++i) {
    loom_low_lower_require_source_value_storage(context, sources[i]);
  }
}

static iree_status_t loom_amdgpu_select_dispatch_row(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_lower_dispatch_row_t* row,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  if (row == NULL || row->select == NULL) {
    return iree_ok_status();
  }
  return row->select(context, source_op, row, out_plan);
}

static iree_status_t loom_amdgpu_select_plan_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  const loom_amdgpu_lower_dispatch_row_t* row =
      loom_amdgpu_find_lower_dispatch_row(source_op->kind);
  return loom_amdgpu_select_dispatch_row(context, source_op, row, out_plan);
}

static iree_status_t loom_amdgpu_select_op(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_low_lower_plan_t* out_plan) {
  (void)user_data;
  return loom_amdgpu_select_plan_id(context, source_op, out_plan);
}

static iree_status_t loom_amdgpu_preselect_op(void* user_data,
                                              loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_low_lower_plan_t* out_plan) {
  (void)user_data;
  *out_plan = loom_low_lower_plan_empty();
  const loom_amdgpu_lower_dispatch_row_t* row =
      loom_amdgpu_find_lower_dispatch_row(source_op->kind);
  const loom_amdgpu_preselect_policy_t preselect_policy =
      loom_amdgpu_dispatch_row_preselect_policy(row);
  switch (preselect_policy) {
    case LOOM_AMDGPU_PRESELECT_VALUE_PLAN:
      return loom_amdgpu_preselect_value_plan(context, source_op, out_plan);
    case LOOM_AMDGPU_PRESELECT_PLAN_ID:
    case LOOM_AMDGPU_PRESELECT_PLAN_ID_FMA_DIAGNOSTIC: {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_select_dispatch_row(context, source_op, row, out_plan));
      if (preselect_policy == LOOM_AMDGPU_PRESELECT_PLAN_ID_FMA_DIAGNOSTIC &&
          loom_low_lower_plan_is_empty(*out_plan)) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_emit_fmaf_literal_operand_form_diagnostic(context,
                                                                  source_op));
      }
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

static void loom_amdgpu_mark_plan_storage_demands(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t plan) {
  (void)user_data;
  const loom_amdgpu_lower_dispatch_row_t* row =
      loom_amdgpu_find_lower_dispatch_row(plan.id);
  const loom_amdgpu_storage_policy_t storage_policy =
      loom_amdgpu_dispatch_row_storage_policy(row);
  switch (storage_policy) {
    case LOOM_AMDGPU_STORAGE_PLAN_SOURCE_ARRAY_1:
      loom_amdgpu_mark_plan_sources_storage(context, plan.target_data, 1);
      return;
    case LOOM_AMDGPU_STORAGE_PLAN_SOURCE_ARRAY_2:
      loom_amdgpu_mark_plan_sources_storage(context, plan.target_data, 2);
      return;
    case LOOM_AMDGPU_STORAGE_PLAN_SOURCE_ARRAY_3:
      loom_amdgpu_mark_plan_sources_storage(context, plan.target_data, 3);
      return;
    case LOOM_AMDGPU_STORAGE_ATOMIC:
      loom_amdgpu_mark_atomic_plan_storage_demands(
          context, source_op,
          (const loom_amdgpu_atomic_plan_t*)plan.target_data);
      return;
    case LOOM_AMDGPU_STORAGE_PREFETCH:
      loom_amdgpu_mark_prefetch_plan_storage_demands(
          context, source_op,
          (const loom_amdgpu_prefetch_plan_t*)plan.target_data);
      return;
    case LOOM_AMDGPU_STORAGE_FRAGMENT_MEMORY:
      loom_amdgpu_mark_fragment_memory_plan_storage_demands(
          context, source_op,
          (const loom_amdgpu_fragment_memory_plan_t*)plan.target_data);
      return;
    case LOOM_AMDGPU_STORAGE_MEMORY_PLAN:
      loom_amdgpu_mark_memory_access_plan_storage_demands(
          context, source_op,
          (const loom_amdgpu_memory_access_plan_t*)plan.target_data);
      return;
    case LOOM_AMDGPU_STORAGE_VALUE_PLAN:
      loom_amdgpu_mark_value_plan_storage_demands(context, source_op, plan);
      return;
    default:
      loom_low_lower_require_source_operands_storage(context, source_op);
      return;
  }
}

static iree_string_view_t loom_amdgpu_workgroup_reduce_plan_key(
    const loom_amdgpu_workgroup_reduce_plan_t* plan) {
  switch (plan->publication_kind) {
    case LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_LDS:
      return IREE_SV("amdgpu.workgroup_reduce.publication.lds");
    case LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_REDUNDANT_SUBGROUP:
      return IREE_SV("amdgpu.workgroup_reduce.publication.redundant_subgroup");
    case LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_LEADER_WORKITEM:
      return IREE_SV("amdgpu.workgroup_reduce.publication.leader_workitem");
    case LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_REDUNDANT_SUBGROUP_LEADER_LANE:
      return IREE_SV(
          "amdgpu.workgroup_reduce.publication."
          "redundant_subgroup_leader_lane");
    default:
      return iree_string_view_empty();
  }
}

static iree_string_view_t loom_amdgpu_plan_key(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t plan) {
  (void)user_data;
  (void)context;
  (void)source_op;
  const loom_amdgpu_lower_dispatch_row_t* row =
      loom_amdgpu_find_lower_dispatch_row(plan.id);
  switch (loom_amdgpu_dispatch_row_report_policy(row)) {
    case LOOM_AMDGPU_REPORT_WORKGROUP_REDUCE_PUBLICATION:
      if (plan.target_data == NULL) {
        return iree_string_view_empty();
      }
      return loom_amdgpu_workgroup_reduce_plan_key(
          (const loom_amdgpu_workgroup_reduce_plan_t*)plan.target_data);
    default:
      return iree_string_view_empty();
  }
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

#define LOOM_AMDGPU_RULE_SET_FRAGMENTS(F)                                   \
  F(ARITHMETIC, loom_amdgpu_arithmetic_contract_fragment,                   \
    loom_amdgpu_arithmetic_lower_rule_set)                                  \
  F(BUFFER, loom_amdgpu_buffer_contract_fragment,                           \
    loom_amdgpu_buffer_lower_rule_set)                                      \
  F(CONFIG, loom_amdgpu_config_contract_fragment,                           \
    loom_amdgpu_config_lower_rule_set)                                      \
  F(INTEGER, loom_amdgpu_integer_contract_fragment,                         \
    loom_amdgpu_integer_lower_rule_set)                                     \
  F(COMPARE, loom_amdgpu_compare_contract_fragment,                         \
    loom_amdgpu_compare_lower_rule_set)                                     \
  F(DOT, loom_amdgpu_dot_contract_fragment, loom_amdgpu_dot_lower_rule_set) \
  F(REDUCE, loom_amdgpu_reduce_contract_fragment,                           \
    loom_amdgpu_reduce_lower_rule_set)                                      \
  F(ASYNC, loom_amdgpu_async_contract_fragment,                             \
    loom_amdgpu_async_lower_rule_set)                                       \
  F(VIEW, loom_amdgpu_view_contract_fragment, loom_amdgpu_view_lower_rule_set)

// clang-format off
typedef enum loom_amdgpu_rule_set_index_e {
#define LOOM_AMDGPU_RULE_SET_ENUM(name, fragment, rule_set) \
  LOOM_AMDGPU_RULE_SET_INDEX_##name,
  LOOM_AMDGPU_RULE_SET_FRAGMENTS(LOOM_AMDGPU_RULE_SET_ENUM)
#undef LOOM_AMDGPU_RULE_SET_ENUM
  LOOM_AMDGPU_RULE_SET_INDEX_COUNT_,
} loom_amdgpu_rule_set_index_t;

static const loom_low_lower_rule_set_t* const kAmdgpuRuleSets[] = {
#define LOOM_AMDGPU_RULE_SET_ROW(name, fragment, rule_set) &rule_set,
  LOOM_AMDGPU_RULE_SET_FRAGMENTS(LOOM_AMDGPU_RULE_SET_ROW)
#undef LOOM_AMDGPU_RULE_SET_ROW
};

static const loom_target_contract_binding_t kAmdgpuContractBindings[] = {
#define LOOM_AMDGPU_CONTRACT_BINDING_ROW(name, fragment, rule_set) \
  {&fragment, LOOM_AMDGPU_RULE_SET_INDEX_##name},
  LOOM_AMDGPU_RULE_SET_FRAGMENTS(LOOM_AMDGPU_CONTRACT_BINDING_ROW)
#undef LOOM_AMDGPU_CONTRACT_BINDING_ROW
  {&loom_amdgpu_matrix_contract_fragment, UINT16_MAX},
};
// clang-format on

static_assert(IREE_ARRAYSIZE(kAmdgpuRuleSets) ==
                  LOOM_AMDGPU_RULE_SET_INDEX_COUNT_,
              "AMDGPU rule-set index enum must match the rule-set table");

static const loom_low_lower_policy_t kAmdgpuLowLowerPolicy = {
    .name = IREE_SVL("amdgpu-register-lower"),
    .error_catalog = &loom_amdgpu_error_catalog,
    .map_type = {.fn = loom_amdgpu_map_type, .user_data = NULL},
    .map_value = {.fn = loom_amdgpu_map_value, .user_data = NULL},
    .map_contract_value = {.fn = loom_amdgpu_map_contract_value,
                           .user_data = NULL},
    .map_argument = {.fn = loom_amdgpu_map_argument, .user_data = NULL},
    .emit_preamble = {.fn = loom_amdgpu_emit_preamble, .user_data = NULL},
    .emit_entry_setup = {.fn = loom_amdgpu_emit_entry_setup, .user_data = NULL},
    .prepare_branch = {.fn = loom_amdgpu_prepare_branch, .user_data = NULL},
    .materialize_branch_arg = {.fn = loom_amdgpu_materialize_branch_arg,
                               .user_data = NULL},
    .emit_cond_branch = {.fn = loom_amdgpu_emit_cond_branch, .user_data = NULL},
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kAmdgpuRuleSets),
            .values = kAmdgpuRuleSets,
        },
    .contract_bindings = kAmdgpuContractBindings,
    .contract_binding_count = IREE_ARRAYSIZE(kAmdgpuContractBindings),
    .descriptor_matrix =
        {
            .options = loom_amdgpu_descriptor_matrix_options,
            .query = loom_amdgpu_descriptor_matrix_query,
            .attrs = loom_amdgpu_descriptor_matrix_attrs,
            .user_data = NULL,
        },
    .preselect_op = {.fn = loom_amdgpu_preselect_op, .user_data = NULL},
    .select_op = {.fn = loom_amdgpu_select_op, .user_data = NULL},
    .mark_plan_storage_demands = {.fn = loom_amdgpu_mark_plan_storage_demands,
                                  .user_data = NULL},
    .plan_key = {.fn = loom_amdgpu_plan_key, .user_data = NULL},
    .emit_op = {.fn = loom_amdgpu_emit_op, .user_data = NULL},
};

const loom_target_low_legality_provider_t
    loom_amdgpu_low_legality_provider_storage = {
        .name = IREE_SVL("amdgpu"),
        .builtin_dialect_bits =
            (1u << LOOM_DIALECT_INDEX) | (1u << LOOM_DIALECT_BUFFER) |
            (1u << LOOM_DIALECT_SCALAR) | (1u << LOOM_DIALECT_VIEW) |
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
    {.contract_set_key = IREE_SVL("amdgpu.cdna3.core"), .policy = &kAmdgpuLowLowerPolicy},
    {.contract_set_key = IREE_SVL("amdgpu.cdna4.core"), .policy = &kAmdgpuLowLowerPolicy},
    {.contract_set_key = IREE_SVL("amdgpu.rdna3.core"), .policy = &kAmdgpuLowLowerPolicy},
    {.contract_set_key = IREE_SVL("amdgpu.rdna4.core"), .policy = &kAmdgpuLowLowerPolicy},
    {.contract_set_key = IREE_SVL("amdgpu.rdna4.gfx125x.core"), .policy = &kAmdgpuLowLowerPolicy},
  };
  // clang-format on
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
