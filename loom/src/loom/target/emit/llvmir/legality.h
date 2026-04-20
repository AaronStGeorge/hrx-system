// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// LLVMIR target legality checks over source Loom IR.
//
// Legality runs before structured LLVM IR construction. It owns diagnostics
// about the source subset, selected target records, and target-family contracts
// that must hold before lowering starts allocating LLVMIR module state.

#ifndef LOOM_TARGET_LLVMIR_LEGALITY_H_
#define LOOM_TARGET_LLVMIR_LEGALITY_H_

#include "loom/ir/ir.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_llvmir_target_legality_context_t
    loom_llvmir_target_legality_context_t;
typedef struct loom_llvmir_target_legality_provider_t
    loom_llvmir_target_legality_provider_t;

typedef enum loom_llvmir_target_legality_code_e {
  LOOM_LLVMIR_TARGET_LEGALITY_OK = 0,
  LOOM_LLVMIR_TARGET_LEGALITY_INVALID_TARGET,
  LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_ABI,
  LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_FUNCTION,
  LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_TYPE,
  LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_CONTROL_FLOW,
  LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_OP,
  LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_INTRINSIC,
  LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_TARGET_CONTRACT,
} loom_llvmir_target_legality_code_t;

typedef enum loom_llvmir_target_legality_event_kind_e {
  LOOM_LLVMIR_TARGET_LEGALITY_EVENT_TARGET_CONTRACT_SELECTED = 0,
} loom_llvmir_target_legality_event_kind_t;

typedef struct loom_llvmir_target_legality_diagnostic_t {
  // Stable diagnostic category.
  loom_llvmir_target_legality_code_t code;
  // Provider that produced the diagnostic, or empty for generic checks.
  iree_string_view_t provider_name;
  // Source op name associated with the diagnostic, or empty for target-level
  // diagnostics.
  iree_string_view_t op_name;
  // Stable reason string.
  iree_string_view_t detail;
  // Optional target-family detail, such as a rejected contract family.
  iree_string_view_t target_detail;
} loom_llvmir_target_legality_diagnostic_t;

typedef struct loom_llvmir_target_legality_event_t {
  // Event category.
  loom_llvmir_target_legality_event_kind_t kind;
  // Provider that produced the event.
  iree_string_view_t provider_name;
  // Source op name associated with the event.
  iree_string_view_t op_name;
  // Stable event reason string.
  iree_string_view_t detail;
  // Optional target-family detail, such as a selected contract descriptor.
  iree_string_view_t target_detail;
} loom_llvmir_target_legality_event_t;

typedef void (*loom_llvmir_target_legality_event_fn_t)(
    void* user_data, const loom_llvmir_target_legality_event_t* event);

typedef iree_status_t (*loom_llvmir_target_legality_try_op_fn_t)(
    const loom_llvmir_target_legality_provider_t* provider,
    loom_llvmir_target_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

struct loom_llvmir_target_legality_provider_t {
  // Stable provider name used for diagnostics.
  iree_string_view_t name;
  // Attempts to verify one source op. Sets |out_handled| false when the op does
  // not belong to this provider.
  loom_llvmir_target_legality_try_op_fn_t try_verify_op;
};

typedef struct loom_llvmir_target_legality_options_t {
  // LLVMIR codegen target snapshot selected for this compilation unit.
  const loom_target_snapshot_t* snapshot;
  // Export and ABI plan selected for this compilation unit.
  const loom_target_export_plan_t* export_plan;
  // Legalization and target-contract config selected for this compilation
  // unit.
  const loom_target_config_t* config;
  // Optional target-specific legality providers consulted by generic checks.
  const loom_llvmir_target_legality_provider_t* const* providers;
  // Number of provider pointers in |providers|.
  iree_host_size_t provider_count;
  // Optional observer for non-fatal legality events such as target contract
  // selection.
  loom_llvmir_target_legality_event_fn_t event_fn;
  // Opaque payload passed to |event_fn|.
  void* event_user_data;
} loom_llvmir_target_legality_options_t;

// Verifies that |module| is legal for LLVMIR lowering under |options|.
//
// |out_diagnostic| is optional. When provided, it is always overwritten with
// either OK or the first failing diagnostic. Non-fatal target-contract
// selections are reported through |options->event_fn|.
iree_status_t loom_llvmir_verify_target_legality(
    const loom_module_t* module,
    const loom_llvmir_target_legality_options_t* options,
    loom_llvmir_target_legality_diagnostic_t* out_diagnostic);

// Returns the source module being checked.
const loom_module_t* loom_llvmir_target_legality_module(
    const loom_llvmir_target_legality_context_t* context);

// Returns the selected target snapshot.
const loom_target_snapshot_t* loom_llvmir_target_legality_snapshot(
    const loom_llvmir_target_legality_context_t* context);

// Returns the selected export plan.
const loom_target_export_plan_t* loom_llvmir_target_legality_export_plan(
    const loom_llvmir_target_legality_context_t* context);

// Returns the selected target config.
const loom_target_config_t* loom_llvmir_target_legality_config(
    const loom_llvmir_target_legality_context_t* context);

// Emits a failing legality diagnostic and returns a matching status.
iree_status_t loom_llvmir_target_legality_fail(
    loom_llvmir_target_legality_context_t* context,
    const loom_llvmir_target_legality_provider_t* provider,
    loom_llvmir_target_legality_code_t code, const loom_op_t* op,
    iree_string_view_t detail, iree_string_view_t target_detail);

// Emits a non-fatal legality event when an event observer was provided.
void loom_llvmir_target_legality_emit_event(
    loom_llvmir_target_legality_context_t* context,
    const loom_llvmir_target_legality_provider_t* provider,
    loom_llvmir_target_legality_event_kind_t kind, const loom_op_t* op,
    iree_string_view_t detail, iree_string_view_t target_detail);

// Resolves a source string attribute from |string_id|.
iree_status_t loom_llvmir_target_legality_string_attr(
    loom_llvmir_target_legality_context_t* context,
    const loom_llvmir_target_legality_provider_t* provider, const loom_op_t* op,
    iree_string_view_t attr_name, loom_string_id_t string_id,
    iree_string_view_t* out_string);

// Verifies a structured llvmir.intrinsic operand/result shape.
iree_status_t loom_llvmir_target_legality_expect_intrinsic_shape(
    loom_llvmir_target_legality_context_t* context,
    const loom_llvmir_target_legality_provider_t* provider, const loom_op_t* op,
    iree_host_size_t operand_count, iree_host_size_t result_count,
    iree_string_view_t detail);

// Verifies that |op| has exactly one result with |expected_type|.
iree_status_t loom_llvmir_target_legality_expect_scalar_result(
    loom_llvmir_target_legality_context_t* context,
    const loom_llvmir_target_legality_provider_t* provider, const loom_op_t* op,
    loom_scalar_type_t expected_type, iree_string_view_t detail);

// Verifies that one operand is a scalar with |expected_type|.
iree_status_t loom_llvmir_target_legality_expect_scalar_operand(
    loom_llvmir_target_legality_context_t* context,
    const loom_llvmir_target_legality_provider_t* provider, const loom_op_t* op,
    iree_host_size_t operand_ordinal, loom_scalar_type_t expected_type,
    iree_string_view_t detail);

// Resolves the lowered bit width of an integer-like scalar operand.
iree_status_t loom_llvmir_target_legality_expect_integer_operand_bit_width(
    loom_llvmir_target_legality_context_t* context,
    const loom_llvmir_target_legality_provider_t* provider, const loom_op_t* op,
    iree_host_size_t operand_ordinal, iree_string_view_t detail,
    uint32_t* out_bit_width);

// Verifies that one operand is defined by a source scalar/index constant.
iree_status_t loom_llvmir_target_legality_expect_constant_operand(
    loom_llvmir_target_legality_context_t* context,
    const loom_llvmir_target_legality_provider_t* provider, const loom_op_t* op,
    iree_host_size_t operand_ordinal, iree_string_view_t detail);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_LLVMIR_LEGALITY_H_
