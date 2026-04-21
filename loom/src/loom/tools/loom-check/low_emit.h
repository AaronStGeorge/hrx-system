// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared helpers for target-owned loom-check emit providers over low.func.def.

#ifndef LOOM_TOOLS_LOOM_CHECK_LOW_EMIT_H_
#define LOOM_TOOLS_LOOM_CHECK_LOW_EMIT_H_

#include "iree/base/api.h"
#include "loom/codegen/low/packetization.h"
#include "loom/tools/loom-check/execute.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_CHECK_LOW_EMIT_MAX_ALLOCATION_BUDGETS = 8,
};

// Parses a shared low emit scheduling strategy value.
iree_status_t loom_check_low_emit_parse_schedule_strategy(
    iree_string_view_t value, iree_string_view_t option_scope,
    loom_low_schedule_strategy_t* out_strategy);

// Parses one <register-class>=<units> allocation budget token.
iree_status_t loom_check_low_emit_parse_allocation_budget(
    iree_string_view_t token, iree_string_view_t option_scope,
    loom_low_allocation_budget_t* budgets, iree_host_size_t budget_capacity,
    iree_host_size_t* budget_count);

// Finds a module-local low.func.def by symbol name.
iree_status_t loom_check_low_emit_find_low_func_def(
    loom_module_t* module, iree_string_view_t symbol_name,
    loom_op_t** out_low_function);

// Packetizes the selected low function through the registry linked into the
// emit provider request. |out_packetization| stores sidecar pointers allocated
// from request->case_arena.
iree_status_t loom_check_low_emit_packetize_function(
    const loom_check_emit_provider_request_t* request,
    iree_string_view_t function_symbol_name,
    loom_low_schedule_strategy_t schedule_strategy,
    const loom_low_allocation_budget_t* allocation_budgets,
    iree_host_size_t allocation_budget_count,
    loom_low_packetization_t* out_packetization);

// Appends one mnemonic per non-empty, non-label assembly line.
iree_status_t loom_check_low_emit_write_assembly_mnemonics(
    iree_string_view_t assembly, iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_LOW_EMIT_H_
