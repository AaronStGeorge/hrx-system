// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/context.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/func/source_signature.h"
#include "loom/ops/function_contract_verify.h"

static iree_status_t loom_func_emit(iree_diagnostic_emitter_t emitter,
                                    const loom_op_t* op,
                                    const loom_error_def_t* error,
                                    const loom_diagnostic_param_t* params,
                                    iree_host_size_t param_count) {
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

// Resolves the nearest function-like owner only when |op| is nested beneath
// its callable body. Other regions on function-like operations have separate
// result contracts and cannot be exited with func.exit.
static loom_func_like_t loom_func_exit_owner(const loom_module_t* module,
                                             const loom_op_t* op) {
  const loom_op_t* child_op = op;
  for (const loom_op_t* parent_op = op->parent_op; parent_op;
       child_op = parent_op, parent_op = parent_op->parent_op) {
    loom_func_like_t function = loom_func_like_const_cast(module, parent_op);
    if (!loom_func_like_isa(function)) {
      continue;
    }
    return child_op->parent_block && child_op->parent_block->parent_region ==
                                         loom_func_like_body(function)
               ? function
               : (loom_func_like_t){0};
  }
  return (loom_func_like_t){0};
}

iree_status_t loom_func_def_verify(const loom_module_t* module,
                                   const loom_op_t* op,
                                   iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_function_contract_verify(module, op, emitter));
  return loom_func_source_signature_verify(module, op, emitter);
}

iree_status_t loom_func_decl_verify(const loom_module_t* module,
                                    const loom_op_t* op,
                                    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_function_contract_verify(module, op, emitter));
  return loom_func_source_signature_verify(module, op, emitter);
}

iree_status_t loom_func_call_verify(const loom_module_t* module,
                                    const loom_op_t* op,
                                    iree_diagnostic_emitter_t emitter) {
  return loom_function_call_contract_verify(
      module, op, loom_func_call_callee(op), loom_func_call_operands(op),
      loom_func_call_results(op), /*argument_match_flags=*/0, emitter);
}

iree_status_t loom_func_exit_verify(const loom_module_t* module,
                                    const loom_op_t* op,
                                    iree_diagnostic_emitter_t emitter) {
  loom_func_like_t function = loom_func_exit_owner(module, op);
  if (!loom_func_like_isa(function)) {
    const iree_string_view_t actual_ancestor =
        op->parent_op ? loom_op_name(module, op->parent_op) : IREE_SV("none");
    const loom_diagnostic_param_t params[] = {
        loom_param_string(loom_op_name(module, op)),
        loom_param_string(IREE_SV("nearest function-like body")),
        loom_param_string(IREE_SV("function-like body")),
        loom_param_string(actual_ancestor),
    };
    return loom_func_emit(emitter, op, LOOM_ERR_STRUCTURE_029, params,
                          IREE_ARRAYSIZE(params));
  }
  const loom_op_vtable_t* function_vtable =
      loom_context_resolve_op(module->context, function.op->kind);
  if (function_vtable->symbol_kind == LOOM_SYMBOL_TEMPLATE_DEF) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(loom_op_name(module, op)),
        loom_param_string(IREE_SV("runtime callable body")),
        loom_param_string(IREE_SV("function or kernel body")),
        loom_param_string(loom_op_name(module, function.op)),
    };
    return loom_func_emit(emitter, op, LOOM_ERR_STRUCTURE_029, params,
                          IREE_ARRAYSIZE(params));
  }

  loom_region_t* exit_region = loom_func_exit_body(op);
  loom_block_t* exit_block = loom_region_entry_block(exit_region);
  if (!exit_block || !exit_block->last_op) {
    return iree_ok_status();  // Generic structure verification diagnoses this.
  }
  const loom_op_t* exit_terminator = exit_block->last_op;
  const loom_region_descriptor_t* body_descriptor =
      loom_func_like_body_region_descriptor(module, function);
  if (!body_descriptor || !loom_region_descriptor_matches_terminator(
                              body_descriptor, exit_terminator->kind)) {
    const loom_op_vtable_t* expected_vtable =
        body_descriptor ? loom_context_resolve_op(module->context,
                                                  body_descriptor->terminator)
                        : NULL;
    const loom_diagnostic_param_t params[] = {
        loom_param_string(loom_op_name(module, op)),
        loom_param_u32(0),
        loom_param_string(expected_vtable ? loom_op_vtable_name(expected_vtable)
                                          : IREE_SV("function-like return")),
        loom_param_string(loom_op_name(module, exit_terminator)),
    };
    return loom_func_emit(emitter, op, LOOM_ERR_STRUCTURE_018, params,
                          IREE_ARRAYSIZE(params));
  }

  return loom_function_exit_contract_verify(module, function, exit_terminator,
                                            emitter);
}
