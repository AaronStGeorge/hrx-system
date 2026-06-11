// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/target_assignment.h"

#include "loom/ir/module.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/rewriter.h"

static bool loom_run_hal_target_symbol_refs_equal(loom_symbol_ref_t lhs,
                                                  loom_symbol_ref_t rhs) {
  return lhs.module_id == rhs.module_id && lhs.symbol_id == rhs.symbol_id;
}

static iree_status_t loom_run_hal_validate_target_ref(
    const loom_module_t* module, loom_symbol_ref_t target_ref) {
  if (!loom_symbol_ref_is_valid(target_ref)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL target resolver returned a null target "
                            "record symbol");
  }
  if (target_ref.module_id != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL target resolver returned non-local target ref {%u, %u}",
        (unsigned)target_ref.module_id, (unsigned)target_ref.symbol_id);
  }
  if (target_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL target resolver returned target symbol id %u, but module has "
        "only %" PRIhsz " symbols",
        (unsigned)target_ref.symbol_id, module->symbols.count);
  }

  const loom_symbol_t* symbol = &module->symbols.entries[target_ref.symbol_id];
  if (symbol->defining_op == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL target resolver returned unbound target symbol id %u",
        (unsigned)target_ref.symbol_id);
  }
  const loom_target_like_t target =
      loom_target_like_cast(module, symbol->defining_op);
  if (!loom_target_like_isa(target)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL target resolver returned symbol id %u, but its defining op is "
        "not target-like",
        (unsigned)target_ref.symbol_id);
  }
  if (!loom_run_hal_target_symbol_refs_equal(loom_target_like_symbol(target),
                                             target_ref)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL target resolver returned symbol id %u, but the target-like op "
        "defines a different symbol",
        (unsigned)target_ref.symbol_id);
  }
  return iree_ok_status();
}

iree_status_t loom_run_hal_count_targetless_kernels(loom_module_t* module,
                                                    uint32_t* out_count) {
  if (out_count == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL targetless kernel count requires an output "
                            "count");
  }
  *out_count = 0;
  if (module == NULL || module->body == NULL ||
      module->body->block_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL targetless kernel count requires a module "
                            "with a body block");
  }
  loom_op_t* op = NULL;
  loom_block_for_each_op(loom_module_block(module), op) {
    if (loom_kernel_def_isa(op) &&
        !loom_symbol_ref_is_valid(loom_kernel_def_target(op))) {
      ++*out_count;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_run_hal_assign_targetless_kernel_targets(
    const loom_run_hal_artifact_provider_t* artifact_provider,
    const loom_run_hal_device_target_t* device_target, loom_module_t* module,
    loom_run_hal_targetless_kernel_assignment_result_t* out_result) {
  if (out_result != NULL) {
    *out_result = (loom_run_hal_targetless_kernel_assignment_result_t){
        .target_ref = loom_symbol_ref_null(),
    };
  }
  if (artifact_provider == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL targetless kernel assignment requires an "
                            "artifact provider");
  }
  if (device_target == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL targetless kernel assignment requires a "
                            "selected target");
  }
  if (module == NULL || module->body == NULL ||
      module->body->block_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL targetless kernel assignment requires a "
                            "module with a body block");
  }

  uint32_t targetless_kernel_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_run_hal_count_targetless_kernels(module, &targetless_kernel_count));
  if (out_result != NULL) {
    out_result->targetless_kernel_count = targetless_kernel_count;
  }
  if (targetless_kernel_count == 0) {
    return iree_ok_status();
  }
  if (artifact_provider->resolve_device_target_ref == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HAL artifact provider '%.*s' cannot assign targetless kernels for "
        "%.*s targets",
        (int)artifact_provider->name.size, artifact_provider->name.data,
        (int)artifact_provider->target_family_name.size,
        artifact_provider->target_family_name.data);
  }

  loom_symbol_ref_t target_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(artifact_provider->resolve_device_target_ref(
      artifact_provider, module, device_target, &target_ref));
  IREE_RETURN_IF_ERROR(loom_run_hal_validate_target_ref(module, target_ref));

  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, &module->arena));
  iree_status_t status = iree_ok_status();
  uint32_t assigned_kernel_count = 0;
  loom_op_t* op = NULL;
  loom_block_for_each_op(loom_module_block(module), op) {
    if (!loom_kernel_def_isa(op) ||
        loom_symbol_ref_is_valid(loom_kernel_def_target(op))) {
      continue;
    }
    status =
        loom_rewriter_set_attr(&rewriter, op, loom_kernel_def_target_ATTR_INDEX,
                               loom_attr_symbol(target_ref));
    if (!iree_status_is_ok(status)) {
      break;
    }
    ++assigned_kernel_count;
  }
  loom_rewriter_deinitialize(&rewriter);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  if (out_result != NULL) {
    out_result->target_ref = target_ref;
    out_result->assigned_kernel_count = assigned_kernel_count;
    out_result->changed = assigned_kernel_count != 0;
  }
  return iree_ok_status();
}
