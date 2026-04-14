// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_defs.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/kernel/ops.h"

static iree_status_t loom_kernel_emit(iree_diagnostic_emitter_t emitter,
                                      const loom_op_t* op,
                                      const loom_error_def_t* error,
                                      const loom_diagnostic_param_t* params,
                                      iree_host_size_t param_count) {
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_kernel_emit_attribute_value_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attr_name, int64_t actual_value,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(attr_name),
      loom_param_i64(actual_value),
      loom_param_string(expected_constraint),
  };
  return loom_kernel_emit(emitter, op, &loom_err_structure_014, params,
                          IREE_ARRAYSIZE(params));
}

iree_status_t loom_kernel_barrier_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  (void)module;

  uint8_t memory_space = loom_kernel_barrier_memory_space(op);
  if (memory_space != LOOM_KERNEL_BARRIER_MEMORY_SPACE_WORKGROUP) {
    return loom_kernel_emit_attribute_value_constraint(
        emitter, op, IREE_SV("memory_space"), memory_space,
        IREE_SV("workgroup memory space"));
  }

  uint8_t ordering = loom_kernel_barrier_ordering(op);
  if (ordering != LOOM_KERNEL_BARRIER_ORDERING_ACQ_REL) {
    return loom_kernel_emit_attribute_value_constraint(
        emitter, op, IREE_SV("ordering"), ordering,
        IREE_SV("acq_rel ordering"));
  }

  uint8_t scope = loom_kernel_barrier_scope(op);
  if (scope != LOOM_KERNEL_BARRIER_SCOPE_WORKGROUP) {
    return loom_kernel_emit_attribute_value_constraint(
        emitter, op, IREE_SV("scope"), scope, IREE_SV("workgroup scope"));
  }
  return iree_ok_status();
}
